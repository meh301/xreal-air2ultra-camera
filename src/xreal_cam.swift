// xreal_cam.swift — Access XREAL Air 2 Ultra tracking cameras on macOS WITHOUT the official SDK.
//
// The glasses expose their stereo tracking cameras as a standard USB Video Class (UVC)
// device, so AVFoundation can open them like any webcam. The single UVC stream is:
//   * 640 x 482, 8-bit monochrome (delivered by AVFoundation as a 2vuy/UYVY buffer whose
//     bytes are actually raw luma — one byte per pixel, row stride 640).
//   * Rows 0..479  = the 640x480 camera image.
//   * Row 480      = a telemetry row (frame counter at col 19, markers 0x19/0xAD/0xDA).
//   * Row 481      = constant 0x5C padding.
//   * Consecutive UVC frames alternate between the LEFT and RIGHT camera; a frame pair
//     shares one counter value (col 19 of row 480). Effective stereo rate ~30 pairs/s.
//
// Usage:
//   swiftc xreal_cam.swift -o xreal_cam
//   ./xreal_cam <numFrames> <outDir>
// Writes <outDir>/cam0_XXXX.pgm and cam1_XXXX.pgm (raw 8-bit, metadata stripped) plus a
// meta.csv log of (uvcIndex, counter, imageMean).

import AVFoundation
import CoreVideo
import Foundation

let args = CommandLine.arguments
let maxFrames = args.count > 1 ? Int(args[1]) ?? 60 : 60
let outDir = args.count > 2 ? args[2] : "./xreal_out"
try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)

let W = 640, HFULL = 482, HIMG = 480, META_ROW = 480, CTR_COL = 19

func writePGM(_ path: String, _ pixels: UnsafePointer<UInt8>, stride: Int, w: Int, h: Int) {
    var data = Data("P5\n\(w) \(h)\n255\n".utf8)
    for y in 0..<h { data.append(Data(bytes: pixels + y * stride, count: w)) }
    try? data.write(to: URL(fileURLWithPath: path))
}

final class XREALCamera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    var count = 0
    var csv = "uvcIndex,counter,mean\n"
    let sem = DispatchSemaphore(value: 0)

    func start() {
        guard let dev = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .continuityCamera], mediaType: .video, position: .unspecified)
            .devices.first(where: { $0.localizedName.contains("UVC Camera") || $0.manufacturer.contains("XREAL") }) else {
            FileHandle.standardError.write(Data("XREAL UVC camera not found.\n".utf8)); exit(1)
        }
        print("Opening \(dev.localizedName) [\(dev.modelID)]")
        guard let input = try? AVCaptureDeviceInput(device: dev) else { exit(1) }
        session.beginConfiguration()
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.setSampleBufferDelegate(self, queue: DispatchQueue(label: "xreal"))
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        session.startRunning()
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput s: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let px = CMSampleBufferGetImageBuffer(s) else { return }
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        let bpr = CVPixelBufferGetBytesPerRow(px)              // ~1280
        let bufH = CVPixelBufferGetHeight(px)                  // 241 buffer rows
        let lineBytes = 2 * W                                  // 1280 image bytes per buffer row (= 2 image rows)
        guard let base = CVPixelBufferGetBaseAddress(px) else { return }
        let p = base.assumingMemoryBound(to: UInt8.self)

        // Every byte of the 2vuy buffer is one real 8-bit luma pixel. A buffer row holds
        // 2*W=1280 luma bytes (= two 640-wide image rows). Copy into a tight 640xHFULL buffer,
        // skipping any per-row stride padding (bpr may exceed lineBytes).
        var mono = [UInt8](repeating: 0, count: W * HFULL)
        mono.withUnsafeMutableBufferPointer { dst in
            for y in 0..<bufH {
                memcpy(dst.baseAddress! + y * lineBytes, p + y * bpr, lineBytes)
            }
        }

        let counter = Int(mono[META_ROW * W + CTR_COL])
        var sum = 0
        for i in 0..<(HIMG * W) { sum += Int(mono[i]) }
        let mean = Double(sum) / Double(HIMG * W)
        let camIndex = Int(mono[META_ROW * W + 58]) & 1  // row480 col58: 0x20 = cam0, 0x21 = cam1 (到着順は固定でない)

        mono.withUnsafeBufferPointer { buf in
            let name = String(format: "%@/cam%d_%04d.pgm", outDir, camIndex, count)
            writePGM(name, buf.baseAddress!, stride: W, w: W, h: HIMG)   // metadata rows stripped
        }
        csv += "\(count),\(counter),\(String(format: "%.1f", mean))\n"
        if count % 10 == 0 { print("frame \(count) cam\(camIndex) counter=\(counter) mean=\(String(format: "%.1f", mean))") }
        count += 1
        if count >= maxFrames { session.stopRunning(); sem.signal() }
    }
}

let cam = XREALCamera()
cam.start()
_ = cam.sem.wait(timeout: .now() + Double(maxFrames) / 30.0 + 10)
try? cam.csv.write(toFile: "\(outDir)/meta.csv", atomically: true, encoding: .utf8)
print("Done. Captured \(cam.count) frames to \(outDir)/ (meta.csv logged)")
