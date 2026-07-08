import AVFoundation
import CoreVideo
import Foundation

let OUT = "./lighttest_out"
try? FileManager.default.createDirectory(atPath: OUT, withIntermediateDirectories: true)
let DURATION = 15.0

final class Grabber: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    var count = 0
    var t0 = Date()
    var csv = "elapsed_s,mean,minv,maxv,counter\n"
    let sem = DispatchSemaphore(value: 0)

    func start() {
        guard let dev = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .continuityCamera], mediaType: .video, position: .unspecified)
            .devices.first(where: { $0.localizedName.contains("UVC Camera") || $0.manufacturer.contains("XREAL") }) else {
            print("not found"); exit(1)
        }
        print("Opening \(dev.localizedName)")
        guard let input = try? AVCaptureDeviceInput(device: dev) else { exit(1) }
        session.beginConfiguration()
        if session.canAddInput(input) { session.addInput(input) }
        let out = AVCaptureVideoDataOutput()
        out.setSampleBufferDelegate(self, queue: DispatchQueue(label: "g"))
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        t0 = Date()
        session.startRunning()
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput s: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let px = CMSampleBufferGetImageBuffer(s) else { return }
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        let bpr = CVPixelBufferGetBytesPerRow(px)
        let bufH = CVPixelBufferGetHeight(px)
        guard let base = CVPixelBufferGetBaseAddress(px) else { return }
        let p = base.assumingMemoryBound(to: UInt8.self)
        let elapsed = Date().timeIntervalSince(t0)
        // stats over the image bytes (skip last buffer row = metadata)
        var sum = 0, mn = 255, mx = 0
        let rows = bufH - 1
        for y in 0..<rows {
            let row = p + y * bpr
            for x in 0..<(2 * 640) {
                let v = Int(row[x]); sum += v
                if v < mn { mn = v }; if v > mx { mx = v }
            }
        }
        let npx = rows * 2 * 640
        let mean = Double(sum) / Double(npx)
        let counter = Int((p + 480 * (bpr / bufH >= 1 ? bpr : bpr))[0]) // fallback; counter parsed in python instead
        _ = counter
        csv += String(format: "%.3f,%.2f,%d,%d,0\n", elapsed, mean, mn, mx)
        // save a raw every ~0.5s for later frame inspection
        if count % 30 == 0 {
            let d = Data(bytes: base, count: bpr * bufH)
            try? d.write(to: URL(fileURLWithPath: String(format: "%@/t%05.1f.raw", OUT, elapsed)))
        }
        count += 1
        if elapsed >= DURATION { session.stopRunning(); sem.signal() }
    }
}

let g = Grabber()
g.start()
_ = g.sem.wait(timeout: .now() + DURATION + 8)
try? g.csv.write(toFile: "\(OUT)/trace.csv", atomically: true, encoding: .utf8)
print("Done. frames=\(g.count), duration=\(DURATION)s, trace.csv written")
