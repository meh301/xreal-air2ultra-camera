import AVFoundation
import CoreVideo
import Foundation

let OUT = "./frames"
try? FileManager.default.createDirectory(atPath: OUT, withIntermediateDirectories: true)

final class Grabber: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    var count = 0
    let maxFrames = 120
    let sem = DispatchSemaphore(value: 0)

    func start() {
        guard let dev = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.external, .continuityCamera],
            mediaType: .video, position: .unspecified)
            .devices.first(where: { $0.localizedName.contains("UVC Camera") || $0.manufacturer.contains("XREAL") }) else {
            print("XREAL UVC camera not found"); exit(1)
        }
        print("Opening: \(dev.localizedName) [\(dev.modelID)]")
        do {
            let input = try AVCaptureDeviceInput(device: dev)
            session.beginConfiguration()
            if session.canAddInput(input) { session.addInput(input) }
            let output = AVCaptureVideoDataOutput()
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "grab"))
            if session.canAddOutput(output) { session.addOutput(output) }
            session.commitConfiguration()
            session.startRunning()
        } catch {
            print("Error opening device: \(error)"); exit(1)
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let px = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(px, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        let w = CVPixelBufferGetWidth(px)
        let h = CVPixelBufferGetHeight(px)
        let bpr = CVPixelBufferGetBytesPerRow(px)
        let base = CVPixelBufferGetBaseAddress(px)!
        let dataLen = bpr * h
        if count == 0 {
            let fmt = CVPixelBufferGetPixelFormatType(px)
            let b = [UInt8((fmt>>24)&0xff),UInt8((fmt>>16)&0xff),UInt8((fmt>>8)&0xff),UInt8(fmt&0xff)]
            print("Frame: \(w)x\(h) bytesPerRow=\(bpr) pixfmt=\(String(bytes:b,encoding:.ascii) ?? "?") totalBytes=\(dataLen)")
        }
        // Save raw bytes of every frame
        let d = Data(bytes: base, count: dataLen)
        try? d.write(to: URL(fileURLWithPath: "\(OUT)/frame_\(count).raw"))
        count += 1
        if count >= maxFrames { session.stopRunning(); sem.signal() }
    }
}

let g = Grabber()
g.start()
if g.sem.wait(timeout: .now() + 15) == .timedOut {
    print("Timed out. Frames captured: \(g.count)")
} else {
    print("Done. Frames captured: \(g.count)")
}
