// cam_probe.swift — keep the vendor-HID open while watching the UVC camera, send a series
// of MCU commands, and report which command changes the camera image (brightness / contrast
// / banding). Goal: find the command that enables the cameras / IR illuminator / SLAM mode.
//
// Build: swiftc cam_probe.swift -o cam_probe -framework AVFoundation -framework IOKit -framework CoreFoundation
// Run:   ./cam_probe            (scans a default command set)
import AVFoundation
import IOKit
import IOKit.hid
import Foundation

// ---------- CRC32 + packet ----------
let crcTable: [UInt32] = { (0..<256).map { i -> UInt32 in
    var c = UInt32(i); for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1 }; return c } }()
func crc32(_ b: ArraySlice<UInt8>) -> UInt32 {
    var r: UInt32 = 0xffffffff; for x in b { r = (r >> 8) ^ crcTable[Int((r ^ UInt32(x)) & 0xff)] }; return r ^ 0xffffffff }
func mcuPacket(_ cmdId: UInt16, _ data: [UInt8]) -> [UInt8] {
    var p = [UInt8](repeating: 0, count: 64); p[0] = 0xfd
    let length = UInt16(data.count + 17)
    func p16(_ v: UInt16, _ o: Int) { p[o] = UInt8(v & 0xff); p[o+1] = UInt8(v >> 8) }
    func p32(_ v: UInt32, _ o: Int) { for k in 0..<4 { p[o+k] = UInt8((v >> (8*UInt32(k))) & 0xff) } }
    p16(length, 5); p32(0x1337, 7); p32(0, 11); p16(cmdId, 15)
    for (i, b) in data.enumerated() { p[22+i] = b }
    p32(crc32(p[5..<(5+Int(length))]), 1); return p }

// ---------- camera ----------
final class Cam: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession(); let lock = NSLock()
    var mean = 0.0, std = 0.0, sat = 0.0, framediff = 0.0
    var prev: [UInt8]?
    func start() {
        guard let d = AVCaptureDevice.DiscoverySession(deviceTypes: [.external, .continuityCamera],
            mediaType: .video, position: .unspecified).devices.first(where: {
                $0.localizedName.contains("UVC Camera") || $0.manufacturer.contains("XREAL") }) else {
            print("camera not found"); exit(1) }
        let input = try! AVCaptureDeviceInput(device: d)
        session.beginConfiguration(); session.addInput(input)
        let o = AVCaptureVideoDataOutput(); o.setSampleBufferDelegate(self, queue: DispatchQueue(label: "cam"))
        session.addOutput(o); session.commitConfiguration(); session.startRunning()
    }
    func captureOutput(_ o: AVCaptureOutput, didOutput s: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let px = CMSampleBufferGetImageBuffer(s) else { return }
        CVPixelBufferLockBaseAddress(px, .readOnly); defer { CVPixelBufferUnlockBaseAddress(px, .readOnly) }
        let bpr = CVPixelBufferGetBytesPerRow(px), bufH = CVPixelBufferGetHeight(px)
        guard let base = CVPixelBufferGetBaseAddress(px) else { return }
        let p = base.assumingMemoryBound(to: UInt8.self)
        var img = [UInt8](repeating: 0, count: 640*480)
        let line = 1280
        for y in 0..<bufH { let n = min(line, 640*480 - y*line); if n > 0 { memcpy(&img[y*line], p + y*bpr, n) } }
        var sum = 0.0, sq = 0.0, st = 0
        for v in img { let d = Double(v); sum += d; sq += d*d; if v > 250 { st += 1 } }
        let n = Double(img.count); let m = sum/n
        var fd = 0.0
        if let pr = prev { var s = 0; for i in 0..<img.count { s += abs(Int(img[i]) - Int(pr[i])) }; fd = Double(s)/n }
        prev = img
        lock.lock(); mean = m; std = (sq/n - m*m).squareRoot(); sat = Double(st)/n*100; framediff = fd; lock.unlock()
    }
    func snap() -> (Double,Double,Double,Double) { lock.lock(); defer { lock.unlock() }; return (mean,std,sat,framediff) }
}

// ---------- HID (keep open) ----------
var devices: [IOHIDDevice] = []
let mgr = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
IOHIDManagerSetDeviceMatching(mgr, [kIOHIDVendorIDKey: 0x3318, kIOHIDProductIDKey: 0x0426] as CFDictionary)
IOHIDManagerRegisterDeviceMatchingCallback(mgr, { _,_,_,dev in
    if ((IOHIDDeviceGetProperty(dev, kIOHIDPrimaryUsagePageKey as CFString) as? Int) ?? 0) == 0x41 {
        IOHIDDeviceOpen(dev, IOOptionBits(kIOHIDOptionsTypeNone)); devices.append(dev) } }, nil)
IOHIDManagerScheduleWithRunLoop(mgr, CFRunLoopGetCurrent(), CFRunLoopMode.defaultMode.rawValue)
IOHIDManagerOpen(mgr, IOOptionBits(kIOHIDOptionsTypeNone))
CFRunLoopRunInMode(.defaultMode, 0.5, false)
print("opened \(devices.count) vendor-HID interfaces")

func send(_ cmdId: UInt16, _ data: [UInt8]) {
    let pkt = mcuPacket(cmdId, data)
    for d in devices { _ = pkt.withUnsafeBufferPointer { IOHIDDeviceSetReport(d, kIOHIDReportTypeOutput, 0, $0.baseAddress!, 64) } }
}

func avgStats(_ cam: Cam, _ secs: Double) -> (Double,Double,Double,Double) {
    let steps = Int(secs/0.05); var a = [0.0,0,0,0]
    for _ in 0..<steps { let s = cam.snap(); a[0]+=s.0; a[1]+=s.1; a[2]+=s.2; a[3]+=s.3
        CFRunLoopRunInMode(.defaultMode, 0.05, false) }
    let n = Double(max(1,steps)); return (a[0]/n,a[1]/n,a[2]/n,a[3]/n) }

// ---------- run ----------
let cam = Cam(); cam.start()
CFRunLoopRunInMode(.defaultMode, 1.5, false)   // warm up
let base = avgStats(cam, 0.6)
print(String(format: "baseline: mean=%.1f std=%.1f sat%%=%.2f framediff=%.1f\n", base.0, base.1, base.2, base.3))
print("cmd    data  ->  mean   std   sat%%  fdiff   (Δmean)")

// candidate command ids to probe (skip 0x08 = set display mode). data [0x01] to "enable".
var cmds: [(UInt16,[UInt8])] = []
for c: UInt16 in 1...0x1f where c != 0x08 { cmds.append((c, [0x01])) }
// a few two-byte / zero-data variants of interest
cmds += [(0x19,[0x01]),(0x1a,[0x01]),(0x1d,[0x01]),(0x6c01,[0x01]),(0x6c02,[0x01])]

for (c, d) in cmds {
    send(c, d)
    let after = avgStats(cam, 0.7)
    let dm = after.0 - base.0
    let flag = (abs(dm) > 8 || abs(after.2 - base.2) > 4) ? "  <== CHANGED" : ""
    print(String(format: "0x%04x  %@   ->  %5.1f %5.1f %5.2f %6.1f  (%+.1f)%@",
                 c, d.map{String(format:"%02x",$0)}.joined(), after.0, after.1, after.2, after.3, dm, flag))
}
print("\ndone.")
