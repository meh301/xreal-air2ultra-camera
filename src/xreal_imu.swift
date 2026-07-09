// xreal_imu.swift — native macOS reader for the XREAL Air 2 Ultra 1 kHz IMU.
// Counterpart of python/xreal_imu.py (which also runs on macOS via hidapi);
// packet formats are documented in docs/PROTOCOL.md.
//
// Build: swiftc -O src/xreal_imu.swift -o xreal_imu -framework IOKit -framework CoreFoundation
//        (or just `make`)
// Usage:
//   ./xreal_imu                     # live console output (1 line/s)
//   ./xreal_imu --csv out.csv       # also log every sample
//   ./xreal_imu --config out.json   # dump the factory calibration JSON, exit
//   ./xreal_imu --info              # serial + firmware versions, exit
//   ./xreal_imu --seconds N         # stop after N seconds
//
// The tool broadcasts commands to all vendor HID interfaces (usage page 0x41)
// and lets the replies identify the right one — the MCU channel answers 0xFD
// packets, the IMU channel answers 0xAA packets and streams 01-02 reports.

import Foundation
import IOKit
import IOKit.hid

let VID = 0x3318, PID = 0x0426

// ---- CRC-32 (zlib) -----------------------------------------------------------
let crcTable: [UInt32] = (0..<256).map { i -> UInt32 in
    var c = UInt32(i)
    for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1 }
    return c
}
func crc32(_ buf: ArraySlice<UInt8>) -> UInt32 {
    var r: UInt32 = 0xffffffff
    for b in buf { r = (r >> 8) ^ crcTable[Int((r ^ UInt32(b)) & 0xff)] }
    return r ^ 0xffffffff
}

// ---- packet builders ------------------------------------------------------------
func put16(_ p: inout [UInt8], _ v: UInt16, _ o: Int) { p[o] = UInt8(v & 0xff); p[o + 1] = UInt8(v >> 8) }
func put32(_ p: inout [UInt8], _ v: UInt32, _ o: Int) { for k in 0..<4 { p[o + k] = UInt8((v >> (8 * k)) & 0xff) } }

/// IMU-channel command: aa | crc32 | len u16 (= data+3) | cmd u8 | data
func imuPacket(_ cmd: UInt8, _ data: [UInt8] = []) -> [UInt8] {
    var p = [UInt8](repeating: 0, count: 64)
    p[0] = 0xaa
    put16(&p, UInt16(data.count + 3), 5)
    p[7] = cmd
    for (i, b) in data.enumerated() { p[8 + i] = b }
    put32(&p, crc32(p[5..<(5 + data.count + 3)]), 1)
    return p
}

/// MCU-channel command: fd | crc32 | len u16 (= data+17) | req u32 | ts u32 | cmd u16 | rsvd[5] | data
func mcuPacket(_ cmd: UInt16, _ data: [UInt8] = []) -> [UInt8] {
    var p = [UInt8](repeating: 0, count: 64)
    p[0] = 0xfd
    let length = UInt16(data.count + 17)
    put16(&p, length, 5)
    put32(&p, 0x1337, 7)
    put16(&p, cmd, 15)
    for (i, b) in data.enumerated() { p[22 + i] = b }
    put32(&p, crc32(p[5..<(5 + Int(length))]), 1)
    return p
}

// ---- little-endian readers --------------------------------------------------------
func u16(_ b: UnsafePointer<UInt8>, _ o: Int) -> Int { Int(b[o]) | Int(b[o + 1]) << 8 }
func u32(_ b: UnsafePointer<UInt8>, _ o: Int) -> UInt32 {
    UInt32(b[o]) | UInt32(b[o + 1]) << 8 | UInt32(b[o + 2]) << 16 | UInt32(b[o + 3]) << 24
}
func u64(_ b: UnsafePointer<UInt8>, _ o: Int) -> UInt64 {
    (0..<8).reduce(UInt64(0)) { $0 | UInt64(b[o + $1]) << (8 * UInt64($1)) }
}
func i24(_ b: UnsafePointer<UInt8>, _ o: Int) -> Int32 {
    let v = Int32(b[o]) | Int32(b[o + 1]) << 8 | Int32(b[o + 2]) << 16
    return (v & 0x80_0000) != 0 ? v - 0x100_0000 : v
}

// ---- args --------------------------------------------------------------------------
let args = CommandLine.arguments
func argValue(_ flag: String) -> String? {
    guard let i = args.firstIndex(of: flag), args.count > i + 1 else { return nil }
    return args[i + 1]
}
let csvPath = argValue("--csv")
let configPath = argValue("--config")
let wantInfo = args.contains("--info")
let seconds = argValue("--seconds").flatMap { Double($0) }

// ---- device plumbing ------------------------------------------------------------------
var devices: [IOHIDDevice] = []
var csvHandle: FileHandle?

// latest command replies, keyed by channel head byte
var aaReply: (cmd: UInt8, data: [UInt8])?
var fdReply: (cmd: UInt16, data: [UInt8])?

// streaming stats
var sampleCount = 0, lastPrint = Date()
var totalSamples = 0

func handleImuReport(_ r: UnsafePointer<UInt8>, _ len: Int) {
    guard len >= 58, r[0] == 0x01, r[1] == 0x02 else { return }
    let ts = u64(r, 4)
    let gm = Float(u16(r, 12)), gd = Float(u32(r, 14))
    let am = Float(u16(r, 27)), ad = Float(u32(r, 29))
    let g = (0..<3).map { Float(i24(r, 18 + 3 * $0)) * gm / gd }   // deg/s
    let a = (0..<3).map { Float(i24(r, 33 + 3 * $0)) * am / ad }   // g
    sampleCount += 1
    totalSamples += 1
    if let h = csvHandle {
        let line = "\(ts),\(u32(r, 54)),\(u16(r, 2)),"
            + String(format: "%.4f,%.4f,%.4f,%.5f,%.5f,%.5f\n",
                     g[0], g[1], g[2], a[0], a[1], a[2])
        h.write(Data(line.utf8))
    }
    let dt = Date().timeIntervalSince(lastPrint)
    if dt >= 1 {
        let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).squareRoot()
        print(String(format: "%6.0f Hz  gyro=(%+8.3f,%+8.3f,%+8.3f) deg/s  " +
                     "accel=(%+6.3f,%+6.3f,%+6.3f) g  |a|=%5.3f  t_raw=%d",
                     Double(sampleCount) / dt, g[0], g[1], g[2], a[0], a[1], a[2],
                     mag, u16(r, 2)))
        sampleCount = 0
        lastPrint = Date()
    }
}

let inputCB: IOHIDReportCallback = { _, _, _, _, _, report, length in
    switch report[0] {
    case 0x01: handleImuReport(report, length)
    case 0xaa:
        guard length >= 8 else { return }
        let l = u16(report, 5)
        aaReply = (report[7], (8..<min(length, 8 + max(0, l - 3))).map { report[$0] })
    case 0xfd:
        guard length >= 22 else { return }
        let l = u16(report, 5)
        fdReply = (UInt16(u16(report, 15)),
                   (22..<min(length, 22 + max(0, l - 17))).map { report[$0] })
    default: break
    }
}

var reportBuffers: [UnsafeMutablePointer<UInt8>] = []
let mgr = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
IOHIDManagerSetDeviceMatching(mgr, [kIOHIDVendorIDKey: VID, kIOHIDProductIDKey: PID] as CFDictionary)
IOHIDManagerRegisterDeviceMatchingCallback(mgr, { _, _, _, device in
    let up = (IOHIDDeviceGetProperty(device, kIOHIDPrimaryUsagePageKey as CFString) as? Int) ?? 0
    if up != 0x41 { return }                          // vendor interfaces only
    devices.append(device)
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: 1024)
    reportBuffers.append(buf)
    IOHIDDeviceOpen(device, IOOptionBits(kIOHIDOptionsTypeNone))
    IOHIDDeviceRegisterInputReportCallback(device, buf, 1024, inputCB, nil)
}, nil)
IOHIDManagerScheduleWithRunLoop(mgr, CFRunLoopGetCurrent(), CFRunLoopMode.defaultMode.rawValue)
IOHIDManagerOpen(mgr, IOOptionBits(kIOHIDOptionsTypeNone))
CFRunLoopRunInMode(.defaultMode, 0.5, false)
guard !devices.isEmpty else {
    FileHandle.standardError.write(Data("XREAL vendor HID interfaces not found - glasses plugged in?\n".utf8))
    exit(1)
}

func broadcast(_ packet: [UInt8]) {
    for dev in devices {
        _ = packet.withUnsafeBufferPointer {
            IOHIDDeviceSetReport(dev, kIOHIDReportTypeOutput, 0, $0.baseAddress!, packet.count)
        }
    }
}

/// Send an IMU-channel command and wait for its reply.
func imuCommand(_ cmd: UInt8, _ data: [UInt8] = [], timeout: Double = 2) -> [UInt8]? {
    aaReply = nil
    broadcast(imuPacket(cmd, data))
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        CFRunLoopRunInMode(.defaultMode, 0.02, true)
        if let r = aaReply, r.cmd == cmd { return r.data }
    }
    return nil
}

func mcuCommand(_ cmd: UInt16, _ data: [UInt8] = [], timeout: Double = 2) -> [UInt8]? {
    fdReply = nil
    broadcast(mcuPacket(cmd, data))
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        CFRunLoopRunInMode(.defaultMode, 0.02, true)
        if let r = fdReply, r.cmd == cmd { return r.data }
    }
    return nil
}

// ---- modes -------------------------------------------------------------------------------
if wantInfo {
    // reply payload = 1 status byte + ascii text (ids per the Air-family MCU
    // protocol, wheaney/nrealAirLinuxDriver device_mcu.h)
    for (label, cmd) in [("serial       ", UInt16(0x15)), ("MCU app FW   ", 0x26),
                         ("DP7911 FW    ", 0x16), ("DSP app FW   ", 0x21)] {
        if let d = mcuCommand(cmd), d.count > 1 {
            print("\(label): \(String(bytes: d.dropFirst(), encoding: .ascii) ?? "?")")
        } else {
            print("\(label): no reply")
        }
    }
    exit(0)
}

if let path = configPath {
    _ = imuCommand(0x19, [0x00])                       // stream off while reading
    guard let lenReply = imuCommand(0x14), lenReply.count >= 4 else {
        print("config length request failed"); exit(1)
    }
    let total = Int(u32(lenReply, 0))
    var cfg = [UInt8]()
    while cfg.count < total {
        guard let part = imuCommand(0x15), !part.isEmpty else {
            print("config read stalled at \(cfg.count)/\(total)"); exit(1)
        }
        cfg.append(contentsOf: part)
    }
    try? Data(cfg.prefix(total)).write(to: URL(fileURLWithPath: path))
    print("factory calibration (\(total) bytes) -> \(path)")
    exit(0)
}

if let path = csvPath {
    FileManager.default.createFile(atPath: path, contents: nil)
    csvHandle = FileHandle(forWritingAtPath: path)
    csvHandle?.write(Data("ts_ns,counter,temp_raw,gx_dps,gy_dps,gz_dps,ax_g,ay_g,az_g\n".utf8))
}

guard imuCommand(0x19, [0x01]) != nil else {
    print("no ack for IMU stream enable"); exit(1)
}
print("IMU stream on (1 kHz). Ctrl+C to stop.")
let deadline = seconds.map { Date().addingTimeInterval($0) }
while deadline.map({ Date() < $0 }) ?? true {
    CFRunLoopRunInMode(.defaultMode, 0.25, true)
}
csvHandle?.closeFile()
print("captured \(totalSamples) samples")
