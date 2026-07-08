// hid_cmd.swift — send an MCU command to the XREAL Air 2 Ultra vendor-HID interfaces and
// dump all input reports. Validates our packet builder + CRC by fetching the serial number.
//
// Build: swiftc hid_cmd.swift -o hid_cmd -framework IOKit -framework CoreFoundation
// Run:   ./hid_cmd <cmdIdHex> [dataHex] [listenSeconds] [sendLen]
//   e.g. ./hid_cmd 15            -> get serial number (cmd 0x15, no data)
//        ./hid_cmd 19 01         -> enable IMU stream
import IOKit
import IOKit.hid
import Foundation

let VID = 0x3318, PID = 0x0426

// ---- CRC-32 (zlib/PKZIP) ----
let crcTable: [UInt32] = {
    (0..<256).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1 }
        return c
    }
}()
func crc32(_ buf: ArraySlice<UInt8>) -> UInt32 {
    var r: UInt32 = 0xffffffff
    for b in buf { r = (r >> 8) ^ crcTable[Int((r ^ UInt32(b)) & 0xff)] }
    return r ^ 0xffffffff
}

// ---- build 64-byte MCU packet ----
func mcuPacket(cmdId: UInt16, data: [UInt8]) -> [UInt8] {
    var p = [UInt8](repeating: 0, count: 64)
    p[0] = 0xfd
    let length = UInt16(data.count + 17)
    func put16(_ v: UInt16, _ o: Int) { p[o] = UInt8(v & 0xff); p[o+1] = UInt8(v >> 8) }
    func put32(_ v: UInt32, _ o: Int) { for k in 0..<4 { p[o+k] = UInt8((v >> (8*UInt32(k))) & 0xff) } }
    put16(length, 5)
    put32(0x1337, 7)      // request_id
    put32(0, 11)          // timestamp
    put16(cmdId, 15)      // cmd_id
    for (i, b) in data.enumerated() { p[22 + i] = b }
    let crc = crc32(p[5..<(5 + Int(length))])
    put32(crc, 1)
    return p
}

// ---- args ----
let a = CommandLine.arguments
guard a.count >= 2, let cmdId = UInt16(a[1], radix: 16) else {
    print("usage: hid_cmd <cmdIdHex> [dataHex] [seconds] [sendLen]"); exit(1)
}
let data: [UInt8] = a.count >= 3 ? stride(from: 0, to: a[2].count, by: 2).compactMap {
    let s = a[2].index(a[2].startIndex, offsetBy: $0)
    let e = a[2].index(s, offsetBy: 2, limitedBy: a[2].endIndex) ?? a[2].endIndex
    return UInt8(a[2][s..<e], radix: 16)
} : []
let seconds = a.count >= 4 ? Double(a[3]) ?? 3 : 3
let sendLen = a.count >= 5 ? Int(a[4]) ?? 64 : 64

let packet = mcuPacket(cmdId: cmdId, data: data)
print("Packet (cmd 0x\(String(cmdId,radix:16)) data \(data.map{String(format:"%02x",$0)}.joined())):")
print("  " + packet.prefix(28).map { String(format: "%02x", $0) }.joined(separator: " ") + " ...")

// ---- open all matching HID devices, index them ----
var devices: [IOHIDDevice] = []
var indexOf: [ObjectIdentifier: Int] = [:]
var buffers: [UnsafeMutablePointer<UInt8>] = []

func asciiOf(_ p: UnsafePointer<UInt8>, _ n: Int) -> String {
    String((0..<n).map { c -> Character in let v = p[c]; return (v >= 32 && v < 127) ? Character(UnicodeScalar(v)) : "." })
}

let inputCB: IOHIDReportCallback = { _, _, sender, _, reportID, report, length in
    let dev = Unmanaged<IOHIDDevice>.fromOpaque(sender!).takeUnretainedValue()
    let idx = indexOf[ObjectIdentifier(dev)] ?? -1
    let hex = (0..<min(length, 40)).map { String(format: "%02x", report[$0]) }.joined(separator: " ")
    let asc = asciiOf(report, min(length, 40))
    print("  IN[dev\(idx)] len=\(length) id=\(reportID): \(hex)   |\(asc)|")
}

let mgr = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
IOHIDManagerSetDeviceMatching(mgr, [kIOHIDVendorIDKey: VID, kIOHIDProductIDKey: PID] as CFDictionary)
IOHIDManagerRegisterDeviceMatchingCallback(mgr, { _, _, _, device in
    let up = (IOHIDDeviceGetProperty(device, kIOHIDPrimaryUsagePageKey as CFString) as? Int) ?? 0
    if up != 0x41 { return }                         // only the vendor control interfaces
    let idx = devices.count
    devices.append(device)
    indexOf[ObjectIdentifier(device)] = idx
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: 512)
    buffers.append(buf)
    IOHIDDeviceOpen(device, IOOptionBits(kIOHIDOptionsTypeNone))
    IOHIDDeviceRegisterInputReportCallback(device, buf, 512, inputCB, nil)
    print("opened dev\(idx) (usagePage 0x41)")
}, nil)
IOHIDManagerScheduleWithRunLoop(mgr, CFRunLoopGetCurrent(), CFRunLoopMode.defaultMode.rawValue)
IOHIDManagerOpen(mgr, IOOptionBits(kIOHIDOptionsTypeNone))
CFRunLoopRunInMode(.defaultMode, 0.5, false)          // let matching callbacks run

// ---- send the command to every 0x41 interface ----
for (i, dev) in devices.enumerated() {
    let r = packet.withUnsafeBufferPointer {
        IOHIDDeviceSetReport(dev, kIOHIDReportTypeOutput, 0, $0.baseAddress!, sendLen)
    }
    print("SetReport -> dev\(i): \(r == kIOReturnSuccess ? "ok" : String(format:"err 0x%x",r))")
}

print("listening \(seconds)s ...")
CFRunLoopRunInMode(.defaultMode, seconds, false)
print("done.")
