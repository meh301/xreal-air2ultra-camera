// hid_read.swift — open every XREAL Air 2 Ultra HID interface and dump input reports.
// Build: swiftc hid_read.swift -o hid_read -framework IOKit -framework CoreFoundation
// Run:   ./hid_read [seconds]
import IOKit
import IOKit.hid
import Foundation

let VID = 0x3318, PID = 0x0426
let seconds = CommandLine.arguments.count > 1 ? Double(CommandLine.arguments[1]) ?? 5 : 5

var reportBuffers: [ObjectIdentifier: UnsafeMutablePointer<UInt8>] = [:]
var counts: [ObjectIdentifier: Int] = [:]
var firstHex: [ObjectIdentifier: Bool] = [:]

func devInfo(_ d: IOHIDDevice) -> String {
    func num(_ k: String) -> Int { (IOHIDDeviceGetProperty(d, k as CFString) as? Int) ?? -1 }
    let loc = num(kIOHIDLocationIDKey)
    let up = num(kIOHIDPrimaryUsagePageKey)
    let u = num(kIOHIDPrimaryUsageKey)
    let maxIn = num(kIOHIDMaxInputReportSizeKey)
    return "loc=0x\(String(loc, radix:16)) usagePage=0x\(String(up, radix:16)) usage=0x\(String(u, radix:16)) maxIn=\(maxIn)"
}

let inputCB: IOHIDReportCallback = { ctx, result, sender, type, reportID, report, length in
    let dev = Unmanaged<IOHIDDevice>.fromOpaque(sender!).takeUnretainedValue()
    let id = ObjectIdentifier(dev)
    counts[id, default: 0] += 1
    let n = counts[id]!
    // print first 3 reports fully, then just counts
    if n <= 3 {
        let bytes = (0..<min(length, 64)).map { String(format: "%02x", report[$0]) }.joined(separator: " ")
        print("[\(devInfo(dev))] report#\(n) id=\(reportID) len=\(length):\n    \(bytes)\(length>64 ? " ..." : "")")
    }
}

let mgr = IOHIDManagerCreate(kCFAllocatorDefault, IOOptionBits(kIOHIDOptionsTypeNone))
let match: [String: Any] = [kIOHIDVendorIDKey: VID, kIOHIDProductIDKey: PID]
IOHIDManagerSetDeviceMatching(mgr, match as CFDictionary)

let matchCB: IOHIDDeviceCallback = { ctx, result, sender, device in
    let maxIn = (IOHIDDeviceGetProperty(device, kIOHIDMaxInputReportSizeKey as CFString) as? Int) ?? 512
    let buf = UnsafeMutablePointer<UInt8>.allocate(capacity: max(1, maxIn))
    reportBuffers[ObjectIdentifier(device)] = buf
    let open = IOHIDDeviceOpen(device, IOOptionBits(kIOHIDOptionsTypeNone))
    print("MATCH \(devInfo(device)) open=\(open == kIOReturnSuccess ? "ok" : String(format:"err 0x%x",open))")
    IOHIDDeviceRegisterInputReportCallback(device, buf, max(1, maxIn), inputCB, nil)
}
IOHIDManagerRegisterDeviceMatchingCallback(mgr, matchCB, nil)
IOHIDManagerScheduleWithRunLoop(mgr, CFRunLoopGetCurrent(), CFRunLoopMode.defaultMode.rawValue)
let opened = IOHIDManagerOpen(mgr, IOOptionBits(kIOHIDOptionsTypeNone))
print("Manager open: \(opened == kIOReturnSuccess ? "ok" : String(format:"err 0x%x",opened))  — listening \(seconds)s")

CFRunLoopRunInMode(.defaultMode, seconds, false)
print("\n=== input report counts per interface ===")
for (id, c) in counts { print("  device \(id): \(c) reports in \(seconds)s") }
if counts.isEmpty { print("  (no input reports received)") }
