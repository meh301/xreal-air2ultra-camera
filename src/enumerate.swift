import AVFoundation
import Foundation

func fourCC(_ code: FourCharCode) -> String {
    let bytes = [UInt8((code >> 24) & 0xff), UInt8((code >> 16) & 0xff),
                 UInt8((code >> 8) & 0xff), UInt8(code & 0xff)]
    return String(bytes: bytes, encoding: .ascii) ?? "?\(code)"
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.external, .builtInWideAngleCamera, .continuityCamera, .deskViewCamera],
    mediaType: .video,
    position: .unspecified)

print("=== Video capture devices ===")
for d in discovery.devices {
    print("\nName: \(d.localizedName)")
    print("  uniqueID: \(d.uniqueID)")
    print("  modelID:  \(d.modelID)")
    print("  manufacturer: \(d.manufacturer)")
    print("  transportType: \(d.transportType)")
    print("  connected: \(d.isConnected)")
    if d.localizedName.contains("UVC") || d.modelID.contains("13080") || d.manufacturer.contains("XREAL") {
        print("  --- FORMATS (\(d.formats.count)) ---")
        for f in d.formats {
            let desc = f.formatDescription
            let dims = CMVideoFormatDescriptionGetDimensions(desc)
            let sub = fourCC(CMFormatDescriptionGetMediaSubType(desc))
            let ranges = f.videoSupportedFrameRateRanges.map { "\($0.minFrameRate)-\($0.maxFrameRate)" }.joined(separator: ",")
            print("   \(dims.width)x\(dims.height) [\(sub)] fps:\(ranges)")
        }
    }
}
print("\n=== Authorization status: \(AVCaptureDevice.authorizationStatus(for: .video).rawValue) ===")
