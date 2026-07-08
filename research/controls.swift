import AVFoundation
import Foundation
guard let d = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.external, .continuityCamera], mediaType: .video, position: .unspecified)
    .devices.first(where: { $0.localizedName.contains("UVC Camera") || $0.manufacturer.contains("XREAL") }) else {
    print("not found"); exit(1) }
print("Device: \(d.localizedName)")
print("exposure modes supported: locked=\(d.isExposureModeSupported(.locked)) custom=\(d.isExposureModeSupported(.custom)) auto=\(d.isExposureModeSupported(.autoExpose)) continuous=\(d.isExposureModeSupported(.continuousAutoExposure))")
print("current exposureMode raw: \(d.exposureMode.rawValue)")
print("focus modes: locked=\(d.isFocusModeSupported(.locked)) continuous=\(d.isFocusModeSupported(.continuousAutoFocus))")
print("whiteBalance auto supported: \(d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance))")
print("activeFormat: \(d.activeFormat.formatDescription)")
