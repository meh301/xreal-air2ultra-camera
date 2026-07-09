# XREAL Air 2 Ultra stereo viewer — Android

Live descrambled preview of the Air 2 Ultra tracking cameras on an Android
phone, over the USB-C port.

Android's camera APIs don't expose external UVC devices, so this app talks to
the glasses directly:

```
UsbManager (permission + file descriptor)
        └─ libusb 1.0.27  (LIBUSB_OPTION_NO_DEVICE_DISCOVERY + wrapped fd)
             └─ libuvc 0.0.7  (stream negotiation, 640×241 fake-YUY2 @ 60 fps)
                  └─ xreal_core.c  (fingerprint, byte-order fix, descramble,
                                    FPN removal — same algorithms as the
                                    desktop tools)
                       └─ Kotlin UI (side-by-side bitmap, clean/raw toggle,
                                     snapshot)
```

## Build

Requirements: Android SDK + NDK (any recent one; set `ndk.version` in
[`gradle.properties`](gradle.properties) to the version you have), JDK 17+,
`git` on PATH.

```sh
./fetch_deps.sh          # Windows: powershell -File fetch_deps.ps1
./gradlew :app:assembleDebug
```

`fetch_deps` clones libusb v1.0.27 and libuvc v0.0.7 (pinned tags) into
`app/src/main/cpp/third_party/` — they are not vendored in the repo. The
native build uses ndk-build (no CMake needed); libusb is compiled as its own
shared library (`libusb1.0.so`) to keep the LGPL boundary clean, libuvc is
built without libjpeg since the stream is uncompressed.

Or open `android/` in Android Studio and run — but run `fetch_deps` first.

## Use

1. Install the APK (`adb install app/build/outputs/apk/debug/app-debug.apk`).
2. Plug the glasses into the phone's USB-C port. The app offers to open (or
   auto-opens) via the `USB_DEVICE_ATTACHED` filter; accept the USB permission
   prompt if one appears.
3. The portrait test screen fills up:
   - **top half** — the live stereo pair (left | right), descrambled and
     denoised. *Raw/Clean* toggles the view, *Snapshot* saves a PNG under
     `Android/data/org.air2ultra.stereocam/files/Pictures/`.
   - **bottom half** — two rolling oscilloscope graphs of the full 1 kHz IMU
     stream (angular rate and acceleration, ~4 s window, autoscaled; samples
     cross JNI in batches so nothing is dropped). The IMU is read from HID
     interface 2 over the same libusb handle as the camera; if claiming that
     interface fails the camera still works. A quaternion is still fused
     natively and available via `XrealNative.nativeGrabImu` — it's just not
     displayed.
4. When the phone drives the glasses' display (DisplayPort alt-mode), the
   clean stereo pair is also rendered onto it fullscreen as **SBS
   passthrough** (left camera → left eye; put the glasses in 3D/SBS mode).
   *Swap L/R* flips the eyes if the mapping looks inverted.

The app requires the current glasses firmware (MCU `12.1.00.498+`); older
firmware uses a different telemetry layout — update at
<https://ota.xreal.com/ultra-update?version=1>.

The phone must support USB host mode (any modern phone does). The glasses
also drive the display via DisplayPort alt-mode at the same time; the camera
stream is independent of that.

## Status

The native pipeline is identical to the validated desktop implementation
(bit-exact port, verified against the Python reference), and the
libusb/libuvc wiring follows the upstream-documented Android flow. It has
**not yet been tested against the glasses on a physical phone** — if
streaming fails on your device, `adb logcat -s xrealcam` shows where
(permission, stream negotiation, or bandwidth). Isochronous bandwidth is the
most likely failure point on older phones or through hubs; plug the glasses
in directly.
