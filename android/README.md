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

### Basalt VIO backend (research branch)

The vSLAM backend is [Basalt (Monado fork)](https://gitlab.freedesktop.org/mateosss/basalt),
compiled to `libbasalt.so` for arm64-v8a and loaded at runtime through the
VIT interface. Build it once with:

```powershell
powershell -ExecutionPolicy Bypass -File android\build_basalt.ps1
```

which downloads its own toolchain (NDK r27c, CMake, Ninja → `~/Android/toolchains`,
~4 GB), builds oneTBB/fmt/basalt against the OpenCV Android SDK, and drops
the stripped libraries into `app/src/main/jniLibs/arm64-v8a/`. Rebuild the
APK afterwards. Without this step the app still builds and runs — the pose
view then says "Basalt not loaded" and a built-in tracker draws the points.

## Use

1. Install the APK (`adb install app/build/outputs/apk/debug/app-debug.apk`).
2. Plug the glasses into the phone's USB-C port. The app offers to open (or
   auto-opens) via the `USB_DEVICE_ATTACHED` filter. **Grant the camera and
   microphone prompts** — the glasses are a composite USB device containing a
   UVC camera and microphones, and Android 10+ refuses USB access to such a
   device unless the app holds those runtime permissions (nothing is captured
   through Android's own camera/audio APIs). Then accept the USB permission
   dialog if one appears.
3. The portrait research screen fills up (this branch is the vSLAM research
   app; the stable test app lives on `main`):
   - **top half** — *tracking* pane (left camera with the tracked features
     as green dots) | *depth* pane (stereo disparity, colorized blue=far →
     red=near, computed at the sensors' 30 Hz on the SLAM worker thread).
   - **bottom half** — the 3D pose/map view: ground grid, world axes, the
     headset drawn as a frustum at its current pose plus a breadcrumb
     trail. Orientation comes from the on-device AHRS; position stays at
     the origin until the Basalt VIO backend lands (docs/VSLAM.md).
   - **buttons, row 1** — *Rst* resets the SLAM system (pose to origin,
     features and trail dropped), *Pts* shows/hides the tracked features
     (phone pane and glasses overlay; green = Basalt's own optical-flow
     keypoints, orange = the built-in fallback tracker used only when
     libbasalt.so is absent), *Dep* toggles the stereo-depth computation,
     *Snap* saves a PNG of the panes to Pictures/XREAL.
   - **buttons, row 2** — *Eye:…* cycles what the glasses show in 3D mode:
     **Cam** (camera passthrough) → **Dep** (colorized stereo depth,
     world-aligned per eye) → **AR** (black background, tracked points
     floating over the real world) → **Off**; *L|Dep / L|R* switches the
     phone's right pane between the depth map and the right camera;
     *SBS/3D* switches the glasses between plain side-by-side and
     calibrated per-eye stereo.
4. When the phone drives the glasses' display (DisplayPort alt-mode), the
   world-aligned passthrough renders onto it natively (front-buffer GLES +
   IMU timewarp, always on), with the tracked features overlaid as points
   that ride the same timewarp as the image.

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
