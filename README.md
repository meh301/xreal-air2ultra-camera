# XREAL Air 2 Ultra — stereo camera viewer

**Read the stereo tracking cameras of the XREAL Air 2 Ultra on macOS, Windows, Linux and Android — no SDK, no kext, no drivers.**

[日本語のREADMEはこちら](README.ja.md)

The Air 2 Ultra exposes its two tracking cameras as a standard USB Video Class (UVC)
device, but the video stream is **block-scrambled** and looks like noise in a normal
webcam viewer. This project descrambles it in real time and gives you a clean
640×480 stereo grayscale feed — the raw input you'd need for VIO/SLAM experiments,
robotics, or just poking at the hardware.

![clean stereo preview](docs/images/stereo_preview.png)

## Quick start

### Windows / Linux (and macOS too)

Requirements: Python 3.9+ with `numpy` and `opencv-python`, and an Air 2 Ultra
plugged in via USB-C. On Windows you also want `ffmpeg` on PATH
(`winget install ffmpeg`) — see the platform notes.

```sh
pip install numpy opencv-python
python python/preview_clean.py
```

A window opens with the live left/right camera feed, descrambled and denoised,
at the full 60 fps UVC rate. The XREAL stream is auto-detected among your
cameras (it is fingerprinted by its telemetry row, so other webcams are
skipped); `python python/xreal_uvc.py` scans and lists what was found.

Keys: `c` toggle clean/scrambled view · `s` save snapshot PNG · `space` pause · `q` quit

Every clean stereo pair is also published to a **named shared-memory
framebuffer** (`xreal_stereo_fb`, 960×640 grayscale + seqlock header), so your
own process — Python, C++, Rust, Unity, … — can consume the live feed with ~1 ms
latency and no sockets. `python python/xreal_fb.py --show` is a demo consumer;
the 64-byte header layout is documented in [python/xreal_fb.py](python/xreal_fb.py).
`--headless` runs the publisher without a window.

Platform notes:
- **Windows**: install ffmpeg (`winget install ffmpeg`). OpenCV's Windows
  backends cannot deliver this device's stream raw (measured: MSMF starts no
  stream, DirectShow force-converts to BGR), so the tools automatically capture
  through ffmpeg's dshow input instead, which grabs the native pin format.
- **Linux**: your user needs access to `/dev/video*` (usually the `video`
  group: `sudo usermod -aG video $USER`, then re-login). The glasses appear as
  a regular `uvcvideo` device; no udev rules needed just to view.

### macOS (native tools)

Requirements: Xcode Command Line Tools (`xcode-select --install`).

```sh
make
./preview_clean
```

macOS will ask for camera permission for your terminal on first run.

### Android

Android has no camera API for external UVC devices, so [`android/`](android/)
contains a native app (libusb + libuvc + the descrambler in C) that opens the
glasses through the USB host API. Build it with Android Studio or Gradle:

```sh
cd android
./fetch_deps.sh        # fetch_deps.ps1 on Windows — clones libusb/libuvc (pinned tags)
./gradlew :app:assembleDebug
```

Install the APK, plug the glasses into the phone's USB-C port, accept the USB
permission prompt, and the live stereo preview starts. See
[android/README.md](android/README.md) for details.

## Tools

| Tool | What it does |
|------|--------------|
| `python/preview_clean.py` | Cross-platform real-time stereo viewer (descramble + fixed-pattern-noise removal). Also `--snap out.png` for a headless snapshot, `--test in.pgm prefix` to verify descrambling offline. |
| `python/xreal_cam.py` | Cross-platform recorder: `python xreal_cam.py <numFrames> <outDir>` writes raw `cam0_*.pgm` / `cam1_*.pgm` (still scrambled) plus `meta.csv`. Output is identical to the macOS recorder. |
| `python/xreal_uvc.py` | Capture module used by the two tools (backend scan, ffmpeg fallback, telemetry fingerprint, byte-order fix). Run directly to scan for the XREAL stream. |
| `python/xreal_fb.py` | Shared-memory framebuffer: layout definition, writer/reader classes, and a demo consumer (`python xreal_fb.py --show`). |
| `python/xreal_imu.py` | **1 kHz IMU reader** (needs `pip install hidapi`): live console output, `--csv` logging, `--fb` shared-memory ring, and `--config calib.json` to dump the on-device factory calibration (IMU biases/noises + fisheye intrinsics and camera↔IMU extrinsics of both tracking cameras — everything VIO needs). |
| `preview_clean` (macOS) | Native Swift version of the viewer, 60 fps. Same keys and flags. |
| `xreal_cam` (macOS) | Native Swift version of the recorder. |
| `enumerate` (macOS) | List AVFoundation cameras and the XREAL device's formats. |
| `android/` | Android app: USB host API + libusb/libuvc, descramble + denoise in C, live side-by-side preview with snapshot. |
| `python/process_clean.py` | Offline pipeline: `python3 python/process_clean.py <capDir> <outDir>` descrambles + cleans a recording (from either recorder) into PNGs and a side-by-side `stereo_feed.mp4`. |
| `python/xreal_descramble.py` | Minimal single-frame descrambler, useful as a reference implementation. |
| `research/` | Reverse-engineering tools (vendor HID commands, UVC controls, USB descriptors). Most build on macOS and Linux. See [research/README.md](research/README.md). |

## How it works (short version)

- The glasses enumerate as a normal UVC webcam (`640×482 @ 60 fps`, nominally YUV but
  actually **one byte = one 8-bit mono pixel**). Rows 0–479 are the image, row 480 is
  telemetry, row 481 is padding.
- Each frame's 307,200 image bytes are shuffled as **128 blocks × 2,400 bytes** in a
  fixed permutation, with a phase that rotates every frame. Sync is recovered by
  finding the block that starts in the fisheye lens's black border.
- Consecutive UVC frames alternate between the two cameras. **The order is not fixed** —
  byte 58 of the telemetry row (`0x20`/`0x21`) tells you which camera a frame came from.
- The sensors are mounted rotated 90° (and 180° opposed to each other), so the
  descrambler also rotates; output is 480×640 portrait per eye.
- Remaining vertical stripes (column fixed-pattern noise) are estimated online and
  subtracted.
- Because the YUV label is fake, some UVC stacks deliver the two bytes of each 16-bit
  pair swapped. The telemetry markers make this detectable, and all capture paths in
  this repo normalize it automatically.

The glasses also stream a **1 kHz IMU** over a vendor HID interface and store
their complete factory calibration (fisheye intrinsics + camera↔IMU extrinsics
for both tracking cameras, IMU biases and noise densities) as JSON on the
device — `python/xreal_imu.py` reads both.

Full protocol notes (USB layout, telemetry row map, scramble algorithm, IMU
packet format and calibration blob, UVC exposure controls, vendor HID protocol,
per-OS capture notes): **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

## Credits

- The block-reorder table was discovered by
  [mazeasdamien/myXreal](https://github.com/mazeasdamien/myXreal) (`stereo_camera.cpp`).
- The vendor HID packet format is documented in
  [badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs).
- The Android app builds on [libusb](https://github.com/libusb/libusb) (LGPL-2.1,
  kept as its own shared library) and [libuvc](https://github.com/libuvc/libuvc) (BSD).
- Developed with [Claude Code](https://claude.com/claude-code) (Claude Fable 5) —
  the reverse-engineering analysis, tools, and documentation in this repo were built
  in collaboration with the AI agent.

## Disclaimer

This is an unofficial, reverse-engineered project, not affiliated with or endorsed by
XREAL. It only *reads* the camera stream over standard UVC — it sends no commands that
modify the device — but use it at your own risk.

## License

[MIT](LICENSE)
