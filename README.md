# XREAL Air 2 Ultra — stereo cameras & IMU, no SDK

**Everything the Air 2 Ultra's sensors offer, on Windows, Linux, macOS and
Android: the two tracking cameras descrambled in real time, the 1 kHz IMU, and
the factory calibration stored on the device — no SDK, no kext, no drivers.**

[日本語のREADMEはこちら](README.ja.md)

The glasses expose their stereo tracking cameras as a standard UVC webcam, but
the stream is **block-scrambled** and looks like noise in a normal viewer. This
project descrambles it live into a clean 640×480 stereo grayscale feed, reads
the IMU that shares the same nanosecond clock, and pulls the fisheye
intrinsics + camera↔IMU extrinsics off the device — the complete raw input for
VIO/SLAM, robotics, or just poking at the hardware.

![clean stereo preview](docs/images/stereo_preview.png)

## Quick start

### Windows / Linux / macOS (Python)

```sh
pip install numpy opencv-python hidapi
python python/preview_clean.py       # live stereo viewer, 60 fps
python python/xreal_imu_scope.py     # IMU oscilloscope + 3D attitude
```

Viewer keys: `c` clean/scrambled · `s` snapshot · `space` pause · `q` quit.
The XREAL is auto-detected among your cameras by its telemetry fingerprint.
On Windows also install ffmpeg (`winget install ffmpeg`); on Linux the IMU
needs a udev rule — both explained under [Platform notes](#platform-notes).

### macOS (native tools)

```sh
xcode-select --install   # once
make
./preview_clean          # stereo viewer (same keys/flags as the Python one)
./xreal_imu              # IMU reader (--csv / --config / --info)
./glasses_passthrough    # camera feed onto the glasses' own displays (SBS)
```

macOS asks for camera permission for your terminal on first run. The Python
tools above work on macOS too.

### Android

Android's camera APIs can't reach external UVC devices, so
[`android/`](android/) is a native app (libusb + libuvc + the descrambler in
C) built on the USB host API:

```sh
cd android
./fetch_deps.sh          # fetch_deps.ps1 on Windows — clones libusb/libuvc (pinned tags)
./gradlew :app:assembleDebug
```

Install the APK and plug the glasses into the phone: live stereo preview plus
a 1 kHz IMU readout with fused orientation. Details:
[android/README.md](android/README.md).

## Tools

**Cameras**

| Command | What it does |
|---------|--------------|
| `python python/preview_clean.py` | Real-time stereo viewer; publishes every clean pair to the shared-memory framebuffer. `--headless` (no window), `--snap out.png` (one snapshot), `--test in.pgm pfx` (offline descramble check). |
| `python python/xreal_cam.py <N> <dir>` | Record N raw (still scrambled) frames as `cam{0,1}_*.pgm` + `meta.csv`; output identical to the macOS recorder. |
| `python python/process_clean.py <dir> <out>` | Offline pipeline: descramble + denoise a recording into PNGs and a side-by-side `stereo_feed.mp4`. |
| `python python/xreal_uvc.py` | Scan and diagnose capture backends (this file is also the capture library the other tools import). |

**IMU & calibration**

| Command | What it does |
|---------|--------------|
| `python python/xreal_imu.py` | 1 kHz gyro/accel reader. `--quat` adds a host-side quaternion, `--csv` logs every sample, `--fb` publishes the shared-memory ring, `--config c.json` dumps the factory calibration, `--info` prints serial + firmware versions. |
| `python python/xreal_imu_scope.py` | Rolling oscilloscope (gyro + accel) with a 3D attitude view; `b` re-zeros the gyro bias. Runs alongside the camera viewer. |

**Native macOS binaries** — `make` builds `preview_clean`, `xreal_cam`,
`xreal_imu` and `enumerate` (device/format lister), mirroring their Python
counterparts, plus one macOS-only tool:

| Command | What it does |
|---------|--------------|
| `./glasses_passthrough` | Stereo passthrough onto the glasses themselves — left camera → left eye, right camera → right eye, fullscreen side-by-side on the XREAL display (put the glasses in 3D/SBS mode). `--list` shows displays, `--display N` picks one, `--window` previews without glasses. Keys: `x` swap eyes · `r` rotate · `m` mirror · `s` toggle SBS. |

**Reference & research** — [`python/xreal_descramble.py`](python/xreal_descramble.py)
is the minimal single-frame descrambler other implementations are checked
against; [`research/`](research/README.md) holds the reverse-engineering tools
(vendor HID commands, UVC controls, USB descriptors).

## Using the data from your own code

Built for feeding a VIO/SLAM pipeline without touching USB yourself:

- **Camera framebuffer** — every clean stereo pair lands in named shared
  memory `xreal_stereo_fb` (960×640 grayscale, seqlock header carrying the
  pair counter and the device exposure timestamp). ~1 ms latency, any
  language. Demo consumer: `python python/xreal_fb.py --show`; the 64-byte
  header layout is documented in [python/xreal_fb.py](python/xreal_fb.py).
- **IMU ring** — `xreal_imu.py --fb` publishes every sample to the
  `xreal_imu_fb` ring (layout in [python/xreal_imu.py](python/xreal_imu.py)).
- **One clock** — camera exposure timestamps and IMU timestamps are the same
  free-running nanosecond counter (measured on-device; the camera carries the
  low 32 bits). No cross-clock calibration needed; unwrap recipe in
  [PROTOCOL.md](docs/PROTOCOL.md#clock-domains-cameras-vs-imu).
- **Factory calibration** — `xreal_imu.py --config` dumps fisheye624
  intrinsics and camera↔IMU extrinsics for both cameras plus IMU biases and
  noise densities: a ready-made VIO parameter set. (It includes your device
  serial — keep dumps out of public repos.)
- **Orientation** — the device streams raw inertial data only; quaternions
  come from the host-side Madgwick filter in
  [python/xreal_ahrs.py](python/xreal_ahrs.py).

## How it works (short version)

- The glasses enumerate as a normal UVC webcam (`640×482 @ 60 fps`, nominally
  YUV but actually **one byte = one 8-bit mono pixel**). Rows 0–479 are the
  image, row 480 is telemetry, row 481 is padding.
- Each frame's 307,200 image bytes are shuffled as **128 blocks × 2,400 bytes**
  in a fixed permutation whose phase rotates every frame. Sync is recovered by
  finding the block that starts in the fisheye lens's black border.
- Consecutive UVC frames alternate between the two cameras, but **the order is
  not fixed** — a telemetry byte identifies the camera.
- The sensors are mounted rotated 90° (and 180° opposed to each other), so the
  descrambler also rotates; output is 480×640 portrait per eye.
- Remaining vertical stripes (column fixed-pattern noise) are estimated online
  and subtracted.
- Telemetry comes in two **firmware dialects** (different metadata layouts):
  dialect B is the current firmware (`12.1.00.498`, confirmed latest by the
  official updater), dialect A is older. Some UVC stacks additionally
  byte-swap the fake-YUV pairs. Every capture path in this repo detects and
  normalizes all of it automatically.

Full protocol documentation — USB layout, telemetry dialects (and which
firmware versions they map to), scramble algorithm, IMU packet format,
calibration blob, clock domains, UVC exposure controls, vendor HID protocol,
per-OS capture notes: **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

## Platform notes

- **Windows**: install ffmpeg (`winget install ffmpeg`). OpenCV's Windows
  backends cannot deliver this device's stream raw (MSMF starts no stream,
  DirectShow force-converts to BGR), so the tools capture through ffmpeg's
  dshow input, which grabs the native pin format.
- **Linux**: camera viewing needs `/dev/video*` access (usually the `video`
  group: `sudo usermod -aG video $USER`, then re-login). IMU access needs
  readable hidraw nodes: `sudo cp linux/99-xreal-air2ultra.rules
  /etc/udev/rules.d/ && sudo udevadm control --reload && sudo udevadm trigger`,
  then replug.
- **macOS**: grant the camera permission prompt on first run of the viewer.

## Credits

- The block-reorder table was discovered by
  [mazeasdamien/myXreal](https://github.com/mazeasdamien/myXreal)
  (`stereo_camera.cpp`) — also worth a look as a Windows-native C++ sibling
  project (ImGui dashboard, stereo rectification, VR scene on the glasses).
- The vendor HID packet formats are documented in
  [badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs) and
  [wheaney/nrealAirLinuxDriver](https://gitlab.com/wheaney/nrealAirLinuxDriver).
- The Android app builds on [libusb](https://github.com/libusb/libusb)
  (LGPL-2.1, kept as its own shared library) and
  [libuvc](https://github.com/libuvc/libuvc) (BSD).
- Developed with [Claude Code](https://claude.com/claude-code) (Claude Fable 5) —
  the reverse-engineering analysis, tools, and documentation in this repo were
  built in collaboration with the AI agent.

## Disclaimer

This is an unofficial, reverse-engineered project, not affiliated with or
endorsed by XREAL. It only reads the sensor streams and sends the documented,
community-validated query/enable commands — but use it at your own risk.

## License

[MIT](LICENSE)
