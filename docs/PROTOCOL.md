# XREAL Air 2 Ultra camera protocol notes

Reverse-engineering notes for the tracking-camera path. Everything below was measured
on a retail Air 2 Ultra on macOS; no official documentation exists.
(Capturing the same stream on Windows/Linux/Android: see
[Capturing on other platforms](#capturing-on-other-platforms) at the end.)

## USB device layout

```
idVendor  0x3318 (13080)  "XREAL"
idProduct 0x0426 (1062)   "XREAL Air 2 Ultra"
```

Composite device (bDeviceClass 239, IAD), one configuration:

| IF | Class/SubClass | Meaning |
|----|----------------|---------|
| 0, 1, 2 | HID (3) | vendor HID, UsagePage 0x41 — MCU control / IMU / 3DoF |
| **3** | **Video Control (14/1)** | UVC VideoControl — "UVC Camera 0" |
| **4** | **Video Streaming (14/2)** | UVC VideoStreaming (camera pixels) |
| 5, 6, 7 | Audio (1) | audio control + streaming |
| 8 | HID (3) | vendor HID |

Only interfaces 3+4 are needed for the camera; AVFoundation binds them automatically.
No vendor "unlock" command is required — the UVC stream is live on enumeration.

## UVC stream format

AVFoundation delivers a `2vuy` (UYVY) buffer of 640×241 (308,480 bytes). The YUV label
is a lie: **every byte is one 8-bit monochrome pixel**. Treated as bytes, the frame is a
tight 640×482 grayscale raster:

```
rows 0..479   640x480 camera image (block-scrambled, see below)
row  480      telemetry row
row  481      constant 0x5C padding
```

### Telemetry row (row 480)

The layout below is what the **current firmware** produces — MCU app FW
`12.1.00.498_20241115` (DSP `12.1.00.001_1.0.1.7_1.1`), confirmed the latest
available by the official updater (checked 2026-07). It is the only layout
these tools support; parsing is implemented in
`python/xreal_uvc.py::_classify` and the Android core.

| col | Meaning |
|-----|---------|
| 0–3 | **u32 LE exposure timestamp [ns]** — low 32 bits of the device clock, **the same timebase as the IMU stream** (see [Clock domains](#clock-domains-cameras-vs-imu)); wraps every 4.295 s |
| 4 | constant `0x64` |
| 18 | **pair counter** — shared by both frames of a stereo pair |
| 22–27 | auxiliary payload, present only on aux frames (see below) |
| 34 | constant `0x01` |
| 51–56 | **frame dimensions as LE u16: 640, 480, 640** — use as the fingerprint |
| 59 | **camera bit: `0x01` = cam0 = the physical RIGHT camera, `0x00` = cam1 = LEFT** (verified on-device; all viewers draw cam1 in the left pane) |
| 60 | constant `0x80` |
| 62, 63 | per-camera values (purpose unknown) |

Quirk: **every 2nd frame of the bit-1 camera is an "aux" frame** whose row 481
is *not* `0x5C` padding but carries data (as do cols 22–27 that frame) —
purpose unknown (IMU/temperature/config echo?). The image rows are unaffected.
Fingerprints must therefore not require the padding row on every frame.

Consecutive UVC frames alternate between the two cameras and a pair shares one
counter value (~30 stereo pairs/s at 60 fps UVC). **Do not infer camera
identity from arrival order** — frame drops shift it, and pre-update firmware
was even observed delivering pairs in inconsistent order. The camera bit is
the only reliable discriminator (verified against descramble orientation and,
historically, image-correlation ground truth).

**Older firmware** used a different telemetry layout (counter at col 19,
camera ID `0x20`/`0x21` at col 58, marker bytes `0xAD,0xDA` at cols 22–23, no
known timestamp field). It is **not supported** — the tools print a hint when
they see it. Update the glasses at
<https://ota.xreal.com/ultra-update?version=1>.

## Block scrambling

The 307,200 image bytes are transmitted as **128 blocks × 2,400 bytes**, permuted by a
fixed reorder table (`REORDER` in [`python/xreal_descramble.py`](../python/xreal_descramble.py);
table discovered by [mazeasdamien/myXreal](https://github.com/mazeasdamien/myXreal)).
The permutation itself never changes, but its **phase rotates from frame to frame**:
logical block `t` lives in raw block `REORDER[(align + t) % 128]`, with `align` unknown
per frame.

**Phase sync:** logical block 0 always starts at the sensor's first readout line, which
lies in the fisheye lens's black vignette border. The raw block whose first 128 bytes
have the smallest sum is therefore block 0; looking it up in `REORDER` yields `align`.
(This could theoretically mis-sync on a frame with a region darker than the vignette,
but in practice it is rock solid.)

**Pixel order:** after un-permuting, the byte stream is the sensor read out in
640-byte lines that correspond to *columns* of the upright image — the sensors are
mounted rotated 90°, and the two cameras are mounted 180° opposed to each other. The
descrambler's LUT therefore transposes into a 480×640 portrait raster, using
`(y, x) = (c, 479−r)` for cam0 and `(639−c, r)` for cam1 (line index `r`, byte
within line `c`). The horizontal sense was verified against the real scene
on-device — the mapping originally lifted from myXreal (`(c, r)` / `(639−c,
479−r)`) produced horizontally mirrored images.

## Sensor artifacts and cleanup

- **8-tap column fixed-pattern noise**: vertical stripes, static per camera.
  Removed by estimating the per-column median of a horizontal high-pass, accumulated
  across frames (EMA online in `preview_clean`, median across frames offline).
- Mild per-frame row banding (temporal illumination), removed per frame the same way
  along rows.

## UVC controls (video-control interface 3)

Readable/writable via control transfers on ep0 (e.g. libusb) *while* AVFoundation is
streaming, without claiming the interface. Tools: `research/uvc_ctl.c`,
`research/get_uvc_desc.c`.

- **Camera Terminal (unit 1):** AutoExposureMode (cs `0x02`) — `0x04` ShutterPriority
  accepted, `0x01` full-manual rejected; ExposureTimeAbs (cs `0x04`) — range 1..159 =
  0.1–15.9 ms, default 60, must be set while streaming, brightness follows it exactly;
  Zoom; Roll.
- **Processing Unit (unit 2):** Brightness, Contrast, …, PowerLineFrequency.
- **Extension Unit (unit 4):** GUID `41769ea2-04de-e347-8b2b-f4341aff003b`, selectors
  unknown (all probes returned PIPE errors). Likely where OV580-style sensor/strobe
  configuration lives. **Open question.**

## Vendor HID protocol (MCU control, interface 0)

Packet format (64 bytes, `IOHIDDeviceSetReport` Output, report ID 0), per
[badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs):

```
[0]      0xFD
[1..5]   CRC32 (zlib, little-endian) over bytes [5 .. 5+length]
[5..7]   length = data.len + 17
[7..11]  request_id (echoed in reply)
[11..15] timestamp (0 ok)
[15..17] cmd_id
[17..22] reserved
[22..]   data
```

Validated on-device: cmd `0x15` returns the internal serial number; cmd `0x19` with
data `01` enables the IMU stream (ack `0x04`). Tools: `research/hid_cmd.swift`,
`research/hid_read.swift`, `research/cam_probe.swift`.

## IMU stream (vendor HID interface 2)

Fully decoded and validated on-device (`python/xreal_imu.py`; framing per
[badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs)).
Interface 2 speaks a shorter sibling of the `0xFD` protocol:

```
[0]     0xAA
[1..5]  CRC32 (zlib, little-endian) over bytes [5 .. 5+length]
[5..7]  length = data.len + 3
[7]     cmd_id
[8..]   data
```

| cmd | Meaning |
|-----|---------|
| `0x19` data `01`/`00` | IMU stream on / off (reply data `00`) |
| `0x14` | u32 byte length of the **factory calibration JSON** |
| `0x15` | next chunk of that JSON |

With the stream on, 512-byte input reports arrive at **1000 Hz** (measured
1000.0 Hz, no drops):

| bytes | Content |
|-------|---------|
| 0–1 | signature `01 02` |
| 2–3 | u16 temperature, raw (~1080 warm; scale unverified) |
| 4–11 | u64 timestamp [ns] |
| 12–13, 14–17 | u16 multiplier, u32 divisor for the gyro |
| 18–26 | 3 × i24 gyro x,y,z — `raw*mul/div` = deg/s |
| 27–28, 29–32 | u16 multiplier, u32 divisor for the accelerometer |
| 33–41 | 3 × i24 accel x,y,z — `raw*mul/div` = g |
| 42–47 | u16 mul + u32 div for the magnetometer |
| 48–53 | 3 × i16 mag x,y,z — **read 0 on the tested unit** |
| 54–57 | u32 secondary counter, 976,562 ns/tick = the sensor's internal 1024 Hz ODR |
| 62 | flag, 0/1 (~40% of reports 1; meaning unknown) |
| 63 | small cycling counter, 0–7 (meaning unknown) |
| 64–65 | u16, ~0xC09x, rises slowly as the device warms — possibly a second temperature/voltage |
| rest | zero (verified over 5,000 reports) |

Values are in the raw sensor frame (at rest: \|accel\| = 1.010 g, gyro bias
≈ (−1.0, −0.5, 0.0) deg/s on the tested unit). ar-drivers-rs maps to its
world convention as `(-x, z, y)` and subtracts the config biases.

**Sensor frame vs the factory calibration frame** (established on-device,
research branch): every quantity in the calibration JSON (`imu_q_cam`,
`imu_p_cam`, the biases) lives in a camera-style device frame — x right,
y **down**, z **forward** (the display extrinsics pin it). The raw stream
does NOT use that frame:

- the **accelerometer** streams ENU-style — x right, y forward, z up
  (worn level it reads +1 g on z): `a_dev = (ax, −az, ay)`;
- the **gyroscope** is additionally **mirrored** (opposite rotation
  handedness — the same firmware quirk family as the mirrored camera
  stream): with the accel mapping applied to it, attitude starts correct
  but every rotation runs backwards. It needs its own mapping:
  `w_dev = (−gx, gz, −gy)`.

Any consumer fusing this IMU against the factory calibration (VIO,
timewarp) must apply both remaps; fusing gyro against accel alone (plain
AHRS) "works" with a shared mapping but yields a mirrored world.

**The device streams raw inertial data only — there is no onboard
orientation/quaternion output.** The vendor stack fuses host-side, and so
does this repo: `python/xreal_ahrs.py` (6-axis Madgwick + gyro-bias capture)
powers `xreal_imu.py --quat` and the attitude view in `xreal_imu_scope.py`.
Other things the glasses can report: the MCU channel (interface 0) emits
async events — cmd `0x6c05` = button press per ar-drivers-rs (interfaces 0/1
are otherwise silent until spoken to; passively confirmed), and interface 8
is a standard HID consumer-control endpoint (the brightness/volume buttons as
media keys, handled by the OS).

### Factory calibration JSON

The `0x14`/`0x15` blob (~55 kB) is the complete VIO calibration set,
written at the factory:

- `IMU.device_1`: `accel_bias`, `gyro_bias` (+ a per-temperature bias table),
  `imu_noises` = [gyro noise, gyro walk, accel noise, accel walk],
  `accel_q_gyro`, `gyro_q_mag` (JPL quaternions).
- `SLAM_camera.device_1/device_2`: **both tracking cameras**, `camera_model:
  fisheye624`, `fc`/`cc`/`kc` intrinsics at `resolution [480, 640]` (exactly
  the descrambled output of this repo), and `imu_p_cam`/`imu_q_cam`
  camera↔IMU extrinsics (translation [m] + JPL quaternion). The two camera
  positions imply a ~13.7 cm stereo baseline.
- `display`: per-eye virtual-display intrinsics + pose, and a
  `display_distortion` mesh.
- `RGB_camera.num_of_cameras = 0` (no RGB camera on the Ultra).

`python python/xreal_imu.py --config calib.json` dumps it. Note it contains
the device serial (`FSN`).

### Clock domains: cameras vs. IMU

Validated on-device: **the camera exposure timestamps and the IMU timestamps
are the same clock** — a free-running nanosecond counter (starts near device
power-on).

- IMU reports carry the full u64 ns value; camera frames carry the low u32
  (telemetry cols 0–3), wrapping every 2³² ns = 4.295 s.
- Measurement (12 s, 720 frames, 13k IMU samples, both streams host-stamped
  with the same monotonic clock): every camera stamp lands `(IMU clock at
  arrival − stamp) = 70…104 ms`, i.e. one tight cluster = USB/driver/pipe
  delivery latency; ±6.6 ms spread is delivery jitter, not stamp jitter. A
  different clock would have spread the differences uniformly over 4.295 s.
- The stamps are hardware exposure times: the two frames of a stereo pair are
  stamped **within ~16 µs of each other** (median; the sensors are triggered
  together), and same-camera spacing is exactly 33.333 ms.
- Unwrap recipe for VIO: with any recent IMU u64 `t_imu`, the full camera
  time is the value congruent to the u32 stamp (mod 2³²) nearest to
  `t_imu − latency`. Since the wrap period (4.295 s) vastly exceeds the
  latency (<0.15 s), this is unambiguous:
  `t_cam = t_imu − ((t_imu − stamp32) mod 2³²)`, taking the mod into
  `[0, 2³²)`.
- So cameras and IMU need **no cross-clock calibration** — feed both
  timestamps straight into a VIO pipeline (the camera-side latency does not
  matter; only stamp correctness does).

## Capturing on other platforms

The stream is plain UVC, so any OS UVC stack can deliver it — the only trick is
getting the **untouched bytes** out, because the nominal `640×241 @ 60 fps`
"YUY2/UYVY" format is a lie (every byte is one mono pixel of the 640×482 raster).

- **OpenCV** (Linux V4L2, macOS AVFoundation): open the device and set
  `CAP_PROP_CONVERT_RGB = 0`; `read()` then returns the raw 308,480-byte buffer
  instead of a decoded BGR image. Never request a different width/height — that
  would insert a converter. `python/xreal_uvc.py` implements this, plus device
  auto-detection.
- **Windows**: measured on OpenCV 5.0, *neither* backend can deliver this
  device raw — MSMF opens it (properties readable) but every `ReadSample`
  fails, and DirectShow streams but force-converts to BGR24 even with
  `CONVERT_RGB=0` accepted (conversion clips the fake-YUV bytes; the data is
  unrecoverable). What does work: **ffmpeg's dshow input**, which connects the
  capture pin at its native `yuyv422 640×241` type and pipes untouched frames
  (`ffmpeg -f dshow -video_size 640x241 -pixel_format yuyv422 -i video=<dev>
  -f rawvideo -`). `xreal_uvc.py` prefers this path on Windows automatically.
- **Android**: no external-UVC camera API exists; `android/` opens the device via
  the USB host API and hands the fd to libusb (`LIBUSB_OPTION_NO_DEVICE_DISCOVERY`
  set before init, then `uvc_wrap`) with libuvc doing the stream negotiation.
- **Byte order**: because the fourcc is fake, a stack that "converts" between
  YUY2 and UYVY swaps the two bytes of each 16-bit pair without touching real
  luma. Whether you get wire order or swapped order is backend-dependent.
  Detect it from the telemetry row: `0xAD,0xDA` at cols 22,23 = native order;
  `0xDA,0xAD` = swap every 16-bit pair before descrambling (`0x19` moving from
  col 5 to col 4 confirms it). Row 481 being constant `0x5C` is order-invariant
  and makes a good device fingerprint on systems with several cameras.
- **Frame shape**: depending on backend, the raw frame arrives as `241×1280`,
  `241×640×2`, `1×308480`, etc. Only the byte count (308,480) matters.

## Open questions

- Extension Unit 4 selectors (sensor/IR-strobe/SLAM-mode configuration).
- Fusing IMU + stereo into VIO/SLAM — all raw ingredients are now available
  (1 kHz IMU, ~30 Hz stereo pairs, factory intrinsics/extrinsics/noise
  parameters).
- Why the magnetometer fields read zero (no sensor populated, or needs an
  enable command?).
- The exact scale of the IMU temperature field.
- Purpose of the block scrambling (DMA artifact vs. light obfuscation) — unknown.
- Whether any UVC stack in the wild actually delivers the byte-swapped order
  (macOS delivers `2vuy` as-is; the detection exists because it costs nothing
  and fails loudly if ignored).
