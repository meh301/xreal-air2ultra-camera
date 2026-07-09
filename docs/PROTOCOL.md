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

| col | Meaning |
|-----|---------|
| 2, 3 | timestamp-like values |
| 5 | constant `0x19` |
| 6 | frame counter, high byte |
| 19 | **frame counter, low byte** — shared by both frames of a stereo pair |
| 22, 23 | constant markers `0xAD`, `0xDA` |
| 25, 26, 27, 35 | config-like constants |
| 58 | **camera ID: `0x20` = cam0, `0x21` = cam1** |

Consecutive UVC frames alternate between the two cameras and a pair shares one counter
value (~30 stereo pairs/s at 60 fps UVC). **Do not infer camera identity from arrival
order**: the L/R order within a pair is not fixed (in one 117-frame capture, 48 pairs
arrived one way and 10 the other; frame drops shift it further). Column 58 is the only
reliable discriminator — it matched image-correlation ground truth on all 117 frames
tested.

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
`(y, x) = (c, r)` for cam0 and `(639-c, 479-r)` for cam1.

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
data `01` enables the IMU stream (ack `0x04`). The IMU channel (interface 2) uses a
different header (`0xAA`, request_id `0x0437`). Tools: `research/hid_cmd.swift`,
`research/hid_read.swift`, `research/cam_probe.swift`.

## Capturing on other platforms

The stream is plain UVC, so any OS UVC stack can deliver it — the only trick is
getting the **untouched bytes** out, because the nominal `640×241 @ 60 fps`
"YUY2/UYVY" format is a lie (every byte is one mono pixel of the 640×482 raster).

- **OpenCV** (Windows MSMF/DirectShow, Linux V4L2, macOS AVFoundation): open the
  device and set `CAP_PROP_CONVERT_RGB = 0`; `read()` then returns the raw
  308,480-byte buffer instead of a decoded BGR image. Never request a different
  width/height — that would insert a converter. `python/xreal_uvc.py` implements
  this, plus device auto-detection.
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
- Driving the IMU stream and fusing it with the stereo feed (VIO/SLAM).
- Purpose of the block scrambling (DMA artifact vs. light obfuscation) — unknown.
- Whether any UVC stack in the wild actually delivers the byte-swapped order
  (macOS delivers `2vuy` as-is; the detection exists because it costs nothing
  and fails loudly if ignored).
