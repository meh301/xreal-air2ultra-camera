# XREAL Air 2 Ultra — stereo camera viewer for macOS

**Read the stereo tracking cameras of the XREAL Air 2 Ultra on macOS — no SDK, no kext, no drivers.**

[日本語のREADMEはこちら](README.ja.md)

The Air 2 Ultra exposes its two tracking cameras as a standard USB Video Class (UVC)
device, but the video stream is **block-scrambled** and looks like noise in a normal
webcam viewer. This project descrambles it in real time and gives you a clean
640×480 stereo grayscale feed — the raw input you'd need for VIO/SLAM experiments,
robotics, or just poking at the hardware.

![clean stereo preview](docs/images/stereo_preview.png)

## Quick start

Requirements: a Mac with Xcode Command Line Tools (`xcode-select --install`),
and an Air 2 Ultra plugged in via USB-C.

```sh
make
./preview_clean
```

That's it. macOS will ask for camera permission for your terminal on first run.
A window opens with the live left/right camera feed, descrambled and denoised.

Keys: `c` toggle clean/scrambled view · `s` save snapshot PNG · `space` pause · `q` quit

## Tools

| Tool | What it does |
|------|--------------|
| `preview_clean` | Real-time stereo viewer (descramble + fixed-pattern-noise removal), 60 fps. Also `--snap out.png` for a headless snapshot, `--test in.pgm prefix` to verify descrambling offline. |
| `glasses_passthrough` | Stereo passthrough onto the glasses themselves: left camera → left eye, right camera → right eye. Fullscreen side-by-side on the XREAL display (put the glasses in 3D/SBS mode). `--list` shows displays, `--display N` picks one, `--window` previews the SBS composite in a normal window. Keys: `x` swap eyes · `r` rotate · `m` mirror · `s` toggle SBS. |
| `xreal_cam` | Recorder: `./xreal_cam <numFrames> <outDir>` writes raw `cam0_*.pgm` / `cam1_*.pgm` (still scrambled) plus `meta.csv` with the per-pair frame counter. |
| `enumerate` | List AVFoundation cameras and the XREAL device's formats. |
| `python/process_clean.py` | Offline pipeline: `python3 python/process_clean.py <capDir> <outDir>` descrambles + cleans a recording into PNGs and a side-by-side `stereo_feed.mp4`. Needs `numpy opencv-python pillow`. |
| `python/xreal_descramble.py` | Minimal single-frame descrambler, useful as a reference implementation. |
| `research/` | Reverse-engineering tools (vendor HID commands, UVC controls, USB descriptors). See [research/README.md](research/README.md). |

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

Full protocol notes (USB layout, telemetry row map, scramble algorithm, UVC exposure
controls, vendor HID protocol): **[docs/PROTOCOL.md](docs/PROTOCOL.md)**.

## Credits

- The block-reorder table was discovered by
  [mazeasdamien/myXreal](https://github.com/mazeasdamien/myXreal) (`stereo_camera.cpp`).
- The vendor HID packet format is documented in
  [badicsalex/ar-drivers-rs](https://github.com/badicsalex/ar-drivers-rs).
- Developed with [Claude Code](https://claude.com/claude-code) (Claude Fable 5) —
  the reverse-engineering analysis, tools, and documentation in this repo were built
  in collaboration with the AI agent.

## Disclaimer

This is an unofficial, reverse-engineered project, not affiliated with or endorsed by
XREAL. It only *reads* the camera stream over standard UVC — it sends no commands that
modify the device — but use it at your own risk.

## License

[MIT](LICENSE)
