#!/usr/bin/env python3
"""Record raw (still scrambled) frames from the XREAL Air 2 Ultra on
Windows / Linux / macOS. Cross-platform port of src/xreal_cam.swift.

Usage:
  python xreal_cam.py [numFrames] [outDir] [--camera N] [--backend NAME]

Writes <outDir>/cam0_XXXX.pgm and cam1_XXXX.pgm (raw 8-bit, telemetry rows
stripped, identical to the macOS recorder) plus meta.csv with
(uvcIndex, counter, imageMean). Feed the directory to process_clean.py to get
descrambled, denoised PNGs and a stereo mp4.
"""

import argparse
import os
import sys
import time

from xreal_uvc import H_IMG, W, backend_by_name, find_camera


def write_pgm(path, img):
    with open(path, "wb") as fh:
        fh.write(f"P5\n{W} {H_IMG}\n255\n".encode())
        fh.write(img.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("numFrames", nargs="?", type=int, default=60)
    ap.add_argument("outDir", nargs="?", default="./xreal_out")
    ap.add_argument("--camera", type=int, help="capture index (default: scan)")
    ap.add_argument("--backend", help="msmf / dshow / v4l2 / avfoundation / any")
    args = ap.parse_args()

    cap = find_camera(index=args.camera,
                      backend=backend_by_name(args.backend) if args.backend else None,
                      latest_only=False)   # a recorder wants every frame
    if cap is None:
        sys.exit("XREAL UVC camera not found (plugged in? old firmware? "
                 "update at https://ota.xreal.com/ultra-update?version=1)")
    print(f"Opening XREAL stream at {cap.description}")

    os.makedirs(args.outDir, exist_ok=True)
    csv = ["uvcIndex,counter,mean\n"]
    count = 0
    deadline = time.time() + args.numFrames / 30.0 + 10

    with cap:
        while count < args.numFrames and time.time() < deadline:
            f = cap.read()
            if f is None:
                continue
            write_pgm(os.path.join(args.outDir, f"cam{f.cam}_{count:04d}.pgm"), f.image)
            csv.append(f"{count},{f.counter},{f.mean:.1f}\n")
            if count % 10 == 0:
                print(f"frame {count} cam{f.cam} counter={f.counter} mean={f.mean:.1f}")
            count += 1

    with open(os.path.join(args.outDir, "meta.csv"), "w") as fh:
        fh.writelines(csv)
    print(f"Done. Captured {count} frames to {args.outDir}/ (meta.csv logged)")


if __name__ == "__main__":
    main()
