#!/usr/bin/env python3
"""Real-time clean stereo preview of the XREAL Air 2 Ultra tracking cameras
for Windows / Linux / macOS. Cross-platform port of src/preview_clean.swift.

Per frame:
  1. block descramble (128 blocks x 2400 bytes, same algorithm as
     xreal_descramble.py)
  2. column FPN (vertical stripe) removal: EMA of the horizontal high-pass
     column median
  3. row banding removal + histogram equalization
then the two cameras are shown side by side (480x640 x2 = 960x640).

Usage:
  python preview_clean.py                     # GUI preview
  python preview_clean.py --snap out.png      # headless: save one stereo snapshot
  python preview_clean.py --test in.pgm pfx   # descramble a single PGM offline
  python preview_clean.py --camera N --backend msmf|dshow|v4l2|avfoundation|any

Keys:  c = clean/scrambled view   s = save snapshot   space = pause   q/esc = quit
"""

import argparse
import sys
import time

import cv2
import numpy as np

from xreal_uvc import (Cleaner, Descrambler, H_IMG, OH, OW, W,
                       backend_by_name, equalize, find_camera)

WIN = "XREAL Air 2 Ultra - clean stereo preview"


def hud(img, text):
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, 255, 1, cv2.LINE_AA)


def offline_test(pgm_path, out_prefix):
    raw = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.size < W * H_IMG:
        sys.exit(f"cannot read {pgm_path}")
    flat = raw.reshape(-1)[: W * H_IMG]
    for name, is_right in (("right", True), ("left", False)):
        img = Descrambler(is_right)(flat)
        cv2.imwrite(f"{out_prefix}_{name}.png", img)
    print(f"saved {out_prefix}_right.png / _left.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, help="capture index (default: scan)")
    ap.add_argument("--backend", help="msmf / dshow / v4l2 / avfoundation / any")
    ap.add_argument("--snap", metavar="OUT.png", help="headless single snapshot")
    ap.add_argument("--test", nargs=2, metavar=("IN.pgm", "OUTPREFIX"),
                    help="descramble one raw PGM offline and exit")
    args = ap.parse_args()

    if args.test:
        offline_test(*args.test)
        return

    cap = find_camera(index=args.camera,
                      backend=backend_by_name(args.backend) if args.backend else None,
                      verbose=True)
    if cap is None:
        sys.exit("XREAL UVC camera not found. Is the Air 2 Ultra plugged in? "
                 "Try 'python xreal_uvc.py' to scan; see README for OS camera "
                 "permission notes.")
    print(f"Opening XREAL stream at {cap.description}")

    descramblers = [Descrambler(is_right=True), Descrambler(is_right=False)]  # cam0, cam1
    cleaners = [Cleaner(), Cleaner()]
    latest_clean = [None, None]
    latest_raw = [None, None]
    show_clean, paused = True, False
    snap_request, snap_count = False, 0
    fps, fps_count, fps_start = 0.0, 0, time.time()
    deadline = time.time() + 20 if args.snap else None

    if not args.snap:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 2 * OW, OH)

    with cap:
        while True:
            if deadline and time.time() > deadline:
                sys.exit("snap timed out")

            f = cap.read()
            if f is None:
                print("stream stalled, retrying...")
                continue

            if not paused:
                if f.mean < 5:      # black frames right after startup
                    continue
                flat = f.image.reshape(-1)
                latest_clean[f.cam] = cleaners[f.cam](descramblers[f.cam](flat))
                latest_raw[f.cam] = equalize(f.image)

                fps_count += 1
                dt = time.time() - fps_start
                if dt >= 1:
                    fps, fps_count, fps_start = fps_count / dt, 0, time.time()

            if all(v is not None for v in latest_clean + latest_raw):
                pair = latest_clean if show_clean else latest_raw
                combined = np.hstack(pair)

                if args.snap:
                    snap_count += 1
                    if snap_count >= 15:      # let the FPN EMA settle first
                        cv2.imwrite(args.snap, combined)
                        print(f"snapshot saved: {args.snap}")
                        return
                    continue
                if snap_request:
                    snap_request = False
                    path = f"preview_snap_{int(time.time())}.png"
                    cv2.imwrite(path, combined)
                    print(f"snapshot saved: {path}")

                view = combined.copy()
                hud(view, f"XREAL Air 2 Ultra  L | R  ctr={f.counter}  {fps:.0f} fps  "
                          f"[{'CLEAN' if show_clean else 'SCRAMBLED'}]"
                          f"{'  PAUSED' if paused else ''}  "
                          "(c:view s:snap space:pause q:quit)")
                cv2.imshow(WIN, view)

            if not args.snap:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("c"):
                    show_clean = not show_clean
                elif key == ord("s"):
                    snap_request = True
                elif key == ord(" "):
                    paused = not paused
                elif key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
