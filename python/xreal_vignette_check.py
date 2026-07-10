#!/usr/bin/env python3
"""Does the descrambled raster match the FACTORY calibration's raster
convention? Objective test, no judgment calls.

The factory principal points sit ~7 px off the raster center in
mirror-paired directions (device_1/left cc_x = 246.5, device_2/right
cc_x = 232.6, center = 239.5). The fisheye vignette circle in a real frame
is centered on the optical axis, so measuring the circle's center in OUR
raster and comparing against cc_x decides whether the current raster is in
the factory convention or its x-mirror (the two hypotheses differ by 14 px,
the measurement is good to ~2 px).

Usage:
  python xreal_vignette_check.py SNAPSHOT.png [SNAPSHOT2.png ...]

Accepts: a single 480x640 portrait camera image, a 960x640 side-by-side
pair (main-branch app snapshot: left pane = device_1/left cam, right pane =
device_2/right cam), or the research app snapshot (left pane = device_1;
the right pane is a depth map and is skipped). Equalization and the green
tracker dots do not move the vignette boundary.
"""

import sys

import cv2
import numpy as np

CC = {"device_1 (left cam)": 246.55, "device_2 (right cam)": 232.56}
CENTER = 239.5


def circle_center_x(gray):
    """x of the bright fisheye disc's center via row-wise extent midpoints."""
    # threshold: vignette corners are near-black; the disc is much brighter
    thr = max(12, int(np.percentile(gray, 20)))
    mask = gray > thr
    mids, weights = [], []
    for y in range(gray.shape[0]):
        xs = np.flatnonzero(mask[y])
        if len(xs) < 40:
            continue
        # use rows where the disc boundary is inside the frame on both
        # sides (otherwise the midpoint is clamped by the image edge)
        if xs[0] == 0 and xs[-1] == gray.shape[1] - 1:
            continue
        mids.append(0.5 * (xs[0] + xs[-1]))
        weights.append(xs[-1] - xs[0])
    if len(mids) < 30:
        return None
    return float(np.average(mids, weights=weights))


def judge(name, cx, cc_x):
    mirrored_cc = 2 * CENTER - cc_x
    d_factory = abs(cx - cc_x)
    d_mirror = abs(cx - mirrored_cc)
    verdict = ("MATCHES factory convention" if d_factory < d_mirror
               else "MIRRORED relative to factory convention")
    conf = abs(d_factory - d_mirror)
    print(f"  {name}: vignette center x = {cx:.1f}")
    print(f"    factory cc_x = {cc_x:.1f} (d={d_factory:.1f})   "
          f"mirrored cc_x = {mirrored_cc:.1f} (d={d_mirror:.1f})")
    print(f"    -> raster {verdict}  (margin {conf:.1f} px)")


def main(paths):
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{p}: cannot read")
            continue
        h, w = img.shape
        print(f"{p}: {w}x{h}")
        if (w, h) == (480, 640):
            cx = circle_center_x(img)
            if cx is None:
                print("  vignette not measurable (image too bright/cropped?)")
                continue
            judge("device_1 (left cam)?", cx, CC["device_1 (left cam)"])
        elif (w, h) == (960, 640):
            left, right = img[:, :480], img[:, 480:]
            cxl = circle_center_x(left)
            if cxl is not None:
                judge("left pane / device_1", cxl, CC["device_1 (left cam)"])
            else:
                print("  left pane: vignette not measurable")
            cxr = circle_center_x(right)
            # research-app snapshots have a depth map here: detectable by
            # the measurement failing or landing nowhere near either cc
            if cxr is not None:
                judge("right pane / device_2", cxr, CC["device_2 (right cam)"])
            else:
                print("  right pane: not measurable (depth pane?) — skipped")
        else:
            print("  unsupported size (want 480x640 or 960x640)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1:])
