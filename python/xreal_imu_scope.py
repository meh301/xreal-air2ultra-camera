#!/usr/bin/env python3
"""Live oscilloscope for the XREAL Air 2 Ultra IMU (1 kHz).

Two rolling 4-second panels: angular rate (deg/s, autoscaled) and
acceleration (g, autoscaled, with |a| in white). Runs happily alongside the
camera viewer - the IMU is a separate HID interface.

Usage:
  python xreal_imu_scope.py            # live window
  python xreal_imu_scope.py --snap out.png   # capture ~2 s, save one frame

Keys:  space = pause   q/esc = quit
"""

import argparse
import collections
import threading
import time

import cv2
import numpy as np

from xreal_imu import XrealImu

W, PANEL_H, GAP = 1000, 290, 34
SPAN_S = 4.0
CHAN_COLORS = [(80, 80, 255), (80, 220, 80), (255, 200, 60)]   # x, y, z (BGR)
MAG_COLOR = (240, 240, 240)


class Recorder:
    def __init__(self, imu):
        self.buf = collections.deque(maxlen=int(SPAN_S * 1100))
        self.latest = None
        self.rate = 0.0
        self._n, self._t0 = 0, time.time()
        self._stop = False
        self._thread = threading.Thread(target=self._run, args=(imu,), daemon=True)
        self._thread.start()

    def _run(self, imu):
        for s in imu.samples():
            if self._stop:
                return
            if s is None:
                # stalled - e.g. another process sent stream-off; re-enable
                try:
                    imu.stream(True)
                except RuntimeError:
                    pass
                continue
            self.buf.append((s.ts_ns, s.gyro_dps, s.accel_g))
            self.latest = s
            self._n += 1
            dt = time.time() - self._t0
            if dt >= 1:
                self.rate, self._n, self._t0 = self._n / dt, 0, time.time()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=2)


def draw_panel(canvas, y0, series, labels, unit, extra=None):
    """series: (N, C) float array; draws autoscaled traces into the panel."""
    cv2.rectangle(canvas, (0, y0), (W - 1, y0 + PANEL_H), (28, 28, 28), -1)
    if len(series) < 2:
        return
    lim = max(1e-3, 1.15 * float(np.abs(series).max()),
              1.15 * float(np.abs(extra).max()) if extra is not None else 0)
    mid = y0 + PANEL_H // 2
    scale = (PANEL_H // 2 - 6) / lim
    cv2.line(canvas, (0, mid), (W, mid), (70, 70, 70), 1)
    for frac in (0.5, 1.0):
        for sgn in (-1, 1):
            yy = int(mid - sgn * frac * lim * scale)
            cv2.line(canvas, (0, yy), (W, yy), (45, 45, 45), 1)
    xs = np.linspace(0, W - 1, len(series)).astype(np.int32)

    def trace(vals, color, thick=1):
        ys = (mid - vals * scale).astype(np.int32)
        pts = np.stack([xs, np.clip(ys, y0, y0 + PANEL_H - 1)], 1)
        cv2.polylines(canvas, [pts], False, color, thick, cv2.LINE_AA)

    if extra is not None:
        trace(np.asarray(extra), MAG_COLOR, 1)
    for c in range(series.shape[1]):
        trace(series[:, c], CHAN_COLORS[c])
    cv2.putText(canvas, f"+{lim:.3g} {unit}", (W - 110, y0 + 16),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (140, 140, 140), 1, cv2.LINE_AA)
    for i, lbl in enumerate(labels):
        cv2.putText(canvas, lbl, (8 + 52 * i, y0 + 16), cv2.FONT_HERSHEY_PLAIN,
                    1.0, CHAN_COLORS[i] if i < 3 else MAG_COLOR, 1, cv2.LINE_AA)


def render(rec):
    canvas = np.zeros((2 * PANEL_H + 2 * GAP, W, 3), np.uint8)
    snap = list(rec.buf)
    if snap:
        gyro = np.array([g for _, g, _ in snap], np.float32)
        acc = np.array([a for _, _, a in snap], np.float32)
        mag = np.linalg.norm(acc, axis=1)
        s = rec.latest
        head = (f"XREAL Air 2 Ultra IMU  {rec.rate:4.0f} Hz   "
                f"gyro ({s.gyro_dps[0]:+7.2f},{s.gyro_dps[1]:+7.2f},{s.gyro_dps[2]:+7.2f}) deg/s   "
                f"accel ({s.accel_g[0]:+5.2f},{s.accel_g[1]:+5.2f},{s.accel_g[2]:+5.2f}) g   "
                f"(space:pause q:quit)")
        cv2.putText(canvas, head, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0,
                    (0, 255, 102), 1, cv2.LINE_AA)
        draw_panel(canvas, GAP, gyro, ("gx", "gy", "gz"), "deg/s")
        draw_panel(canvas, GAP + PANEL_H + GAP, acc, ("ax", "ay", "az", "|a|"),
                   "g", extra=mag)
        cv2.putText(canvas, f"angular rate  (last {SPAN_S:.0f} s)", (8, GAP - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(canvas, "acceleration", (8, GAP + PANEL_H + GAP - 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (140, 140, 140), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snap", metavar="OUT.png", help="capture ~2 s and save one frame")
    args = ap.parse_args()

    with XrealImu() as imu:
        imu.stream(True)
        rec = Recorder(imu)
        if args.snap:
            time.sleep(2.2)
            cv2.imwrite(args.snap, render(rec))
            print(f"saved {args.snap}")
            rec.stop()
            return
        win = "XREAL IMU scope"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, W, 2 * PANEL_H + 2 * GAP)
        paused = False
        frozen = None
        while True:
            if not paused:
                frozen = render(rec)
            cv2.imshow(win, frozen)
            k = cv2.waitKey(33) & 0xFF
            if k == ord(" "):
                paused = not paused
            elif k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        rec.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
