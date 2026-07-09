#!/usr/bin/env python3
"""Live oscilloscope + attitude view for the XREAL Air 2 Ultra IMU (1 kHz).

Two rolling 4-second panels (angular rate and acceleration) plus a 3D
attitude cube driven by a host-side 6-axis Madgwick filter - the glasses
stream raw data only, so the quaternion is computed here (gyro bias is
captured during the first second; hold the glasses still at launch for best
results; yaw is unreferenced without a magnetometer). Runs happily alongside
the camera viewer - the IMU is a separate HID interface.

Usage:
  python xreal_imu_scope.py            # live window
  python xreal_imu_scope.py --snap out.png   # capture ~2 s, save one frame

Keys:  space = pause   b = re-capture gyro bias (hold still)   q/esc = quit
"""

import argparse
import collections
import math
import threading
import time

import cv2
import numpy as np

from xreal_ahrs import ImuOrientation, quat_rotate, quat_to_euler_deg
from xreal_imu import XrealImu

W, PANEL_H, GAP = 1000, 290, 34
SIDE_W = 330                                                   # attitude panel
SPAN_S = 4.0
CHAN_COLORS = [(80, 80, 255), (80, 220, 80), (255, 200, 60)]   # x, y, z (BGR)
MAG_COLOR = (240, 240, 240)


class Recorder:
    def __init__(self, imu):
        self.buf = collections.deque(maxlen=int(SPAN_S * 1100))
        self.latest = None
        self.rate = 0.0
        self.ahrs = ImuOrientation()
        self.quat = None
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
            self.quat = self.ahrs.feed(s) or self.quat
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


# a stylized "glasses" wireframe in the sensor frame + its edge list
_GL_V = [(-0.8, -0.18, -0.28), (0.8, -0.18, -0.28), (0.8, 0.18, -0.28),
         (-0.8, 0.18, -0.28), (-0.8, -0.18, 0.28), (0.8, -0.18, 0.28),
         (0.8, 0.18, 0.28), (-0.8, 0.18, 0.28),
         (-0.78, 0.18, 0.2), (-0.78, 1.1, 0.2),      # left temple arm
         (0.78, 0.18, 0.2), (0.78, 1.1, 0.2)]        # right temple arm
_GL_E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7), (8, 9), (10, 11)]
_AXES = [((1.15, 0, 0), (80, 80, 255), "x"),
         ((0, 1.15, 0), (80, 220, 80), "y"),
         ((0, 0, 1.15), (255, 200, 60), "z")]


def _project(v, cx, cy, scale):
    """Earth frame -> screen: +x right, +z up, camera pitched 25 deg down."""
    a = math.radians(25)
    x, y, z = v
    depth = y * math.cos(a) - z * math.sin(a)
    up = y * math.sin(a) + z * math.cos(a)
    return (int(cx + scale * x), int(cy - scale * up)), depth


def draw_attitude(canvas, rec, x0):
    cx, cy, scale = x0 + SIDE_W // 2, GAP + 160, 95
    cv2.rectangle(canvas, (x0, GAP), (x0 + SIDE_W - 6, GAP + 320), (28, 28, 28), -1)
    cv2.putText(canvas, "attitude (host-side Madgwick)", (x0 + 8, GAP - 5),
                cv2.FONT_HERSHEY_PLAIN, 0.9, (140, 140, 140), 1, cv2.LINE_AA)
    q = rec.quat
    if q is None:
        msg = "capturing gyro bias" if rec.ahrs.capturing else "waiting for data"
        cv2.putText(canvas, msg + " - hold still", (x0 + 20, cy),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 200, 255), 1, cv2.LINE_AA)
        return
    pts = {}
    for i, v in enumerate(_GL_V):
        pts[i], _ = _project(quat_rotate(q, v), cx, cy, scale)
    for a, b in _GL_E:
        cv2.line(canvas, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
    for v, color, lbl in _AXES:
        p, _ = _project(quat_rotate(q, v), cx, cy, scale)
        cv2.line(canvas, (cx, cy), p, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (p[0] + 3, p[1] - 3), cv2.FONT_HERSHEY_PLAIN,
                    1.0, color, 1, cv2.LINE_AA)
    yaw, pitch, roll = quat_to_euler_deg(q)
    rows = [f"q  {q[0]:+6.3f} {q[1]:+6.3f}", f"   {q[2]:+6.3f} {q[3]:+6.3f}",
            f"yaw   {yaw:+7.1f} deg (drifts, no mag)",
            f"pitch {pitch:+7.1f} deg", f"roll  {roll:+7.1f} deg"]
    if not rec.ahrs.still_capture:
        rows.append("bias capture was noisy - press b to redo")
    for i, row in enumerate(rows):
        cv2.putText(canvas, row, (x0 + 12, GAP + 340 + 20 * i),
                    cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 102), 1, cv2.LINE_AA)


def render(rec):
    canvas = np.zeros((2 * PANEL_H + 2 * GAP, W + SIDE_W, 3), np.uint8)
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
        draw_attitude(canvas, rec, W + 6)
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
        cv2.resizeWindow(win, W + SIDE_W, 2 * PANEL_H + 2 * GAP)
        paused = False
        frozen = None
        while True:
            if not paused:
                frozen = render(rec)
            cv2.imshow(win, frozen)
            k = cv2.waitKey(33) & 0xFF
            if k == ord(" "):
                paused = not paused
            elif k == ord("b"):        # re-zero: fresh bias capture + attitude
                rec.ahrs = ImuOrientation()
                rec.quat = None
            elif k in (ord("q"), 27):
                break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        rec.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
