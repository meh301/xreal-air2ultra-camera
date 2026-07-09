#!/usr/bin/env python3
"""Stereo passthrough onto the glasses' own displays for Windows / Linux.
Cross-platform port of src/glasses_passthrough.swift.

The Air 2 Ultra shows up as an external monitor (DisplayPort alt-mode). Put
the glasses in 3D/SBS mode, run this tool, and the left camera is rendered
into the left half of that display (= left eye), the right camera into the
right half. Each clean stereo pair is also published to the shared-memory
framebuffer (same as preview_clean.py), so downstream consumers keep working.

Usage:
  python glasses_passthrough.py                # auto-pick the non-primary display
  python glasses_passthrough.py --list         # list displays and exit
  python glasses_passthrough.py --display N    # use display N from --list
  python glasses_passthrough.py --window       # windowed preview (no glasses needed)
  python glasses_passthrough.py --geometry X,Y,W,H   # manual placement fallback

Keys:  x = swap eyes   r = rotate 90°   m = mirror   s = SBS/mono toggle
       space = pause   q/esc = quit
"""

import argparse
import platform
import re
import subprocess
import sys
import time

import cv2
import numpy as np

from xreal_align import Aligner
from xreal_fb import FramebufferWriter
from xreal_uvc import Cleaner, Descrambler, OH, OW, backend_by_name, find_camera

WIN = "XREAL passthrough"


# ---- display enumeration ---------------------------------------------------------

def _windows_monitors():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()          # physical pixel coordinates
    monitors = []

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", ctypes.c_wchar * 32)]

    proc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                              ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def cb(hmon, hdc, lprect, lparam):
        mi = MONITORINFOEXW()
        mi.cbSize = ctypes.sizeof(MONITORINFOEXW)
        user32.GetMonitorInfoW(hmon, ctypes.byref(mi))
        r = mi.rcMonitor
        monitors.append({"name": mi.szDevice, "x": r.left, "y": r.top,
                         "w": r.right - r.left, "h": r.bottom - r.top,
                         "primary": bool(mi.dwFlags & 1)})
        return 1

    user32.EnumDisplayMonitors(0, 0, proc(cb), 0)
    return monitors


def _linux_monitors():
    try:
        out = subprocess.run(["xrandr", "--query"], capture_output=True,
                             text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    monitors = []
    for line in out.splitlines():
        if " connected" not in line:
            continue
        m = re.search(r"(\d+)x(\d+)\+(\d+)\+(\d+)", line)
        if not m:
            continue
        w, h, x, y = map(int, m.groups())
        monitors.append({"name": line.split()[0], "x": x, "y": y, "w": w, "h": h,
                         "primary": " primary " in line})
    return monitors


def list_monitors():
    return {"Windows": _windows_monitors,
            "Linux": _linux_monitors}.get(platform.system(), lambda: [])()


def pick_glasses_monitor(monitors):
    """The glasses are just 'some external display': prefer non-primary."""
    ext = [m for m in monitors if not m["primary"]]
    return ext[0] if ext else None


def set_display_mode(mode):
    """Switch the glasses' display mode over the MCU HID channel:
    1 = mirror (both eyes see everything), 3 = SBS 60 Hz (left half -> left
    eye only — required for stereo/aligned passthrough). Best effort."""
    try:
        from xreal_imu import mcu_query
        got = mcu_query(0x08, bytes([mode]))
        ok = got is not None and got[0] == 0
        print(f"glasses display mode -> {mode} "
              f"({'ok' if ok else 'no ack — switch 3D mode manually'})")
        return ok
    except Exception as e:
        print(f"display mode switch unavailable ({e}); "
              "enable the glasses' 3D/SBS mode manually")
        return False


# ---- composition -----------------------------------------------------------------

def draw_eye(canvas, img, x0, w, rot, mirror):
    """Aspect-fit `img` (after rotation/mirror) into canvas[:, x0:x0+w]."""
    if rot:
        img = np.rot90(img, rot)
    if mirror:
        img = np.fliplr(img)
    h = canvas.shape[0]
    scale = min(w / img.shape[1], h / img.shape[0])
    dw, dh = max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))
    resized = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_NEAREST)
    ox, oy = x0 + (w - dw) // 2, (h - dh) // 2
    canvas[oy:oy + dh, ox:ox + dw] = resized


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--camera", type=int, help="capture index (default: scan)")
    ap.add_argument("--backend", help="ffmpeg / msmf / dshow / v4l2 / any")
    ap.add_argument("--list", action="store_true", help="list displays and exit")
    ap.add_argument("--display", type=int, help="display index from --list")
    ap.add_argument("--window", action="store_true", help="windowed preview")
    ap.add_argument("--geometry", metavar="X,Y,W,H",
                    help="place fullscreen manually (skips display detection)")
    ap.add_argument("--align", metavar="CALIB.json",
                    help="1:1 world-aligned passthrough using the factory "
                         "calibration (dump it with xreal_imu.py --config); "
                         "disables the r/m keys, geometry is calibrated")
    ap.add_argument("--depth", type=float, default=float("inf"),
                    help="assumed scene distance in meters for --align "
                         "(default: infinity)")
    args = ap.parse_args()

    monitors = list_monitors()
    if args.list:
        if not monitors:
            print("no displays found (or unsupported session; use --geometry)")
        for i, m in enumerate(monitors):
            print(f"  [{i}] {m['name']}  {m['w']}x{m['h']} @ {m['x']},{m['y']}"
                  f"{'  (primary)' if m['primary'] else ''}")
        return

    def pick_target(mons):
        if args.geometry:
            x, y, w, h = map(int, args.geometry.split(","))
            return {"name": "manual", "x": x, "y": y, "w": w, "h": h}
        if args.display is not None:
            if not 0 <= args.display < len(mons):
                sys.exit(f"--display {args.display} out of range (see --list)")
            return mons[args.display]
        return pick_glasses_monitor(mons)

    target = None
    switched_3d = False
    if not args.window:
        target = pick_target(monitors)
        if target is None:
            print("No external display found - falling back to --window. "
                  "(Glasses plugged in and in extend mode?)", file=sys.stderr)
        elif not args.geometry:
            # per-eye stereo (SBS 60 Hz - the hardware's limit); the DP link
            # renegotiates (geometry may change), so switch first and
            # re-enumerate
            switched_3d = set_display_mode(3)
            if switched_3d:
                time.sleep(3.0)
                target = pick_target(list_monitors()) or target

    cap = find_camera(index=args.camera,
                      backend=backend_by_name(args.backend) if args.backend else None)
    if cap is None:
        sys.exit("XREAL UVC camera not found (plugged in? old firmware? "
                 "update at https://ota.xreal.com/ultra-update?version=1)")
    print(f"Opening XREAL stream at {cap.description}")

    if target and not args.window:
        print(f"Fullscreen on {target['name']} {target['w']}x{target['h']} "
              f"@ {target['x']},{target['y']}")
        cw, ch = target["w"], target["h"]
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.moveWindow(WIN, target["x"] + 10, target["y"] + 10)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cw, ch = 1280, 360
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, cw, ch)

    aligner = Aligner(args.align, half_size=(cw // 2, ch),
                      depth=args.depth) if args.align else None
    if aligner:
        print("1:1 world-aligned mode (42x25 deg per eye from the calibration)")

    descr = [Descrambler(is_right=True), Descrambler(is_right=False)]  # cam0, cam1
    cleaners = [Cleaner(), Cleaner()]
    clean = [np.empty((OH, OW), np.uint8) for _ in range(2)]
    have = [False, False]
    dscr = np.empty((OH, OW), np.uint8)
    pair = np.empty((OH, 2 * OW), np.uint8)     # published to the framebuffer
    canvas = np.zeros((ch, cw), np.uint8)
    fbw = FramebufferWriter()

    swap, rot, mirror, sbs, paused = False, 0, False, True, False
    fps, n, t0 = 0.0, 0, time.time()

    try:
        with cap:
            while True:
                f = cap.read()
                if f is None:
                    print("stream stalled, retrying...")
                    continue
                if not paused and f.mean >= 5:
                    c = f.cam
                    descr[c](f.image.reshape(-1), out=dscr)
                    cleaners[c](dscr, out=clean[c])
                    have[c] = True
                    n += 1
                    if time.time() - t0 >= 1:
                        fps, n, t0 = n / (time.time() - t0), 0, time.time()
                    if all(have):
                        # cam1 is the physical LEFT camera (verified on-device)
                        pair[:, :OW] = clean[1]
                        pair[:, OW:] = clean[0]
                        fbw.publish(pair, f.counter, f.device_ts)
                        canvas[:] = 0
                        if aligner:
                            lv, rv = aligner.warp(clean[1], clean[0])
                            if swap:
                                lv, rv = rv, lv
                            canvas[:, :cw // 2] = lv
                            canvas[:, cw // 2:] = rv if sbs else lv
                        else:
                            left, right = ((clean[0], clean[1]) if swap
                                           else (clean[1], clean[0]))
                            draw_eye(canvas, left, 0, cw // 2, rot, mirror)
                            draw_eye(canvas, right if sbs else left, cw // 2,
                                     cw - cw // 2, rot, mirror)
                        if args.window or target is None:
                            cv2.putText(canvas,
                                        f"L|R ctr={f.counter} {fps:.0f} fps  "
                                        "x:swap r:rot m:mirror s:sbs q:quit",
                                        (8, 18), cv2.FONT_HERSHEY_PLAIN, 1.0, 255, 1,
                                        cv2.LINE_AA)
                        cv2.imshow(WIN, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("x"):
                    swap = not swap
                elif key == ord("v") and aligner:
                    aligner.set_variant(aligner.variant + 1)
                    print(f"alignment variant -> {aligner.variant}")
                elif key == ord("r"):
                    rot = (rot + 1) % 4
                elif key == ord("m"):
                    mirror = not mirror
                elif key == ord("s"):
                    sbs = not sbs
                elif key == ord(" "):
                    paused = not paused
                elif key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        fbw.close()
        cv2.destroyAllWindows()
        if switched_3d:
            set_display_mode(1)   # leave the glasses as found


if __name__ == "__main__":
    main()
