#!/usr/bin/env python3
"""Cross-platform capture + realtime processing for the XREAL Air 2 Ultra
stereo tracking cameras.

The glasses enumerate as a normal UVC webcam whose nominal format is 640x241
"YUY2/UYVY" (16 bpp). That label is a lie: every byte of the buffer is one
8-bit grayscale pixel of a tight 640x482 frame (rows 0-479 image, row 480
telemetry, row 481 constant 0x5C padding). See docs/PROTOCOL.md.

This module grabs those raw bytes through OpenCV with CONVERT_RGB disabled,
which works on:
    Windows  - Media Foundation (CAP_MSMF) or DirectShow (CAP_DSHOW)
    Linux    - V4L2 (CAP_V4L2)
    macOS    - AVFoundation (CAP_AVFOUNDATION; the native Swift tools in src/
               are the better option there)

Because the fourcc is fake, some backends may hand us the byte stream with
the two bytes of each 16-bit "YUV" pair swapped (a YUY2<->UYVY relabel). The
telemetry row makes the true order detectable: markers 0xAD,0xDA sit at
columns 22,23. `find_camera()` fingerprints candidate cameras with this and
fixes the byte order transparently, so callers always see the same layout the
macOS tools see.

Run directly to scan for cameras:  python xreal_uvc.py
"""

import platform
import sys
import time

import cv2
import numpy as np

from xreal_descramble import REORDER, NB, BS

# frame geometry (see docs/PROTOCOL.md)
W, H_IMG, H_FULL = 640, 480, 482
META_ROW, PAD_ROW, PAD_VAL = 480, 481, 0x5C
CTR_COL, CAM_COL = 19, 58
FRAME_BYTES = W * H_FULL                       # 308480
OW, OH = 480, 640                              # descrambled portrait size

_REORDER = np.asarray(REORDER, np.int32)
_INV_REORDER = np.empty(NB, np.int32)
_INV_REORDER[_REORDER] = np.arange(NB, dtype=np.int32)


# ---- descramble (vectorized equivalent of xreal_descramble.py) ---------------

def build_lut(is_right):
    """lut[i] = flat output index of the i-th byte of the descrambled stream.

    Byte i of the stream is pixel i%640 of sensor line i//640; the sensors sit
    rotated 90 deg (and the two cameras 180 deg opposed), so sensor lines are
    columns of the upright 480x640 portrait image.
    """
    idx = np.arange(NB * BS, dtype=np.int32)
    r, c = idx // W, idx % W
    if is_right:
        y, x = c, r
    else:
        y, x = (W - 1) - c, (H_IMG - 1) - r
    return y * OW + x


class Descrambler:
    """Fast per-camera descrambler (identical output to xreal_descramble.py)."""

    def __init__(self, is_right):
        lut = build_lut(is_right)
        self.inv = np.empty_like(lut)
        self.inv[lut] = np.arange(lut.size, dtype=np.int32)   # gather form

    def __call__(self, raw_flat):
        """raw_flat: 307200 scrambled bytes -> (640, 480) uint8 image."""
        blocks = raw_flat.reshape(NB, BS)
        # sync: logical block 0 starts in the fisheye's black border, so the
        # raw block whose first 128 bytes sum lowest is REORDER[align]
        sync = int(np.argmin(blocks[:, :128].sum(axis=1, dtype=np.int32)))
        align = _INV_REORDER[sync]
        order = _REORDER[(align + np.arange(NB)) % NB]
        stream = blocks[order].reshape(-1)
        return stream[self.inv].reshape(OH, OW)


# ---- realtime cleanup (port of the Cleaner in src/preview_clean.swift) -------

def equalize(img):
    return cv2.equalizeHist(img)


class Cleaner:
    """Column FPN (vertical stripes) + row banding removal, online version.

    Column FPN is static per camera: median of a horizontal high-pass,
    accumulated across frames with an EMA. Row banding varies over time and is
    removed per frame.
    """

    R = 15   # high-pass box radius, matches the Swift viewer

    def __init__(self):
        self.stripe = None

    def __call__(self, img):
        f = img.astype(np.float32)
        k = 2 * self.R + 1

        # medians run on every 2nd sample: same estimate, ~half the cost,
        # keeps the pipeline comfortably inside the 60 fps UVC frame budget
        hp = f - cv2.blur(f, (k, 1), borderType=cv2.BORDER_REPLICATE)
        cur = np.median(hp[::2], axis=0)
        if self.stripe is None:
            self.stripe = cur
        else:
            self.stripe = 0.95 * self.stripe + 0.05 * cur
        f -= self.stripe[None, :]

        hp = f - cv2.blur(f, (1, k), borderType=cv2.BORDER_REPLICATE)
        f -= np.median(hp[:, ::2], axis=1)[:, None]

        return equalize(np.clip(f, 0, 255).astype(np.uint8))


# ---- capture ------------------------------------------------------------------

_BACKENDS = {
    "msmf": cv2.CAP_MSMF,
    "dshow": cv2.CAP_DSHOW,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
    "any": cv2.CAP_ANY,
}
_BACKEND_NAMES = {v: k for k, v in _BACKENDS.items()}


def backend_by_name(name):
    try:
        return _BACKENDS[name.lower()]
    except KeyError:
        raise SystemExit(f"unknown backend '{name}' (choose from {', '.join(_BACKENDS)})")


def default_backends():
    return {
        "Windows": [cv2.CAP_MSMF, cv2.CAP_DSHOW],
        "Linux": [cv2.CAP_V4L2],
        "Darwin": [cv2.CAP_AVFOUNDATION],
    }.get(platform.system(), [cv2.CAP_ANY])


def _classify(flat):
    """'ok' / 'swapped' / None for one raw frame's bytes.

    Row 481 must be (mostly) the 0x5C padding constant; the 0xAD,0xDA telemetry
    markers at row 480 cols 22,23 then reveal whether the backend swapped the
    bytes of each 16-bit pair (fake-YUY2 vs fake-UYVY delivery).
    """
    rows = flat.reshape(H_FULL, W)
    if np.count_nonzero(rows[PAD_ROW] == PAD_VAL) < W - 64:
        return None
    t = rows[META_ROW]
    if t[22] == 0xAD and t[23] == 0xDA:
        return "ok"
    if t[22] == 0xDA and t[23] == 0xAD:
        return "swapped"
    return None


def _unswap(flat):
    return flat.reshape(-1, 2)[:, ::-1].reshape(-1)


def _raw_bytes(frame):
    """Whatever array shape a backend returns, as flat bytes (or None)."""
    if frame is None:
        return None
    flat = np.asarray(frame).reshape(-1)
    if flat.dtype != np.uint8:
        flat = flat.view(np.uint8)
    return flat if flat.size == FRAME_BYTES else None


class XrealFrame:
    """One UVC frame: full 482-row raster + parsed telemetry."""

    __slots__ = ("gray", "counter", "cam")

    def __init__(self, gray):
        self.gray = gray                                  # (482, 640) uint8
        self.counter = int(gray[META_ROW, CTR_COL])       # shared by a stereo pair
        self.cam = int(gray[META_ROW, CAM_COL]) & 1       # 0x20=cam0, 0x21=cam1

    @property
    def image(self):
        return self.gray[:H_IMG]                          # (480, 640), scrambled

    @property
    def mean(self):
        return float(self.gray[:H_IMG].mean())


class XrealCapture:
    """An opened, byte-order-normalized XREAL camera stream."""

    def __init__(self, cap, backend, index, swapped):
        self.cap, self.backend, self.index, self.swapped = cap, backend, index, swapped

    @property
    def description(self):
        return f"{_BACKEND_NAMES.get(self.backend, self.backend)}:{self.index}" + (
            " (byte-swapped delivery)" if self.swapped else "")

    def read(self, tries=8):
        """Next frame, or None if the stream keeps misbehaving."""
        for _ in range(tries):
            ok, frame = self.cap.read()
            flat = _raw_bytes(frame) if ok else None
            if flat is None:
                continue
            if self.swapped:
                flat = _unswap(flat)
            return XrealFrame(flat.reshape(H_FULL, W))
        return None

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


def _open_raw(index, backend):
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None
    # the whole trick: hand over the bytes, do not decode the fake YUV
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # best effort, lowers latency
    return cap


def _probe(cap, tries=40):
    """Read a few frames and fingerprint them ('ok'/'swapped'/None)."""
    for _ in range(tries):
        ok, frame = cap.read()
        flat = _raw_bytes(frame) if ok else None
        if flat is None:
            continue
        mode = _classify(flat)
        if mode:
            return mode
    return None


def find_camera(index=None, backend=None, verbose=False, max_index=10):
    """Scan for the XREAL stream and return an XrealCapture, or None.

    index/backend pin the search; otherwise every plausible backend/index is
    probed and validated with the telemetry fingerprint, so other webcams on
    the machine are skipped automatically.
    """
    try:  # silence per-index open errors from OpenCV during the scan
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    backends = [backend] if backend is not None else default_backends()
    indices = [index] if index is not None else range(max_index)
    for be in backends:
        misses = 0
        for i in indices:
            cap = _open_raw(i, be)
            if cap is None:
                misses += 1
                if misses >= 3 and index is None:
                    break
                continue
            misses = 0
            mode = _probe(cap)
            if verbose:
                print(f"  {_BACKEND_NAMES.get(be, be)}:{i} -> "
                      f"{mode or 'not an XREAL stream'}")
            if mode:
                return XrealCapture(cap, be, i, mode == "swapped")
            cap.release()
    return None


# ---- scan mode (the enumerate.swift equivalent) --------------------------------

def _scan():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    print("Scanning for the XREAL Air 2 Ultra UVC stream...")
    for be in default_backends():
        print(f"backend {_BACKEND_NAMES.get(be, be)}:")
        found_any = False
        misses = 0
        for i in range(10):
            cap = _open_raw(i, be)
            if cap is None:
                misses += 1
                if misses >= 3:
                    break
                continue
            misses = 0
            found_any = True
            t0 = time.time()
            ok, frame = cap.read()
            size = np.asarray(frame).size if ok and frame is not None else 0
            mode = _classify(_raw_bytes(frame)) if size == FRAME_BYTES else None
            verdict = {"ok": "XREAL stream", "swapped": "XREAL stream (byte-swapped)",
                       None: "some other camera" if size else "no frame"}[mode]
            print(f"  index {i}: raw frame {size} bytes "
                  f"({(time.time()-t0)*1000:.0f} ms) -> {verdict}")
            cap.release()
        if not found_any:
            print("  no cameras")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(__doc__)
        sys.exit(0)
    _scan()
