#!/usr/bin/env python3
"""Cross-platform capture + realtime processing for the XREAL Air 2 Ultra
stereo tracking cameras.

The glasses enumerate as a normal UVC webcam whose nominal format is 640x241
"YUY2/UYVY" (16 bpp). That label is a lie: every byte of the buffer is one
8-bit grayscale pixel of a tight 640x482 frame (rows 0-479 image, row 480
telemetry, row 481 padding). See docs/PROTOCOL.md.

Capture paths (auto-selected by find_camera):
  * OpenCV with CONVERT_RGB=0 (Linux V4L2, macOS AVFoundation, and Windows
    machines where a backend cooperates).
  * ffmpeg's dshow input on Windows: both OpenCV Windows backends proved
    unable to deliver raw bytes from this device (MSMF starts no stream,
    DirectShow force-converts to BGR, which clips the fake-YUV bytes), while
    ffmpeg connects the DirectShow pin at its native yuyv422 type and pipes
    the untouched frames.

Firmware telemetry dialects (both auto-detected per frame):
  A: the layout in docs/PROTOCOL.md - counter col 19, camera id 0x20/0x21 at
     col 58, markers 0xAD,0xDA at cols 22,23, padding row always 0x5C.
  B: (newer firmware) counter col 18, camera bit at col 59, frame dimensions
     640/480/640 as LE u16 at cols 51-56 (used as the fingerprint), padding
     row 0x5C except on every 2nd cam-bit-1 frame, which carries extra data
     in row 481 and cols 22-27.

Because the fourcc is fake, a UVC stack may also deliver the two bytes of
each 16-bit pair swapped; the fingerprint detects that and read() fixes it.

Run directly to scan for cameras:  python xreal_uvc.py
"""

import platform
import re
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from xreal_descramble import REORDER, NB, BS

# frame geometry (see docs/PROTOCOL.md)
W, H_IMG, H_FULL = 640, 480, 482
META_ROW, PAD_ROW, PAD_VAL = 480, 481, 0x5C
FRAME_BYTES = W * H_FULL                       # 308480
OW, OH = 480, 640                              # descrambled portrait size

XREAL_VIDPID = "vid_3318&pid_0426"

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
    """Fast per-camera descrambler (identical output to xreal_descramble.py).

    All work buffers are preallocated; __call__ performs two vectorized
    gathers and no Python-level per-pixel work. Not thread-safe (use one
    instance per camera on one thread).
    """

    def __init__(self, is_right):
        lut = build_lut(is_right)
        self.inv = np.empty_like(lut)
        self.inv[lut] = np.arange(lut.size, dtype=np.int32)   # gather form
        self._stream = np.empty((NB, BS), np.uint8)
        self._out = np.empty(NB * BS, np.uint8)
        self._tail = np.arange(NB, dtype=np.int32)

    def __call__(self, raw_flat, out=None):
        """raw_flat: 307200 scrambled bytes -> (640, 480) uint8 image."""
        blocks = raw_flat.reshape(NB, BS)
        # sync: logical block 0 starts in the fisheye's black border, so the
        # raw block whose first 128 bytes sum lowest is REORDER[align]
        sync = int(np.argmin(blocks[:, :128].sum(axis=1, dtype=np.int32)))
        align = _INV_REORDER[sync]
        order = _REORDER[(align + self._tail) % NB]
        np.take(blocks, order, axis=0, out=self._stream)
        flat_out = self._out if out is None else out.reshape(-1)
        np.take(self._stream.reshape(-1), self.inv, out=flat_out)
        return flat_out.reshape(OH, OW)


# ---- realtime cleanup (port of the Cleaner in src/preview_clean.swift) -------

def equalize(img, out=None):
    return cv2.equalizeHist(img, dst=out)


class Cleaner:
    """Column FPN (vertical stripes) + row banding removal, online version.

    Column FPN is static per camera: median of a horizontal high-pass,
    accumulated across frames with an EMA (re-estimated every `est_every`
    frames - it converges after ~15 estimates and then barely moves). Row
    banding varies over time and is removed every frame.
    """

    R = 15   # high-pass box radius, matches the Swift viewer

    def __init__(self, est_every=3):
        self.stripe = None
        self.est_every = est_every
        self._n = 0
        self._f = np.empty((OH, OW), np.float32)
        self._blur = np.empty((OH, OW), np.float32)
        self._u8 = np.empty((OH, OW), np.uint8)

    def __call__(self, img, out=None):
        f, k = self._f, 2 * self.R + 1
        np.copyto(f, img)   # uint8 -> float32

        if self.stripe is None or self._n % self.est_every == 0:
            cv2.blur(f, (k, 1), dst=self._blur, borderType=cv2.BORDER_REPLICATE)
            # median over every 2nd row: same estimate, half the cost
            cur = np.median((f - self._blur)[::2], axis=0)
            if self.stripe is None:
                self.stripe = cur
            else:
                self.stripe = 0.95 * self.stripe + 0.05 * cur
        self._n += 1
        f -= self.stripe[None, :]

        cv2.blur(f, (1, k), dst=self._blur, borderType=cv2.BORDER_REPLICATE)
        f -= np.median((f - self._blur)[:, ::2], axis=1)[:, None]

        np.clip(f, 0, 255, out=f)
        u8 = self._u8
        np.copyto(u8, f, casting="unsafe")
        return equalize(u8, out=out)


# ---- telemetry dialects --------------------------------------------------------

class _DialectA:
    """Layout documented in docs/PROTOCOL.md (earlier firmware)."""
    name = "A"

    @staticmethod
    def parse(trow):
        # col 19 = pair counter, col 58 = 0x20 (cam0) / 0x21 (cam1);
        # cam0 uses the right-LUT
        return int(trow[19]), int(trow[58]) & 1


class _DialectB:
    """Newer firmware: dims marker at 51-56, counter 18, camera bit 59."""
    name = "B"

    @staticmethod
    def parse(trow):
        # cam bit 1 is the camera that decodes upright with the right-LUT,
        # i.e. the same physical camera dialect A calls cam0
        return int(trow[18]), 1 - (int(trow[59]) & 1)


_B_NATIVE = np.array([0x00, 0x80, 0x02, 0xE0, 0x01, 0x80, 0x02, 0x00], np.uint8)
_B_SWAPPED = np.array([0x80, 0x00, 0xE0, 0x02, 0x80, 0x01, 0x00, 0x02], np.uint8)


def _classify(flat):
    """(order, dialect) for one raw frame's bytes, or None.

    order is 'ok' or 'swapped' (fake-YUY2 vs fake-UYVY delivery: a byte swap
    within every 16-bit pair).
    """
    rows = flat.reshape(H_FULL, W)
    t = rows[META_ROW]
    # dialect B: 640,480,640 as little-endian u16 at cols 51-56
    if np.array_equal(t[50:58], _B_NATIVE):
        return "ok", _DialectB
    if np.array_equal(t[50:58], _B_SWAPPED):
        return "swapped", _DialectB
    # dialect A: 0xAD,0xDA markers + constant 0x5C padding row
    if np.count_nonzero(rows[PAD_ROW] == PAD_VAL) >= W - 64:
        if t[22] == 0xAD and t[23] == 0xDA:
            return "ok", _DialectA
        if t[22] == 0xDA and t[23] == 0xAD:
            return "swapped", _DialectA
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

    def __init__(self, gray, dialect):
        self.gray = gray                                  # (482, 640) uint8
        self.counter, self.cam = dialect.parse(gray[META_ROW])

    @property
    def image(self):
        return self.gray[:H_IMG]                          # (480, 640), scrambled

    @property
    def mean(self):
        return float(self.gray[:H_IMG].mean())


# ---- capture: OpenCV path ------------------------------------------------------

_BACKENDS = {
    "ffmpeg": "ffmpeg",
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


class XrealCapture:
    """An opened, byte-order-normalized XREAL camera stream (OpenCV)."""

    def __init__(self, cap, backend, index, swapped, dialect):
        self.cap, self.backend, self.index = cap, backend, index
        self.swapped, self.dialect = swapped, dialect

    @property
    def description(self):
        return (f"{_BACKEND_NAMES.get(self.backend, self.backend)}:{self.index}"
                f" dialect {self.dialect.name}"
                + (" (byte-swapped delivery)" if self.swapped else ""))

    def read(self, tries=8):
        """Next frame, or None if the stream keeps misbehaving."""
        for _ in range(tries):
            ok, frame = self.cap.read()
            flat = _raw_bytes(frame) if ok else None
            if flat is None:
                continue
            if self.swapped:
                flat = _unswap(flat)
            return XrealFrame(flat.reshape(H_FULL, W), self.dialect)
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


def _looks_like_xreal(cap):
    """Cheap property check, no frame reads needed."""
    return (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) in ((640.0, 241.0), (640.0, 482.0))


def _probe(cap, budget_s=6.0):
    """Read frames for up to budget_s and fingerprint them."""
    t0 = time.time()
    decoded = 0
    while time.time() - t0 < budget_s:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        flat = _raw_bytes(frame)
        if flat is None:
            # frames arrive but never at the raw size: backend is decoding
            # (e.g. Windows DirectShow force-converts to BGR) - give up early
            decoded += 1
            if decoded >= 5:
                return None
            continue
        got = _classify(flat)
        if got:
            return got
    return None


# ---- capture: ffmpeg/dshow path (Windows) --------------------------------------

def _ffmpeg_exe():
    return shutil.which("ffmpeg")


def ffmpeg_find_device(exe):
    """(friendly_name, moniker) of the XREAL dshow device, or None."""
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=20).stderr
    except (OSError, subprocess.TimeoutExpired):
        return None
    last_video = None
    devices = []   # (name, alternative_name)
    for line in out.splitlines():
        m = re.search(r'"([^"]+)"\s+\(video\)', line)
        if m:
            last_video = m.group(1)
            continue
        m = re.search(r'Alternative name\s+"([^"]+)"', line)
        if m and last_video is not None:
            devices.append((last_video, m.group(1)))
            last_video = None
    for name, alt in devices:
        if XREAL_VIDPID in alt.lower():
            return name, alt
    for name, alt in devices:
        if "uvc camera" in name.lower():
            return name, alt
    return None


class FFmpegCapture:
    """XREAL stream via `ffmpeg -f dshow` piping raw yuyv422 frames.

    A drain thread reads the pipe at full stream rate so the device never
    stalls; read() hands out frames as they arrive (latest_only=True keeps
    only the newest frame - right for a live viewer; False queues every
    frame - right for a recorder).
    """

    def __init__(self, exe, device, latest_only=True):
        self.device_name, moniker = device
        self._proc = subprocess.Popen(
            [exe, "-hide_banner", "-loglevel", "error",
             "-f", "dshow", "-video_size", f"{W}x241", "-pixel_format", "yuyv422",
             "-rtbufsize", "8M", "-i", f"video={moniker}",
             "-f", "rawvideo", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        self.swapped = False
        self.dialect = None
        self._latest_only = latest_only
        self._cond = threading.Condition()
        self._frames = []            # pending frames (len<=1 if latest_only)
        self._eof = False
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    @property
    def description(self):
        d = self.dialect.name if self.dialect else "?"
        return (f"ffmpeg:dshow:{self.device_name} dialect {d}"
                + (" (byte-swapped delivery)" if self.swapped else ""))

    def _drain(self):
        stream = self._proc.stdout
        while True:
            data = stream.read(FRAME_BYTES)
            if not data or len(data) < FRAME_BYTES:
                break
            flat = np.frombuffer(data, np.uint8)
            with self._cond:
                if self._latest_only:
                    self._frames[:] = [flat]
                else:
                    self._frames.append(flat)
                self._cond.notify()
        with self._cond:
            self._eof = True
            self._cond.notify()

    def read(self, timeout=2.0):
        """Next frame, or None on timeout/EOF."""
        deadline = time.time() + timeout
        while True:
            with self._cond:
                while not self._frames and not self._eof:
                    left = deadline - time.time()
                    if left <= 0 or not self._cond.wait(left):
                        return None
                if not self._frames:
                    return None      # EOF
                flat = self._frames.pop(0)
            if self.dialect is None:
                got = _classify(flat)
                if got is None:
                    if time.time() < deadline:
                        continue     # startup black frame
                    return None
                order, self.dialect = got
                self.swapped = order == "swapped"
            if self.swapped:
                flat = _unswap(flat)
            return XrealFrame(flat.reshape(H_FULL, W).copy(), self.dialect)

    def release(self):
        if self._proc.poll() is None:
            self._proc.kill()
        self._proc.wait()
        self._thread.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


# ---- device discovery -----------------------------------------------------------

def find_camera(index=None, backend=None, verbose=False, max_index=10,
                latest_only=True):
    """Scan for the XREAL stream; returns XrealCapture / FFmpegCapture or None.

    index/backend pin the search (backend 'ffmpeg' forces the ffmpeg path).
    On Windows ffmpeg is preferred when available - it is the only capture
    path that demonstrably delivers raw bytes from this device there.
    latest_only tunes the ffmpeg path: True drops stale frames (viewer),
    False keeps every frame (recorder).
    """
    try:  # silence per-index open errors from OpenCV during the scan
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass

    want_ffmpeg = backend == "ffmpeg" or (
        backend is None and platform.system() == "Windows")
    if want_ffmpeg:
        exe = _ffmpeg_exe()
        if exe:
            dev = ffmpeg_find_device(exe)
            if dev:
                cap = FFmpegCapture(exe, dev, latest_only=latest_only)
                if cap.read(timeout=6.0) is not None:
                    if verbose:
                        print(f"  ffmpeg:dshow:{dev[0]} -> ok")
                    return cap
                cap.release()
                if verbose:
                    print(f"  ffmpeg:dshow:{dev[0]} -> no classifiable frames")
            elif verbose:
                print("  ffmpeg: no XREAL dshow device")
        elif verbose:
            print("  ffmpeg not on PATH, falling back to OpenCV")
        if backend == "ffmpeg":
            return None

    backends = [backend] if backend is not None else default_backends()
    indices = [index] if index is not None else range(max_index)
    # first pass: only devices whose properties scream XREAL (640x241);
    # second pass: probe everything else that opened
    for only_candidates in (True, False):
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
                if only_candidates and not _looks_like_xreal(cap):
                    cap.release()
                    continue
                got = _probe(cap)
                if verbose:
                    print(f"  {_BACKEND_NAMES.get(be, be)}:{i} -> "
                          f"{'%s/%s' % (got[0], got[1].name) if got else 'not an XREAL stream'}")
                if got:
                    order, dialect = got
                    return XrealCapture(cap, be, i, order == "swapped", dialect)
                cap.release()
    return None


# ---- scan mode (the enumerate.swift equivalent) --------------------------------

def _scan():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    print("Scanning for the XREAL Air 2 Ultra UVC stream...")
    if platform.system() == "Windows":
        exe = _ffmpeg_exe()
        if exe:
            dev = ffmpeg_find_device(exe)
            print(f"ffmpeg/dshow: {'found ' + dev[0] if dev else 'no XREAL device'}")
            if dev:
                with FFmpegCapture(exe, dev) as cap:
                    f = cap.read(timeout=6.0)
                    print(f"  streams: {'yes, ' + cap.description if f else 'NO FRAMES'}")
        else:
            print("ffmpeg/dshow: ffmpeg not on PATH (winget install ffmpeg)")
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
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            note = ""
            if _looks_like_xreal(cap):
                got = _probe(cap)
                note = (" <- XREAL, raw %s, dialect %s" % (got[0], got[1].name)
                        if got else " <- XREAL-shaped but no raw frames")
            print(f"  index {i}: {w:.0f}x{h:.0f}{note}")
            cap.release()
        if not found_any:
            print("  no cameras")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(__doc__)
        sys.exit(0)
    _scan()
