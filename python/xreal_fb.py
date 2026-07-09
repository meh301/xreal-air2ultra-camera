#!/usr/bin/env python3
"""Named shared-memory framebuffer for the clean XREAL stereo feed.

The viewer (preview_clean.py) publishes every descrambled + denoised stereo
pair here so *any* other process - Python, C/C++, Rust, Unity, ... - can map
the block by name and consume the live feed without sockets or disk.

    name  : "xreal_stereo_fb"   (shm_open name / Windows section name)
    layout: 64-byte header + 960*640 grayscale pixels (left eye cols 0-479,
            right eye cols 480-959, row-major, 8 bit)

    header (little-endian):
      off  0  u32  magic 0x42465258 ("XRFB")
      off  4  u32  version = 1
      off  8  u32  width  (960)
      off 12  u32  height (640)
      off 16  u32  seq    - seqlock: odd while the writer is mid-update;
                            re-read until it is even and unchanged
      off 20  u32  pair counter (from the telemetry row)
      off 24  f64  time.time() at publish
      off 32  ..   reserved to 64

Run directly to consume the feed:  python xreal_fb.py [--show]
"""

import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np

FB_NAME = "xreal_stereo_fb"
FB_W, FB_H = 960, 640
FB_MAGIC = 0x42465258
HDR_SIZE = 64
FB_SIZE = HDR_SIZE + FB_W * FB_H


class FramebufferWriter:
    def __init__(self, name=FB_NAME):
        try:
            self.shm = shared_memory.SharedMemory(name=name, create=True, size=FB_SIZE)
        except FileExistsError:   # stale block from a previous run: reuse
            self.shm = shared_memory.SharedMemory(name=name)
        self._seq = 0
        buf = self.shm.buf
        struct.pack_into("<IIII", buf, 0, FB_MAGIC, 1, FB_W, FB_H)
        self._pixels = np.ndarray((FB_H, FB_W), np.uint8, buf, HDR_SIZE)

    def publish(self, stereo, counter):
        """stereo: (640, 960) uint8. One copy into the mapped region."""
        buf = self.shm.buf
        self._seq += 1                                   # odd: update begins
        struct.pack_into("<I", buf, 16, self._seq & 0xFFFFFFFF)
        np.copyto(self._pixels, stereo)
        struct.pack_into("<Id", buf, 20, counter & 0xFFFFFFFF, time.time())
        self._seq += 1                                   # even: stable
        struct.pack_into("<I", buf, 16, self._seq & 0xFFFFFFFF)

    def close(self):
        self.shm.close()
        try:
            self.shm.unlink()      # no-op on Windows
        except FileNotFoundError:
            pass


class FramebufferReader:
    def __init__(self, name=FB_NAME):
        self.shm = shared_memory.SharedMemory(name=name)   # raises if absent
        magic, ver, w, h = struct.unpack_from("<IIII", self.shm.buf, 0)
        if magic != FB_MAGIC:
            raise RuntimeError("shared memory block is not an XRFB framebuffer")
        self.size = (h, w)
        self._pixels = np.ndarray((h, w), np.uint8, self.shm.buf, HDR_SIZE)
        self._last_seq = 0

    def read(self):
        """Latest stable frame as (image copy, counter, timestamp), or None
        if nothing new since the previous call."""
        for _ in range(1000):
            seq1, = struct.unpack_from("<I", self.shm.buf, 16)
            if seq1 & 1:
                continue                        # writer mid-update
            if seq1 == self._last_seq:
                return None                     # nothing new
            img = self._pixels.copy()
            counter, ts = struct.unpack_from("<Id", self.shm.buf, 20)
            seq2, = struct.unpack_from("<I", self.shm.buf, 16)
            if seq1 == seq2:                    # torn read? retry
                self._last_seq = seq1
                return img, counter, ts
        return None

    def close(self):
        self.shm.close()


def main():
    show = "--show" in sys.argv
    print(f"Attaching to '{FB_NAME}' (start preview_clean.py first)...")
    while True:
        try:
            rd = FramebufferReader()
            break
        except FileNotFoundError:
            time.sleep(0.5)
    print(f"attached: {rd.size[1]}x{rd.size[0]} stereo feed")
    if show:
        import cv2
    n, t0 = 0, time.time()
    try:
        while True:
            got = rd.read()
            if got is None:
                time.sleep(0.002)
                continue
            img, counter, ts = got
            n += 1
            if time.time() - t0 >= 1:
                age_ms = (time.time() - ts) * 1000
                print(f"{n} pairs/s  counter={int(counter)}  latency={age_ms:.0f} ms "
                      f"mean={img.mean():.0f}")
                n, t0 = 0, time.time()
            if show:
                cv2.imshow("xreal_fb consumer", img)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    rd.close()


if __name__ == "__main__":
    main()
