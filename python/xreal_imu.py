#!/usr/bin/env python3
"""1 kHz IMU reader for the XREAL Air 2 Ultra (Windows / Linux / macOS).

The glasses stream their IMU over vendor HID interface 2. Commands on that
interface use an 0xAA-framed packet (a shorter sibling of the 0xFD MCU
protocol, see docs/PROTOCOL.md):

    aa | crc32 u32 | length u16 (= len(data)+3) | cmd u8 | data...
    crc32 = zlib CRC-32 over bytes [5 : 5+length]

  cmd 0x19 data 01/00   IMU stream on / off
  cmd 0x14              -> u32 length of the calibration JSON
  cmd 0x15              -> next chunk of the calibration JSON

Once enabled, 512-byte input reports arrive at 1000 Hz:

    [0:2]   signature 01 02
    [2:4]   u16 temperature, raw
    [4:12]  u64 timestamp [ns]
    [12:14] u16 gyro multiplier      } raw * mul / div
    [14:18] u32 gyro divisor         }   = angular rate [deg/s]
    [18:27] 3 x i24 gyro x,y,z
    [27:29] u16 accel multiplier     } raw * mul / div
    [29:33] u32 accel divisor        }   = acceleration [g]
    [33:42] 3 x i24 accel x,y,z
    [42:48] u16 mul + u32 div for the magnetometer (values read 0 here)
    [48:54] 3 x i16 mag x,y,z
    [54:58] u32 secondary counter, ~976.56 us/tick (the sensor's 1024 Hz ODR)
    rest    zero padding

Values are reported in the raw sensor frame; the factory calibration
(biases, camera<->IMU extrinsics, fisheye624 intrinsics of both tracking
cameras) is stored on the device as JSON — dump it with --config.

Usage:
  python xreal_imu.py                    # live console output
  python xreal_imu.py --csv imu.csv      # also log every sample
  python xreal_imu.py --fb               # also publish shm ring "xreal_imu_fb"
  python xreal_imu.py --config calib.json  # dump factory calibration, exit
  python xreal_imu.py --seconds 10       # stop after N seconds

Needs `pip install hidapi`. On Linux run as root or add a udev rule for
VID 3318 (see README).
"""

import argparse
import struct
import sys
import time
import zlib

import hid

VID, PID, IMU_INTERFACE, MCU_INTERFACE = 0x3318, 0x0426, 2, 0
SIG = b"\x01\x02"

# shared-memory ring (--fb): 64B header + capacity slots of 48 bytes
#   header: u32 magic 'XRIM', u32 version=1, u32 capacity, u32 slot_size,
#           u64 write_count (slots [0, write_count) are valid,
#           newest slot = (write_count-1) % capacity), rest reserved
#   slot:   u64 ts_ns, u32 counter, u16 temp_raw, u16 pad,
#           f32 gyro x,y,z [deg/s], f32 accel x,y,z [g], f32 pad
IMU_FB_NAME = "xreal_imu_fb"
IMU_FB_MAGIC = 0x4D495258
IMU_FB_CAPACITY = 4096
IMU_FB_SLOT = 48
IMU_FB_HDR = 64


class ImuSample:
    __slots__ = ("ts_ns", "counter", "temp_raw", "gyro_dps", "accel_g")

    def __init__(self, report):
        b = report
        self.temp_raw, = struct.unpack_from("<H", b, 2)
        self.ts_ns, = struct.unpack_from("<Q", b, 4)
        gm, gd = struct.unpack_from("<HI", b, 12)
        am, ad = struct.unpack_from("<HI", b, 27)
        i24 = lambda o: int.from_bytes(b[o:o + 3], "little", signed=True)
        self.gyro_dps = tuple(i24(18 + 3 * i) * gm / gd for i in range(3))
        self.accel_g = tuple(i24(33 + 3 * i) * am / ad for i in range(3))
        self.counter, = struct.unpack_from("<I", b, 54)


def _packet(cmd, data=b""):
    body = struct.pack("<HB", len(data) + 3, cmd) + data
    return bytes([0xAA]) + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF) + body


# ---- MCU channel (interface 0, 0xFD framing) — device info queries -----------

def _mcu_packet(cmd, data=b""):
    body = struct.pack("<HIIH5s", len(data) + 17, 0x1337, 0, cmd, b"\0" * 5) + data
    return bytes([0xFD]) + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF) + body


def mcu_query(cmd, data=b"", tries=50):
    """One read command on the MCU interface; returns (status, payload)."""
    path = next((d["path"] for d in hid.enumerate(VID, PID)
                 if d["interface_number"] == MCU_INTERFACE), None)
    if path is None:
        raise FileNotFoundError("MCU HID interface 0 not found")
    dev = hid.device()
    dev.open_path(path)
    try:
        dev.write(b"\x00" + _mcu_packet(cmd, data).ljust(64, b"\0"))
        for _ in range(tries):
            buf = dev.read(1024, timeout_ms=250)
            if not buf:
                return None
            b = bytes(buf)
            if b[:1] == b"\xfd":
                length, = struct.unpack_from("<H", b, 5)
                rcmd, = struct.unpack_from("<H", b, 15)
                if rcmd == cmd:
                    payload = b[22:22 + max(0, length - 17)]
                    return (payload[0] if payload else None,
                            payload[1:].decode("ascii", "replace").strip("\x00"))
        return None
    finally:
        dev.close()


def print_info():
    """Serial + firmware versions (command ids per the Air-family MCU
    protocol, wheaney/nrealAirLinuxDriver device_mcu.h)."""
    for label, cmd in (("serial       ", 0x15), ("MCU app FW   ", 0x26),
                       ("DP7911 FW    ", 0x16), ("DSP app FW   ", 0x21)):
        got = mcu_query(cmd)
        print(f"{label}: {got[1] if got else 'no reply'}")


class XrealImu:
    """The IMU HID channel. Use as a context manager; iterate samples()."""

    def __init__(self):
        path = next((d["path"] for d in hid.enumerate(VID, PID)
                     if d["interface_number"] == IMU_INTERFACE), None)
        if path is None:
            raise FileNotFoundError(
                "XREAL Air 2 Ultra HID interface 2 not found - glasses plugged in? "
                "(on Linux: udev rule or root needed)")
        self.dev = hid.device()
        self.dev.open_path(path)
        self._streaming = False

    def command(self, cmd, data=b"", tries=100):
        """Send one 0xAA command, return its reply payload (or None)."""
        self.dev.write(b"\x00" + _packet(cmd, data).ljust(64, b"\0"))
        for _ in range(tries):
            buf = self.dev.read(1024, timeout_ms=250)
            if not buf:
                return None
            b = bytes(buf)
            if b[:1] == b"\xaa":                      # command reply
                length, rcmd = struct.unpack_from("<HB", b, 5)
                if rcmd == cmd:
                    return b[8:8 + max(0, length - 3)]
        return None

    def stream(self, on):
        if self.command(0x19, b"\x01" if on else b"\x00") is None:
            raise RuntimeError("no ack for IMU stream command")
        self._streaming = on

    def config_json(self):
        """Fetch the factory calibration JSON stored on the glasses."""
        was = self._streaming
        if was:
            self.stream(False)
        raw = self.command(0x14)
        if raw is None or len(raw) < 4:
            raise RuntimeError("config length request failed")
        total, = struct.unpack("<I", raw[:4])
        cfg = bytearray()
        while len(cfg) < total:
            part = self.command(0x15)
            if part is None:
                raise RuntimeError(f"config read stalled at {len(cfg)}/{total}")
            cfg.extend(part)
        if was:
            self.stream(True)
        return bytes(cfg[:total])

    def samples(self):
        """Generator of ImuSample. Call stream(True) first."""
        while True:
            buf = self.dev.read(1024, timeout_ms=500)
            if not buf:
                yield None                            # stall marker
                continue
            b = bytes(buf)
            if b[:2] == SIG:
                yield ImuSample(b)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # deliberately leave the stream running: it is free-running and
        # costless with no listener, and turning it off here would cut the
        # feed under any other process reading it in parallel
        self.dev.close()


class ImuRingWriter:
    """Publish samples to the named shared-memory ring (layout above)."""

    def __init__(self):
        from multiprocessing import shared_memory
        size = IMU_FB_HDR + IMU_FB_CAPACITY * IMU_FB_SLOT
        try:
            self.shm = shared_memory.SharedMemory(name=IMU_FB_NAME, create=True,
                                                  size=size)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name=IMU_FB_NAME)
        struct.pack_into("<IIII", self.shm.buf, 0,
                         IMU_FB_MAGIC, 1, IMU_FB_CAPACITY, IMU_FB_SLOT)
        struct.pack_into("<Q", self.shm.buf, 16, 0)
        self.count = 0

    def publish(self, s):
        off = IMU_FB_HDR + (self.count % IMU_FB_CAPACITY) * IMU_FB_SLOT
        struct.pack_into("<QIHH3f3f4x", self.shm.buf, off,
                         s.ts_ns, s.counter, s.temp_raw, 0,
                         *s.gyro_dps, *s.accel_g)
        self.count += 1
        struct.pack_into("<Q", self.shm.buf, 16, self.count)  # publish after write

    def close(self):
        self.shm.close()
        try:
            self.shm.unlink()          # no-op on Windows
        except FileNotFoundError:
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", metavar="OUT.csv", help="log every sample")
    ap.add_argument("--fb", action="store_true",
                    help=f'publish to shared-memory ring "{IMU_FB_NAME}"')
    ap.add_argument("--config", metavar="OUT.json",
                    help="dump the factory calibration JSON and exit")
    ap.add_argument("--quat", action="store_true",
                    help="fuse a 6-axis quaternion host-side (Madgwick; the "
                         "device itself streams raw data only)")
    ap.add_argument("--info", action="store_true",
                    help="print serial + firmware versions (MCU channel) and exit")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    args = ap.parse_args()

    if args.info:
        print_info()
        return

    with XrealImu() as imu:
        if args.config:
            cfg = imu.config_json()
            with open(args.config, "wb") as fh:
                fh.write(cfg)
            print(f"factory calibration ({len(cfg)} bytes) -> {args.config}")
            return

        csv = open(args.csv, "w") if args.csv else None
        if csv:
            csv.write("ts_ns,counter,temp_raw,gx_dps,gy_dps,gz_dps,ax_g,ay_g,az_g"
                      + (",qw,qx,qy,qz" if args.quat else "") + "\n")
        ring = ImuRingWriter() if args.fb else None
        if ring:
            print(f'publishing to shared memory "{IMU_FB_NAME}"')
        ahrs = None
        if args.quat:
            from xreal_ahrs import ImuOrientation, quat_to_euler_deg
            ahrs = ImuOrientation()
            print("estimating gyro bias, hold the glasses still for ~1 s...")

        imu.stream(True)
        print("IMU stream on (1 kHz). Ctrl+C to stop.")
        n, t0 = 0, time.time()
        end = time.time() + args.seconds if args.seconds else None
        try:
            for s in imu.samples():
                if end and time.time() > end:
                    break
                if s is None:
                    print("stream stalled...", flush=True)
                    continue
                n += 1
                q = ahrs.feed(s) if ahrs else None
                if csv:
                    csv.write(f"{s.ts_ns},{s.counter},{s.temp_raw},"
                              f"{s.gyro_dps[0]:.4f},{s.gyro_dps[1]:.4f},{s.gyro_dps[2]:.4f},"
                              f"{s.accel_g[0]:.5f},{s.accel_g[1]:.5f},{s.accel_g[2]:.5f}"
                              + (",%.6f,%.6f,%.6f,%.6f" % tuple(q) if q else
                                 ",,,," if ahrs else "") + "\n")
                if ring:
                    ring.publish(s)
                dt = time.time() - t0
                if dt >= 1:
                    if q:
                        yaw, pitch, roll = quat_to_euler_deg(q)
                        print(f"{n/dt:6.0f} Hz  q=({q[0]:+6.3f},{q[1]:+6.3f},{q[2]:+6.3f},"
                              f"{q[3]:+6.3f})  yaw={yaw:+7.1f} pitch={pitch:+6.1f} "
                              f"roll={roll:+7.1f} deg"
                              + ("" if ahrs.still_capture else "  [bias capture was noisy]"),
                              flush=True)
                    else:
                        a = s.accel_g
                        mag = (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5
                        print(f"{n/dt:6.0f} Hz  gyro=({s.gyro_dps[0]:+8.3f},{s.gyro_dps[1]:+8.3f},"
                              f"{s.gyro_dps[2]:+8.3f}) deg/s  accel=({a[0]:+6.3f},{a[1]:+6.3f},"
                              f"{a[2]:+6.3f}) g  |a|={mag:5.3f}  t_raw={s.temp_raw}",
                              flush=True)
                    n, t0 = 0, time.time()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            if csv:
                csv.close()
            if ring:
                ring.close()


if __name__ == "__main__":
    main()
