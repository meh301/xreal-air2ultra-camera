#!/usr/bin/env python3
"""Ground-truth test of the IMU stream's rotation signs — no fusion, no
rendering, no judgment calls: perform three prompted physical rotations and
read the raw gyro channel signs directly.

Why this exists: the accelerometer pins the chip's up axis (z reads +1 g at
rest), but gravity says nothing about rotation signs. A 6-axis AHRS then
builds a self-consistent world out of ANY channel-sign convention, so a
correct-looking attitude view cannot distinguish a healthy gyro from one
with flipped channels. This tool can: it compares each raw gyro channel
against a known physical rotation.

Run, follow the three prompts (each: hold the pose, press Enter, then
rotate smoothly for ~2 s):

  1) glasses FLAT on the desk (lenses horizontal, arms folded), rotate the
     whole pair COUNTERCLOCKWISE seen from above
  2) glasses flat, then TIP THE FRONT (lens side) UPWARD ~45 deg and back
  3) glasses flat, then TILT THE RIGHT SIDE (right lens) DOWN ~45 deg and back

It prints the dominant raw channel and sign for each motion plus the mean
rest accel. No interpretation is applied — paste the output as-is.
"""

import sys
import time

from xreal_imu import XrealImu


def collect(imu, seconds):
    t0 = time.time()
    acc = [0.0, 0.0, 0.0]
    gyr = [0.0, 0.0, 0.0]
    n = 0
    for s in imu.samples():
        if s is None:
            continue
        for i in range(3):
            gyr[i] += s.gyro_dps[i]
            acc[i] += s.accel_g[i]
        n += 1
        if time.time() - t0 > seconds:
            break
    return [g / max(n, 1) for g in gyr], [a / max(n, 1) for a in acc], n


def main():
    tests = [
        ("FLAT on desk -> rotate COUNTERCLOCKWISE (seen from above)", "yaw"),
        ("FLAT on desk -> tip the FRONT (lenses) UP ~45 deg", "pitch"),
        ("FLAT on desk -> tilt the RIGHT side DOWN ~45 deg", "roll"),
    ]
    with XrealImu() as imu:
        imu.stream(True)
        print("draining startup...")
        collect(imu, 0.5)

        input("rest the glasses FLAT and still, then press Enter...")
        g0, a0, n = collect(imu, 1.0)
        print(f"  rest: accel=({a0[0]:+.3f},{a0[1]:+.3f},{a0[2]:+.3f}) g  "
              f"gyro bias=({g0[0]:+.3f},{g0[1]:+.3f},{g0[2]:+.3f}) deg/s  "
              f"[{n} samples]")

        for prompt, name in tests:
            input(f"\n{name.upper()}: {prompt}\n  press Enter, then move "
                  "smoothly for ~2 s...")
            g, a, n = collect(imu, 2.0)
            g = [gi - bi for gi, bi in zip(g, g0)]
            dom = max(range(3), key=lambda i: abs(g[i]))
            print(f"  mean gyro=({g[0]:+7.2f},{g[1]:+7.2f},{g[2]:+7.2f}) deg/s"
                  f"  -> dominant: {'xyz'[dom]} {'+' if g[dom] > 0 else '-'}"
                  f"  [{n} samples]")

        imu.stream(False)
    print("\npaste the four result lines verbatim.")


if __name__ == "__main__":
    sys.exit(main())
