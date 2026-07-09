#!/usr/bin/env python3
"""Host-side orientation (quaternion) for the XREAL Air 2 Ultra IMU.

The glasses stream only raw gyro/accel at 1 kHz (docs/PROTOCOL.md) - there is
no onboard orientation output; even the vendor stack fuses host-side. This
module provides that fusion: a standard 6-axis Madgwick AHRS plus automatic
gyro-bias capture, yielding a gravity-referenced attitude in the sensor frame.

Without a magnetometer (the mag fields read zero on the Ultra), yaw has no
absolute reference and drifts slowly - fine for visualization; a vSLAM system
should consume the raw stream and estimate biases itself.
"""

import math


class MadgwickAHRS:
    """Madgwick's 6-axis (gyro+accel) gradient-descent filter.

    Direct port of the IMU update from the reference implementation
    (x-io MadgwickAHRS). q is the sensor->earth rotation as (w, x, y, z).
    """

    def __init__(self, beta=0.05):
        self.q = [1.0, 0.0, 0.0, 0.0]
        self.beta = beta

    def update(self, gx, gy, gz, ax, ay, az, dt):
        """gx..gz in rad/s, ax..az in any consistent unit, dt in seconds."""
        q1, q2, q3, q4 = self.q

        qdot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz)
        qdot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy)
        qdot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx)
        qdot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx)

        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 0.0:
            ax, ay, az = ax / norm, ay / norm, az / norm
            _2q1, _2q2, _2q3, _2q4 = 2 * q1, 2 * q2, 2 * q3, 2 * q4
            _4q1, _4q2, _4q3 = 4 * q1, 4 * q2, 4 * q3
            _8q2, _8q3 = 8 * q2, 8 * q3
            q1q1, q2q2, q3q3, q4q4 = q1 * q1, q2 * q2, q3 * q3, q4 * q4

            s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay
            s2 = (_4q2 * q4q4 - _2q4 * ax + 4 * q1q1 * q2 - _2q1 * ay - _4q2
                  + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az)
            s3 = (4 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3
                  + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az)
            s4 = 4 * q2q2 * q4 - _2q2 * ax + 4 * q3q3 * q4 - _2q3 * ay
            n = math.sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4)
            if n > 0.0:
                b = self.beta / n
                qdot1 -= b * s1
                qdot2 -= b * s2
                qdot3 -= b * s3
                qdot4 -= b * s4

        q1 += qdot1 * dt
        q2 += qdot2 * dt
        q3 += qdot3 * dt
        q4 += qdot4 * dt
        n = math.sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4) or 1.0
        self.q = [q1 / n, q2 / n, q3 / n, q4 / n]
        return self.q


def quat_to_euler_deg(q):
    """(yaw, pitch, roll) in degrees, ZYX convention, from (w,x,y,z)."""
    w, x, y, z = q
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def quat_rotate(q, v):
    """Rotate vector v (sensor frame) into the earth frame by q=(w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v
    # v' = v + 2*r x (r x v + w*v), r = (x,y,z)
    rx, ry, rz = (y * vz - z * vy + w * vx,
                  z * vx - x * vz + w * vy,
                  x * vy - y * vx + w * vz)
    return (vx + 2 * (y * rz - z * ry),
            vy + 2 * (z * rx - x * rz),
            vz + 2 * (x * ry - y * rx))


class ImuOrientation:
    """Bias capture + Madgwick, fed with xreal_imu.ImuSample objects.

    Phase 1 (~1 s): accumulate gyro to estimate the bias (works if the
    glasses are reasonably still). Phase 2: track, with an aggressive beta
    for the first second so the attitude snaps to gravity, then settle.
    """

    CAPTURE_N = 1000

    def __init__(self):
        self.filter = MadgwickAHRS(beta=0.5)
        self.bias = (0.0, 0.0, 0.0)
        self.still_capture = True
        self._acc = [0.0, 0.0, 0.0]
        self._minmax = [[1e9, -1e9], [1e9, -1e9], [1e9, -1e9]]
        self._n = 0
        self._last_ts = None
        self._tracked = 0

    @property
    def capturing(self):
        return self._n < self.CAPTURE_N

    def feed(self, s):
        """Feed one ImuSample; returns the quaternion, or None while the
        bias capture is still running."""
        if self.capturing:
            for i, v in enumerate(s.gyro_dps):
                self._acc[i] += v
                self._minmax[i][0] = min(self._minmax[i][0], v)
                self._minmax[i][1] = max(self._minmax[i][1], v)
            self._n += 1
            if not self.capturing:
                self.bias = tuple(a / self._n for a in self._acc)
                # >3 deg/s swing during capture = the user was moving
                self.still_capture = all(hi - lo < 3.0 for lo, hi in self._minmax)
                self._last_ts = s.ts_ns
            return None

        dt = (s.ts_ns - self._last_ts) / 1e9 if self._last_ts else 1e-3
        self._last_ts = s.ts_ns
        dt = min(max(dt, 0.0), 0.005)
        self._tracked += 1
        if self._tracked == 1000:      # attitude has converged: settle down
            self.filter.beta = 0.05
        g = [math.radians(v - b) for v, b in zip(s.gyro_dps, self.bias)]
        return self.filter.update(*g, *s.accel_g, dt)
