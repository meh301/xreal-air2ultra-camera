#!/usr/bin/env python3
"""World-aligned (1:1) passthrough mapping from the factory calibration.

Goal: a pixel on the glasses' virtual display shows exactly the piece of the
world that sits behind it, so the passthrough view lines up with reality.

For every display pixel of one eye:
  1. unproject through that eye's virtual-display intrinsics (`k_*_display`
     from the calibration JSON; ~2484 px focal over 1920x1080 = ~42x25 deg),
  2. rotate the ray display->IMU->camera using the calibrated poses
     (`target_q_*_display`, `imu_q_cam`; quaternions are JPL xyzw per the
     calibration's own comment),
  3. project through the same-side tracking camera's fisheye624 model
     (fc/cc + 12 kc: 6 radial theta-polynomial, 2 tangential, 4 thin-prism),
  4. sample the descrambled 480x640 image there.

Left eye uses SLAM_camera device_1 (= cam1, the physical left camera), right
eye device_2 (= cam0). Rays assume a scene at `depth` (default: infinity =
rotation only); nearby objects will show parallax swim since the cameras sit
a few cm from the eyes - that is inherent to reprojection-free passthrough.

The maps are built once (numpy) and applied per frame with cv2.remap.
Used by glasses_passthrough.py --align <calib.json>; dump the calibration
with `python xreal_imu.py --config calib.json`.
"""

import json

import cv2
import numpy as np


def quat_to_rot(q, conj=True):
    """3x3 rotation matrix from a quaternion given as [x, y, z, w].

    conj flips the vector part - JPL and Hamilton conventions differ exactly
    by this, which is what the runtime "variant" cycling toggles per
    quaternion (variant 3 = both conjugated = the JPL reading, expected
    correct for this calibration).
    """
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    if conj:
        x, y, z = -x, -y, -z
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def fisheye624_project(pts, fc, cc, kc):
    """Project Nx3 camera-frame points with the fisheye624 model.

    kc = [k0..k5, p0, p1, s0..s3]: theta-polynomial radial distortion,
    tangential, thin-prism (the model projectaria calls
    FisheyeRadTanThinPrism). Returns Nx2 pixel coords.
    """
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.hypot(x, y)
    theta = np.arctan2(r, z)
    t2 = theta * theta
    poly = 1 + t2 * (kc[0] + t2 * (kc[1] + t2 * (kc[2] + t2 * (
        kc[3] + t2 * (kc[4] + t2 * kc[5])))))
    th_d = theta * poly
    safe_r = np.where(r > 1e-9, r, 1.0)
    xr = np.where(r > 1e-9, th_d * x / safe_r, 0.0)
    yr = np.where(r > 1e-9, th_d * y / safe_r, 0.0)

    r2 = xr * xr + yr * yr
    p0, p1 = kc[6], kc[7]
    s0, s1, s2, s3 = kc[8], kc[9], kc[10], kc[11]
    dx = 2 * p0 * xr * yr + p1 * (r2 + 2 * xr * xr) + s0 * r2 + s1 * r2 * r2
    dy = p0 * (r2 + 2 * yr * yr) + 2 * p1 * xr * yr + s2 * r2 + s3 * r2 * r2

    u = fc[0] * (xr + dx) + cc[0]
    v = fc[1] * (yr + dy) + cc[1]
    return np.stack([u, v], axis=1)


def build_eye_maps(calib, eye, out_size, depth=np.inf, variant=3):
    """cv2.remap maps for one eye.

    calib: parsed calibration JSON. eye: 'left'/'right'. out_size: (w, h) of
    the rendered half (960x1080 for SBS: the panel half is stretched 2x
    horizontally by the glasses, so display column u = 2*u_half).
    Returns (map_x, map_y) float32, sampling the eye's descrambled 480x640
    camera image.
    """
    disp = calib["display"]
    cam = calib["SLAM_camera"]["device_1" if eye == "left" else "device_2"]
    K = np.array(disp[f"k_{eye}_display"], float).reshape(3, 3)
    R_imu_disp = quat_to_rot(disp[f"target_q_{eye}_display"], conj=bool(variant & 1))
    p_imu_disp = np.array(disp[f"target_p_{eye}_display"], float)
    R_imu_cam = quat_to_rot(cam["imu_q_cam"], conj=bool(variant >> 1 & 1))
    p_imu_cam = np.array(cam["imu_p_cam"], float)

    w, h = out_size
    full_w, full_h = disp.get("resolution", [1920, 1080])
    # SBS half -> full display pixel coordinates
    u = (np.arange(w) + 0.5) * (full_w / w) - 0.5
    v = (np.arange(h) + 0.5) * (full_h / h) - 0.5
    uu, vv = np.meshgrid(u, v)

    Kinv = np.linalg.inv(K)
    rays = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], axis=1) @ Kinv.T
    rays_imu = rays @ R_imu_disp.T

    if np.isinf(depth):
        pts_cam = rays_imu @ R_imu_cam           # rotation only
    else:
        pts_imu = rays_imu / rays_imu[:, 2:3] * depth + p_imu_disp
        pts_cam = (pts_imu - p_imu_cam) @ R_imu_cam

    uv = fisheye624_project(pts_cam, cam["fc"], cam["cc"], cam["kc"])
    behind = pts_cam[:, 2] <= 1e-6
    uv[behind] = -1e4                            # sample outside -> black
    map_x = uv[:, 0].reshape(h, w).astype(np.float32)
    map_y = uv[:, 1].reshape(h, w).astype(np.float32)
    return map_x, map_y


def load_calib(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Aligner:
    """Precomputed 1:1 warp for both eyes; call with the two cleaned images."""

    def __init__(self, calib_path, half_size=(960, 1080), depth=np.inf,
                 variant=3):
        self._calib = load_calib(calib_path)
        self._half_size, self._depth = half_size, depth
        self.variant = variant
        self._build()

    def _build(self):
        self.maps = {
            eye: build_eye_maps(self._calib, eye, self._half_size,
                                self._depth, self.variant)
            for eye in ("left", "right")
        }

    def set_variant(self, variant):
        self.variant = variant & 3
        self._build()

    def warp(self, img_cam1_left, img_cam0_right):
        """(left_view, right_view) for the two descrambled 480x640 images."""
        lx, ly = self.maps["left"]
        rx, ry = self.maps["right"]
        left = cv2.remap(img_cam1_left, lx, ly, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
        right = cv2.remap(img_cam0_right, rx, ry, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)
        return left, right
