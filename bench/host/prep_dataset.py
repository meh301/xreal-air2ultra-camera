#!/usr/bin/env python
"""Convert a EuRoC / TUM-VI / MSD sequence into a replay pack.

Usage:
    prep_dataset.py euroc|tumvi|msd <archive_or_dir> <out_pack_dir> [--seq NAME]

<archive_or_dir> may be:
  * a .zip / .tar archive (extracted idempotently next to itself), or
  * an already-extracted directory (the sequence dir itself, or any parent
    containing one or more sequences -- a sequence is a dir containing mav0/).
If the input contains multiple sequences, pass --seq to pick one (the CLI
converts exactly one sequence per invocation; run_all.py iterates them all).

Conventions (see README.md for the full rationale):
  * calib.txt rotation/translation are cam->IMU (p_imu = R q * p_cam + p).
      - EuRoC sensor.yaml T_BS is sensor->body(=IMU): used directly.
      - TUM-VI Kalibr camchain T_cam_imu is IMU->cam: INVERTED here.
      - MSD Basalt calibration.json T_imu_cam is cam->IMU: used directly.
  * EuRoC radtan images are REMAPPED to a virtual kb4/equidistant camera
    (same fx,fy,cx,cy, k1..k4 = 0), so EuRoC pack frames are resampled.
  * MSD: cam0 = left, cam1 = right (Basalt/EuRoC-ASL ordering).
  * TUM-VI 16-bit PNGs are converted to 8-bit via >> 8.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from pack_common import (DATASETS, check_baseline, die, ensure_extracted,
                         format_calib, info, median_rate_hz, rot_to_quat_xyzw,
                         se3_inverse, warn, write_frames_csv, write_gt_tum,
                         write_imu_bin, write_meta)

MSD_HF_REPO = "collabora/monado-slam-datasets"

# Fallback IMU noise densities [gyro_n, gyro_bias_rw, accel_n, accel_bias_rw]
# in rad/s/sqrt(Hz), rad/s^2/sqrt(Hz), m/s^2/sqrt(Hz), m/s^3/sqrt(Hz).
TUMVI_DEFAULT_NOISES = [0.00016, 0.000022, 0.0028, 0.00086]   # TUM-VI paper/page
MSD_DEFAULT_NOISES = [0.00035, 0.0001, 0.00667, 0.001]        # benchmark defaults


# ---------------------------------------------------------------------------
# generic EuRoC-ASL directory helpers
# ---------------------------------------------------------------------------

def discover_sequences(input_path):
    """Return sorted [(seq_name, seq_root)] where seq_root contains mav0/."""
    root = ensure_extracted(input_path)
    seqs = []
    if (root / "mav0").is_dir():
        seqs.append((root.name, root))
    else:
        for mav in sorted(root.rglob("mav0")):
            if mav.is_dir():
                seqs.append((mav.parent.name, mav.parent))
    if not seqs:
        die(f"no sequence (dir containing mav0/) found under {root}")
    return seqs


def parse_csv_rows(path, ncols_min):
    """Parse a EuRoC-style CSV: '#' comments, comma- or space-separated,
    first column kept as exact int (timestamps overflow float64)."""
    if not path.exists():
        die(f"missing csv: {path}")
    rows = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split(",") if "," in ln else ln.split()
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < ncols_min:
            die(f"malformed line in {path} (need >= {ncols_min} cols): {ln!r}")
        rows.append(parts)
    if not rows:
        die(f"no data rows in {path}")
    return rows


def load_yaml(path):
    if not path.exists():
        die(f"missing yaml: {path}")
    txt = path.read_text(encoding="utf-8", errors="replace")
    # Strip OpenCV-style '%YAML:1.0' directives PyYAML rejects.
    lines = [ln for ln in txt.splitlines() if not ln.lstrip().startswith("%")]
    try:
        return yaml.safe_load("\n".join(lines))
    except yaml.YAMLError as e:
        die(f"failed to parse {path}: {e}")


def list_cam_frames(cam_dir):
    """[(ts_ns, image_path)] from camX/data.csv, else from data/*.png names."""
    data_csv = cam_dir / "data.csv"
    out = []
    if data_csv.exists():
        for parts in parse_csv_rows(data_csv, 2):
            out.append((int(parts[0]), cam_dir / "data" / parts[1]))
    else:
        img_dir = cam_dir / "data"
        if not img_dir.is_dir():
            die(f"no data.csv and no data/ dir in {cam_dir}")
        for p in sorted(img_dir.glob("*.png")):
            try:
                out.append((int(p.stem), p))
            except ValueError:
                die(f"cannot parse timestamp from image name {p.name} "
                    f"in {img_dir}")
    if not out:
        die(f"no frames found for camera {cam_dir}")
    out.sort(key=lambda x: x[0])
    return out


def pair_stereo(left_frames, right_frames, seq):
    lmap = dict(left_frames)
    rmap = dict(right_frames)
    common = sorted(set(lmap) & set(rmap))
    if not common:
        die(f"{seq}: no common timestamps between cam0 and cam1")
    dropped = (len(lmap) - len(common)) + (len(rmap) - len(common))
    if dropped:
        warn(f"{seq}: dropped {dropped} unpaired frames "
             f"(kept {len(common)} stereo pairs)")
    return [(ts, lmap[ts], rmap[ts]) for ts in common]


def load_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        die(f"failed to read image {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)   # 16-bit PNG -> 8-bit (TUM-VI)
    elif img.dtype != np.uint8:
        die(f"unsupported image dtype {img.dtype} in {path}")
    return np.ascontiguousarray(img)


def read_imu_csv(imu_csv):
    """EuRoC-format imu csv: ts_ns, wx wy wz [rad/s], ax ay az [m/s^2]."""
    rows = parse_csv_rows(imu_csv, 7)
    ts = np.array([int(r[0]) for r in rows], dtype=np.int64)
    vals = np.array([[float(x) for x in r[1:7]] for r in rows], dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    if not np.array_equal(order, np.arange(len(ts))):
        warn(f"{imu_csv}: timestamps not sorted -- sorting")
        ts, vals = ts[order], vals[order]
    return ts, vals[:, 0:3], vals[:, 3:6]


def read_gt_csv(gt_csv):
    """EuRoC-style GT csv: ts, px py pz, qw qx qy qz [, ...] -> TUM rows.
    Timestamps auto-detected as ns (>1e12) or seconds."""
    rows = parse_csv_rows(gt_csv, 8)
    out = []
    for r in rows:
        # timestamp may be integer ns or float seconds
        traw = r[0]
        if "." in traw or float(traw) < 1e12:
            ts_ns = int(round(float(traw) * 1e9))
        else:
            ts_ns = int(traw)
        p = [float(r[1]), float(r[2]), float(r[3])]
        qw, qx, qy, qz = (float(r[4]), float(r[5]), float(r[6]), float(r[7]))
        n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if abs(n - 1.0) > 1e-3:
            die(f"{gt_csv}: quaternion norm {n:.4f} at ts {traw} -- column "
                f"order assumption (ts,p,q_wxyz) looks wrong")
        out.append((ts_ns, p, [qx / n, qy / n, qz / n, qw / n]))
    out.sort(key=lambda x: x[0])
    return out


# ---------------------------------------------------------------------------
# EuRoC
# ---------------------------------------------------------------------------

def _euroc_cam_yaml(seq_root, cam):
    y = load_yaml(seq_root / "mav0" / cam / "sensor.yaml")
    intr = [float(v) for v in y["intrinsics"]]           # fu fv cu cv
    dist = [float(v) for v in y["distortion_coefficients"]]  # k1 k2 p1 p2
    model = str(y.get("distortion_model", "")).lower()
    if "radial" not in model and "radtan" not in model:
        die(f"EuRoC {cam}: expected radial-tangential distortion, "
            f"got {model!r}")
    T_BS = np.array(y["T_BS"]["data"], dtype=float).reshape(4, 4)  # cam->IMU
    res = [int(v) for v in y["resolution"]]              # [W, H]
    return intr, dist, T_BS, res


def build_equidistant_remap(W, H, intr, dist_radtan):
    """Map from a virtual kb4 camera (same fx,fy,cx,cy, k1..k4=0, i.e. pure
    equidistant r = f*theta) back into the source pinhole+radtan image."""
    fx, fy, cx, cy = intr
    k1, k2, p1, p2 = dist_radtan[:4]
    u = (np.arange(W, dtype=np.float64) - cx) / fx
    v = (np.arange(H, dtype=np.float64) - cy) / fy
    mx, my = np.meshgrid(u, v)                     # (H, W)
    theta = np.sqrt(mx * mx + my * my)             # equidistant: r_n = theta
    valid = theta < np.deg2rad(88.0)
    s = np.where(theta > 1e-9, np.sin(theta) / np.maximum(theta, 1e-12), 1.0)
    X, Y, Z = s * mx, s * my, np.cos(theta)
    z = np.maximum(Z, 1e-9)
    x, y = X / z, Y / z                            # pinhole normalized
    r2 = x * x + y * y
    rad = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    map_x = (fx * xd + cx).astype(np.float32)
    map_y = (fy * yd + cy).astype(np.float32)
    map_x[~valid] = -1.0
    map_y[~valid] = -1.0
    return map_x, map_y


def prep_euroc(seq_root, seq):
    mav = seq_root / "mav0"
    cams = {}
    for cam in ("cam0", "cam1"):
        intr, dist, T_BS, res = _euroc_cam_yaml(seq_root, cam)
        cams[cam] = dict(intr=intr, dist=dist, T=T_BS, res=res)

    imu_yaml = load_yaml(mav / "imu0" / "sensor.yaml")
    try:
        noises = [float(imu_yaml["gyroscope_noise_density"]),
                  float(imu_yaml["gyroscope_random_walk"]),
                  float(imu_yaml["accelerometer_noise_density"]),
                  float(imu_yaml["accelerometer_random_walk"])]
    except KeyError as e:
        die(f"EuRoC imu0/sensor.yaml missing noise key {e}")

    calib = {"noises": noises}
    for side, cam in (("left", "cam0"), ("right", "cam1")):
        c = cams[cam]
        calib[side] = {
            "pinhole": c["intr"],
            "dist": [0.0, 0.0, 0.0, 0.0],   # virtual pure-equidistant kb4
            "q_xyzw": rot_to_quat_xyzw(c["T"][:3, :3]),
            "p": c["T"][:3, 3].tolist(),
        }

    # remap maps (built once per camera, from sensor.yaml resolution)
    remaps = {}
    for cam in ("cam0", "cam1"):
        W, H = cams[cam]["res"]
        remaps[cam] = build_equidistant_remap(W, H, cams[cam]["intr"],
                                              cams[cam]["dist"])
    info(f"{seq}: EuRoC radtan -> virtual kb4 remap maps built "
         f"({cams['cam0']['res'][0]}x{cams['cam0']['res'][1]})")

    def loader(path, cam):
        img = load_gray(path)
        mx, my = remaps[cam]
        if img.shape != mx.shape:
            die(f"{seq}: image {path} is {img.shape[::-1]}, sensor.yaml says "
                f"{mx.shape[::-1]}")
        return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    pairs = pair_stereo(list_cam_frames(mav / "cam0"),
                        list_cam_frames(mav / "cam1"), seq)
    imu = read_imu_csv(mav / "imu0" / "data.csv")
    gt_csv = mav / "state_groundtruth_estimate0" / "data.csv"
    gt_rows = read_gt_csv(gt_csv)
    return calib, pairs, loader, imu, gt_rows


# ---------------------------------------------------------------------------
# TUM-VI
# ---------------------------------------------------------------------------

def _find_camchain(seq_root):
    cands = sorted(seq_root.rglob("camchain*.yaml"))
    return cands[0] if cands else None


def prep_tumvi(seq_root, seq):
    mav = seq_root / "mav0"
    camchain_path = _find_camchain(seq_root)
    cams = {}
    if camchain_path is not None:
        info(f"{seq}: using Kalibr camchain {camchain_path}")
        chain = load_yaml(camchain_path)
        for cam in ("cam0", "cam1"):
            if cam not in chain:
                die(f"{camchain_path} has no {cam} entry")
            c = chain[cam]
            model = str(c.get("distortion_model", "")).lower()
            if model != "equidistant":
                die(f"TUM-VI {cam}: expected equidistant distortion, "
                    f"got {model!r}")
            if "T_cam_imu" in c:
                T_cam_imu = np.array(c["T_cam_imu"], dtype=float).reshape(4, 4)
            elif cam == "cam1" and "T_cn_cnm1" in c and "T" in cams.get("cam0", {}):
                # cam1 relative to cam0: T_cam1_imu = T_cn_cnm1 * T_cam0_imu
                T_cam0_imu = se3_inverse(cams["cam0"]["T"])
                T_cam_imu = (np.array(c["T_cn_cnm1"], dtype=float).reshape(4, 4)
                             @ T_cam0_imu)
            else:
                die(f"{camchain_path}: {cam} has no T_cam_imu")
            cams[cam] = dict(
                intr=[float(v) for v in c["intrinsics"]],
                dist=[float(v) for v in c["distortion_coeffs"]],
                # Kalibr T_cam_imu maps IMU-frame points to cam frame
                # (IMU->cam); invert to get cam->IMU.
                T=se3_inverse(T_cam_imu))
    else:
        warn(f"{seq}: no camchain*.yaml found -- falling back to "
             f"mav0/camX/sensor.yaml (verify convention!)")
        for cam in ("cam0", "cam1"):
            y = load_yaml(mav / cam / "sensor.yaml")
            model = str(y.get("distortion_model", "")).lower()
            if "equidistant" not in model:
                die(f"TUM-VI {cam} sensor.yaml: expected equidistant, "
                    f"got {model!r}")
            cams[cam] = dict(
                intr=[float(v) for v in y["intrinsics"]],
                dist=[float(v) for v in y["distortion_coefficients"]],
                # EuRoC-format sensor.yaml T_BS is sensor->body: cam->IMU.
                T=np.array(y["T_BS"]["data"], dtype=float).reshape(4, 4))

    # noises: mav0/imu0/sensor.yaml, else dso/imu_config.yaml (present in the
    # 512_16 tars, "inflated" Allan values), else dataset-page values
    noises = None
    keys = ("gyroscope_noise_density", "gyroscope_random_walk",
            "accelerometer_noise_density", "accelerometer_random_walk")
    for cand in (mav / "imu0" / "sensor.yaml",
                 seq_root / "dso" / "imu_config.yaml"):
        if cand.exists():
            y = load_yaml(cand) or {}
            if all(k in y for k in keys):
                noises = [float(y[k]) for k in keys]
                info(f"{seq}: IMU noises from {cand.name}: {noises}")
                break
    if noises is None:
        noises = list(TUMVI_DEFAULT_NOISES)
        info(f"{seq}: IMU noises not in tar -- using TUM-VI dataset-page "
             f"values {noises}")

    calib = {"noises": noises}
    for side, cam in (("left", "cam0"), ("right", "cam1")):
        c = cams[cam]
        calib[side] = {"pinhole": c["intr"], "dist": c["dist"],
                       "q_xyzw": rot_to_quat_xyzw(c["T"][:3, :3]),
                       "p": c["T"][:3, 3].tolist()}

    def loader(path, cam):
        return load_gray(path)   # >>8 for 16-bit handled inside

    pairs = pair_stereo(list_cam_frames(mav / "cam0"),
                        list_cam_frames(mav / "cam1"), seq)
    imu = read_imu_csv(mav / "imu0" / "data.csv")

    # GT: prefer dso/gt_imu.csv (time-aligned, expressed in the IMU frame);
    # fall back to raw mocap0/data.csv.
    gt_candidates = [p for p in (seq_root / "dso" / "gt_imu.csv",
                                 mav / "mocap0" / "data.csv",
                                 mav / "gt" / "data.csv") if p.exists()]
    if not gt_candidates:
        die(f"{seq}: no ground truth found (tried dso/gt_imu.csv, "
            f"mav0/mocap0/data.csv, mav0/gt/data.csv)")
    info(f"{seq}: ground truth from {gt_candidates[0]}")
    gt_rows = read_gt_csv(gt_candidates[0])
    return calib, pairs, loader, imu, gt_rows


# ---------------------------------------------------------------------------
# MSD (Monado SLAM Datasets, Basalt calibration)
# ---------------------------------------------------------------------------

def _find_msd_calib_json(seq_root, input_root):
    """Look for <device>/extras/calibration.json near the sequence; download
    that single file from HuggingFace if it was not fetched with the zips."""
    cands = []
    d = seq_root
    for _ in range(4):                       # seq dir and a few parents
        cands.append(d / "extras" / "calibration.json")
        if d.parent == d:
            break
        d = d.parent
    if input_root.is_dir():
        cands.extend(sorted(input_root.rglob("extras/calibration.json")))
    for c in cands:
        if c.exists():
            return c

    warn("MSD calibration.json not found locally -- fetching from "
         f"HuggingFace ({MSD_HF_REPO})")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        die("huggingface_hub not installed and calibration.json not found "
            "locally; pip install huggingface_hub")
    api = HfApi()
    try:
        files = api.list_repo_files(MSD_HF_REPO, repo_type="dataset")
    except Exception as e:
        die(f"could not list {MSD_HF_REPO}: {e}")
    hits = [f for f in files if f.endswith("extras/calibration.json")]
    # MOO* sequences belong to the Odyssey+ device (MO_odyssey_plus).
    dev_hits = [f for f in hits if "odyssey" in f.lower()] or hits
    if not dev_hits:
        die(f"no extras/calibration.json in {MSD_HF_REPO}; found: {hits}")
    if len(dev_hits) > 1:
        warn(f"multiple calibration.json candidates, using first: {dev_hits}")
    local = hf_hub_download(repo_id=MSD_HF_REPO, repo_type="dataset",
                            filename=dev_hits[0],
                            local_dir=str(input_root / "_hf_extras"))
    return Path(local)


def _msd_noises(calib_dir, calib_json):
    """Noises from calibration.extra.json (preferred), then noise fields in
    calibration.json, then benchmark defaults. Basalt per-axis arrays are
    reduced with the mean."""

    def reduce(v):
        a = np.abs(np.asarray(v, dtype=float)).ravel()
        return float(np.mean(a))

    def flatten(obj, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)) and not (
                        isinstance(v, list) and v
                        and isinstance(v[0], (int, float))):
                    flatten(v, out)
                else:
                    out[k.lower()] = v
        elif isinstance(obj, list):
            for v in obj:
                flatten(v, out)

    key_map = [  # calib.txt order; candidate key names per slot
        ("gyro_noise", ["gyro_noise_std", "gyroscope_noise_density",
                        "gyro_noise_density"]),
        ("gyro_bias_rw", ["gyro_bias_std", "gyroscope_random_walk",
                          "gyro_random_walk"]),
        ("accel_noise", ["accel_noise_std", "accelerometer_noise_density",
                         "accel_noise_density"]),
        ("accel_bias_rw", ["accel_bias_std", "accelerometer_random_walk",
                           "accel_random_walk"]),
    ]
    sources = []
    extra = calib_dir / "calibration.extra.json"
    if extra.exists():
        sources.append(("calibration.extra.json",
                        json.loads(extra.read_text(encoding="utf-8"))))
    sources.append(("calibration.json", calib_json))

    for src_name, src in sources:
        flat = {}
        flatten(src, flat)
        vals, missing = [], []
        for slot, names in key_map:
            hit = next((n for n in names if n in flat), None)
            if hit is None:
                missing.append(slot)
            else:
                vals.append(reduce(flat[hit]))
        if not missing:
            info(f"MSD IMU noises from {src_name}: {vals}")
            return vals
    info(f"MSD IMU noises not found in extras -- using defaults "
         f"{MSD_DEFAULT_NOISES}")
    return list(MSD_DEFAULT_NOISES)


def prep_msd(seq_root, seq, input_root):
    mav = seq_root / "mav0"
    calib_path = _find_msd_calib_json(seq_root, Path(input_root))
    info(f"{seq}: Basalt calibration from {calib_path}")
    cj = json.loads(calib_path.read_text(encoding="utf-8"))
    v = cj.get("value0", cj)
    try:
        T_list = v["T_imu_cam"]
        intr_list = v["intrinsics"]
    except KeyError as e:
        die(f"{calib_path}: not a Basalt calibration (missing {e})")
    if len(T_list) < 2 or len(intr_list) < 2:
        die(f"{calib_path}: expected >= 2 cameras, got "
            f"{len(T_list)} extrinsics / {len(intr_list)} intrinsics")

    calib = {"noises": _msd_noises(calib_path.parent, cj)}
    # MSD/Basalt convention: cam list order == cam0, cam1 dirs; cam0 = left.
    for side, i in (("left", 0), ("right", 1)):
        it = intr_list[i]
        ctype = str(it.get("camera_type", "")).lower()
        ii = it["intrinsics"]
        if ctype == "kb4":
            model, dist = "kb4", [float(ii[k]) for k in ("k1", "k2", "k3", "k4")]
        elif ctype == "pinhole-radtan8":
            # Basalt-native RT8 passes straight through the VIT interface
            # (PinholeRadtan8Camera, 8 coeffs + rpmax) — same model the MSD
            # paper's own Basalt baseline ran.
            model = "rt8"
            dist = [float(ii[k]) for k in ("k1", "k2", "p1", "p2",
                                           "k3", "k4", "k5", "k6")]
            dist.append(float(ii.get("rpmax", 0.0)))
        else:
            die(f"{calib_path}: cam{i} camera_type={it.get('camera_type')!r}, "
                f"only kb4 / pinhole-radtan8 supported")
        if calib.get("model", model) != model:
            die(f"{calib_path}: mixed camera models between cams")
        calib["model"] = model
        Td = T_list[i]
        # Basalt T_imu_cam maps cam-frame points into the IMU frame
        # (p_imu = T_imu_cam * p_cam) => already cam->IMU, use directly.
        q = np.array([Td["qx"], Td["qy"], Td["qz"], Td["qw"]], dtype=float)
        q /= np.linalg.norm(q)
        calib[side] = {
            "pinhole": [float(ii[k]) for k in ("fx", "fy", "cx", "cy")],
            "dist": dist,
            "q_xyzw": q.tolist(),
            "p": [float(Td["px"]), float(Td["py"]), float(Td["pz"])],
        }

    def loader(path, cam):
        return load_gray(path)

    pairs = pair_stereo(list_cam_frames(mav / "cam0"),
                        list_cam_frames(mav / "cam1"), seq)
    imu = read_imu_csv(mav / "imu0" / "data.csv")
    gt_candidates = [p for p in (mav / "gt" / "data.csv",
                                 mav / "state_groundtruth_estimate0" / "data.csv",
                                 mav / "mocap0" / "data.csv",
                                 mav / "vicon0" / "data.csv") if p.exists()]
    if not gt_candidates:
        die(f"{seq}: no ground truth csv found under {mav}")
    info(f"{seq}: ground truth from {gt_candidates[0]}")
    gt_rows = read_gt_csv(gt_candidates[0])
    return calib, pairs, loader, imu, gt_rows


# ---------------------------------------------------------------------------
# pack writing
# ---------------------------------------------------------------------------

def write_pack(dataset, seq, out_dir, calib, pairs, loader, imu, gt_rows):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    check_baseline(calib, ctx=f"{dataset}/{seq}")

    imu_ts, gyro, accel = imu
    n_imu = write_imu_bin(out, imu_ts, gyro, accel)
    write_gt_tum(out, gt_rows)
    (out / "calib.txt").write_text(format_calib(calib), encoding="ascii",
                                   newline="\n")

    W = H = None
    n = len(pairs)
    raw_path = out / "frames.raw"
    with open(raw_path, "wb") as raw:
        for i, (ts, lpath, rpath) in enumerate(pairs):
            left = loader(lpath, "cam0")
            right = loader(rpath, "cam1")
            if W is None:
                H, W = left.shape
                if H < 2 or W < 2:
                    die(f"{seq}: implausible frame size {W}x{H}")
            for name, img in (("left", left), ("right", right)):
                if img.shape != (H, W):
                    die(f"{seq}: {name} frame {i} is "
                        f"{img.shape[1]}x{img.shape[0]}, expected {W}x{H}")
                raw.write(img.tobytes())
            if (i + 1) % 200 == 0 or i + 1 == n:
                info(f"  frames {i + 1}/{n}")
    write_frames_csv(out, [ts for ts, _, _ in pairs])

    meta = {
        "dataset": dataset, "seq": seq, "W": W, "H": H,
        "fps": round(median_rate_hz([ts for ts, _, _ in pairs]), 3),
        "imu_hz": round(median_rate_hz(imu_ts), 3),
        "compressed": 0, "n_frames": n,
    }
    write_meta(out, meta)   # written last: acts as pack-complete marker
    info(f"{dataset}/{seq}: pack complete -> {out}")
    info(f"  {n} stereo pairs @ {meta['fps']:.2f} fps, {W}x{H}, "
         f"{n_imu} imu records @ {meta['imu_hz']:.1f} Hz, "
         f"frames.raw {raw_path.stat().st_size / 1e6:.1f} MB")
    return meta


def prep_one(dataset, seq_root, out_dir, input_root=None):
    seq_root = Path(seq_root)
    seq = seq_root.name
    if dataset == "euroc":
        parts = prep_euroc(seq_root, seq)
    elif dataset == "tumvi":
        parts = prep_tumvi(seq_root, seq)
    elif dataset == "msd":
        parts = prep_msd(seq_root, seq, input_root or seq_root.parent)
    else:
        die(f"unknown dataset {dataset!r} (want one of {DATASETS})")
    return write_pack(dataset, seq, out_dir, *parts)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=DATASETS)
    ap.add_argument("input", help="archive (.zip/.tar) or extracted directory")
    ap.add_argument("out_pack_dir")
    ap.add_argument("--seq", default=None,
                    help="sequence name if the input contains several")
    args = ap.parse_args()

    inp = Path(args.input)
    seqs = discover_sequences(inp)
    if args.seq:
        seqs = [(n, r) for n, r in seqs if n == args.seq]
        if not seqs:
            die(f"--seq {args.seq!r} not found; available: "
                f"{[n for n, _ in discover_sequences(inp)]}")
    if len(seqs) > 1:
        die("input contains multiple sequences: "
            f"{[n for n, _ in seqs]} -- pick one with --seq NAME "
            f"(or use run_all.py to convert them all)")
    name, root = seqs[0]
    root_for_extras = inp if inp.is_dir() else inp.parent
    prep_one(args.dataset, root, args.out_pack_dir, input_root=root_for_extras)


if __name__ == "__main__":
    main()
