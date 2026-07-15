"""Shared helpers for the SLAM benchmark replay-pack tooling.

Replay pack layout (frozen spec -- keep in sync with the C replay harness):

    <pack_dir>/
      meta.txt      key=value: dataset seq W H fps imu_hz compressed n_frames
      calib.txt     kb4 calibration, text (see format_calib below)
      imu.bin       32-byte LE records: int64 ts_ns, f32 gx gy gz (deg/s),
                    f32 ax ay az (units of g = 9.80665)
      frames.csv    header-less "ts_ns,idx" lines (idx = frame pair index)
      frames.raw    L8 gray pairs: [left W*H][right W*H] per frames.csv line
      gt.tum        ground truth "t_sec tx ty tz qx qy qz qw" (scoring only)
"""

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

G_ACCEL = 9.80665
RAD2DEG = 180.0 / np.pi

# int64 ts + 3x f32 gyro + 3x f32 accel = 8 + 12 + 12 = 32 bytes, little-endian.
IMU_DTYPE = np.dtype([("ts_ns", "<i8"),
                      ("gyro_dps", "<f4", (3,)),
                      ("accel_g", "<f4", (3,))])
assert IMU_DTYPE.itemsize == 32, "imu.bin record must be exactly 32 bytes"

DATASETS = ("euroc", "tumvi", "msd")

# Nominal camera rates used by compress_voca.py and score.py completion math
# (per benchmark convention: euroc/tumvi 20 fps, MSD MOO 30 fps).
DATASET_FPS = {"euroc": 20.0, "tumvi": 20.0, "msd": 30.0}

# Divergence thresholds (score.py): ATE metres, RTE centimetres.
ATE_DIVERGE_M = {"euroc": 10.0, "tumvi": 10.0, "msd": 1.0}
RTE_DIVERGE_CM = 10.0


def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def info(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# meta.txt
# ---------------------------------------------------------------------------

_META_TYPES = {"dataset": str, "seq": str, "W": int, "H": int,
               "fps": float, "imu_hz": float, "compressed": int,
               "n_frames": int}


def write_meta(pack_dir, meta):
    keys = ["dataset", "seq", "W", "H", "fps", "imu_hz", "compressed", "n_frames"]
    missing = [k for k in keys if k not in meta]
    if missing:
        die(f"internal: meta missing keys {missing}")
    lines = [f"{k}={meta[k]}" for k in keys]
    (Path(pack_dir) / "meta.txt").write_text("\n".join(lines) + "\n",
                                             encoding="ascii")


def read_meta(pack_dir):
    p = Path(pack_dir) / "meta.txt"
    if not p.exists():
        die(f"not a replay pack (missing meta.txt): {pack_dir}")
    meta = {}
    for ln in p.read_text(encoding="ascii").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        if "=" not in ln:
            die(f"malformed meta.txt line in {p}: {ln!r}")
        k, v = ln.split("=", 1)
        k, v = k.strip(), v.strip()
        typ = _META_TYPES.get(k, str)
        try:
            meta[k] = typ(v)
        except ValueError:
            die(f"malformed meta.txt value in {p}: {ln!r}")
    missing = [k for k in _META_TYPES if k not in meta]
    if missing:
        die(f"meta.txt in {pack_dir} missing keys: {missing}")
    if meta["dataset"] not in DATASETS:
        die(f"meta.txt dataset={meta['dataset']!r} not one of {DATASETS}")
    return meta


# ---------------------------------------------------------------------------
# calib.txt
# ---------------------------------------------------------------------------

def _fmt(vals):
    return " ".join(f"{float(v):.12g}" for v in vals)


def format_calib(calib):
    """calib = {'left': {'pinhole':[fx,fy,cx,cy], 'dist':[k1..k4],
                         'q_xyzw':[qx,qy,qz,qw], 'p':[x,y,z]},
                'right': {...}, 'noises': [gn, gbrw, an, abrw]}
    Rotation q/p are cam->IMU: p_imu = R(q) * p_cam + p.
    Noise units (SI): rad/s/sqrt(Hz), rad/s^2/sqrt(Hz),
                      m/s^2/sqrt(Hz), m/s^3/sqrt(Hz)."""
    lines = ["model kb4"]
    for side in ("left", "right"):
        c = calib[side]
        if len(c["pinhole"]) != 4 or len(c["dist"]) != 4:
            die(f"internal: calib {side} needs 4 pinhole + 4 dist values")
        q = np.asarray(c["q_xyzw"], dtype=float)
        if abs(np.linalg.norm(q) - 1.0) > 1e-6:
            die(f"internal: calib {side} quaternion not normalized: {q}")
        lines.append(f"{side}_pinhole {_fmt(c['pinhole'])}")
        lines.append(f"{side}_dist {_fmt(c['dist'])}")
        lines.append(f"{side}_q_xyzw {_fmt(q)}")
        lines.append(f"{side}_p {_fmt(c['p'])}")
    if len(calib["noises"]) != 4:
        die("internal: calib noises must be [gyro_n, gyro_brw, accel_n, accel_brw]")
    lines.append(f"noises {_fmt(calib['noises'])}")
    return "\n".join(lines) + "\n"


def check_baseline(calib, ctx=""):
    """Sanity: stereo baseline should be ~2-30 cm for these headset/MAV rigs."""
    b = float(np.linalg.norm(np.asarray(calib["left"]["p"]) -
                             np.asarray(calib["right"]["p"])))
    if not (0.02 <= b <= 0.30):
        warn(f"{ctx}: stereo baseline {b*100:.2f} cm outside sane range "
             f"(2..30 cm) -- check extrinsics convention!")
    else:
        info(f"{ctx}: stereo baseline {b*100:.2f} cm (sane)")
    return b


# ---------------------------------------------------------------------------
# SO3 / SE3 helpers
# ---------------------------------------------------------------------------

def rot_to_quat_xyzw(R):
    """Rotation matrix -> quaternion [qx,qy,qz,qw], robust (Shepperd)."""
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or abs(np.linalg.det(R) - 1.0) > 1e-4:
        die(f"internal: not a rotation matrix (det={np.linalg.det(R):.6f})")
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=float)
    return q / np.linalg.norm(q)


def quat_xyzw_to_rot(q):
    qx, qy, qz, qw = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def se3_inverse(T):
    """Invert a 4x4 homogeneous transform."""
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        die(f"internal: SE3 must be 4x4, got {T.shape}")
    Rinv = T[:3, :3].T
    out = np.eye(4)
    out[:3, :3] = Rinv
    out[:3, 3] = -Rinv @ T[:3, 3]
    return out


# ---------------------------------------------------------------------------
# frames.csv / frames.raw / imu.bin / gt.tum I/O
# ---------------------------------------------------------------------------

def write_frames_csv(pack_dir, ts_list):
    with open(Path(pack_dir) / "frames.csv", "w", encoding="ascii",
              newline="\n") as f:
        for idx, ts in enumerate(ts_list):
            f.write(f"{int(ts)},{idx}\n")


def read_frames_csv(pack_dir):
    p = Path(pack_dir) / "frames.csv"
    if not p.exists():
        die(f"missing frames.csv in {pack_dir}")
    out = []
    for ln in p.read_text(encoding="ascii").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) != 2:
            die(f"malformed frames.csv line in {p}: {ln!r}")
        out.append((int(parts[0]), int(parts[1])))
    return out


def open_frames_raw(pack_dir, meta, mode="r"):
    """Memory-map frames.raw as (n_frames, 2, H, W) uint8 (2 = left,right)."""
    p = Path(pack_dir) / "frames.raw"
    n, H, W = meta["n_frames"], meta["H"], meta["W"]
    expect = n * 2 * H * W
    if mode == "r":
        if not p.exists():
            die(f"missing frames.raw in {pack_dir}")
        actual = p.stat().st_size
        if actual != expect:
            die(f"frames.raw size mismatch in {pack_dir}: "
                f"{actual} bytes, expected {expect} (= {n}*2*{H}*{W})")
    return np.memmap(p, dtype=np.uint8, mode=mode, shape=(n, 2, H, W))


def write_imu_bin(pack_dir, ts_ns, gyro_rad_s, accel_m_s2):
    """ts_ns int64 array; gyro rad/s, accel m/s^2 -> converts to deg/s and g."""
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    n = len(ts_ns)
    rec = np.empty(n, dtype=IMU_DTYPE)
    rec["ts_ns"] = ts_ns
    rec["gyro_dps"] = (np.asarray(gyro_rad_s, dtype=np.float64)
                       * RAD2DEG).astype(np.float32)
    rec["accel_g"] = (np.asarray(accel_m_s2, dtype=np.float64)
                      / G_ACCEL).astype(np.float32)
    rec.tofile(str(Path(pack_dir) / "imu.bin"))
    return n


def read_imu_bin(pack_dir):
    p = Path(pack_dir) / "imu.bin"
    if not p.exists():
        die(f"missing imu.bin in {pack_dir}")
    if p.stat().st_size % IMU_DTYPE.itemsize != 0:
        die(f"imu.bin size {p.stat().st_size} not a multiple of 32 in {pack_dir}")
    return np.fromfile(str(p), dtype=IMU_DTYPE)


def write_gt_tum(pack_dir, rows):
    """rows: iterable of (ts_ns:int, p:(3,), q_xyzw:(4,))."""
    n = 0
    with open(Path(pack_dir) / "gt.tum", "w", encoding="ascii",
              newline="\n") as f:
        for ts_ns, p, q in rows:
            f.write(f"{ts_ns * 1e-9:.9f} "
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
            n += 1
    if n == 0:
        die(f"no ground-truth rows written for {pack_dir}")
    return n


def median_rate_hz(ts_ns):
    ts = np.asarray(ts_ns, dtype=np.int64)
    if len(ts) < 2:
        die("cannot compute rate from <2 timestamps")
    dt = np.diff(ts)
    med = float(np.median(dt))
    if med <= 0:
        die("non-increasing timestamps while computing rate")
    return 1e9 / med


# ---------------------------------------------------------------------------
# Archive extraction (idempotent)
# ---------------------------------------------------------------------------

def _looks_like_html(path):
    with open(path, "rb") as f:
        head = f.read(512).lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def find_7z():
    for cand in ("7z", "7za"):
        w = shutil.which(cand)
        if w:
            return w
    for cand in (r"C:\Program Files\7-Zip\7z.exe",
                 r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(cand).exists():
            return cand
    return None


def _extract_split_zip(zip_path, target):
    """PKWARE split archive: <stem>.z01, .z02, ..., <stem>.zip (last part)."""
    parts = sorted(zip_path.parent.glob(zip_path.stem + ".z[0-9][0-9]"),
                   key=lambda p: int(p.suffix[2:]))
    if not parts:
        die(f"internal: {zip_path} flagged as split but no .zNN parts found")
    for p in parts + [zip_path]:
        if _looks_like_html(p):
            die(f"{p} is an HTML page, not an archive part -- the download "
                f"failed; re-download it")
    seven = find_7z()
    if seven:
        info(f"  joining split zip via 7z: {zip_path.name} "
             f"(+{len(parts)} parts)")
        r = subprocess.run([seven, "x", "-y", f"-o{target}", str(zip_path)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return
        warn(f"7z failed on {zip_path} (exit {r.returncode}) -- trying "
             f"pure-python concat join")
    else:
        warn("7z not found -- trying pure-python split-zip join")
    # Fallback: concatenate parts in order (z01..zNN then .zip). zipfile finds
    # the central directory from the end, so the joined stream usually reads
    # fine despite the 4-byte spanning marker at the start of part 1.
    joined = target.parent / (zip_path.stem + ".joined.zip.tmp")
    try:
        with open(joined, "wb") as out:
            for p in parts + [zip_path]:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 1 << 22)
        if not zipfile.is_zipfile(joined):
            die(f"joined split zip is unreadable: install 7-Zip and retry "
                f"({zip_path})")
        with zipfile.ZipFile(joined) as z:
            z.extractall(target)
    finally:
        joined.unlink(missing_ok=True)


def ensure_extracted(archive_or_dir):
    """Extract an archive next to itself (idempotent via .extract_done marker).
    If given a directory, returns it unchanged."""
    p = Path(archive_or_dir)
    if p.is_dir():
        return p
    if not p.exists():
        die(f"input not found: {p}")
    if p.suffix.lower() in (".z01", ".z02", ".z03"):
        die(f"pass the final .zip of the split set, not {p.name}")
    stem = p.stem  # 'foo.zip' -> 'foo', 'foo.tar' -> 'foo'
    target = p.parent / stem
    marker = target / ".extract_done"
    if marker.exists():
        info(f"  already extracted: {target}")
        return target
    if _looks_like_html(p):
        die(f"{p} is an HTML page, not an archive (the download failed -- "
            f"the server returned an error/landing page; re-download it)")
    if target.exists():
        warn(f"{target} exists without completion marker -- re-extracting")
    target.mkdir(parents=True, exist_ok=True)
    is_split = (p.suffix.lower() == ".zip"
                and (p.parent / (stem + ".z01")).exists())
    with open(p, "rb") as f:
        magic = f.read(4)
    info(f"  extracting {p.name} -> {target} ...")
    try:
        if is_split:
            _extract_split_zip(p, target)
        elif zipfile.is_zipfile(p):
            with zipfile.ZipFile(p) as z:
                z.extractall(target)
        elif tarfile.is_tarfile(p):
            with tarfile.open(p) as t:
                try:
                    t.extractall(target, filter="data")  # py>=3.12
                except TypeError:
                    t.extractall(target)
        elif magic.startswith(b"PK"):
            die(f"{p} starts like a zip but has no readable central "
                f"directory -- truncated / still-downloading file?")
        else:
            die(f"unrecognized archive format: {p}")
    except (zipfile.BadZipFile, tarfile.TarError, EOFError, OSError) as e:
        die(f"extraction of {p} failed ({e}) -- archive truncated or "
            f"corrupt? re-download it")
    marker.write_text("")
    return target


def run_cmd(cmd, cwd=None, what="command"):
    """Run a subprocess, dying with the tail of its output on failure."""
    r = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout or "").splitlines()[-30:])
        die(f"{what} failed (exit {r.returncode}):\n"
            f"  {' '.join(str(c) for c in cmd)}\n{tail}")
    return r
