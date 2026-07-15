# bench/host -- host-side SLAM benchmark tooling

Python tooling that turns EuRoC / TUM-VI / MSD sequences into **replay packs**
for the C replay harness, optionally produces the VOCA-compressed variant,
and scores estimated trajectories against ground truth.

## Environment

* Python: `F:\xreal_depth\depthenv\Scripts\python.exe` (3.10.1)
  * required: numpy, opencv, PyYAML (all present), **evo 1.36.5**
    (installed via `pip install evo` on 2026-07-15), huggingface_hub
    (only needed for the MSD calibration.json fallback download)
* ffmpeg with libx264 on PATH (verified: 6.0 gyan.dev build)
* 7-Zip (`7z` on PATH or default install dir) is preferred for the multipart
  MSD MOO12 archive (`.zip` + `.z01` + `.z02`); if absent, a pure-python
  concat-join fallback is attempted, and a clear error is raised if that
  fails. `py7zr` is NOT used -- split ZIPs are PKWARE format, not 7z volumes.

## Scripts

```
prep_dataset.py  euroc|tumvi|msd <archive_or_dir> <out_pack_dir> [--seq NAME]
compress_voca.py <pack_dir> <out_pack_dir> [--bitrate 500k] [--keep-temp] [--force]
score.py         <est.tum> <gt.tum> --dataset euroc|tumvi|msd [--json out.json]
                 [--fps F] [--t-max-diff 0.01]
run_all.py       [--root F:\slam_bench] [--packs DIR] [--datasets euroc,tumvi,msd]
                 [--compress] [--bitrate 500k] [--only SUBSTR]
                 [--replay-cmd "TEMPLATE {pack} {est}"] [--results results.md]
```

All scripts import `pack_common.py` (must stay in the same directory).

## Replay pack format (frozen spec)

```
<pack_dir>/
  meta.txt      key=value: dataset=euroc|tumvi|msd seq=<name> W= H= fps=
                imu_hz= compressed=0|1 n_frames=
  calib.txt     model kb4
                left_pinhole fx fy cx cy
                left_dist k1 k2 k3 k4
                left_q_xyzw qx qy qz qw     # rotation cam->IMU (R_imu_cam)
                left_p x y z                # camera origin in IMU frame [m]
                right_* (same four lines)
                noises gyro_n gyro_bias_rw accel_n accel_bias_rw   # SI
  imu.bin       32-byte LE records: int64 ts_ns, f32 gx gy gz (DEG/S),
                f32 ax ay az (units of g = 9.80665)
  frames.csv    header-less "ts_ns,idx"; idx = pair index;
                left offset = idx*2*W*H, right = +W*H
  frames.raw    concatenated L8 gray pairs in timestamp order
  gt.tum        "t_sec tx ty tz qx qy qz qw" (scoring only)
```

`meta.txt` is written **last**, so its presence marks a complete pack
(run_all.py uses this for idempotent skipping).

## Convention decisions (documented per spec)

### Extrinsics: everything is normalized to cam->IMU
`left_q_xyzw` / `left_p` satisfy `p_imu = R(q) * p_cam + p`.

* **EuRoC** `mav0/camX/sensor.yaml` `T_BS` maps sensor(cam) points to
  body(=IMU) frame, i.e. already cam->IMU: **used directly**.
* **TUM-VI** Kalibr camchain (`dso/camchain.yaml` inside the tar) stores
  `T_cam_imu`, which maps IMU-frame points into the camera frame (IMU->cam):
  **inverted** here (`R = R_cam_imu^T`, `p = -R_cam_imu^T t`). If cam1 only
  had `T_cn_cnm1`, it is composed with cam0 (`T_cam1_imu = T_cn_cnm1 *
  T_cam0_imu`) before inverting. Verified against the real room1 camchain:
  the resulting baseline is ~10.1 cm, matching `T_cn_cnm1`'s -0.101 m x-offset.
  If no camchain is found, `mav0/camX/sensor.yaml` `T_BS` is used as a
  fallback (with a warning).
* **MSD** Basalt `calibration.json` stores `T_imu_cam`, which in
  Basalt/Sophus convention maps camera-frame points into the IMU frame
  (`p_imu = T_imu_cam * p_cam`), i.e. already cam->IMU: **used directly**
  (settled per the Basalt convention and validated by the baseline check).
* Every conversion runs a **baseline sanity check**: it warns loudly if
  `|left_p - right_p|` falls outside 2..30 cm.

### Cameras
* Left = `cam0`, right = `cam1` for **all three** datasets. For MSD this
  follows the Basalt/EuRoC-ASL ordering used by the Monado SLAM Datasets
  (cam0 is the first entry of `T_imu_cam`/`intrinsics` in calibration.json).
* **EuRoC images are RESAMPLED** (footnote for any report): EuRoC cameras are
  pinhole+radtan4, which kb4 cannot express. Each image is remapped once
  (cv2.remap, bilinear) into a **virtual equidistant/kb4 camera** with the
  same fx,fy,cx,cy and `k1..k4 = 0` (pure `r = f*theta`). The map is built
  per camera by unprojecting each virtual pixel (theta = r_norm), projecting
  the ray through the radtan model, rays beyond 88 deg marked invalid
  (borders read as black). TUM-VI and MSD are kb4-native and passed through
  untouched (`distortion_coeffs`/`k1..k4` copied verbatim).
* TUM-VI 16-bit PNGs are converted to 8-bit via `>> 8`.

### IMU
* Source csv (`ts_ns, w[rad/s] x3, a[m/s^2] x3`) is converted to **deg/s**
  and **g** for imu.bin (the C harness expects those units).
* Noise densities (`noises` line, SI units as in the spec):
  * EuRoC: from `mav0/imu0/sensor.yaml`.
  * TUM-VI: from `mav0/imu0/sensor.yaml` if present, else
    `dso/imu_config.yaml` inside the tar (present in the 512_16 tars; these
    are the "inflated" Allan values: gyro 0.00016 / 0.000022,
    accel 0.0028 / 0.00086), else those same dataset-page values hardcoded.
  * MSD: from `extras/calibration.extra.json` if present, else noise fields
    (`gyro_noise_std`, `gyro_bias_std`, `accel_noise_std`, `accel_bias_std`,
    per-axis arrays reduced by mean) in `calibration.json`, else the
    documented defaults `{0.00035, 0.0001, 0.00667, 0.001}`. The chosen
    source is printed during prep.

### Ground truth
* All GT is written as TUM (`t_sec tx ty tz qx qy qz qw`); source csvs are
  EuRoC-style `ts, p, q_wxyz` -- note the **wxyz -> xyzw** reorder and
  ns -> s conversion. Quaternion norms are checked to catch column-order
  mistakes.
* EuRoC: `mav0/state_groundtruth_estimate0/data.csv`.
* TUM-VI: prefers `dso/gt_imu.csv` (time-aligned, expressed in the IMU
  frame; format `# timestamp[ns],tx,ty,tz,qw,qx,qy,qz`, verified against the
  real room1 tar) over raw `mav0/mocap0/data.csv`.
* MSD: `mav0/gt/data.csv` (falls back to state_groundtruth_estimate0 /
  mocap0 / vicon0 if absent).

### MSD calibration discovery
`<device>/extras/calibration.json` is searched: next to the sequence, up to
3 parent levels, then recursively under the input root. If the HF snapshot
was filtered to `*MOO*` (which excludes extras/), the single file is
downloaded via `huggingface_hub.hf_hub_download` from
`collabora/monado-slam-datasets` (the `extras/calibration.json` whose repo
path contains "odyssey", cached under `<msd_root>\_hf_extras`).

## compress_voca.py

Reconstructs per-camera PNG dirs from frames.raw and runs the exact VOCA
two-pass x264 per camera:

```
pass1: ffmpeg -y -f concat -safe 0 -r <FPS> -i <list> -c:v libx264 -b:v <BITRATE>
       -pix_fmt yuv420p -r <FPS> -vf setpts=PTS-STARTPTS
       -x264opts partitions=p8x8,p4x4,i8x8:keyint=1000:me=umh:merange=64:subme=6:bframes=0:ref=1
       -passlogfile <log> -pass 1 -an -f null NUL
pass2: same -> -pass 2 -an -f mp4 <cam>.mp4
```

then decodes back (`ffmpeg -i out.mp4 -f rawvideo -pix_fmt gray`) and writes
a new pack with `compressed=1`. Decoded frame count is verified against
n_frames (hard error on mismatch). Encoder FPS is the dataset nominal rate
(euroc/tumvi **20**, msd **30**), not the measured fps. Pass-1 output goes to
`NUL` (Windows). The mp4s are kept in `<out>/_voca_tmp/` for reference; PNG
and rawvideo temps are deleted (use `--keep-temp` to keep everything).
imu/calib/gt/frames.csv are copied verbatim.

## score.py

Uses **evo 1.36.5 as a library** (equivalent CLI forms in parentheses):

* **ATE** = APE translation RMSE [m] after Umeyama **SE(3) alignment, no
  scale** (`evo_ape tum gt est -a`).
* **RTE** = RPE translation RMSE, `--delta 6 --delta_unit f`, consecutive
  pairs (**without** `--all_pairs`), reported in **cm** (m x 100).
* Association: nearest timestamps, `--t-max-diff 0.01 s` default.
* **Divergence**: ATE > 10 m (**1 m for msd**) or RTE > 10 cm marks the run
  diverged and both metrics become `inf` (raw values preserved in the JSON
  as `ate_raw_m`/`rte_raw_cm`). An empty/missing-pose estimate or fewer than
  8 matched poses also counts as diverged.
* **completion%** = est poses within the GT time span (half-frame slack) /
  expected frame count, where expected = `gt_span_sec * fps + 1` and fps is
  the dataset nominal rate (euroc/tumvi 20, msd 30; override with `--fps`),
  capped at 100.
* Exit code 0 for any computed score (diverged included); nonzero only for
  operational errors.

## run_all.py

Orchestrator: discovers archives/extracted sequences under
`<root>/{euroc,tumvi,msd}`, preps each sequence into `<packs>/<dataset>/<seq>`
(skipped when `meta.txt` exists), optionally builds `<seq>_voca`, obtains an
estimate, scores it, and writes a markdown results table.

**The C replay harness is not wired yet** -- see the `run_replay()` TODO
(local-container vs adb execution sketched there). Until then either:

* pass `--replay-cmd "replay.exe {pack} {est}"` -- a shell template run per
  sequence with `{pack}`/`{est}` substituted (put quotes around the
  placeholders if your paths contain spaces), or
* run the harness manually and drop `est.tum` into each pack dir; re-running
  run_all.py picks existing est.tum files up and scores them.

Sequences without an estimate are listed as `pending` and excluded from the
aggregates. Aggregation follows VOCA conventions: **SR** = non-diverged /
attempted; **AVG** over non-diverged sequences only (exclude any-failed);
**MED** over all attempted sequences with failures counted as `inf`.

## Notes / gotchas

* Archives are extracted **next to themselves** (idempotent via an
  `.extract_done` marker inside the extraction dir).
* The tooling detects HTML-error-page downloads masquerading as archives and
  truncated (still-downloading) zips, and refuses them with a clear message
  -- the initial EuRoC downloads on F: hit exactly this (the ETH
  research-collection URLs returned the SPA landing page).
* frames.raw for a full sequence is large (e.g. TUM-VI room: ~1.5 GB); packs
  are streamed frame-by-frame, memory use stays flat.
* Kalibr `timeshift_cam_imu` (TUM-VI camchain) is ignored: cam and IMU
  timestamps in the euroc-format tars are already on a common clock.
* EuRoC `machine_hall.zip`-style multi-sequence archives are supported:
  `prep_dataset.py` lists the sequences and asks for `--seq`; `run_all.py`
  iterates all of them automatically.
