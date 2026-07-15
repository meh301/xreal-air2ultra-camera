# SLAM benchmark harness (`benchmark` branch)

Benchmarks OUR stack — Basalt VIO (VIT interface) + the xr_map session
loop-closure/healing layer — against published systems on EuRoC, TUM-VI room,
and the Monado SLAM Dataset (MSD MOO group), under the causal protocol of
VOCA (arXiv 2607.00189) / MSD (arXiv 2508.00088): causal per-frame poses,
SE3-Umeyama ATE, RTE Δ=6 frames, their divergence thresholds, optional
500 kbps x264 input degradation.

## Architecture

```
host (Windows, python+ffmpeg)          runner (either)
  bench/host/prep_dataset.py             A) remote Linux container:
    dataset -> replay pack                  bench/container/build_basalt_linux.sh
  bench/host/compress_voca.py              bench/replay/Makefile.linux
    pack -> 500kbps pack                  B) on-device (arm64, adb):
  bench/host/score.py                       bench/build_replay.ps1
    est.tum vs gt.tum -> ATE/RTE            push libs + xr_replay, run in
  bench/host/run_all.py                     /data/local/tmp like qnn-net-run
    orchestration + results table
```

One `xr_replay` run emits BOTH trajectories (`*_vio.tum` = raw Basalt
odometry, `*_map.tum` = with xr_map loop-closure corrections) because the
map layer has no feedback into VIO.

## Replay binary

Per-dataset resolutions are compile-time (`-DXR_OW/-DXR_OH`):
euroc 752x480 · tumvi 512x512 · msdmoo 640x480 · device 480x640.

```
xr_replay --pack <dir> --out <prefix> [--toml basalt.toml] [--inflight N]
          [--fast] [--no-map] [--xfeat model.onnx]
```

Feed pacing: `--inflight` (default 6) blocks the feeder when more than N
pushed pairs have no pose yet — approximates blocking-queue flow control on
the app's drop-under-backpressure libbasalt. The bench libbasalt (stock
blocking queues, `bench/container/patch_basalt_bench.py` applies only the
cam-calib-optional patch) gives natural flow control instead; `--fast`
disables observation pacing when using it.

## Basalt variants

- app lib (jniLibs): our patched build — drops under backpressure (realtime).
- bench lib: same lineage, blocking queues — deterministic completeness,
  faithful to the Basalt baselines in the MSD/VOCA papers.

## Comparability notes (state these when reporting)

- MSD Table IV = uncompressed causal VIO (IMU on, LC off) — our VIO-only
  column compares cleanly.
- VOCA tables = 500 kbps causal stereo VO, NO IMU, LC disabled — our numbers
  under their input degradation, not a same-class comparison.
- OKVIS2's causal+LC results are the only published causal loop-closure
  opponent (their alignment is 4-DoF yaw, slightly stricter than SE3).
- EuRoC images are host-remapped radtan->equidistant (kb4) for VIT (footnote
  in reports); TUM-VI/MSD kb4 passes through natively.
- Warmup: the first ~300 IMU samples of each sequence gate frame feeding
  (Basalt gravity init) — a few skipped frames at 200 Hz IMU datasets,
  counted in the completion stat.
```