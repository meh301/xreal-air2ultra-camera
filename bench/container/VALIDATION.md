# STAGE-3 landmark-factor coupling (`XR_LMFACT`) — container build + smoke

Stage 3 of map→VIO coupling: after the session map verifies a closure /
relocalization, the PnP **inlier 2D-3D pairs** (query pixel + matched map
landmark 3D) are fed into Basalt's estimator as **reprojection factors with
FIXED 3D** — unary factors on the observing frame's pose, no landmark state
added. This is the OKVIS2 mechanism: per-point arbitration inside the
optimizer, safe at ANY revisit age (the pose-prior channel's 30 s
revisit-age gate excludes exactly the room-scale closures this recovers).

Everything is flag-gated: with `XR_LMFACT` unset (or a stock libbasalt
without the symbol) behavior is bit-identical to before.

## What changed

App repo (branch `benchmark`):
- `android/app/src/main/cpp/xr_slam.c/.h` — `xr_slam_landmark_factors(...)`,
  dlsym-gated on `vit_tracker_xreal_landmark_factors` (returns -1 when the
  backend lacks it, 0 when the tracker is down).
- `android/app/src/main/cpp/xr_map.c` — `XR_LMFACT` flag; `reloc_pnp` gained
  optional out-params emitting up to 32 geometric-inlier (uv, session-xyz)
  pairs (existing callers pass NULL); `lmfact_post()` transforms the pairs
  session→odom via CORR⁻¹ and posts them at the two `tight_post_prior` call
  sites (the gated sub-threshold branch — with NO revisit-age gate — and the
  confirmed-closure TIGHT branch).
- `bench/container/patch_lmfact.py` — idempotent patcher for the container's
  basalt clone (subsumes the pose-prior patch_tight hunks: applies them first
  when missing, then the stage-3 hunks; safe to run any number of times).

Basalt (nested repo `android/app/src/main/cpp/third_party/basalt`, branch
`xr-lmfact` — merge into its `main` before an Android build):
- `include/basalt/vi_estimator/vio_estimator.h` — `setXrLandmarkFactors`
  virtual no-op.
- `include/basalt/vi_estimator/sqrt_keypoint_vio.h` — override + thread-safe
  buffer (≤ 8 frames × ≤ 32 factors, keyed by frame t_ns; re-post for the
  same frame+cam replaces).
- `src/vi_estimator/sqrt_keypoint_vio.cpp` — application in `optimize()`
  right after the pose-prior block (post-`get_dense_H_b`): for each buffered
  batch whose t_ns is in the window, add H += JᵀWJ, b += JᵀWr on that frame's
  pose block. Residual r = proj(T_i_c⁻¹ T_w_i⁻¹ p_w) − uv; J = J_proj(2×3) ·
  R_cw · [−I | hat(p_w − t_wi)] under the decoupled increment convention
  (fixed 3D ⇒ no landmark Jacobian block). Huber at 3σ, hard cutoff at 20σ,
  W = 1/σ_px². Batches expire 0.5 s behind the newest state.
- `src/vit/vit_tracker.cpp` / `include/basalt/vit/vit_tracker.hpp` —
  `extern "C" vit_tracker_xreal_landmark_factors` export.

Verified offline (no compiler on the Windows box): NDK-clang
`-fsyntax-only` passes on `xr_map.c`/`xr_slam.c`; the estimator block
typechecks against the real basalt-headers (GenericCamera/Sophus/Eigen,
float and double); the analytic Jacobian matches central differences to
5e-10 and a simulated GN step with basalt's `inc = -solve(H,b)` convention
converges (see the derivation comment in the injected block).

## Build (container)

```bash
cd /root/xreal                    # repo root, branch benchmark
python3 bench/container/patch_lmfact.py bench/container/basalt-linux
bash bench/container/build_basalt_linux.sh
```

`patch_lmfact.py` is idempotent and ordering-safe w.r.t. patch_tight.py /
patch_basalt_bench.py. The build script re-runs patch_basalt_bench and
rebuilds `libbasalt.so` + the three replay binaries (the replay picks up the
new `xr_map.c`/`xr_slam.c` automatically).

Symbol check (proves the export made it in):

```bash
nm -D bench/container/lib/libbasalt.so | grep xreal_landmark
# expect: T vit_tracker_xreal_landmark_factors
```

## Smoke run

```bash
cd bench/replay
XR_TIGHT=1 XR_TIGHTSUB=1 XR_LMFACT=1 \
LD_LIBRARY_PATH=/root/xreal/bench/container/lib \
./xr_replay_tumvi --pack <pack-dir> --out /tmp/lmfact_smoke \
    [--xfeat <xfeat.onnx> --lglue <lglue.onnx> --vpr <eigenplaces.onnx>]
```

`XR_LMFACT=1` alone is also a valid arm (factors post only from the
sub-gate branch; confirmed closures keep the classic snap+deform).
`XR_TIGHT=1 XR_LMFACT=1` is the intended stage-3 configuration.

Log lines that prove the chain end to end (stderr / stdout):

1. `session map: LANDMARK-FACTOR map->VIO coupling ON`
   — flag parsed (xr_map.c).
2. `Basalt VIT backend loaded, interface X.Y.Z (+xreal pose prior) (+lm factors)`
   — the dlsym found the new symbol (xr_slam.c).
3. `session map: LMFACT posted N landmark factors`
   — a verified pass emitted inlier pairs and the post succeeded (xr_map.c,
   fires per posting).
4. `[xr] LMFACT active: A/N fixed-3D landmark factors applied to frame T`
   — the estimator actually folded factors into the dense H,b (basalt
   stderr, printed ONCE on first application).

If (3) appears without (4): the frame left the optimization window before
the next `optimize()` (or every residual exceeded the 20σ cutoff) —
check timestamp domains first.

## A/B

Same pack, same seed, three arms:

```bash
for arm in "XR_TIGHT=1 XR_TIGHTSUB=1" "XR_TIGHT=1 XR_TIGHTSUB=1 XR_LMFACT=1" "XR_LMFACT=1"; do
  env $arm LD_LIBRARY_PATH=... ./xr_replay_tumvi --pack ... --out /tmp/ab_$RANDOM
done
```

Score with `bench/host/score.py` as usual; the hunt is the TUM-long
corridors 1–3 and the rooms/EuRoC/MSD gap where OKVIS2+LC absorbs sub-gate
corrections continuously (medians vs medians).

## Known risks / not verified without a full build

- The container clone (mateosss/basalt) is a different lineage from the
  nested repo; the stage-3 hunks anchor on text patch_tight already proved
  present there, and `Vec2/Vec4/frame_poses.getPose()/calib.intrinsics` all
  exist in that lineage's `sqrt_keypoint_vio` — but the container build is
  the first real compile of the merged file.
- Timestamp domain: factors are keyed by the map query frame's `work.ts`
  and matched EXACTLY against estimator state timestamps (same source as
  the popped poses, so they should match; the pose-prior channel only ever
  compared inequalities). If cam_time_offset handling ever changes, the
  exact-match lookup is the first thing to break — log line (4) missing is
  the symptom.
- Factors are not added to the marginalization prior (same design as the
  pose prior): their information vanishes when the frame leaves the window.
  Intentional — keeps the channel weak and removable.
