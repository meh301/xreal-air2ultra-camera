# XREAL vSLAM pipeline — architecture reference

_Generated 2026-07-19 from a code-grounded, adversarially-verified audit of the pipeline (workflow wf_cb0c294e). Every load-bearing claim cites file:line. For the device-deployment handoff._

## One-paragraph overview

Stereo visual-inertial SLAM for XREAL AR glasses. **Basalt** provides the VIO (square-root keyframe estimator, 15-DoF state, its own KLT patch optical flow — NOT neural features), treated as drifting odometry. A **GLIM-style bounded session map** (xr_map.c) sits on top on a dedicated low-priority thread: it stores keyframes, retrieves loop/reloc candidates with a **VPR embedding** (EigenPlaces-512 / MegaLoc-8448), matches with **XFeat** descriptors + **LighterGlue**, verifies with **gravity-aligned 2-point PnP RANSAC**, and heals drift by deforming the pose chain — feeding corrections back to the live pose (and, under XR_LMFACT, landmark factors back into the VIO cost). Target: Snapdragon 8 Elite Gen 5; neural models run on the HTP/NPU (XFeat) or CPU (VPR/LighterGlue today, NPU at Gen 5). All neural inference is ONNX-Runtime dlopen'd — a lean build silently falls back to classical CPU paths.

## Calibration: used / assumed / dropped (the device-deployment table)

| Quantity | Status | Note |
|---|---|---|
| Camera intrinsics + distortion | **USED** | device fisheye624 **refit to KB4** — tangential/thin-prism dropped |
| Stereo cam-IMU extrinsics T_i_c | **USED, fixed** | never refined online |
| IMU noise densities | **USED** | continuous-time; discretized internally; default if absent |
| IMU accel/gyro bias (additive) | **USED** | factory offset + estimated online |
| IMU accel/gyro **scale + misalignment** | **ASSUMED IDENTITY** | only additive bias modeled — the confirmed altitude-twist driver |
| Cam-IMU **time offset (td)** | **DROPPED (=0)** | no path in the VIT interface; apply commented out |
| **Photometric / vignette** | **offline dropped; online substitute** | no vignette LUT, but on-device EMA radial-gain + contrast-stretch conditions the feed (xreal_jni.c:482-577) |
| Gravity | fixed (gauge) | world z=up; correct, not a defect |

---

## VIO Frontend & Feature Tracking

### Estimator uses Basalt KLT patch tracking, not neural features
The VIO estimator tracks features with **Basalt's own inverse-compositional patch (KLT-style) optical flow**, not XFeat. The factory default is `optical_flow_type = "frame_to_frame"` (`vio_config.cpp:50`), instantiated as `FrameToFrameOpticalFlow` (`optical_flow.cpp:63-77`). A `patch`/`PatchOpticalFlow` and a `multiscale_frame_to_frame` variant also exist but are not selected. On-device/bench use the VIT `create()` path with no config file, so C++ defaults apply (`vit_tracker.cpp:171-178`; `xr_slam.c:328`). Tracking is a Gauss-Newton SE2 patch alignment iterated coarse-to-fine (`frame_to_frame_optical_flow.h:377-438`); new points come from `cv::FAST` corners (`keypoints.cpp:164`).

### Detection, patch, pyramid
- **Pattern**: `optical_flow_pattern = 51` (`vio_config.cpp:57`) = `Pattern51` = `Pattern52 * 0.5`, i.e. **52 sample points** at half radius (`patterns.h:137-146`).
- **Pyramid**: `optical_flow_levels = 3` (`vio_config.cpp:59`) → levels 0-3 (4 octaves), each `scale = 1<<level` (`frame_to_frame_optical_flow.h:383-388`).
- **Detection grid**: 50 px cells, 1 pt/cell, adaptive FAST threshold halving 40→5 (`vio_config.cpp:51-54`, `keypoints.cpp:160-188`).
- **Robustness**: forward-backward re-track, reject if `dist2 > 0.04` (`frame_to_frame_optical_flow.h:354-364`).

### Stereo
Cam0→camN cross-camera matching via the same patch tracker with a depth-based guess, then an epipolar (essential-matrix) filter, `optical_flow_epipolar_error = 0.005` (`frame_to_frame_optical_flow.h:673-731`; `vio_config.cpp:60`). `cam_count` is asserted > 1 (`vit_tracker.cpp:168`).

### Photometric / vignette — NOT applied
Only **per-patch mean normalization** is done (`data /= mean`; `patch.h:97-98,190-199`) — affine brightness invariance, not a response/vignette correction. `Calibration.vignette` exists (`calibration.hpp:158`) but is touched only by offline calibration tooling (`vignette.cpp`, `cam_calib.cpp`, `poses_optimize.h:252`). The runtime path `apply_cam_calibration` sets extrinsics/intrinsics/resolution and **never vignette** (`vit_tracker.cpp:263-306`); the on-device KB4 build has no vignette field at all (`xr_slam.c:291-319`).

### Camera model
Runtime VIT interface supports only NONE(pinhole)/KB4/RT8 (`vit_tracker.cpp:277-298`). On-device = **KB4** fitted from XREAL fisheye624 (`xr_slam.c:298-305`). Bench `calib.txt` = kb4 or rt8 (`xr_replay_main.c:96-98`). The `ds`/`eucm` JSONs in `basalt/data/` are for the standalone Basalt CLI (`vio.cpp --config-path`), not this pipeline.

### Images / resolution
8-bit L8 pushed, stored as uint16 `<<8`, FAST reads `>>8` (`vit_tracker.cpp:509`, `keypoints.cpp:156`). On-device default 480×640 portrait (`xreal_core.h:26-31`). Bench (`prep_dataset.py`): EuRoC 752×480 remapped to a virtual equidistant KB4 (`:19-20,224-256`); TUM-VI 512×512 kb4, 16-bit→8-bit (`:285-356`); MSD indoor 960×960 kb4; MSD outdoor/dance 640×480 rt8 (`:494-513`). (Model/remap/bit-depth are code-verified at these cites; the exact pixel dimensions are read from each dataset's own calibration at runtime, not hard-coded here.)

### XFeat's actual role
XFeat (`xr_xfeat`) runs only inside `xr_map.c` for descriptors/loop closure (`xr_map.c:4173-4256`) — **not** in the estimator frontend. A seed hook (`setXrSeeds`/`xr_slam_seed_keypoints`) can feed XFeat maxima as extra FAST detection seeds, but "KLT still does the tracking" (`xr_replay_main.c:508-510`). This path is **dormant on-device** (no caller of `xr_slam_seed_keypoints` or `xr_map_get_reseed` in the app tree); it is wired only in the bench replay, gated by `XR_SEED=1` + `--xfeat` (`xr_replay_main.c:437-439,511-521`).

## VIO backend estimator

The backend is Basalt's `SqrtKeypointVioEstimator<Scalar>` (`android/app/src/main/cpp/third_party/basalt/src/vi_estimator/sqrt_keypoint_vio.cpp`, header `.../include/basalt/vi_estimator/sqrt_keypoint_vio.h`). It runs on its own `processing_thread` (`.cpp:366`) that pops synced optical-flow frames and IMU data from bounded queues (`.cpp:115-116`).

**Online state (15-DoF).** Each frame state is a `PoseVelBiasState`: `SE3 T_w_i` (6) + `vel_w_i` (3) + `bias_gyro` (3) + `bias_accel` (3) = `POSE_VEL_BIAS_SIZE = 15` (`basalt-headers/.../imu/imu_types.h:48,104,166,239-240`). No camera extrinsics, time offset, or gravity live in the estimated state — those are fixed calibration.

**IMU preintegration.** Between frames a `IntegratedImuMeasurement` is built from bias-corrected accel/gyro (`.cpp:302-336`), used to predict the next state (`meas->predictState(...,g,...)`, `.cpp:409`) and as an inertial factor in linearization (`ImuLinData ild = {g,...}`, `.cpp:1264`).

**Sliding window.** `vio_max_kfs=7`, `vio_max_states=3` (`src/utils/vio_config.cpp:76-77`). The fork adds long-term keyframes (`ltkfs`) held outside the `max_kfs` budget (`.cpp:717`) and optionally fixed in the solve (`vio_fix_long_term_keyframes`, default false, `.cpp:1395-1405`).

**Keyframe selection = connectivity.** A KF is taken when the tracked/(tracked+untracked) ratio in cam0 drops below `vio_new_kf_keypoints_thresh` (0.7) after `vio_min_frames_after_kf` (5) (`.cpp:454-456`). Marginalization picks the KF to drop by `KF_MARG_DEFAULT`: oldest KF whose tracked-landmark fraction falls under `vio_kf_marg_feature_ratio` (0.1), else a DSO-style distance score (`.cpp:814-868`).

**Marginalization = sqrt/ABS_QR.** `vio_sqrt_marg=true`, `vio_linearization_type=ABS_QR` (`vio_config.cpp:73-74`); the marg prior is carried as a square-root information factor via `MargHelper::marginalizeHelperSqrtToSqrt` (`.cpp:1072`), then delta-corrected (`.cpp:1170-1172`).

**Landmark parameterization.** Inverse-depth: `direction` (Stereographic) + scalar `inv_dist`, host-anchored, DLT-triangulated with a metric-baseline gate `vio_min_triangulation_dist` (0.05) and `inv_dist∈(0,3)` (`.cpp:534-539`).

**Fixed gravity as gauge.** `g` is a `const Vec3` set once (`.cpp:63`), never optimized. The marg prior pins position (`head<3>`) and yaw (index 5) at weight `1e8` (`.cpp:87-89`), fixing the 4 unobservable DoF; roll/pitch stay observable through the gravity-referenced IMU factors.

**Map→VIO coupling that SHIPS (committed app tree).** Two unconditional grafts in `optimize()`, fed by the map thread through `vit_tracker_xreal_landmark_factors`/`_pose_prior` → `setXrLandmarkFactors`/`setXrPosePrior` (`vit_tracker.cpp:739-742,725-731`):
1. **XR_TIGHT pose prior** — weak unary SE(3) prior on the newest state toward `E*T`, added as `H+=W, b+=W·r` (`.cpp:1416-1440`), timestamp-expired.
2. **XR_LMFACT fixed-3D landmark factors** — re-observed map points as Huber-robust unary reprojection factors on the observing frame's pose only (no landmark state; map 3D authoritative), `H.block<6,6>+=`, `b.segment<6>+=` (`.cpp:1442-1522`). The map posts inlier 2D-3D pairs transformed odom←session and encodes persist/transient in the sigma sign (`xr_map.c:1212-1236`).

## Session map & loop closure

A GLIM-style bounded session map sits on top of Basalt VIO, which it treats as untouched drifting odometry (`xr_map.h:1-18`). All heavy work runs on one dedicated `nice(19)` map thread (`xr_map.c:5615`); the SLAM worker only posts keyframes through a single non-blocking mailbox that **drops when the thread is busy** (`xr_map_offer2`, `xr_map.c:5726`). `process_keyframe` snapshots the mailbox, does extraction/embedding/matching/PnP/deform **lock-free**, then takes `MAP_LOCK` only for a brief write/publish phase — the priority-inversion fix (`xr_map.c:4313`, `4852`).

**Keyframe storage.** `KF[]` is a fixed pool of `XR_MAP_MAX_KF=3333` slots (`xr_map.h:38`, raised from 400 on 2026-07-19), each holding odom pose `q/p`, corrected session pose `qc/pc`, up to 200 keypoints+descriptors, landmark 3D/uv, a VPR embedding `emb[XR_VPR_MAX_DIM]`, and a submap `seg` id (`xr_map.c:170-204`). Storage gates: motion (`KF_DIST_M=0.30`, `KF_ANGLE_COS`=15° at `xr_map.c:43-44`), `STORE_MIN_LM=20` landmarks, and `STORE_MIN_INTERVAL_NS=350ms` (`xr_map.c:66-68`). Eviction is a rolling least-recently-useful cap; `last_used` refreshes on loop matches and on spatial proximity (`KF_NEAR_M=2m`), so revisited space never rolls off (`xr_map.c:5498-5550`, `4892-4898`).

**Descriptors.** Default is CPU BAD/TEBLID (256-bit Hamming, anchored at VIO landmarks then FAST-grid corners; `bad_extract`, `xr_map.c:2420`). XFeat int8 64-D (ONNX) is optional and default-off (`USE_XFEAT=0`, `xr_map.c:395`, `4242`).

**Retrieval + verification.** When a VPR model is registered, per-keyframe embeddings pre-rank a `VPR_SHORTLIST=12` by cosine (`xr_map.c:4405-4477`); otherwise a spatial gate/full scan runs. Shortlisted keyframes are descriptor-matched (greedy NN+margin, or LighterGlue for XFeat), then verified by covisibility-pooled **gravity-aligned 2-point PnP RANSAC** solving yaw+translation only (`reloc_pnp` `xr_map.c:2925`, `pnp2_ransac` `2673`). Acceptance needs ≥8 pairs, ≥33% near-inlier ratio, and inliers spread over `COVIS_MIN_KF=3` distinct keyframes (`xr_map.c:2582`, `93`, `3266`).

**Closure application.** A verified alignment `D` must be confirmed by a second agreeing query within `CONFIRM_WINDOW_NS=12s` (`xr_map.c:4979`). Healthy drift closures **deform** the tail: default `graph_deform` distributes `D` by session path length (`xr_map.c:1773`); it is effectively 4-DoF because `D` is yaw-only. `XR_PGO`/`XR_PGO4DOF` (Gauss-Seidel, explicit yaw projection, `1862`/`1928`) and `XR_EDGEGRAPH` (persistent closure-edge whole-chain relax, `2062`) are default-off alternatives. Post-loss recovery only *registers* the live frame, never deforming the frozen-correct map (`xr_map.c:5259`).

**Submaps/healing.** Unrecovered loss after `SEG_OPEN_NS=10s` opens a new segment (`CUR_SEG++`); a cross-segment confirmed closure rigidly **welds** it (`seg_weld`, `xr_map.c:2197`, `5202`). Recovery lifecycle HEALTHY/LOST/RECOVERED is driven by `xr_map_freeze_storage` from the IMU thread (`xr_map.c:124`, `4130`).

**Relocalization.** Stationary query-only offers run at `QUERY_INTERVAL_NS=2.5s`; LOST does full scans; `xr_map_probe`/`xr_map_probe_burst` are blocking bench cold-probes (`xr_map.c:5650`, `5680`).

**Feedback to VIO.** By default corrections do **not** re-enter Basalt — `session = D∘odom` is applied only to the displayed pose downstream (`xr_map_get_correction`, consumed at `bench/replay/xr_replay_main.c:273`, `xreal_jni.c:1158`). Two corrections split display vs map: `LIVE` snaps only with recovery on, `CORR` always advances (`xr_map.c:408-409`, `5377-5391`). Real map→estimator coupling (`tight_post_prior`→`xr_slam_pose_prior`, `lmfact_post`→`xr_slam_landmark_factors`) is gated by `XR_TIGHT`/`XR_LMFACT` and no-ops without extended libbasalt symbols (`xr_slam.c:184`).

## Neural model inventory

All neural models run through ONNX Runtime, `dlopen`'d at runtime (`libonnxruntime.so`) — nothing links against it, so a lean build silently drops to classical fallbacks. Five models exist; four are wired live, one is dormant.

### XFeat — dense feature extraction (ACTIVE, map only)
Keypoints + 64‑D descriptors for the **session map**, at keyframe rate (~1–2 Hz) on the map thread — **not** in the Basalt VIO frontend (VIO uses its own optical flow; the map's non‑XFeat descriptor is BAD/TEBLID). Dual path behind one contract (xr_xfeat.c:1‑13): **NPU** dense backbone as a QAIRT‑native **A8W8** EPContext (u8 IO, 3.6 ms on SD888 HTP), with NMS/top‑K/bilinear descriptor sampling reproduced in C; **CPU** fallback (full `xfeat.onnx`, or `xfeat_dense_dyn.onnx`). Resolution fixed **480×640** (`IMG_W=XR_OW=480`, `IMG_H=XR_OH=640`, xr_xfeat.c:37‑38). NPU IO shapes: score `[1,1,640,480]`, desc `[1,64,80,60]`, reliability `[1,1,80,60]` (xr_xfeat.c:240‑241). Called from xr_map.c:4256 (`xr_xfeat_extract_anchored`); NPU inference holds `XR_NPU_GATE`. Target: **NPU‑HTP** (QNN), CPU‑EP fallback.

### VPR / place recognition — EigenPlaces / MegaLoc (ACTIVE, map only)
One L2‑normalized global embedding per keyframe to pre‑rank loop/reloc candidates (xr_vpr.h:1‑16). Input `[1,1,H,W]` raw gray 0..255; output unit‑norm `[1,DIM]`, DIM discovered at bring‑up (`XR_VPR_MAX_DIM 8448`, xr_vpr.c:161). **EigenPlaces‑512** is the deploy/plumbing model; **MegaLoc‑8448** swaps behind the same interface. Container arm registry: the `vpr` arm uses EigenPlaces‑512, while `megaloc`/`xdlg6`/`full` use MegaLoc‑8448 (EVALUATION.md:147‑150). Compute: **CPU‑EP** on device; opt‑in **CUDA EP** for host/container replay (`XR_ORT_CUDA`, xr_vpr.c:80‑108) — MegaLoc fp32 is ~915 MB / ~229 ms/embed on CPU. On‑device NPU context arrives with Gen 5 (xr_vpr.h:11‑13). fp32 in‑repo. Called from xr_map.c:4306.

### LighterGlue — learned matcher (ACTIVE, map verification only)
XFeat's trimmed LightGlue; replaces greedy NN+margin for loop/reloc PnP correspondences, run **only on candidate keyframes** (`c==0` budget, xr_map.c:2990‑2996), never in the retrieval scan. Static shape, `XR_LGLUE_N=512` slots/side (xr_lighterglue.h:24); kpts normalized, int8 descs dequant /127. **fp32**, CPU‑EP (CUDA on bench container, xr_lighterglue.c:75‑98) — no NPU path. Official 6‑layer L6.

### LiteAnyStereo / LAS2 — stereo depth (ACTIVE, shipping depth worker)
Metric stereo depth on NPU (xr_liteanystereo.h:1‑11), invoked by the depth worker (xreal_jni.c:1495‑1527), SGM CPU fallback. Two slots: **192×256 fast** (~28‑32 ms), **288×384 MID** (~65‑72 ms). **A16W8** native, **u16** IO, 1‑channel input; disparity→`depth=f*base/disp`. Shipped A16W8@192. Target: **NPU‑HTP**, gated by `XR_NPU_GATE`.

### ZipDepth — monocular depth (DORMANT)
Mono metric depth, input `(1,3,384,384)` RGB/255, output `(1,1,384,384)` affine‑metric (xr_zipdepth.h:9‑14). **DORMANT**: compiled in (Android.mk) and header included, but `xr_zipdepth_init`/`xr_zipdepth_run` have **no caller** in the pipeline — superseded by LAS2. Its on‑device QNN `graphCreate` wall was never solved (xr_zipdepth.c:186‑227).

## Calibration: used vs dropped

The VIO (Basalt behind Monado's VIT interface) is fed calibration **programmatically** — never a Basalt calib JSON. Two feed paths exist:

- **Device** — `xr_slam_start` (`xr_slam.c:322`), from the 82-float factory blob delivered via `nativeSetAlignment` (`xreal_jni.c:2671-2717`): per-eye fisheye624 intrinsics, `q_cam`/`p_cam`, and IMU `gyro_bias`/`accel_bias`/`noises`.
- **Bench/dataset replay** — `xr_slam_start_raw` (`xr_slam.c:609`), from `calib.txt` (`model`, `*_pinhole`, `*_dist`, `*_q_xyzw`, `*_p`, `noises`) written by `pack_common.py:format_calib` (`pack_common.py:110-136`) and parsed by `read_cam`/`read_calib` (`xr_replay_main.c:61-105`).

Both push `vit_camera_calibration_t`/`vit_imu_calibration_t` into Basalt's `Calibration` in `apply_cam_calibration`/`apply_imu_calibration` (`vit_tracker.cpp:263-359`).

**USED**
- **Camera intrinsics + distortion.** Device: fisheye624 is **refit to KB4** by least squares over the FoV (`fit_kb4`, `xr_slam.c:232-277`); only 4 radial coeffs survive — tangential/thin-prism `kc` terms are **dropped** (`xr_slam.h:9-12`, `xr_slam.c:229-231`), always emitted as `KB4` (`xr_slam.c:303`). Bench: `fx,fy,cx,cy`+dist pass straight through as KB4 (`k[0..3]`) or RT8 (`k[0..7]`+rpmax) (`xr_slam.c:643-657`) → `GenericCamera` (`vit_tracker.cpp:283-300`).
- **Stereo extrinsics `T_i_c`** (cam→IMU). Built from quaternion+translation into a 4×4 `T_imu_cam` (`xr_slam.c:310-319`, `658-667`) → `calib.T_i_c` (`vit_tracker.cpp:267-273`). Held fixed.
- **IMU noise densities** `{gyro_noise, gyro_bias_rw, accel_noise, accel_bias_rw}` → `noise_std`/`bias_std` (`xr_slam.c:363-366`, `683-686`; `vit_tracker.cpp:339-340,355-356`); defaults `{0.00035,0.0001,0.00667,0.001}` if absent (`xr_slam.c:356,678`). Bench allows an `XR_IMU_NOISE_SCALE` multiplier (`xr_replay_main.c:447-453`).
- **IMU accel/gyro bias (additive).** Device feeds factory bias as `offset = -bias` (`xr_slam.c:361-362`). Bench leaves `offset = 0` (memset, `xr_slam.c:675`) → **estimated online** (`xr_slam.h:79-82`). The estimator's time-varying bias state runs on top regardless.

**DROPPED / ASSUMED**
- **IMU scale + misalignment.** `accel.transform`/`gyro.transform` set to **identity** (`xr_slam.c:359-360,681-682`); `vit_tracker.cpp:333-334,349-350` compute `transform − I` = 0, so `calib_accel_bias`/`calib_gyro_bias` carry zero scale/misalignment. Never calibrated.
- **cam-IMU time offset `td`.** `cam_time_offset_ns` defaults 0 (`calibration.hpp:62`), never set; the apply line is **commented out** in VIO and VO (`sqrt_keypoint_vio.cpp:258`, `sqrt_keypoint_vo.cpp:213`).
- **Vignette/photometric.** `calib.vignette` exists (`calibration.hpp:158`) but `apply_cam_calibration` never touches it — always empty.
- **Gravity.** Fixed gauge `g = (0,0,−9.81)` (`imu_types.h:63`) passed to the estimator (`vit_tracker.cpp:388`); not per-device, not in `calib.txt`.

Per-device (factory blob): intrinsics, extrinsics, biases, noises. Dataset-provided (`calib.txt`): all intrinsics/extrinsics/noises, biases zeroed. Common to both: identity IMU scale, `td=0`/unused, empty vignette, fixed gravity.

## Device deployment & what is lost on-device

Offline benchmarks feed Basalt a per-dataset calibration JSON (intrinsics, cam-IMU extrinsics, IMU noise, time offset — e.g. `data/euroc/euroc_ds_calib.json`) that datasets provide. On the XREAL glasses (SD 8 Elite Gen 5) every one of these must be re-derived per unit or is dropped.

**Camera intrinsics/distortion.** The factory blob is fetched over the HID IMU channel (cmds `0x14/0x15`, `xreal_jni.c:1774-1798`) and parsed as an 82-float alignment array (`xreal_jni.c:2671-2702`): per eye `K,q_display,q_cam,fc,cc,kc[12],p_cam`. The native camera is fisheye624 (12 `kc`), but for VIO it is **least-squares down-fit to a 4-param KB4** model, explicitly dropping tangential/thin-prism terms as "~0" (`xr_slam.c:231-277`, applied `298-305`). The usable image circle must match the sensor — the device asset ships `optical_flow_image_safe_radius: 305.0` (`basalt_vio_config.json:17`; TUNEABLES.md:227, "wrong = ...").

**Cam-IMU extrinsics + time offset.** Extrinsics come from `q_cam`/`p_cam` (`xr_slam.c:310-319`); the stereo baseline is `p_cam[right]-p_cam[left]`, which gates VIO startup (`xreal_jni.c:2709-2717`). **There is no time-offset (td) path**: `vit_camera_calibration_t` has no offset field (`vit_interface.h:253-264`), and Basalt's internal `cam_time_offset_ns` defaults to 0 (`calibration.hpp:62`) with no way to set it — on-device td is implicitly zero.

**IMU calibration.** Factory gyro/accel biases are injected as `offset = -bias` (`xr_slam.c:361-362`; the raw stream is uncorrected). The four continuous-time noise densities come from the blob or default to `{3.5e-4, 1e-4, 6.67e-3, 1e-3}` (`xr_slam.c:356`), discretized internally by `σ·sqrt(update_rate)` (`calibration.hpp:190,196`; enter Allan-variance densities, not discrete — TUNEABLES.md:221-224). **Scale/misalignment is hard-set to identity** — only the transform diagonal is written (`xr_slam.c:359-360`, raw path `681-682`); the blob carries no intrinsic matrix, so cross-axis scale/misalignment is assumed perfect. Global reweighting is via `XR_IMU_NOISE_SCALE` (TUNEABLES.md:60).

**Photometric/vignette.** Offline photometric calibration (per-lens vignette LUT / response) is not used. Instead an **online** EMA radial-bin vignette gain + contrast stretch conditions the Basalt feed (`xreal_jni.c:482-577`, pushed at `621`), with `XR_ZNCC=2` arming ZNCC only on exposure transients (TUNEABLES.md:55,70,79). A true per-lens photometric capture is absent.

**Real-time.** Device asset sets `vio_enforce_realtime: true` (`basalt_vio_config.json:44`) — drops frames rather than lag head motion (TUNEABLES.md:230); bench runs false. BA window is capped `max_kfs 7 / max_states 3` (`json:32-33`). NPU work is serialized through the `XR_NPU_GATE` mutex (`xr_liteanystereo.c:25,123,288`, a V68 workaround) and the map thread is duty-capped ~25% (`xr_map.c:4179-4182`). Model budgets (SD888 [M]→Gen5 [P]): XFeat dense A8W8 3.6→1-1.5 ms, LAS2 A16W8@192 32→9-12 ms, EigenPlaces-512 16.6→~5 ms, MegaLoc fp16 18-35 ms (GEN5_DEPLOYMENT.md:34-56).

**Degradation if wrong.** IMU scale error is unabsorbable and drives vertical/altitude twist — our VIO's altitude is 85% of drive error vs OKVIS2's 5%; a 2% accel-scale inject degrades V +8% (ITERATIONS.md:2570-2597). td/lever-arm error produces velocity-scaled compounding — the 4Seasons drive scale error compounds 3.2× at 2 min, with cam-IMU time offset named a prime suspect (ITERATIONS.md:1194-1197). Missing photometric calibration breaks brightness-constancy under lighting/exposure swings, destabilizing the patch tracker (the online vignette/ZNCC paths exist precisely to mitigate this).

