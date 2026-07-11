# vSLAM for the research app — evaluation & integration plan

Goal (from the research track): a **full stereo-inertial SLAM pipeline running
in-app on the phone** — lightweight, fast, low-latency; "real-world
localization" rather than SOTA benchmark accuracy. Wishlist: hooks for
loosely-coupled GNSS fusion later (Ai-Thinker GP-02 module), stereo depth
estimation for AR occlusion masking, and visualization of the tracked
landmarks on the passthrough.

What this repo already provides — and it is exactly the input a VIO system
wants: synchronized 30 Hz/eye stereo, 1 kHz IMU on the **same nanosecond
clock** with hardware exposure timestamps, and the factory calibration
(fisheye624 intrinsics, camera↔IMU extrinsics, IMU noise densities) fetched
from the device.

## Candidates

| System | Type | Stereo+IMU | Loop closure / reloc | GNSS hook | ROS-free embedding | License | Phone fit |
|---|---|---|---|---|---|---|---|
| **Basalt** (Collabora/Monado fork) | optimization VIO + mapper | ✓ | partial → improving (persistent mapping in progress) | ✗ (add loose fusion ourselves) | ✓ (built for embedding; `vit_interface`) | BSD-3 | **proven — it is what Monado uses for headset inside-out tracking** |
| **VINS-Fusion** (HKUST) | optimization VIO + pose graph | ✓ | ✓ | **✓ built-in** (`global_fusion`: GPS + VIO pose graph) | ✗ (ROS-coupled; ports exist, surgery needed) | GPLv3 | good (VINS-Mobile lineage ran on phones) |
| **ORB-SLAM3** | feature SLAM, atlas | ✓ | ✓✓ (best reloc/multi-map) | ✗ | ✓ (needs Pangolin removed) | GPLv3 | heavy; needs feature-count tuning at 480×640 |
| **OpenVINS / MINS** (RPNG) | MSCKF filter | ✓ | ✗ (odometry only) | MINS: ✓ GNSS | mostly (ROS-wrapped, core separable) | GPLv3 | lightest CPU of all |
| **HybVIO** (Spectacular AI) | hybrid filter + loose SLAM | ✓ | loose SLAM module | ✗ | ✓ | GPLv3 ("not for production") | excellent benchmark speed on consumer HW |
| **OKVIS2-X** (2025) | keyframe VI-SLAM | ✓ | ✓ | **✓ GNSS built-in** | partially | BSD-ish (check) | new; also configurable with **dense depth** — worth tracking |

Excluded: ARCore (closed, and bound to the phone's own cameras — it cannot
consume the glasses' external UVC stream), Kimera (research-grade, heavy),
stella-vslam (no IMU).

## Accuracy: Basalt vs VINS-Fusion (published benchmarks)

On the standard stereo-inertial benchmarks (EuRoC, TUM-VI), the consistent
ordering is **ORB-SLAM3 > Basalt > VINS-Fusion**: ORB-SLAM3 averages
~3.6 cm ATE on EuRoC stereo-inertial; Basalt lands roughly in the 5–8 cm
band (winning some sequences outright); VINS-Fusion typically ~10 cm —
about 30–50 % higher error than Basalt — while also being the heaviest to
de-ROS. So between the two asked about: **Basalt is both more accurate and
significantly lighter/faster**; VINS-Fusion's remaining edge is purely its
built-in GNSS `global_fusion` module, which we replicate externally anyway.

## Deployment shape per environment

- **Indoors — relocalization is key**, and the lab's building-wide fog
  compute goal implies **cloud-hosted maps revisited across sessions and
  devices**. That architecture wants a *split* design regardless of
  tracker: a light on-device VIO for low-latency pose, plus a map service
  (fog node) that ingests keyframes/landmarks, builds and stores maps
  long-term, and answers relocalization queries — the visual-positioning
  pattern. Basalt's persistent mapping is younger than ORB-SLAM3's atlas,
  but with the map/reloc layer living on the fog side, the on-device
  tracker's native map-save maturity stops being the deciding factor.
- **Outdoors — no relocalization**: GP-02 GNSS provides initial position
  and (from motion) heading; the 4-DoF alignment estimator anchors the VIO
  trajectory to ENU continuously.

## Recommendation

**Primary: Basalt, the Monado/Collabora fork.** It is the only candidate
that is *in production for precisely our problem* — camera+IMU inside-out
tracking of a head-mounted device on constrained hardware, latency-first. No
ROS anywhere, BSD licensed, C++ that embeds as a library behind a small
tracking interface, and the fork is actively gaining persistent
mapping/relocalization. Its optical-flow front end is cheap at our 480×640.
What it lacks — GNSS — is the piece we planned to add *loosely* anyway: a
loose GPS fusion is an external alignment (similarity/pose-graph between the
VIO trajectory and GNSS fixes, the same pattern as VINS' `global_fusion`
node) and does not need to live inside the estimator.

**Fallback: ORB-SLAM3** if on-device saved-map relocalization must land
*before* the fog map service exists — its atlas save/load and DBoW2
relocalization are the most mature available, at the cost of CPU weight and
GPLv3. **VINS-Fusion** is deprioritized on accuracy (see benchmarks above);
we borrow its `global_fusion` *pattern* for GNSS rather than its codebase.
Keep an eye on **OKVIS2-X** (GNSS + dense depth in one system) as it
matures.

## Integration plan (phased)

1. **Calibration bridge** — none of the candidates speak `fisheye624`.
   Write a converter that samples our validated projector
   (`python/xreal_align.py::fisheye624_project`) on a grid and refits
   Kannala-Brandt (kb4/KB8 — Basalt and ORB-SLAM3 both support KB) plus the
   extrinsics/noise into the target config format. Expected sub-pixel fit
   over the usable FOV.
2. **Feed adapter (native)** — the app already owns descrambled stereo and
   1 kHz IMU with shared-clock timestamps in C. Add a `xr_slam_feed` shim
   that hands frames + IMU into Basalt's input queues in-process. Two
   caveats already known:
   - feed the FPN-cleaned but **NOT histogram-equalized** image — the
     per-frame global equalization flickers brightness and would hurt
     feature matching (add a tap before `equalize` in the pipeline);
   - each eye updates at 30 Hz alternating; Basalt accepts per-camera
     timestamps, so feed frames individually rather than as forced pairs.
3. **Pose out, points out** — consume Basalt's pose + landmark stream;
   project landmarks through the existing display model
   (`xr_align_ray`/`xr_align_project` inverses are not needed — landmarks
   are world points: transform world→IMU-now→display via the calibrated
   poses) and draw them as a GL point pass in `xreal_gles.c` on top of the
   passthrough. Timewarp already provides IMU-now.
4. **Depth for occlusion — target 30 Hz, not 60** — independent of the VIO
   choice. 60 Hz depth is physically pointless here: each camera refreshes
   at 30 Hz, so a faster depth loop would recompute identical inputs.
   Instead, compute depth at the full sensor rate (30 Hz) and let the
   existing timewarp carry the occlusion mask to display cadence — the mask
   rides the same distortion mesh as the image, so occlusion stays locked
   to the warped view between camera frames.
   Feasibility on CPU: rectify with the factory extrinsics (13.7 cm
   baseline), run OpenCV SGBM at **half resolution (~320×240, 48–64
   disparities): ~5–10 ms/pair on one phone big core = comfortable 30 Hz**;
   full 480×640 SGBM (~30–60 ms) would not hold 30 Hz on CPU. Recover edge
   quality by joint-bilateral/guided upsampling of the half-res disparity
   with the full-res image (~2 ms). GPU (GLES compute / TFLite-GPU stereo
   net) is the upgrade path only if half-res + guided upsample proves too
   coarse for convincing occlusion boundaries — not needed for v1.
5. **GNSS (GP-02) loose fusion** — outdoors-first: NMEA fixes feed a small
   alignment estimator (4-DoF similarity between VIO odometry and ENU),
   modeled on VINS `global_fusion`, giving initial position + heading and
   continuous drift anchoring; indoors it simply disengages. No changes to
   the VIO core.
6. **Fiducial anchors (future enhancement)** — AprilTag (BSD C library) or
   OpenCV ArUco detection on the tracking frames (~5–15 ms at 480×640 with
   decimation, run at a few Hz, not per-frame). Uses: instant relocalization
   anchors indoors, deterministic outdoor↔indoor handover (tag at the
   entrance defines the transform between the GNSS-anchored outdoor frame
   and the building map), and shared reference frames across devices in the
   fog — tags are the cheap ground truth that visual maps drift around.
7. **WebSocket telemetry/map client** — a background client in the research
   app (OkHttp WebSocket; the research branch accepts the dependency)
   streaming to a fog/management server:
   - `pose` (~30 Hz): timestamped 6-DoF pose + tracking state, for live
     visualization in a management interface;
   - `landmarks` (incremental): new/updated map points for remote map view;
   - `keyframe` (on keyframe creation, optional/throttled): compressed
     image + feature data — the ingredient the fog map service needs to
     build and store building-wide maps long-term;
   - `map` (on demand): serialized map blob upload/download for session
     persistence until the server-side map builder exists.
   JSON first for debuggability, binary (flatbuffers) once bandwidth
   matters. The server side (storage, map merging, relocalization service,
   web viewer) is a separate fog-infrastructure work package.

## Performance expectations (phone-class ARM, our resolution)

- Basalt front end at 480×640×2 @30 Hz/eye: well under one big core.
- IMU at 1 kHz: negligible.
- SGBM half-res: ~10 ms/pair on a big core (run at 10–15 Hz for occlusion).
- Landmark overlay: trivial (a point-sprite draw in the existing renderer).

## Implementation status (research branch)

The research app skeleton is **live** — everything around the SLAM core
exists and runs, with a lightweight stand-in front end where Basalt will sit:

| Piece | File(s) | Status |
|---|---|---|
| Stereo rectification (factory calib → 240×320 portrait pinhole pair, x = 13.7 cm baseline, f=200, bilinear) | `android/.../cpp/xr_stereo.c` | ✓ built from `imu_p_cam`/`imu_q_cam` + fisheye624 at runtime |
| Stereo depth — 9×7 census + **4-path SGM** (P1=10, adaptive P2 120/48 at edges) + uniqueness + LR consistency + subpixel (¼ px) + speckle region filter + occlusion hole-fill + median | `xr_stereo.c` | ✓ NEON-vectorized (u16×8 SGM inner loop, vectorized WTA); scalar SGM measured 62 ms on-device, NEON pass expected ~3-4× faster; auto-drops to every-2nd-pair if a pass exceeds 26 ms so tracking never starves |
| Sparse feature tracker (grid-seeded Shi-Tomasi corners, 7×7 SAD, gyro-predicted search, forward-backward check, subpixel) | `xr_track.c` | ✓ stand-in for Basalt's optical-flow front end |
| Non-equalized tap for trackers (plan item 2 caveat) | `xreal_core.c` `xr_clean(..., do_equalize)` | ✓ display gets equalized, SLAM gets raw |
| SLAM worker thread (rectify + track + depth per stereo pair, newest-wins) | `xreal_jni.c` `slam_worker` | ✓ |
| Landmark overlay on the passthrough (GL_POINTS pass riding the same timewarp dR as the image) | `xreal_gles.c`, `xr_align_ray_to_display()` | ✓ |
| Glasses eye modes: camera / depth (per-eye world-aligned) / AR points / off | `xr_gles_set_eye_mode`, Eye button | ✓ |
| Accumulated landmark cloud in the 3D viewer — Basalt's estimator landmarks (stable ids + inverse depths via VIT `POSE_FEATURES`) triangulated to world xyz and accumulated client-side (4096 cap, id-hashed) | `xr_slam_poll` landmarks, `nativeGrabMap`, PoseMapView cloud | ✓ **note: this is a visualization of VIO structure, not a relocalizable map** — Basalt's VIT path is sliding-window odometry (3 states + 7 keyframes, marginalized); no loop closure, no relocalization, no persistent map on-device. Real mapping/reloc = the fog-side map service (plan §Deployment) or an ORB-SLAM3-class swap if on-device reloc becomes a hard requirement |
| Flicker-free contrast stretch on the Basalt feed (EMA'd 2%/98% percentiles — per-frame HE flickers, raw frames starve FAST on grain) + detection density config (grid 40, min threshold 3) | `frame_cb` stretch, `assets/basalt_vio_config.json` | ✓ |
| Phone UI: tracking pane \| depth pane, 3D pose/map view, buttons Rst/Pts/Dep/SBS/Snap | `MainActivity.kt`, `PoseMapView.kt` | ✓ orientation from AHRS; position zeros until Basalt |
| Pose export | `nativeGrabPose` (q, p, tracked, depth ms, flags; bit3 = Basalt live) | ✓ |
| **Basalt backend** — the Monado fork built for Android (arm64), loaded at runtime via the VIT C interface | `xr_slam.c`, `android/build_basalt.ps1`, jniLibs: `libbasalt.so` + `libtbb.so` + `libc++_shared.so` | ✓ **live**: 6-DoF pose + velocity, per-camera features (u,v,depth) drive the overlay; fisheye624→kb4 refit on the fly; without the lib the app falls back to the built-in tracker |
| GNSS (GP-02), fiducials, WebSocket client | — | deferred per plan |

### Basalt wiring notes

- `android/build_basalt.ps1` reproduces the whole native build: NDK r27c +
  CMake/Ninja (downloaded to `~/Android/toolchains`), oneTBB 2021.13
  (tbbmalloc off — its version scripts don't link on Android), fmt 9.1,
  OpenCV 4.10 Android SDK (static), then basalt with
  `BASALT_BUILD_SHARED_LIBRARY_ONLY=ON`, float-only instantiation,
  `CXX_MARCH=armv8.2-a+fp16+dotprod` (Snapdragon 888 floor). Output ~2.8 MB
  stripped. Local patch applied by the script: `find_package(TBB CONFIG)`
  (basalt's bundled FindTBB predates oneTBB) and cereal's Boost-dependent
  perf tests skipped via `-DSKIP_PERFORMANCE_COMPARISON=ON`.
- Runtime flow: `nativeSetAlignment` → `xr_slam_start` (dlopen, KB4 refit of
  the radial θ-polynomial by linear least squares — exact, tangential terms
  ~0 — plus `T_imu_cam` from `imu_q_cam`/`imu_p_cam`, IMU noise defaults) →
  UVC thread pushes L8 pairs (left, right; exposure ns), IMU thread pushes
  1 kHz rad/s + m/s² → SLAM worker polls poses+features at 30 Hz.
- IMU noise densities are EuRoC-class defaults for now; biases are
  estimated online. If drift says it matters, wire the factory blob's
  `imu` section through `vit_imu_calibration`.
- The AHRS stays as the timewarp's 1 kHz pose source; Basalt is the map/
  pose truth. Fusing Basalt orientation into the warp is a later
  refinement (needs care at 30 Hz vs 1 kHz).

### Basalt build plan (next session)

1. **Fetch**: `basalt-monado` (Collabora fork) + submodules; deps are
   header-only or CMake-friendly: Eigen, Sophus, cereal, {fmt}, TBB
   (oneTBB builds for Android; or swap its task usage for a thread pool),
   basalt's own `basalt-headers`. No ROS, no Pangolin needed when built
   headless with the `vit_interface`.
2. **Build shape**: one `libbasalt_vit.so` prebuilt per ABI via a separate
   CMake toolchain build (NDK toolchain file), then linked into `xrealcam`
   by ndk-build as a `PREBUILT_SHARED_LIBRARY` — keeps our fast ndk-build
   iteration.
3. **Calibration**: converter refits fisheye624 → Kannala-Brandt (kb4) by
   sampling `xr_align_project` on a grid (plan item 1); emit basalt's JSON
   calib (T_imu_cam from `imu_p_cam`/`imu_q_cam`, noise densities from the
   `imu` block of the device JSON).
4. **Wire-in**: replace `xr_track_step` inside `slam_worker` with the VIT
   frame+IMU feed (IMU samples forwarded from the existing 1 kHz drain
   thread), fill `nativeGrabPose` q/p from VIT pose output, hand VIT
   landmarks to `xr_gles_set_points` instead of tracker rays. The depth
   path stays as-is.

### Session mapping & relocalization architecture (implemented; fog side pending)

GLIM's module split (odometry / local mapping / global mapping) transposed
to our visual-inertial stack — three frames, three owners:

1. **Odometry — Basalt, untouched.** Drifting `odom` frame at 30 Hz.
   Basalt cannot consume pose corrections and doesn't need to.
2. **Session map — on-device (`xr_map.c`).** Motion-gated keyframes
   (≥ 0.3 m / 15°) holding the immutable odom pose, a pose-graph
   **corrected pose** (`qc/pc`), and ~200 descriptors — mini-ORB (FAST-9 +
   intensity-centroid orientation + rotated 256-bit BRIEF from a fixed
   seeded pattern, no OpenCV) or **XFeat** (the primary; mini-ORB is too
   weak for reliable reloc on these grainy sensors). Landmarks are stored
   in the owning keyframe's odom frame — **anchor-local**: their session
   position rides on `qc/pc`, so moving a keyframe moves its whole point
   cloud for free. **Bounded**: ≤ 200 keyframes, rolling
   least-recently-useful eviction (spatial recency keeps the current space
   resident). The loop pipeline:
   - **candidates** — brute-force descriptor matching (**top-K** clusters
     when relocalizing, since a repetitive scene can hand raw scoring to
     the wrong keyframe);
   - **verification** — a **gravity-aligned 2D-3D PnP** relocalization (the
     map supplies 3D, the query supplies *pixels* — the ORB-SLAM reloc
     pattern, *not* 3D-3D; both odom worlds are z-up so the unknown reduces
     to yaw + translation, a 2-point closed form + RANSAC + linear refine),
     with **covisibility-pooled** one-to-one correspondences and inliers
     that must be backed by ≥ 3 distinct observing keyframes, then a
     **second verified frame must agree** (temporal confirmation) before
     anything is applied;
   - **closure** — a confirmed closure **deforms the 4-DoF keyframe pose
     graph**: the matched anchor and everything before it hold still, the
     drifted tail is pulled toward the correction in proportion to its odom
     path length (drift accrues with distance), and every keyframe's
     anchor-local landmarks follow, so the historical map co-localizes onto
     the established one — real closure, *not* just a live-pose snap. The
     displayed cloud is DERIVED from the keyframe graph
     (`xr_map_get_cloud`) and deforms with it — there is no flat cloud to
     rigidly retransform. With recovery on, the live `T_session←odom`
     correction also snaps so the pose follows the healed map; with it off
     the map still heals but the pose stays odometry-continuous (the
     GNSS-fusion mode). A **confirmed-recovery lifecycle** (healthy → lost
     on a shake → relocalizing → recovered) keeps storage frozen until a
     closure is verified, not on a fixed timer. Basalt never sees any of it.
3. **Global / cloud map — same format, fog side.** The anchored keyframe
   graph *is* the cloud format: `xr_map_save`/`xr_map_load` persist and
   restore it on-device (the substrate for cross-session persistence and
   the fog map service; the WebSocket `keyframe`/`landmarks` payloads are
   the same struct). A loaded map is the reference the live session
   relocalizes INTO — the first verified closure **registers** the two
   frames (`T_global←session`) and the graph heals as usual. Fog-side
   long-term map building and full session↔cloud co-optimization
   (small_gicp, MIT, the natural registration library) remain the pending
   work package; on-device save/load/register exists now.

Borrowed from GLIM/koide3's CPU playbook: the module split itself, bounded
incremental containers instead of grow-forever structures, keyframe
decimation so expensive work touches few nodes, and (when session scale
ever demands it) voxel-hashed association à la small_gicp instead of
trees. small_gicp (MIT) is also the natural library for the session→cloud
registration step on the fog side.

### Performance & localization research menu (beyond koide3, surveyed 2026-07)

**Localization quality**
- **XFeat (CVPR 2024)** — learned detector/descriptor with real-time
  *sparse inference on CPU at VGA*, ~5× faster than comparable learned
  features, far more robust than ORB-class descriptors on noisy/
  low-contrast images (exactly our sensors). Adoption path: (1) session
  keyframes only (~1–2 Hz → negligible cost, big reloc-quality win over
  the built-in mini-ORB) via ONNX Runtime Mobile; (2) per-frame on the
  Xperia's Hexagon NPU later. ONNX exports exist incl. LightGlue pairing.
- **Photometric compensation** — the fisheye vignette violates the
  brightness-constancy assumption of both Basalt's patch tracker and any
  KLT at the image periphery (TUM's online photometric calibration line
  of work). We can go beyond the shipped temporal-stable contrast
  stretch: fit the vignette profile once (flat-scene or from the optics)
  and divide it out of the tracker feed → more usable features toward
  the edges, better geometry conditioning. Low effort, targets our
  single worst image defect.
- **OKVIS2 / OKVIS2-X** (BSD) — full VI-SLAM with *built-in* real-time
  loop closure (pose-graph edges created by marginalizing common
  observations, revived into landmarks on closure) + DBoW2
  relocalization; 2-X adds GNSS (→ GP-02) and dense-depth support. This
  is the integrated alternative if the session-layer route (Basalt +
  pose-graph-lite) underdelivers: one port replaces three custom layers.
  Cost: heavier than Basalt, new Android port effort.
- **iBoW-LCD / OBIndex2** — incremental bag of binary words: no offline
  vocabulary, dynamic-island temporal grouping. Overkill for the bounded
  session map (brute force wins there) but the right shape for FOG-side
  global place recognition over unbounded maps.

**Compute**
- NEON-batched Hamming with early-out for descriptor matching (only
  when session scale demands it).
- SIMD FAST (OpenCV-style) if keyframe rate ever grows.
- XFeat→NPU (QNN/HTP) on SM8850: moves the whole descriptor stage off
  the CPU.
- GPU compute SGM for depth (below) remains the biggest single CPU
  reclamation available.

### GPU depth (queued behind Basalt)

The CPU SGM is NEON-vectorized and auto-decimates, but Basalt will want
those CPU cycles — moving depth to the GPU is the plan once the backend
lands (or sooner if NEON still can't hold 30 Hz on the target phone):

- **GLES 3.1 compute SGM**: census + cost volume are embarrassingly
  parallel; the 4 directional scans parallelize across the perpendicular
  axis (240–320 invocations each, 48-wide disparity vectors in shared
  memory). Needs its own EGL context + thread so it never contends with
  the latency-critical front-buffer present. At 240×320×48 this is well
  inside mobile-GPU real-time budgets (embedded-GPU SGM literature:
  VGA×128 disp at 42 fps on a Tegra X1).
- **Alternative**: a small stereo network via TFLite GPU delegate
  (MADNet-class) — denser output on low texture, but a heavyweight
  dependency; evaluate only if SGM quality stays unsatisfying.

Sources: [Basalt for Monado (Collabora)](https://www.collabora.com/news-and-blog/blog/2022/04/05/visual-inertial-tracking-support-for-monado-openxr/),
[Monado](https://monado.freedesktop.org/),
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion),
[HybVIO](https://github.com/SpectacularAI/HybVIO),
[OKVIS2-X](https://arxiv.org/pdf/2510.04612),
[ORB-SLAM3 paper (EuRoC/TUM-VI stereo-inertial benchmark tables)](https://arxiv.org/pdf/2007.11898v1),
[Comparison of modern open-source visual SLAM](https://ar5iv.labs.arxiv.org/html/2108.01654),
[ORB-SLAM3 / OpenVINS overview](https://github.com/topics/visual-inertial-odometry).
