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

Sources: [Basalt for Monado (Collabora)](https://www.collabora.com/news-and-blog/blog/2022/04/05/visual-inertial-tracking-support-for-monado-openxr/),
[Monado](https://monado.freedesktop.org/),
[VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-Fusion),
[HybVIO](https://github.com/SpectacularAI/HybVIO),
[OKVIS2-X](https://arxiv.org/pdf/2510.04612),
[ORB-SLAM3 paper (EuRoC/TUM-VI stereo-inertial benchmark tables)](https://arxiv.org/pdf/2007.11898v1),
[Comparison of modern open-source visual SLAM](https://ar5iv.labs.arxiv.org/html/2108.01654),
[ORB-SLAM3 / OpenVINS overview](https://github.com/topics/visual-inertial-odometry).
