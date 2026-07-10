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

**Fallback / GPS-first alternative: VINS-Fusion** if the GP-02 integration
becomes the near-term priority — its `global_fusion` module is exactly the
loose GPS+VIO fusion we described, at the cost of de-ROS-ing the codebase
and a heavier optimizer. **ORB-SLAM3** only if persistent multi-session maps
and relocalization become the core requirement. Keep an eye on **OKVIS2-X**
(GNSS + dense depth in one system) as it matures.

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
4. **Depth for occlusion** — independent of the VIO choice. Rectify the
   stereo pair with the factory extrinsics (13.7 cm baseline) and run
   OpenCV SGBM at 480×640 or half-res on CPU (~10–20 ms — fine at
   occlusion-mask cadence, which can lag the display loop). Upgrade path: a
   small TFLite/NNAPI stereo network if SGBM quality disappoints. Output: a
   per-eye depth texture the GLES renderer can test AR content against.
5. **GNSS (GP-02) loose fusion** — later phase: feed NMEA fixes into a
   small alignment estimator (4-DoF similarity between VIO odometry and
   ENU) modeled on VINS `global_fusion`; publishes a drift-corrected global
   pose without touching the VIO core. Works identically over Basalt.

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
[ORB-SLAM3 / OpenVINS overview](https://github.com/topics/visual-inertial-odometry).
