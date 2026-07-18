#!/usr/bin/env python3
"""Stage 16 (VIO estimator architecture): parallax keyframe trigger —
XR_KFPAR.

THE corridor fix, and the largest single VIO lever per the benchmark
analysis. Stock basalt's ONLY keyframe trigger is connectivity-ratio
(sqrt_keypoint_vio.cpp ~455): take a KF when the fraction of TRACKED
keypoints drops below vio_new_kf_keypoints_thresh. Under pure FORWARD
motion (corridors), features sit near the focus-of-expansion and stay
tracked -> connectivity stays high -> almost NO keyframes spawn -> the
sliding-window BASELINE collapses -> depth/scale along the optical axis
go unobservable -> the 50cm pure-VIO corridor drift (corridor2 50.6,
corridor3 49.8, the worst numbers in the fleet).

Fix (OKVIS2/DSO-standard disparity keyframing): add a SECOND trigger on
median pixel PARALLAX since the last keyframe. Track each cam0 keypoint's
pixel position at the last KF; each frame, take the median displacement
of the still-tracked keypoints; force a KF when it exceeds a threshold
even if connectivity is high. This manufactures the triangulation
baseline the connectivity rule refuses to. Near-zero regression risk on
well-parallaxed EuRoC (the trigger only ADDS keyframes when parallax is
genuinely high). Env-gated XR_KFPAR=<threshold_px> (0/unset = stock).
Applies after stage 13."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header: last-KF keypoint pixels + threshold helper ----------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = "  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;"
new = old + """

  /* stage 16 (XR_KFPAR): cam0 keypoint pixels at the last keyframe, for
   * the median-parallax keyframe trigger. */
  std::map<KeypointId, Eigen::Matrix<Scalar, 2, 1>> xr_last_kf_kps;
  static float xr_kfpar_px() {
    static const float v = [] {
      const char* e = getenv("XR_KFPAR");
      return e && *e ? (float)atof(e) : 0.0f;   /* 0 = disabled (stock) */
    }();
    return v;
  }"""
assert t.count(old) == 1, "hdr anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp: parallax trigger before the connectivity check ---------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")
old = """  if (Scalar(connected[0]) / (connected[0] + unconnected_obs[0].size()) < Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;"""
new = """  if (Scalar(connected[0]) / (connected[0] + unconnected_obs[0].size()) < Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;

  /* stage 16 (XR_KFPAR): parallax trigger. Median pixel displacement of
   * still-tracked cam0 keypoints since the last keyframe; force a KF when
   * the baseline is triangulation-worthy even if connectivity stays high
   * (the pure-forward-motion corridor case the connectivity rule misses). */
  if (xr_kfpar_px() > 0 && !take_kf && !xr_last_kf_kps.empty() &&
      frames_after_kf > config.vio_min_frames_after_kf) {
    std::vector<Scalar> xr_par;
    xr_par.reserve(opt_flow_meas->keypoints[0].size());
    for (const auto& kv : opt_flow_meas->keypoints[0]) {
      auto it = xr_last_kf_kps.find(kv.first);
      if (it == xr_last_kf_kps.end()) continue;
      const Eigen::Matrix<Scalar, 2, 1> cur =
          kv.second.translation().template cast<Scalar>();
      xr_par.push_back((cur - it->second).norm());
    }
    if (xr_par.size() >= 8) {
      std::nth_element(xr_par.begin(), xr_par.begin() + xr_par.size() / 2,
                       xr_par.end());
      if (xr_par[xr_par.size() / 2] > Scalar(xr_kfpar_px())) take_kf = true;
    }
  }"""
assert t.count(old) == 1, "trigger anchor"
t = t.replace(old, new)

# ---- cpp: snapshot cam0 keypoints when a keyframe is taken -------------
old = """  if (take_kf) {
    // Triangulate new points from one of the observations (with sufficient
    // baseline) and make keyframe
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);"""
new = """  if (take_kf) {
    // Triangulate new points from one of the observations (with sufficient
    // baseline) and make keyframe
    take_kf = false;
    frames_after_kf = 0;
    kf_ids.emplace(last_state_t_ns);

    /* stage 16: reset the parallax reference to this keyframe's cam0 kps */
    if (xr_kfpar_px() > 0) {
      xr_last_kf_kps.clear();
      for (const auto& kv : opt_flow_meas->keypoints[0])
        xr_last_kf_kps[kv.first] =
            kv.second.translation().template cast<Scalar>();
    }"""
assert t.count(old) == 1, "snapshot anchor"
t = t.replace(old, new)
f.write_text(t, encoding="utf-8")
print("stage 16 (XR_KFPAR parallax keyframe trigger) applied")
