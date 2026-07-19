#!/usr/bin/env python3
"""Stage 17 (VIO estimator architecture): translation-distance keyframe
trigger — XR_KFDIST.

Companion to the parallax trigger (stage 16), for the regime where
parallax mis-fires: FAST forward/outdoor motion (the 4Seasons drives).
The unbounded-drive experiment showed the drive map stays at ~1500
keyframes REGARDLESS of the cap or the map distance gate, because the
VIO's connectivity-based take_kf fires only ~every 0.6s -> ~10-16m
keyframe spacing at driving speed -> too sparse for the session map to
relocalize (drive reloc 0-10%). A translation-distance take_kf fires
RELIABLY on the drives (translation accumulates monotonically, unlike
FoE parallax), densifying the VIO keyframes and thus the session map that
stores on them. XR_KFDIST=<metres> (0/unset = stock). Applies after stage
13 (raw take_kf structure)."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header ------------------------------------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = "  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;"
new = old + """

  /* stage 17 (XR_KFDIST): translation-distance keyframe trigger. */
  Eigen::Matrix<Scalar, 3, 1> xr_last_kf_pos = Eigen::Matrix<Scalar, 3, 1>::Zero();
  bool xr_last_kf_pos_set = false;
  static float xr_kfdist_m() {
    static const float v = [] {
      const char* e = getenv("XR_KFDIST");
      return e && *e ? (float)atof(e) : 0.0f;   /* 0 = disabled (stock) */
    }();
    return v;
  }"""
assert t.count(old) == 1, "hdr anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp: distance trigger after the connectivity check ----------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")
old = """  if (Scalar(connected[0]) / (connected[0] + unconnected_obs[0].size()) < Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;"""
new = """  if (Scalar(connected[0]) / (connected[0] + unconnected_obs[0].size()) < Scalar(config.vio_new_kf_keypoints_thresh) &&
      frames_after_kf > config.vio_min_frames_after_kf)
    take_kf = true;

  /* stage 17 (XR_KFDIST): force a keyframe when the platform has
   * translated more than the threshold since the last KF, even if
   * connectivity stays high — the fast-forward-motion (drive) case where
   * the connectivity rule under-samples and the map goes too sparse to
   * relocalize. */
  if (xr_kfdist_m() > 0 && !take_kf && xr_last_kf_pos_set &&
      frames_after_kf > config.vio_min_frames_after_kf &&
      !frame_states.empty()) {
    const Eigen::Matrix<Scalar, 3, 1> xr_pos =
        frame_states.rbegin()->second.getState().T_w_i.translation();
    if ((xr_pos - xr_last_kf_pos).norm() > Scalar(xr_kfdist_m()))
      take_kf = true;
  }"""
assert t.count(old) == 1, "trigger anchor"
t = t.replace(old, new)

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

    /* stage 17: reset the distance reference to this keyframe's position */
    if (xr_kfdist_m() > 0 && !frame_states.empty()) {
      xr_last_kf_pos = frame_states.rbegin()->second.getState().T_w_i.translation();
      xr_last_kf_pos_set = true;
    }"""
assert t.count(old) == 1, "snapshot anchor"
t = t.replace(old, new)
f.write_text(t, encoding="utf-8")
print("stage 17 (XR_KFDIST translation-distance keyframe trigger) applied")
