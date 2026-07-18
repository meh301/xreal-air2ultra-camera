#!/usr/bin/env python3
"""Stage 8d: per-frame covisibility matching for injected landmarks.

8c exposed the structural mismatch: the app's re-match stream (map
thread, 1-2 Hz) is far sparser than basalt's ~150 ms non-kf frame
lifetime, so pending observations died before any take_kf could create
a landmark (zero injections end-to-end). The OKVIS2 answer is
detection-by-covisibility: the ESTIMATOR matches map landmarks against
live features every frame. 8d builds exactly that: each injected anchor
(odom-frame 3D) is projected into every frame; a flow keypoint within
3 px is an observation. Creation happens on take_kf measures hosted at
the current frame with the last two matches; afterwards observations
accumulate at frame rate and the landmark lives basalt's natural
BA/marginalization life. Applies on top of 8+8b+8c."""
import re
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header: anchor registry --------------------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = """  /* per-landmark pending obs until two distinct frames exist (QR needs
   * >=2 observation rows) */
  std::map<int, std::vector<XrInjItem>> xr_inj_pending;"""
new = """  /* per-landmark pending obs until two distinct frames exist (QR needs
   * >=2 observation rows) */
  std::map<int, std::vector<XrInjItem>> xr_inj_pending;
  /* stage-8d anchor registry: odom-frame 3D per injected map landmark,
   * matched against live flow keypoints EVERY frame (covisibility) */
  struct XrInjLm {
    Scalar x, y, z;
    int created;
    int nm;
    int64_t m_ns[2];
    Scalar m_uv[2][2];
  };
  std::map<int, XrInjLm> xr_inj_lms;"""
assert t.count(old) == 1, "hdr 8d anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp: replace the whole consume body --------------------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

pat = re.compile(
    r"      std::vector<XrInjItem> xr_items;.*?"
    r"std::cerr << \"\[xr\] LMINJ \+\" << xr_made << \" landmarks, \+\" << xr_obs\n"
    r"                  << \" obs \(lmdb \" << lmdb\.numLandmarks\(\) << \"\)\" << std::endl;\n"
    r"      \}",
    re.S)
new_body = """      /* drain posts -> anchor registry (the app tells us WHICH map
       * landmarks matter and WHERE they are; observation now happens
       * here, every frame, against the live flow keypoints) */
      std::vector<XrInjItem> xr_items;
      {
        std::lock_guard<std::mutex> l(xr_inj_mutex);
        xr_items.swap(xr_inj_queue);
      }
      for (const auto& xit : xr_items) {
        auto& e = xr_inj_lms[xit.id];
        e.x = Scalar(xit.x);
        e.y = Scalar(xit.y);
        e.z = Scalar(xit.z);
      }
      if (xr_inj_lms.size() > 512) {       /* bounded: shed pending first */
        for (auto it = xr_inj_lms.begin();
             it != xr_inj_lms.end() && xr_inj_lms.size() > 384;)
          it = (!it->second.created) ? xr_inj_lms.erase(it) : std::next(it);
      }
      /* per-frame covisibility matcher */
      const int64_t xr_now = opt_flow_meas->t_ns;
      int xr_obs = 0, xr_made = 0;
      if ((frame_states.count(xr_now) || frame_poses.count(xr_now)) &&
          !xr_inj_lms.empty()) {
        SE3 xr_T_c_w =
            (getPoseStateWithLin(xr_now).getPose() * calib.T_i_c[0]).inverse();
        for (auto it = xr_inj_lms.begin(); it != xr_inj_lms.end();) {
          auto& e = it->second;
          const int xr_bid = 0x40000000 + it->first;
          if (e.created && !lmdb.landmarkExists(xr_bid)) {
            it = xr_inj_lms.erase(it);     /* marginalized out: done */
            continue;
          }
          Vec3 xr_pc = xr_T_c_w * Vec3(e.x, e.y, e.z);
          const Scalar xr_d = xr_pc.norm();
          if (!(xr_d > Scalar(0.3) && xr_d < Scalar(60.0))) { ++it; continue; }
          Vec2 xr_pr;
          if (!calib.intrinsics[0].project(xr_pc, xr_pr)) { ++it; continue; }
          Scalar xr_best = Scalar(9.0);    /* (3 px)^2 gate */
          Vec2 xr_uv;
          bool xr_hit = false;
          for (const auto& kv : opt_flow_meas->keypoints.at(0)) {
            Vec2 kp = kv.second.translation().template cast<Scalar>();
            const Scalar d2 = (kp - xr_pr).squaredNorm();
            if (d2 < xr_best) { xr_best = d2; xr_uv = kp; xr_hit = true; }
          }
          if (!xr_hit) { ++it; continue; }
          if (e.created) {
            auto& lm = lmdb.getLandmark(xr_bid);
            if (lm.obs.count(TimeCamId(xr_now, 0)) == 0) {
              KeypointObservation<Scalar> ko;
              ko.kpt_id = xr_bid;
              ko.pos = xr_uv;
              lmdb.addObservation(TimeCamId(xr_now, 0), ko);
              xr_obs++;
            }
          } else {
            const int s = e.nm & 1;
            if (e.nm == 0 || e.m_ns[(e.nm - 1) & 1] != xr_now) {
              e.m_ns[s] = xr_now;
              e.m_uv[s][0] = xr_uv(0);
              e.m_uv[s][1] = xr_uv(1);
              e.nm++;
            }
            const int s0 = (e.nm - 1) & 1, s1 = e.nm & 1;
            if (kf_ids.count(xr_now) > 0 && e.nm >= 2 && e.m_ns[s0] == xr_now &&
                (frame_states.count(e.m_ns[s1]) ||
                 frame_poses.count(e.m_ns[s1]))) {
              Vec4 xr_p4;
              xr_p4 << xr_pc / xr_d, Scalar(1.0) / xr_d;
              Landmark<Scalar> xr_lm;
              xr_lm.host_kf_id = TimeCamId(xr_now, 0);
              xr_lm.direction = StereographicParam<Scalar>::project(xr_p4);
              xr_lm.inv_dist = xr_p4[3];
              xr_lm.id = xr_bid;
              lmdb.addLandmark(xr_bid, xr_lm);
              KeypointObservation<Scalar> k0;
              k0.kpt_id = xr_bid;
              k0.pos = Vec2(e.m_uv[s0][0], e.m_uv[s0][1]);
              lmdb.addObservation(TimeCamId(e.m_ns[s0], 0), k0);
              KeypointObservation<Scalar> k1;
              k1.kpt_id = xr_bid;
              k1.pos = Vec2(e.m_uv[s1][0], e.m_uv[s1][1]);
              lmdb.addObservation(TimeCamId(e.m_ns[s1], 0), k1);
              e.created = 1;
              xr_made++;
            }
          }
          ++it;
        }
      }
      static int xr_inj_logn = 0;
      if ((xr_made || xr_obs) && xr_inj_logn < 40) {
        xr_inj_logn++;
        std::cerr << "[xr] LMINJ +" << xr_made << " landmarks, +" << xr_obs
                  << " obs (lmdb " << lmdb.numLandmarks() << ")" << std::endl;
      }"""
t2, n = pat.subn(new_body, t)
assert n == 1, f"cpp 8d subn: {n}"
f.write_text(t2, encoding="utf-8")
print("stage 8d (per-frame covisibility matching) applied")
