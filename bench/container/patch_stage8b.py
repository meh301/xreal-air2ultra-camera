#!/usr/bin/env python3
"""Stage 8b: fix the 1-observation QR underflow in XR_LMINJ.

Basalt's LandmarkBlockAbsDynamic::performQR assumes >=2 observation rows
(flow landmarks are born from a triangulated pair); an injected landmark
with a single observation segfaults makeHouseholder. Buffer injections
per landmark until two observations from DISTINCT frames exist (both
still in the window), then create the landmark with both. Applies on top
of stage 8."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header: pending buffer member --------------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = """  std::mutex xr_inj_mutex;
  std::vector<XrInjItem> xr_inj_queue;"""
new = """  std::mutex xr_inj_mutex;
  std::vector<XrInjItem> xr_inj_queue;
  /* per-landmark pending obs until two distinct frames exist (QR needs
   * >=2 observation rows) */
  std::map<int, std::vector<XrInjItem>> xr_inj_pending;"""
assert t.count(old) == 1, "hdr pending anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp: buffered creation ---------------------------------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")
old = """        if (lmdb.landmarkExists(xr_bid)) {
          if (lmdb.getLandmark(xr_bid).obs.count(xr_tcid_obs) == 0) {
            lmdb.addObservation(xr_tcid_obs, xr_kobs);
            xr_obs++;
          }
          continue;
        }
        if (kf_ids.empty()) continue;
        const int64_t xr_host = *kf_ids.rbegin();  /* newest keyframe */
        if (frame_states.count(xr_host) == 0 && frame_poses.count(xr_host) == 0)
          continue;
        SE3 xr_T_w_c = getPoseStateWithLin(xr_host).getPose() * calib.T_i_c[xit.cam_id];
        Vec3 xr_pc = xr_T_w_c.inverse() * Vec3(Scalar(xit.x), Scalar(xit.y), Scalar(xit.z));
        const Scalar xr_d = xr_pc.norm();
        if (!(xr_d > Scalar(0.3) && xr_d < Scalar(60.0))) continue;
        Vec4 xr_p4;
        xr_p4 << xr_pc / xr_d, Scalar(1.0) / xr_d;
        Landmark<Scalar> xr_lm;
        xr_lm.host_kf_id = TimeCamId(xr_host, xit.cam_id);
        xr_lm.direction = StereographicParam<Scalar>::project(xr_p4);
        xr_lm.inv_dist = xr_p4[3];
        xr_lm.id = xr_bid;
        lmdb.addLandmark(xr_bid, xr_lm);
        lmdb.addObservation(xr_tcid_obs, xr_kobs);
        xr_made++;
        xr_obs++;
      }"""
new = """        if (lmdb.landmarkExists(xr_bid)) {
          if (lmdb.getLandmark(xr_bid).obs.count(xr_tcid_obs) == 0) {
            lmdb.addObservation(xr_tcid_obs, xr_kobs);
            xr_obs++;
          }
          continue;
        }
        /* buffer until TWO observations from distinct frames exist —
         * the QR landmark block needs >=2 observation rows */
        auto& xr_pend = xr_inj_pending[xit.id];
        bool xr_dup = false;
        for (const auto& pe : xr_pend)
          if (pe.t_ns == xit.t_ns) { xr_dup = true; break; }
        if (!xr_dup) xr_pend.push_back(xit);
        /* prune entries whose frames left the window */
        xr_pend.erase(std::remove_if(xr_pend.begin(), xr_pend.end(),
            [&](const XrInjItem& pe) {
              return frame_states.count(pe.t_ns) == 0 &&
                     frame_poses.count(pe.t_ns) == 0;
            }), xr_pend.end());
        if (xr_pend.size() < 2) continue;
        if (kf_ids.empty()) continue;
        const int64_t xr_host = *kf_ids.rbegin();  /* newest keyframe */
        if (frame_states.count(xr_host) == 0 && frame_poses.count(xr_host) == 0)
          continue;
        SE3 xr_T_w_c = getPoseStateWithLin(xr_host).getPose() * calib.T_i_c[xit.cam_id];
        Vec3 xr_pc = xr_T_w_c.inverse() * Vec3(Scalar(xit.x), Scalar(xit.y), Scalar(xit.z));
        const Scalar xr_d = xr_pc.norm();
        if (!(xr_d > Scalar(0.3) && xr_d < Scalar(60.0))) { xr_inj_pending.erase(xit.id); continue; }
        Vec4 xr_p4;
        xr_p4 << xr_pc / xr_d, Scalar(1.0) / xr_d;
        Landmark<Scalar> xr_lm;
        xr_lm.host_kf_id = TimeCamId(xr_host, xit.cam_id);
        xr_lm.direction = StereographicParam<Scalar>::project(xr_p4);
        xr_lm.inv_dist = xr_p4[3];
        xr_lm.id = xr_bid;
        lmdb.addLandmark(xr_bid, xr_lm);
        for (const auto& pe : xr_pend) {
          KeypointObservation<Scalar> xr_po;
          xr_po.kpt_id = xr_bid;
          xr_po.pos = Eigen::Matrix<Scalar, 2, 1>(Scalar(pe.u), Scalar(pe.v));
          lmdb.addObservation(TimeCamId(pe.t_ns, pe.cam_id), xr_po);
          xr_obs++;
        }
        xr_inj_pending.erase(xit.id);
        xr_made++;
      }
      /* global pending hygiene: drop landmarks whose buffered frames all
       * left the window (bounded memory) */
      if (xr_inj_pending.size() > 2048) xr_inj_pending.clear();"""
assert t.count(old) == 1, "cpp buffered anchor"
t = t.replace(old, new)
# <algorithm> for remove_if — ensure present
if "#include <algorithm>" not in t:
    t = t.replace("#include <basalt", "#include <algorithm>\n#include <basalt", 1)
f.write_text(t, encoding="utf-8")
print("stage 8b (buffered two-obs creation) applied")
