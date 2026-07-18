#!/usr/bin/env python3
"""Stage 8: XR_LMINJ — closure-landmark INJECTION into the live lmdb.

The per-sequence decomposition (site /analysis.html) shows the dominant
gap vs OKVIS2+LC is closure ABSORPTION (theirs 1.7-31.9x, ours 1.0-1.2x)
and the OKVIS2 mechanism is landmark reactivation: closures turn cached
observations back into live landmarks that participate in bundle
adjustment with correct linearization and marginalize through the
estimator's own path. Stage 8 is the tractable equivalent for Basalt:
every verified closure inlier becomes a REAL lmdb landmark (map 3D as
initialization, hosted at the newest keyframe) with streaming
observations added by the app's re-match channel (LMTRACK). The fixed-3D
factor channel (stages 3-7) stays on as the map anchor; the injected
landmarks add multi-frame structure consistency that fixed factors
cannot provide. Env-gated: XR_LMINJ.
Apply after stages 2-7."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- 1. vio_estimator.h: virtual default no-op --------------------------
f = W / "include/basalt/vi_estimator/vio_estimator.h"
t = f.read_text(encoding="utf-8")
old = """  virtual void setXrLandmarkFactors(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world, int n,
                                    float sigma_px) {
    (void)t_ns; (void)cam_id; (void)uv; (void)xyz_world; (void)n; (void)sigma_px;
  }"""
new = old + """

  /* XREAL stage-8: inject verified closure landmarks into the live lmdb
   * as optimizable landmarks with streaming observations (the OKVIS2
   * reactivation mechanism, gated upstream). Default: no-op. */
  virtual void setXrInjectLandmarks(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world,
                                    const int32_t* ids, int n) {
    (void)t_ns; (void)cam_id; (void)uv; (void)xyz_world; (void)ids; (void)n;
  }"""
assert t.count(old) == 1, "vio_estimator anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 2. sqrt_keypoint_vio.h: queue + override ---------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = """  std::mutex xr_lm_mutex;
  std::vector<XrLmBatch> xr_lm_batches;"""
new = """  std::mutex xr_lm_mutex;
  std::vector<XrLmBatch> xr_lm_batches;

  /* XREAL stage-8 injection queue (consumed in measure()) */
  struct XrInjItem {
    int64_t t_ns; int cam_id; float u, v; float x, y, z; int32_t id;
  };
  std::mutex xr_inj_mutex;
  std::vector<XrInjItem> xr_inj_queue;
  void setXrInjectLandmarks(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world,
                            const int32_t* ids, int n) override {
    if (!uv || !xyz_world || !ids || n <= 0) return;
    std::lock_guard<std::mutex> l(xr_inj_mutex);
    if (xr_inj_queue.size() > 4096) return;      /* bounded backlog */
    for (int i = 0; i < n; i++)
      xr_inj_queue.push_back({t_ns, cam_id, uv[2 * i], uv[2 * i + 1],
                              xyz_world[3 * i], xyz_world[3 * i + 1],
                              xyz_world[3 * i + 2], ids[i]});
  }"""
assert t.count(old) == 1, "sqrt h anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 3. sqrt_keypoint_vio.cpp: consume in measure() ---------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")
old = """    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }"""
new = """    num_points_kf[opt_flow_meas->t_ns] = num_points_added;
  } else {
    frames_after_kf++;
  }

  /* XREAL stage-8 (XR_LMINJ): consume queued closure-landmark
   * injections. A verified closure inlier becomes a REAL lmdb landmark
   * hosted at the newest keyframe, initialized at the map 3D, with a
   * streaming observation on the posting frame; the app's re-match
   * channel keeps adding observations on later frames. The landmark
   * participates in BA and marginalizes through the estimator's own
   * path — arbitrated, correctly linearized, reversible until host
   * marginalization. */
  {
    static int xr_inj_on = -1;
    if (xr_inj_on < 0) {
      const char* e = getenv("XR_LMINJ");
      xr_inj_on = (e && *e && *e != '0') ? 1 : 0;
      if (xr_inj_on) std::cerr << "[xr] LMINJ landmark injection ON" << std::endl;
    }
    if (xr_inj_on) {
      std::vector<XrInjItem> xr_items;
      {
        std::lock_guard<std::mutex> l(xr_inj_mutex);
        xr_items.swap(xr_inj_queue);
      }
      int xr_made = 0, xr_obs = 0;
      for (const auto& xit : xr_items) {
        if (frame_states.count(xit.t_ns) == 0 && frame_poses.count(xit.t_ns) == 0)
          continue;                    /* posting frame left the window */
        const int xr_bid = 0x40000000 + xit.id;
        TimeCamId xr_tcid_obs(xit.t_ns, xit.cam_id);
        KeypointObservation<Scalar> xr_kobs;
        xr_kobs.kpt_id = xr_bid;
        xr_kobs.pos = Eigen::Matrix<Scalar, 2, 1>(Scalar(xit.u), Scalar(xit.v));
        if (lmdb.landmarkExists(xr_bid)) {
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
      }
      static int xr_inj_logn = 0;
      if ((xr_made || xr_obs) && xr_inj_logn < 40) {
        xr_inj_logn++;
        std::cerr << "[xr] LMINJ +" << xr_made << " landmarks, +" << xr_obs
                  << " obs (lmdb " << lmdb.numLandmarks() << ")" << std::endl;
      }
    }
  }"""
assert t.count(old) == 1, "cpp consume anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 4. vit_tracker.hpp: Tracker method decl ----------------------------
f = W / "include/basalt/vit/vit_tracker.hpp"
t = f.read_text(encoding="utf-8")
old = """  void xreal_set_landmark_factors(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world, int n,
                                  float sigma_px);"""
new = old + """

  /* XREAL stage-8: closure-landmark injection (ids pair with uv/xyz). */
  void xreal_inject_landmarks(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world, const int32_t *ids,
                              int n);"""
assert t.count(old) == 1, "vit hpp anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 5. vit_tracker.cpp: impl + C export --------------------------------
f = W / "src/vit/vit_tracker.cpp"
t = f.read_text(encoding="utf-8")
old = """void Tracker::xreal_set_landmark_factors(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world, int n,
                                         float sigma_px) {
  if (!impl_ || !impl_->vio) return;
  impl_->vio->setXrLandmarkFactors(t_ns, cam_id, uv, xyz_world, n, sigma_px);
}"""
new = old + """

void Tracker::xreal_inject_landmarks(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world,
                                     const int32_t *ids, int n) {
  if (!impl_ || !impl_->vio) return;
  impl_->vio->setXrInjectLandmarks(t_ns, cam_id, uv, xyz_world, ids, n);
}"""
assert t.count(old) == 1, "vit cpp impl anchor"
t = t.replace(old, new)
old = """extern "C" vit_result_t vit_tracker_xreal_landmark_factors(vit_tracker_t *tracker, int64_t t_ns, int32_t cam_id,
                                                           const float *uv, const float *xyz_world, int32_t n,
                                                           float sigma_px) {
  if (!tracker || !uv || !xyz_world || n <= 0) return VIT_ERROR_INVALID_VALUE;
  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));
  t->xreal_set_landmark_factors(t_ns, cam_id, uv, xyz_world, n, sigma_px);
  return VIT_SUCCESS;
}"""
new = old + """

/* XREAL stage-8 extension: closure-landmark injection into the lmdb. */
extern "C" vit_result_t vit_tracker_xreal_inject_landmarks(vit_tracker_t *tracker, int64_t t_ns, int32_t cam_id,
                                                           const float *uv, const float *xyz_world,
                                                           const int32_t *ids, int32_t n) {
  if (!tracker || !uv || !xyz_world || !ids || n <= 0) return VIT_ERROR_INVALID_VALUE;
  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));
  t->xreal_inject_landmarks(t_ns, cam_id, uv, xyz_world, ids, n);
  return VIT_SUCCESS;
}"""
assert t.count(old) == 1, "vit cpp export anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

print("stage 8 (XR_LMINJ closure-landmark injection) applied")
