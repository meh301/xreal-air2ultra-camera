#!/usr/bin/env python3
"""Apply the XR STAGE-3 landmark-factor edits (setXrLandmarkFactors /
vit_tracker_xreal_landmark_factors) to the container's basalt clone.
Idempotent; follows the patch_tight.py pattern.

The stage-3 hunks anchor on the stage-2 pose-prior hunks, so this script
FIRST applies the pose-prior edits when they are missing (subsumes
patch_tight.py — running both in either order is safe: every hunk is
guarded by a presence check). Mirrors the commits on the app-side basalt
clone (branch xr-lmfact in android/app/src/main/cpp/third_party/basalt).

Usage: patch_lmfact.py [path-to-basalt-clone]
"""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/xreal/bench/container/basalt-linux")
done = []


def read(rel):
    return (W / rel).read_text(encoding="utf-8")


def write(rel, t):
    (W / rel).write_text(t, encoding="utf-8", newline="\n")


def insert(rel, anchor, text, guard, before=False):
    """Insert text next to anchor unless guard is already present."""
    t = read(rel)
    if guard in t:
        done.append(f"{rel}: already patched ({guard})")
        return
    assert anchor in t, f"{rel}: anchor not found: {anchor[:60]!r}"
    t = t.replace(anchor, (text + anchor) if before else (anchor + text), 1)
    write(rel, t)
    done.append(f"{rel}: patched ({guard})")


# ======================= stage 2: pose prior (prereq) ======================
# Same edits as patch_tight.py — applied only when missing.

insert("include/basalt/vi_estimator/vio_estimator.h",
       "virtual void takeLongTermKeyframe() {};",
       """

  /* XREAL map->VIO tight coupling: weak unary SE(3) pose prior request on
   * the newest state. E = world-frame left correction (target = E * T);
   * expiry compares against state timestamps. Default: unsupported no-op. */
  virtual void setXrPosePrior(const Sophus::SE3d& E, double sigma_t, double sigma_r, int64_t expiry_t_ns) {
    (void)E; (void)sigma_t; (void)sigma_r; (void)expiry_t_ns;
  }
""", guard="setXrPosePrior")

t = read("include/basalt/vi_estimator/sqrt_keypoint_vio.h")
if "xr_prior_mutex" not in t:
    assert "#include <thread>" in t
    t = t.replace("#include <thread>", "#include <mutex>\n#include <thread>", 1)
    anchor = "void takeLongTermKeyframe() override;"
    assert anchor in t
    t = t.replace(anchor, anchor + """

  /* XREAL map->VIO tight coupling (see VioEstimatorBase). Thread-safe. */
  void setXrPosePrior(const Sophus::SE3d& E, double sigma_t, double sigma_r, int64_t expiry_t_ns) override {
    std::lock_guard<std::mutex> l(xr_prior_mutex);
    xr_prior_E = E;
    xr_prior_sigma_t = sigma_t;
    xr_prior_sigma_r = sigma_r;
    xr_prior_expiry = expiry_t_ns;
    xr_prior_active = true;
  }
  std::mutex xr_prior_mutex;
  Sophus::SE3d xr_prior_E;
  double xr_prior_sigma_t = 0.07, xr_prior_sigma_r = 0.035;
  int64_t xr_prior_expiry = 0;
  bool xr_prior_active = false;
""", 1)
    write("include/basalt/vi_estimator/sqrt_keypoint_vio.h", t)
    done.append("sqrt_keypoint_vio.h: pose-prior patched")
else:
    done.append("sqrt_keypoint_vio.h: pose-prior already present")

t = read("src/vi_estimator/sqrt_keypoint_vio.cpp")
if "xr_prior_active" not in t:
    anchor = 'stats.add("get_dense_H_b", t.reset()).format("ms");'
    assert anchor in t
    t = t.replace(anchor, """/* XREAL map->VIO tight coupling: weak unary SE(3) prior on the NEWEST
           * state, target = E * T. Decoupled increments make J = I, so the
           * prior lands as H += W, b += W * r on the state's pose block. */
          {
            std::lock_guard<std::mutex> lp(xr_prior_mutex);
            if (xr_prior_active && !frame_states.empty()) {
              const int64_t newest_ts = frame_states.rbegin()->first;
              if (newest_ts > xr_prior_expiry) {
                xr_prior_active = false;
              } else if (aom.abs_order_map.count(newest_ts)) {
                const auto& [idx, bsize] = aom.abs_order_map.at(newest_ts);
                UNUSED(bsize);
                const SE3 T_cur = frame_states.at(newest_ts).getState().T_w_i;
                const SE3 E_c = xr_prior_E.template cast<Scalar>();
                const SE3 Tt = E_c * T_cur;
                const Vec3 rt = T_cur.translation() - Tt.translation();
                const Vec3 rr = (T_cur.so3() * Tt.so3().inverse()).log();
                const Scalar wt = Scalar(1.0) / Scalar(xr_prior_sigma_t * xr_prior_sigma_t);
                const Scalar wr = Scalar(1.0) / Scalar(xr_prior_sigma_r * xr_prior_sigma_r);
                for (int c = 0; c < 3; c++) {
                  H(idx + c, idx + c) += wt;
                  b(idx + c) += wt * rt(c);
                  H(idx + 3 + c, idx + 3 + c) += wr;
                  b(idx + 3 + c) += wr * rr(c);
                }
              }
            }
          }

          """ + anchor, 1)
    write("src/vi_estimator/sqrt_keypoint_vio.cpp", t)
    done.append("sqrt_keypoint_vio.cpp: pose-prior patched")
else:
    done.append("sqrt_keypoint_vio.cpp: pose-prior already present")

insert("include/basalt/vit/vit_tracker.hpp",
       " private:\n  struct Implementation;\n  std::unique_ptr<Implementation> impl_;\n};",
       """  /* XREAL map->VIO tight coupling: forward a weak pose prior. */
  void xreal_set_pose_prior(const double q_xyzw[4], const double p[3], double sigma_t, double sigma_r,
                            int64_t expiry_t_ns);

""", guard="xreal_set_pose_prior", before=True)

t = read("src/vit/vit_tracker.cpp")
if "xreal_set_pose_prior" not in t:
    anchor = "}  // namespace basalt::vit_implementation"
    assert anchor in t
    t = t.replace(anchor, """void Tracker::xreal_set_pose_prior(const double q_xyzw[4], const double p[3], double sigma_t, double sigma_r,
                                   int64_t expiry_t_ns) {
  if (!impl_ || !impl_->vio) return;
  Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]);
  q.normalize();
  Sophus::SE3d E(q, Eigen::Vector3d(p[0], p[1], p[2]));
  impl_->vio->setXrPosePrior(E, sigma_t, sigma_r, expiry_t_ns);
}

}  // namespace basalt::vit_implementation

/* XREAL extension: reachable via dlsym, no header coupling required. */
extern "C" vit_result_t vit_tracker_xreal_pose_prior(vit_tracker_t *tracker, const double q_xyzw[4],
                                                     const double p[3], double sigma_t, double sigma_r,
                                                     int64_t expiry_t_ns) {
  if (!tracker) return VIT_ERROR_INVALID_VALUE;
  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));
  t->xreal_set_pose_prior(q_xyzw, p, sigma_t, sigma_r, expiry_t_ns);
  return VIT_SUCCESS;
}
""", 1)
    write("src/vit/vit_tracker.cpp", t)
    done.append("vit_tracker.cpp: pose-prior patched")
else:
    done.append("vit_tracker.cpp: pose-prior already present")

# ========================= stage 3: landmark factors =======================

# 1. base virtual (anchors on the stage-2 virtual)
insert("include/basalt/vi_estimator/vio_estimator.h",
       """  virtual void setXrPosePrior(const Sophus::SE3d& E, double sigma_t, double sigma_r, int64_t expiry_t_ns) {
    (void)E; (void)sigma_t; (void)sigma_r; (void)expiry_t_ns;
  }
""",
       """
  /* XREAL map->VIO tight coupling, stage 3: re-observed MAP LANDMARKS as
   * reprojection factors with FIXED 3D (no landmark state — unary factors on
   * the frame at t_ns). uv = 2n pixels in cam_id, xyz_world = 3n world-frame
   * (odometry world) metres, sigma_px = measurement std. The optimizer
   * arbitrates each point against the IMU/vision factors (the OKVIS2
   * mechanism). Default: unsupported no-op. */
  virtual void setXrLandmarkFactors(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world, int n,
                                    float sigma_px) {
    (void)t_ns; (void)cam_id; (void)uv; (void)xyz_world; (void)n; (void)sigma_px;
  }
""", guard="setXrLandmarkFactors")

# 2. sqrt header: includes + buffered batches + thread-safe setter
t = read("include/basalt/vi_estimator/sqrt_keypoint_vio.h")
if "xr_lm_batches" not in t:
    assert "#include <mutex>\n#include <thread>" in t, \
        "sqrt_keypoint_vio.h: expected the stage-2 include block"
    t = t.replace("#include <mutex>\n#include <thread>",
                  "#include <cstring>\n#include <mutex>\n#include <thread>\n#include <vector>", 1)
    anchor = "bool xr_prior_active = false;"
    assert anchor in t, "sqrt_keypoint_vio.h: stage-2 pose-prior members missing"
    t = t.replace(anchor, anchor + """

  /* XREAL stage-3 tight coupling (see VioEstimatorBase): fixed-3D map
   * landmark reprojection factors, buffered per frame timestamp like the
   * pose priors. Thread-safe setter; applied in optimize() while the frame
   * is in the window, expired by frame time. */
  static constexpr int XR_LM_MAX_FACTORS = 96;          /* per frame */
  static constexpr int XR_LM_MAX_BATCHES = 24;          /* kf batches persist */
  static constexpr int64_t XR_LM_EXPIRY_NS = 500000000; /* ~0.5 s */
  struct XrLmBatch {
    int64_t t_ns;
    int cam_id;
    int n;
    float sigma_px;
    float uv[2 * XR_LM_MAX_FACTORS];
    float xyz[3 * XR_LM_MAX_FACTORS];
  };
  void setXrLandmarkFactors(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world, int n,
                            float sigma_px) override {
    if (!uv || !xyz_world || n <= 0) return;
    if (n > XR_LM_MAX_FACTORS) n = XR_LM_MAX_FACTORS;
    std::lock_guard<std::mutex> l(xr_lm_mutex);
    XrLmBatch* slot = nullptr;
    for (auto& bch : xr_lm_batches)  /* re-post for the same frame+cam replaces */
      if (bch.t_ns == t_ns && bch.cam_id == cam_id) slot = &bch;
    if (!slot) {
      if (xr_lm_batches.size() >= XR_LM_MAX_BATCHES) xr_lm_batches.erase(xr_lm_batches.begin());
      xr_lm_batches.emplace_back();
      slot = &xr_lm_batches.back();
    }
    slot->t_ns = t_ns;
    slot->cam_id = cam_id;
    slot->n = n;
    slot->sigma_px = sigma_px > 0 ? sigma_px : 2.0f;
    std::memcpy(slot->uv, uv, sizeof(float) * 2 * (size_t)n);
    std::memcpy(slot->xyz, xyz_world, sizeof(float) * 3 * (size_t)n);
  }
  std::mutex xr_lm_mutex;
  std::vector<XrLmBatch> xr_lm_batches;
""", 1)
    write("include/basalt/vi_estimator/sqrt_keypoint_vio.h", t)
    done.append("sqrt_keypoint_vio.h: lmfact patched")
else:
    done.append("sqrt_keypoint_vio.h: lmfact already present")

# 3. optimize() application (before the same stats anchor stage 2 used, so it
# lands right after the pose-prior block)
t = read("src/vi_estimator/sqrt_keypoint_vio.cpp")
if "xr_lm_batches" not in t:
    anchor = 'stats.add("get_dense_H_b", t.reset()).format("ms");'
    assert anchor in t
    t = t.replace(anchor, """/* XREAL stage-3 tight coupling: re-observed MAP LANDMARK
           * reprojection factors with FIXED 3D — unary factors on the
           * observing frame's pose (no landmark state; the map's 3D is
           * authoritative). Under the decoupled increment convention
           * (t += dt, R <- exp(phi) R), with v = p_w - t_wi:
           *   p_c(dt,phi) = R_ic^T (R_wi^T exp(-phi) (v - dt) - t_ic)
           *   d p_c / d [dt|phi] = R_cw [-I | hat(v)],  R_cw = (R_wi R_ic)^T
           *   J = d_proj_d_pc(2x3) * that (2x6), residual r = proj - uv.
           * Huber-robustified (map points can be wrong) — each point is
           * arbitrated by the optimizer against IMU/vision factors, which
           * is what lets closures at ANY revisit age contribute (the
           * OKVIS2 mechanism the pose-prior channel cannot reproduce). */
          {
            std::lock_guard<std::mutex> lg_lm(xr_lm_mutex);
            if (!xr_lm_batches.empty() && !frame_states.empty()) {
              const int64_t xr_newest_ts = frame_states.rbegin()->first;
              for (size_t xbi = 0; xbi < xr_lm_batches.size();) {
                const XrLmBatch& bch = xr_lm_batches[xbi];
                const bool xr_alive = frame_states.count(bch.t_ns) > 0 ||
                                      frame_poses.count(bch.t_ns) > 0;
                if (!xr_alive && bch.t_ns + XR_LM_EXPIRY_NS < xr_newest_ts) {
                  /* frame gone AND stale: the post lost the state-window
                   * race — drop. Batches on LIVE keyframes persist for
                   * seconds and FOLD at kf-marginalization (XR_LMMARG,
                   * the OKVIS2 anchoring). */
                  xr_lm_batches.erase(xr_lm_batches.begin() + xbi);
                  continue;
                }
                ++xbi;
                if (bch.cam_id < 0 || bch.cam_id >= (int)calib.T_i_c.size()) continue;
                if (aom.abs_order_map.count(bch.t_ns) == 0) continue;
                const auto& [xr_idx, xr_size] = aom.abs_order_map.at(bch.t_ns);
                UNUSED(xr_size);
                SE3 T_w_i_lm;
                if (frame_states.count(bch.t_ns) > 0) {
                  T_w_i_lm = frame_states.at(bch.t_ns).getState().T_w_i;
                } else if (frame_poses.count(bch.t_ns) > 0) {
                  T_w_i_lm = frame_poses.at(bch.t_ns).getPose();
                } else {
                  continue;
                }
                const SE3& T_i_c = calib.T_i_c[bch.cam_id];
                const Eigen::Matrix<Scalar, 3, 3> R_cw = (T_w_i_lm.so3() * T_i_c.so3()).inverse().matrix();
                const SE3 T_c_w = (T_w_i_lm * T_i_c).inverse();
                const Scalar w_meas = Scalar(1.0) / Scalar(bch.sigma_px * bch.sigma_px);
                const Scalar huber_px = Scalar(3.0) * Scalar(bch.sigma_px);
                const Scalar cutoff_px = Scalar(20.0) * Scalar(bch.sigma_px);
                Eigen::Matrix<Scalar, 6, 6> H_lm;
                H_lm.setZero();
                Eigen::Matrix<Scalar, 6, 1> b_lm;
                b_lm.setZero();
                int applied = 0;
                for (int k = 0; k < bch.n; k++) {
                  const Vec3 p_w(Scalar(bch.xyz[3 * k]), Scalar(bch.xyz[3 * k + 1]), Scalar(bch.xyz[3 * k + 2]));
                  const Vec3 p_c = T_c_w * p_w;
                  Vec4 p_c4;
                  p_c4 << p_c(0), p_c(1), p_c(2), Scalar(1);
                  Vec2 proj;
                  Eigen::Matrix<Scalar, 2, 4> d_proj_d_p3d;
                  if (!calib.intrinsics[bch.cam_id].project(p_c4, proj, &d_proj_d_p3d)) continue;
                  const Vec2 r(proj(0) - Scalar(bch.uv[2 * k]), proj(1) - Scalar(bch.uv[2 * k + 1]));
                  const Scalar rn = r.norm();
                  if (!(rn < cutoff_px)) continue;   /* gross outlier (or NaN) */
                  const Vec3 v_wi = p_w - T_w_i_lm.translation();
                  Eigen::Matrix<Scalar, 3, 6> d_pc_d_xi;
                  d_pc_d_xi.template leftCols<3>() = -R_cw;
                  d_pc_d_xi.template rightCols<3>() = R_cw * Sophus::SO3<Scalar>::hat(v_wi);
                  const Eigen::Matrix<Scalar, 2, 6> J_lm = d_proj_d_p3d.template leftCols<3>() * d_pc_d_xi;
                  const Scalar w_huber = rn <= huber_px ? Scalar(1) : huber_px / rn;
                  const Scalar w = w_meas * w_huber;
                  H_lm.noalias() += w * (J_lm.transpose() * J_lm);
                  b_lm.noalias() += w * (J_lm.transpose() * r);
                  applied++;
                }
                if (applied > 0) {
                  H.template block<6, 6>(xr_idx, xr_idx) += H_lm;
                  b.template segment<6>(xr_idx) += b_lm;
                  static bool xr_lm_announced = false;
                  if (!xr_lm_announced) {
                    xr_lm_announced = true;
                    std::cerr << "[xr] LMFACT active: " << applied << "/" << bch.n
                              << " fixed-3D landmark factors applied to frame " << bch.t_ns << std::endl;
                  }
                }
              }
            }
          }

          """ + anchor, 1)
    write("src/vi_estimator/sqrt_keypoint_vio.cpp", t)
    done.append("sqrt_keypoint_vio.cpp: lmfact patched")
else:
    done.append("sqrt_keypoint_vio.cpp: lmfact already present")

# 4. vit_tracker.hpp: method decl
insert("include/basalt/vit/vit_tracker.hpp",
       " private:\n  struct Implementation;\n  std::unique_ptr<Implementation> impl_;\n};",
       """  /* XREAL stage-3 tight coupling: fixed-3D map landmark reprojection
   * factors for frame t_ns (uv = 2n pixels in cam_id, xyz_world = 3n
   * odometry-world metres, sigma_px = measurement std). */
  void xreal_set_landmark_factors(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world, int n,
                                  float sigma_px);

""", guard="xreal_set_landmark_factors", before=True)

# 5. vit_tracker.cpp: member impl + C export
t = read("src/vit/vit_tracker.cpp")
if "xreal_set_landmark_factors" not in t:
    anchor = "}  // namespace basalt::vit_implementation"
    assert anchor in t
    t = t.replace(anchor, """void Tracker::xreal_set_landmark_factors(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world, int n,
                                         float sigma_px) {
  if (!impl_ || !impl_->vio) return;
  impl_->vio->setXrLandmarkFactors(t_ns, cam_id, uv, xyz_world, n, sigma_px);
}

}  // namespace basalt::vit_implementation

/* XREAL stage-3 extension: fixed-3D map landmark reprojection factors. */
extern "C" vit_result_t vit_tracker_xreal_landmark_factors(vit_tracker_t *tracker, int64_t t_ns, int32_t cam_id,
                                                           const float *uv, const float *xyz_world, int32_t n,
                                                           float sigma_px) {
  if (!tracker || !uv || !xyz_world || n <= 0) return VIT_ERROR_INVALID_VALUE;
  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));
  t->xreal_set_landmark_factors(t_ns, cam_id, uv, xyz_world, n, sigma_px);
  return VIT_SUCCESS;
}
""", 1)
    write("src/vit/vit_tracker.cpp", t)
    done.append("vit_tracker.cpp: lmfact patched")
else:
    done.append("vit_tracker.cpp: lmfact already present")



# ==================== stage 4: marg-persistent factors =====================
# Without this, factor information EVAPORATES when its frame marginalizes
# (applied in optimize() only). OKVIS2 persists reobservation info through
# marginalization — the rooms-regime mechanism. Env-gated: XR_LMMARG.

t = read("src/vi_estimator/sqrt_keypoint_vio.cpp")
if "XR_LMMARG" not in t:
    anchor = """      if (is_lin_sqrt && marg_data.is_sqrt) {
        lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
      } else {
        lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);
      }"""
    assert t.count(anchor) >= 1, "marg get_dense anchor missing"
    inject = anchor + """

      /* XREAL stage-4 (XR_LMMARG): persist landmark-factor information
       * through marginalization. Batches attached to frames being REMOVED
       * fold into the marg linearization here — as extra residual rows in
       * sqrt mode, as an H/b block otherwise — so the map's pull survives
       * the frame instead of evaporating with it. */
      {
        static int xr_lmmarg_on = -1;
        if (xr_lmmarg_on < 0) {
          const char* e = getenv("XR_LMMARG");
          xr_lmmarg_on = (e && *e && *e != '0') ? 1 : 0;
          if (xr_lmmarg_on)
            std::cerr << "[xr] LMMARG marg-persistent landmark factors ON" << std::endl;
        }
        if (xr_lmmarg_on) {
          std::lock_guard<std::mutex> lg_lm(xr_lm_mutex);
          for (size_t xbi = 0; xbi < xr_lm_batches.size();) {
            const XrLmBatch& bch = xr_lm_batches[xbi];
            const bool being_removed =
                kfs_to_marg.count(bch.t_ns) > 0 || bch.t_ns == last_state_to_marg;
            if (!being_removed || aom.abs_order_map.count(bch.t_ns) == 0 ||
                bch.cam_id < 0 || bch.cam_id >= (int)calib.T_i_c.size()) {
              ++xbi;
              continue;
            }
            const auto& [xr_idx, xr_bsize] = aom.abs_order_map.at(bch.t_ns);
            UNUSED(xr_bsize);
            SE3 T_w_i_lm;
            if (frame_states.count(bch.t_ns) > 0) {
              T_w_i_lm = frame_states.at(bch.t_ns).getState().T_w_i;
            } else if (frame_poses.count(bch.t_ns) > 0) {
              T_w_i_lm = frame_poses.at(bch.t_ns).getPose();
            } else {
              ++xbi;
              continue;
            }
            const SE3& T_i_c = calib.T_i_c[bch.cam_id];
            const Eigen::Matrix<Scalar, 3, 3> R_cw =
                (T_w_i_lm.so3() * T_i_c.so3()).inverse().matrix();
            const SE3 T_c_w = (T_w_i_lm * T_i_c).inverse();
            const Scalar w_meas = Scalar(1.0) / Scalar(bch.sigma_px * bch.sigma_px);
            const Scalar huber_px = Scalar(3.0) * Scalar(bch.sigma_px);
            const Scalar cutoff_px = Scalar(20.0) * Scalar(bch.sigma_px);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 6> Jrows(2 * bch.n, 6);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rrows(2 * bch.n);
            int nr = 0;
            for (int k = 0; k < bch.n; k++) {
              const Vec3 p_w(Scalar(bch.xyz[3 * k]), Scalar(bch.xyz[3 * k + 1]),
                             Scalar(bch.xyz[3 * k + 2]));
              const Vec3 p_c = T_c_w * p_w;
              Vec4 p_c4;
              p_c4 << p_c(0), p_c(1), p_c(2), Scalar(1);
              Vec2 proj;
              Eigen::Matrix<Scalar, 2, 4> d_proj_d_p3d;
              if (!calib.intrinsics[bch.cam_id].project(p_c4, proj, &d_proj_d_p3d)) continue;
              const Vec2 r(proj(0) - Scalar(bch.uv[2 * k]),
                           proj(1) - Scalar(bch.uv[2 * k + 1]));
              const Scalar rn = r.norm();
              if (!(rn < cutoff_px)) continue;
              const Vec3 v_wi = p_w - T_w_i_lm.translation();
              Eigen::Matrix<Scalar, 3, 6> d_pc_d_xi;
              d_pc_d_xi.template leftCols<3>() = -R_cw;
              d_pc_d_xi.template rightCols<3>() = R_cw * Sophus::SO3<Scalar>::hat(v_wi);
              const Eigen::Matrix<Scalar, 2, 6> J_lm =
                  d_proj_d_p3d.template leftCols<3>() * d_pc_d_xi;
              const Scalar w_huber = rn <= huber_px ? Scalar(1) : huber_px / rn;
              const Scalar sw = std::sqrt(w_meas * w_huber);
              Jrows.template block<2, 6>(2 * nr, 0) = sw * J_lm;
              rrows.template segment<2>(2 * nr) = sw * r;
              nr++;
            }
            if (nr > 0) {
              if (is_lin_sqrt && marg_data.is_sqrt) {
                const Eigen::Index r0 = Q2Jp_or_H.rows();
                Q2Jp_or_H.conservativeResize(r0 + 2 * nr, Eigen::NoChange);
                Q2r_or_b.conservativeResize(r0 + 2 * nr);
                Q2Jp_or_H.bottomRows(2 * nr).setZero();
                Q2Jp_or_H.block(r0, xr_idx, 2 * nr, 6) = Jrows.topRows(2 * nr);
                Q2r_or_b.tail(2 * nr) = rrows.head(2 * nr);
              } else {
                Q2Jp_or_H.block(xr_idx, xr_idx, 6, 6) +=
                    Jrows.topRows(2 * nr).transpose() * Jrows.topRows(2 * nr);
                Q2r_or_b.segment(xr_idx, 6) +=
                    Jrows.topRows(2 * nr).transpose() * rrows.head(2 * nr);
              }
              static bool xr_lmmarg_announced = false;
              if (!xr_lmmarg_announced) {
                xr_lmmarg_announced = true;
                std::cerr << "[xr] LMMARG folded " << nr
                          << " landmark factors into the marg prior" << std::endl;
              }
              xr_lm_batches.erase(xr_lm_batches.begin() + xbi);
              continue;   /* consumed */
            }
            ++xbi;
          }
        }
      }"""
    t = t.replace(anchor, inject, 1)
    write("src/vi_estimator/sqrt_keypoint_vio.cpp", t)
    done.append("sqrt_keypoint_vio.cpp: lmmarg (stage 4) patched")
else:
    done.append("sqrt_keypoint_vio.cpp: lmmarg already present")

print("\n".join(done))
