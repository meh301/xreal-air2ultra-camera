#!/usr/bin/env python3
"""Stage 14 (forensic attack step 3): closure factors enter the LM
accept/reject test — XR_ABSORB.

THE corridor-gap fix. Today the map/closure factors (pose prior + fixed-3D
landmark reprojection) are added to the dense H,b (optimize(), ~1884-2009)
so the solved increment `inc` DOES include their pull — but neither
`error_total` (from lqr->linearizeProblem) nor `after_error_total`
(computeError + computeMargPriorError + computeImuError) includes their
residual. So f_diff = error_total - after_error_total measures only how
vision+imu+marg changed: when a closure pulls the pose to satisfy the
loop (raising the vision residual), f_diff goes negative and the LM step
is REJECTED. The optimizer structurally rejects exactly the meter-scale
corrections closures induce — the measured 1.0-1.2x absorption vs OKVIS2's
1.7-31.9x, and the corridor1-4 ATE gap vs OKVIS2+LC.

Fix: evaluate the SAME factor error (same residuals, weights, huber as the
H,b block) at the pre-step state (added to error_total once per outer
iteration) and the post-step state (added to after_error_total each
backtrack). f_diff then credits the factor-residual reduction and a
closure that lowers total cost is accepted. Env-gated XR_ABSORB (off =
bit-identical to today). Applies after stage 13."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header: decls -----------------------------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = "  void xrvReadvance();"
new = """  void xrvReadvance();

  /* stage 14: robust error of the applied XR map/closure factors at the
   * current state, scale-consistent with the dense H,b construction, so
   * the LM accept/reject test accounts for closures (XR_ABSORB). */
  Scalar_ xrAbsorbError();
  static bool xr_absorb_on() {
    static const bool v = [] {
      const char* e = getenv("XR_ABSORB");
      return e && *e && *e != '0';
    }();
    return v;
  }"""
assert t.count(old) == 1, "hdr anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp: function definition (before optimize) ------------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")
anchor = ("template <class Scalar_>\n"
          "bool SqrtKeypointVioEstimator<Scalar_>::optimize() {")
assert t.count(anchor) == 1, "optimize anchor"
defn = """template <class Scalar_>
Scalar_ SqrtKeypointVioEstimator<Scalar_>::xrAbsorbError() {
  Scalar err = 0;
  /* pose prior (mirrors the H,b block: r = T (-) E*T, err = 0.5 r' W r) */
  {
    std::lock_guard<std::mutex> lp(xr_prior_mutex);
    if (xr_prior_active && !frame_states.empty()) {
      const int64_t newest_ts = frame_states.rbegin()->first;
      if (newest_ts <= xr_prior_expiry &&
          frame_states.count(newest_ts)) {
        const SE3 T_cur = frame_states.at(newest_ts).getState().T_w_i;
        const SE3 E_c = xr_prior_E.template cast<Scalar>();
        const SE3 Tt = E_c * T_cur;
        const Vec3 rt = T_cur.translation() - Tt.translation();
        const Vec3 rr = (T_cur.so3() * Tt.so3().inverse()).log();
        const Scalar wt = Scalar(1.0) / Scalar(xr_prior_sigma_t * xr_prior_sigma_t);
        const Scalar wr = Scalar(1.0) / Scalar(xr_prior_sigma_r * xr_prior_sigma_r);
        err += Scalar(0.5) * (wt * rt.squaredNorm() + wr * rr.squaredNorm());
      }
    }
  }
  /* fixed-3D landmark reprojection factors (mirrors the H,b block: same
   * sigma decode, w_meas, huber/cauchy weight, cutoff; err += 0.5 w r^2
   * with the IRLS weight, consistent with b += w J' r) */
  {
    std::lock_guard<std::mutex> lg_lm(xr_lm_mutex);
    static int xr_cauchy = -1;
    if (xr_cauchy < 0) {
      const char* ce = getenv("XR_LM_CAUCHY");
      xr_cauchy = (ce && *ce && *ce != '0') ? 1 : 0;
    }
    for (const auto& bch : xr_lm_batches) {
      if (bch.cam_id < 0 || bch.cam_id >= (int)calib.T_i_c.size()) continue;
      SE3 T_w_i_lm;
      if (frame_states.count(bch.t_ns) > 0)
        T_w_i_lm = frame_states.at(bch.t_ns).getState().T_w_i;
      else if (frame_poses.count(bch.t_ns) > 0)
        T_w_i_lm = frame_poses.at(bch.t_ns).getPose();
      else
        continue;
      const SE3& T_i_c = calib.T_i_c[bch.cam_id];
      const SE3 T_c_w = (T_w_i_lm * T_i_c).inverse();
      Scalar xr_sraw3 = std::abs(bch.sigma_px);
      if (xr_sraw3 > Scalar(500.0)) xr_sraw3 -= Scalar(1000.0);
      const Scalar xsg = xr_sraw3;
      const Scalar w_meas = Scalar(1.0) / Scalar(xsg * xsg);
      const Scalar huber_px = Scalar(3.0) * Scalar(xsg);
      const Scalar cutoff_px = Scalar(20.0) * Scalar(xsg);
      for (int k = 0; k < bch.n; k++) {
        const Vec3 p_w(Scalar(bch.xyz[3 * k]), Scalar(bch.xyz[3 * k + 1]),
                       Scalar(bch.xyz[3 * k + 2]));
        const Vec3 p_c = T_c_w * p_w;
        Vec4 p_c4;
        p_c4 << p_c(0), p_c(1), p_c(2), Scalar(1);
        Vec2 proj;
        if (!calib.intrinsics[bch.cam_id].project(p_c4, proj)) continue;
        const Vec2 r(proj(0) - Scalar(bch.uv[2 * k]),
                     proj(1) - Scalar(bch.uv[2 * k + 1]));
        const Scalar rn = r.norm();
        if (!(rn < cutoff_px)) continue;
        const Scalar w_huber =
            xr_cauchy ? Scalar(1.0) / (Scalar(1.0) + (rn / huber_px) * (rn / huber_px))
                      : (rn <= huber_px ? Scalar(1) : huber_px / rn);
        err += Scalar(0.5) * w_meas * w_huber * rn * rn;
      }
    }
  }
  return err;
}

""" + anchor
t = t.replace(anchor, defn)

# ---- cpp: add to error_total (once per outer iteration) ----------------
old = """        stats.add("performQR", t.reset()).format("ms");
        times[0] += tt.elapsed_ns();
      }"""
new = """        stats.add("performQR", t.reset()).format("ms");
        times[0] += tt.elapsed_ns();
      }

      /* stage 14 (XR_ABSORB): credit the map/closure factor error in the
       * pre-step reference so f_diff sees closure-residual reduction. */
      if (xr_absorb_on()) error_total += xrAbsorbError();"""
assert t.count(old) == 1, "error_total anchor"
t = t.replace(old, new)

# ---- cpp: add to after_error_total (each backtrack) --------------------
old = """        Scalar after_error_total = after_update_vision_and_inertial_error + after_update_marg_prior_error;"""
new = """        Scalar after_error_total = after_update_vision_and_inertial_error + after_update_marg_prior_error;
        if (xr_absorb_on()) after_error_total += xrAbsorbError();"""
assert t.count(old) == 1, "after_error_total anchor"
t = t.replace(old, new)
f.write_text(t, encoding="utf-8")
print("stage 14 (XR_ABSORB closure-in-cost-test) applied")
