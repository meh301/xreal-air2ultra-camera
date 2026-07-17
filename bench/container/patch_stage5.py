import sys
from pathlib import Path
W = Path(sys.argv[1] if len(sys.argv) > 1 else "/root/xreal/bench/container/basalt-linux")
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

# ---- stage 5a: FOLD-TIME residual arbitration -------------------------
# The fold decision moves from post-time gates (all failed) to fold-time:
# re-evaluate the batch against the CURRENT state, which has had seconds
# of IMU+vision evidence since the closure. Median residual must be tight.
old = """            Eigen::Matrix<Scalar, Eigen::Dynamic, 6> Jrows(2 * bch.n, 6);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rrows(2 * bch.n);
            int nr = 0;"""
new = """            Eigen::Matrix<Scalar, Eigen::Dynamic, 6> Jrows(2 * bch.n, 6);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rrows(2 * bch.n);
            int nr = 0;
            /* stage 5a: fold-time arbitration. Collect residual norms
             * against the CURRENT (evidence-refined) state first; only a
             * batch whose MEDIAN residual stayed tight earns the fold. A
             * wrong-place closure decoheres from the trajectory over the
             * seconds between post and marg — this is the temporal
             * arbitration the post-time gates could not do. */
            std::vector<Scalar> xr_rns;
            xr_rns.reserve(bch.n);"""
assert t.count(old) == 1, "5a rows anchor"
t = t.replace(old, new)

old = """              const Scalar rn = r.norm();
              if (!(rn < cutoff_px)) continue;
              const Vec3 v_wi = p_w - T_w_i_lm.translation();"""
assert t.count(old) >= 1
# only patch the FIRST occurrence (stage-4 fold loop comes before stage-3
# optimize loop in file order? stage-4 is in marginalize() ~line 900-1000,
# stage-3 optimize ~1600 — stage-4 first). Verify by context: the stage-4
# block contains xr_rns now.
i4 = t.index("xr_rns.reserve")
iocc = t.index(old, i4)
t = t[:iocc] + """              const Scalar rn = r.norm();
              if (!(rn < cutoff_px)) continue;
              xr_rns.push_back(rn);
              const Vec3 v_wi = p_w - T_w_i_lm.translation();""" + t[iocc + len(old):]

old = """            if (nr > 0) {
              if (is_lin_sqrt && marg_data.is_sqrt) {"""
new = """            /* fold-time verdict: median residual of surviving points */
            bool xr_fold_ok = nr > 0;
            if (xr_fold_ok && !xr_rns.empty()) {
              std::nth_element(xr_rns.begin(), xr_rns.begin() + xr_rns.size() / 2,
                               xr_rns.end());
              const Scalar med = xr_rns[xr_rns.size() / 2];
              Scalar fold_max = Scalar(4.0);
              { const char* e = getenv("XR_LMMARG_FOLD_PX");
                if (e && *e) fold_max = Scalar(atof(e)); }
              if (!(med < fold_max)) {
                xr_fold_ok = false;
                static int xr_rej_n = 0;
                if (xr_rej_n < 20) {
                  xr_rej_n++;
                  std::cerr << "[xr] LMMARG fold REJECTED at marg (median "
                            << (double)med << "px)" << std::endl;
                }
              }
            }
            if (xr_fold_ok) {
              if (is_lin_sqrt && marg_data.is_sqrt) {"""
assert t.count(old) == 1, "5a verdict anchor"
t = t.replace(old, new)

# the fold block's closing: batch must be CONSUMED either way (rejected
# batches must not retry forever)
old = """              xr_lm_batches.erase(xr_lm_batches.begin() + xbi);
              continue;   /* consumed */
            }
            ++xbi;"""
new = """              xr_lm_batches.erase(xr_lm_batches.begin() + xbi);
              continue;   /* consumed (folded) */
            }
            if (nr > 0) {   /* evaluated and rejected: consume, never retry */
              xr_lm_batches.erase(xr_lm_batches.begin() + xbi);
              continue;
            }
            ++xbi;"""
assert t.count(old) == 1, "5a consume anchor"
t = t.replace(old, new)

# ---- item 7: Cauchy kernel option (XR_LM_CAUCHY) in BOTH factor paths --
# w_huber -> cauchy weight w = 1/(1 + (r/huber)^2) when env set.
occ = t.count("const Scalar w_huber = rn <= huber_px ? Scalar(1) : huber_px / rn;")
assert occ == 2, f"kernel sites: {occ}"
t = t.replace(
    "const Scalar w_huber = rn <= huber_px ? Scalar(1) : huber_px / rn;",
    """static int xr_cauchy = -1;
              if (xr_cauchy < 0) {
                const char* ce = getenv("XR_LM_CAUCHY");
                xr_cauchy = (ce && *ce && *ce != '0') ? 1 : 0;
              }
              const Scalar w_huber = xr_cauchy
                  ? Scalar(1.0) / (Scalar(1.0) + (rn / huber_px) * (rn / huber_px))
                  : (rn <= huber_px ? Scalar(1) : huber_px / rn);""")

# need <algorithm> and <vector> for nth_element — usually present; ensure
if "#include <algorithm>" not in t:
    t = t.replace("#include <basalt", "#include <algorithm>\n#include <basalt", 1)

f.write_text(t, encoding="utf-8")
print("stage 5a (fold-time arbitration) + cauchy option applied")
