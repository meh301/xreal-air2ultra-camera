#!/usr/bin/env python3
"""Stage 15 (VIO frontend architecture): regularized affine-photometric
patch residual — XR_AFFINE.

The principled version of the parked XR_ZNCC graft. Stock basalt's patch
residual is mean-normalized only (gain-invariant, NOT bias/exposure
invariant), so auto-exposure ramps (MH_05) and flicker (V2_03) inject a DC
offset that inflates every residual and breaks tracks. XR_ZNCC did
zero-mean/unit-variance (gain+bias invariant, V2_03 -60%) but divided by
the raw patch sigma, which on low-texture patches (flat rooms/slides)
amplifies noise -> harm; a hard sigma floor (znf_ab) only partly fixed it.

The fix is a Tikhonov REGULARIZER: the per-patch gain regularized toward
identity. In the ZNCC parameterization that is a smooth shrinkage of the
normalization scale:
    sigma_reg = sqrt(sigma^2 + lambda^2)
- high-texture (sigma >> lambda): sigma_reg -> sigma (full affine
  gain+bias invariance -> keeps the exposure/flicker win)
- low-texture (sigma << lambda): sigma_reg -> lambda (a bounded constant
  scaling, the gain shrinks toward 1 -> no noise amplification)
lambda is a real intensity-contrast scale (XR_AFFINE_LAMBDA, default 600
in the 16-bit patch units; the znf_ab probe showed sigma lives in the
100s-1000s). Unlike XR_ZNCC this is SAFE always-on (the regularizer, not a
transient gate, is what makes it safe on flat scenes), so it can be the
main-path residual. Env-gated XR_AFFINE for the A/B. Applies after stage 9
(reuses the ZNCC branch scaffolding)."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "include/basalt/optical_flow/patch.h"
t = f.read_text(encoding="utf-8")

# 1. helpers: XR_AFFINE level + lambda, next to xr_zncc_floor
old = """  static inline float xr_zncc_floor() {
    static const float v = [] {
      const char* e = getenv("XR_ZNCC_FLOOR");
      return e && *e ? (float)atof(e) : 1e-3f;
    }();
    return v;
  }"""
new = old + """
  /* stage 15: regularized affine-photometric main-path residual. */
  static inline bool xr_affine() {
    static const bool v = [] {
      const char* e = getenv("XR_AFFINE");
      return e && *e && *e != '0';
    }();
    return v;
  }
  /* Tikhonov gain-regularization scale (16-bit patch intensity units);
   * sigma_reg = sqrt(sigma^2 + lambda^2). */
  static inline float xr_affine_lambda() {
    static const float v = [] {
      const char* e = getenv("XR_AFFINE_LAMBDA");
      return e && *e ? (float)atof(e) : 600.0f;
    }();
    return v;
  }"""
assert t.count(old) == 1, "helper anchor"
t = t.replace(old, new)

# 2. the branch gate: XR_AFFINE also enters the normalized-residual branch
n_gate = t.count("if (xr_zncc()) {")
assert n_gate == 3, f"expected 3 zncc gates, found {n_gate}"
t = t.replace("if (xr_zncc()) {", "if (xr_zncc() || xr_affine()) {")

# 3. replace the sigma floor with the smooth regularizer when affine
# (setData + setDataJacSe2)
old_floor = "if (sigma < Scalar(xr_zncc_floor())) sigma = Scalar(xr_zncc_floor());"
n_floor = t.count(old_floor)
assert n_floor == 2, f"expected 2 sigma floors, found {n_floor}"
t = t.replace(old_floor,
    "if (xr_affine())\n"
    "        sigma = std::sqrt(sigma * sigma +\n"
    "                          Scalar(xr_affine_lambda()) * Scalar(xr_affine_lambda()));\n"
    "      else if (sigma < Scalar(xr_zncc_floor()))\n"
    "        sigma = Scalar(xr_zncc_floor());")

# residual()'s target sigma
old_tsig = "if (tsig < Scalar(1e-3)) tsig = Scalar(1e-3);"
if t.count(old_tsig) == 0:
    old_tsig = "if (tsig < Scalar(xr_zncc_floor())) tsig = Scalar(xr_zncc_floor());"
assert t.count(old_tsig) == 1, "residual tsig anchor"
t = t.replace(old_tsig,
    "if (xr_affine())\n"
    "        tsig = std::sqrt(tsig * tsig +\n"
    "                         Scalar(xr_affine_lambda()) * Scalar(xr_affine_lambda()));\n"
    "      else if (tsig < Scalar(xr_zncc_floor()))\n"
    "        tsig = Scalar(xr_zncc_floor());")

f.write_text(t, encoding="utf-8")
print("stage 15 (XR_AFFINE regularized affine-photometric residual) applied")
