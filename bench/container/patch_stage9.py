#!/usr/bin/env python3
"""Stage 9 (VIO surgery V1c): ZNCC patch normalization — XR_ZNCC.

Diagnosed (ledger): both deterministic micro-jump events are photometric
transients — MH_05 t+79.4 is an auto-exposure RAMP (+1.9 mean/frame on
mean-50 images), V2_03 t+34-37 is violent flicker (+38/-17 swings).
Basalt's patch residual is mean-normalized: gain-invariant but NOT
bias/black-level invariant (large distortion on dim images) and noisy
(divide by small mean). ZNCC normalization — (I - mu)/sigma on template
and target alike — is invariant to gain AND bias; the Jacobian scales by
1/sigma (d mu/d sigma terms are second order; standard IC-ZNCC
approximation). Env-gated XR_ZNCC so fz19 semantics are untouched when
off. Applies to include/basalt/optical_flow/patch.h."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "include/basalt/optical_flow/patch.h"
t = f.read_text(encoding="utf-8")

# 0. env flag helper (file-local, read once)
anchor = ("  template <typename ImgT>\n"
          "  static void setData(const ImgT &img,")
assert t.count(anchor) == 1
t = t.replace(anchor, """  static inline bool xr_zncc() {
    static const bool v = [] {
      const char* e = getenv("XR_ZNCC");
      return e && *e && *e != '0';
    }();
    return v;
  }

""" + anchor, 1)  # helper placed BEFORE the template line

# 1. setData: optional ZNCC second pass
old = """    mean = sum / num_valid_points;
    data /= mean;
  }"""
new = """    mean = sum / num_valid_points;
    if (xr_zncc()) {
      /* zero-mean / unit-variance: gain AND bias invariant */
      Scalar var = 0;
      int nv = 0;
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (data[i] >= 0) { const Scalar d = data[i] - mean; var += d * d; nv++; }
      Scalar sigma = nv > 1 ? std::sqrt(var / nv) : Scalar(0);
      if (sigma < Scalar(1e-3)) sigma = Scalar(1e-3);
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (data[i] >= 0)
          data[i] = (data[i] - mean) / sigma + Scalar(10); /* +10: keep
             valid samples positive — data[i] >= 0 is the validity test
             everywhere; the offset cancels in the residual difference */
      mean = sigma;   /* carry sigma in the mean slot (validity checks) */
      return;
    }
    data /= mean;
  }"""
assert t.count(old) == 1, "setData anchor"
t = t.replace(old, new)

# 2. setDataJacSe2: ZNCC normalization + Jacobian scale
old = """    mean = sum / num_valid_points;

    const Scalar mean_inv = num_valid_points / sum;"""
assert t.count(old) == 1, "jac anchor A"
t = t.replace(old, """    mean = sum / num_valid_points;

    if (xr_zncc()) {
      Scalar var = 0;
      int nv = 0;
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (data[i] >= 0) { const Scalar d = data[i] - mean; var += d * d; nv++; }
      Scalar sigma = nv > 1 ? std::sqrt(var / nv) : Scalar(0);
      if (sigma < Scalar(1e-3)) sigma = Scalar(1e-3);
      const Scalar sig_inv = Scalar(1) / sigma;
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (data[i] >= 0)
          data[i] = (data[i] - mean) * sig_inv + Scalar(10);
      J_se2 *= sig_inv;
      mean = sigma;
      return;
    }

    const Scalar mean_inv = num_valid_points / sum;""")

# 3. residual(): ZNCC branch replacing the whole normalization loop
old = """    int num_residuals = 0;

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        residual[i] = num_valid_points * val / sum - data[i];
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }

    return num_residuals > PATTERN_SIZE / 2;"""
assert t.count(old) == 1, "residual block anchor"
new = """    int num_residuals = 0;

    if (xr_zncc()) {
      /* ZNCC: zero-mean/unit-variance over the mutually-valid samples —
       * gain AND bias invariant (exposure ramps, flicker, black level) */
      Scalar tsum = 0;
      int tn = 0;
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (residual[i] >= 0 && data[i] >= 0) { tsum += residual[i]; tn++; }
      if (tn <= PATTERN_SIZE / 2) { residual.setZero(); return false; }
      const Scalar tmean = tsum / tn;
      Scalar tvar = 0;
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (residual[i] >= 0 && data[i] >= 0) {
          const Scalar d = residual[i] - tmean;
          tvar += d * d;
        }
      Scalar tsig = std::sqrt(tvar / tn);
      if (tsig < Scalar(1e-3)) tsig = Scalar(1e-3);
      for (int i = 0; i < PATTERN_SIZE; i++) {
        if (residual[i] >= 0 && data[i] >= 0) {
          residual[i] = ((residual[i] - tmean) / tsig + Scalar(10)) - data[i];
          num_residuals++;
        } else {
          residual[i] = 0;
        }
      }
      return num_residuals > PATTERN_SIZE / 2;
    }

    for (int i = 0; i < PATTERN_SIZE; i++) {
      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        residual[i] = num_valid_points * val / sum - data[i];
        num_residuals++;

      } else {
        residual[i] = 0;
      }
    }

    return num_residuals > PATTERN_SIZE / 2;"""
t = t.replace(old, new)

if "#include <cstdlib>" not in t:
    t = t.replace("#pragma once", "#pragma once\n#include <cstdlib>\n#include <cmath>", 1)
f.write_text(t, encoding="utf-8")
print("stage 9 (XR_ZNCC patch normalization) applied")
