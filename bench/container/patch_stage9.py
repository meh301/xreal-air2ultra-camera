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
anchor = "  static void setData(const ImgT &img,"
assert t.count(anchor) == 1
t = t.replace(anchor, """  static inline bool xr_zncc() {
    static const bool v = [] {
      const char* e = getenv("XR_ZNCC");
      return e && *e && *e != '0';
    }();
    return v;
  }

""" + anchor, 1)

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
        if (data[i] >= 0) data[i] = (data[i] - mean) / sigma;
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
        if (data[i] >= 0) data[i] = (data[i] - mean) * sig_inv;
      J_se2 *= sig_inv;
      mean = sigma;
      return;
    }

    const Scalar mean_inv = num_valid_points / sum;""")

# 3. residual(): ZNCC on the target samples
old = """      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        residual[i] = num_valid_points * val / sum - data[i];"""
assert t.count(old) == 1, "residual anchor"
t = t.replace(old, """      if (residual[i] >= 0 && data[i] >= 0) {
        const Scalar val = residual[i];
        if (xr_zncc()) {
          residual[i] = val;   /* normalized in the second pass below */
        } else
        residual[i] = num_valid_points * val / sum - data[i];""")

# 3b. add the ZNCC second pass after the residual loop. Anchor: the loop
# closes and num_residuals is returned/used — find the tail.
old2 = """        num_residuals++;
      } else {
        residual[i] = 0;
      }
    }"""
assert t.count(old2) == 1, "residual tail anchor"
t = t.replace(old2, """        num_residuals++;
      } else {
        residual[i] = 0;
      }
    }
    if (xr_zncc() && num_residuals > 1) {
      /* second pass: zero-mean/unit-var the target samples, then diff */
      const Scalar tmean = sum / num_residuals;
      Scalar tvar = 0;
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (residual[i] != 0 && data[i] >= 0) {
          const Scalar d = residual[i] - tmean;
          tvar += d * d;
        }
      Scalar tsig = std::sqrt(tvar / num_residuals);
      if (tsig < Scalar(1e-3)) tsig = Scalar(1e-3);
      for (int i = 0; i < PATTERN_SIZE; i++)
        if (residual[i] != 0 && data[i] >= 0)
          residual[i] = (residual[i] - tmean) / tsig - data[i];
    }""")

if "#include <cstdlib>" not in t:
    t = t.replace("#pragma once", "#pragma once\n#include <cstdlib>\n#include <cmath>", 1)
f.write_text(t, encoding="utf-8")
print("stage 9 (XR_ZNCC patch normalization) applied")
