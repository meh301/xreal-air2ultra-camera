#!/usr/bin/env python3
"""Stage 6: per-batch fold-arbitration class on the sigma channel.

The app encodes room-class persist batches as |sigma| + 1000 (sign still
means persist/transient). Room-class batches skip fold-time arbitration
entirely (s5b/s5c: rooms show a clean dose-response — arbitration only
taxes the prize there), corridor-class keeps the XR_LMMARG_FOLD_PX
median-residual verdict (corr3 replicated: unarbitrated corridor folds
are poison). Apply after patch_lmfact.py stages 2-5a + lmmarg_gate.py.
"""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

# stage-4 fold site (12-space indent): decode the +1000 room-class offset
old = """            const Scalar xsg = std::abs(bch.sigma_px);
            const Scalar w_meas = Scalar(1.0) / Scalar(xsg * xsg);"""
new = """            Scalar xr_sraw = std::abs(bch.sigma_px);
            const bool xr_noarb = xr_sraw > Scalar(500.0); /* +1000 = room-class */
            if (xr_noarb) xr_sraw -= Scalar(1000.0);
            const Scalar xsg = xr_sraw;
            const Scalar w_meas = Scalar(1.0) / Scalar(xsg * xsg);"""
assert t.count(old) == 1, "stage-4 xsg anchor"
t = t.replace(old, new)

# stage-3 optimize site (16-space indent): decode magnitude only
old = """                const Scalar xsg = std::abs(bch.sigma_px);
                const Scalar w_meas = Scalar(1.0) / Scalar(xsg * xsg);"""
new = """                Scalar xr_sraw3 = std::abs(bch.sigma_px);
                if (xr_sraw3 > Scalar(500.0)) xr_sraw3 -= Scalar(1000.0);
                const Scalar xsg = xr_sraw3;
                const Scalar w_meas = Scalar(1.0) / Scalar(xsg * xsg);"""
assert t.count(old) == 1, "stage-3 xsg anchor"
t = t.replace(old, new)

# fold verdict: room-class batches bypass the residual arbitration
old = """              Scalar fold_max = Scalar(4.0);
              { const char* e = getenv("XR_LMMARG_FOLD_PX");
                if (e && *e) fold_max = Scalar(atof(e)); }"""
new = """              Scalar fold_max = Scalar(4.0);
              { const char* e = getenv("XR_LMMARG_FOLD_PX");
                if (e && *e) fold_max = Scalar(atof(e)); }
              if (xr_noarb) fold_max = Scalar(1e30); /* room-class: no arb */"""
assert t.count(old) == 1, "verdict anchor"
t = t.replace(old, new)

f.write_text(t, encoding="utf-8")
print("stage 6 (per-batch room-class no-arb) applied")
