#!/usr/bin/env python3
"""Stage 7: FIX the persist-sign destruction in setXrLandmarkFactors.

The original stage-3 setter sanitized sigma with `sigma_px > 0 ? sigma_px
: 2.0f`, which rewrites every NEGATIVE (transient) sigma to +2.0 — the
stage-4 persist-sign convention was destroyed at the door from day one.
Every batch (transient closures, LMTRACK re-posts) has been fold-eligible
regardless of the app-side persist gates; only the FOLD_PX arbitration
axis ever functioned. Preserve the sign; sanitize magnitude only.
Apply to include/basalt/vi_estimator/sqrt_keypoint_vio.h.
"""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")

old = "    slot->sigma_px = sigma_px > 0 ? sigma_px : 2.0f;"
new = ("    /* SIGN carries persist(+)/transient(-) — never rewrite it.\n"
       "     * Magnitude may carry the +1000 room-class offset (stage 6). */\n"
       "    slot->sigma_px =\n"
       "        (sigma_px != 0.0f && sigma_px > -1.0e4f && sigma_px < 1.0e4f)\n"
       "            ? sigma_px : 2.0f;")
assert t.count(old) == 1, "setter sigma anchor"
t = t.replace(old, new)

f.write_text(t, encoding="utf-8")
print("stage 7 (persist-sign fix) applied")
