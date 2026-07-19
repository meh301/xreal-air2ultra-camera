#!/usr/bin/env python3
"""Stage 20 (VIO frontend, SHIPS): XR_TRIPAR — minimum-parallax gate on
new-landmark triangulation.

Audit finding #2 (verified 2026-07-19): the VIO's new-keyframe triangulation
(sqrt_keypoint_vio.cpp, the addLandmark loop ~line 522-540) accepts a
DLT-triangulated point on only a fixed metric-baseline gate
(min_triang_distance2) + an inverse-depth range ([3] in (0,3.0)), with NO
minimum parallax angle / conditioning test, and stores it as one scalar
inverse-distance with no variance. In open scenes 76-80% of triangulations
are low-parallax far features whose depth is essentially noise; they enter
FULL-WEIGHT and drag horizontal/scale accuracy.

A/B (reduced set, VIO track, --no-map, n=3): magistrale1 H -19%, magistrale2
H -35% (robust across all runs, zero regression on corridor2 despite
rejecting 35% of triangulations). VERTICAL essentially unchanged -> this is a
HORIZONTAL/scale win, not the altitude-twist fix. Ships default-ON via the
best-config env (XR_TRIPAR=1). Env: XR_TRIPAR (on/off), XR_TRIPAR_DEG (min
parallax degrees, default 1.0). OFF = bit-identical to stock.

Parallax angle = acos( f0 . (R_0_1 * f1) ) where f0,f1 are the two host
bearing rays and T_0_1 maps cam1->host frame (same convention triangulate()
uses internally). Reject (continue to the next obs pair; a wider-baseline
pair may still triangulate the landmark) when the angle is below threshold."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

assert "xr_tripar_on" not in t, "TRIPAR already applied"

# 1. includes + file-scope (anonymous-namespace) helpers, inserted right
#    before `namespace basalt {`.
anchor = "namespace basalt {"
assert t.count(anchor) >= 1, "namespace anchor"
helpers = """#include <atomic>
#include <cmath>
#include <cstdlib>
namespace {
inline bool xr_tripar_on() {
  static const bool v = [] { const char* e = getenv("XR_TRIPAR"); return e && *e && *e != '0'; }();
  return v;
}
inline double xr_tripar_deg() {
  static const double v = [] { const char* e = getenv("XR_TRIPAR_DEG"); double d = (e && *e) ? atof(e) : 1.0; return d > 0.0 ? d : 1.0; }();
  return v;
}
std::atomic<long> xr_tripar_seen{0}, xr_tripar_rej{0};
}  // namespace
namespace basalt {"""
t = t.replace(anchor, helpers, 1)

# 2. the parallax gate, inserted after the triangulate() call and before the
#    accept-if.
tri = ("Vec4 p0_triangulated = triangulate(p0_3d.template head<3>(), "
       "p1_3d.template head<3>(), T_0_1);")
assert t.count(tri) == 1, "triangulate anchor"
gate = tri + """
          if (xr_tripar_on()) {
            static const Scalar xr_tp_min_cos =
                Scalar(std::cos(xr_tripar_deg() * M_PI / 180.0));
            const Vec3 xr_f0 = p0_3d.template head<3>().normalized();
            const Vec3 xr_f1 = (T_0_1.so3() * p1_3d.template head<3>()).normalized();
            Scalar xr_cos = xr_f0.dot(xr_f1);
            if (xr_cos > Scalar(1)) xr_cos = Scalar(1);
            if (xr_cos < Scalar(-1)) xr_cos = Scalar(-1);
            xr_tripar_seen++;
            if (xr_cos > xr_tp_min_cos) {  // parallax below threshold -> reject
              const long xr_r = ++xr_tripar_rej;
              if (xr_r == 1 || xr_r % 4096 == 0)
                std::cerr << "[xr] TRIPAR reject " << xr_r << " / seen "
                          << xr_tripar_seen.load() << std::endl;
              continue;
            }
          }"""
t = t.replace(tri, gate)
f.write_text(t, encoding="utf-8")
print("stage 20 (XR_TRIPAR min-parallax triangulation gate) applied")
