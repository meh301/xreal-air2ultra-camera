#!/usr/bin/env python3
"""Stage 12 (XRV P3b): closure-triggered re-advance — the vit API chain.

Exposes vit_tracker_xreal_readvance(): the map layer calls it when a
verified closure is APPLIED, arming xrv_readvance_pending so the next
marg event rebuilds the sqrt prior from the retained window (stage 11)
with the closure's landmark factors already live — fresh linearization
points at exactly the moment absorption matters (the DM-VIO closure
mechanic). Requires XR_DELAYMARG=1 for the capture substrate; the map
side gates the calls with XR_RADV. Applies after stage 11."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- 1. vio_estimator.h: virtual no-op ----------------------------------
f = W / "include/basalt/vi_estimator/vio_estimator.h"
t = f.read_text(encoding="utf-8")
old = """  virtual void setXrInjectLandmarks(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world,
                                    const int32_t* ids, int n) {
    (void)t_ns; (void)cam_id; (void)uv; (void)xyz_world; (void)ids; (void)n;
  }"""
new = old + """

  /* XREAL stage-12: arm a delayed-marg re-advance (P3b closure trigger).
   * Default: no-op. */
  virtual void setXrReadvance() {}"""
assert t.count(old) == 1, "vio_estimator anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 2. sqrt_keypoint_vio.h: override -----------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = """  bool xrv_readvance_pending = false;
  void xrvReadvance();"""
new = """  bool xrv_readvance_pending = false;   /* set cross-thread by the map
      layer's closure trigger; consumed once per marg — a lost write
      merely delays the rebuild one event (benign) */
  void xrvReadvance();
  void setXrReadvance() override {
    if (xrv_delaymarg_on()) xrv_readvance_pending = true;
  }"""
assert t.count(old) == 1, "sqrt h anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 3. vit_tracker.hpp: Tracker method decl ----------------------------
f = W / "include/basalt/vit/vit_tracker.hpp"
t = f.read_text(encoding="utf-8")
old = """  void xreal_inject_landmarks(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world, const int32_t *ids,
                              int n);"""
new = old + """

  /* XREAL stage-12: closure-triggered delayed-marg re-advance. */
  void xreal_readvance();"""
assert t.count(old) == 1, "vit hpp anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- 4. vit_tracker.cpp: impl + C export --------------------------------
f = W / "src/vit/vit_tracker.cpp"
t = f.read_text(encoding="utf-8")
old = """void Tracker::xreal_inject_landmarks(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world,
                                     const int32_t *ids, int n) {
  if (!impl_ || !impl_->vio) return;
  impl_->vio->setXrInjectLandmarks(t_ns, cam_id, uv, xyz_world, ids, n);
}"""
new = old + """

void Tracker::xreal_readvance() {
  if (!impl_ || !impl_->vio) return;
  impl_->vio->setXrReadvance();
}"""
assert t.count(old) == 1, "vit cpp impl anchor"
t = t.replace(old, new)
old = """extern "C" vit_result_t vit_tracker_xreal_inject_landmarks(vit_tracker_t *tracker, int64_t t_ns, int32_t cam_id,
                                                           const float *uv, const float *xyz_world,
                                                           const int32_t *ids, int32_t n) {
  if (!tracker || !uv || !xyz_world || !ids || n <= 0) return VIT_ERROR_INVALID_VALUE;
  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));
  t->xreal_inject_landmarks(t_ns, cam_id, uv, xyz_world, ids, n);
  return VIT_SUCCESS;
}"""
new = old + """

/* XREAL stage-12 extension: closure-triggered re-advance. */
extern "C" vit_result_t vit_tracker_xreal_readvance(vit_tracker_t *tracker) {
  if (!tracker) return VIT_ERROR_INVALID_VALUE;
  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));
  t->xreal_readvance();
  return VIT_SUCCESS;
}"""
assert t.count(old) == 1, "vit cpp export anchor"
f.write_text(t.replace(old, new), encoding="utf-8")
print("stage 12 (XR_RADV closure->readvance API chain) applied")
