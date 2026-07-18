#!/usr/bin/env python3
"""Stage 9b (VIO surgery V1c-2): transient-gated ZNCC — XR_ZNCC=2.

The always-on A/B (zn_em, ledger) split cleanly: V2_03 flicker -60%
(the designed-for case) but low-texture scenes HARMED (slides2 +43%,
MOO02 +23%) — ZNCC's contrast flattening destroys signal where contrast
IS the signal. (The sigma-floor bracket was invalid: basalt images are
16-bit, floors 3/6 in 8-bit units never engaged.) v2 = enable the ZNCC
branch only while a PHOTOMETRIC TRANSIENT is detected: windowed
frame-mean delta over ~10 frames catches both flicker swings (+38/-17
per frame) and slow exposure ramps (+1.9/frame accumulating), with a
hold window. XR_ZNCC=1 stays always-on; XR_ZNCC=2 transient-gated.
XR_ZNCC_DMEAN (8-bit units over the window, default 3.0),
XR_ZNCC_HOLD_MS (default 2000). Applies after stage 9."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- patch.h: level + shared gate flag ----------------------------------
f = W / "include/basalt/optical_flow/patch.h"
t = f.read_text(encoding="utf-8")
old = """  static inline bool xr_zncc() {
    static const bool v = [] {
      const char* e = getenv("XR_ZNCC");
      return e && *e && *e != '0';
    }();
    return v;
  }"""
new = """  /* v5: normalization mode BOUND AT PATCH CREATION — gate flapping
   * otherwise corrupts live tracks (template under one normalization,
   * target under the other). */
  bool xr_zncc_bound = false;

  static inline int xr_zncc_level() {
    static const int v = [] {
      const char* e = getenv("XR_ZNCC");
      return e && *e ? atoi(e) : 0;
    }();
    return v;
  }
  /* level 2 gate: armed by the flow's photometric-transient detector
   * (frame-mean jump), read per-patch. Per-instantiation static is fine:
   * writer (processFrame) and readers (setData/residual) share PatchT. */
  static std::atomic<int>& xr_zncc_gate() {
    static std::atomic<int> v{0};
    return v;
  }
  static inline bool xr_zncc() {
    const int lv = xr_zncc_level();
    if (lv <= 0) return false;
    if (lv == 1) return true;
    return xr_zncc_gate().load(std::memory_order_relaxed) != 0;
  }"""
assert t.count(old) == 1, "zncc helper anchor"
t = t.replace(old, new)
if "#include <atomic>" not in t:
    t = t.replace("#pragma once", "#pragma once\n#include <atomic>", 1)
old = """    setDataJacSe2(img, pos, mean, data, J_se2);"""
new = """    setDataJacSe2(img, pos, mean, data, J_se2);
    xr_zncc_bound = xr_zncc();   /* v5: bind normalization at creation */"""
assert t.count(old) == 1, "setFromImage anchor"
t = t.replace(old, new)
f.write_text(t, encoding="utf-8")

# ---- frame_to_frame_optical_flow.h: the detector ------------------------
f = W / "include/basalt/optical_flow/frame_to_frame_optical_flow.h"
t = f.read_text(encoding="utf-8")
if "#include <algorithm>" not in t:
    t = t.replace("#pragma once", "#pragma once
#include <algorithm>
#include <cstring>", 1)
old = """  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }"""
new = """  void processFrame(int64_t curr_t_ns, OpticalFlowInput::Ptr& new_img_vec) {
    for (const auto& v : new_img_vec->img_data) {
      if (!v.img.get()) return;
    }

    /* XR_ZNCC=2 (stage 9b): photometric-transient detector. Windowed
     * frame-mean delta (8-bit units) arms the patch ZNCC branch for a
     * hold window — flicker swings AND slow exposure ramps both
     * accumulate past the threshold; static scenes (incl. dark rooms
     * where contrast flattening hurts) keep stock normalization. */
    if (PatchT::xr_zncc_level() >= 2) {
      static const float xr_dthr = [] {
        const char* e = getenv("XR_ZNCC_DMEAN");
        return e && *e ? (float)atof(e) : 3.0f;
      }();
      static const int64_t xr_hold_ns = [] {
        const char* e = getenv("XR_ZNCC_HOLD_MS");
        return (int64_t)((e && *e ? atof(e) : 2000.0) * 1e6);
      }();
      const auto& xr_im = *new_img_vec->img_data[0].img;
      /* v4 UNIFORMITY TEST: 4x4 cell means — a GLOBAL photometric event
       * (exposure ramp, flicker) shifts every cell together; a scene
       * change (projector slide flip, moving light) shifts a region.
       * Arm only on uniform shifts: median cell |delta| above threshold
       * AND >=13/16 cells agreeing in sign. */
      float xr_cm[16];
      {
        const size_t cw = xr_im.w / 4, ch = xr_im.h / 4;
        for (int cy = 0; cy < 4; cy++)
          for (int cx = 0; cx < 4; cx++) {
            double s = 0;
            int n = 0;
            for (size_t y = cy * ch; y < (size_t)(cy + 1) * ch; y += 8)
              for (size_t x = cx * cw; x < (size_t)(cx + 1) * cw; x += 8) {
                s += xr_im(x, y);
                n++;
              }
            xr_cm[cy * 4 + cx] = n ? (float)(s / n) * (1.0f / 256.0f) : 0.f;
          }
      }
      /* XR_ZNCC_UNIFORM=1 (default): arm only on uniform shifts
       * (median cell delta + sign agreement). =0: v3 semantics — global
       * mean delta, no uniformity requirement (single long armings;
       * the variant that held V2_03's -60%). */
      static const int xr_uni = [] {
        const char* e = getenv("XR_ZNCC_UNIFORM");
        return e && *e ? atoi(e) : 1;
      }();
      static float xr_ring[10][16];
      static int xr_ri = 0, xr_rn = 0;
      static int64_t xr_active_until = 0;
      float xr_best_med = 0, xr_best_glob = 0;
      int xr_best_pos = 0, xr_best_neg = 0;
      for (int i = 0; i < xr_rn; i++) {
        float ad[16];
        int pos = 0, neg = 0;
        float gsum = 0;
        for (int c = 0; c < 16; c++) {
          const float d = xr_cm[c] - xr_ring[i][c];
          ad[c] = d < 0 ? -d : d;
          gsum += d;
          if (d > 0) pos++; else neg++;
        }
        const float g = gsum < 0 ? -gsum / 16.f : gsum / 16.f;
        if (g > xr_best_glob) xr_best_glob = g;
        std::nth_element(ad, ad + 8, ad + 16);
        if (ad[8] > xr_best_med) {
          xr_best_med = ad[8];
          xr_best_pos = pos;
          xr_best_neg = neg;
        }
      }
      memcpy(xr_ring[xr_ri], xr_cm, sizeof xr_cm);
      xr_ri = (xr_ri + 1) % 10;
      if (xr_rn < 10) xr_rn++;
      const bool xr_fire =
          xr_uni ? (xr_best_med > xr_dthr &&
                    (xr_best_pos >= 13 || xr_best_neg >= 13))
                 : (xr_best_glob > xr_dthr);
      if (xr_rn >= 5 && xr_fire)
        xr_active_until = curr_t_ns + xr_hold_ns;
      const int xr_on = curr_t_ns < xr_active_until ? 1 : 0;
      static int xr_logn = 0;
      if (xr_on && !PatchT::xr_zncc_gate().load(std::memory_order_relaxed) &&
          xr_logn < 20) {
        xr_logn++;
        std::cout << "[xr] ZNCC transient armed: dmed=" << xr_best_med
                  << " pos=" << xr_best_pos << std::endl;
      }
      PatchT::xr_zncc_gate().store(xr_on, std::memory_order_relaxed);
    }"""
assert t.count(old) == 1, "processFrame anchor"
t = t.replace(old, new)
f.write_text(t, encoding="utf-8")
print("stage 9b (transient-gated ZNCC) applied")
