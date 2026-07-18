#!/usr/bin/env python3
"""Stage 8c: host injected landmarks the way flow landmarks are hosted.

The 8b host (newest kf_ids entry) is not guaranteed to be in the
optimization's AbsOrderMap (this fork moves keyframes to a long-term
set) -> BASALT_ASSERT(aom.count(host)) fired. Flow landmarks are hosted
at the CURRENT frame during a take_kf measure — guaranteed in the aom
with the same lifecycle. Do the same: landmark creation only when
take_kf, hosted at the current frame; observation-appends to existing
landmarks continue every frame. Applies on top of 8b."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

old = """        if (xr_pend.size() < 2) continue;
        if (kf_ids.empty()) continue;
        const int64_t xr_host = *kf_ids.rbegin();  /* newest keyframe */
        if (frame_states.count(xr_host) == 0 && frame_poses.count(xr_host) == 0)
          continue;
        SE3 xr_T_w_c = getPoseStateWithLin(xr_host).getPose() * calib.T_i_c[xit.cam_id];"""
new = """        if (xr_pend.size() < 2) continue;
        /* create only on a take_kf measure, hosted at the CURRENT frame —
         * the exact hosting contract flow landmarks use (guaranteed in
         * the AbsOrderMap, natural marginalization lifecycle) */
        if (!take_kf) continue;
        const int64_t xr_host = opt_flow_meas->t_ns;
        if (frame_states.count(xr_host) == 0 && frame_poses.count(xr_host) == 0)
          continue;
        SE3 xr_T_w_c = getPoseStateWithLin(xr_host).getPose() * calib.T_i_c[xit.cam_id];"""
assert t.count(old) == 1, "8c host anchor"
t = t.replace(old, new)

f.write_text(t, encoding="utf-8")
print("stage 8c (take_kf hosting) applied")
