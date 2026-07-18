#!/usr/bin/env python3
"""Stage 8e: injected landmarks marginalize with their HOST, never as
"lost". The lost-landmark criterion is flow connectivity (kpt_id present
in current optical-flow observations); injected ids (0x40000000+) are
never flow-tracked, so they were classified lost EVERY frame and
marg-linearized while hosted at keyframes newer than last_state_to_marg
-> BASALT_ASSERT(host in marg aom). Exempt the injected id range from
lost_landmaks; the host-marg path (all frame_poses are in the marg aom)
retires them naturally. Applies on top of 8..8d."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

old = """      if (!connected) { lost_landmaks.emplace(kv.first); }
    }
  }"""
new = """      if (!connected) { lost_landmaks.emplace(kv.first); }
    }
    /* XREAL stage-8: injected landmarks are not flow-tracked, so the
     * connectivity test always calls them lost — which would marg-
     * linearize them while hosted at kfs newer than last_state_to_marg
     * (host-not-in-aom assert). They retire with their HOST kf instead. */
    for (auto it = lost_landmaks.begin(); it != lost_landmaks.end();)
      it = (*it >= (KeypointId)0x40000000) ? lost_landmaks.erase(it)
                                           : std::next(it);
  }"""
assert t.count(old) == 1, "8e lost anchor"
f.write_text(t.replace(old, new), encoding="utf-8")
print("stage 8e (host-lifecycle marg for injected landmarks) applied")
