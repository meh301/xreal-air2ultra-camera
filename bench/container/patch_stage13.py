#!/usr/bin/env python3
"""Stage 13 (XRV P4): NATIVE REVIVAL + closure-scoped re-advance.

The point of the rebuild, made native. Three pieces:

1. RETIRED STORE (XR_REVIVE): when a landmark dies (host-marged or
   flow-lost), its full observation history is retired into a bounded
   store keyed by the BRIDGE id (the id the map layer sends back at
   closures: raw flow id for native landmarks, map id for injected
   ones). Retirement replaces oblivion.

2. REVIVAL: when the stage-8 consume path creates an injected landmark
   whose bridge id is in the retired store, the retired observations on
   still-alive keyframes are GRAFTED onto it. The revived landmark then
   constrains the old keyframe poses AND the new track simultaneously —
   the OKVIS2 edge<->observation duality, aimed squarely at the corr1
   rigid-return-residual.

3. CLOSURE-SCOPED RE-ADVANCE (XR_REVIVE_RADV, needs XR_DELAYMARG): a
   revival arms the stage-11 rebuild ONLY when a retained marg event
   still references the revived landmark's original id, and the rebuild
   starts AT that event instead of the full window. No closure-relevant
   recoverable info in the window -> no rebuild (the corr3-protecting
   property the global trigger lacked, by construction).

Applies after stage 12."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header ------------------------------------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = """  std::deque<XrvMargEvent> xrv_marg_events;
  int64_t xrv_marg_seq = 0;"""
new = """  std::deque<XrvMargEvent> xrv_marg_events;
  int64_t xrv_marg_seq = 0;

  /* XRV P4 (stage 13): retired-landmark store — observation histories
   * of dead landmarks, keyed by the bridge id the map layer sends back
   * at closures. Bounded FIFO by retirement time. */
  struct XrvRetiredLm {
    int64_t retired_ns;
    KeypointId orig_id;
    Eigen::aligned_map<TimeCamId, Eigen::Matrix<Scalar, 2, 1>> obs;
  };
  std::map<int, XrvRetiredLm> xrv_retired;
  int xrv_readvance_start = 1 << 30;   /* scoped rebuild: oldest event
      referencing a revived id; sentinel = full window */
  static bool xrv_revive_on() {
    static const bool v = [] {
      const char* e = getenv("XR_REVIVE");
      return e && *e && *e != '0';
    }();
    return v;
  }
  static bool xrv_revive_radv_on() {
    static const bool v = [] {
      const char* e = getenv("XR_REVIVE_RADV");
      return e && *e && *e != '0';
    }();
    return v;
  }"""
assert t.count(old) == 1, "hdr anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp ---------------------------------------------------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

# 1. capture block runs for revive too; retire before deque push;
#    deque/prior copy stays delaymarg-only
old = """    /* XRV P2a (stage 10, XR_DELAYMARG): snapshot everything this event
     * consumes BEFORE erasure — the reversible-marg substrate. */
    if (xrv_delaymarg_on()) {"""
new = """    /* XRV P2a (stage 10, XR_DELAYMARG): snapshot everything this event
     * consumes BEFORE erasure — the reversible-marg substrate. Stage 13
     * rides the same snapshot for retirement (XR_REVIVE). */
    if (xrv_delaymarg_on() || xrv_revive_on()) {"""
assert t.count(old) == 1, "capture guard anchor"
t = t.replace(old, new)

old = """      ev.prior_before = marg_data;
      xrv_marg_events.push_back(std::move(ev));
      while ((int)xrv_marg_events.size() > xrv_d) xrv_marg_events.pop_front();"""
new = """      /* stage 13 retirement: dead landmarks' observation histories go
       * to the bounded store under their bridge id (what the map layer
       * sends back at a future closure). */
      if (xrv_revive_on()) {
        auto xr_retire = [&](int id, const Landmark<Scalar>& lm) {
          const int bid = id >= 0x40000000 ? id - 0x40000000 : id;
          auto& r = xrv_retired[bid];
          r.retired_ns = last_state_to_marg;
          r.orig_id = (KeypointId)id;
          r.obs.clear();
          for (const auto& ob : lm.obs) r.obs[ob.first] = ob.second;
        };
        for (const auto& hl : ev.host_lms) xr_retire(hl.first, hl.second);
        for (const auto& ll : ev.lost_lms) xr_retire(ll.first, ll.second);
        while (xrv_retired.size() > 4096) {
          auto oldest = xrv_retired.begin();
          for (auto r2 = xrv_retired.begin(); r2 != xrv_retired.end(); ++r2)
            if (r2->second.retired_ns < oldest->second.retired_ns)
              oldest = r2;
          xrv_retired.erase(oldest);
        }
      }
      if (!xrv_delaymarg_on()) {
        /* revive-only mode: no retained-window substrate */
      } else {
      ev.prior_before = marg_data;
      xrv_marg_events.push_back(std::move(ev));
      while ((int)xrv_marg_events.size() > xrv_d) xrv_marg_events.pop_front();"""
assert t.count(old) == 1, "capture tail anchor"
t = t.replace(old, new)

# close the delaymarg-only brace after the capture log block
old = """        std::cerr << "[xr] DELAYMARG capture #" << e2.seq << ": states="
                  << e2.states.size() << " imu=" << e2.imu.size()
                  << " host_lms=" << e2.host_lms.size() << " lost_lms=" << e2.lost_lms.size() << " dropped_obs="
                  << e2.dropped_obs.size() << " (deque "
                  << xrv_marg_events.size() << ")" << std::endl;
      }
    }"""
new = """        std::cerr << "[xr] DELAYMARG capture #" << e2.seq << ": states="
                  << e2.states.size() << " imu=" << e2.imu.size()
                  << " host_lms=" << e2.host_lms.size() << " lost_lms=" << e2.lost_lms.size() << " dropped_obs="
                  << e2.dropped_obs.size() << " (deque "
                  << xrv_marg_events.size() << ")" << std::endl;
      }
      }
    }"""
assert t.count(old) == 1, "capture log anchor"
t = t.replace(old, new)

# 2. revival graft in the consume creation branch
old = """              e.created = 1;
              xr_made++;"""
new = """              e.created = 1;
              xr_made++;
              /* XRV P4 (stage 13): NATIVE REVIVAL — this map landmark
               * was a previously-retired estimator landmark: graft its
               * historical observations on still-alive keyframes. The
               * revived landmark constrains the OLD poses and the new
               * track simultaneously. */
              if (xrv_revive_on()) {
                auto rit = xrv_retired.find(it->first);
                if (rit != xrv_retired.end()) {
                  int xr_g = 0;
                  for (const auto& ob : rit->second.obs) {
                    const int64_t tf = ob.first.frame_id;
                    if (!frame_poses.count(tf) && !frame_states.count(tf))
                      continue;
                    auto& lm2 = lmdb.getLandmark(xr_bid);
                    if (lm2.obs.count(ob.first)) continue;
                    KeypointObservation<Scalar> kg;
                    kg.kpt_id = xr_bid;
                    kg.pos = ob.second;
                    lmdb.addObservation(ob.first, kg);
                    xr_g++;
                  }
                  /* closure-scoped re-advance: only if a retained event
                   * still references the ORIGINAL id — start there. */
                  if (xr_g > 0 && xrv_revive_radv_on() &&
                      xrv_delaymarg_on()) {
                    const KeypointId oid = rit->second.orig_id;
                    for (size_t ei = 0; ei < xrv_marg_events.size();
                         ei++) {
                      const auto& ev2 = xrv_marg_events[ei];
                      bool hit = false;
                      for (const auto& o2 : ev2.dropped_obs)
                        if ((KeypointId)o2.lm_id == oid) {
                          hit = true;
                          break;
                        }
                      if (!hit)
                        for (const auto& h2 : ev2.host_lms)
                          if ((KeypointId)h2.first == oid) {
                            hit = true;
                            break;
                          }
                      if (!hit)
                        for (const auto& l2 : ev2.lost_lms)
                          if ((KeypointId)l2.first == oid) {
                            hit = true;
                            break;
                          }
                      if (hit) {
                        xrv_readvance_pending = true;
                        if ((int)ei < xrv_readvance_start)
                          xrv_readvance_start = (int)ei;
                        break;
                      }
                    }
                  }
                  if (xr_g > 0) xr_revived++;
                  xrv_retired.erase(rit);
                }
              }"""
assert t.count(old) == 1, "revival anchor"
t = t.replace(old, new)

# 3. revival counter + log
old = """      const int64_t xr_now = opt_flow_meas->t_ns;
      int xr_obs = 0, xr_made = 0;"""
new = """      const int64_t xr_now = opt_flow_meas->t_ns;
      int xr_obs = 0, xr_made = 0, xr_revived = 0;"""
assert t.count(old) == 1, "counter anchor"
t = t.replace(old, new)
old = """      static int xr_inj_logn = 0;
      if ((xr_made || xr_obs) && xr_inj_logn < 40) {
        xr_inj_logn++;
        std::cerr << "[xr] LMINJ +" << xr_made << " landmarks, +" << xr_obs
                  << " obs (lmdb " << lmdb.numLandmarks() << ")" << std::endl;
      }"""
new = """      static int xr_inj_logn = 0;
      if ((xr_made || xr_obs) && xr_inj_logn < 40) {
        xr_inj_logn++;
        std::cerr << "[xr] LMINJ +" << xr_made << " landmarks, +" << xr_obs
                  << " obs (lmdb " << lmdb.numLandmarks() << ")" << std::endl;
      }
      static int xr_rev_logn = 0;
      if (xr_revived && xr_rev_logn < 40) {
        xr_rev_logn++;
        std::cerr << "[xr] REVIVED " << xr_revived
                  << " landmarks w/ historical obs (retired store "
                  << xrv_retired.size() << ")" << std::endl;
      }"""
assert t.count(old) == 1, "log anchor"
t = t.replace(old, new)

# 4. scoped start in the engine
old = """  if (xrv_marg_events.empty()) return;
  const auto& ev0 = xrv_marg_events.front();"""
new = """  if (xrv_marg_events.empty()) return;
  /* stage 13: closure-scoped rebuild — start at the oldest event a
   * revival referenced; sentinel means the full retained window. */
  size_t xr_s0 = 0;
  if (xrv_readvance_start > 0 &&
      (size_t)xrv_readvance_start < xrv_marg_events.size())
    xr_s0 = (size_t)xrv_readvance_start;
  xrv_readvance_start = 1 << 30;
  const auto& ev0 = xrv_marg_events[xr_s0];"""
assert t.count(old) == 1, "engine head anchor"
t = t.replace(old, new)

# 4b. every event loop in the engine starts at xr_s0
n = t.count("  for (const auto& ev : xrv_marg_events) {")
assert n == 3, n
t = t.replace(
    "  for (const auto& ev : xrv_marg_events) {",
    """  for (size_t xr_ei = xr_s0; xr_ei < xrv_marg_events.size(); xr_ei++) {
    const auto& ev = xrv_marg_events[xr_ei];""")
old = """  for (const auto& ev : xrv_marg_events)
    for (const auto& im : ev.imu) {"""
new = """  for (size_t xr_ei = xr_s0; xr_ei < xrv_marg_events.size(); xr_ei++)
    for (const auto& im : xrv_marg_events[xr_ei].imu) {"""
assert t.count(old) == 1, "imu loop anchor"
t = t.replace(old, new)

# 4c. log the scope
old = """    std::cerr << "[xr] READVANCE ok: prior rebuilt over \""""
if t.count(old) != 1:
    old = """    std::cerr << "[xr] READVANCE ok: prior rebuilt over """
assert t.count(old) >= 1, "ok log anchor"
t = t.replace(
    '"[xr] READVANCE ok: prior rebuilt over "',
    '"[xr] READVANCE ok (scope " << xr_s0 << "/" << xrv_marg_events.size()\n              << "): prior rebuilt over "', 1)
f.write_text(t, encoding="utf-8")
print("stage 13 (XR_REVIVE native revival + scoped re-advance) applied")
