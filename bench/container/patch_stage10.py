#!/usr/bin/env python3
"""Stage 10 (XRV P2a): delayed-marginalization capture layer — XR_DELAYMARG.

The keystone of the estimator-core rebuild (bench/XRV_DESIGN.md): make
marginalization reversible for a window of D events. P2a captures, at
every marg event and BEFORE anything is erased, a deep snapshot of what
the event consumes: the marged state ids and estimates, their IMU
preintegrations, every landmark observation about to be dropped
(host-marged landmarks in full; surviving landmarks' obs on marged
frames), and the pre-event sqrt prior. Bounded deque of D events.
ZERO behavior change (capture only) — the P2b twin prior and the P3
closure-triggered re-advance consume this store. Env-gated XR_DELAYMARG.
"""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header: event store ------------------------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = "  std::map<int, XrInjLm> xr_inj_lms;"
new = """  std::map<int, XrInjLm> xr_inj_lms;

  /* XRV P2a (stage 10): retained marginalization events — everything a
   * marg event consumes, snapshotted BEFORE erasure, so the prior can
   * later be re-derived with fresh linearization (P2b/P3). */
  struct XrvMargState {
    int64_t t_ns;
    SE3 T_w_i;                       /* current estimate at capture */
    Vec3 vel, bg, ba;
    SE3 T_w_i_lin;                   /* FEJ lin point at capture */
    Eigen::Matrix<Scalar, 15, 1> delta;
    bool was_linearized;
    bool had_vel_bias;
  };
  struct XrvMargObs {
    int lm_id;
    TimeCamId tcid;
    Eigen::Matrix<Scalar, 2, 1> pos;
  };
  struct XrvMargEvent {
    int64_t seq;
    int64_t last_state_to_marg;
    std::vector<XrvMargState> states;          /* marged states/poses */
    std::vector<std::pair<int64_t, IntegratedImuMeasurement<Scalar>>>
        imu;                                   /* keyed by state t_ns */
    Eigen::aligned_vector<std::pair<int, Landmark<Scalar>>>
        host_lms;                              /* host-marged, full copy */
    std::vector<XrvMargObs> dropped_obs;       /* survivors' lost obs */
    Eigen::aligned_vector<std::pair<int, Landmark<Scalar>>>
        lost_lms;                              /* flow-lost, folded at marg */
    MargLinData<Scalar> prior_before;          /* pre-event sqrt prior */
  };
  std::deque<XrvMargEvent> xrv_marg_events;
  int64_t xrv_marg_seq = 0;
  static bool xrv_delaymarg_on() {
    static const bool v = [] {
      const char* e = getenv("XR_DELAYMARG");
      return e && *e && *e != '0';
    }();
    return v;
  }"""
assert t.count(old) == 1, "hdr anchor"
t = t.replace(old, new)
if "#include <deque>" not in t:
    t = t.replace("#pragma once", "#pragma once\n#include <deque>", 1)
f.write_text(t, encoding="utf-8")

# ---- cpp: capture before erasure ---------------------------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")
old = """    {
      BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() == false);
      frame_states.at(last_state_to_marg).setLinTrue();
    }

    for (const int64_t id : states_to_marg_all) {"""
new = """    {
      BASALT_ASSERT(frame_states.at(last_state_to_marg).isLinearized() == false);
      frame_states.at(last_state_to_marg).setLinTrue();
    }

    /* XRV P2a (stage 10, XR_DELAYMARG): snapshot everything this event
     * consumes BEFORE erasure — the reversible-marg substrate. */
    if (xrv_delaymarg_on()) {
      static int xrv_d = [] {
        const char* e = getenv("XR_DELAYMARG_D");
        int v = e && *e ? atoi(e) : 6;
        return v > 0 ? v : 6;
      }();
      XrvMargEvent ev;
      ev.seq = xrv_marg_seq++;
      ev.last_state_to_marg = last_state_to_marg;
      auto snap_state = [&](int64_t id) {
        XrvMargState s;
        s.t_ns = id;
        if (frame_states.count(id)) {
          const auto& sw = frame_states.at(id);
          const auto& st = sw.getState();
          s.T_w_i = st.T_w_i;
          s.vel = st.vel_w_i;
          s.bg = st.bias_gyro;
          s.ba = st.bias_accel;
          s.T_w_i_lin = sw.getStateLin().T_w_i;
          s.delta = sw.isLinearized() ? sw.getDelta()
                                      : Eigen::Matrix<Scalar, 15, 1>::Zero();
          s.was_linearized = sw.isLinearized();
          s.had_vel_bias = true;
        } else if (frame_poses.count(id)) {
          const auto& pw = frame_poses.at(id);
          s.T_w_i = pw.getPose();
          s.vel.setZero();
          s.bg.setZero();
          s.ba.setZero();
          s.T_w_i_lin = pw.isLinearized() ? pw.getPoseLin() : pw.getPose();
          s.delta.setZero();
          if (pw.isLinearized())
            s.delta.template head<6>() = pw.getDelta();
          s.was_linearized = pw.isLinearized();
          s.had_vel_bias = false;
        } else {
          return;
        }
        ev.states.push_back(s);
      };
      for (const int64_t id : states_to_marg_all) snap_state(id);
      for (const int64_t id : states_to_marg_vel_bias) snap_state(id);
      for (const int64_t id : poses_to_marg) snap_state(id);
      for (const int64_t id : kfs_to_marg) snap_state(id);
      for (const int64_t id : states_to_marg_all)
        if (imu_meas.count(id)) ev.imu.push_back({id, imu_meas.at(id)});
      for (const int64_t id : states_to_marg_vel_bias)
        if (imu_meas.count(id)) ev.imu.push_back({id, imu_meas.at(id)});
      /* landmarks: host-marged in full; survivors' obs on marged frames */
      for (const auto& kv : lmdb.getLandmarks()) {
        const auto& lm = kv.second;
        const int64_t host = lm.host_kf_id.frame_id;
        const bool host_goes = kfs_to_marg.count(host) > 0 ||
                               poses_to_marg.count(host) > 0 ||
                               states_to_marg_all.count(host) > 0;
        if (host_goes) {
          ev.host_lms.push_back({(int)kv.first, lm});
          continue;
        }
        for (const auto& ob : lm.obs) {
          const int64_t tf = ob.first.frame_id;
          if (kfs_to_marg.count(tf) > 0 || poses_to_marg.count(tf) > 0 ||
              states_to_marg_all.count(tf) > 0) {
            XrvMargObs o;
            o.lm_id = (int)kv.first;
            o.tcid = ob.first;
            o.pos = ob.second;
            ev.dropped_obs.push_back(o);
          }
        }
      }
      /* flow-lost landmarks: stock FOLDS them into the prior at this
       * event then removes them — most of the event's vision info.
       * Snapshot in full so the re-advance can replay the fold. */
      if (config.vio_marg_lost_landmarks)
        for (const auto& xr_lmid : lost_landmaks)
          if (lmdb.landmarkExists(xr_lmid))
            ev.lost_lms.push_back({(int)xr_lmid, lmdb.getLandmark(xr_lmid)});
      ev.prior_before = marg_data;
      xrv_marg_events.push_back(std::move(ev));
      while ((int)xrv_marg_events.size() > xrv_d) xrv_marg_events.pop_front();
      static int xrv_logn = 0;
      if (xrv_logn < 12) {
        xrv_logn++;
        const auto& e2 = xrv_marg_events.back();
        std::cerr << "[xr] DELAYMARG capture #" << e2.seq << ": states="
                  << e2.states.size() << " imu=" << e2.imu.size()
                  << " host_lms=" << e2.host_lms.size() << " lost_lms=" << e2.lost_lms.size() << " dropped_obs="
                  << e2.dropped_obs.size() << " (deque "
                  << xrv_marg_events.size() << ")" << std::endl;
      }
    }

    for (const int64_t id : states_to_marg_all) {"""
assert t.count(old) == 1, "cpp capture anchor"
t = t.replace(old, new)
f.write_text(t, encoding="utf-8")
print("stage 10 (XR_DELAYMARG P2a capture) applied")
