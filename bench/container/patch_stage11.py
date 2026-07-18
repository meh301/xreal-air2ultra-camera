#!/usr/bin/env python3
"""Stage 11 (XRV P3a): the re-advance engine — XR_READVANCE.

Consumes the stage-10 capture: rebuilds the sqrt marg prior from the
oldest retained pre-event prior by RESURRECTING every captured dead
state and landmark observation, re-linearizing the whole retained
window at CURRENT estimates with the estimator's own linearization
machinery, re-marginalizing the dead variables via MargHelper, and
burying the resurrected objects again. This makes the last D
marginalization events reversible with fresh linearization points —
the DM-VIO effect inside basalt's sqrt form.

P3a trigger: XR_READVANCE_TEST=N re-advances every N marg events
(mechanics validation). The closure trigger (P3b) sets
xrv_readvance_pending from the closure bridge instead.
Applies after stage 10."""
import sys
from pathlib import Path

W = Path(sys.argv[1] if len(sys.argv) > 1 else
         "/root/xreal/bench/container/basalt-linux")

# ---- header: pending flag + method decl ---------------------------------
f = W / "include/basalt/vi_estimator/sqrt_keypoint_vio.h"
t = f.read_text(encoding="utf-8")
old = """  std::deque<XrvMargEvent> xrv_marg_events;
  int64_t xrv_marg_seq = 0;"""
new = """  std::deque<XrvMargEvent> xrv_marg_events;
  int64_t xrv_marg_seq = 0;
  bool xrv_readvance_pending = false;
  void xrvReadvance();"""
assert t.count(old) == 1, "hdr anchor"
f.write_text(t.replace(old, new), encoding="utf-8")

# ---- cpp ---------------------------------------------------------------
f = W / "src/vi_estimator/sqrt_keypoint_vio.cpp"
t = f.read_text(encoding="utf-8")

# 1. test trigger: after the capture block's deque push
old = """      while ((int)xrv_marg_events.size() > xrv_d) xrv_marg_events.pop_front();"""
new = """      while ((int)xrv_marg_events.size() > xrv_d) xrv_marg_events.pop_front();
      static int xrv_test_n = [] {
        const char* e = getenv("XR_READVANCE_TEST");
        return e && *e ? atoi(e) : 0;
      }();
      if (xrv_test_n > 0 && (ev.seq % xrv_test_n) == (xrv_test_n - 1))
        xrv_readvance_pending = true;"""
assert t.count(old) == 1, "trigger anchor"
t = t.replace(old, new)

# 2. invoke point: in measure(), right after the LMINJ consume block
#    (before optimize_and_marg would run again next frame). Anchor on the
#    lost_landmaks construction which follows the injection block.
old = """  std::unordered_set<KeypointId> lost_landmaks;
  if (config.vio_marg_lost_landmarks) {"""
new = """  if (xrv_delaymarg_on() && xrv_readvance_pending) {
    xrv_readvance_pending = false;
    xrvReadvance();
  }

  std::unordered_set<KeypointId> lost_landmaks;
  if (config.vio_marg_lost_landmarks) {"""
assert t.count(old) == 1, "invoke anchor"
t = t.replace(old, new)

# 3. the engine — append before the explicit template instantiations
old = "// instantiate templates"
if t.count(old) != 1:
    # fall back: append before the final instantiation lines
    old = "template class SqrtKeypointVioEstimator<float>;"
assert t.count(old) >= 1, "instantiation anchor"
engine = """
template <class Scalar_>
void SqrtKeypointVioEstimator<Scalar_>::xrvReadvance() {
  if (xrv_marg_events.empty()) return;
  const auto& ev0 = xrv_marg_events.front();

  /* ---- 0. PROMOTE: alive kf poses that were vel-bias-marged during
   *         the window come back up to 15-dof states (pose part at the
   *         pose's own FEJ lin point, vel/bias at captured values) so
   *         the oldest prior's 15-dof columns and the captured IMU
   *         chain line up; vel/bias re-marged, pose kept, object
   *         demoted back after the rebuild. */
  Eigen::aligned_map<int64_t, PoseStateWithLin<Scalar>> promoted_saved;
  for (const auto& ev : xrv_marg_events) {
    for (const auto& s : ev.states) {
      if (!s.had_vel_bias) continue;
      if (frame_states.count(s.t_ns)) continue;
      auto itp = frame_poses.find(s.t_ns);
      if (itp == frame_poses.end()) continue;
      if (promoted_saved.count(s.t_ns)) continue;
      promoted_saved.emplace(s.t_ns, itp->second);
      PoseVelBiasState<Scalar> st;
      st.t_ns = s.t_ns;
      st.T_w_i = itp->second.isLinearized() ? itp->second.getPoseLin()
                                            : itp->second.getPose();
      st.vel_w_i = s.vel;
      st.bias_gyro = s.bg;
      st.bias_accel = s.ba;
      PoseVelBiasStateWithLin<Scalar> pvb(st);
      pvb.setLinTrue();
      if (itp->second.isLinearized()) {
        Eigen::Matrix<Scalar, 15, 1> inc;
        inc.setZero();
        inc.template head<6>() = itp->second.getDelta();
        pvb.applyInc(inc);
      }
      frame_poses.erase(itp);
      frame_states[s.t_ns] = pvb;
    }
  }

  /* ---- 1. RESURRECT: dead states (stored estimates, fresh lin) and
   *         dead landmarks/observations ---- */
  std::set<int64_t> res_poses, res_states;
  for (const auto& ev : xrv_marg_events) {
    for (const auto& s : ev.states) {
      if (frame_poses.count(s.t_ns) || frame_states.count(s.t_ns)) continue;
      /* size contract comes from the oldest prior's order when present */
      auto it = ev0.prior_before.order.abs_order_map.find(s.t_ns);
      const bool as_state =
          (it != ev0.prior_before.order.abs_order_map.end() &&
           it->second.second == POSE_VEL_BIAS_SIZE) ||
          (it == ev0.prior_before.order.abs_order_map.end() &&
           s.had_vel_bias);
      /* prior-order vars must satisfy the FEJ contract: resurrect at
       * the CAPTURED LIN POINT and replay the captured delta, so the
       * old prior's columns stay valid AND estimate = lin (+) delta
       * reproduces the captured estimate. Vars outside the old prior
       * get a fresh linearization at the captured estimate. */
      const bool in_prior =
          it != ev0.prior_before.order.abs_order_map.end();
      const bool lin_replay = in_prior && s.was_linearized;
      if (as_state) {
        PoseVelBiasState<Scalar> st;
        st.t_ns = s.t_ns;
        st.T_w_i = lin_replay ? s.T_w_i_lin : s.T_w_i;
        st.vel_w_i = s.vel;
        st.bias_gyro = s.bg;
        st.bias_accel = s.ba;
        if (lin_replay) {
          st.vel_w_i -= s.delta.template segment<3>(6);
          st.bias_gyro -= s.delta.template segment<3>(9);
          st.bias_accel -= s.delta.template segment<3>(12);
        }
        frame_states[s.t_ns] = PoseVelBiasStateWithLin<Scalar>(st);
        if (in_prior) {
          frame_states[s.t_ns].setLinTrue();
          if (lin_replay) frame_states[s.t_ns].applyInc(s.delta);
        }
        res_states.insert(s.t_ns);
      } else {
        frame_poses[s.t_ns] = PoseStateWithLin<Scalar>(
            s.t_ns, lin_replay ? s.T_w_i_lin : s.T_w_i);
        if (in_prior) {
          frame_poses[s.t_ns].setLinTrue();
          if (lin_replay)
            frame_poses[s.t_ns].applyInc(s.delta.template head<6>());
        }
        res_poses.insert(s.t_ns);
      }
    }
  }
  std::vector<int> res_lm_ids;
  std::vector<std::pair<int, TimeCamId>> res_obs;
  for (const auto& ev : xrv_marg_events) {
    /* aom membership rule (the aom itself is built in step 2): all
     * poses; states only if resurrected, promoted, or the current
     * prior-connectivity state */
    auto xr_aom_frame = [&](int64_t f) {
      if (frame_poses.count(f)) return true;
      if (!frame_states.count(f)) return false;
      return res_states.count(f) > 0 || promoted_saved.count(f) > 0 ||
             marg_data.order.abs_order_map.count(f) > 0;
    };
    for (const auto& hl : ev.host_lms) {
      if (lmdb.landmarkExists(hl.first)) continue;
      const int64_t host = hl.second.host_kf_id.frame_id;
      if (!frame_poses.count(host) && !frame_states.count(host)) continue;
      /* addLandmark copies neither obs nor the host->target index —
       * every observation must go through addObservation; >=2 in-aom
       * obs required (1-obs landmark QR is degenerate). */
      int n_ok = 0;
      for (const auto& ob : hl.second.obs)
        if (xr_aom_frame(ob.first.frame_id)) n_ok++;
      if (n_ok < 2) continue;
      lmdb.addLandmark(hl.first, hl.second);
      for (const auto& ob : hl.second.obs) {
        if (!xr_aom_frame(ob.first.frame_id)) continue;
        KeypointObservation<Scalar> ko;
        ko.kpt_id = hl.first;
        ko.pos = ob.second;
        lmdb.addObservation(ob.first, ko);
      }
      res_lm_ids.push_back(hl.first);
    }
    for (const auto& ob : ev.dropped_obs) {
      if (!lmdb.landmarkExists(ob.lm_id)) continue;
      if (!frame_poses.count(ob.tcid.frame_id) &&
          !frame_states.count(ob.tcid.frame_id))
        continue;
      auto& lm = lmdb.getLandmark(ob.lm_id);
      if (lm.obs.count(ob.tcid)) continue;
      KeypointObservation<Scalar> ko;
      ko.kpt_id = ob.lm_id;
      ko.pos = ob.pos;
      lmdb.addObservation(ob.tcid, ko);
      res_obs.push_back({ob.lm_id, ob.tcid});
    }
  }

  /* ---- 2. AOM: the oldest prior's vars in ITS order, then the alive
   *         vars created since ---- */
  AbsOrderMap aom;
  auto add_var = [&](int64_t id, int size) {
    if (aom.abs_order_map.count(id)) return;
    aom.abs_order_map[id] = std::make_pair(aom.total_size, size);
    aom.total_size += size;
    aom.items++;
  };
  bool aom_ok = true;
  for (const auto& kv : ev0.prior_before.order.abs_order_map) {
    const int64_t id = kv.first;
    const int size = kv.second.second;
    if (!frame_poses.count(id) && !frame_states.count(id)) aom_ok = false;
    add_var(id, size);
  }
  for (const auto& kv : frame_poses)
    add_var(kv.first, POSE_SIZE);
  /* alive states: only the prior-connectivity state — stock keeps the
   * young sliding states OUT of the marg prior (their FEJ flag is not
   * set); plus resurrected and promoted vars */
  for (const auto& kv : frame_states) {
    if (res_states.count(kv.first) || promoted_saved.count(kv.first) ||
        marg_data.order.abs_order_map.count(kv.first))
      add_var(kv.first, POSE_VEL_BIAS_SIZE);
  }
  if (!aom_ok) {
    /* a prior var neither alive nor captured: bail out cleanly */
    for (const int64_t id : res_poses) frame_poses.erase(id);
    for (const int64_t id : res_states) frame_states.erase(id);
    for (const auto& ro : res_obs)
      if (lmdb.landmarkExists(ro.first))
        lmdb.removeObservations(ro.first, {ro.second});
    for (const int id : res_lm_ids)
      if (lmdb.landmarkExists(id)) lmdb.removeLandmark(id);
    for (auto& pr : promoted_saved) {
      frame_states.erase(pr.first);
      frame_poses.emplace(pr.first, pr.second);
    }
    std::cerr << "[xr] READVANCE bail: prior var unresolvable" << std::endl;
    return;
  }

  /* ---- 3. re-linearize prior + captured IMU + all reachable landmark
   *         observations with the estimator's own machinery ---- */
  typename LinearizationBase<Scalar, POSE_SIZE>::Options lqr_options;
  lqr_options.lb_options.huber_parameter = huber_thresh;
  lqr_options.lb_options.obs_std_dev = obs_std_dev;
  lqr_options.linearization_type = config.vio_linearization_type;

  ImuLinData<Scalar> ild = {g, gyro_bias_sqrt_weight,
                            accel_bias_sqrt_weight, {}};
  /* ImuBlock::linearizeImu requires BOTH endpoints as 15-dof
   * frame_states (it .at()s the states map) — include a factor only
   * when that holds; pose-demoted endpoints' inertial info remains in
   * the base prior (v1 approximation). */
  auto xr_imu_ok = [&](int64_t a, int64_t b) {
    auto ia = aom.abs_order_map.find(a);
    auto ib = aom.abs_order_map.find(b);
    return ia != aom.abs_order_map.end() &&
           ib != aom.abs_order_map.end() &&
           ia->second.second == POSE_VEL_BIAS_SIZE &&
           ib->second.second == POSE_VEL_BIAS_SIZE &&
           frame_states.count(a) > 0 && frame_states.count(b) > 0;
  };
  /* captured (dead) factors ONLY — live imu factors stay live in
   * optimize(); folding them into the prior would double-count */
  for (const auto& ev : xrv_marg_events)
    for (const auto& im : ev.imu) {
      int64_t s0 = im.second.get_start_t_ns();
      int64_t s1 = s0 + im.second.get_dt_ns();
      if (ild.imu_meas.count(im.first)) continue;
      if (xr_imu_ok(s0, s1)) ild.imu_meas[im.first] = &im.second;
    }

  std::set<int64_t> dead_kfs;
  for (const int64_t id : res_poses) dead_kfs.insert(id);
  for (const int64_t id : res_states) dead_kfs.insert(id);
  int64_t xr_last = dead_kfs.empty() ? 0 : *dead_kfs.rbegin();

  std::unordered_set<KeypointId> no_lost;
  std::set<FrameId> no_fixed;
  auto lqr = LinearizationBase<Scalar, POSE_SIZE>::create(
      this, aom, lqr_options,
      const_cast<MargLinData<Scalar>*>(&ev0.prior_before), &ild, &dead_kfs,
      &no_lost, xr_last, &no_fixed);
  lqr->linearizeProblem();
  lqr->performQR();

  MatX Q2Jp_or_H;
  VecX Q2r_or_b;
  if (marg_data.is_sqrt)
    lqr->get_dense_Q2Jp_Q2r(Q2Jp_or_H, Q2r_or_b);
  else
    lqr->get_dense_H_b(Q2Jp_or_H, Q2r_or_b);

  /* ---- 4. re-marginalize the resurrected columns ---- */
  std::set<int> idx_to_keep, idx_to_marg;
  for (const auto& kv : aom.abs_order_map) {
    const int start = kv.second.first;
    const int size = kv.second.second;
    const bool dead = dead_kfs.count(kv.first) > 0;
    const bool prom = promoted_saved.count(kv.first) > 0;
    for (int i = 0; i < size; i++) {
      const bool m = dead || (prom && i >= (int)POSE_SIZE);
      (m ? idx_to_marg : idx_to_keep).emplace(start + i);
    }
  }
  MatX marg_H_new;
  VecX marg_b_new;
  if (marg_data.is_sqrt)
    MargHelper<Scalar>::marginalizeHelperSqrtToSqrt(
        Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
        marg_b_new);
  else
    MargHelper<Scalar>::marginalizeHelperSqToSq(
        Q2Jp_or_H, Q2r_or_b, idx_to_keep, idx_to_marg, marg_H_new,
        marg_b_new);

  /* diagnostic: captured-prior fold residual (needs resurrected vars
   * still alive) */
  Scalar xr_r0 = Scalar(0), xr_b0 = Scalar(0);
  {
    VecX d0;
    computeDelta(ev0.prior_before.order, d0);
    xr_r0 = (ev0.prior_before.H * d0 + ev0.prior_before.b).norm();
    xr_b0 = ev0.prior_before.b.norm();
  }

  /* ---- 5. BURY the resurrected objects ---- */
  for (const auto& ro : res_obs)
    if (lmdb.landmarkExists(ro.first))
      lmdb.removeObservations(ro.first, {ro.second});
  for (const int id : res_lm_ids)
    if (lmdb.landmarkExists(id)) lmdb.removeLandmark(id);
  for (const int64_t id : res_poses) frame_poses.erase(id);
  for (const int64_t id : res_states) frame_states.erase(id);
  for (auto& pr : promoted_saved) {
    frame_states.erase(pr.first);
    frame_poses.emplace(pr.first, pr.second);
  }

  /* ---- 6. install the re-advanced prior over the survivors, in the
   *         kept-column order (aom order restricted to survivors) ---- */
  AbsOrderMap new_order;
  for (const auto& kv : aom.abs_order_map) {
    if (dead_kfs.count(kv.first)) continue;
    const int keep_sz = promoted_saved.count(kv.first)
                            ? (int)POSE_SIZE : kv.second.second;
    new_order.abs_order_map[kv.first] =
        std::make_pair(new_order.total_size, keep_sz);
    new_order.total_size += keep_sz;
    new_order.items++;
  }
  if (marg_H_new.cols() != new_order.total_size) {
    std::cerr << "[xr] READVANCE bail: size mismatch " << marg_H_new.cols()
              << " vs " << new_order.total_size << std::endl;
    return;
  }
  {
    static const bool dry = [] {
      const char* e = getenv("XR_READVANCE_DRY");
      return e && *e && *e != '0';
    }();
    if (new_order.total_size == marg_data.order.total_size) {
      MatX Hi = marg_data.is_sqrt
                    ? MatX(marg_data.H.transpose() * marg_data.H)
                    : marg_data.H;
      VecX bi = marg_data.is_sqrt
                    ? VecX(marg_data.H.transpose() * marg_data.b)
                    : marg_data.b;
      MatX Hn = marg_data.is_sqrt ? MatX(marg_H_new.transpose() * marg_H_new)
                                  : marg_H_new;
      VecX bn = marg_data.is_sqrt ? VecX(marg_H_new.transpose() * marg_b_new)
                                  : marg_b_new;
      VecX si = VecX::Zero(new_order.items), sn = VecX::Zero(new_order.items);
      VecX di = VecX::Zero(new_order.items), dn = VecX::Zero(new_order.items);
      int vi = 0;
      for (const auto& kv : new_order.abs_order_map) {
        const int o = kv.second.first, sz = kv.second.second;
        si[vi] = bi.segment(o, sz).norm();
        sn[vi] = bn.segment(o, sz).norm();
        di[vi] = Hi.block(o, o, sz, sz).norm();
        dn[vi] = Hn.block(o, o, sz, sz).norm();
        vi++;
      }
      /* effective residual at the current deltas: what optimize() sees */
      VecX cur_delta;
      computeDelta(new_order, cur_delta);
      const Scalar ri = (marg_data.H * cur_delta + marg_data.b).norm();
      const Scalar rn = (marg_H_new * cur_delta + marg_b_new).norm();
      std::cerr << "[xr] READVANCE diff: |Hi-Hn|=" << (Hi - Hn).norm()
                << " |Hi|=" << Hi.norm() << " r_eff_i=" << ri
                << " r_eff_n=" << rn << " r_fold0=" << xr_r0
                << " |b0|=" << xr_b0 << std::endl;
      std::cerr << "  Hdiag_i=" << di.transpose() << std::endl;
      std::cerr << "  Hdiag_n=" << dn.transpose() << std::endl;
      std::cerr << "  b_i=" << si.transpose() << std::endl;
      std::cerr << "  b_n=" << sn.transpose() << std::endl;
    } else {
      std::cerr << "[xr] READVANCE diff: ORDER MISMATCH inc="
                << marg_data.order.total_size << " new="
                << new_order.total_size << std::endl;
    }
    if (dry) {
      xrv_marg_events.clear();
      return;
    }
  }
  marg_data.H = marg_H_new;
  marg_data.b = marg_b_new;
  marg_data.order = new_order;
  xrv_marg_events.clear();
  static int xr_ra_log = 0;
  if (xr_ra_log < 12) {
    xr_ra_log++;
    std::cerr << "[xr] READVANCE ok: prior rebuilt over "
              << new_order.items << " vars (" << new_order.total_size
              << " dof), " << dead_kfs.size() << " re-marged, "
              << promoted_saved.size() << " promoted" << std::endl;
  }
}

"""
i = t.index(old)
t = t[:i] + engine + t[i:]
f.write_text(t, encoding="utf-8")
print("stage 11 (XR_READVANCE re-advance engine) applied")
