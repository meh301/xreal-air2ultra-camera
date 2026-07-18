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
      if (as_state) {
        PoseVelBiasState<Scalar> st;
        st.t_ns = s.t_ns;
        st.T_w_i = s.T_w_i;
        st.vel_w_i = s.vel;
        st.bias_gyro = s.bg;
        st.bias_accel = s.ba;
        frame_states[s.t_ns] = PoseVelBiasStateWithLin<Scalar>(st);
        res_states.insert(s.t_ns);
      } else {
        frame_poses[s.t_ns] = PoseStateWithLin<Scalar>(s.t_ns, s.T_w_i);
        res_poses.insert(s.t_ns);
      }
    }
  }
  std::vector<int> res_lm_ids;
  std::vector<std::pair<int, TimeCamId>> res_obs;
  for (const auto& ev : xrv_marg_events) {
    for (const auto& hl : ev.host_lms) {
      if (lmdb.landmarkExists(hl.first)) continue;
      const int64_t host = hl.second.host_kf_id.frame_id;
      if (!frame_poses.count(host) && !frame_states.count(host)) continue;
      lmdb.addLandmark(hl.first, hl.second);
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
  for (const auto& kv : frame_states)
    add_var(kv.first, POSE_VEL_BIAS_SIZE);
  if (!aom_ok) {
    /* a prior var neither alive nor captured: bail out cleanly */
    for (const int64_t id : res_poses) frame_poses.erase(id);
    for (const int64_t id : res_states) frame_states.erase(id);
    for (const auto& ro : res_obs)
      if (lmdb.landmarkExists(ro.first))
        lmdb.removeLandmarkObservations(ro.first, {ro.second});
    for (const int id : res_lm_ids)
      if (lmdb.landmarkExists(id)) lmdb.removeLandmark(id);
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
  for (const auto& kv : imu_meas) {
    int64_t s0 = kv.second.get_start_t_ns();
    int64_t s1 = s0 + kv.second.get_dt_ns();
    if (aom.abs_order_map.count(s0) && aom.abs_order_map.count(s1))
      ild.imu_meas[kv.first] = &kv.second;
  }
  for (const auto& ev : xrv_marg_events)
    for (const auto& im : ev.imu) {
      int64_t s0 = im.second.get_start_t_ns();
      int64_t s1 = s0 + im.second.get_dt_ns();
      if (ild.imu_meas.count(im.first)) continue;
      if (aom.abs_order_map.count(s0) && aom.abs_order_map.count(s1))
        ild.imu_meas[im.first] = &im.second;
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
    for (int i = 0; i < size; i++)
      (dead ? idx_to_marg : idx_to_keep).emplace(start + i);
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

  /* ---- 5. BURY the resurrected objects ---- */
  for (const auto& ro : res_obs)
    if (lmdb.landmarkExists(ro.first))
      lmdb.removeLandmarkObservations(ro.first, {ro.second});
  for (const int id : res_lm_ids)
    if (lmdb.landmarkExists(id)) lmdb.removeLandmark(id);
  for (const int64_t id : res_poses) frame_poses.erase(id);
  for (const int64_t id : res_states) frame_states.erase(id);

  /* ---- 6. install the re-advanced prior over the survivors, in the
   *         kept-column order (aom order restricted to survivors) ---- */
  AbsOrderMap new_order;
  for (const auto& kv : aom.abs_order_map) {
    if (dead_kfs.count(kv.first)) continue;
    new_order.abs_order_map[kv.first] =
        std::make_pair(new_order.total_size, kv.second.second);
    new_order.total_size += kv.second.second;
    new_order.items++;
  }
  if (marg_H_new.cols() != new_order.total_size) {
    std::cerr << "[xr] READVANCE bail: size mismatch " << marg_H_new.cols()
              << " vs " << new_order.total_size << std::endl;
    return;
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
              << " dof), " << dead_kfs.size() << " re-marged" << std::endl;
  }
}

"""
i = t.index(old)
t = t[:i] + engine + t[i:]
f.write_text(t, encoding="utf-8")
print("stage 11 (XR_READVANCE re-advance engine) applied")
