/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include <deque>

#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <basalt/imu/preintegration.h>
#include <basalt/utils/vis_matrices.h>
#include <basalt/utils/time_utils.hpp>

#include <basalt/vi_estimator/sqrt_ba_base.h>
#include <basalt/vi_estimator/vio_estimator.h>

#include <basalt/imu/preintegration.h>

namespace basalt {

template <class Scalar_>
class SqrtKeypointVioEstimator : public VioEstimatorBase, public SqrtBundleAdjustmentBase<Scalar_> {
 public:
  using Scalar = Scalar_;

  typedef std::shared_ptr<SqrtKeypointVioEstimator> Ptr;

  static const int N = 9;
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using VecN = Eigen::Matrix<Scalar, N, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
  using MatN3 = Eigen::Matrix<Scalar, N, 3>;
  using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using SE3 = Sophus::SE3<Scalar>;
  using UIMAT = vis::UIMAT;

  using typename SqrtBundleAdjustmentBase<Scalar>::RelLinData;
  using typename SqrtBundleAdjustmentBase<Scalar>::AbsLinData;

  using BundleAdjustmentBase<Scalar>::computeError;
  using BundleAdjustmentBase<Scalar>::get_current_points;
  using BundleAdjustmentBase<Scalar>::computeDelta;
  using BundleAdjustmentBase<Scalar>::computeProjections;
  using BundleAdjustmentBase<Scalar>::triangulate;
  using BundleAdjustmentBase<Scalar>::backup;
  using BundleAdjustmentBase<Scalar>::restore;
  using BundleAdjustmentBase<Scalar>::getPoseStateWithLin;
  using BundleAdjustmentBase<Scalar>::computeModelCostChange;

  using SqrtBundleAdjustmentBase<Scalar>::linearizeHelper;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeAbsHelper;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeRel;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeAbs;
  using SqrtBundleAdjustmentBase<Scalar>::updatePoints;
  using SqrtBundleAdjustmentBase<Scalar>::updatePointsAbs;
  using SqrtBundleAdjustmentBase<Scalar>::linearizeMargPrior;
  using SqrtBundleAdjustmentBase<Scalar>::computeMargPriorError;
  using SqrtBundleAdjustmentBase<Scalar>::computeMargPriorModelCostChange;
  using SqrtBundleAdjustmentBase<Scalar>::checkNullspace;
  using SqrtBundleAdjustmentBase<Scalar>::checkEigenvalues;

  SqrtKeypointVioEstimator(const Eigen::Vector3d& g, const basalt::Calibration<double>& calib, const VioConfig& config);

  void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i, const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  void scheduleResetState() override;
  bool resetState(typename IntegratedImuMeasurement<Scalar>::Ptr& meas);

  void initialize(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) override;

  virtual ~SqrtKeypointVioEstimator() { maybe_join(); }

  inline void maybe_join() override {
    if (processing_thread) {
      processing_thread->join();
      processing_thread.reset();
    }
  }

  void addIMUToQueue(const ImuData<double>::Ptr& data) override;
  void addVisionToQueue(const OpticalFlowResult::Ptr& data) override;
  void takeLongTermKeyframe() override;

  /* XREAL map->VIO tight coupling (see VioEstimatorBase). Thread-safe. */
  void setXrPosePrior(const Sophus::SE3d& E, double sigma_t, double sigma_r, int64_t expiry_t_ns) override {
    std::lock_guard<std::mutex> l(xr_prior_mutex);
    xr_prior_E = E;
    xr_prior_sigma_t = sigma_t;
    xr_prior_sigma_r = sigma_r;
    xr_prior_expiry = expiry_t_ns;
    xr_prior_active = true;
  }
  std::mutex xr_prior_mutex;
  Sophus::SE3d xr_prior_E;
  double xr_prior_sigma_t = 0.07, xr_prior_sigma_r = 0.035;
  int64_t xr_prior_expiry = 0;
  bool xr_prior_active = false;

  /* XREAL stage-3 tight coupling (see VioEstimatorBase): fixed-3D map
   * landmark reprojection factors, buffered per frame timestamp like the
   * pose priors. Thread-safe setter; applied in optimize() while the frame
   * is in the window, expired by frame time. */
  static constexpr int XR_LM_MAX_FACTORS = 96;          /* per frame */
  static constexpr int XR_LM_MAX_BATCHES = 24;          /* kf batches persist */
  static constexpr int64_t XR_LM_EXPIRY_NS = 500000000; /* ~0.5 s */
  struct XrLmBatch {
    int64_t t_ns;
    int cam_id;
    int n;
    float sigma_px;
    float uv[2 * XR_LM_MAX_FACTORS];
    float xyz[3 * XR_LM_MAX_FACTORS];
  };
  void setXrLandmarkFactors(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world, int n,
                            float sigma_px) override {
    if (!uv || !xyz_world || n <= 0) return;
    if (n > XR_LM_MAX_FACTORS) n = XR_LM_MAX_FACTORS;
    std::lock_guard<std::mutex> l(xr_lm_mutex);
    XrLmBatch* slot = nullptr;
    for (auto& bch : xr_lm_batches)  /* re-post for the same frame+cam replaces */
      if (bch.t_ns == t_ns && bch.cam_id == cam_id) slot = &bch;
    if (!slot) {
      if (xr_lm_batches.size() >= XR_LM_MAX_BATCHES) xr_lm_batches.erase(xr_lm_batches.begin());
      xr_lm_batches.emplace_back();
      slot = &xr_lm_batches.back();
    }
    slot->t_ns = t_ns;
    slot->cam_id = cam_id;
    slot->n = n;
    /* SIGN carries persist(+)/transient(-) — never rewrite it.
     * Magnitude may carry the +1000 room-class offset (stage 6). */
    slot->sigma_px =
        (sigma_px != 0.0f && sigma_px > -1.0e4f && sigma_px < 1.0e4f)
            ? sigma_px : 2.0f;
    std::memcpy(slot->uv, uv, sizeof(float) * 2 * (size_t)n);
    std::memcpy(slot->xyz, xyz_world, sizeof(float) * 3 * (size_t)n);
  }
  std::mutex xr_lm_mutex;
  std::vector<XrLmBatch> xr_lm_batches;

  /* XREAL stage-8 injection queue (consumed in measure()) */
  struct XrInjItem {
    int64_t t_ns; int cam_id; float u, v; float x, y, z; int32_t id;
  };
  std::mutex xr_inj_mutex;
  std::vector<XrInjItem> xr_inj_queue;
  /* per-landmark pending obs until two distinct frames exist (QR needs
   * >=2 observation rows) */
  std::map<int, std::vector<XrInjItem>> xr_inj_pending;
  /* stage-8d anchor registry: odom-frame 3D per injected map landmark,
   * matched against live flow keypoints EVERY frame (covisibility) */
  struct XrInjLm {
    Scalar x, y, z;
    int created;
    int nm;
    int64_t m_ns[2];
    Scalar m_uv[2][2];
  };
  std::map<int, XrInjLm> xr_inj_lms;

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
  /* XR_TRIPAR: minimum-parallax gate on new-landmark triangulation. */
  static bool xr_tripar_on() {
    static const bool v = [] {
      const char* e = getenv("XR_TRIPAR");
      return e && *e && *e != '0';
    }();
    return v;
  }
  static double xr_tripar_deg() {
    static const double v = [] {
      const char* e = getenv("XR_TRIPAR_DEG");
      double d = (e && *e) ? atof(e) : 1.0;
      return (d > 0.0) ? d : 1.0;
    }();
    return v;
  }
  static bool xrv_revive_radv_on() {
    static const bool v = [] {
      const char* e = getenv("XR_REVIVE_RADV");
      return e && *e && *e != '0';
    }();
    return v;
  }
  bool xrv_readvance_pending = false;   /* set cross-thread by the map
      layer's closure trigger; consumed once per marg — a lost write
      merely delays the rebuild one event (benign) */
  void xrvReadvance();
  void setXrReadvance() override {
    if (xrv_delaymarg_on()) xrv_readvance_pending = true;
  }
  static bool xrv_delaymarg_on() {
    static const bool v = [] {
      const char* e = getenv("XR_DELAYMARG");
      return e && *e && *e != '0';
    }();
    return v;
  }
  void setXrInjectLandmarks(int64_t t_ns, int cam_id, const float* uv, const float* xyz_world,
                            const int32_t* ids, int n) override {
    if (!uv || !xyz_world || !ids || n <= 0) return;
    std::lock_guard<std::mutex> l(xr_inj_mutex);
    if (xr_inj_queue.size() > 4096) return;      /* bounded backlog */
    for (int i = 0; i < n; i++)
      xr_inj_queue.push_back({t_ns, cam_id, uv[2 * i], uv[2 * i + 1],
                              xyz_world[3 * i], xyz_world[3 * i + 1],
                              xyz_world[3 * i + 2], ids[i]});
  }



  typename ImuData<Scalar>::Ptr popFromImuDataQueue();

  bool measure(const OpticalFlowResult::Ptr& opt_flow_meas, const typename IntegratedImuMeasurement<Scalar>::Ptr& meas);

  // int64_t propagate();
  // void addNewState(int64_t data_t_ns);

  bool optimize_and_marg(const std::map<int64_t, int>& num_points_connected,
                         const std::unordered_set<KeypointId>& lost_landmaks);

  bool marginalize(const std::map<int64_t, int>& num_points_connected,
                   const std::unordered_set<KeypointId>& lost_landmaks);
  bool optimize();

  bool show_uimat(UIMAT m) const;

  void debug_finalize() override;

  void logMargNullspace();
  Eigen::VectorXd checkMargNullspace() const;
  Eigen::VectorXd checkMargEigenvalues() const;

  int64_t get_t_ns() const { return frame_states.at(last_state_t_ns).getState().t_ns; }
  const SE3& get_T_w_i() const { return frame_states.at(last_state_t_ns).getState().T_w_i; }
  const Vec3& get_vel_w_i() const { return frame_states.at(last_state_t_ns).getState().vel_w_i; }

  const PoseVelBiasState<Scalar>& get_state() const { return frame_states.at(last_state_t_ns).getState(); }
  PoseVelBiasState<Scalar> get_state(int64_t t_ns) const {
    PoseVelBiasState<Scalar> state;

    auto it = frame_states.find(t_ns);

    if (it != frame_states.end()) { return it->second.getState(); }

    auto it2 = frame_poses.find(t_ns);
    if (it2 != frame_poses.end()) { state.T_w_i = it2->second.getPose(); }

    return state;
  }

  void setMaxStates(size_t val) override { max_states = val; }
  void setMaxKfs(size_t val) override { max_kfs = val; }

  Eigen::aligned_vector<SE3> getFrameStates() const {
    Eigen::aligned_vector<SE3> res;

    for (const auto& kv : frame_states) { res.push_back(kv.second.getState().T_w_i); }

    return res;
  }

  Eigen::aligned_vector<SE3> getFramePoses() const {
    Eigen::aligned_vector<SE3> res;

    for (const auto& kv : frame_poses) { res.push_back(kv.second.getPose()); }

    return res;
  }

  Eigen::aligned_map<int64_t, SE3> getAllPosesMap() const {
    Eigen::aligned_map<int64_t, SE3> res;

    for (const auto& kv : frame_poses) { res[kv.first] = kv.second.getPose(); }

    for (const auto& kv : frame_states) { res[kv.first] = kv.second.getState().T_w_i; }

    return res;
  }

  Sophus::SE3d getT_w_i_init() override { return T_w_i_init.template cast<double>(); }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  using BundleAdjustmentBase<Scalar>::frame_poses;
  using BundleAdjustmentBase<Scalar>::frame_states;
  using BundleAdjustmentBase<Scalar>::lmdb;
  using BundleAdjustmentBase<Scalar>::obs_std_dev;
  using BundleAdjustmentBase<Scalar>::huber_thresh;
  using BundleAdjustmentBase<Scalar>::calib;

 private:
  OpticalFlowResult::Ptr prev_frame;
  OpticalFlowResult::Ptr curr_frame;

  bool take_kf;
  int frames_after_kf;
  size_t frame_count = 0;
  std::set<int64_t> kf_ids;
  std::set<int64_t> ltkfs;  // Long term keyframes
  bool take_ltkf;           // Whether the next keyframe should be made into ltkfs
  Eigen::aligned_map<int64_t, size_t> frame_idx;

  int64_t last_state_t_ns;
  Eigen::aligned_map<int64_t, IntegratedImuMeasurement<Scalar>> imu_meas;

  const Vec3 g;

  // Input

  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

  /* stage 16 (XR_KFPAR): cam0 keypoint pixels at the last keyframe, for
   * the median-parallax keyframe trigger. */
  std::map<KeypointId, Eigen::Matrix<Scalar, 2, 1>> xr_last_kf_kps;
  static float xr_kfpar_px() {
    static const float v = [] {
      const char* e = getenv("XR_KFPAR");
      return e && *e ? (float)atof(e) : 0.0f;   /* 0 = disabled (stock) */
    }();
    return v;
  }

  std::map<int64_t, int> num_points_kf;

  // Marginalization
  MargLinData<Scalar> marg_data;

  // Used only for debug and log purporses.
  MargLinData<Scalar> nullspace_marg_data;

  Vec3 gyro_bias_sqrt_weight, accel_bias_sqrt_weight;

  size_t max_states;
  size_t max_kfs;

  SE3 T_w_i_init;

  bool initialized;
  bool opt_started;
  bool schedule_reset;

  VioConfig config;

  constexpr static Scalar vee_factor = Scalar(2.0);
  constexpr static Scalar initial_vee = Scalar(2.0);
  Scalar lambda, min_lambda, max_lambda, lambda_vee;

  std::shared_ptr<std::thread> processing_thread;

  // timing and stats
  ExecutionStats stats_all_;
  ExecutionStats stats_sums_;
};
}  // namespace basalt
