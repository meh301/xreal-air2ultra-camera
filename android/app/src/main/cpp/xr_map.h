/* xr_map.h — per-session map layer on top of the Basalt odometry.
 *
 * GLIM-style module split (odometry / local map / global map): Basalt is
 * the drifting odometry and is never touched; this layer keeps a bounded
 * session map — motion-gated keyframes with descriptors and anchor-local
 * landmark geometry, on a rolling cap. Loop / relocalization candidates
 * come from descriptor matching; verification is a gravity-aligned 2D-3D
 * PnP (top-K covisibility-pooled clusters); a confirmed closure DEFORMS
 * the keyframe pose graph (4-DoF, error distributed by odom path length)
 * so the drifted map co-localizes with the established one — real closure,
 * not just a live-pose snap. The same anchored graph is the on-disk
 * cloud-map format (save/load below).
 *
 * Descriptors: XFeat (ONNX Runtime, when the model is staged — CVPR'24
 * learned features, far more robust than ORB on these grainy sensors) or
 * the built-in mini-ORB fallback. All heavy work (inference, matching)
 * runs on a dedicated low-priority map thread; the SLAM worker only
 * offers keyframes through a non-blocking mailbox.
 */
#ifndef XR_MAP_H
#define XR_MAP_H

#include <stdint.h>

enum {
    XR_MAP_MAX_KF = 200,       /* session cap */
    XR_MAP_KP_PER_KF = 200,    /* keypoints/descriptors per keyframe */
};

/* Stage the XFeat ONNX model (app files dir); call before streaming.
 * Without it the mini-ORB fallback is used. */
void xr_map_set_model(const char *onnx_path);

/* Register the VPR place-recognition model (xr_vpr): per-keyframe
 * embeddings + top-K appearance retrieval replace the brute candidate
 * scan once frames embed. Optional; no-op path without it. */
void xr_map_set_vpr_model(const char *onnx_path);

/* Runtime descriptor selector: 0 = mini-ORB (default), 1 = XFeat (needs
 * the model staged + ONNX Runtime present). Clears the keyframe store on
 * a real change since the two descriptor types cannot cross-match. */
/* Returns the descriptor in effect afterwards (0 = BAD/TEBLID, 1 = XFeat);
 * a request for XFeat is rejected (returns 0) when ORT/the model is absent. */
int xr_map_set_use_xfeat(int on);

/* 1 when XFeat is actually loaded and usable (model + ORT), else 0. */
int xr_map_xfeat_ready(void);

/* Mapping (default) grows the session map; localization-only freezes it:
 * the current view is still matched against the stored keyframes (the
 * relocalization query) but nothing new is stored. */
void xr_map_set_mapping(int on);

/* Loop RECOVERY on (default) / off. Loop closure itself — candidate
 * detection, geometric verification, keyframe-chain relaxation — always
 * runs; this only gates whether a verified closure SNAPS the live pose
 * (the T_session<-odom update). Off = the future GNSS-fusion mode: the
 * map keeps healing itself, the displayed pose stays odometry-
 * continuous and an external reference owns global placement. */
void xr_map_set_recovery(int on);

/* Note violent motion (shake/loss) from the IMU thread. Drives the
 * confirmed-recovery lifecycle: storage freezes and STAYS frozen until a
 * relocalization is verified — not merely until the shake stops — so a
 * stable-but-wrong post-shake odometry cannot lay a competing map.
 * Relocalization queries keep running throughout. */
void xr_map_freeze_storage(int shaking);

/* Recovery lifecycle for the panel: 0 healthy, 1 lost/relocalizing,
 * 2 recovered (map just healed, storage resuming). */
int xr_map_recovery_state(void);

/* Left-camera geometry for PnP verification (ORB-SLAM-style reloc: the
 * stored map supplies 3D, the query supplies pixels): pixel->camera-ray
 * unprojector plus the camera->IMU extrinsics. Wire once when the SLAM
 * bridge has its calibration. */
void xr_map_set_geom(int (*unproject)(float u, float v, float ray_cam[3]),
                     const float R_ic[9], const float p_ic[3]);

void xr_map_reset(void);

/* Offer a keyframe from the SLAM worker (non-blocking; drops when the
 * map thread is busy). Processed when moved >= 0.3 m or turned >= 15 deg
 * since the last processed one. img = the conditioned left frame
 * (480x640, same feed Basalt sees); landmarks ride along for the future
 * PnP verification. The store is a ROLLING cap: when full, the least-
 * recently-useful keyframe is evicted — usefulness refreshes on loop
 * matches AND on spatial proximity (< 2 m) of the current pose, so
 * staying in the same space never loses its keyframes; only unvisited
 * places roll off. In localization-only mode nothing is stored. */
void xr_map_offer(const float q[4], const float p[3], uint64_t ts_ns,
                  const uint8_t *img,
                  const int32_t *lm_id, const float (*lm_xyz)[3],
                  const float (*lm_uv)[2], int n_lm);

int xr_map_num_keyframes(void);      /* thread-safe (atomic) */

/* Most recent loop/reloc candidate (descriptor stage): returns 1 and the
 * two keyframe timestamps + match count, or 0 when none happened yet. */
int xr_map_last_candidate(uint64_t *ts_a, uint64_t *ts_b, int *matches);

/* Loop/reloc statistics: total accepted candidates this session, and the
 * matched keyframe's stored (odom-frame) position + match count of the
 * most recent one. Returns 1 when at least one has occurred. */
int xr_map_loop_stats(int *count, float pos[3], int *matches);

/* The most recent verification attempt, for on-screen diagnosis: fills
 * the 3D pair and inlier counts and returns the outcome — 0 none yet,
 * 1 below the candidate bar, 2 too few 3D pairs, 3 too few inliers,
 * 4 alignment good but beyond the snap caps, 5 applied. */
int xr_map_verify_stats(int *pairs, int *inliers);

/* Last loop-search cost for the health line: keyframes match-scored,
 * candidate clusters verified, match+PnP microseconds, lock-wait microseconds.
 * Any out pointer may be NULL. */
void xr_map_perf(int *searched, int *candidates, int *match_us, int *lock_us);

/* The most recent match's keyframe landmarks (SESSION-frame world xyz,
 * n x 3 floats — stored coords through the matched node's correction) —
 * the AR loop/reloc flash. Returns the count copied (<= max). Their
 * offset from the live cloud of the same physical spot is the residual
 * drift the pose graph has not yet absorbed. */
int xr_map_loop_points(float *xyz, int max);

/* The authoritative displayed map: the session-frame point cloud DERIVED
 * from the keyframe graph (every landmark through its owning keyframe's
 * corrected pose, deduped by id). Rebuilt whenever the graph changes, so a
 * closure that deforms the keyframes deforms this cloud with them — the
 * visible healing. Feed it to the AR/3D view instead of a flat cloud that
 * would have to be rigidly shifted. Returns the count copied (<= max). */
/* Copy the keyframe-derived display cloud (session frame). *inout_gen is the
 * generation the caller last saw; on return it holds the current generation.
 * Returns the point count, or -1 (no lock taken, buffer untouched) when the
 * cloud is unchanged since that generation. Pass NULL to always copy. */
int xr_map_get_cloud(float *xyz, int max, unsigned *inout_gen);

/* NOTE — cloud-map persistence (save/load of the anchored graph) is the
 * NEXT phase, not here yet. Doing it safely needs SUBMAPS: a loaded map
 * must be an immutable reference; landmark ids namespaced per session
 * (Basalt's ids restart and would collide); the live session kept in its
 * own submap and only REGISTERED against the reference (a single
 * T_ref<-live transform), never chain-deformed into it. Building it before
 * that corrupts the reference on the first cross-session match. */

/* Live session correction D (q wxyz, p): session_pose = D ∘ odom_pose,
 * session_point = R(D)·p_odom + p(D). Identity until the first VERIFIED
 * loop closure (gravity-aligned 2D-3D PnP: the map supplies 3D, the query
 * supplies pixels). A confirmed closure deforms the keyframe pose graph
 * AND, with recovery enabled, snaps this correction so the live pose
 * follows the healed map; with recovery off the map still heals but the
 * correction holds (odometry-continuous, for external global placement).
 * Returns a generation counter (bumps on every update; also on reset). */
int xr_map_get_correction(float q[4], float p[3]);

#endif
