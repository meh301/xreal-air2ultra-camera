/* xr_map.h — per-session map layer on top of the Basalt odometry.
 *
 * GLIM-style module split (odometry / local map / global map): Basalt is
 * the drifting odometry and is never touched; this layer keeps a bounded
 * session map — motion-gated keyframes with descriptors and the landmark
 * geometry — pruned by a history timeout so memory stays flat. Loop /
 * relocalization candidates come from descriptor matching; the geometric
 * verification and the T_session<-odom correction are the next stage.
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

/* The most recent match's keyframe landmarks (SESSION-frame world xyz,
 * n x 3 floats — stored coords through the matched node's correction) —
 * the AR loop/reloc flash. Returns the count copied (<= max). Their
 * offset from the live cloud of the same physical spot is the residual
 * drift the pose graph has not yet absorbed. */
int xr_map_loop_points(float *xyz, int max);

/* Live session correction D (q wxyz, p): session_pose = D ∘ odom_pose,
 * session_point = R(D)·p_odom + p(D). Identity until the first VERIFIED
 * loop closure (3D-3D RANSAC over matched landmark pairs). In mapping
 * mode a closure also relaxes the keyframe chain (per-keyframe pose
 * graph, error distributed by odom path length); in localization-only
 * mode it relocalizes the live pose against the frozen map. Returns a
 * generation counter (bumps on every update; also on reset). */
int xr_map_get_correction(float q[4], float p[3]);

#endif
