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

void xr_map_reset(void);

/* Offer a keyframe from the SLAM worker (non-blocking; drops when the
 * map thread is busy). Stored when moved >= 0.3 m or turned >= 15 deg
 * since the last stored keyframe. img = the processed left frame
 * (480x640, same feed Basalt sees). Landmarks (odom-frame xyz + left-cam
 * uv + stable ids) ride along for the future PnP verification. */
void xr_map_offer(const float q[4], const float p[3], uint64_t ts_ns,
                  const uint8_t *img,
                  const int32_t *lm_id, const float (*lm_xyz)[3],
                  const float (*lm_uv)[2], int n_lm);

int xr_map_num_keyframes(void);      /* thread-safe (atomic) */

/* Most recent loop/reloc candidate (descriptor stage): returns 1 and the
 * two keyframe timestamps + match count, or 0 when none happened yet. */
int xr_map_last_candidate(uint64_t *ts_a, uint64_t *ts_b, int *matches);

#endif
