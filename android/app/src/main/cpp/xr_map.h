/* xr_map.h — per-session map layer on top of the Basalt odometry.
 *
 * GLIM-style module split (odometry / local map / global map): Basalt is
 * the drifting odometry and is never touched; this layer keeps a bounded
 * session map — motion-gated keyframes with ORB-style descriptors and the
 * landmark geometry — pruned by a history timeout so memory stays flat.
 * It detects loop/relocalization candidates by descriptor matching; the
 * geometric verification and the resulting T_session<-odom correction are
 * the next stage. The fog side consumes the same keyframes long-term
 * (docs/VSLAM.md); session-to-cloud matching happens against that store.
 *
 * Single-threaded: everything is called from the SLAM worker.
 */
#ifndef XR_MAP_H
#define XR_MAP_H

#include <stdint.h>

enum {
    XR_MAP_MAX_KF = 200,       /* session cap */
    XR_MAP_KP_PER_KF = 200,    /* keypoints/descriptors per keyframe */
};

void xr_map_reset(void);

/* Prune keyframes older than timeout_ns (the session "history timeout"). */
void xr_map_tick(uint64_t now_ts_ns, uint64_t timeout_ns);

/* Offer a keyframe: stored when moved >= 0.3 m or turned >= 15 deg since
 * the last stored one. img = descrambled left camera (480x640, cleaned).
 * Landmarks (odom-frame xyz + left-cam uv + stable ids) ride along for
 * the future PnP verification. Returns 1 when a keyframe was stored. */
int xr_map_maybe_keyframe(const float q[4], const float p[3], uint64_t ts_ns,
                          const uint8_t *img,
                          const int32_t *lm_id, const float (*lm_xyz)[3],
                          const float (*lm_uv)[2], int n_lm);

int xr_map_num_keyframes(void);      /* thread-safe (atomic) */

/* Most recent loop/reloc candidate (descriptor stage): returns 1 and the
 * two keyframe timestamps + match count, or 0 when none happened yet. */
int xr_map_last_candidate(uint64_t *ts_a, uint64_t *ts_b, int *matches);

#endif
