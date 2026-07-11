/* xr_slam.h — Basalt VIO behind Monado's VIT interface, loaded at runtime.
 *
 * libbasalt.so (built by android/build_basalt.ps1, packaged in jniLibs) is
 * dlopen'd on demand; when it is absent — e.g. an armeabi-v7a build — every
 * entry point degrades gracefully and the app falls back to the built-in
 * tracker. The bridge feeds the raw (non-equalized) camera frames plus the
 * 1 kHz IMU, and returns 6-DoF poses and the tracked features.
 *
 * Calibration is passed programmatically: the factory fisheye624 intrinsics
 * are refit to Kannala-Brandt (kb4) — the radial theta-polynomial maps
 * exactly, the tiny tangential/thin-prism terms are dropped — and the
 * camera-in-IMU transforms come straight from imu_q_cam / imu_p_cam.
 */
#ifndef XR_SLAM_H
#define XR_SLAM_H

#include <stdint.h>

#include "xreal_align.h"

enum { XR_SLAM_MAX_FEATURES = 256 };

typedef struct {
    uint64_t ts;            /* pose timestamp, IMU clock ns */
    float q[4];             /* orientation, IMU in world, Hamilton wxyz */
    float p[3];             /* position [m], world */
    float v[3];             /* linear velocity [m/s] */
    int n_features;         /* landmarks projected into cam0 (left) */
    float feat_uv[XR_SLAM_MAX_FEATURES][2];   /* left-camera pixels */
    float feat_ray[XR_SLAM_MAX_FEATURES][3];  /* IMU-frame rays */
    int32_t feat_id[XR_SLAM_MAX_FEATURES];    /* stable feature ids */
    int n_features_r;       /* landmarks projected into cam1 (right) */
    float feat_uv_r[XR_SLAM_MAX_FEATURES][2]; /* right-camera pixels */
    /* triangulated landmarks in WORLD coordinates (from the estimator's
     * inverse depths) with their stable ids — the raw material of a map */
    int n_landmarks;
    int32_t lm_id[XR_SLAM_MAX_FEATURES];
    float lm_xyz[XR_SLAM_MAX_FEATURES][3];
    float lm_uv[XR_SLAM_MAX_FEATURES][2];  /* their left-cam pixels */
} xr_slam_state;

/* dlopen libbasalt.so and resolve the VIT symbols; idempotent.
 * Returns 1 when the backend is available. */
int xr_slam_load(void);

/* Path to the unified Basalt config file (num-threads, VioConfig path —
 * written by the app into its files dir). Call before xr_slam_start;
 * without it Basalt runs on library defaults (all cores, desktop tuning). */
void xr_slam_set_config(const char *unified_config_path);

/* Create + start the tracker from the factory calibration (left = the eye
 * calib holding cam1, right = cam0). gyro_bias [rad/s] / accel_bias [m/s^2]
 * are the factory bias vectors (the raw stream is uncorrected!) and noises
 * = {gyro_noise, gyro_bias_rw, accel_noise, accel_bias_rw} from the blob's
 * imu_noises; pass NULL to fall back to generic defaults.
 * Returns 0 on success. */
int xr_slam_start(const xr_eye_calib *left, const xr_eye_calib *right,
                  int variant, const float gyro_bias[3],
                  const float accel_bias[3], const float noises[4]);

void xr_slam_stop(void);
void xr_slam_reset(void);
int xr_slam_running(void);

/* Left-camera geometry for the session map's PnP relocalization: unit
 * ray (CAMERA frame) for a pixel, and the camera->IMU extrinsics.
 * Return 0 when available (after xr_slam_start configured the kb4 fit). */
int xr_slam_unproject0(float u, float v, float ray_cam[3]);
int xr_slam_cam0_geom(float R_ic[9], float p_ic[3]);

/* Push one IMU sample (single producer: the IMU drain thread).
 * gyro in deg/s and accel in g are converted internally. */
void xr_slam_push_imu(uint64_t ts_ns, const float gyro_dps[3],
                      const float accel_g[3]);

/* Push a stereo pair (single producer: the UVC callback thread).
 * left/right are the descrambled non-equalized 480x640 images. */
void xr_slam_push_pair(const uint8_t *left, const uint8_t *right,
                       uint64_t ts_ns);

/* Drain the pose queue; keeps the newest. Returns 1 if `out` was updated
 * with a fresh state, 0 if nothing new arrived. */
int xr_slam_poll(xr_slam_state *out);

#endif
