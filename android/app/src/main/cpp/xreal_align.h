/* xreal_align.h — 1:1 world-aligned passthrough mapping (C port of
 * python/xreal_align.py). Each eye of the glasses is its own virtual display
 * (~42x25 deg, calibrated intrinsics + pose); for every display pixel we cast
 * a ray, rotate it display->IMU->same-side camera and project through the
 * fisheye624 model, giving a precomputed sample index into the descrambled
 * 480x640 camera image.
 */
#ifndef XREAL_ALIGN_H
#define XREAL_ALIGN_H

#include <stdint.h>

/* per-eye numbers lifted from the factory calibration JSON */
typedef struct {
    float K[9];        /* k_{left,right}_display, row-major 3x3 */
    float q_disp[4];   /* target_q_{eye}_display, xyzw */
    float q_cam[4];    /* imu_q_cam of the same-side camera, xyzw */
    float fc[2], cc[2], kc[12];   /* fisheye624 of that camera */
    float p_cam[3];    /* imu_p_cam of that camera [m] (stereo baseline) */
} xr_eye_calib;

/* Rotation-convention variant: bit 0 = conjugate the display quaternion,
 * bit 1 = conjugate the camera quaternion. Variant 2 (camera conjugated,
 * display not) was verified on-device — the passthrough lines up with the
 * real world through the glasses. */
enum { XR_ALIGN_VARIANT_DEFAULT = 2 };

/* Stage 1: display pixel (calibrated 1920x1080 coordinates) -> ray in the
 * IMU frame (not normalized). Static per mesh vertex. */
void xr_align_ray(const xr_eye_calib *eye, int variant,
                  float u_disp, float v_disp, float ray_imu[3]);

/* Stage 2: IMU-frame ray -> camera pixel via the fisheye model. Called per
 * frame when timewarp rotates the rays. Returns 0, or -1 if the ray points
 * behind the camera. */
int xr_align_project(const xr_eye_calib *eye, int variant,
                     const float ray_imu[3], float *u_cam, float *v_cam);

/* Both stages composed (no timewarp). */
int xr_align_uv(const xr_eye_calib *eye, int variant,
                float u_disp, float v_disp, float *u_cam, float *v_cam);

/* Inverse of stage 1: IMU-frame ray -> display pixel (calibrated 1920x1080
 * coordinates) of this eye. Used to overlay world points/landmarks on the
 * passthrough. Returns 0, or -1 if the ray points behind the display. */
int xr_align_ray_to_display(const xr_eye_calib *eye, int variant,
                            const float ray_imu[3], float *u_disp, float *v_disp);

/* Build the sample map for one eye: out_idx[y*w + x] = index into the
 * 480x640 camera image (or -1 for out-of-view). (w, h) is the rendered
 * per-eye size; full_w/full_h the calibrated display resolution the K matrix
 * refers to (1920x1080). */
void xr_align_build(const xr_eye_calib *eye, int variant, int w, int h,
                    int full_w, int full_h, int32_t *out_idx);

#endif
