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
} xr_eye_calib;

/* Rotation-convention variant: bit 0 = conjugate the display quaternion,
 * bit 1 = conjugate the camera quaternion. Variant 2 (camera conjugated,
 * display not) was verified on-device — the passthrough lines up with the
 * real world through the glasses. */
enum { XR_ALIGN_VARIANT_DEFAULT = 2 };

/* Build the sample map for one eye: out_idx[y*w + x] = index into the
 * 480x640 camera image (or -1 for out-of-view). (w, h) is the rendered
 * per-eye size; full_w/full_h the calibrated display resolution the K matrix
 * refers to (1920x1080). depth_m <= 0 means infinity. */
void xr_align_build(const xr_eye_calib *eye, int variant, int w, int h,
                    int full_w, int full_h, int32_t *out_idx);

#endif
