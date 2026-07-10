/* xr_stereo.h — stereo rectification + depth for AR occlusion.
 *
 * Rectifies the two descrambled 480x640 images into a portrait 240x320
 * pinhole pair whose x-axis is the calibrated baseline (13.7 cm), then runs
 * census-transform block matching (48 disparities) — ~10 ms/pair on one
 * phone core, i.e. comfortable at the sensors' 30 Hz. Depth never needs to
 * run faster: the cameras refresh at 30 Hz and the timewarp carries the
 * result to display cadence.
 */
#ifndef XR_STEREO_H
#define XR_STEREO_H

#include <stdint.h>

#include "xreal_align.h"

enum { XS_W = 240, XS_H = 320, XS_DISP = 48 };

typedef struct {
    /* rectified-pixel -> source-pixel sample maps (nearest), -1 = invalid */
    int32_t map[2][XS_W * XS_H];       /* [0]=left(cam1), [1]=right(cam0) */
    float f_rect, baseline_m;          /* rectified focal px, baseline */
    float R_rect_imu[9];               /* rectified frame -> IMU frame */
    uint32_t census[2][XS_W * XS_H];   /* scratch */
    uint8_t rect[2][XS_W * XS_H];      /* rectified images (kept for tracker) */
    uint8_t disp_raw[XS_W * XS_H];
} xr_stereo;

/* Build the rectification from the factory calibration (left = the eye
 * calib holding cam1/device_1, right = cam0/device_2), plus the camera
 * positions in the IMU frame. */
void xr_stereo_init(xr_stereo *s,
                    const xr_eye_calib *left, const xr_eye_calib *right,
                    const float p_left[3], const float p_right[3],
                    int variant);

/* Rectify both 480x640 images and compute the disparity map.
 * out_disp: XS_W*XS_H, 0 = invalid, else disparity in pixels. */
void xr_stereo_depth(xr_stereo *s, const uint8_t *img_left,
                     const uint8_t *img_right, uint8_t *out_disp);

/* depth [m] for a disparity value (0 -> 0). */
static inline float xr_stereo_z(const xr_stereo *s, uint8_t disp) {
    return disp ? s->f_rect * s->baseline_m / (float)disp : 0.0f;
}

#endif
