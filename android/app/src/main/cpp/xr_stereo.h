/* xr_stereo.h — stereo rectification + depth for AR occlusion.
 *
 * Rectifies the two descrambled 480x640 images into a portrait 240x320
 * pinhole pair whose x-axis is the calibrated baseline (13.7 cm), then runs
 * the standard real-time depth recipe (Hirschmueller's Semi-Global
 * Matching): 9x7 census matching cost, SGM aggregation along 4 paths,
 * uniqueness + left-right consistency checks, subpixel refinement, speckle
 * region filter, median. Depth never needs to run faster than the sensors'
 * 30 Hz: the timewarp carries the result to display cadence.
 */
#ifndef XR_STEREO_H
#define XR_STEREO_H

#include <stdint.h>

#include "xreal_align.h"

enum {
    XS_W = 240, XS_H = 320, XS_DISP = 48,
    XS_SCALE = 4,                       /* disparity output: quarter pixels */
    XS_MAX_OUT = XS_DISP * XS_SCALE,    /* largest possible output value */
    /* Full-resolution LEFT rectify dedicated to ZipDepth: same virtual camera
     * (FOV) as the 240x320 pair — the rectified focal scales with resolution —
     * but sampled at the native 480x640 sensor detail, so the depth model is fed
     * a sharp image instead of a 240x320->384 upscale. */
    ZDR_W = XS_W * 2, ZDR_H = XS_H * 2  /* 480x640 */
};

typedef struct {
    /* rectified-pixel -> source-pixel maps: nearest index (validity + the
     * tracker's dot back-mapping) and 12.4 fixed-point bilinear coords */
    int32_t map[2][XS_W * XS_H];       /* [0]=left(cam1), [1]=right(cam0) */
    uint16_t mapx[2][XS_W * XS_H];
    uint16_t mapy[2][XS_W * XS_H];
    float f_rect, baseline_m;          /* rectified focal px, baseline */
    float R_rect_imu[9];               /* rectified frame -> IMU frame */
    uint8_t rect[2][XS_W * XS_H];      /* rectified images (kept for tracker) */

    /* full-res LEFT rectify for ZipDepth (mapx_hi==0xFFFF => invalid pixel) */
    uint16_t mapx_hi[ZDR_W * ZDR_H];
    uint16_t mapy_hi[ZDR_W * ZDR_H];
    uint8_t rect_hi[ZDR_W * ZDR_H];

    /* scratch (large — the owner should be static/heap, not stack) */
    uint64_t census[2][XS_W * XS_H];
    uint8_t cost[XS_W * XS_H * XS_DISP];
    uint16_t sum[XS_W * XS_H * XS_DISP];
    uint16_t lrow[2][XS_W * XS_DISP];  /* vertical-path row buffers */
    uint16_t lmin[XS_W];
    uint8_t disp_int[XS_W * XS_H];     /* integer WTA (LR + speckle) */
    uint8_t disp_q[XS_W * XS_H];       /* quarter-pixel, pre-median */
    uint8_t dispR[XS_W * XS_H];        /* right-image WTA for the LR check */
    int32_t stack[XS_W * XS_H];        /* speckle flood-fill */
    int32_t region[XS_W * XS_H];
} xr_stereo;

/* Build the rectification from the factory calibration (left = the eye
 * calib holding cam1/device_1, right = cam0/device_2), plus the camera
 * positions in the IMU frame. */
void xr_stereo_init(xr_stereo *s,
                    const xr_eye_calib *left, const xr_eye_calib *right,
                    const float p_left[3], const float p_right[3],
                    int variant);

/* Rectify one camera image (cam 0 = left) into s->rect[cam], bilinear. */
void xr_stereo_rectify(xr_stereo *s, int cam, const uint8_t *src);

/* Rectify the LEFT camera image at full 480x640 resolution into s->rect_hi
 * (bilinear) — the sharp input for ZipDepth. `src` is the 480x640 descrambled
 * left image (the same buffer passed as cam 0 to xr_stereo_rectify). */
void xr_stereo_rectify_hi(xr_stereo *s, const uint8_t *src);

/* Compute the disparity map from the rectified pair (call
 * xr_stereo_rectify for both cameras first).
 * out_disp: XS_W*XS_H, 0 = invalid, else disparity in QUARTER pixels. */
void xr_stereo_depth(xr_stereo *s, uint8_t *out_disp);

/* depth [m] for a quarter-pixel disparity value (0 -> 0). */
static inline float xr_stereo_z(const xr_stereo *s, uint8_t disp_q) {
    return disp_q ? s->f_rect * s->baseline_m * (float)XS_SCALE / (float)disp_q
                  : 0.0f;
}

#endif
