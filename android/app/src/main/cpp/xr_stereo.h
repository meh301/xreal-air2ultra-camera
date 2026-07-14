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
    /* Depth-net rectify: same virtual camera (FOV) as the 240x320 pair — the
     * rectified focal scales with resolution — built DIRECTLY at the model's
     * input size (192x256, f=160). One resample from the sensor instead of the
     * old rectify-to-480x640 + bilinear-downsample two-step: ~2-3.5 ms/pass
     * cheaper, sharper (single filtering), and it exactly matches how the
     * quantization calibration data was rectified. */
    ZDR_W = (XS_W * 4) / 5, ZDR_H = (XS_H * 4) / 5  /* 192x256 */
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

    /* model-res rectify for the stereo depth net: BOTH cameras at ZDR_W x
     * ZDR_H ([0]=left, [1]=right). mapx_hi==0xFFFF => invalid pixel. */
    uint16_t mapx_hi[2][ZDR_W * ZDR_H];
    uint16_t mapy_hi[2][ZDR_W * ZDR_H];
    uint8_t rect_hi[2][ZDR_W * ZDR_H];

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

/* Rectify camera `cam` (0=left, 1=right) at model resolution (ZDR_W x ZDR_H)
 * into s->rect_hi[cam] (bilinear) — the direct inputs for the stereo depth net.
 * `src` is that camera's 480x640 descrambled image (same buffer as rectify). */
void xr_stereo_rectify_hi(xr_stereo *s, int cam, const uint8_t *src);

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
