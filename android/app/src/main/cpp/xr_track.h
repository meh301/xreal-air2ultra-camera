/* xr_track.h — lightweight IMU-aided sparse feature tracker.
 *
 * Runs on the rectified left image (240x320): grid-seeded gradient corners,
 * SAD patch tracking with a gyro-predicted search window. This is the live
 * "tracking" front-end for visualization and plumbing; the Basalt backend
 * replaces it behind the same xr_slam interface (Basalt's own front end is
 * likewise a sparse optical flow).
 */
#ifndef XR_TRACK_H
#define XR_TRACK_H

#include <stdint.h>

#include "xr_stereo.h"

enum { XT_MAX = 120 };

typedef struct {
    float x, y;        /* rect-image pixel */
    uint16_t age;      /* frames tracked */
    uint8_t alive;
} xr_feature;

typedef struct {
    xr_feature pt[XT_MAX];
    int count;
    uint8_t prev[XS_W * XS_H];
    int have_prev;
    uint32_t frame;
} xr_track;

void xr_track_reset(xr_track *t);

/* Track into `img` (rectified left, XS_W x XS_H). pred_dx/pred_dy: predicted
 * whole-image shift in pixels from the gyro (rotation flow), used to center
 * the search windows. Returns the number of live features. */
int xr_track_step(xr_track *t, const uint8_t *img, float pred_dx, float pred_dy);

#endif
