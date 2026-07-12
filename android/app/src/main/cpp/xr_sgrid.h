/* xr_sgrid.h — sparse-stereo metric anchors for depth calibration.
 *
 * A cheap alternative to full SGM: block-match a coarse GRID of points in the
 * rectified left image against the right along the epipolar line, keep only the
 * confident, unique, textured matches, and turn each into a metric
 * inverse-depth. These sparse points are the metric backbone for calibrating
 * the monocular (ZipDepth) map — dense enough to cover regions where VIO
 * landmarks are absent, cheap enough to run every frame.
 *
 * Everything is in the RECTIFIED LEFT pixel frame (the same frame ZipDepth runs
 * on), so anchors index straight into the dense map with no coordinate mapping.
 * Convention matches xr_stereo: left (x,y) matches right (x-d, y), z = f*B/d. */
#ifndef XR_SGRID_H
#define XR_SGRID_H

#include <stdint.h>

typedef struct {
    float x, y;        /* rectified-left pixel (where to sample the dense map) */
    float invz;        /* measured metric inverse-depth, 1/metres */
} xr_anchor;

/* Match a grid of points; write up to max_out confident anchors to out, return
 * the count. left/right are w*h 8-bit rectified images; f_rect in pixels,
 * baseline_m in metres. */
int xr_sgrid_match(const uint8_t *left, const uint8_t *right, int w, int h,
                   float f_rect, float baseline_m,
                   xr_anchor *out, int max_out);

/* Project VIO landmarks (world metric 3D) into the SAME rectified-left frame
 * as the grid, giving far-range metric anchors that stereo disparity can't
 * reach. lm_xyz is n_lm world points; (q_iw, p_iw) is the IMU->world pose
 * (Hamilton wxyz); R_rect_imu maps the rectified frame to IMU; p_ic is the left
 * camera centre in the IMU frame (eye_calib.p_cam). Writes up to max_out
 * in-frame, in-front anchors; returns the count. */
int xr_lm_anchors(const float (*lm_xyz)[3], int n_lm,
                  const float q_iw[4], const float p_iw[3],
                  const float R_rect_imu[9], float f_rect, float cx, float cy,
                  const float p_ic[3], int w, int h,
                  xr_anchor *out, int max_out);

#endif
