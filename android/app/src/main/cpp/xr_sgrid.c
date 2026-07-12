/* xr_sgrid.c — see xr_sgrid.h. */
#include "xr_sgrid.h"

#include <stdlib.h>

#define SG_STEP     16     /* grid spacing (px) */
#define SG_HALF     3      /* match window half-size -> 7x7 */
#define SG_DMAX     47     /* max disparity searched */
#define SG_DMIN     1      /* d=0 is depth=inf; skip it */
#define SG_UNIQ_NUM 5      /* uniqueness: best*6/5 must still beat 2nd-best */
#define SG_UNIQ_DEN 6
#define SG_MIN_GRAD 200    /* min horizontal texture (SAD vs 1px shift) */

/* SAD of the (2*half+1)^2 window centred at (lx,ly) in L vs (rx,ry) in R. */
static int sad_win(const uint8_t *L, const uint8_t *R, int w,
                   int lx, int ly, int rx, int ry, int half) {
    int s = 0;
    for (int dy = -half; dy <= half; dy++) {
        const uint8_t *lr = L + (size_t)(ly + dy) * w + lx;
        const uint8_t *rr = R + (size_t)(ry + dy) * w + rx;
        for (int dx = -half; dx <= half; dx++) {
            int diff = (int)lr[dx] - (int)rr[dx];
            s += diff < 0 ? -diff : diff;
        }
    }
    return s;
}

int xr_sgrid_match(const uint8_t *left, const uint8_t *right, int w, int h,
                   float f_rect, float baseline_m,
                   xr_anchor *out, int max_out) {
    if (!left || !right || max_out <= 0 || f_rect <= 0.0f || baseline_m <= 0.0f)
        return 0;
    const float fB = f_rect * baseline_m;          /* d/fB = inverse depth */
    const int lo = SG_HALF, hix = w - SG_HALF, hiy = h - SG_HALF;
    int n = 0;
    int cost[SG_DMAX + 1];

    for (int y = lo + SG_STEP / 2; y < hiy && n < max_out; y += SG_STEP) {
        for (int x = lo + SG_STEP / 2; x < hix && n < max_out; x += SG_STEP) {
            /* texture gate: a window with little horizontal structure can't be
             * matched unambiguously along the epipolar, so skip it */
            if (x - 1 < lo) continue;
            if (sad_win(left, left, w, x, y, x - 1, y, SG_HALF) < SG_MIN_GRAD)
                continue;

            /* the right window at x-d must stay in-bounds */
            int dmax = x - lo;
            if (dmax > SG_DMAX) dmax = SG_DMAX;
            if (dmax < SG_DMIN) continue;

            int best = 0x7fffffff, second = 0x7fffffff, bd = -1;
            for (int d = SG_DMIN; d <= dmax; d++) {
                int c = sad_win(left, right, w, x, y, x - d, y, SG_HALF);
                cost[d] = c;
                if (c < best) { second = best; best = c; bd = d; }
                else if (c < second) second = c;
            }
            if (bd < SG_DMIN) continue;
            /* uniqueness: the runner-up must be clearly worse */
            if ((long)best * SG_UNIQ_DEN >= (long)second * SG_UNIQ_NUM) continue;

            /* sub-pixel: parabola through cost[bd-1], cost[bd], cost[bd+1] */
            float dsub = (float)bd;
            if (bd > SG_DMIN && bd < dmax) {
                int cm = cost[bd - 1], c0 = cost[bd], cp = cost[bd + 1];
                int denom = cm - 2 * c0 + cp;
                if (denom > 0) {
                    float off = 0.5f * (float)(cm - cp) / (float)denom;
                    if (off > -1.0f && off < 1.0f) dsub += off;
                }
            }

            out[n].x = (float)x;
            out[n].y = (float)y;
            out[n].invz = dsub / fB;               /* metric inverse-depth */
            n++;
        }
    }
    return n;
}

/* Hamilton wxyz -> 3x3 rotation (row-major), the IMU->world convention the VIO
 * pose uses. */
static void wxyz_rot(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
}

int xr_lm_anchors(const float (*lm_xyz)[3], int n_lm,
                  const float q_iw[4], const float p_iw[3],
                  const float R_rect_imu[9], float f_rect, float cx, float cy,
                  const float p_ic[3], int w, int h,
                  xr_anchor *out, int max_out) {
    if (!lm_xyz || n_lm <= 0 || max_out <= 0 || f_rect <= 0.0f) return 0;
    float Rwi[9];
    wxyz_rot(q_iw, Rwi);                    /* IMU->world */
    const float *Rr = R_rect_imu;          /* rectified->IMU */
    int n = 0;
    for (int i = 0; i < n_lm && n < max_out; i++) {
        /* world -> IMU frame: X_imu = Rwi^T (X_w - p) */
        float d[3] = { lm_xyz[i][0] - p_iw[0], lm_xyz[i][1] - p_iw[1],
                       lm_xyz[i][2] - p_iw[2] };
        float xi[3] = {
            Rwi[0] * d[0] + Rwi[3] * d[1] + Rwi[6] * d[2],
            Rwi[1] * d[0] + Rwi[4] * d[1] + Rwi[7] * d[2],
            Rwi[2] * d[0] + Rwi[5] * d[1] + Rwi[8] * d[2],
        };
        /* relative to the left camera centre (still in IMU axes) */
        xi[0] -= p_ic[0]; xi[1] -= p_ic[1]; xi[2] -= p_ic[2];
        /* IMU -> rectified frame: X_rect = R_rect_imu^T X */
        float xr = Rr[0] * xi[0] + Rr[3] * xi[1] + Rr[6] * xi[2];
        float yr = Rr[1] * xi[0] + Rr[4] * xi[1] + Rr[7] * xi[2];
        float zr = Rr[2] * xi[0] + Rr[5] * xi[1] + Rr[8] * xi[2];
        if (zr <= 0.1f) continue;          /* behind / too close */
        float u = f_rect * xr / zr + cx;
        float v = f_rect * yr / zr + cy;
        if (u < 0.0f || u >= (float)w || v < 0.0f || v >= (float)h) continue;
        out[n].x = u;
        out[n].y = v;
        out[n].invz = 1.0f / zr;
        n++;
    }
    return n;
}
