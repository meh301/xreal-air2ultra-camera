/* xr_stereo.c — see xr_stereo.h.
 *
 * Depth pipeline per pair: bilinear rectify -> 9x7 census -> hamming cost
 * volume -> SGM along 4 paths (P1/P2 smoothness, the piece plain block
 * matching lacks) -> WTA with uniqueness + subpixel -> left-right
 * consistency -> speckle region filter -> 3x3 median. This is the classic
 * embedded real-time recipe (census + 4-path SGM + LR + speckle).
 */
#include "xr_stereo.h"

#include <math.h>
#include <string.h>

#include "xreal_core.h"

/* rectified pinhole: portrait 240x320, x along the baseline */
#define F_RECT 200.0f
#define CX ((XS_W - 1) * 0.5f)
#define CY ((XS_H - 1) * 0.5f)

/* SGM penalties for the 9x7 census cost (range 0..62): P1 tolerates the
 * 1-disparity steps of slanted surfaces, P2 charges real discontinuities */
#define P1 10
#define P2 120
#define C_PAD 62               /* cost where the window leaves the image */
#define UNIQ_NUM 17            /* reject if 2nd best < best * 17/16 */
#define UNIQ_DEN 16
#define SPECKLE_MIN 60         /* smallest surviving region, px */
#define SPECKLE_TOL 1          /* integer-disparity connectivity */

static void normalize3(float v[3]) {
    float n = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (n > 1e-12f) { v[0] /= n; v[1] /= n; v[2] /= n; }
}

static void cross3(const float a[3], const float b[3], float o[3]) {
    o[0] = a[1] * b[2] - a[2] * b[1];
    o[1] = a[2] * b[0] - a[0] * b[2];
    o[2] = a[0] * b[1] - a[1] * b[0];
}

void xr_stereo_init(xr_stereo *s,
                    const xr_eye_calib *left, const xr_eye_calib *right,
                    const float p_left[3], const float p_right[3],
                    int variant) {
    /* rectified frame: x = baseline direction (left -> right camera),
     * z = mean forward of the two cameras projected off x, y = z cross x */
    float bx[3] = { p_right[0] - p_left[0], p_right[1] - p_left[1],
                    p_right[2] - p_left[2] };
    s->baseline_m = sqrtf(bx[0] * bx[0] + bx[1] * bx[1] + bx[2] * bx[2]);
    normalize3(bx);

    /* forward: use the IMU frame's +z as the nominal look direction (the
     * display/camera z axes are within ~10 deg of it), orthogonalized */
    float fz[3] = { 0, 0, 1 };
    float d = fz[0] * bx[0] + fz[1] * bx[1] + fz[2] * bx[2];
    fz[0] -= d * bx[0]; fz[1] -= d * bx[1]; fz[2] -= d * bx[2];
    normalize3(fz);
    float by[3];
    cross3(fz, bx, by);        /* y = z cross x (right-handed, y down-ish) */
    normalize3(by);

    /* columns of R_rect_imu are the rect axes expressed in the IMU frame */
    s->R_rect_imu[0] = bx[0]; s->R_rect_imu[1] = by[0]; s->R_rect_imu[2] = fz[0];
    s->R_rect_imu[3] = bx[1]; s->R_rect_imu[4] = by[1]; s->R_rect_imu[5] = fz[1];
    s->R_rect_imu[6] = bx[2]; s->R_rect_imu[7] = by[2]; s->R_rect_imu[8] = fz[2];
    s->f_rect = F_RECT;

    const xr_eye_calib *cams[2] = { left, right };
    for (int c = 0; c < 2; c++) {
        for (int y = 0; y < XS_H; y++) {
            for (int x = 0; x < XS_W; x++) {
                float rr[3] = { (x - CX) / F_RECT, (y - CY) / F_RECT, 1.0f };
                float ray_imu[3] = {
                    s->R_rect_imu[0] * rr[0] + s->R_rect_imu[1] * rr[1] + s->R_rect_imu[2] * rr[2],
                    s->R_rect_imu[3] * rr[0] + s->R_rect_imu[4] * rr[1] + s->R_rect_imu[5] * rr[2],
                    s->R_rect_imu[6] * rr[0] + s->R_rect_imu[7] * rr[1] + s->R_rect_imu[8] * rr[2],
                };
                size_t i = (size_t)y * XS_W + x;
                float u, v;
                s->map[c][i] = -1;
                s->mapx[c][i] = s->mapy[c][i] = 0;
                if (xr_align_project(cams[c], variant, ray_imu, &u, &v) == 0 &&
                    u >= 0.0f && u < XR_OW - 1.001f &&
                    v >= 0.0f && v < XR_OH - 1.001f) {
                    s->map[c][i] = (int32_t)lroundf(v) * XR_OW + (int32_t)lroundf(u);
                    s->mapx[c][i] = (uint16_t)(u * 16.0f);
                    s->mapy[c][i] = (uint16_t)(v * 16.0f);
                }
            }
        }
    }
}

/* bilinear rectification via the 12.4 fixed-point maps */
void xr_stereo_rectify(xr_stereo *s, int c, const uint8_t *src) {
    uint8_t *dst = s->rect[c];
    for (size_t i = 0; i < (size_t)XS_W * XS_H; i++) {
        if (s->map[c][i] < 0) { dst[i] = 0; continue; }
        uint32_t mx = s->mapx[c][i], my = s->mapy[c][i];
        uint32_t fx = mx & 15, fy = my & 15;
        const uint8_t *p = src + (my >> 4) * XR_OW + (mx >> 4);
        uint32_t a = p[0] * (16 - fx) + p[1] * fx;
        uint32_t b = p[XR_OW] * (16 - fx) + p[XR_OW + 1] * fx;
        dst[i] = (uint8_t)((a * (16 - fy) + b * fy + 128) >> 8);
    }
}

/* 9x7 census transform (62 neighbours -> 62-bit signature); border rows and
 * columns keep signature 0 and are excluded via cost padding */
static void census9x7(const uint8_t *img, uint64_t *out) {
    memset(out, 0, sizeof(uint64_t) * XS_W * XS_H);
    for (int y = 3; y < XS_H - 3; y++) {
        for (int x = 4; x < XS_W - 4; x++) {
            uint8_t c = img[y * XS_W + x];
            uint64_t sig = 0;
            for (int dy = -3; dy <= 3; dy++) {
                const uint8_t *row = img + (y + dy) * XS_W + x;
                for (int dx = -4; dx <= 4; dx++) {
                    if (!dx && !dy) continue;
                    sig = (sig << 1) | (row[dx] < c);
                }
            }
            out[y * XS_W + x] = sig;
        }
    }
}

/* one SGM step: Lc(d) = C(d) + min(Lp(d), Lp(d+-1)+P1, minLp+P2) - minLp */
static inline uint16_t sgm_update(const uint8_t *C, const uint16_t *Lp,
                                  uint16_t minLp, uint16_t *Lc) {
    uint16_t minc = 0xFFFF;
    uint16_t cap = (uint16_t)(minLp + P2);
    for (int d = 0; d < XS_DISP; d++) {
        uint16_t best = Lp[d];
        if (d > 0 && (uint16_t)(Lp[d - 1] + P1) < best) best = (uint16_t)(Lp[d - 1] + P1);
        if (d < XS_DISP - 1 && (uint16_t)(Lp[d + 1] + P1) < best) best = (uint16_t)(Lp[d + 1] + P1);
        if (cap < best) best = cap;
        uint16_t v = (uint16_t)(C[d] + best - minLp);
        Lc[d] = v;
        if (v < minc) minc = v;
    }
    return minc;
}

static inline uint16_t sgm_seed(const uint8_t *C, uint16_t *Lc) {
    uint16_t minc = 0xFFFF;
    for (int d = 0; d < XS_DISP; d++) {
        Lc[d] = C[d];
        if (C[d] < minc) minc = C[d];
    }
    return minc;
}

void xr_stereo_depth(xr_stereo *s, uint8_t *out_disp) {
    census9x7(s->rect[0], s->census[0]);
    census9x7(s->rect[1], s->census[1]);

    /* matching cost: hamming distance of the census signatures */
    for (int y = 0; y < XS_H; y++) {
        int inner_y = y >= 3 && y < XS_H - 3;
        for (int x = 0; x < XS_W; x++) {
            uint8_t *C = s->cost + ((size_t)y * XS_W + x) * XS_DISP;
            uint64_t cl = s->census[0][y * XS_W + x];
            for (int d = 0; d < XS_DISP; d++) {
                int xr = x - d;
                C[d] = (inner_y && x >= 4 && x < XS_W - 4 && xr >= 4)
                       ? (uint8_t)__builtin_popcountll(cl ^ s->census[1][y * XS_W + xr])
                       : C_PAD;
            }
        }
    }

    /* SGM aggregation, 4 paths */
    memset(s->sum, 0, sizeof s->sum);
    uint16_t L[2][XS_DISP];
    for (int pass = 0; pass < 2; pass++) {                 /* left<->right */
        int x0 = pass ? XS_W - 1 : 0, dx = pass ? -1 : 1;
        for (int y = 0; y < XS_H; y++) {
            const uint8_t *Crow = s->cost + (size_t)y * XS_W * XS_DISP;
            uint16_t *Srow = s->sum + (size_t)y * XS_W * XS_DISP;
            uint16_t *Lp = L[0], *Lc = L[1];
            uint16_t m = sgm_seed(Crow + (size_t)x0 * XS_DISP, Lp);
            for (int d = 0; d < XS_DISP; d++) Srow[(size_t)x0 * XS_DISP + d] += Lp[d];
            for (int x = x0 + dx; x >= 0 && x < XS_W; x += dx) {
                m = sgm_update(Crow + (size_t)x * XS_DISP, Lp, m, Lc);
                uint16_t *Sd = Srow + (size_t)x * XS_DISP;
                for (int d = 0; d < XS_DISP; d++) Sd[d] += Lc[d];
                uint16_t *t = Lp; Lp = Lc; Lc = t;
            }
        }
    }
    for (int pass = 0; pass < 2; pass++) {                 /* top<->bottom */
        int y0 = pass ? XS_H - 1 : 0, dy = pass ? -1 : 1;
        for (int y = y0; y >= 0 && y < XS_H; y += dy) {
            const uint8_t *Crow = s->cost + (size_t)y * XS_W * XS_DISP;
            uint16_t *Srow = s->sum + (size_t)y * XS_W * XS_DISP;
            uint16_t *prev = s->lrow[pass ? 1 : 0];        /* previous row's L */
            for (int x = 0; x < XS_W; x++) {
                uint16_t *Lc = prev + (size_t)x * XS_DISP; /* updated in place */
                uint16_t tmp[XS_DISP];
                uint16_t m;
                if (y == y0) m = sgm_seed(Crow + (size_t)x * XS_DISP, tmp);
                else m = sgm_update(Crow + (size_t)x * XS_DISP, Lc, s->lmin[x], tmp);
                memcpy(Lc, tmp, sizeof tmp);
                s->lmin[x] = m;
                uint16_t *Sd = Srow + (size_t)x * XS_DISP;
                for (int d = 0; d < XS_DISP; d++) Sd[d] += Lc[d];
            }
        }
    }

    /* WTA + uniqueness + subpixel (quarter-pixel), plus the right image's
     * WTA from the same aggregated volume for the LR check */
    memset(s->dispR, 0, sizeof s->dispR);
    for (int y = 0; y < XS_H; y++) {
        const uint16_t *Srow = s->sum + (size_t)y * XS_W * XS_DISP;
        uint8_t *Dq = s->disp_q + (size_t)y * XS_W;
        uint8_t *Di = s->disp_int + (size_t)y * XS_W;
        for (int x = 0; x < XS_W; x++) {
            const uint16_t *Sd = Srow + (size_t)x * XS_DISP;
            int best = 0;
            for (int d = 1; d < XS_DISP; d++)
                if (Sd[d] < Sd[best]) best = d;
            uint32_t bc = Sd[best];
            int unique = 1;
            for (int d = 0; d < XS_DISP; d++) {
                if (d >= best - 1 && d <= best + 1) continue;
                if ((uint32_t)Sd[d] * UNIQ_DEN < bc * UNIQ_NUM) { unique = 0; break; }
            }
            if (!unique || best == 0 || x - best < 4) {
                Dq[x] = 0; Di[x] = 0;
                continue;
            }
            int dq = best * XS_SCALE;
            if (best > 0 && best < XS_DISP - 1) {
                int denom = (int)Sd[best - 1] - 2 * (int)Sd[best] + (int)Sd[best + 1];
                if (denom > 0) {
                    int num = (int)Sd[best - 1] - (int)Sd[best + 1];
                    int off = (num * XS_SCALE) / (2 * denom);   /* in [-2, 2] */
                    if (off > XS_SCALE / 2) off = XS_SCALE / 2;
                    if (off < -XS_SCALE / 2) off = -XS_SCALE / 2;
                    dq += off;
                }
            }
            Dq[x] = (uint8_t)(dq < 1 ? 1 : dq);
            Di[x] = (uint8_t)best;
        }
        /* right-image WTA: Sr(xr, d) = S(xr + d, d) */
        uint8_t *Dr = s->dispR + (size_t)y * XS_W;
        for (int xr = 0; xr < XS_W; xr++) {
            int best = 0;
            uint16_t bc = 0xFFFF;
            for (int d = 0; d < XS_DISP && xr + d < XS_W; d++) {
                uint16_t v = Srow[(size_t)(xr + d) * XS_DISP + d];
                if (v < bc) { bc = v; best = d; }
            }
            Dr[xr] = (uint8_t)best;
        }
    }

    /* left-right consistency */
    for (int y = 0; y < XS_H; y++) {
        uint8_t *Dq = s->disp_q + (size_t)y * XS_W;
        uint8_t *Di = s->disp_int + (size_t)y * XS_W;
        const uint8_t *Dr = s->dispR + (size_t)y * XS_W;
        for (int x = 0; x < XS_W; x++) {
            if (!Di[x]) continue;
            int xr = x - Di[x];
            int dd = (int)Dr[xr] - (int)Di[x];
            if (dd < -1 || dd > 1) { Dq[x] = 0; Di[x] = 0; }
        }
    }

    /* speckle filter: 4-connected regions of similar disparity smaller than
     * SPECKLE_MIN are noise */
    memset(s->region, 0, sizeof s->region);
    int32_t next_label = 0;
    for (size_t seed = 0; seed < (size_t)XS_W * XS_H; seed++) {
        if (!s->disp_int[seed] || s->region[seed]) continue;
        int32_t label = ++next_label;
        int top = 0;
        s->stack[top++] = (int32_t)seed;
        s->region[seed] = label;
        int count = 0;
        int start = top - 1;                    /* members stay in the stack */
        while (top > start + count) {
            int32_t i = s->stack[start + count++];
            int x = i % XS_W, y = i / XS_W;
            int di = s->disp_int[i];
            const int nx[4] = { x - 1, x + 1, x, x };
            const int ny[4] = { y, y, y - 1, y + 1 };
            for (int k = 0; k < 4; k++) {
                if (nx[k] < 0 || nx[k] >= XS_W || ny[k] < 0 || ny[k] >= XS_H)
                    continue;
                int32_t j = ny[k] * XS_W + nx[k];
                if (s->region[j] || !s->disp_int[j]) continue;
                int dj = s->disp_int[j];
                if (dj - di > SPECKLE_TOL || di - dj > SPECKLE_TOL) continue;
                s->region[j] = label;
                s->stack[top++] = j;
            }
        }
        if (count < SPECKLE_MIN)
            for (int k = 0; k < count; k++) {
                int32_t i = s->stack[start + k];
                s->disp_q[i] = 0;
                /* keep disp_int for connectivity of already-visited px */
            }
    }

    /* 3x3 median (zeros participate: also erodes lone survivors) */
    memset(out_disp, 0, (size_t)XS_W * XS_H);
    for (int y = 1; y < XS_H - 1; y++) {
        for (int x = 1; x < XS_W - 1; x++) {
            uint8_t v[9];
            int k = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    v[k++] = s->disp_q[(y + dy) * XS_W + x + dx];
            for (int i = 0; i <= 4; i++) {
                int m = i;
                for (int j = i + 1; j < 9; j++)
                    if (v[j] < v[m]) m = j;
                uint8_t t = v[i]; v[i] = v[m]; v[m] = t;
            }
            out_disp[y * XS_W + x] = v[4];
        }
    }
}
