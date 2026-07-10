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

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define XS_NEON 1
#endif

/* rectified pinhole: portrait 240x320, x along the baseline */
#define F_RECT 200.0f
#define CX ((XS_W - 1) * 0.5f)
#define CY ((XS_H - 1) * 0.5f)

/* SGM penalties for the 9x7 census cost (range 0..62): P1 tolerates the
 * 1-disparity steps of slanted surfaces, P2 charges real discontinuities.
 * P2 drops at intensity edges (adaptive P2): a strong image gradient is
 * evidence of a genuine depth boundary, so jumps get cheaper there. */
#define P1 10
#define P2 120
#define P2_EDGE 48             /* P2 across an intensity edge */
#define EDGE_DIFF 16           /* |dI| that counts as an edge */
#define C_PAD 62               /* cost where the window leaves the image */
#define UNIQ_NUM 17            /* reject if 2nd best < best * 17/16 */
#define UNIQ_DEN 16
#define SPECKLE_MIN 60         /* smallest surviving region, px */
#define SPECKLE_TOL 1          /* integer-disparity connectivity */
#define FILL_MAX 60            /* longest invalid run the hole fill closes */

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

/* one SGM step: Lc(d) = C(d) + min(Lp(d), Lp(d+-1)+P1, minLp+p2) - minLp,
 * accumulated into the 4-path sum as it goes. Returns min(Lc). */
#ifdef XS_NEON
static inline uint16_t sgm_update(const uint8_t *C, const uint16_t *Lp,
                                  uint16_t minLp, uint16_t p2,
                                  uint16_t *Lc, uint16_t *Sacc) {
    const uint16x8_t vP1 = vdupq_n_u16(P1);
    const uint16x8_t vminLp = vdupq_n_u16(minLp);
    const uint16x8_t vcap = vdupq_n_u16((uint16_t)(minLp + p2));
    const uint16x8_t big = vdupq_n_u16(0xFFFF);
    uint16x8_t run = big;
    uint16x8_t blk[XS_DISP / 8];
    for (int b = 0; b < XS_DISP / 8; b++) blk[b] = vld1q_u16(Lp + 8 * b);
    for (int b = 0; b < XS_DISP / 8; b++) {
        uint16x8_t l = blk[b];
        uint16x8_t lm1 = vextq_u16(b ? blk[b - 1] : big, l, 7);
        uint16x8_t lp1 = vextq_u16(l, b < XS_DISP / 8 - 1 ? blk[b + 1] : big, 1);
        /* saturating +P1 keeps the 0xFFFF pads from wrapping to tiny values */
        uint16x8_t m = vminq_u16(l, vminq_u16(vqaddq_u16(lm1, vP1),
                                              vqaddq_u16(lp1, vP1)));
        m = vminq_u16(m, vcap);
        uint16x8_t c16 = vmovl_u8(vld1_u8(C + 8 * b));
        uint16x8_t v = vsubq_u16(vaddq_u16(c16, m), vminLp);
        vst1q_u16(Lc + 8 * b, v);
        vst1q_u16(Sacc + 8 * b, vaddq_u16(vld1q_u16(Sacc + 8 * b), v));
        run = vminq_u16(run, v);
    }
#ifdef __aarch64__
    return vminvq_u16(run);
#else
    uint16x4_t r = vmin_u16(vget_low_u16(run), vget_high_u16(run));
    r = vpmin_u16(r, r);
    r = vpmin_u16(r, r);
    return vget_lane_u16(r, 0);
#endif
}

static inline uint16_t sgm_seed(const uint8_t *C, uint16_t *Lc, uint16_t *Sacc) {
    uint16x8_t run = vdupq_n_u16(0xFFFF);
    for (int b = 0; b < XS_DISP / 8; b++) {
        uint16x8_t v = vmovl_u8(vld1_u8(C + 8 * b));
        vst1q_u16(Lc + 8 * b, v);
        vst1q_u16(Sacc + 8 * b, vaddq_u16(vld1q_u16(Sacc + 8 * b), v));
        run = vminq_u16(run, v);
    }
#ifdef __aarch64__
    return vminvq_u16(run);
#else
    uint16x4_t r = vmin_u16(vget_low_u16(run), vget_high_u16(run));
    r = vpmin_u16(r, r);
    r = vpmin_u16(r, r);
    return vget_lane_u16(r, 0);
#endif
}
#else
static inline uint16_t sgm_update(const uint8_t *C, const uint16_t *Lp,
                                  uint16_t minLp, uint16_t p2,
                                  uint16_t *Lc, uint16_t *Sacc) {
    uint16_t minc = 0xFFFF;
    uint16_t cap = (uint16_t)(minLp + p2);
    for (int d = 0; d < XS_DISP; d++) {
        uint16_t best = Lp[d];
        if (d > 0 && (uint16_t)(Lp[d - 1] + P1) < best) best = (uint16_t)(Lp[d - 1] + P1);
        if (d < XS_DISP - 1 && (uint16_t)(Lp[d + 1] + P1) < best) best = (uint16_t)(Lp[d + 1] + P1);
        if (cap < best) best = cap;
        uint16_t v = (uint16_t)(C[d] + best - minLp);
        Lc[d] = v;
        Sacc[d] = (uint16_t)(Sacc[d] + v);
        if (v < minc) minc = v;
    }
    return minc;
}

static inline uint16_t sgm_seed(const uint8_t *C, uint16_t *Lc, uint16_t *Sacc) {
    uint16_t minc = 0xFFFF;
    for (int d = 0; d < XS_DISP; d++) {
        Lc[d] = C[d];
        Sacc[d] = (uint16_t)(Sacc[d] + C[d]);
        if (C[d] < minc) minc = C[d];
    }
    return minc;
}
#endif

/* adaptive P2 from the intensity step between a pixel and its path
 * predecessor */
static inline uint16_t p2_of(uint8_t a, uint8_t b) {
    int d = (int)a - (int)b;
    if (d < 0) d = -d;
    return d > EDGE_DIFF ? P2_EDGE : P2;
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
            const uint8_t *irow = s->rect[0] + (size_t)y * XS_W;
            uint16_t *Srow = s->sum + (size_t)y * XS_W * XS_DISP;
            uint16_t *Lp = L[0], *Lc = L[1];
            uint16_t m = sgm_seed(Crow + (size_t)x0 * XS_DISP, Lp,
                                  Srow + (size_t)x0 * XS_DISP);
            for (int x = x0 + dx; x >= 0 && x < XS_W; x += dx) {
                m = sgm_update(Crow + (size_t)x * XS_DISP, Lp, m,
                               p2_of(irow[x], irow[x - dx]),
                               Lc, Srow + (size_t)x * XS_DISP);
                uint16_t *t = Lp; Lp = Lc; Lc = t;
            }
        }
    }
    for (int pass = 0; pass < 2; pass++) {                 /* top<->bottom */
        int y0 = pass ? XS_H - 1 : 0, dy = pass ? -1 : 1;
        uint16_t *prev = s->lrow[pass];                    /* previous row's L */
        for (int y = y0; y >= 0 && y < XS_H; y += dy) {
            const uint8_t *Crow = s->cost + (size_t)y * XS_W * XS_DISP;
            const uint8_t *irow = s->rect[0] + (size_t)y * XS_W;
            uint16_t *Srow = s->sum + (size_t)y * XS_W * XS_DISP;
            for (int x = 0; x < XS_W; x++) {
                uint16_t *Lr = prev + (size_t)x * XS_DISP;
                uint16_t m;
                if (y == y0) {
                    m = sgm_seed(Crow + (size_t)x * XS_DISP, Lr,
                                 Srow + (size_t)x * XS_DISP);
                } else {
                    uint16_t tmp[XS_DISP];
                    m = sgm_update(Crow + (size_t)x * XS_DISP, Lr, s->lmin[x],
                                   p2_of(irow[x], irow[x - dy * XS_W]),
                                   tmp, Srow + (size_t)x * XS_DISP);
                    memcpy(Lr, tmp, sizeof tmp);
                }
                s->lmin[x] = m;
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
            int unique;
#ifdef XS_NEON
            uint16x8_t vm = vld1q_u16(Sd);
            for (int b = 1; b < XS_DISP / 8; b++)
                vm = vminq_u16(vm, vld1q_u16(Sd + 8 * b));
#ifdef __aarch64__
            uint16_t bv = vminvq_u16(vm);
#else
            uint16x4_t r = vmin_u16(vget_low_u16(vm), vget_high_u16(vm));
            r = vpmin_u16(r, r);
            r = vpmin_u16(r, r);
            uint16_t bv = vget_lane_u16(r, 0);
#endif
            while (Sd[best] != bv) best++;
            uint32_t bc = bv;
            /* count entries under the ambiguity threshold, then discount
             * the winner and its two neighbours */
            uint16_t thr = (uint16_t)((bc * UNIQ_NUM) / UNIQ_DEN);
            uint16x8_t vthr = vdupq_n_u16(thr);
            uint16x8_t acc = vdupq_n_u16(0);
            for (int b = 0; b < XS_DISP / 8; b++)
                acc = vsubq_u16(acc, vcltq_u16(vld1q_u16(Sd + 8 * b), vthr));
#ifdef __aarch64__
            int cnt = vaddvq_u16(acc);
#else
            uint32x4_t a32 = vpaddlq_u16(acc);
            uint64x2_t a64 = vpaddlq_u32(a32);
            int cnt = (int)(vgetq_lane_u64(a64, 0) + vgetq_lane_u64(a64, 1));
#endif
            cnt -= Sd[best] < thr;
            if (best > 0) cnt -= Sd[best - 1] < thr;
            if (best < XS_DISP - 1) cnt -= Sd[best + 1] < thr;
            unique = cnt == 0;
#else
            for (int d = 1; d < XS_DISP; d++)
                if (Sd[d] < Sd[best]) best = d;
            uint32_t bc = Sd[best];
            unique = 1;
            for (int d = 0; d < XS_DISP; d++) {
                if (d >= best - 1 && d <= best + 1) continue;
                if ((uint32_t)Sd[d] * UNIQ_DEN < bc * UNIQ_NUM) { unique = 0; break; }
            }
#endif
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

    /* hole filling: close short invalid scanline runs with the smaller
     * (more distant) edge disparity — occluded and textureless pixels
     * belong to the background. Long runs stay unknown. */
    for (int y = 0; y < XS_H; y++) {
        uint8_t *Dq = s->disp_q + (size_t)y * XS_W;
        int x = 0;
        while (x < XS_W) {
            if (Dq[x]) { x++; continue; }
            int x1 = x;
            while (x1 < XS_W && !Dq[x1]) x1++;
            if (x > 0 && x1 < XS_W && x1 - x <= FILL_MAX) {
                uint8_t l = Dq[x - 1], r = Dq[x1];
                uint8_t f = l < r ? l : r;
                for (int k = x; k < x1; k++)
                    if (s->map[0][(size_t)y * XS_W + k] >= 0) Dq[k] = f;
            }
            x = x1;
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
