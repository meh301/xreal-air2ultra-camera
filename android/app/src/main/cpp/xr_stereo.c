/* xr_stereo.c — see xr_stereo.h. */
#include "xr_stereo.h"

#include <math.h>
#include <string.h>

#include "xreal_core.h"

/* rectified pinhole: portrait 240x320, x along the baseline */
#define F_RECT 200.0f
#define CX ((XS_W - 1) * 0.5f)
#define CY ((XS_H - 1) * 0.5f)

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
                float u, v;
                int32_t idx = -1;
                if (xr_align_project(cams[c], variant, ray_imu, &u, &v) == 0) {
                    int ui = (int)lroundf(u), vi = (int)lroundf(v);
                    if (ui >= 0 && ui < XR_OW && vi >= 0 && vi < XR_OH)
                        idx = vi * XR_OW + ui;
                }
                s->map[c][(size_t)y * XS_W + x] = idx;
            }
        }
    }
}

/* 5x5 census transform (24 neighbours -> 24-bit signature) */
static void census(const uint8_t *img, uint32_t *out) {
    memset(out, 0, sizeof(uint32_t) * XS_W * XS_H);
    for (int y = 2; y < XS_H - 2; y++) {
        for (int x = 2; x < XS_W - 2; x++) {
            uint8_t c = img[y * XS_W + x];
            uint32_t sig = 0;
            for (int dy = -2; dy <= 2; dy++)
                for (int dx = -2; dx <= 2; dx++) {
                    if (!dx && !dy) continue;
                    sig = (sig << 1) | (img[(y + dy) * XS_W + x + dx] < c);
                }
            out[y * XS_W + x] = sig;
        }
    }
}

void xr_stereo_depth(xr_stereo *s, const uint8_t *img_left,
                     const uint8_t *img_right, uint8_t *out_disp) {
    /* rectify (nearest) */
    for (int c = 0; c < 2; c++) {
        const uint8_t *src = c == 0 ? img_left : img_right;
        for (size_t i = 0; i < XS_W * XS_H; i++) {
            int32_t m = s->map[c][i];
            s->rect[c][i] = m >= 0 ? src[m] : 0;
        }
    }
    census(s->rect[0], s->census[0]);
    census(s->rect[1], s->census[1]);

    /* block matching on census hamming cost, 3-row aggregation,
     * uniqueness-checked winner */
    memset(s->disp_raw, 0, sizeof s->disp_raw);
    for (int y = 3; y < XS_H - 3; y++) {
        for (int x = XS_DISP + 2; x < XS_W - 2; x++) {
            int best = 1 << 30, second = 1 << 30, best_d = 0;
            for (int d = 0; d < XS_DISP; d++) {
                int cost = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    size_t li = (size_t)(y + dy) * XS_W + x;
                    cost += __builtin_popcount(s->census[0][li] ^
                                               s->census[1][li - d]);
                }
                if (cost < best) { second = best; best = cost; best_d = d; }
                else if (cost < second) second = cost;
            }
            /* uniqueness + minimum-texture gates */
            if (best_d > 0 && best * 8 < second * 7 && best < 30)
                s->disp_raw[y * XS_W + x] = (uint8_t)best_d;
        }
    }

    /* 3x3 median to knock out speckle */
    memset(out_disp, 0, XS_W * XS_H);
    for (int y = 1; y < XS_H - 1; y++) {
        for (int x = 1; x < XS_W - 1; x++) {
            uint8_t v[9];
            int k = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    v[k++] = s->disp_raw[(y + dy) * XS_W + x + dx];
            /* partial selection sort to the median */
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
