/* xr_track.c — see xr_track.h. */
#include "xr_track.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PATCH 3            /* 7x7 patch */
#define SEARCH 8           /* +-8 px around the prediction */
#define SAD_MAX 2200       /* reject worse matches (7x7 u8) */
#define FB_TOL 2           /* forward-backward: max round-trip error, px */
#define MIN_DIST 10        /* min spacing between features, px */
#define MIN_EIG 1200       /* corner-quality gate (see harris_min_eig) */
#define MARGIN (PATCH + SEARCH + 2)

void xr_track_reset(xr_track *t) {
    memset(t, 0, sizeof *t);
}

static int sad7(const uint8_t *a, const uint8_t *b) {
    int s = 0;
    for (int dy = -PATCH; dy <= PATCH; dy++)
        for (int dx = -PATCH; dx <= PATCH; dx++)
            s += abs((int)a[dy * XS_W + dx] - (int)b[dy * XS_W + dx]);
    return s;
}

/* smaller eigenvalue of the 5x5 gradient structure tensor (Shi-Tomasi
 * "good features" score): high only where the patch has gradients in TWO
 * directions, i.e. a corner — plain gradient magnitude also fires on edges,
 * which slide along themselves and track poorly. Gradients are /4 to keep
 * the sums in comfortable integer range. */
static int harris_min_eig(const uint8_t *img, int x, int y) {
    int gxx = 0, gyy = 0, gxy = 0;
    for (int dy = -2; dy <= 2; dy++) {
        const uint8_t *p = img + (y + dy) * XS_W + x;
        for (int dx = -2; dx <= 2; dx++) {
            int gx = ((int)p[dx + 1] - (int)p[dx - 1]) >> 2;
            int gy = ((int)p[dx + XS_W] - (int)p[dx - XS_W]) >> 2;
            gxx += gx * gx;
            gyy += gy * gy;
            gxy += gx * gy;
        }
    }
    float tr2 = (gxx - gyy) * 0.5f;
    float lmin = (gxx + gyy) * 0.5f -
                 sqrtf(tr2 * tr2 + (float)gxy * (float)gxy);
    return (int)lmin;
}

static int too_close(const xr_track *t, float x, float y) {
    for (int i = 0; i < XT_MAX; i++) {
        if (!t->pt[i].alive) continue;
        float dx = t->pt[i].x - x, dy = t->pt[i].y - y;
        if (dx * dx + dy * dy < (float)(MIN_DIST * MIN_DIST)) return 1;
    }
    return 0;
}

/* seed new corners on a grid: best Shi-Tomasi pixel per empty cell */
static void reseed(xr_track *t, const uint8_t *img) {
    enum { CX_ = 6, CY_ = 8 };
    int cell_w = (XS_W - 2 * MARGIN) / CX_, cell_h = (XS_H - 2 * MARGIN) / CY_;
    int occupied[CY_][CX_];
    memset(occupied, 0, sizeof occupied);
    for (int i = 0; i < XT_MAX; i++) {
        if (!t->pt[i].alive) continue;
        int cx = ((int)t->pt[i].x - MARGIN) / cell_w;
        int cy = ((int)t->pt[i].y - MARGIN) / cell_h;
        if (cx >= 0 && cx < CX_ && cy >= 0 && cy < CY_) occupied[cy][cx] = 1;
    }
    for (int cy = 0; cy < CY_; cy++) {
        for (int cx = 0; cx < CX_; cx++) {
            if (occupied[cy][cx] || t->count >= XT_MAX) continue;
            int x0 = MARGIN + cx * cell_w, y0 = MARGIN + cy * cell_h;
            int best = 0, bx = -1, by = -1;
            for (int y = y0; y < y0 + cell_h; y += 2) {
                for (int x = x0; x < x0 + cell_w; x += 2) {
                    int score = harris_min_eig(img, x, y);
                    if (score > best) { best = score; bx = x; by = y; }
                }
            }
            if (best < MIN_EIG) continue;      /* no corner in this cell */
            if (too_close(t, (float)bx, (float)by)) continue;
            for (int i = 0; i < XT_MAX; i++) {
                if (t->pt[i].alive) continue;
                t->pt[i].x = (float)bx;
                t->pt[i].y = (float)by;
                t->pt[i].age = 0;
                t->pt[i].alive = 1;
                t->count++;
                break;
            }
        }
    }
}

int xr_track_step(xr_track *t, const uint8_t *img, float pred_dx, float pred_dy) {
    if (t->have_prev) {
        for (int i = 0; i < XT_MAX; i++) {
            if (!t->pt[i].alive) continue;
            int px = (int)lroundf(t->pt[i].x);
            int py = (int)lroundf(t->pt[i].y);
            int cx = (int)lroundf(t->pt[i].x + pred_dx);
            int cy = (int)lroundf(t->pt[i].y + pred_dy);
            if (px < MARGIN || px >= XS_W - MARGIN ||
                py < MARGIN || py >= XS_H - MARGIN ||
                cx < MARGIN || cx >= XS_W - MARGIN ||
                cy < MARGIN || cy >= XS_H - MARGIN) {
                t->pt[i].alive = 0;
                t->count--;
                continue;
            }
            const uint8_t *ref = t->prev + py * XS_W + px;
            int best = 1 << 30, bx = cx, by = cy;
            for (int dy = -SEARCH; dy <= SEARCH; dy++) {
                for (int dx = -SEARCH; dx <= SEARCH; dx++) {
                    int s = sad7(img + (cy + dy) * XS_W + cx + dx, ref);
                    if (s < best) { best = s; bx = cx + dx; by = cy + dy; }
                }
            }
            int ok = best <= SAD_MAX;

            /* forward-backward check: track the found patch back into the
             * previous image; a good feature round-trips to where it began,
             * a drifter doesn't */
            if (ok) {
                const uint8_t *fwd = img + by * XS_W + bx;
                int fbest = 1 << 30, fx = px, fy = py;
                for (int dy = -FB_TOL - 1; dy <= FB_TOL + 1; dy++) {
                    for (int dx = -FB_TOL - 1; dx <= FB_TOL + 1; dx++) {
                        int s = sad7(t->prev + (py + dy) * XS_W + px + dx, fwd);
                        if (s < fbest) { fbest = s; fx = px + dx; fy = py + dy; }
                    }
                }
                if (abs(fx - px) > FB_TOL || abs(fy - py) > FB_TOL) ok = 0;
            }
            if (!ok) {
                t->pt[i].alive = 0;
                t->count--;
                continue;
            }

            /* subpixel: 1D parabola fits on the SAD surface around the peak */
            float sx = (float)bx, sy = (float)by;
            {
                const uint8_t *c = img + by * XS_W + bx;
                int sl = sad7(c - 1, ref), sr = sad7(c + 1, ref);
                int su = sad7(c - XS_W, ref), sd = sad7(c + XS_W, ref);
                int dxx = sl - 2 * best + sr, dyy = su - 2 * best + sd;
                if (dxx > 0) {
                    float o = 0.5f * (float)(sl - sr) / (float)dxx;
                    if (o > 0.5f) o = 0.5f; if (o < -0.5f) o = -0.5f;
                    sx += o;
                }
                if (dyy > 0) {
                    float o = 0.5f * (float)(su - sd) / (float)dyy;
                    if (o > 0.5f) o = 0.5f; if (o < -0.5f) o = -0.5f;
                    sy += o;
                }
            }
            t->pt[i].x = sx;
            t->pt[i].y = sy;
            if (t->pt[i].age < 65535) t->pt[i].age++;
        }
    }
    if (!t->have_prev || t->count < 90 || t->frame % 15 == 0)
        reseed(t, img);
    memcpy(t->prev, img, sizeof t->prev);
    t->have_prev = 1;
    t->frame++;
    return t->count;
}
