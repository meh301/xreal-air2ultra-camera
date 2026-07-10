/* xr_track.c — see xr_track.h. */
#include "xr_track.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PATCH 3            /* 7x7 patch */
#define SEARCH 8           /* +-8 px around the prediction */
#define SAD_MAX 2200       /* reject worse matches (7x7 u8) */
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

/* seed new corners on a grid: strongest gradient pixel per empty cell */
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
                    const uint8_t *p = img + y * XS_W + x;
                    int g = abs((int)p[1] - (int)p[-1]) +
                            abs((int)p[XS_W] - (int)p[-XS_W]);
                    if (g > best) { best = g; bx = x; by = y; }
                }
            }
            if (best < 24) continue;           /* textureless cell */
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
            if (best > SAD_MAX) {
                t->pt[i].alive = 0;
                t->count--;
                continue;
            }
            t->pt[i].x = (float)bx;
            t->pt[i].y = (float)by;
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
