/* xr_depthcal.c — see xr_depthcal.h. */
#include "xr_depthcal.h"

#include <math.h>

#define CAL_ITERS    5       /* IRLS reweighting passes */
#define CAL_MIN_ANCH 6       /* fewer anchors than this: don't attempt a fit */
#define CAL_MIN_IN   5       /* fewer inliers than this: reject the fit */
#define CAL_EMA      0.30f   /* new-fit weight in the temporal smoothing */
#define CAL_INLIER_K 2.5f    /* inlier gate: |residual| < k * robust scale */

/* Robust affine fit y ~= a*s + b by iteratively re-weighted least squares with
 * a Huber influence: the first pass is unweighted, then each pass down-weights
 * points whose residual exceeds an adaptive scale (mean abs residual), so a
 * handful of gross landmark / stereo outliers can't drag the line. Everything
 * accumulated in double — s and y span a wide inverse-depth range. */
int xr_depthcal_fit(xr_depthcal *c, const float *s, const float *y, int n) {
    if (n < CAL_MIN_ANCH) return 0;

    double a = 1.0, b = 0.0, scale = 0.0;
    for (int iter = 0; iter < CAL_ITERS; iter++) {
        double Sw = 0, Ss = 0, Sy = 0, Sss = 0, Ssy = 0;
        for (int i = 0; i < n; i++) {
            double wi = 1.0;
            if (iter > 0) {                         /* Huber weight */
                double ar = fabs(a * s[i] + b - y[i]);
                wi = (ar <= scale || scale <= 0.0) ? 1.0 : scale / ar;
            }
            Sw  += wi;
            Ss  += wi * s[i];
            Sy  += wi * y[i];
            Sss += wi * (double)s[i] * s[i];
            Ssy += wi * (double)s[i] * y[i];
        }
        double det = Sss * Sw - Ss * Ss;
        if (fabs(det) < 1e-12) return 0;            /* anchors collinear in s */
        a = (Ssy * Sw - Ss * Sy) / det;
        b = (Sss * Sy - Ss * Ssy) / det;

        /* adaptive robust scale for the next pass's Huber gate */
        double asum = 0;
        for (int i = 0; i < n; i++) asum += fabs(a * s[i] + b - y[i]);
        scale = 1.5 * (asum / n) + 1e-9;
    }

    /* inlier count + RMS on the final model */
    int n_in = 0;
    double sse = 0;
    for (int i = 0; i < n; i++) {
        double r = a * s[i] + b - y[i];
        if (fabs(r) < CAL_INLIER_K * scale) { n_in++; sse += r * r; }
    }
    if (n_in < CAL_MIN_IN) return 0;

    float af = (float)a, bf = (float)b;
    if (!isfinite(af) || !isfinite(bf)) return 0;
    if (c->have) {                                  /* temporal smoothing */
        c->a = (1.0f - CAL_EMA) * c->a + CAL_EMA * af;
        c->b = (1.0f - CAL_EMA) * c->b + CAL_EMA * bf;
    } else {
        c->a = af;
        c->b = bf;
    }
    c->have = 1;
    c->n_in = n_in;
    c->rms = (float)sqrt(sse / n_in);
    return 1;
}

#define DCAL_MAX_ANCH 1024   /* cap the fit set; plenty for grid + landmarks */

/* bilinear sample of a w*h float map at (fx,fy), clamped to the border */
static float sample_bilin(const float *m, int w, int h, float fx, float fy) {
    if (fx < 0) fx = 0; if (fx > w - 1) fx = (float)(w - 1);
    if (fy < 0) fy = 0; if (fy > h - 1) fy = (float)(h - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
    float ax = fx - x0, ay = fy - y0;
    const float *r0 = m + (size_t)y0 * w, *r1 = m + (size_t)y1 * w;
    float a = r0[x0] * (1 - ax) + r0[x1] * ax;
    float b = r1[x0] * (1 - ax) + r1[x1] * ax;
    return a * (1 - ay) + b * ay;
}

int xr_depthcal_update(xr_depthcal *c, const float *dense, int w, int h,
                       const xr_anchor *anchors, int n) {
    if (!dense || !anchors) return 0;
    float s[DCAL_MAX_ANCH], y[DCAL_MAX_ANCH];
    int m = 0;
    for (int i = 0; i < n && m < DCAL_MAX_ANCH; i++) {
        if (anchors[i].invz <= 0.0f) continue;      /* skip invalid anchors */
        s[m] = sample_bilin(dense, w, h, anchors[i].x, anchors[i].y);
        y[m] = anchors[i].invz;
        m++;
    }
    return xr_depthcal_fit(c, s, y, m);
}

void xr_depthcal_apply(const xr_depthcal *c, const float *s_in, float *z_out,
                       int n, float zmin, float zmax) {
    if (!c->have) {
        for (int i = 0; i < n; i++) z_out[i] = 0.0f;
        return;
    }
    float a = c->a, b = c->b;
    float izmin = 1.0f / zmax, izmax = 1.0f / zmin;   /* invz clamps */
    for (int i = 0; i < n; i++) {
        float invz = a * s_in[i] + b;
        if (invz <= izmin) { z_out[i] = (invz <= 0.0f) ? 0.0f : zmax; continue; }
        if (invz >= izmax) { z_out[i] = zmin; continue; }
        z_out[i] = 1.0f / invz;
    }
}
