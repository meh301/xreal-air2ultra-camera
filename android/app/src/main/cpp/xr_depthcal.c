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
