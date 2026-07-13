/* xr_depthcal.c — see xr_depthcal.h. */
#include "xr_depthcal.h"

#include <math.h>

#define CAL_NCOEF    20      /* deg-3 spatial affine: A,B degree-3 => 10+10 coef */
#define CAL_ITERS    5       /* IRLS reweighting passes */
#define CAL_MIN_IN   5       /* fewer inliers than this: reject the fit */
#define CAL_EMA      0.30f   /* new-fit weight in the temporal smoothing */
#define CAL_INLIER_K 2.5f    /* inlier gate: |residual| < k * robust scale */

/* Anchor-count gates for the spatial polynomial degree. A degree-d fit has
 * D=2/6/12/20 coefficients (d=0/1/2/3); we want a healthy oversampling over
 * noisy live stereo anchors before trusting the higher-order spatial terms, and
 * we fall back rather than overfit. Tunable once on-device anchor yields are seen
 * (the calibration health log prints deg + inliers). */
#define CAL_MIN_DEG3 100     /* >= this many anchors: full spatial deg-3 (20 coef) */
#define CAL_MIN_DEG2 60      /* >= this: deg-2 spatial (12 coef) */
#define CAL_MIN_DEG1 20      /* >= this: deg-1 spatial (6 coef) */
#define CAL_MIN_DEG0 6       /* >= this: plain global affine (2 coef); else no fit */

/* Feature row for (s,u,v) up to the given degree, matching the coefficient order
 * consumed in xr_depthcal_apply:
 *   [ s, 1,                                             (deg0)
 *     s*u, s*v, u, v,                                   (deg1)
 *     s*u^2, s*v^2, s*u*v, u^2, v^2, u*v,               (deg2)
 *     s*u^3, s*v^3, s*u^2*v, s*u*v^2, u^3, v^3, u^2*v, u*v^2 ]   (deg3)
 * Grouping by s this is exactly invz = A(u,v)*s + B(u,v) with A,B degree-`deg`. */
static int feat_row(float f[CAL_NCOEF], float s, float u, float v, int deg) {
    float u2 = u * u, v2 = v * v, uv = u * v;
    f[0] = s; f[1] = 1.0f;
    if (deg == 0) return 2;
    f[2] = s * u; f[3] = s * v; f[4] = u; f[5] = v;
    if (deg == 1) return 6;
    f[6] = s * u2; f[7] = s * v2; f[8] = s * uv;
    f[9] = u2; f[10] = v2; f[11] = uv;
    if (deg == 2) return 12;
    f[12] = s * u2 * u; f[13] = s * v2 * v; f[14] = s * u2 * v; f[15] = s * u * v2;
    f[16] = u2 * u; f[17] = v2 * v; f[18] = u2 * v; f[19] = u * v2;
    return 20;
}

/* Solve the SPD system M x = rhs (D<=CAL_NCOEF) by Cholesky; M is overwritten
 * with its factor. Returns 0 if not positive-definite (degenerate geometry). */
static int solve_spd(double M[CAL_NCOEF][CAL_NCOEF], const double rhs[CAL_NCOEF],
                     double x[CAL_NCOEF], int D) {
    for (int j = 0; j < D; j++) {                 /* M = L L^T, lower triangle */
        double d = M[j][j];
        for (int k = 0; k < j; k++) d -= M[j][k] * M[j][k];
        if (d <= 1e-18) return 0;
        M[j][j] = sqrt(d);
        for (int i = j + 1; i < D; i++) {
            double sm = M[i][j];
            for (int k = 0; k < j; k++) sm -= M[i][k] * M[j][k];
            M[i][j] = sm / M[j][j];
        }
    }
    for (int i = 0; i < D; i++) {                 /* forward:  L y = rhs */
        double sm = rhs[i];
        for (int k = 0; k < i; k++) sm -= M[i][k] * x[k];
        x[i] = sm / M[i][i];
    }
    for (int i = D - 1; i >= 0; i--) {            /* back:  L^T x = y */
        double sm = x[i];
        for (int k = i + 1; k < D; k++) sm -= M[k][i] * x[k];
        x[i] = sm / M[i][i];
    }
    return 1;
}

static int deg_dim(int deg) {
    return deg == 3 ? 20 : deg == 2 ? 12 : deg == 1 ? 6 : 2;
}

/* Robust IRLS fit of invz ~= feat(s,u,v).coef in D dims (Huber down-weighting,
 * first pass unweighted). Writes the full CAL_NCOEF-vector (unused terms left 0),
 * the inlier count and RMS. Returns 1 if enough inliers, else 0. Doubles
 * throughout — s and invz span a wide range and the normal matrix is ill-cond. */
static int robust_fit(const float *s, const float *u, const float *v,
                      const float *y, int n, int deg,
                      float coef_out[CAL_NCOEF], int *n_in_out, float *rms_out) {
    int D = deg_dim(deg);
    double coef[CAL_NCOEF] = {0};
    coef[0] = 1.0;                                /* seed scale a~1, shift b~0 */
    double scale = 0.0;

    for (int iter = 0; iter < CAL_ITERS; iter++) {
        double M[CAL_NCOEF][CAL_NCOEF] = {{0}}, rhs[CAL_NCOEF] = {0};
        for (int i = 0; i < n; i++) {
            float f[CAL_NCOEF]; feat_row(f, s[i], u[i], v[i], deg);
            double wi = 1.0;
            if (iter > 0) {
                double pred = 0.0;
                for (int c = 0; c < D; c++) pred += coef[c] * f[c];
                double ar = fabs(pred - y[i]);
                wi = (ar <= scale || scale <= 0.0) ? 1.0 : scale / ar;
            }
            for (int r = 0; r < D; r++) {
                rhs[r] += wi * f[r] * y[i];
                for (int c = 0; c <= r; c++) M[r][c] += wi * (double)f[r] * f[c];
            }
        }
        for (int r = 0; r < D; r++)               /* mirror to upper triangle */
            for (int c = r + 1; c < D; c++) M[r][c] = M[c][r];

        double tr = 0.0;                          /* tiny ridge for stability */
        for (int r = 0; r < D; r++) tr += M[r][r];
        double ridge = 1e-6 * tr / D;
        for (int r = 0; r < D; r++) M[r][r] += ridge;

        double x[CAL_NCOEF];
        if (!solve_spd(M, rhs, x, D)) return 0;
        for (int c = 0; c < D; c++) coef[c] = x[c];

        double asum = 0.0;                        /* adaptive Huber scale */
        for (int i = 0; i < n; i++) {
            float f[CAL_NCOEF]; feat_row(f, s[i], u[i], v[i], deg);
            double pred = 0.0;
            for (int c = 0; c < D; c++) pred += coef[c] * f[c];
            asum += fabs(pred - y[i]);
        }
        scale = 1.5 * (asum / n) + 1e-9;
    }

    int n_in = 0; double sse = 0.0;
    for (int i = 0; i < n; i++) {
        float f[CAL_NCOEF]; feat_row(f, s[i], u[i], v[i], deg);
        double pred = 0.0;
        for (int c = 0; c < D; c++) pred += coef[c] * f[c];
        double r = pred - y[i];
        if (fabs(r) < CAL_INLIER_K * scale) { n_in++; sse += r * r; }
    }
    if (n_in < CAL_MIN_IN) return 0;

    for (int c = 0; c < CAL_NCOEF; c++) {
        float cf = (float)coef[c];
        if (!isfinite(cf)) return 0;
        coef_out[c] = (c < D) ? cf : 0.0f;        /* zero unused higher terms */
    }
    *n_in_out = n_in;
    *rms_out = (float)sqrt(sse / n_in);
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
    if (!dense || !anchors || w < 2 || h < 2) return 0;
    static float s[DCAL_MAX_ANCH], us[DCAL_MAX_ANCH],
                 vs[DCAL_MAX_ANCH], y[DCAL_MAX_ANCH];
    float iu = 2.0f / (w - 1), iv = 2.0f / (h - 1);   /* pixel -> [-1,1] */
    int m = 0;
    for (int i = 0; i < n && m < DCAL_MAX_ANCH; i++) {
        if (anchors[i].invz <= 0.0f) continue;        /* skip invalid anchors */
        s[m]  = sample_bilin(dense, w, h, anchors[i].x, anchors[i].y);
        us[m] = anchors[i].x * iu - 1.0f;
        vs[m] = anchors[i].y * iv - 1.0f;
        y[m]  = anchors[i].invz;
        m++;
    }
    int deg = (m >= CAL_MIN_DEG3) ? 3 :
              (m >= CAL_MIN_DEG2) ? 2 :
              (m >= CAL_MIN_DEG1) ? 1 :
              (m >= CAL_MIN_DEG0) ? 0 : -1;
    if (deg < 0) return 0;

    float nc[CAL_NCOEF]; int n_in; float rms;
    if (!robust_fit(s, us, vs, y, m, deg, nc, &n_in, &rms)) return 0;

    if (c->have) {                                    /* temporal smoothing */
        for (int k = 0; k < CAL_NCOEF; k++)
            c->c[k] = (1.0f - CAL_EMA) * c->c[k] + CAL_EMA * nc[k];
    } else {
        for (int k = 0; k < CAL_NCOEF; k++) c->c[k] = nc[k];
    }
    c->have = 1;
    c->n_in = n_in;
    c->rms = rms;
    c->deg = deg;
    return 1;
}

void xr_depthcal_apply(const xr_depthcal *c, const float *s_in, float *z_out,
                       int w, int h, float zmin, float zmax) {
    if (!c->have) {
        for (int i = 0; i < w * h; i++) z_out[i] = 0.0f;
        return;
    }
    const float *k = c->c;
    float izmin = 1.0f / zmax, izmax = 1.0f / zmin;    /* invz clamps */
    float iu = 2.0f / (w - 1), iv = 2.0f / (h - 1);
    for (int yy = 0; yy < h; yy++) {
        float v = yy * iv - 1.0f, v2 = v * v, v3 = v2 * v;
        /* A(u,v) and B(u,v) are degree-3; fold the v-powers into per-row u
         * coefficients (unused higher terms are 0, so this covers deg 0..3):
         *   A = A0 + A1*u + A2*u^2 + A3*u^3,  B likewise. */
        float A0 = k[0] + k[3] * v + k[7] * v2 + k[13] * v3;
        float A1 = k[2] + k[8] * v + k[15] * v2;
        float A2 = k[6] + k[14] * v;
        float A3 = k[12];
        float B0 = k[1] + k[5] * v + k[10] * v2 + k[17] * v3;
        float B1 = k[4] + k[11] * v + k[19] * v2;
        float B2 = k[9] + k[18] * v;
        float B3 = k[16];
        for (int xx = 0; xx < w; xx++) {
            int idx = yy * w + xx;
            float u = xx * iu - 1.0f;
            float A = ((A3 * u + A2) * u + A1) * u + A0;    /* Horner */
            float B = ((B3 * u + B2) * u + B1) * u + B0;
            float invz = A * s_in[idx] + B;
            if (invz <= izmin) { z_out[idx] = (invz <= 0.0f) ? 0.0f : zmax; continue; }
            if (invz >= izmax) { z_out[idx] = zmin; continue; }
            z_out[idx] = 1.0f / invz;
        }
    }
}
