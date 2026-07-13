/* xr_depthcal.h — spatially-varying metric calibration of a monocular depth map.
 *
 * ZipDepth (monocular depth) is only affine-correct: its output relates to true
 * inverse-depth by an unknown scale + shift. On this IR-sensitive camera that
 * affine is NOT global — IR reflectance gradients bias the model in a smooth
 * position-dependent way, so a single (a,b) leaves large residuals. So we fit
 *
 *     invz(x,y) ~= A(u,v)*s(x,y) + B(u,v),   u,v = image coords normalized to [-1,1]
 *
 * with A and B POLYNOMIALS in (u,v). The polynomial order is the real lever on
 * accuracy (offline, quantized model, held-out, robust fit): global affine
 * delta1~0.53, degree-1 ~0.73, degree-2 ~0.84, degree-3 ~0.88 — and the gain
 * survives realistic stereo-anchor noise. Raw anchor COUNT, by contrast,
 * saturates the fit by ~140 points, so density mainly buys noise-averaging and
 * enough support for the higher orders. We therefore fit up to degree 3 (A,B
 * degree-3 => 20 coefficients) from the SAME sparse metric anchors as before
 * (xr_sgrid stereo grid + xr_lm_anchors VIO landmarks), robustly (IRLS Huber),
 * EMA-smoothed across frames, and DEGRADE GRACEFULLY by anchor count:
 *   >=100 -> deg3(20c),  >=60 -> deg2(12c),  >=20 -> deg1(6c),  >=6 -> global(2c).
 * So it is never worse than the old global fit and won't overfit a sparse frame.
 * The output is clamped to [zmin,zmax] m, which also bounds any high-order
 * extrapolation in anchor-free regions. Fit is per-frame, so it adapts to each
 * scene (indoor/outdoor) with no retrain. Cost is a tiny <=20x20 solve + a
 * per-pixel polynomial. Runtime-agnostic: nothing here knows about ONNX/the NPU. */
#ifndef XR_DEPTHCAL_H
#define XR_DEPTHCAL_H

#include "xr_sgrid.h"      /* xr_anchor */

typedef struct {
    float c[20];       /* invz ~= sum c[k]*feat_k(s,u,v); up to the deg-3 spatial affine */
    int   have;        /* a usable fit exists */
    int   n_in;        /* inliers in the last fit (quality signal) */
    float rms;         /* inlier RMS residual, inverse-depth units */
    int   deg;         /* polynomial degree actually used (0/1/2/3 by anchor count) */
} xr_depthcal;

/* Fit from the dense model map + anchors: bilinearly sample `dense` at each
 * anchor pixel to form s, take y = anchor.invz and (u,v) = the anchor pixel
 * normalized to [-1,1], robustly fit invz ~= A(u,v)*s + B(u,v), and EMA-smooth
 * into c. `dense` must be in an INVERSE-DEPTH-correlated space (the caller passes
 * 1/depth for a depth-emitting model). w,h are the map dims. Returns 1 if the fit
 * is usable, else 0 (the previous fit keeps driving apply). */
int xr_depthcal_update(xr_depthcal *c, const float *dense, int w, int h,
                       const xr_anchor *anchors, int n);

/* Apply the current spatial fit to a dense model map: z_out[i] = metric depth
 * (metres) = 1/(A(u,v)*s_in[i] + B(u,v)), clamped to [zmin,zmax]; a non-positive
 * inverse-depth (behind the camera / diverged) becomes 0. w,h are the map dims
 * (needed to recover (u,v) per pixel). If !c->have, fills 0. */
void xr_depthcal_apply(const xr_depthcal *c, const float *s_in, float *z_out,
                       int w, int h, float zmin, float zmax);

#endif
