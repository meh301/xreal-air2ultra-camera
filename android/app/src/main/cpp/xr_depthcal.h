/* xr_depthcal.h — metric calibration of a monocular depth map.
 *
 * ZipDepth (and monocular depth in general) is only affine-correct: its output
 * relates to true inverse-depth by an unknown per-scene scale + shift. We fit
 * that affine map from SPARSE METRIC ANCHORS — Basalt VIO landmarks (which
 * carry metric inverse-depth) and a cheap sparse-stereo grid — then apply it to
 * the dense map to get metric depth.
 *
 * The fit is done in INVERSE-DEPTH space, where the relation is affine and the
 * conditioning is even across near + far: y = 1/z_metric ~= a*s + b, with s the
 * model value at each anchor. It is robust (iteratively re-weighted Huber)
 * against the inevitable landmark / stereo-match outliers, and EMA-smoothed
 * across frames so the passthrough depth doesn't shimmer. Runtime-agnostic:
 * nothing here knows about ONNX / the NPU; it only sees numbers. */
#ifndef XR_DEPTHCAL_H
#define XR_DEPTHCAL_H

typedef struct {
    float a, b;        /* invz ~= a*s + b (EMA-smoothed, current best) */
    int   have;        /* a usable fit exists */
    int   n_in;        /* inliers in the last fit (quality signal) */
    float rms;         /* inlier RMS residual, inverse-depth units */
} xr_depthcal;

/* Fit from n anchors: s[i] = dense-model value at the anchor, y[i] = measured
 * metric inverse-depth (1/z, z in metres) there. Robust + EMA-smoothed into c.
 * Returns 1 if the fit is usable (enough inliers, well-conditioned), else 0 and
 * c is left unchanged (the previous fit keeps driving apply). */
int xr_depthcal_fit(xr_depthcal *c, const float *s, const float *y, int n);

/* Apply the current fit to a dense model map: z_out[i] = metric depth (metres)
 * = 1/(a*s_in[i] + b), clamped to [zmin, zmax]; a non-positive inverse-depth
 * (behind the camera / diverged) becomes 0. If !c->have, fills 0. */
void xr_depthcal_apply(const xr_depthcal *c, const float *s_in, float *z_out,
                       int n, float zmin, float zmax);

#endif
