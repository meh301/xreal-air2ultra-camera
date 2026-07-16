/* xr_lighterglue.h — learned keypoint matcher for loop/reloc VERIFICATION.
 *
 * LighterGlue (XFeat's trimmed 3-layer LightGlue) replaces the greedy
 * nearest-neighbour + margin test when generating query↔keyframe
 * correspondences for PnP — attention over BOTH keypoint sets recovers
 * matches that per-descriptor NN loses on repetitive structure (the
 * corridor-recall axis of the reloc benchmark). It only ever runs on the
 * few retrieval candidates that reach geometric verification, never in
 * the per-keyframe retrieval scan (that stays a cheap NN count).
 *
 * Inference through ONNX Runtime (dlopen'd, CPU EP — same contract as
 * xr_vpr). The export is STATIC-shape: exactly LG_N keypoint slots per
 * side; callers' sets are padded with zero descriptors (zero tokens score
 * ~nothing after the internal normalization and are index-filtered from
 * the output). Keypoints are fed normalized to ~[-1,1]:
 * (uv - [W,H]/2) / (max(W,H)/2) — the convention the export traced.
 * Descriptors arrive int8 (map storage, unit-norm * 127) and are
 * dequantized to f32 with /127. */
#ifndef XR_LIGHTERGLUE_H
#define XR_LIGHTERGLUE_H

#include <stdint.h>

#define XR_LGLUE_N 512      /* fixed keypoint slots per side (export shape) */

/* Register the staged ONNX model path (cheap; no I/O). The first match
 * call performs the lazy ORT bring-up on the calling (map) thread. */
void xr_lglue_set_model(const char *onnx_path);

/* 1 once a session is up (after the first successful bring-up). */
int xr_lglue_ready(void);

/* 1 when a model path is registered (session may not be up yet) — the
 * map layer uses this to route correspondence generation. */
int xr_lglue_wanted(void);

/* Match two keypoint sets (pixel uv + int8[64] descriptors, n <= LG_N;
 * larger sets are truncated). W/H = image size for normalization.
 * Fills out_i0/out_i1 (indices into set 0 / set 1) and out_sc (match
 * confidence 0..1) for up to max_out matches above the score floor
 * (XR_LGLUE_MIN env, default 0.35). Returns the match count, or -1 when
 * unavailable / on inference error (caller falls back to NN). */
int xr_lglue_match(const float (*uv0)[2], const int8_t (*d0)[64], int n0,
                   const float (*uv1)[2], const int8_t (*d1)[64], int n1,
                   float w_px, float h_px,
                   int *out_i0, int *out_i1, float *out_sc, int max_out);

#endif
