/* xr_xfeat.h — XFeat (CVPR 2024) keypoints + descriptors for the session
 * map, via ONNX Runtime's C API (libonnxruntime.so from the Gradle AAR,
 * dlopen'd at runtime — nothing links against it). Used at KEYFRAME rate
 * (~1-2 Hz) on the map thread; the fixed 640x480 export matches the
 * portrait camera raster exactly.
 *
 * Two paths, one contract (callers cannot tell them apart):
 *  - NPU: the dense backbone as a QAIRT-native A8W8 HTP context (EPContext
 *    .onnx, u8 IO: score [1,1,640,480] / desc [1,64,80,60] / reliability
 *    [1,1,80,60]); NMS + reliability-weighted top-K + bilinear descriptor
 *    sampling reproduced in C. 3.6 ms on the SD888 HTP.
 *  - CPU: the original full xfeat.onnx (detect/NMS/top-K in-graph):
 *    input "images" [1,3,640,480] float, raw 0..255 (gray replicated x3);
 *    outputs keypoints [N,2] float (x,y), descriptors [N,64] float
 *    (L2-normalized), scores [N] float sorted descending.
 */
#ifndef XR_XFEAT_H
#define XR_XFEAT_H

#include <stdint.h>

/* upper bound on keypoints per extraction (>= xr_map's XR_MAP_KP_PER_KF) */
#define XR_XFEAT_MAX_KP 256

/* Optional NPU dense path: register the staged EPContext model BEFORE
 * xr_xfeat_init runs (cheap — stores the path). Init then tries the HTP
 * first and falls back to the CPU model. */
void xr_xfeat_set_npu_model(const char *epctx_path);

/* 1 when extraction runs the HTP dense pass (vs the CPU graph). */
int xr_xfeat_npu_active(void);

/* Load ONNX Runtime + the model; idempotent. Returns 1 when usable. */
int xr_xfeat_init(const char *model_path);

int xr_xfeat_available(void);

/* Extract up to `max` keypoints from a 480x640 grayscale image. Fills
 * uv[i] = {x, y} and desc[i] = the L2-normalized descriptor quantized to
 * int8 (round(v * 127)); cosine similarity ~= dot / 16129. Returns the
 * number of keypoints, or -1 on inference failure. */
int xr_xfeat_extract(const uint8_t *img, float (*uv)[2],
                     int8_t (*desc)[64], int max);

/* Sample descriptors at ARBITRARY pixel uvs from the dense map cached by
 * the LAST xr_xfeat_extract call (same image, same map-thread pass) —
 * landmark anchoring: every VIO landmark gets an exact descriptor, so
 * loop/reloc verification is no longer starved by the kp<->landmark
 * proximity join. Works behind the NPU dense path and the dense CPU
 * export (xfeat_dense_dyn.onnx); returns -1 under the sparse CPU graph
 * (no dense map to sample). */
int xr_xfeat_sample(const float (*uv)[2], int n, int8_t (*desc)[64]);

/* Extract + anchor-sample ATOMICALLY (one internal lock hold): anchors
 * are guaranteed to come from THIS image's dense map even when another
 * thread (XR_SEED feeding) extracts concurrently. out_anchored = n_anchor
 * on success, 0 when the active path has no dense map. */
int xr_xfeat_extract_anchored(const uint8_t *img, float (*uv)[2],
                              int8_t (*desc)[64], int max,
                              const float (*auv)[2], int n_anchor,
                              int8_t (*adesc)[64], int *out_anchored);

/* 1 when the active path produces a dense map (NPU, or the dense CPU
 * export) — i.e., xr_xfeat_sample will work after the next extract. The
 * map layer uses this to decide whether to reserve keypoint budget for
 * landmark anchors. */
int xr_xfeat_can_sample(void);

#endif
