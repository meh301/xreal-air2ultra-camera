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

#endif
