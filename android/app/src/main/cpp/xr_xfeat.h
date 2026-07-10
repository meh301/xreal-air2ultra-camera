/* xr_xfeat.h — XFeat (CVPR 2024) keypoints + descriptors for the session
 * map, via ONNX Runtime's C API (libonnxruntime.so from the Gradle AAR,
 * dlopen'd at runtime — nothing links against it). Used at KEYFRAME rate
 * (~1-2 Hz) on the map thread; the fixed 640x480 export matches the
 * portrait camera raster exactly.
 *
 * Contract of the bundled model (verified against the export on desktop):
 * input "images" [1,3,640,480] float, raw 0..255 (gray replicated x3);
 * outputs keypoints [N,2] float (x,y), descriptors [N,64] float
 * (L2-normalized), scores [N] float sorted descending.
 */
#ifndef XR_XFEAT_H
#define XR_XFEAT_H

#include <stdint.h>

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
