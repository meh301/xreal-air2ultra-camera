/* xr_zipdepth.h — monocular metric depth (ZipDepth) via ONNX Runtime on the
 * NPU: Qualcomm QNN (Hexagon) on Snapdragon, MediaTek NeuroPilot on Dimensity,
 * CPU fallback otherwise. Mirrors xr_xfeat's dlopen pattern — libonnxruntime.so
 * is loaded at runtime, so a build/device without it (or without the model)
 * just reports unavailable and the depth worker keeps using SGM.
 *
 * I/O contract, taken from the ZipDepth repo (scripts/export.py +
 * zipdepth/inference/predictor.py + model/architecture.py):
 *   input  'image' : (1,3,ZD_IN_H,ZD_IN_W) NCHW, RGB, divided by 255, no
 *                    mean/std. Static shapes -> ZD_IN_* MUST equal the exported
 *                    ONNX resolution.
 *   output 'depth' : (1,1,ZD_IN_H,ZD_IN_W), non-negative (ReLU'd) affine-metric
 *                    depth -> made metric downstream by xr_depthcal.
 * The XREAL cameras are grayscale, so the single channel is replicated to RGB. */
#ifndef XR_ZIPDEPTH_H
#define XR_ZIPDEPTH_H

#include <stdint.h>

/* Load ORT + the model and pick an execution provider. Retryable (returns 0
 * until the .so + model are present); thread-safe (one caller does the work).
 * dsp_dir = the directory holding the device-pulled FastRPC/DSP libs
 * (libcdsprpc.so + the DSP HAL, staged from assets/qnn_dsp); NULL/empty if not
 * staged (then the QNN HTP device can't be created and it falls back to CPU). */
int xr_zipdepth_init(const char *model_path, const char *dsp_dir);
int xr_zipdepth_available(void);

/* Run on an in_w*in_h grayscale rectified image (resized to the model's ZD_IN_*
 * input); write out_w*out_h affine-metric DEPTH (the model output resized to the
 * OUTPUT dims). Input and output dims are decoupled so ZipDepth can be fed a
 * full-res rectified image while the depth map stays at the anchor/display res.
 * Returns 0 on success, -1 otherwise. The caller inverts + calibrates. */
int xr_zipdepth_run(const uint8_t *gray, int in_w, int in_h,
                    float *depth_out, int out_w, int out_h);

#endif
