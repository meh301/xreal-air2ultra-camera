/* xr_liteanystereo.h — metric stereo depth (LiteAnyStereo V2-S, CVPR2026) on the
 * NPU via ONNX Runtime + QNN (Hexagon HTP). The model is a 2-input stereo net
 * (rectified left+right, 480x640) that outputs disparity DIRECTLY -> metric depth
 * = f*baseline/disp, no monocular model, no anchors, no calibration. Loaded from
 * an AoT QNN context binary wrapped as an EPContext ONNX (QnnContext_createFrom
 * Binary -> skips the on-device online graph-prepare that fails on SD888).
 *
 * Replaces the ZipDepth-mono + xr_sgrid anchors + xr_depthcal calibration stack.
 * dlopen/QNN pattern mirrors xr_zipdepth: without libonnxruntime.so or the model
 * it reports unavailable and the depth worker falls back to SGM (xr_stereo_depth).
 */
#ifndef XR_LITEANYSTEREO_H
#define XR_LITEANYSTEREO_H

#include <stdint.h>

/* Load ORT + the EPContext model and pick the QNN HTP execution provider.
 * Retryable (returns 0 until the .so + model are present); thread-safe.
 * dsp_dir = directory with device-pulled FastRPC/DSP libs (or NULL/empty). */
int xr_las2_init(const char *model_path, const char *dsp_dir);
int xr_las2_available(void);

/* Run on the rectified pair (each ZDR_W*ZDR_H = 480x640 grayscale, [0]=left).
 * Writes metric DEPTH (metres, 0 = invalid) into depth_out[out_w*out_h],
 * subsampled from the 480x640 disparity. f_hi = full-res rectified focal (px),
 * base = stereo baseline (m). depth is resolution-invariant so the subsample
 * carries the metric straight through. Returns 0 on success, -1 otherwise. */
int xr_las2_run(const uint8_t *left, const uint8_t *right, float f_hi, float base,
                float *depth_out, int out_w, int out_h);

#endif
