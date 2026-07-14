/* xr_liteanystereo.h — metric stereo depth (LiteAnyStereo V2-S, CVPR2026) on the
 * NPU via ONNX Runtime + QNN (Hexagon HTP). 1-channel folded models (first conv
 * + ImageNet norm folded in; host-verified exact) run DIRECTLY on the rectified
 * pair -> disparity -> metric depth = f*baseline/disp; no monocular model, no
 * anchors, no calibration. Loaded from AoT QNN context binaries wrapped as
 * EPContext ONNX (QnnContext_createFromBinary; the v68 skel is integer-only, so
 * only QAIRT-native-quantized contexts deserialize — see memory las2-deployment).
 *
 * TWO model slots (demo quality tiers): slot 0 = 192x256 fast (~28-32 ms),
 * slot 1 = 288x384 MID (~65-72 ms). Graph I/O is u16-quantized; the per-model
 * scales come from the context metadata and are passed at init. Without ORT /
 * the model a slot reports unavailable and the depth worker falls back to SGM. */
#ifndef XR_LITEANYSTEREO_H
#define XR_LITEANYSTEREO_H

#include <stdint.h>

enum { XR_LAS2_FAST = 0, XR_LAS2_MID = 1, XR_LAS2_SLOTS = 2 };

/* Load ORT + the EPContext model into `slot` and pick the QNN HTP execution
 * provider. w/h = the model's input size; in_scale/out_scale = the context's
 * u16 quantization scales (offset 0). Retryable (returns 0 until the .so +
 * model are present); one caller at a time. dsp_dir = optional dir with
 * device-pulled FastRPC/DSP libs. */
int xr_las2_init(int slot, const char *model_path, const char *dsp_dir,
                 int w, int h, float in_scale, float out_scale);
int xr_las2_available(int slot);

/* Run `slot` on a rectified pair AT THAT MODEL'S RESOLUTION (w*h grayscale,
 * [0]=left). Writes metric DEPTH (metres, 0 = invalid) into
 * depth_out[out_w*out_h], subsampled from the model-res disparity. f_model =
 * rectified focal at the MODEL resolution (px), base = baseline (m). Depth is
 * resolution-invariant so the subsample carries the metric straight through.
 * Returns 0 on success, -1 otherwise. */
int xr_las2_run(int slot, const uint8_t *left, const uint8_t *right,
                float f_model, float base, float *depth_out,
                int out_w, int out_h);

#endif
