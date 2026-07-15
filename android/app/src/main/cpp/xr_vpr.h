/* xr_vpr.h — visual place recognition embeddings for the session map.
 *
 * One L2-normalized global descriptor per keyframe (EigenPlaces-512 today;
 * MegaLoc behind the same interface later — the embedding dimension is the
 * only contract). The map uses them to PRE-RANK loop-closure/relocalization
 * candidates by appearance (a dot product per keyframe) so the expensive
 * per-keypoint descriptor matching runs only on a short list, replacing the
 * O(all-keyframes) brute scan. Embeddings are model-version-locked: maps
 * store the raw vectors, so query and store must come from the same model.
 *
 * Inference runs through ONNX Runtime (dlopen'd, CPU EP — the benchmark
 * container path; the on-device NPU context variant arrives with the
 * Snapdragon Gen 5 port and must take XR_NPU_GATE). The model takes
 * [1,1,H,W] raw gray 0..255 (our EigenPlaces export bakes ImageNet
 * normalization into the graph) and returns a unit-norm [1,DIM] embedding.
 */
#ifndef XR_VPR_H
#define XR_VPR_H

#include <stdint.h>

/* Upper bound across supported models: EigenPlaces-512 .. MegaLoc-8448.
 * The ACTUAL dimension is discovered from the model's output shape at
 * bring-up; embeddings from different models/dims never compare. */
#define XR_VPR_MAX_DIM 8448

/* Register the staged ONNX model path (cheap; no I/O). The first embed
 * call performs the lazy ORT bring-up on the calling (map) thread. */
void xr_vpr_set_model(const char *onnx_path);

/* 1 once a session is up (i.e., after the first successful embed). */
int xr_vpr_ready(void);

/* Embedding dimension of the active model (0 until a session is up). */
int xr_vpr_dim(void);

/* Embed one XR_OW x XR_OH grayscale frame. Returns the embedding
 * dimension (>0) and fills emb[0..dim) L2-normalized on success,
 * -1 when unavailable or on inference error. */
int xr_vpr_embed(const uint8_t *img, float emb[XR_VPR_MAX_DIM]);

#endif
