/* xr_lighterglue.c — see xr_lighterglue.h. ORT CPU inference, one session
 * for the process, modeled on xr_vpr's dlopen'd path (nothing links
 * libonnxruntime). Map-thread only, like every matcher stage. */
#include "xr_lighterglue.h"

#include <dlfcn.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *meminfo;
    char path[512];
    atomic_int ready;
    int failed;                        /* permanent: don't retry every call */
    float min_score;
    /* static-shape input planes (map thread only) */
    float kpts[2][XR_LGLUE_N][2];
    float desc[2][XR_LGLUE_N][64];
} G;

static int ort_ok(OrtStatus *st, const char *what) {
    if (!st) return 1;
    LOGE("LGLUE/ORT %s: %s", what, G.api->GetErrorMessage(st));
    G.api->ReleaseStatus(st);
    return 0;
}

void xr_lglue_set_model(const char *onnx_path) {
    if (!onnx_path || !onnx_path[0]) return;
    strncpy(G.path, onnx_path, sizeof G.path - 1);
}

int xr_lglue_ready(void) {
    return atomic_load_explicit(&G.ready, memory_order_acquire);
}

int xr_lglue_wanted(void) {
    return G.path[0] != 0 && !G.failed;
}

static int lglue_try_init(void) {
    void *dl = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        LOGI("LGLUE: libonnxruntime.so not present (%s) — matcher off",
             dlerror());
        return 0;
    }
    const OrtApiBase *(*get_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(dl, "OrtGetApiBase");
    if (!get_base) return 0;
    G.api = get_base()->GetApi(ORT_API_VERSION);
    if (!G.api) return 0;
    if (!ort_ok(G.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xrlglue", &G.env),
                "CreateEnv"))
        return 0;
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(G.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    G.api->SetIntraOpNumThreads(opts, 2);
    G.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    if (!ort_ok(G.api->CreateSession(G.env, G.path, opts, &G.session),
                "CreateSession")) {
        G.api->ReleaseSessionOptions(opts);
        return 0;
    }
    G.api->ReleaseSessionOptions(opts);
    if (!ort_ok(G.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &G.meminfo), "CreateCpuMemoryInfo"))
        return 0;
    const char *e = getenv("XR_LGLUE_MIN");
    G.min_score = e && *e ? (float)atof(e) : 0.35f;
    atomic_store_explicit(&G.ready, 1, memory_order_release);
    LOGI("LGLUE ready: %s (%d slots, min score %.2f)", G.path, XR_LGLUE_N,
         (double)G.min_score);
    return 1;
}

/* fill one side's padded planes: normalized kpts + dequantized descs */
static void lglue_fill(int side, const float (*uv)[2], const int8_t (*d)[64],
                       int n, float w_px, float h_px) {
    float sx = w_px * 0.5f, sy = h_px * 0.5f;
    float sc = 1.0f / (w_px > h_px ? sx : sy);
    memset(G.kpts[side], 0, sizeof G.kpts[side]);
    memset(G.desc[side], 0, sizeof G.desc[side]);
    for (int i = 0; i < n; i++) {
        G.kpts[side][i][0] = (uv[i][0] - sx) * sc;
        G.kpts[side][i][1] = (uv[i][1] - sy) * sc;
        for (int c = 0; c < 64; c++)
            G.desc[side][i][c] = (float)d[i][c] * (1.0f / 127.0f);
    }
}

int xr_lglue_match(const float (*uv0)[2], const int8_t (*d0)[64], int n0,
                   const float (*uv1)[2], const int8_t (*d1)[64], int n1,
                   float w_px, float h_px,
                   int *out_i0, int *out_i1, float *out_sc, int max_out) {
    if (!atomic_load_explicit(&G.ready, memory_order_acquire)) {
        if (G.failed || !G.path[0]) return -1;
        if (!lglue_try_init()) {       /* map thread; one attempt only */
            G.failed = 1;
            return -1;
        }
    }
    if (n0 > XR_LGLUE_N) n0 = XR_LGLUE_N;
    if (n1 > XR_LGLUE_N) n1 = XR_LGLUE_N;
    if (n0 < 4 || n1 < 4) return 0;
    lglue_fill(0, uv0, d0, n0, w_px, h_px);
    lglue_fill(1, uv1, d1, n1, w_px, h_px);

    const int64_t kshape[3] = { 1, XR_LGLUE_N, 2 };
    const int64_t dshape[3] = { 1, XR_LGLUE_N, 64 };
    OrtValue *in[4] = { NULL, NULL, NULL, NULL };
    if (!ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                    G.meminfo, G.kpts[0], sizeof G.kpts[0], kshape, 3,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[0]), "kpts0") ||
        !ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                    G.meminfo, G.kpts[1], sizeof G.kpts[1], kshape, 3,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[1]), "kpts1") ||
        !ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                    G.meminfo, G.desc[0], sizeof G.desc[0], dshape, 3,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[2]), "desc0") ||
        !ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                    G.meminfo, G.desc[1], sizeof G.desc[1], dshape, 3,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in[3]), "desc1")) {
        for (int i = 0; i < 4; i++)
            if (in[i]) G.api->ReleaseValue(in[i]);
        return -1;
    }
    const char *ins[4] = { "kpts0", "kpts1", "desc0", "desc1" };
    const char *outs[2] = { "matches", "scores" };
    OrtValue *ov[2] = { NULL, NULL };
    OrtStatus *st = G.api->Run(G.session, NULL, ins,
                               (const OrtValue *const *)in, 4, outs, 2, ov);
    for (int i = 0; i < 4; i++) G.api->ReleaseValue(in[i]);
    if (!ort_ok(st, "Run")) {
        for (int i = 0; i < 2; i++)
            if (ov[i]) G.api->ReleaseValue(ov[i]);
        return -1;
    }

    int nm = 0;
    int64_t *mi = NULL;
    float *sc = NULL;
    OrtTensorTypeAndShapeInfo *tsi = NULL;
    int64_t mshape[2] = { 0, 0 };
    size_t nd = 0;
    if (ort_ok(G.api->GetTensorTypeAndShape(ov[0], &tsi), "match shape")) {
        G.api->GetDimensionsCount(tsi, &nd);
        G.api->GetDimensions(tsi, mshape, nd < 2 ? nd : 2);
        G.api->ReleaseTensorTypeAndShapeInfo(tsi);
    }
    if (ort_ok(G.api->GetTensorMutableData(ov[0], (void **)&mi), "match data") &&
        ort_ok(G.api->GetTensorMutableData(ov[1], (void **)&sc), "score data") &&
        mi && sc) {
        int k = (int)mshape[0];
        for (int m = 0; m < k && nm < max_out; m++) {
            int i0 = (int)mi[m * 2], i1 = (int)mi[m * 2 + 1];
            if (i0 < 0 || i0 >= n0 || i1 < 0 || i1 >= n1) continue;   /* pads */
            if (sc[m] < G.min_score) continue;
            out_i0[nm] = i0;
            out_i1[nm] = i1;
            out_sc[nm] = sc[m];
            nm++;
        }
    }
    G.api->ReleaseValue(ov[0]);
    G.api->ReleaseValue(ov[1]);
    return nm;
}
