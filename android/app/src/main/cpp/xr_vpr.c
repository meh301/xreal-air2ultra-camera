/* xr_vpr.c — see xr_vpr.h. ORT CPU inference, one session for the process,
 * modeled on xr_xfeat's dlopen'd CPU path (nothing links libonnxruntime). */
#include "xr_vpr.h"

#include <dlfcn.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"
#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *meminfo;
    char in_name[64];
    char out_name[64];
    char path[512];
    atomic_int ready;
    int failed;                       /* permanent: don't retry every frame */
    int dim;                          /* model output dim, set at bring-up */
    float input[XR_OH * XR_OW];       /* map thread only */
} V;

static int ort_ok(OrtStatus *st, const char *what) {
    if (!st) return 1;
    LOGE("VPR/ORT %s: %s", what, V.api->GetErrorMessage(st));
    V.api->ReleaseStatus(st);
    return 0;
}

void xr_vpr_set_model(const char *onnx_path) {
    if (!onnx_path || !onnx_path[0]) return;
    strncpy(V.path, onnx_path, sizeof V.path - 1);
}

int xr_vpr_ready(void) {
    return atomic_load_explicit(&V.ready, memory_order_acquire);
}

int xr_vpr_dim(void) {
    return atomic_load_explicit(&V.ready, memory_order_acquire) ? V.dim : 0;
}

static int vpr_try_init(void) {
    void *dl = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        LOGI("VPR: libonnxruntime.so not present (%s) — retrieval off",
             dlerror());
        return 0;
    }
    const OrtApiBase *(*get_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(dl, "OrtGetApiBase");
    if (!get_base) return 0;
    V.api = get_base()->GetApi(ORT_API_VERSION);
    if (!V.api) return 0;
    if (!ort_ok(V.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xrvpr", &V.env),
                "CreateEnv"))
        return 0;
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(V.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    V.api->SetIntraOpNumThreads(opts, 2);
    V.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    if (!ort_ok(V.api->CreateSession(V.env, V.path, opts, &V.session),
                "CreateSession")) {
        V.api->ReleaseSessionOptions(opts);
        return 0;
    }
    V.api->ReleaseSessionOptions(opts);
    if (!ort_ok(V.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &V.meminfo), "CreateCpuMemoryInfo"))
        return 0;
    OrtAllocator *alloc = NULL;
    V.api->GetAllocatorWithDefaultOptions(&alloc);
    char *name = NULL;
    if (!ort_ok(V.api->SessionGetInputName(V.session, 0, alloc, &name),
                "GetInputName"))
        return 0;
    strncpy(V.in_name, name, sizeof V.in_name - 1);
    alloc->Free(alloc, name);
    if (!ort_ok(V.api->SessionGetOutputName(V.session, 0, alloc, &name),
                "GetOutputName"))
        return 0;
    strncpy(V.out_name, name, sizeof V.out_name - 1);
    alloc->Free(alloc, name);
    /* discover the embedding dimension from the output shape [1, D] */
    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    int64_t d[2] = { 0, 0 };
    size_t nd = 0;
    if (!ort_ok(V.api->SessionGetOutputTypeInfo(V.session, 0, &ti),
                "OutputTypeInfo"))
        return 0;
    V.api->CastTypeInfoToTensorInfo(ti, &tsi);
    if (tsi) {
        V.api->GetDimensionsCount(tsi, &nd);
        V.api->GetDimensions(tsi, d, nd < 2 ? nd : 2);
    }
    V.api->ReleaseTypeInfo(ti);
    V.dim = (int)d[1];
    if (V.dim <= 0 || V.dim > XR_VPR_MAX_DIM) {
        LOGE("VPR: unusable embedding dim %d (max %d)", V.dim, XR_VPR_MAX_DIM);
        return 0;
    }
    atomic_store_explicit(&V.ready, 1, memory_order_release);
    LOGI("VPR ready: %s (%d-D, in=%s out=%s)", V.path, V.dim,
         V.in_name, V.out_name);
    return 1;
}

int xr_vpr_embed(const uint8_t *img, float emb[XR_VPR_MAX_DIM]) {
    if (!atomic_load_explicit(&V.ready, memory_order_acquire)) {
        if (V.failed || !V.path[0]) return -1;
        if (!vpr_try_init()) {         /* map thread; one attempt only */
            V.failed = 1;
            return -1;
        }
    }
    const int plane = XR_OH * XR_OW;
    for (int i = 0; i < plane; i++) V.input[i] = (float)img[i];
    const int64_t shape[4] = { 1, 1, XR_OH, XR_OW };
    OrtValue *in = NULL;
    if (!ort_ok(V.api->CreateTensorWithDataAsOrtValue(
                    V.meminfo, V.input, sizeof V.input, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in), "CreateTensor"))
        return -1;
    const char *ins[1] = { V.in_name };
    const char *outs[1] = { V.out_name };
    OrtValue *ov = NULL;
    OrtStatus *st = V.api->Run(V.session, NULL, ins,
                               (const OrtValue *const *)&in, 1, outs, 1, &ov);
    V.api->ReleaseValue(in);
    if (!ort_ok(st, "Run")) return -1;
    float *e = NULL;
    int ok = ort_ok(V.api->GetTensorMutableData(ov, (void **)&e), "out data");
    if (ok && e) {
        /* graph L2-normalizes; renormalize anyway so stored dot products
         * are true cosines even under numeric drift */
        float n = 0;
        for (int i = 0; i < V.dim; i++) n += e[i] * e[i];
        n = n > 1e-12f ? 1.0f / sqrtf(n) : 0.0f;
        for (int i = 0; i < V.dim; i++) emb[i] = e[i] * n;
    }
    V.api->ReleaseValue(ov);
    return ok ? V.dim : -1;
}
