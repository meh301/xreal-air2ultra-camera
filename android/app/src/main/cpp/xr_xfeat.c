/* xr_xfeat.c — see xr_xfeat.h. */
#include "xr_xfeat.h"

#include <dlfcn.h>
#include <math.h>
#include <string.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"
#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define IMG_W XR_OW      /* 480 */
#define IMG_H XR_OH      /* 640 */

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *meminfo;
    char in_name[64];
    char out_names[3][64];   /* keypoints, descriptors, scores */
    int ready;
    float input[3 * IMG_H * IMG_W];
} X;

static int ort_ok(OrtStatus *st, const char *what) {
    if (!st) return 1;
    LOGE("XFeat/ORT %s: %s", what, X.api->GetErrorMessage(st));
    X.api->ReleaseStatus(st);
    return 0;
}

int xr_xfeat_init(const char *model_path) {
    static int attempted;
    if (X.ready) return 1;
    if (attempted) return 0;
    attempted = 1;

    void *dl = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        LOGI("XFeat: libonnxruntime.so not present (%s) — mini-ORB fallback",
             dlerror());
        return 0;
    }
    const OrtApiBase *(*get_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(dl, "OrtGetApiBase");
    if (!get_base) {
        LOGE("XFeat: OrtGetApiBase missing");
        return 0;
    }
    X.api = get_base()->GetApi(ORT_API_VERSION);
    if (!X.api) {
        LOGE("XFeat: ORT api version %d unavailable", ORT_API_VERSION);
        return 0;
    }
    if (!ort_ok(X.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xfeat", &X.env),
                "CreateEnv"))
        return 0;
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(X.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    X.api->SetIntraOpNumThreads(opts, 2);
    X.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    if (!ort_ok(X.api->CreateSession(X.env, model_path, opts, &X.session),
                "CreateSession")) {
        X.api->ReleaseSessionOptions(opts);
        return 0;
    }
    X.api->ReleaseSessionOptions(opts);
    if (!ort_ok(X.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &X.meminfo), "CreateCpuMemoryInfo"))
        return 0;

    /* discover the io names (single input; outputs mapped by position:
     * the export order is keypoints, descriptors, scores) */
    OrtAllocator *alloc = NULL;
    X.api->GetAllocatorWithDefaultOptions(&alloc);
    char *name = NULL;
    if (!ort_ok(X.api->SessionGetInputName(X.session, 0, alloc, &name),
                "GetInputName"))
        return 0;
    strncpy(X.in_name, name, sizeof X.in_name - 1);
    alloc->Free(alloc, name);
    for (int i = 0; i < 3; i++) {
        if (!ort_ok(X.api->SessionGetOutputName(X.session, (size_t)i, alloc,
                                                &name), "GetOutputName"))
            return 0;
        strncpy(X.out_names[i], name, sizeof X.out_names[i] - 1);
        alloc->Free(alloc, name);
    }
    X.ready = 1;
    LOGI("XFeat ready: %s (in=%s outs=%s,%s,%s)", model_path, X.in_name,
         X.out_names[0], X.out_names[1], X.out_names[2]);
    return 1;
}

int xr_xfeat_available(void) {
    return X.ready;
}

int xr_xfeat_extract(const uint8_t *img, float (*uv)[2],
                     int8_t (*desc)[64], int max) {
    if (!X.ready) return -1;

    /* gray u8 -> float, replicated to 3 channels (raw 0..255 range) */
    const int plane = IMG_H * IMG_W;
    for (int i = 0; i < plane; i++) X.input[i] = (float)img[i];
    memcpy(X.input + plane, X.input, sizeof(float) * plane);
    memcpy(X.input + 2 * plane, X.input, sizeof(float) * plane);

    const int64_t shape[4] = { 1, 3, IMG_H, IMG_W };
    OrtValue *in = NULL;
    if (!ort_ok(X.api->CreateTensorWithDataAsOrtValue(
                    X.meminfo, X.input, sizeof X.input, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
                "CreateTensor"))
        return -1;

    const char *ins[1] = { X.in_name };
    const char *outs[3] = { X.out_names[0], X.out_names[1], X.out_names[2] };
    OrtValue *ov[3] = { NULL, NULL, NULL };
    OrtStatus *st = X.api->Run(X.session, NULL, ins,
                               (const OrtValue *const *)&in, 1, outs, 3, ov);
    X.api->ReleaseValue(in);
    if (!ort_ok(st, "Run")) return -1;

    int n = -1;
    OrtTensorTypeAndShapeInfo *info = NULL;
    if (ort_ok(X.api->GetTensorTypeAndShape(ov[0], &info), "GetShape")) {
        size_t dims = 0;
        int64_t d[4] = { 0 };
        X.api->GetDimensionsCount(info, &dims);
        X.api->GetDimensions(info, d, dims < 4 ? dims : 4);
        X.api->ReleaseTensorTypeAndShapeInfo(info);
        n = (int)d[0];
    }
    float *kp = NULL, *dsc = NULL;
    if (n > 0 &&
        ort_ok(X.api->GetTensorMutableData(ov[0], (void **)&kp), "kp data") &&
        ort_ok(X.api->GetTensorMutableData(ov[1], (void **)&dsc), "desc data")) {
        if (n > max) n = max;          /* scores are sorted descending */
        for (int i = 0; i < n; i++) {
            uv[i][0] = kp[i * 2];
            uv[i][1] = kp[i * 2 + 1];
            for (int k = 0; k < 64; k++) {
                float v = dsc[i * 64 + k] * 127.0f;
                desc[i][k] = (int8_t)(v > 127.f ? 127 : v < -127.f ? -127
                                      : lroundf(v));
            }
        }
    } else if (n < 0) {
        n = -1;
    }
    for (int i = 0; i < 3; i++)
        if (ov[i]) X.api->ReleaseValue(ov[i]);
    return n;
}
