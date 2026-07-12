/* xr_zipdepth.c — see xr_zipdepth.h. */
#include "xr_zipdepth.h"

#include <dlfcn.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/system_properties.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* Exported ONNX resolution — MUST match scripts/export.py --width/--height.
 * 384x384 is the tested default; for the 3:4 rectified frame, exporting at
 * 288x384 preserves aspect (both multiples of 32) and is preferable. */
#define ZD_IN_W 384
#define ZD_IN_H 384

/* MediaTek NeuroPilot ORT execution-provider name. CONFIRM against the
 * NeuroPilot-enabled ORT build; a wrong name just falls back to CPU. */
#define ZD_MTK_EP "NeuroPilot"

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *meminfo;
    char in_name[64];
    char out_name[64];
    atomic_int ready;                 /* release-published once usable */
    float input[3 * ZD_IN_H * ZD_IN_W];   /* NCHW RGB, init-thread only */
} Z;

static int ort_ok(OrtStatus *st, const char *what) {
    if (!st) return 1;
    LOGE("ZipDepth/ORT %s: %s", what, Z.api->GetErrorMessage(st));
    Z.api->ReleaseStatus(st);
    return 0;
}

static void lc(char *s) { for (; *s; s++) if (*s >= 'A' && *s <= 'Z') *s += 32; }

/* Pick an NPU execution provider by SoC; leave CPU (default) if none fits or
 * the provider isn't in this ORT build (append returns an error -> fall back). */
static void zd_append_ep(OrtSessionOptions *opts) {
    char soc[PROP_VALUE_MAX] = {0}, board[PROP_VALUE_MAX] = {0};
    int rc0 = __system_property_get("ro.soc.manufacturer", soc);
    int rc1 = __system_property_get("ro.board.platform", board);
    (void)rc0; (void)rc1;
    lc(soc); lc(board);
    int qcom = strstr(soc, "qti") || strstr(soc, "qualcomm") ||
               strstr(board, "kona") || strstr(board, "lahaina") ||
               strstr(board, "taro") || strstr(board, "kalama") ||
               strstr(board, "pineapple") || strstr(board, "sm8");
    int mtk = strstr(soc, "mediatek") || strstr(board, "mt6");

    OrtStatus *st = NULL;
    if (qcom) {
        /* Hexagon HTP (NPU) backend at burst clocks. The backend .so and the
         * per-arch V*Stub/V*Skel libs must be packaged in jniLibs (see
         * android/fetch_qnn.ps1); libcdsprpc.so is a device system lib. */
        const char *k[] = { "backend_path", "htp_performance_mode" };
        const char *v[] = { "libQnnHtp.so", "burst" };
        st = Z.api->SessionOptionsAppendExecutionProvider(opts, "QNN", k, v, 2);
        if (!st) { LOGI("ZipDepth: QNN (Hexagon HTP, burst) EP"); return; }
    } else if (mtk) {
        st = Z.api->SessionOptionsAppendExecutionProvider(opts, ZD_MTK_EP,
                                                          NULL, NULL, 0);
        if (!st) { LOGI("ZipDepth: MediaTek (NeuroPilot) EP"); return; }
    }
    if (st) {
        LOGE("ZipDepth: NPU EP unavailable (%s) — CPU",
             Z.api->GetErrorMessage(st));
        Z.api->ReleaseStatus(st);
    } else {
        LOGI("ZipDepth: CPU EP (no NPU match: soc='%s' board='%s')", soc, board);
    }
}

static int zd_try_init(const char *model_path) {
    void *dl = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        LOGI("ZipDepth: libonnxruntime.so not present (%s) — SGM depth",
             dlerror());
        return 0;
    }
    const OrtApiBase *(*get_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(dl, "OrtGetApiBase");
    if (!get_base) { LOGE("ZipDepth: OrtGetApiBase missing"); return 0; }
    Z.api = get_base()->GetApi(ORT_API_VERSION);
    if (!Z.api) { LOGE("ZipDepth: ORT api %d unavailable", ORT_API_VERSION); return 0; }

    if (!ort_ok(Z.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "zipdepth", &Z.env),
                "CreateEnv"))
        return 0;
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(Z.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    zd_append_ep(opts);                       /* NPU EP or CPU */
    ort_ok(Z.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL),
           "SetGraphOpt");
    if (!ort_ok(Z.api->CreateSession(Z.env, model_path, opts, &Z.session),
                "CreateSession")) {
        Z.api->ReleaseSessionOptions(opts);
        return 0;
    }
    Z.api->ReleaseSessionOptions(opts);
    if (!ort_ok(Z.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &Z.meminfo), "CreateCpuMemoryInfo"))
        return 0;

    OrtAllocator *alloc = NULL;
    ort_ok(Z.api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocator");
    char *name = NULL;
    if (!ort_ok(Z.api->SessionGetInputName(Z.session, 0, alloc, &name),
                "GetInputName"))
        return 0;
    strncpy(Z.in_name, name, sizeof Z.in_name - 1);
    alloc->Free(alloc, name);
    if (!ort_ok(Z.api->SessionGetOutputName(Z.session, 0, alloc, &name),
                "GetOutputName"))
        return 0;
    strncpy(Z.out_name, name, sizeof Z.out_name - 1);
    alloc->Free(alloc, name);

    atomic_store_explicit(&Z.ready, 1, memory_order_release);
    LOGI("ZipDepth ready: %s (%dx%d, in=%s out=%s)", model_path,
         ZD_IN_W, ZD_IN_H, Z.in_name, Z.out_name);
    return 1;
}

int xr_zipdepth_init(const char *model_path) {
    if (atomic_load_explicit(&Z.ready, memory_order_acquire)) return 1;
    static atomic_int busy;
    if (atomic_exchange(&busy, 1)) return 0;
    int ok = zd_try_init(model_path);
    if (!ok) atomic_store(&busy, 0);
    return ok;
}

int xr_zipdepth_available(void) {
    return atomic_load_explicit(&Z.ready, memory_order_acquire);
}

/* bilinear sample of a w*h grayscale image at (fx,fy) */
static float samp_u8(const uint8_t *g, int w, int h, float fx, float fy) {
    if (fx < 0) fx = 0; if (fx > w - 1) fx = (float)(w - 1);
    if (fy < 0) fy = 0; if (fy > h - 1) fy = (float)(h - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
    float ax = fx - x0, ay = fy - y0;
    const uint8_t *r0 = g + (size_t)y0 * w, *r1 = g + (size_t)y1 * w;
    float a = r0[x0] * (1 - ax) + r0[x1] * ax;
    float b = r1[x0] * (1 - ax) + r1[x1] * ax;
    return a * (1 - ay) + b * ay;
}

/* bilinear sample of a w*h float image at (fx,fy) */
static float samp_f(const float *m, int w, int h, float fx, float fy) {
    if (fx < 0) fx = 0; if (fx > w - 1) fx = (float)(w - 1);
    if (fy < 0) fy = 0; if (fy > h - 1) fy = (float)(h - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
    float ax = fx - x0, ay = fy - y0;
    const float *r0 = m + (size_t)y0 * w, *r1 = m + (size_t)y1 * w;
    float a = r0[x0] * (1 - ax) + r0[x1] * ax;
    float b = r1[x0] * (1 - ax) + r1[x1] * ax;
    return a * (1 - ay) + b * ay;
}

int xr_zipdepth_run(const uint8_t *gray, int w, int h, float *depth_out) {
    if (!atomic_load_explicit(&Z.ready, memory_order_acquire)) return -1;
    if (!gray || !depth_out || w <= 0 || h <= 0) return -1;

    /* preprocess: resize gray -> ZD input, /255, replicate to 3 channels */
    const int plane = ZD_IN_H * ZD_IN_W;
    const float sx = (float)w / ZD_IN_W, sy = (float)h / ZD_IN_H;
    for (int yz = 0; yz < ZD_IN_H; yz++) {
        for (int xz = 0; xz < ZD_IN_W; xz++) {
            float v = samp_u8(gray, w, h, (xz + 0.5f) * sx - 0.5f,
                              (yz + 0.5f) * sy - 0.5f) * (1.0f / 255.0f);
            Z.input[yz * ZD_IN_W + xz] = v;     /* R */
        }
    }
    memcpy(Z.input + plane, Z.input, sizeof(float) * plane);       /* G */
    memcpy(Z.input + 2 * plane, Z.input, sizeof(float) * plane);   /* B */

    const int64_t shape[4] = { 1, 3, ZD_IN_H, ZD_IN_W };
    OrtValue *in = NULL;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.meminfo, Z.input, sizeof Z.input, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in), "CreateTensor"))
        return -1;

    const char *ins[1] = { Z.in_name };
    const char *outs[1] = { Z.out_name };
    OrtValue *ov = NULL;
    OrtStatus *st = Z.api->Run(Z.session, NULL, ins,
                               (const OrtValue *const *)&in, 1, outs, 1, &ov);
    Z.api->ReleaseValue(in);
    if (!ort_ok(st, "Run")) return -1;

    float *depth = NULL;
    int ok = ort_ok(Z.api->GetTensorMutableData(ov, (void **)&depth), "out data");
    if (ok && depth) {
        /* resize the ZD_IN depth back to w*h */
        const float rx = (float)ZD_IN_W / w, ry = (float)ZD_IN_H / h;
        for (int yo = 0; yo < h; yo++)
            for (int xo = 0; xo < w; xo++)
                depth_out[yo * w + xo] =
                    samp_f(depth, ZD_IN_W, ZD_IN_H, (xo + 0.5f) * rx - 0.5f,
                           (yo + 0.5f) * ry - 0.5f);
    }
    Z.api->ReleaseValue(ov);
    return ok ? 0 : -1;
}
