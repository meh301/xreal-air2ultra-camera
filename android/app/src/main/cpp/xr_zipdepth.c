/* xr_zipdepth.c — see xr_zipdepth.h. */
#include "xr_zipdepth.h"

#include <dlfcn.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
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
    char dsp_dir[256];                /* device-pulled FastRPC/DSP libs (or "") */
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

/* Detect SoC family from build properties; also returns the board string. */
static void zd_detect_soc(int *qcom, int *mtk, char *board, size_t bn) {
    char soc[PROP_VALUE_MAX] = {0}, b[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.manufacturer", soc);
    __system_property_get("ro.board.platform", b);
    lc(soc); lc(b);
    *qcom = strstr(soc, "qti") || strstr(soc, "qualcomm") ||
            strstr(b, "kona") || strstr(b, "lahaina") ||
            strstr(b, "taro") || strstr(b, "kalama") ||
            strstr(b, "pineapple") || strstr(b, "sm8");
    *mtk = strstr(soc, "mediatek") || strstr(b, "mt6");
    snprintf(board, bn, "%s", b);
}

/* Prepend our native-lib dirs to the two loader paths so QNN's HTP can reach the
 * Hexagon. THE reason this is needed: libQnnHtp.so must dlopen libcdsprpc.so (the
 * FastRPC client) + its DSP/system-lib closure, but /vendor/lib64 + /system/lib64
 * are NOT on an app's linker namespace -- so we bundle those libs in the app
 * native-lib dir (pull_dsp_libs.ps1 -> jniLibs) and:
 *   LD_LIBRARY_PATH   -> the linker finds libcdsprpc.so + the QNN CPU libs
 *   ADSP_LIBRARY_PATH -> FastRPC finds the DSP skel (libQnnHtpV68Skel.so)
 * (Bundling in nativeLibraryDir is what actually makes libcdsprpc loadable --
 * it's on libQnnHtp's own namespace; LD_LIBRARY_PATH is belt-and-suspenders.)
 * Writes the app native-lib dir (dladdr on our own code) into `libdir`. */
static void zd_fastrpc_env(char *libdir, size_t ln) {
    libdir[0] = 0;
    Dl_info info;
    if (dladdr((void *)zd_fastrpc_env, &info) && info.dli_fname) {
        const char *slash = strrchr(info.dli_fname, '/');
        if (slash) snprintf(libdir, ln, "%.*s",
                            (int)(slash - info.dli_fname), info.dli_fname);
    }
    const char *dsp = Z.dsp_dir[0] ? Z.dsp_dir : libdir;
    { const char *e = getenv("LD_LIBRARY_PATH"); char v[1400];
      snprintf(v, sizeof v, "%s:%s%s%s", dsp, libdir,
               (e && e[0]) ? ":" : "", (e && e[0]) ? e : "");
      setenv("LD_LIBRARY_PATH", v, 1); }
    { const char *e = getenv("ADSP_LIBRARY_PATH"); char v[1400];
      snprintf(v, sizeof v, "%s;%s%s%s", libdir, dsp,
               (e && e[0]) ? ";" : "", (e && e[0]) ? e : "");
      setenv("ADSP_LIBRARY_PATH", v, 1); }
    LOGI("ZipDepth: FastRPC libs dsp_dir=%s qnn_dir=%s (LD+ADSP set)", dsp, libdir);
}

/* Build session options (verbose log so the QNN backend prints WHY it rejects a
 * config; CPU-thread cap so a CPU fallback doesn't starve SLAM/render; graph
 * opt), optionally append execution provider `ep` with n key/value pairs, then
 * try to open the model. On success sets Z.session and returns 1; else logs
 * `label` + the reason and returns 0. */
static int zd_open(const char *model_path, const char *ep,
                   const char **k, const char **v, int n, const char *label) {
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(Z.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    ort_ok(Z.api->SetSessionLogSeverityLevel(opts, 0), "SetLogSeverity");
    ort_ok(Z.api->SetIntraOpNumThreads(opts, 2), "SetIntraOpNumThreads");
    ort_ok(Z.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL),
           "SetGraphOpt");
    if (ep) {
        OrtStatus *st = Z.api->SessionOptionsAppendExecutionProvider(
            opts, ep, k, v, n);
        if (st) {
            LOGE("ZipDepth: [%s] EP append failed: %s", label,
                 Z.api->GetErrorMessage(st));
            Z.api->ReleaseStatus(st);
            Z.api->ReleaseSessionOptions(opts);
            return 0;
        }
    }
    OrtStatus *st = Z.api->CreateSession(Z.env, model_path, opts, &Z.session);
    Z.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("ZipDepth: [%s] open failed: %s", label, Z.api->GetErrorMessage(st));
        Z.api->ReleaseStatus(st);
        Z.session = NULL;
        return 0;
    }
    LOGI("ZipDepth: [%s] session opened", label);
    return 1;
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

    int qcom = 0, mtk = 0;
    char board[PROP_VALUE_MAX] = {0};
    zd_detect_soc(&qcom, &mtk, board, sizeof board);

    int ok = 0;
    if (qcom) {
        char libdir[512];
        zd_fastrpc_env(libdir, sizeof libdir);
        char backend[560];
        if (libdir[0]) snprintf(backend, sizeof backend, "%s/libQnnHtp.so", libdir);
        else snprintf(backend, sizeof backend, "libQnnHtp.so");
        /* The HTP device is created inside CreateSession. On-device JIT wants the
         * HTP to AUTO-DETECT the SoC: passing an explicit soc_model/htp_arch
         * builds a QnnHtpDevice platform descriptor that this lahaina/V68 rejects
         * at QnnDevice_create as INVALID_CONFIG (verified: fails with the
         * descriptor even with libcdsprpc present + both QNN 2.31/2.33; ORT only
         * pushes that descriptor when soc/arch are set -- qnn_backend_manager.cc
         * CreateDevice). So try auto-detect FIRST (no soc/arch), then the explicit
         * variants as fallbacks -- first that opens the model wins. fp16 lets the
         * FP32 model run on the HTP (the no-calibration alternative to INT8 QDQ);
         * opt mode 3 = best graph finalization. */
        const char *k0[] = { "backend_path", "enable_htp_fp16_precision",
                             "htp_graph_finalization_optimization_mode" };
        const char *v0[] = { backend, "1", "3" };
        const char *k1[] = { "backend_path", "htp_arch",
                             "enable_htp_fp16_precision",
                             "htp_graph_finalization_optimization_mode" };
        const char *v1[] = { backend, "68", "1", "3" };
        const char *k2[] = { "backend_path", "soc_model",
                             "enable_htp_fp16_precision",
                             "htp_graph_finalization_optimization_mode" };
        const char *v2[] = { backend, "30", "1", "3" };
        LOGI("ZipDepth: QNN HTP init [board='%s'] %s", board, backend);
        ok = zd_open(model_path, "QNN", k0, v0, 3, "QNN HTP auto-detect fp16")
          || zd_open(model_path, "QNN", k1, v1, 4, "QNN HTP arch=68 fp16")
          || zd_open(model_path, "QNN", k2, v2, 4, "QNN HTP soc=30 fp16");
        if (!ok) LOGE("ZipDepth: all QNN HTP configs failed -> CPU fallback");
    } else if (mtk) {
        ok = zd_open(model_path, ZD_MTK_EP, NULL, NULL, 0, "MediaTek NeuroPilot");
    }
    if (!ok && !zd_open(model_path, NULL, NULL, NULL, 0, "CPU"))
        return 0;

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

int xr_zipdepth_init(const char *model_path, const char *dsp_dir) {
    if (atomic_load_explicit(&Z.ready, memory_order_acquire)) return 1;
    static atomic_int busy;
    if (atomic_exchange(&busy, 1)) return 0;
    Z.dsp_dir[0] = 0;
    if (dsp_dir && dsp_dir[0])
        strncpy(Z.dsp_dir, dsp_dir, sizeof Z.dsp_dir - 1);
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
