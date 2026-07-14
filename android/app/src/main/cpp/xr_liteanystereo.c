/* xr_liteanystereo.c — see xr_liteanystereo.h. QNN/ORT setup mirrors xr_zipdepth
 * (same on-device wall: online graph-prepare fails on SD888, so the model is an
 * AoT EPContext binary that loads via QnnContext_createFromBinary). */
#include "xr_liteanystereo.h"

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

/* Model input, per single image: NCHW (1,3,LAS_H,LAS_W), RGB (gray replicated),
 * raw [0,255] (LAS normalizes internally). MUST match the exported/compiled
 * shape (left/right [1,3,256,192]); the AoT context binary is fixed-shape.
 * The model runs at a REDUCED resolution (192x256) to fit the unsigned-PD DSP
 * memory budget (480x640 needs ~244 MB, far past the PD ration; 192x256 ~35 MB).
 * max_disp is locked at 192 by the trained weights, so 192 wide is the floor. */
#define LAS_W 192
#define LAS_H 256
/* The rectified source (xr_stereo rect_hi) is 480x640; xr_las2_run downsamples it
 * to LAS_W x LAS_H and rescales the focal for the metric-depth conversion. */
#define LAS_SRC_W 480
#define LAS_SRC_H 640

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *meminfo;
    char in_name[2][64];              /* left, right */
    char out_name[64];                /* disparity */
    char dsp_dir[256];
    atomic_int ready;
    float in_l[3 * LAS_H * LAS_W];    /* NCHW input buffers, init/run-thread only */
    float in_r[3 * LAS_H * LAS_W];
} Z;

static int ort_ok(OrtStatus *st, const char *what) {
    if (!st) return 1;
    LOGE("LAS2/ORT %s: %s", what, Z.api->GetErrorMessage(st));
    Z.api->ReleaseStatus(st);
    return 0;
}

static void lc(char *s) { for (; *s; s++) if (*s >= 'A' && *s <= 'Z') *s += 32; }

static void las_detect_soc(int *qcom, int *mtk, char *board, size_t bn) {
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

/* Prepend our native-lib dir to the loader paths so QNN's HTP can reach the
 * Hexagon (libcdsprpc + DSP skel). See xr_zipdepth.c for the full rationale. */
static void las_fastrpc_env(char *libdir, size_t ln) {
    libdir[0] = 0;
    Dl_info info;
    if (dladdr((void *)las_fastrpc_env, &info) && info.dli_fname) {
        const char *slash = strrchr(info.dli_fname, '/');
        if (slash) snprintf(libdir, ln, "%.*s",
                            (int)(slash - info.dli_fname), info.dli_fname);
    }
    const char *dsp = Z.dsp_dir[0] ? Z.dsp_dir : libdir;
    { const char *e = getenv("LD_LIBRARY_PATH"); char v[1400];
      snprintf(v, sizeof v, "%s:%s%s%s", dsp, libdir,
               (e && e[0]) ? ":" : "", (e && e[0]) ? e : "");
      setenv("LD_LIBRARY_PATH", v, 1); }
    /* ADSP_LIBRARY_PATH is SEMICOLON-delimited (FastRPC convention), NOT colon
     * like LD_LIBRARY_PATH — a ':' makes the DSP treat the whole string as one
     * bogus path and it can't find libQnnHtpV68Skel.so. Matches zd_fastrpc_env. */
    { const char *e = getenv("ADSP_LIBRARY_PATH"); char v[1400];
      snprintf(v, sizeof v, "%s;%s%s%s", libdir, dsp,
               (e && e[0]) ? ";" : "", (e && e[0]) ? e : "");
      setenv("ADSP_LIBRARY_PATH", v, 1); }
    LOGI("LAS2: FastRPC libs dsp_dir=%s qnn_dir=%s (LD+ADSP set)", dsp, libdir);
}

static int las_open(const char *model_path, const char *ep,
                    const char **k, const char **v, int n, int strict,
                    const char *label) {
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(Z.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    ort_ok(Z.api->SetSessionLogSeverityLevel(opts, 2), "SetLogSeverity");
    ort_ok(Z.api->SetIntraOpNumThreads(opts, 2), "SetIntraOpNumThreads");
    ort_ok(Z.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL),
           "SetGraphOpt");
    if (ep) {
        OrtStatus *st = Z.api->SessionOptionsAppendExecutionProvider(
            opts, ep, k, v, n);
        if (st) {
            LOGE("LAS2: [%s] EP append failed: %s", label,
                 Z.api->GetErrorMessage(st));
            Z.api->ReleaseStatus(st);
            Z.api->ReleaseSessionOptions(opts);
            return 0;
        }
        if (strict)
            ort_ok(Z.api->AddSessionConfigEntry(
                       opts, "session.disable_cpu_ep_fallback", "1"),
                   "DisableCpuFallback");
    }
    OrtStatus *st = Z.api->CreateSession(Z.env, model_path, opts, &Z.session);
    Z.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("LAS2: [%s] open failed: %s", label, Z.api->GetErrorMessage(st));
        Z.api->ReleaseStatus(st);
        Z.session = NULL;
        return 0;
    }
    LOGI("LAS2: [%s] session opened", label);
    return 1;
}

static int las_try_init(const char *model_path) {
    void *dl = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!dl) {
        LOGI("LAS2: libonnxruntime.so not present (%s) — SGM depth", dlerror());
        return 0;
    }
    const OrtApiBase *(*get_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(dl, "OrtGetApiBase");
    if (!get_base) { LOGE("LAS2: OrtGetApiBase missing"); return 0; }
    Z.api = get_base()->GetApi(ORT_API_VERSION);
    if (!Z.api) { LOGE("LAS2: ORT api %d unavailable", ORT_API_VERSION); return 0; }

    if (!ort_ok(Z.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "las2", &Z.env),
                "CreateEnv"))
        return 0;

    int qcom = 0, mtk = 0;
    char board[PROP_VALUE_MAX] = {0};
    las_detect_soc(&qcom, &mtk, board, sizeof board);

    int ok = 0;
    if (qcom) {
        char libdir[512];
        las_fastrpc_env(libdir, sizeof libdir);
        char htp[560];
        if (libdir[0]) snprintf(htp, sizeof htp, "%s/libQnnHtp.so", libdir);
        else           snprintf(htp, sizeof htp, "libQnnHtp.so");
        /* ONE attempt only: the model is a single EPContext node (a whole AoT HTP
         * graph loaded via QnnContext_createFromBinary), so it is all-or-nothing on
         * the Hexagon — there are no boundary ops to split onto the CPU, and the
         * Adreno GPU backend cannot consume an HTP-compiled context binary (the old
         * +cpu-boundary / GPU fallbacks only ever produced confusing errors:
         * "libQnnGpu.so not found", "not compatible with any EP"). Strict (disable
         * CPU-EP fallback) so a QNN failure drops us cleanly to SGM rather than a
         * bogus/slow CPU placement. Failure now falls straight through to SGM. */
        const char *kf[] = { "backend_path", "offload_graph_io_quantization" };
        const char *vf[] = { htp, "0" };
        const char *kh[] = { "backend_path" };
        const char *vh[] = { htp };
        LOGI("LAS2: QNN init [board='%s'] htp=%s", board, htp);
        /* ONLINE-COMPILE test (ORT #26686 workaround for the HTP-v68 offline-
         * EPContext defect): the staged model is now the QDQ graph, not an offline
         * context, so the QNN EP compiles it ON-DEVICE (QnnGraph_create) instead of
         * QnnContext_createFromBinary. Strict full-HTP first, then a CPU boundary
         * (a few QDQ I/O nodes may not map to the HTP). */
        ok = las_open(model_path, "QNN", kf, vf, 2, 1, "QNN HTP full (INT8)")
          || las_open(model_path, "QNN", kh, vh, 1, 0, "QNN HTP +cpu-boundary (INT8)");
        if (!ok)
            LOGE("LAS2: QNN on-device compile failed -> SGM. Capture logcat "
                 "(onnxruntime/QnnHtp) for the QnnGraph_create / op-support reason.");
    } else if (mtk) {
        ok = las_open(model_path, "NeuroPilot", NULL, NULL, 0, 1, "MediaTek NeuroPilot");
    }
    if (!ok) return 0;

    if (!ort_ok(Z.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &Z.meminfo), "CreateCpuMemoryInfo"))
        return 0;

    OrtAllocator *alloc = NULL;
    ort_ok(Z.api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocator");
    char *name = NULL;
    for (int kk = 0; kk < 2; kk++) {
        if (!ort_ok(Z.api->SessionGetInputName(Z.session, kk, alloc, &name),
                    "GetInputName"))
            return 0;
        strncpy(Z.in_name[kk], name, sizeof Z.in_name[kk] - 1);
        alloc->Free(alloc, name);
    }
    if (!ort_ok(Z.api->SessionGetOutputName(Z.session, 0, alloc, &name),
                "GetOutputName"))
        return 0;
    strncpy(Z.out_name, name, sizeof Z.out_name - 1);
    alloc->Free(alloc, name);

    atomic_store_explicit(&Z.ready, 1, memory_order_release);
    LOGI("LAS2 ready: %s (%dx%d, in=%s,%s out=%s)", model_path,
         LAS_W, LAS_H, Z.in_name[0], Z.in_name[1], Z.out_name);
    return 1;
}

int xr_las2_init(const char *model_path, const char *dsp_dir) {
    if (atomic_load_explicit(&Z.ready, memory_order_acquire)) return 1;
    static atomic_int busy;
    int expected = 0;
    if (!atomic_compare_exchange_strong(&busy, &expected, 1))
        return 0;                       /* another caller is doing the work */
    if (dsp_dir) strncpy(Z.dsp_dir, dsp_dir, sizeof Z.dsp_dir - 1);
    int ok = las_try_init(model_path);
    atomic_store(&busy, 0);
    return ok;
}

int xr_las2_available(void) {
    return atomic_load_explicit(&Z.ready, memory_order_acquire);
}

/* bilinear sample of a w*h uint8 image at (fx,fy), clamped to the edge */
static float las_samp_u8(const uint8_t *g, int w, int h, float fx, float fy) {
    if (fx < 0) fx = 0; else if (fx > w - 1) fx = (float)(w - 1);
    if (fy < 0) fy = 0; else if (fy > h - 1) fy = (float)(h - 1);
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1 < w ? x0 + 1 : x0, y1 = y0 + 1 < h ? y0 + 1 : y0;
    float ax = fx - x0, ay = fy - y0;
    const uint8_t *r0 = g + (size_t)y0 * w, *r1 = g + (size_t)y1 * w;
    float a = r0[x0] * (1 - ax) + r0[x1] * ax;
    float b = r1[x0] * (1 - ax) + r1[x1] * ax;
    return a * (1 - ay) + b * ay;
}

int xr_las2_run(const uint8_t *left, const uint8_t *right, float f_hi, float base,
                float *depth_out, int out_w, int out_h) {
    if (!atomic_load_explicit(&Z.ready, memory_order_acquire)) return -1;
    if (!left || !right || !depth_out || out_w <= 0 || out_h <= 0) return -1;

    /* Downsample the 480x640 rectified source to the model's LAS_W x LAS_H, gray
     * replicated to 3 channels, raw [0,255]. */
    const int plane = LAS_H * LAS_W;
    const float rx = (float)LAS_SRC_W / LAS_W, ry = (float)LAS_SRC_H / LAS_H;
    for (int y = 0; y < LAS_H; y++) {
        float syf = (y + 0.5f) * ry - 0.5f;
        for (int x = 0; x < LAS_W; x++) {
            float sxf = (x + 0.5f) * rx - 0.5f;
            Z.in_l[y * LAS_W + x] = las_samp_u8(left,  LAS_SRC_W, LAS_SRC_H, sxf, syf);
            Z.in_r[y * LAS_W + x] = las_samp_u8(right, LAS_SRC_W, LAS_SRC_H, sxf, syf);
        }
    }
    memcpy(Z.in_l + plane, Z.in_l, sizeof(float) * plane);
    memcpy(Z.in_l + 2 * plane, Z.in_l, sizeof(float) * plane);
    memcpy(Z.in_r + plane, Z.in_r, sizeof(float) * plane);
    memcpy(Z.in_r + 2 * plane, Z.in_r, sizeof(float) * plane);

    const int64_t shape[4] = { 1, 3, LAS_H, LAS_W };
    OrtValue *inL = NULL, *inR = NULL;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.meminfo, Z.in_l, sizeof Z.in_l, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inL), "CreateL"))
        return -1;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.meminfo, Z.in_r, sizeof Z.in_r, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inR), "CreateR")) {
        Z.api->ReleaseValue(inL); return -1;
    }

    const char *ins[2] = { Z.in_name[0], Z.in_name[1] };
    const OrtValue *invals[2] = { inL, inR };
    const char *outs[1] = { Z.out_name };
    OrtValue *ov = NULL;
    OrtStatus *st = Z.api->Run(Z.session, NULL, ins,
                               (const OrtValue *const *)invals, 2, outs, 1, &ov);
    Z.api->ReleaseValue(inL);
    Z.api->ReleaseValue(inR);
    if (!ort_ok(st, "Run")) return -1;

    float *disp = NULL;
    int ok = ort_ok(Z.api->GetTensorMutableData(ov, (void **)&disp), "out data");
    if (ok && disp) {
        /* disparity is (1,1,LAS_H,LAS_W) at the MODEL resolution. f_hi is the focal
         * at the 480-wide rectified source, so scale it to the model width — f and
         * disp must share a resolution: f_las = f_hi * LAS_W / LAS_SRC_W. Depth is
         * then subsampled straight to the out_w*out_h publish grid. */
        const float f_las = f_hi * (float)LAS_W / (float)LAS_SRC_W;
        const float sx = (float)LAS_W / out_w, sy = (float)LAS_H / out_h;
        for (int oy = 0; oy < out_h; oy++) {
            int yy = (int)((oy + 0.5f) * sy); if (yy >= LAS_H) yy = LAS_H - 1;
            for (int ox = 0; ox < out_w; ox++) {
                int xx = (int)((ox + 0.5f) * sx); if (xx >= LAS_W) xx = LAS_W - 1;
                float d = disp[yy * LAS_W + xx]; if (d < 0.0f) d = -d;
                depth_out[oy * out_w + ox] =
                    (d > 0.3f) ? (f_las * base / d) : 0.0f;
            }
        }
    }
    Z.api->ReleaseValue(ov);
    return ok ? 0 : -1;
}
