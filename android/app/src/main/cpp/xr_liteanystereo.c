/* xr_liteanystereo.c — see xr_liteanystereo.h. Two model slots (fast 192x256 /
 * MID 288x384), one shared ORT env + QNN backend. Only QAIRT-native-quantized
 * AoT contexts load on the integer-only v68 skel (memory: las2-deployment). */
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

/* largest slot (MID 288x384) sizes the shared input staging buffers */
#define LAS_MAX_PX (288 * 384)

/* see xr_liteanystereo.h — serializes all HTP session creation + Run() */
pthread_mutex_t XR_NPU_GATE = PTHREAD_MUTEX_INITIALIZER;

/* Per-slot model state. Inputs are NCHW (1,1,h,w), 1-CHANNEL (first conv +
 * ImageNet norm folded in — host-verified exact vs the 3-ch original), raw
 * [0,255] quantized here to the graph's UFIXED_POINT_16 encoding. */
typedef struct {
    OrtSession *session;
    char in_name[2][64];              /* left, right */
    char out_name[64];                /* disparity */
    int w, h;
    float in_scale, out_scale;        /* u16 quant scales from the ctx meta */
    atomic_int ready;
} las_model;

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtRunOptions *run_opts;          /* per-run DSP clock vote (see init) */
    OrtMemoryInfo *meminfo;
    char dsp_dir[256];
    int ort_up;                       /* shared env/backend initialized */
    las_model m[XR_LAS2_SLOTS];
    uint16_t in_l[LAS_MAX_PX];        /* u16-quantized inputs, run-thread only */
    uint16_t in_r[LAS_MAX_PX];
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

static int las_open(las_model *m, const char *model_path, const char *ep,
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
    pthread_mutex_lock(&XR_NPU_GATE);         /* context deserialize hits the HTP */
    OrtStatus *st = Z.api->CreateSession(Z.env, model_path, opts, &m->session);
    pthread_mutex_unlock(&XR_NPU_GATE);
    Z.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("LAS2: [%s] open failed: %s", label, Z.api->GetErrorMessage(st));
        Z.api->ReleaseStatus(st);
        m->session = NULL;
        return 0;
    }
    LOGI("LAS2: [%s] session opened", label);
    return 1;
}

/* shared ORT env + QNN backend paths, initialized once */
static int las_ensure_ort(void) {
    if (Z.ort_up) return 1;
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
    if (!ort_ok(Z.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &Z.meminfo), "CreateCpuMemoryInfo"))
        return 0;
    /* Per-run DSP clock vote: pin clocks for the inference, release to
     * default after — the DSP idles cold between paced runs and during
     * thermal suspend instead of holding a session-lifetime high vote. */
    if (ort_ok(Z.api->CreateRunOptions(&Z.run_opts), "CreateRunOptions")) {
        ort_ok(Z.api->AddRunConfigEntry(Z.run_opts, "qnn.htp_perf_mode",
                                        "sustained_high_performance"),
               "RunCfg perf_mode");
        ort_ok(Z.api->AddRunConfigEntry(Z.run_opts, "qnn.htp_perf_mode_post_run",
                                        "default"),
               "RunCfg perf_post");
    }
    Z.ort_up = 1;
    return 1;
}

static int las_try_init(int slot, const char *model_path) {
    las_model *m = &Z.m[slot];
    if (!las_ensure_ort()) return 0;

    int qcom = 0, mtk = 0;
    char board[PROP_VALUE_MAX] = {0};
    las_detect_soc(&qcom, &mtk, board, sizeof board);

    int ok = 0;
    char label[64];
    snprintf(label, sizeof label, "QNN HTP full (A16W8 native %dx%d)",
             m->w, m->h);
    if (qcom) {
        char libdir[512];
        las_fastrpc_env(libdir, sizeof libdir);
        char htp[560];
        if (libdir[0]) snprintf(htp, sizeof htp, "%s/libQnnHtp.so", libdir);
        else           snprintf(htp, sizeof htp, "libQnnHtp.so");
        /* ONE strict attempt: the model is a single EPContext node carrying a
         * QAIRT-NATIVE-quantized (A16W8) HTP context compiled --htp_socs
         * sm8350 — the only pipeline whose binaries the INTEGER-ONLY v68 skel
         * deserializes (x86 prepare of any float segment emits fp16 op
         * variants v68 lacks -> err 1002). All-or-nothing on the Hexagon: no
         * boundary ops to split, and the GPU backend can't consume an HTP
         * context. Strict, so failure drops cleanly to SGM. */
        const char *kf[] = { "backend_path", "offload_graph_io_quantization" };
        const char *vf[] = { htp, "0" };
        LOGI("LAS2: QNN init [board='%s'] slot=%d htp=%s", board, slot, htp);
        ok = las_open(m, model_path, "QNN", kf, vf, 2, 1, label);
        if (!ok)
            LOGE("LAS2: QNN HTP did not take the context binary -> SGM. Enable "
                 "DSP FARF (<proc>.farf) + logcat QnnDsp for the deserializer op.");
    } else if (mtk) {
        ok = las_open(m, model_path, "NeuroPilot", NULL, NULL, 0, 1,
                      "MediaTek NeuroPilot");
    }
    if (!ok) return 0;

    OrtAllocator *alloc = NULL;
    ort_ok(Z.api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocator");
    char *name = NULL;
    for (int kk = 0; kk < 2; kk++) {
        if (!ort_ok(Z.api->SessionGetInputName(m->session, kk, alloc, &name),
                    "GetInputName"))
            return 0;
        strncpy(m->in_name[kk], name, sizeof m->in_name[kk] - 1);
        alloc->Free(alloc, name);
    }
    if (!ort_ok(Z.api->SessionGetOutputName(m->session, 0, alloc, &name),
                "GetOutputName"))
        return 0;
    strncpy(m->out_name, name, sizeof m->out_name - 1);
    alloc->Free(alloc, name);

    atomic_store_explicit(&m->ready, 1, memory_order_release);
    LOGI("LAS2 ready: slot %d %s (%dx%d, in=%s,%s out=%s)", slot, model_path,
         m->w, m->h, m->in_name[0], m->in_name[1], m->out_name);
    return 1;
}

int xr_las2_init(int slot, const char *model_path, const char *dsp_dir,
                 int w, int h, float in_scale, float out_scale) {
    if (slot < 0 || slot >= XR_LAS2_SLOTS || w * h > LAS_MAX_PX) return 0;
    las_model *m = &Z.m[slot];
    if (atomic_load_explicit(&m->ready, memory_order_acquire)) return 1;
    static atomic_int busy;
    int expected = 0;
    if (!atomic_compare_exchange_strong(&busy, &expected, 1))
        return 0;                       /* another caller is doing the work */
    if (dsp_dir) strncpy(Z.dsp_dir, dsp_dir, sizeof Z.dsp_dir - 1);
    m->w = w; m->h = h;
    m->in_scale = in_scale; m->out_scale = out_scale;
    int ok = las_try_init(slot, model_path);
    atomic_store(&busy, 0);
    return ok;
}

int xr_las2_available(int slot) {
    if (slot < 0 || slot >= XR_LAS2_SLOTS) return 0;
    return atomic_load_explicit(&Z.m[slot].ready, memory_order_acquire);
}

int xr_las2_run(int slot, const uint8_t *left, const uint8_t *right,
                float f_model, float base, float *depth_out,
                int out_w, int out_h) {
    if (slot < 0 || slot >= XR_LAS2_SLOTS) return -1;
    las_model *m = &Z.m[slot];
    if (!atomic_load_explicit(&m->ready, memory_order_acquire)) return -1;
    if (!left || !right || !depth_out || out_w <= 0 || out_h <= 0) return -1;
    const int w = m->w, h = m->h, plane = w * h;

    /* quantize the model-res rectified gray pair to the graph's u16 input
     * encoding (1 channel — the first conv is folded, no replication) */
    const float qs = 1.0f / m->in_scale;
    for (int i = 0; i < plane; i++) {
        float vl = left[i] * qs, vr = right[i] * qs;
        Z.in_l[i] = vl >= 65535.f ? 65535 : (uint16_t)(vl + 0.5f);
        Z.in_r[i] = vr >= 65535.f ? 65535 : (uint16_t)(vr + 0.5f);
    }

    const int64_t shape[4] = { 1, 1, h, w };
    const size_t nbytes = sizeof(uint16_t) * (size_t)plane;
    OrtValue *inL = NULL, *inR = NULL;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.meminfo, Z.in_l, nbytes, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, &inL), "CreateL"))
        return -1;
    if (!ort_ok(Z.api->CreateTensorWithDataAsOrtValue(
                    Z.meminfo, Z.in_r, nbytes, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, &inR), "CreateR")) {
        Z.api->ReleaseValue(inL); return -1;
    }

    const char *ins[2] = { m->in_name[0], m->in_name[1] };
    const OrtValue *invals[2] = { inL, inR };
    const char *outs[1] = { m->out_name };
    OrtValue *ov = NULL;
    pthread_mutex_lock(&XR_NPU_GATE);         /* exclusive HTP (see header) */
    OrtStatus *st = Z.api->Run(m->session, Z.run_opts, ins,
                               (const OrtValue *const *)invals, 2, outs, 1, &ov);
    pthread_mutex_unlock(&XR_NPU_GATE);
    Z.api->ReleaseValue(inL);
    Z.api->ReleaseValue(inR);
    if (!ort_ok(st, "Run")) return -1;

    uint16_t *disp = NULL;
    int ok = ort_ok(Z.api->GetTensorMutableData(ov, (void **)&disp), "out data");
    if (ok && disp) {
        /* disparity is (1,1,h,w) u16-quantized at the MODEL resolution:
         * d = u16 * out_scale, depth = f_model*base/d, subsampled straight to
         * the out_w*out_h publish grid (depth is resolution-invariant). */
        const float os = m->out_scale;
        const float sx = (float)w / out_w, sy = (float)h / out_h;
        for (int oy = 0; oy < out_h; oy++) {
            int yy = (int)((oy + 0.5f) * sy); if (yy >= h) yy = h - 1;
            for (int ox = 0; ox < out_w; ox++) {
                int xx = (int)((ox + 0.5f) * sx); if (xx >= w) xx = w - 1;
                float d = disp[yy * w + xx] * os;
                depth_out[oy * out_w + ox] =
                    (d > 0.3f) ? (f_model * base / d) : 0.0f;
            }
        }
    }
    Z.api->ReleaseValue(ov);
    return ok ? 0 : -1;
}
