/* xr_xfeat.c — see xr_xfeat.h. Two inference paths behind one contract:
 *
 *   NPU  — XFeat's dense backbone as a QAIRT-native A8W8 HTP context
 *          (EPContext .onnx, u8 quantized IO; 3.6 ms on the SD888 HTP,
 *          score-map corr .984 / desc cosine .998 vs fp32). The dynamic
 *          tail (NMS -> reliability-weighted top-K -> bilinear descriptor
 *          sampling -> L2 norm) is reproduced here in C, mirroring the
 *          official XFeat post-processing the full ONNX graph performs.
 *   CPU  — the original full xfeat.onnx on ORT's CPU EP (fallback when the
 *          QNN EP / context is unavailable: non-qnn builds, non-QC SoCs).
 *
 * Both paths fill the same (uv, int8 desc[64]) keyframe contract, so the
 * map layer cannot tell them apart. */
#include "xr_xfeat.h"

#include <dlfcn.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"
#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define IMG_W XR_OW      /* 480 */
#define IMG_H XR_OH      /* 640 */
#define GRID_W (IMG_W / 8)   /* 60: dense desc/reliability grid */
#define GRID_H (IMG_H / 8)   /* 80 */

/* u8 quantization of the A8W8 context's IO (ctx_xfeat_a8 meta.json). The
 * input scale is 0.998046875 ~= 1.0 with offset 0, so the raw 8-bit gray
 * frame IS the quantized tensor (worst-case 0.5 gray-level bias at 255 —
 * far below the quantizer's own noise). Descriptor dequant is (q-127)*s;
 * s cancels in the L2 norm so only the offset is applied. */
#define XFN_SCORE_SCALE 0.00390625f      /* offset 0 */
#define XFN_REL_SCALE   0.003351456253f  /* offset -1 */
#define XFN_SCORE_QTHR  13               /* 0.05 (official threshold) / scale */
#define XFN_BORDER      4                /* skip rectified frame edges */

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtMemoryInfo *meminfo;
    int ort_up;              /* api/env/meminfo initialized */

    /* CPU path: full xfeat.onnx (detect+NMS+topK+sampling in-graph) */
    OrtSession *session;
    char in_name[64];
    char out_names[3][64];   /* keypoints, descriptors, scores */
    float input[3 * IMG_H * IMG_W];

    /* NPU path: dense EPContext (score/desc/reliability maps, u8 IO) */
    OrtSession *npu_session;
    OrtRunOptions *run_opts; /* per-run DSP clock vote (see LAS2) */
    char npu_in[64];
    char npu_out[3][64];     /* names by session index */
    int role[3];             /* session output index -> 0 score, 1 desc, 2 rel */
    char npu_path[256];
    atomic_int npu_ready;

    atomic_int ready;        /* release-published once EITHER path is usable;
                              * acquire-loaded by availability/extraction (UI +
                              * map threads) so they see the whole session */
} X;

static int ort_ok(OrtStatus *st, const char *what) {
    if (!st) return 1;
    LOGE("XFeat/ORT %s: %s", what, X.api->GetErrorMessage(st));
    X.api->ReleaseStatus(st);
    return 0;
}

/* ---------------- shared ORT bring-up ---------------- */

static int xfeat_ensure_ort(void) {
    if (X.ort_up) return 1;
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
    if (!ort_ok(X.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &X.meminfo), "CreateCpuMemoryInfo"))
        return 0;
    X.ort_up = 1;
    return 1;
}

/* ---------------- NPU path ---------------- */

static void lc(char *s) { for (; *s; s++) if (*s >= 'A' && *s <= 'Z') *s += 32; }

static int xfeat_is_qcom(void) {
    char soc[PROP_VALUE_MAX] = {0}, b[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.manufacturer", soc);
    __system_property_get("ro.board.platform", b);
    lc(soc); lc(b);
    return strstr(soc, "qti") || strstr(soc, "qualcomm") ||
           strstr(b, "kona") || strstr(b, "lahaina") ||
           strstr(b, "taro") || strstr(b, "kalama") ||
           strstr(b, "pineapple") || strstr(b, "sm8") ? 1 : 0;
}

/* Prepend our native-lib dir to the loader paths so QNN's HTP can reach the
 * Hexagon (libcdsprpc + DSP skel). ADSP_LIBRARY_PATH is SEMICOLON-delimited
 * (FastRPC convention). Mirrors las_fastrpc_env; both modules may run this —
 * prepending the same dir twice is harmless. */
static void xfeat_fastrpc_env(char *libdir, size_t ln) {
    libdir[0] = 0;
    Dl_info info;
    if (dladdr((void *)xfeat_fastrpc_env, &info) && info.dli_fname) {
        const char *slash = strrchr(info.dli_fname, '/');
        if (slash) snprintf(libdir, ln, "%.*s",
                            (int)(slash - info.dli_fname), info.dli_fname);
    }
    if (!libdir[0]) return;
    { const char *e = getenv("LD_LIBRARY_PATH"); char v[1400];
      snprintf(v, sizeof v, "%s%s%s", libdir,
               (e && e[0]) ? ":" : "", (e && e[0]) ? e : "");
      setenv("LD_LIBRARY_PATH", v, 1); }
    { const char *e = getenv("ADSP_LIBRARY_PATH"); char v[1400];
      snprintf(v, sizeof v, "%s%s%s", libdir,
               (e && e[0]) ? ";" : "", (e && e[0]) ? e : "");
      setenv("ADSP_LIBRARY_PATH", v, 1); }
}

static int xfeat_try_init_npu(void) {
    if (!X.npu_path[0] || !xfeat_is_qcom()) return 0;

    char libdir[512];
    xfeat_fastrpc_env(libdir, sizeof libdir);
    char htp[560];
    if (libdir[0]) snprintf(htp, sizeof htp, "%s/libQnnHtp.so", libdir);
    else           snprintf(htp, sizeof htp, "libQnnHtp.so");

    OrtSessionOptions *opts = NULL;
    if (!ort_ok(X.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    X.api->SetSessionLogSeverityLevel(opts, 2);
    X.api->SetIntraOpNumThreads(opts, 1);
    X.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    /* ONE strict attempt: single EPContext node, QAIRT-native A8W8, compiled
     * --htp_socs sm8350 (the only pipeline the integer-only v68 skel
     * deserializes — memory: las2-deployment). All-or-nothing on the Hexagon;
     * failure drops cleanly to the CPU model. */
    const char *kf[] = { "backend_path", "offload_graph_io_quantization" };
    const char *vf[] = { htp, "0" };
    OrtStatus *st = X.api->SessionOptionsAppendExecutionProvider(
        opts, "QNN", kf, vf, 2);
    if (st) {
        LOGI("XFeat NPU: QNN EP unavailable (%s) — CPU model",
             X.api->GetErrorMessage(st));
        X.api->ReleaseStatus(st);
        X.api->ReleaseSessionOptions(opts);
        return 0;
    }
    ort_ok(X.api->AddSessionConfigEntry(
               opts, "session.disable_cpu_ep_fallback", "1"),
           "DisableCpuFallback");
    st = X.api->CreateSession(X.env, X.npu_path, opts, &X.npu_session);
    X.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("XFeat NPU: HTP did not take the context (%s) — CPU model",
             X.api->GetErrorMessage(st));
        X.api->ReleaseStatus(st);
        X.npu_session = NULL;
        return 0;
    }

    /* Per-run DSP clock vote: pin for the inference, release after (the map
     * thread runs at keyframe rate — the DSP idles cold in between). */
    if (ort_ok(X.api->CreateRunOptions(&X.run_opts), "CreateRunOptions")) {
        ort_ok(X.api->AddRunConfigEntry(X.run_opts, "qnn.htp_perf_mode",
                                        "sustained_high_performance"),
               "RunCfg perf_mode");
        ort_ok(X.api->AddRunConfigEntry(X.run_opts, "qnn.htp_perf_mode_post_run",
                                        "default"),
               "RunCfg perf_post");
    }

    /* Discover IO names and map the 3 outputs to roles BY SHAPE — never by
     * position or name convention: [1,1,640,480] score, [1,64,80,60] desc,
     * [1,1,80,60] reliability. */
    OrtAllocator *alloc = NULL;
    ort_ok(X.api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocator");
    char *name = NULL;
    if (!ort_ok(X.api->SessionGetInputName(X.npu_session, 0, alloc, &name),
                "npu GetInputName"))
        return 0;
    strncpy(X.npu_in, name, sizeof X.npu_in - 1);
    alloc->Free(alloc, name);
    int mapped = 0;
    for (int i = 0; i < 3; i++) {
        if (!ort_ok(X.api->SessionGetOutputName(X.npu_session, (size_t)i, alloc,
                                                &name), "npu GetOutputName"))
            return 0;
        strncpy(X.npu_out[i], name, sizeof X.npu_out[i] - 1);
        alloc->Free(alloc, name);
        OrtTypeInfo *ti = NULL;
        const OrtTensorTypeAndShapeInfo *tsi = NULL;
        int64_t d[4] = { 0 };
        size_t nd = 0;
        if (!ort_ok(X.api->SessionGetOutputTypeInfo(X.npu_session, (size_t)i,
                                                    &ti), "npu OutputTypeInfo"))
            return 0;
        X.api->CastTypeInfoToTensorInfo(ti, &tsi);
        if (tsi) {
            X.api->GetDimensionsCount(tsi, &nd);
            X.api->GetDimensions(tsi, d, nd < 4 ? nd : 4);
        }
        X.api->ReleaseTypeInfo(ti);
        if (d[1] == 64)                   { X.role[i] = 1; mapped |= 2; }
        else if (d[2] == IMG_H)           { X.role[i] = 0; mapped |= 1; }
        else if (d[2] == GRID_H)          { X.role[i] = 2; mapped |= 4; }
    }
    if (mapped != 7) {
        LOGE("XFeat NPU: unexpected output shapes (mask %d) — CPU model", mapped);
        X.api->ReleaseSession(X.npu_session);
        X.npu_session = NULL;
        return 0;
    }
    atomic_store_explicit(&X.npu_ready, 1, memory_order_release);
    LOGI("XFeat NPU ready: %s (HTP dense A8W8, CPU tail)", X.npu_path);
    return 1;
}

/* CPU tail over the dense maps: 5x5 NMS on the u8 score plane,
 * reliability-weighted ranking, top-`max`, bilinear 64-D descriptor
 * sampling + L2 norm + int8 quantization. */
typedef struct { float sc; uint16_t x, y; } xfn_peak;

static int xfeat_npu_extract(const uint8_t *img, float (*uv)[2],
                             int8_t (*desc)[64], int max) {
    const int64_t shape[4] = { 1, 1, IMG_H, IMG_W };
    OrtValue *in = NULL;
    /* the raw gray frame is already the quantized u8 tensor (scale ~1.0) */
    if (!ort_ok(X.api->CreateTensorWithDataAsOrtValue(
                    X.meminfo, (void *)img, (size_t)(IMG_H * IMG_W), shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &in),
                "npu CreateTensor"))
        return -1;

    const char *ins[1] = { X.npu_in };
    const char *outs[3] = { X.npu_out[0], X.npu_out[1], X.npu_out[2] };
    OrtValue *ov[3] = { NULL, NULL, NULL };
    OrtStatus *st = X.api->Run(X.npu_session, X.run_opts, ins,
                               (const OrtValue *const *)&in, 1, outs, 3, ov);
    X.api->ReleaseValue(in);
    if (!ort_ok(st, "npu Run")) return -1;

    const uint8_t *score = NULL, *dns = NULL, *rel = NULL;
    for (int i = 0; i < 3; i++) {
        void *p = NULL;
        if (!ort_ok(X.api->GetTensorMutableData(ov[i], &p), "npu out data")) {
            for (int k = 0; k < 3; k++) if (ov[k]) X.api->ReleaseValue(ov[k]);
            return -1;
        }
        if (X.role[i] == 0) score = p;
        else if (X.role[i] == 1) dns = p;
        else rel = p;
    }

    /* separable 5x5 max filter, row pass into a static staging plane */
    static uint8_t rowmax[IMG_H * IMG_W];
    for (int y = 0; y < IMG_H; y++) {
        const uint8_t *s = score + y * IMG_W;
        uint8_t *r = rowmax + y * IMG_W;
        for (int x = 0; x < IMG_W; x++) {
            int x0 = x - 2 < 0 ? 0 : x - 2, x1 = x + 2 >= IMG_W ? IMG_W - 1 : x + 2;
            uint8_t m = 0;
            for (int k = x0; k <= x1; k++) if (s[k] > m) m = s[k];
            r[x] = m;
        }
    }

    /* peak scan: score == 5x5 window max && above threshold; ranking value is
     * heatmap * bilinear(reliability) exactly as official detectAndCompute.
     * Kept in a bounded min-heap of the requested size. */
    if (max > XR_XFEAT_MAX_KP) max = XR_XFEAT_MAX_KP;
    static xfn_peak heap[XR_XFEAT_MAX_KP];
    int hn = 0;
    for (int y = XFN_BORDER; y < IMG_H - XFN_BORDER; y++) {
        const uint8_t *s = score + y * IMG_W;
        for (int x = XFN_BORDER; x < IMG_W - XFN_BORDER; x++) {
            uint8_t v = s[x];
            if (v < XFN_SCORE_QTHR) continue;
            uint8_t m = 0;
            for (int dy = -2; dy <= 2; dy++) {
                uint8_t rm = rowmax[(y + dy) * IMG_W + x];
                if (rm > m) m = rm;
            }
            if (v != m) continue;
            /* bilinear reliability at the 1/8 grid (align_corners=false) */
            float gx = (x + 0.5f) * 0.125f - 0.5f;
            float gy = (y + 0.5f) * 0.125f - 0.5f;
            if (gx < 0) gx = 0; if (gx > GRID_W - 1.001f) gx = GRID_W - 1.001f;
            if (gy < 0) gy = 0; if (gy > GRID_H - 1.001f) gy = GRID_H - 1.001f;
            int ix = (int)gx, iy = (int)gy;
            float fx = gx - ix, fy = gy - iy;
            const uint8_t *r0 = rel + iy * GRID_W + ix;
            float rl = (1 - fy) * ((1 - fx) * r0[0] + fx * r0[1]) +
                       fy * ((1 - fx) * r0[GRID_W] + fx * r0[GRID_W + 1]);
            float sc = (v * XFN_SCORE_SCALE) * ((rl - 1.0f) * XFN_REL_SCALE);
            if (hn < max) {                     /* push */
                int i = hn++;
                heap[i] = (xfn_peak){ sc, (uint16_t)x, (uint16_t)y };
                while (i && heap[(i - 1) / 2].sc > heap[i].sc) {
                    xfn_peak t = heap[i]; heap[i] = heap[(i - 1) / 2];
                    heap[(i - 1) / 2] = t; i = (i - 1) / 2;
                }
            } else if (sc > heap[0].sc) {       /* replace root, sift down */
                heap[0] = (xfn_peak){ sc, (uint16_t)x, (uint16_t)y };
                int i = 0;
                for (;;) {
                    int l = 2 * i + 1, r = l + 1, sm = i;
                    if (l < hn && heap[l].sc < heap[sm].sc) sm = l;
                    if (r < hn && heap[r].sc < heap[sm].sc) sm = r;
                    if (sm == i) break;
                    xfn_peak t = heap[i]; heap[i] = heap[sm]; heap[sm] = t;
                    i = sm;
                }
            }
        }
    }

    /* heap -> descending order (contract: scores sorted descending) */
    for (int end = hn - 1; end > 0; end--) {   /* heapsort: min-heap ascends */
        xfn_peak t = heap[0]; heap[0] = heap[end]; heap[end] = t;
        int i = 0;
        for (;;) {
            int l = 2 * i + 1, r = l + 1, sm = i;
            if (l < end && heap[l].sc < heap[sm].sc) sm = l;
            if (r < end && heap[r].sc < heap[sm].sc) sm = r;
            if (sm == i) break;
            xfn_peak tt = heap[i]; heap[i] = heap[sm]; heap[sm] = tt;
            i = sm;
        }
    }

    /* bilinear 64-D descriptor at each keypoint; the u8 dequant scale cancels
     * in the L2 norm, only the -127 offset is applied */
    const int plane = GRID_H * GRID_W;
    for (int i = 0; i < hn; i++) {
        int x = heap[i].x, y = heap[i].y;
        uv[i][0] = (float)x;
        uv[i][1] = (float)y;
        float gx = (x + 0.5f) * 0.125f - 0.5f;
        float gy = (y + 0.5f) * 0.125f - 0.5f;
        if (gx < 0) gx = 0; if (gx > GRID_W - 1.001f) gx = GRID_W - 1.001f;
        if (gy < 0) gy = 0; if (gy > GRID_H - 1.001f) gy = GRID_H - 1.001f;
        int ix = (int)gx, iy = (int)gy;
        float fx = gx - ix, fy = gy - iy;
        float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx;
        float w10 = fy * (1 - fx), w11 = fy * fx;
        const uint8_t *base = dns + iy * GRID_W + ix;
        float v[64];
        float nrm = 0;
        for (int c = 0; c < 64; c++) {
            const uint8_t *p = base + c * plane;
            float d = w00 * (p[0] - 127) + w01 * (p[1] - 127) +
                      w10 * (p[GRID_W] - 127) + w11 * (p[GRID_W + 1] - 127);
            v[c] = d;
            nrm += d * d;
        }
        nrm = nrm > 1e-12f ? 127.0f / sqrtf(nrm) : 0.0f;
        for (int c = 0; c < 64; c++) {
            float q = v[c] * nrm;
            desc[i][c] = (int8_t)(q > 127.f ? 127 : q < -127.f ? -127
                                  : lroundf(q));
        }
    }

    for (int i = 0; i < 3; i++)
        if (ov[i]) X.api->ReleaseValue(ov[i]);
    return hn;
}

/* ---------------- CPU path (original full graph) ---------------- */

static int xfeat_try_init_cpu(const char *model_path) {
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
    LOGI("XFeat CPU ready: %s (in=%s outs=%s,%s,%s)", model_path, X.in_name,
         X.out_names[0], X.out_names[1], X.out_names[2]);
    return 1;
}

static int xfeat_try_init(const char *model_path) {
    if (!xfeat_ensure_ort()) return 0;
    /* NPU first; when the HTP takes the context, skip the CPU session
     * entirely (saves the ~40 MB CPU arena + graph optimization time). */
    int npu = xfeat_try_init_npu();
    int cpu = npu ? 0 : xfeat_try_init_cpu(model_path);
    if (!npu && !cpu) return 0;
    atomic_store_explicit(&X.ready, 1, memory_order_release);
    return 1;
}

/* Thread-safe: one caller runs the slow init; others fall back to ORB
 * until X.ready flips. A failed attempt is retryable (not a permanent
 * one-shot — the model may simply not have been staged yet). */
int xr_xfeat_init(const char *model_path) {
    if (atomic_load_explicit(&X.ready, memory_order_acquire)) return 1;
    static atomic_int busy;
    if (atomic_exchange(&busy, 1)) return 0;
    int ok = xfeat_try_init(model_path);
    if (!ok) atomic_store(&busy, 0);
    return ok;
}

int xr_xfeat_available(void) {
    return atomic_load_explicit(&X.ready, memory_order_acquire);
}

void xr_xfeat_set_npu_model(const char *epctx_path) {
    if (!epctx_path || !epctx_path[0]) return;
    strncpy(X.npu_path, epctx_path, sizeof X.npu_path - 1);
}

int xr_xfeat_npu_active(void) {
    return atomic_load_explicit(&X.npu_ready, memory_order_acquire);
}

int xr_xfeat_extract(const uint8_t *img, float (*uv)[2],
                     int8_t (*desc)[64], int max) {
    if (!atomic_load_explicit(&X.ready, memory_order_acquire)) return -1;

    if (atomic_load_explicit(&X.npu_ready, memory_order_acquire)) {
        int n = xfeat_npu_extract(img, uv, desc, max);
        if (n >= 0) return n;
        if (!X.session) return -1;   /* no CPU session to fall back on */
        LOGE("XFeat NPU extract failed — CPU fallback this frame");
    }
    if (!X.session) return -1;

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
