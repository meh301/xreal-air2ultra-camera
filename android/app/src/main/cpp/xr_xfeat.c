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
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>

#include <android/log.h>

#include <time.h>

#include "ort/onnxruntime_c_api.h"
#include "xr_liteanystereo.h"   /* XR_NPU_GATE — process-wide HTP serializer */
#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define IMG_W XR_OW      /* 480 */
#define IMG_H XR_OH      /* 640 */
#define GRID_W (IMG_W / 8)   /* 60: dense desc/reliability grid */
#define GRID_H (IMG_H / 8)   /* 80 */
/* The dense backbone downsamples by 32 with floor, so a non-multiple-of-32
 * input (4Seasons 800x400 -> 48 rows, not 50) shrinks the output grid and
 * the shape check rejects EVERY frame. Zero-pad the inference input up to
 * the next multiple of 32 and run all dense-grid math on the padded dims;
 * when the frame is already clean (every indoor set, the device) the padded
 * dims collapse to the plain ones and nothing changes. */
#define IMG_PW (((IMG_W + 31) / 32) * 32)
#define IMG_PH (((IMG_H + 31) / 32) * 32)
#define PGRID_W (IMG_PW / 8)
#define PGRID_H (IMG_PH / 8)

/* u8 quantization of the A8W8 context's IO (ctx_xfeat_a8 meta.json). The
 * input scale is 0.998046875 ~= 1.0 with offset 0, so the raw 8-bit gray
 * frame IS the quantized tensor (worst-case 0.5 gray-level bias at 255 —
 * far below the quantizer's own noise). Descriptor dequant is (q-127)*s;
 * s cancels in the L2 norm so only the offset is applied. */
#define XFN_SCORE_SCALE 0.00390625f      /* offset 0 */
#define XFN_REL_SCALE   0.003351456253f  /* offset -1 */
#define XFN_SCORE_QTHR  13               /* 0.05 (official threshold) / scale */
#define XFN_SCORE_THR   0.05f            /* same gate, float-IO contexts */
#define XFN_BORDER      4                /* skip rectified frame edges */

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtMemoryInfo *meminfo;
    int ort_up;              /* api/env/meminfo initialized */

    /* CPU path: full xfeat.onnx (detect+NMS+topK+sampling in-graph), OR the
     * DENSE export (feats/scores/reliability maps, float IO) with the same
     * C tail the NPU path uses — the dense variant additionally enables
     * xr_xfeat_sample (descriptors at arbitrary uvs: landmark anchoring). */
    OrtSession *session;
    char in_name[64];
    char out_names[3][64];   /* sparse: keypoints, descriptors, scores;
                              * dense: by role 0 scores, 1 feats, 2 rel */
    int cpu_dense;           /* CPU session is the dense export */
    int dense_in_ch;         /* input channels the dense graph traced (1/3) */
    float input[3 * IMG_PH * IMG_PW];

    /* last extract's dense descriptor map, for xr_xfeat_sample (map thread
     * only). kind: 0 none, 1 float (CPU dense, PADDED grid), 2 u8 (NPU
     * dense), 3 float (NPU dense, float-IO context, plain grid). */
    int dense_kind;
    float dense_f[64 * PGRID_H * PGRID_W];
    uint8_t dense_u8[64 * GRID_H * GRID_W];

    /* NPU path: dense EPContext (score/desc/reliability maps). The graph IO
     * encoding is DISCOVERED from the session, never assumed: the shipping
     * A8W8 context is u8 in / u8 out, while a v81 native-fp16 context carries
     * FLOAT16 IO (and a plain float export FLOAT) — see xfeat_npu_dtypes. */
    OrtSession *npu_session;
    OrtRunOptions *run_opts; /* per-run DSP clock vote (see LAS2) */
    char npu_in[64];
    char npu_out[3][64];     /* names by session index */
    int role[3];             /* session output index -> 0 score, 1 desc, 2 rel */
    ONNXTensorElementDataType npu_in_dt;
    ONNXTensorElementDataType npu_out_dt[3];  /* by session output index */
    int npu_out_float;       /* outputs are FLOAT/FLOAT16 -> float tail */
    __fp16 npu_in_half[IMG_H * IMG_W];        /* fp16-IO staging */
    char npu_path[256];
    atomic_int npu_ready;
    atomic_int npu_vote;     /* run_opts carries the v81 burst power vote */

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

/* Serializes ALL extraction/sampling state (X.input, the static NMS/heap
 * scratch planes, the dense caches): the map thread AND the XR_SEED
 * feeding thread both call xr_xfeat_extract — unlocked they tear each
 * other's zero-copy inference inputs and the dense cache (review finding:
 * anchored descriptors sampled from the WRONG frame's map). */
static pthread_mutex_t XF_MU = PTHREAD_MUTEX_INITIALIZER;

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

/* Hexagon v81 (SM8850 / 8 Elite Gen 5, board "canoe"). Its unsigned
 * protection domain ACCEPTS HAP_Power votes; the v68 (SM8350) rejects them
 * and a vote there fails the whole Run — so the clock vote below is selected
 * per-SoC, never applied blind. */
static int xfeat_is_v81(void) {
    char soc[PROP_VALUE_MAX] = {0}, b[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.model", soc);
    __system_property_get("ro.board.platform", b);
    lc(soc); lc(b);
    return strstr(b, "canoe") || strstr(soc, "sm8850") ? 1 : 0;
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

/* Abandon a session the HTP accepted but whose contract we cannot honour
 * (unexpected shapes or IO dtype): tear it down completely so init falls
 * through to the CPU model instead of binding memory we misread. */
static int xfeat_npu_reject(const char *why) {
    LOGE("XFeat NPU: %s — CPU model", why);
    if (X.run_opts) { X.api->ReleaseRunOptions(X.run_opts); X.run_opts = NULL; }
    atomic_store_explicit(&X.npu_vote, 0, memory_order_release);
    if (X.npu_session) { X.api->ReleaseSession(X.npu_session); X.npu_session = NULL; }
    return 0;
}

/* Element type of session IO `idx`; leaves *dt untouched when the session
 * cannot describe it (caller's default stands and the shape check rejects). */
static void xfeat_io_dtype(int is_input, size_t idx, int64_t d[4],
                           ONNXTensorElementDataType *dt) {
    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    size_t nd = 0;
    OrtStatus *st = is_input
        ? X.api->SessionGetInputTypeInfo(X.npu_session, idx, &ti)
        : X.api->SessionGetOutputTypeInfo(X.npu_session, idx, &ti);
    if (!ort_ok(st, "npu TypeInfo") || !ti) return;
    X.api->CastTypeInfoToTensorInfo(ti, &tsi);
    if (tsi) {
        if (d) {
            X.api->GetDimensionsCount(tsi, &nd);
            X.api->GetDimensions(tsi, d, nd < 4 ? nd : 4);
        }
        ort_ok(X.api->GetTensorElementType(tsi, dt), "npu element type");
    }
    X.api->ReleaseTypeInfo(ti);
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
    pthread_mutex_lock(&XR_NPU_GATE);      /* context deserialize hits the HTP */
    st = X.api->CreateSession(X.env, X.npu_path, opts, &X.npu_session);
    pthread_mutex_unlock(&XR_NPU_GATE);
    X.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("XFeat NPU: HTP did not take the context (%s) — CPU model",
             X.api->GetErrorMessage(st));
        X.api->ReleaseStatus(st);
        X.npu_session = NULL;
        return 0;
    }

    /* Per-run DSP clock vote. v68: pin for the inference, release after (the
     * map thread runs at keyframe rate — the DSP idles cold in between).
     * v81: burst + a 100 us RPC latency budget, held across runs (no
     * post_run release — the validated recipe; tearing the vote down between
     * paced runs is what lets DCVS settle back to the slow sustained clock,
     * worth 31.5 -> ~9 ms on a reference model). If this firmware refuses the
     * vote anyway the first Run drops it and keeps going (see extract). */
    const int v81 = xfeat_is_v81();
    if (ort_ok(X.api->CreateRunOptions(&X.run_opts), "CreateRunOptions")) {
        if (v81) {
            int voted =
                ort_ok(X.api->AddRunConfigEntry(X.run_opts, "qnn.htp_perf_mode",
                                                "burst"), "RunCfg perf_mode") &&
                ort_ok(X.api->AddRunConfigEntry(X.run_opts,
                                                "qnn.rpc_control_latency", "100"),
                       "RunCfg rpc_latency");
            atomic_store_explicit(&X.npu_vote, voted, memory_order_release);
        } else {
            ort_ok(X.api->AddRunConfigEntry(X.run_opts, "qnn.htp_perf_mode",
                                            "sustained_high_performance"),
                   "RunCfg perf_mode");
            ort_ok(X.api->AddRunConfigEntry(X.run_opts, "qnn.htp_perf_mode_post_run",
                                            "default"),
                   "RunCfg perf_post");
        }
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
    /* Input encoding. UINT8 is the shipping A8W8 contract (input scale
     * 0.998 ~= 1.0, so the raw gray frame IS the tensor); FLOAT/FLOAT16
     * contexts take the same raw [0,255] gray level in the graph's float
     * domain. A u16 context cannot be bound here: unlike LAS2, XFeat's
     * contract carries no quantization scale to encode the input with, and
     * guessing one silently poisons every descriptor. */
    X.npu_in_dt = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    xfeat_io_dtype(1, 0, NULL, &X.npu_in_dt);
    if (X.npu_in_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 &&
        X.npu_in_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
        X.npu_in_dt != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        char why[64];
        snprintf(why, sizeof why, "input dtype %d unsupported", (int)X.npu_in_dt);
        return xfeat_npu_reject(why);
    }
    int mapped = 0;
    for (int i = 0; i < 3; i++) {
        if (!ort_ok(X.api->SessionGetOutputName(X.npu_session, (size_t)i, alloc,
                                                &name), "npu GetOutputName"))
            return 0;
        strncpy(X.npu_out[i], name, sizeof X.npu_out[i] - 1);
        alloc->Free(alloc, name);
        int64_t d[4] = { 0 };
        X.npu_out_dt[i] = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        xfeat_io_dtype(0, (size_t)i, d, &X.npu_out_dt[i]);
        if (d[1] == 64)                   { X.role[i] = 1; mapped |= 2; }
        else if (d[2] == IMG_H)           { X.role[i] = 0; mapped |= 1; }
        else if (d[2] == GRID_H)          { X.role[i] = 2; mapped |= 4; }
    }
    if (mapped != 7) {
        char why[64];
        snprintf(why, sizeof why, "unexpected output shapes (mask %d)", mapped);
        return xfeat_npu_reject(why);
    }
    /* The three dense maps must share one encoding family: all u8 (the A8W8
     * tail, dequantized with XFN_*_SCALE) or all float (fp16/fp32 tail). A
     * mixed set would need per-map scales the context does not hand us. */
    int n_u8 = 0, n_f = 0;
    for (int i = 0; i < 3; i++) {
        if (X.npu_out_dt[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) n_u8++;
        else if (X.npu_out_dt[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                 X.npu_out_dt[i] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) n_f++;
    }
    if (n_u8 != 3 && n_f != 3) {
        char why[80];
        snprintf(why, sizeof why, "mixed output dtypes %d/%d/%d",
                 (int)X.npu_out_dt[0], (int)X.npu_out_dt[1],
                 (int)X.npu_out_dt[2]);
        return xfeat_npu_reject(why);
    }
    X.npu_out_float = n_f == 3;
    /* Only the A8W8 tail exists: it reads the three maps as u8 planes, so a
     * float-IO context would be misread rather than merely slower. XFeat has
     * no fp16 build to feed one anyway — the v81 offline preparer cannot
     * create q::InstanceNorm.SumAndSquares_Tile.tcm, which its norm layer
     * needs — so refuse instead of publishing a session the tail cannot read. */
    if (X.npu_out_float)
        return xfeat_npu_reject("float-IO context has no dense tail");
    atomic_store_explicit(&X.npu_ready, 1, memory_order_release);
    LOGI("XFeat NPU ready: %s (HTP dense, in dtype %d, %s tail%s)", X.npu_path,
         (int)X.npu_in_dt, X.npu_out_float ? "float" : "A8W8",
         atomic_load_explicit(&X.npu_vote, memory_order_acquire)
             ? ", v81 burst vote" : "");
    return 1;
}

/* CPU tail over the dense maps: 5x5 NMS on the u8 score plane,
 * reliability-weighted ranking, top-`max`, bilinear 64-D descriptor
 * sampling + L2 norm + int8 quantization. */
typedef struct { float sc; uint16_t x, y; } xfn_peak;

static int64_t xfn_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

static void xfeat_sample_one(float x, float y, int8_t out[64]);

/* A8W8 tail: the dense maps in the context's u8 encoding. */
static int xfeat_npu_tail_u8(const uint8_t *score, const uint8_t *dns,
                             const uint8_t *rel, float (*uv)[2],
                             int8_t (*desc)[64], int max) {
    /* stash the dense map for xr_xfeat_sample (landmark anchoring) */
    memcpy(X.dense_u8, dns, sizeof X.dense_u8);
    X.dense_kind = 2;

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

    return hn;
}

/* Dense NPU inference: bind the frame in whatever encoding the context
 * declares, run the HTP under the process-wide gate, hand the three dense
 * maps to the tail. The gray frame doubles as the u8 quantized tensor
 * (input scale ~1.0), so the A8W8 path binds it zero-copy. */
static int xfeat_npu_extract(const uint8_t *img, float (*uv)[2],
                             int8_t (*desc)[64], int max) {
    const int64_t t_all = xfn_us();
    const size_t px = (size_t)IMG_H * IMG_W;
    const int64_t shape[4] = { 1, 1, IMG_H, IMG_W };
    OrtValue *in = NULL;
    int bound;
    if (X.npu_in_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        for (size_t i = 0; i < px; i++) X.npu_in_half[i] = (__fp16)img[i];
        bound = ort_ok(X.api->CreateTensorWithDataAsOrtValue(
                           X.meminfo, X.npu_in_half, px * sizeof(__fp16),
                           shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &in),
                       "npu CreateTensor");
    } else if (X.npu_in_dt == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        for (size_t i = 0; i < px; i++) X.input[i] = (float)img[i];
        bound = ort_ok(X.api->CreateTensorWithDataAsOrtValue(
                           X.meminfo, X.input, px * sizeof(float), shape, 4,
                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
                       "npu CreateTensor");
    } else {
        bound = ort_ok(X.api->CreateTensorWithDataAsOrtValue(
                           X.meminfo, (void *)img, px, shape, 4,
                           ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &in),
                       "npu CreateTensor");
    }
    if (!bound) return -1;

    const char *ins[1] = { X.npu_in };
    const char *outs[3] = { X.npu_out[0], X.npu_out[1], X.npu_out[2] };
    OrtValue *ov[3] = { NULL, NULL, NULL };
    pthread_mutex_lock(&XR_NPU_GATE);         /* exclusive HTP (see header) */
    const int64_t t_run = xfn_us();           /* gate wait vs run time split */
    OrtStatus *st = X.api->Run(X.npu_session, X.run_opts, ins,
                               (const OrtValue *const *)&in, 1, outs, 3, ov);
    pthread_mutex_unlock(&XR_NPU_GATE);
    const int64_t t_done = xfn_us();
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

    int hn = xfeat_npu_tail_u8(score, dns, rel, uv, desc, max);

    for (int i = 0; i < 3; i++)
        if (ov[i]) X.api->ReleaseValue(ov[i]);

    /* cadence telemetry: first few + every 32nd extract, with the gate-wait
     * vs HTP-run vs CPU-tail split so contention shows up in logcat */
    static int xfn_n;
    if (xfn_n < 3 || (xfn_n & 31) == 0)
        LOGI("XFeat NPU extract #%d: %d kp, wait %.1f + run %.1f + tail %.1f ms",
             xfn_n, hn, (t_run - t_all) / 1000.0f, (t_done - t_run) / 1000.0f,
             (xfn_us() - t_done) / 1000.0f);
    xfn_n++;
    return hn;
}

/* ---------------- shared bilinear descriptor sampler ---------------- */

/* Bilinear 64-D descriptor at pixel (x,y) from whichever dense map the
 * last extract cached; L2-normalized, int8-quantized (round(v*127)).
 *
 * FLOAT path uses XFeat's exact InterpolateSparse2d convention, validated
 * against torch grid_sample(align_corners=false) to 5e-7 (dense-export
 * agent): g = p * GRID / (IMG - 1) - 0.5, out-of-bounds taps contribute
 * ZERO (no clamp). The u8/NPU path keeps the device-validated
 * (p+0.5)/8-0.5 + clamp variant it has always used. */
static void xfeat_sample_one(float x, float y, int8_t out[64]) {
    float v[64], nrm = 0;
    if (X.dense_kind == 1) {               /* CPU dense map: PADDED dims —
                                              the model saw the padded image */
        const int plane = PGRID_H * PGRID_W;
        float gx = x * (float)PGRID_W / (float)(IMG_PW - 1) - 0.5f;
        float gy = y * (float)PGRID_H / (float)(IMG_PH - 1) - 0.5f;
        float fxf = floorf(gx), fyf = floorf(gy);
        int x0 = (int)fxf, y0 = (int)fyf;
        float wx = gx - fxf, wy = gy - fyf;
        float w[4] = { (1 - wy) * (1 - wx), (1 - wy) * wx,
                       wy * (1 - wx), wy * wx };
        int xs[4] = { x0, x0 + 1, x0, x0 + 1 };
        int ys[4] = { y0, y0, y0 + 1, y0 + 1 };
        for (int c = 0; c < 64; c++) v[c] = 0;
        for (int t = 0; t < 4; t++) {
            if (xs[t] < 0 || xs[t] >= PGRID_W ||
                ys[t] < 0 || ys[t] >= PGRID_H)
                continue;                  /* grid_sample zeros padding */
            const float *p = X.dense_f + ys[t] * PGRID_W + xs[t];
            for (int c = 0; c < 64; c++) v[c] += w[t] * p[c * plane];
        }
        for (int c = 0; c < 64; c++) nrm += v[c] * v[c];
    } else {                               /* u8 (NPU): offset -127, scale
                                              cancels in the L2 norm */
        const int plane = GRID_H * GRID_W;
        float gx = (x + 0.5f) * 0.125f - 0.5f;
        float gy = (y + 0.5f) * 0.125f - 0.5f;
        if (gx < 0) gx = 0; if (gx > GRID_W - 1.001f) gx = GRID_W - 1.001f;
        if (gy < 0) gy = 0; if (gy > GRID_H - 1.001f) gy = GRID_H - 1.001f;
        int ix = (int)gx, iy = (int)gy;
        float fx = gx - ix, fy = gy - iy;
        float w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx;
        float w10 = fy * (1 - fx), w11 = fy * fx;
        const uint8_t *base = X.dense_u8 + iy * GRID_W + ix;
        for (int c = 0; c < 64; c++) {
            const uint8_t *p = base + c * plane;
            float d = w00 * (p[0] - 127) + w01 * (p[1] - 127) +
                      w10 * (p[GRID_W] - 127) + w11 * (p[GRID_W + 1] - 127);
            v[c] = d;
            nrm += d * d;
        }
    }
    nrm = nrm > 1e-12f ? 127.0f / sqrtf(nrm) : 0.0f;
    for (int c = 0; c < 64; c++) {
        float q = v[c] * nrm;
        out[c] = (int8_t)(q > 127.f ? 127 : q < -127.f ? -127 : lroundf(q));
    }
}

int xr_xfeat_sample(const float (*uv)[2], int n, int8_t (*desc)[64]) {
    pthread_mutex_lock(&XF_MU);
    if (!X.dense_kind) {                   /* sparse graph: no dense map */
        pthread_mutex_unlock(&XF_MU);
        return -1;
    }
    for (int i = 0; i < n; i++)
        xfeat_sample_one(uv[i][0], uv[i][1], desc[i]);
    pthread_mutex_unlock(&XF_MU);
    return n;
}

int xr_xfeat_can_sample(void) {
    return X.cpu_dense ||
           atomic_load_explicit(&X.npu_ready, memory_order_acquire);
}

/* ---------------- CPU DENSE path (xfeat_dense_dyn.onnx + C tail) -------- */

/* Float twin of the NPU tail: 5x5 NMS on the float heatmap, reliability-
 * weighted top-K, bilinear descriptor sampling — and the feats map cached
 * for xr_xfeat_sample. */
#define XFC_SCORE_THR 0.05f               /* official detect threshold */

static int xfeat_cpu_dense_extract(const uint8_t *img, float (*uv)[2],
                                   int8_t (*desc)[64], int max) {
    /* fill the (possibly padded) inference plane: real rows at the padded
     * stride, zeros in the pad band (grid_sample's zero-padding convention
     * matches, so padded taps contribute nothing) */
    const int plane_px = IMG_PH * IMG_PW;
    for (int y = 0; y < IMG_H; y++) {
        const uint8_t *s = img + y * IMG_W;
        float *d = X.input + y * IMG_PW;
        for (int x = 0; x < IMG_W; x++) d[x] = (float)s[x];
        for (int x = IMG_W; x < IMG_PW; x++) d[x] = 0.0f;
    }
    if (IMG_PH > IMG_H)
        memset(X.input + IMG_H * IMG_PW, 0,
               sizeof(float) * (size_t)(IMG_PH - IMG_H) * IMG_PW);
    if (X.dense_in_ch == 3) {
        memcpy(X.input + plane_px, X.input, sizeof(float) * plane_px);
        memcpy(X.input + 2 * plane_px, X.input, sizeof(float) * plane_px);
    }
    const int64_t shape[4] = { 1, X.dense_in_ch, IMG_PH, IMG_PW };
    OrtValue *in = NULL;
    if (!ort_ok(X.api->CreateTensorWithDataAsOrtValue(
                    X.meminfo, X.input,
                    sizeof(float) * (size_t)X.dense_in_ch * plane_px, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in),
                "dense CreateTensor"))
        return -1;
    const char *ins[1] = { X.in_name };
    const char *outs[3] = { X.out_names[0], X.out_names[1], X.out_names[2] };
    OrtValue *ov[3] = { NULL, NULL, NULL };
    OrtStatus *st = X.api->Run(X.session, NULL, ins,
                               (const OrtValue *const *)&in, 1, outs, 3, ov);
    X.api->ReleaseValue(in);
    if (!ort_ok(st, "dense Run")) return -1;
    float *score = NULL, *feats = NULL, *rel = NULL;
    if (!ort_ok(X.api->GetTensorMutableData(ov[0], (void **)&score), "scores") ||
        !ort_ok(X.api->GetTensorMutableData(ov[1], (void **)&feats), "feats") ||
        !ort_ok(X.api->GetTensorMutableData(ov[2], (void **)&rel), "rel")) {
        for (int k = 0; k < 3; k++) if (ov[k]) X.api->ReleaseValue(ov[k]);
        return -1;
    }
    /* validate the runtime feats shape against the compiled grid — a
     * wrong-resolution model must not be memcpy'd at the compiled size */
    {
        OrtTensorTypeAndShapeInfo *ti = NULL;
        int64_t d[4] = { 0 };
        size_t nd = 0;
        if (ort_ok(X.api->GetTensorTypeAndShape(ov[1], &ti), "feats shape")) {
            X.api->GetDimensionsCount(ti, &nd);
            X.api->GetDimensions(ti, d, nd < 4 ? nd : 4);
            X.api->ReleaseTensorTypeAndShapeInfo(ti);
        }
        if (d[1] != 64 || d[2] != PGRID_H || d[3] != PGRID_W) {
            LOGE("XFeat dense: feats %dx%dx%d != 64x%dx%d — wrong-resolution "
                 "model, extraction off this frame", (int)d[1], (int)d[2],
                 (int)d[3], PGRID_H, PGRID_W);
            for (int k = 0; k < 3; k++)
                if (ov[k]) X.api->ReleaseValue(ov[k]);
            return -1;
        }
    }
    memcpy(X.dense_f, feats, sizeof X.dense_f);
    X.dense_kind = 1;

    /* separable 5x5 max filter, row pass (float twin of the NPU tail).
     * The score plane is PADDED-stride; peaks are only sought in the real
     * region so the pad band never spawns a keypoint. */
    static float frowmax[IMG_PH * IMG_PW];
    for (int y = 0; y < IMG_H; y++) {
        const float *s = score + y * IMG_PW;
        float *r = frowmax + y * IMG_PW;
        for (int x = 0; x < IMG_W; x++) {
            int x0 = x - 2 < 0 ? 0 : x - 2,
                x1 = x + 2 >= IMG_W ? IMG_W - 1 : x + 2;
            float m = 0;
            for (int k = x0; k <= x1; k++) if (s[k] > m) m = s[k];
            r[x] = m;
        }
    }
    if (max > XR_XFEAT_MAX_KP) max = XR_XFEAT_MAX_KP;
    static xfn_peak heap[XR_XFEAT_MAX_KP];
    int hn = 0;
    const int plane_g = GRID_H * GRID_W;
    (void)plane_g;
    for (int y = XFN_BORDER; y < IMG_H - XFN_BORDER; y++) {
        const float *s = score + y * IMG_PW;
        for (int x = XFN_BORDER; x < IMG_W - XFN_BORDER; x++) {
            float v = s[x];
            if (v < XFC_SCORE_THR) continue;
            float m = 0;
            for (int dy = -2; dy <= 2; dy++) {
                float rm = frowmax[(y + dy) * IMG_PW + x];
                if (rm > m) m = rm;
            }
            if (v != m) continue;
            /* reliability via the same validated InterpolateSparse2d
             * convention as descriptor sampling (zero-padded taps); the
             * model saw the PADDED image, so its dims drive the mapping */
            float gx = (float)x * (float)PGRID_W / (float)(IMG_PW - 1) - 0.5f;
            float gy = (float)y * (float)PGRID_H / (float)(IMG_PH - 1) - 0.5f;
            float fxf = floorf(gx), fyf = floorf(gy);
            int x0g = (int)fxf, y0g = (int)fyf;
            float wx = gx - fxf, wy = gy - fyf;
            float rl = 0;
            const float wt[4] = { (1 - wy) * (1 - wx), (1 - wy) * wx,
                                  wy * (1 - wx), wy * wx };
            const int xs[4] = { x0g, x0g + 1, x0g, x0g + 1 };
            const int ys[4] = { y0g, y0g, y0g + 1, y0g + 1 };
            for (int t = 0; t < 4; t++) {
                if (xs[t] < 0 || xs[t] >= PGRID_W ||
                    ys[t] < 0 || ys[t] >= PGRID_H)
                    continue;
                rl += wt[t] * rel[ys[t] * PGRID_W + xs[t]];
            }
            float sc = v * rl;
            if (hn < max) {
                int i = hn++;
                heap[i] = (xfn_peak){ sc, (uint16_t)x, (uint16_t)y };
                while (i && heap[(i - 1) / 2].sc > heap[i].sc) {
                    xfn_peak t = heap[i]; heap[i] = heap[(i - 1) / 2];
                    heap[(i - 1) / 2] = t; i = (i - 1) / 2;
                }
            } else if (sc > heap[0].sc) {
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
    for (int end = hn - 1; end > 0; end--) {
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
    for (int i = 0; i < hn; i++) {
        uv[i][0] = (float)heap[i].x;
        uv[i][1] = (float)heap[i].y;
        xfeat_sample_one(uv[i][0], uv[i][1], desc[i]);
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

    /* discover the io names. Two possible graphs behind this path:
     *  - sparse (original): outputs mapped by position — keypoints,
     *    descriptors, scores;
     *  - dense (xfeat_dense_dyn.onnx): outputs named feats / scores /
     *    reliability, mapped by NAME into role slots (0 scores, 1 feats,
     *    2 reliability) — shapes are symbolic in the dynamic export, so
     *    name is the only reliable key. */
    OrtAllocator *alloc = NULL;
    X.api->GetAllocatorWithDefaultOptions(&alloc);
    char *name = NULL;
    if (!ort_ok(X.api->SessionGetInputName(X.session, 0, alloc, &name),
                "GetInputName"))
        return 0;
    strncpy(X.in_name, name, sizeof X.in_name - 1);
    alloc->Free(alloc, name);
    int dense_mask = 0;
    char raw[3][64];
    for (int i = 0; i < 3; i++) {
        if (!ort_ok(X.api->SessionGetOutputName(X.session, (size_t)i, alloc,
                                                &name), "GetOutputName"))
            return 0;
        strncpy(raw[i], name, sizeof raw[i] - 1);
        raw[i][sizeof raw[i] - 1] = 0;
        alloc->Free(alloc, name);
    }
    for (int i = 0; i < 3; i++) {
        if (!strcmp(raw[i], "scores") || strstr(raw[i], "score")) {
            strncpy(X.out_names[0], raw[i], sizeof X.out_names[0] - 1);
            dense_mask |= 1;
        } else if (!strcmp(raw[i], "feats")) {
            strncpy(X.out_names[1], raw[i], sizeof X.out_names[1] - 1);
            dense_mask |= 2;
        } else if (!strcmp(raw[i], "reliability")) {
            strncpy(X.out_names[2], raw[i], sizeof X.out_names[2] - 1);
            dense_mask |= 4;
        }
    }
    X.cpu_dense = dense_mask == 7;
    if (!X.cpu_dense)                       /* sparse: positional as before */
        for (int i = 0; i < 3; i++)
            strncpy(X.out_names[i], raw[i], sizeof X.out_names[i] - 1);
    /* input channel count (dense export traced 1-ch gray; sparse is 3-ch) */
    X.dense_in_ch = 3;
    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    if (X.api->SessionGetInputTypeInfo(X.session, 0, &ti) == NULL && ti) {
        int64_t d[4] = { 0 };
        size_t nd = 0;
        X.api->CastTypeInfoToTensorInfo(ti, &tsi);
        if (tsi) {
            X.api->GetDimensionsCount(tsi, &nd);
            X.api->GetDimensions(tsi, d, nd < 4 ? nd : 4);
            if (d[1] == 1) X.dense_in_ch = 1;
        }
        X.api->ReleaseTypeInfo(ti);
    }
    LOGI("XFeat CPU ready: %s (%s graph, in=%s x%dch, outs=%s,%s,%s)",
         model_path, X.cpu_dense ? "DENSE" : "sparse", X.in_name,
         X.dense_in_ch, X.out_names[0], X.out_names[1], X.out_names[2]);
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

/* Thread-safe: concurrent initializers WAIT and then re-check ready
 * instead of failing — the old busy-flag fast-fail let the preload
 * thread's in-flight init silently REJECT xr_map_set_use_xfeat(1), and
 * a whole run would bench BAD/TEBLID under an --xfeat label (review
 * finding). A failed attempt stays retryable. */
int xr_xfeat_init(const char *model_path) {
    if (atomic_load_explicit(&X.ready, memory_order_acquire)) return 1;
    static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&init_mu);
    int ok = atomic_load_explicit(&X.ready, memory_order_acquire)
                 ? 1 : xfeat_try_init(model_path);
    pthread_mutex_unlock(&init_mu);
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

static int xfeat_extract_inner(const uint8_t *img, float (*uv)[2],
                               int8_t (*desc)[64], int max) {
    if (!atomic_load_explicit(&X.ready, memory_order_acquire)) return -1;

    if (atomic_load_explicit(&X.npu_ready, memory_order_acquire)) {
        int n = xfeat_npu_extract(img, uv, desc, max);
        if (n >= 0) return n;
        if (!X.session) return -1;   /* no CPU session to fall back on */
        LOGE("XFeat NPU extract failed — CPU fallback this frame");
    }
    if (!X.session) return -1;
    if (X.cpu_dense) return xfeat_cpu_dense_extract(img, uv, desc, max);

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

int xr_xfeat_extract(const uint8_t *img, float (*uv)[2],
                     int8_t (*desc)[64], int max) {
    pthread_mutex_lock(&XF_MU);
    int n = xfeat_extract_inner(img, uv, desc, max);
    pthread_mutex_unlock(&XF_MU);
    return n;
}

/* Compound extract + landmark-anchor sampling under ONE lock hold: the
 * anchors MUST come from the dense map of THIS extract's image — a
 * separate sample call can race another thread's extract in between and
 * silently store descriptors from the wrong frame (review finding). */
int xr_xfeat_extract_anchored(const uint8_t *img, float (*uv)[2],
                              int8_t (*desc)[64], int max,
                              const float (*auv)[2], int n_anchor,
                              int8_t (*adesc)[64], int *out_anchored) {
    pthread_mutex_lock(&XF_MU);
    int n = xfeat_extract_inner(img, uv, desc, max);
    int a = 0;
    if (n >= 0 && n_anchor > 0 && X.dense_kind) {
        for (int i = 0; i < n_anchor; i++)
            xfeat_sample_one(auv[i][0], auv[i][1], adesc[i]);
        a = n_anchor;
    }
    pthread_mutex_unlock(&XF_MU);
    if (out_anchored) *out_anchored = a;
    return n;
}
