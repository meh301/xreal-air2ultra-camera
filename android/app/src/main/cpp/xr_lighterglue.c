/* xr_lighterglue.c — see xr_lighterglue.h. ORT CPU inference, one session
 * for the process, modeled on xr_vpr's dlopen'd path (nothing links
 * libonnxruntime). Map-thread only, like every matcher stage. */
#include "xr_lighterglue.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>
#include <time.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"
#include "xr_liteanystereo.h"   /* XR_NPU_GATE — process-wide HTP serializer */

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* The NPU context is the attention stack TRUNCATED at the log-assignment
 * matrix: the dynamic LightGlue tail (TopK / mutual-NN / NonZero) cannot be
 * expressed in a static HTP graph, so it is applied here in C — the same split
 * xr_xfeat.c uses for its dense maps. Measured on SM8850: 10.5 ms on the HTP
 * against 17.5 ms for the equivalent CPU graph, and it recovers 99.6% of the
 * fp32 matcher's matches with no spurious pairs over the calibration set.
 * v68 has no NPU path at all and never will: the attention MatMul requires
 * dsp_arch >= v73 ("has incorrect Value 68, expected >= 73"), so SM8350 keeps
 * the full CPU graph. */
#define LG_ASSIGN_N XR_LGLUE_N            /* log-assignment is N x N */

static struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;               /* CPU: full graph, tail in-graph */
    OrtSession *npu_session;            /* v81: truncated graph, tail below */
    OrtRunOptions *run_opts;            /* v81 HTP clock vote */
    OrtMemoryInfo *meminfo;
    char path[512];
    char npu_path[512];
    char npu_out[128];                  /* assignment tensor name from session */
    atomic_int ready;
    atomic_int npu_ready;
    int failed;                        /* permanent: don't retry every call */
    float min_score;
    /* static-shape input planes (map thread only) */
    float kpts[2][XR_LGLUE_N][2];
    float desc[2][XR_LGLUE_N][64];
    /* fp16 staging + the assignment matrix readback (map thread only) */
    __fp16 kpts_h[2][XR_LGLUE_N][2];
    __fp16 desc_h[2][XR_LGLUE_N][64];
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

void xr_lglue_set_npu_model(const char *epctx_path) {
    if (!epctx_path || !epctx_path[0]) return;
    strncpy(G.npu_path, epctx_path, sizeof G.npu_path - 1);
}

int xr_lglue_npu_active(void) {
    return atomic_load_explicit(&G.npu_ready, memory_order_acquire);
}

int xr_lglue_ready(void) {
    return atomic_load_explicit(&G.ready, memory_order_acquire);
}

int xr_lglue_wanted(void) {
    return G.path[0] != 0 && !G.failed;
}

/* v81 only. The context is arch-locked and the attention MatMul needs
 * dsp_arch >= v73, so anything else stays on the CPU graph. */
static int lglue_is_v81(void) {
    char b[PROP_VALUE_MAX] = { 0 }, s[PROP_VALUE_MAX] = { 0 };
    __system_property_get("ro.board.platform", b);
    __system_property_get("ro.soc.model", s);
    for (char *p = b; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;
    for (char *p = s; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;
    return strstr(b, "canoe") || strstr(s, "sm8850") ? 1 : 0;
}

/* Prepend our native-lib dir to the loader paths so the HTP can reach the
 * Hexagon. ADSP_LIBRARY_PATH is SEMICOLON-delimited, LD_LIBRARY_PATH COLON —
 * getting that backwards silently kills DSP loading (see xr_liteanystereo). */
static void lglue_fastrpc_env(char *libdir, size_t ln) {
    libdir[0] = 0;
    Dl_info info;
    if (dladdr((void *)lglue_fastrpc_env, &info) && info.dli_fname) {
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

static int lglue_try_init_npu(void *dl) {
    if (!G.npu_path[0] || !lglue_is_v81()) return 0;

    char libdir[512];
    lglue_fastrpc_env(libdir, sizeof libdir);
    char htp[560];
    if (libdir[0]) snprintf(htp, sizeof htp, "%s/libQnnHtp.so", libdir);
    else           snprintf(htp, sizeof htp, "libQnnHtp.so");

    OrtSessionOptions *opts = NULL;
    if (!ort_ok(G.api->CreateSessionOptions(&opts), "npu CreateSessionOptions"))
        return 0;
    G.api->SetSessionLogSeverityLevel(opts, 2);
    G.api->SetIntraOpNumThreads(opts, 1);
    G.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    const char *kf[] = { "backend_path", "offload_graph_io_quantization" };
    const char *vf[] = { htp, "0" };
    OrtStatus *st = G.api->SessionOptionsAppendExecutionProvider(
        opts, "QNN", kf, vf, 2);
    if (st) {
        LOGI("LGLUE NPU: QNN EP unavailable (%s) — CPU graph",
             G.api->GetErrorMessage(st));
        G.api->ReleaseStatus(st);
        G.api->ReleaseSessionOptions(opts);
        return 0;
    }
    ort_ok(G.api->AddSessionConfigEntry(opts, "session.disable_cpu_ep_fallback", "1"),
           "npu DisableCpuFallback");
    pthread_mutex_lock(&XR_NPU_GATE);      /* context deserialize hits the HTP */
    st = G.api->CreateSession(G.env, G.npu_path, opts, &G.npu_session);
    pthread_mutex_unlock(&XR_NPU_GATE);
    G.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("LGLUE NPU: HTP did not take the context (%s) — CPU graph",
             G.api->GetErrorMessage(st));
        G.api->ReleaseStatus(st);
        G.npu_session = NULL;
        return 0;
    }
    /* v81's unsigned PD accepts power votes (SM8350's does not) */
    if (ort_ok(G.api->CreateRunOptions(&G.run_opts), "npu CreateRunOptions")) {
        ort_ok(G.api->AddRunConfigEntry(G.run_opts, "qnn.htp_perf_mode", "burst"),
               "npu perf_mode");
        ort_ok(G.api->AddRunConfigEntry(G.run_opts, "qnn.rpc_control_latency", "100"),
               "npu rpc_latency");
    }
    /* the assignment tensor's name is generated by the exporter — read it */
    OrtAllocator *alloc = NULL;
    char *name = NULL;
    G.api->GetAllocatorWithDefaultOptions(&alloc);
    if (!ort_ok(G.api->SessionGetOutputName(G.npu_session, 0, alloc, &name),
                "npu GetOutputName") || !name) {
        G.api->ReleaseSession(G.npu_session);
        G.npu_session = NULL;
        return 0;
    }
    strncpy(G.npu_out, name, sizeof G.npu_out - 1);
    alloc->Free(alloc, name);
    atomic_store_explicit(&G.npu_ready, 1, memory_order_release);
    LOGI("LGLUE NPU ready: %s (fp16 HTP attention + C tail, out '%s')",
         G.npu_path, G.npu_out);
    (void)dl;
    return 1;
}

/* The LightGlue tail the truncated graph drops, over the N x N log-assignment
 * matrix A: a pair (a,b) matches when b is a's argmax AND a is b's argmax
 * (mutual nearest neighbour); its confidence is exp(A[a][b]). No threshold is
 * baked into the graph — the XR_LGLUE_MIN floor is applied here, exactly as
 * the full CPU graph's consumer does. Verified against the fp32 model:
 * 240/241 matches recovered over the calibration set with zero spurious. */
static int lglue_tail(const float *A, int n0, int n1,
                      int *out_i0, int *out_i1, float *out_sc, int max_out) {
    static int best_b[LG_ASSIGN_N];      /* map thread only */
    static int best_a[LG_ASSIGN_N];
    for (int b = 0; b < LG_ASSIGN_N; b++) { best_a[b] = -1; }
    /* column argmax in one pass over the matrix, row argmax alongside it */
    static float col_max[LG_ASSIGN_N];
    for (int b = 0; b < LG_ASSIGN_N; b++) col_max[b] = -INFINITY;
    for (int a = 0; a < LG_ASSIGN_N; a++) {
        const float *row = A + (size_t)a * LG_ASSIGN_N;
        float rm = -INFINITY;
        int rb = -1;
        for (int b = 0; b < LG_ASSIGN_N; b++) {
            const float v = row[b];
            if (v > rm) { rm = v; rb = b; }
            if (v > col_max[b]) { col_max[b] = v; best_a[b] = a; }
        }
        best_b[a] = rb;
    }
    int nm = 0;
    for (int a = 0; a < n0 && nm < max_out; a++) {
        const int b = best_b[a];
        if (b < 0 || b >= n1) continue;              /* pad slot */
        if (best_a[b] != a) continue;                /* not mutual */
        const float sc = expf(A[(size_t)a * LG_ASSIGN_N + b]);
        if (sc < G.min_score) continue;
        out_i0[nm] = a;
        out_i1[nm] = b;
        out_sc[nm] = sc;
        nm++;
    }
    return nm;
}

static int lglue_match_npu(int n0, int n1, int *out_i0, int *out_i1,
                           float *out_sc, int max_out) {
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < XR_LGLUE_N; i++) {
            G.kpts_h[s][i][0] = (__fp16)G.kpts[s][i][0];
            G.kpts_h[s][i][1] = (__fp16)G.kpts[s][i][1];
            for (int c = 0; c < 64; c++)
                G.desc_h[s][i][c] = (__fp16)G.desc[s][i][c];
        }
    }
    const int64_t kshape[3] = { 1, XR_LGLUE_N, 2 };
    const int64_t dshape[3] = { 1, XR_LGLUE_N, 64 };
    OrtValue *in[4] = { NULL, NULL, NULL, NULL };
    const OrtValue *cin[4];
    int ok =
        ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                   G.meminfo, G.kpts_h[0], sizeof G.kpts_h[0], kshape, 3,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &in[0]), "npu kpts0") &&
        ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                   G.meminfo, G.kpts_h[1], sizeof G.kpts_h[1], kshape, 3,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &in[1]), "npu kpts1") &&
        ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                   G.meminfo, G.desc_h[0], sizeof G.desc_h[0], dshape, 3,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &in[2]), "npu desc0") &&
        ort_ok(G.api->CreateTensorWithDataAsOrtValue(
                   G.meminfo, G.desc_h[1], sizeof G.desc_h[1], dshape, 3,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &in[3]), "npu desc1");
    if (!ok) {
        for (int i = 0; i < 4; i++) if (in[i]) G.api->ReleaseValue(in[i]);
        return -1;
    }
    for (int i = 0; i < 4; i++) cin[i] = in[i];
    const char *ins[4] = { "kpts0", "kpts1", "desc0", "desc1" };
    const char *outs[1] = { G.npu_out };
    OrtValue *ov[1] = { NULL };
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pthread_mutex_lock(&XR_NPU_GATE);          /* exclusive HTP */
    OrtStatus *st = G.api->Run(G.npu_session, G.run_opts, ins, cin, 4, outs, 1, ov);
    pthread_mutex_unlock(&XR_NPU_GATE);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int i = 0; i < 4; i++) G.api->ReleaseValue(in[i]);
    if (!ort_ok(st, "npu Run")) {
        if (ov[0]) G.api->ReleaseValue(ov[0]);
        return -1;
    }
    /* ORT hands back the graph's declared fp16; widen once into the tail's
     * working matrix rather than converting inside the argmax loops */
    static float A[(size_t)LG_ASSIGN_N * LG_ASSIGN_N];   /* 1 MB, map thread */
    void *p = NULL;
    if (!ort_ok(G.api->GetTensorMutableData(ov[0], &p), "npu out data") || !p) {
        G.api->ReleaseValue(ov[0]);
        return -1;
    }
    const __fp16 *h = (const __fp16 *)p;
    for (size_t i = 0; i < (size_t)LG_ASSIGN_N * LG_ASSIGN_N; i++)
        A[i] = (float)h[i];
    G.api->ReleaseValue(ov[0]);

    const int nm = lglue_tail(A, n0, n1, out_i0, out_i1, out_sc, max_out);
    const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
                      (t1.tv_nsec - t0.tv_nsec) / 1e6;
    static int lgn;
    if (lgn < 4 || (lgn & 15) == 0)
        LOGI("LGLUE match #%d: %.1f ms (%d x %d kpts -> %d matches, v81 fp16 HTP)",
             lgn, ms, n0, n1, nm);
    lgn++;
    return nm;
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
#ifndef __ANDROID__
    /* CUDA EP on the bench container (XR_ORT_CUDA=<dev>, same contract as
     * xr_vpr): a CPU LighterGlue pass costs 30-80 ms and a single search
     * can trigger many — measured to starve the map thread (room1 stores
     * 225 -> 40 with LG on). GPU drops it to a few ms. */
    {
        const char *cud = getenv("XR_ORT_CUDA");
        if (cud && *cud) {
            OrtStatus *(*cuda_ep)(OrtSessionOptions *, int) =
                (OrtStatus * (*)(OrtSessionOptions *, int))
                dlsym(dl, "OrtSessionOptionsAppendExecutionProvider_CUDA");
            if (cuda_ep) {
                OrtStatus *st = cuda_ep(opts, atoi(cud));
                if (st) {
                    LOGI("LGLUE: CUDA EP failed (%s), CPU",
                         G.api->GetErrorMessage(st));
                    G.api->ReleaseStatus(st);
                } else {
                    LOGI("LGLUE: CUDA EP enabled (device %s)", cud);
                }
            }
        }
    }
#endif
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
    /* Try the HTP after the CPU session exists, so a refusal anywhere in the
     * QNN path simply leaves the full CPU graph in charge. */
    lglue_try_init_npu(dl);
    atomic_store_explicit(&G.ready, 1, memory_order_release);
    LOGI("LGLUE ready: %s (%d slots, min score %.2f, %s)", G.path, XR_LGLUE_N,
         (double)G.min_score,
         atomic_load_explicit(&G.npu_ready, memory_order_acquire)
             ? "v81 fp16 HTP + C tail" : "CPU fp32 full graph");
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

    if (atomic_load_explicit(&G.npu_ready, memory_order_acquire)) {
        int n = lglue_match_npu(n0, n1, out_i0, out_i1, out_sc, max_out);
        if (n >= 0) return n;
        LOGE("LGLUE: NPU match failed — CPU graph this call");
    }

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
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    OrtStatus *st = G.api->Run(G.session, NULL, ins,
                               (const OrtValue *const *)in, 4, outs, 2, ov);
    clock_gettime(CLOCK_MONOTONIC, &t1);
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

    /* This is the one model with no accelerator path on Android — the CUDA
     * escape above is compiled out, so it runs fp32 on the ORT CPU EP. On the
     * bench host a pass cost 30-80 ms and starved the map thread, which is why
     * the caller budgets it to the candidate keyframe alone (xr_map.c c==0).
     * Log the real per-call cost so that budget can be judged on this SoC
     * instead of assumed: first few calls, then every 16th. */
    {
        const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 +
                          (t1.tv_nsec - t0.tv_nsec) / 1e6;
        static int lgn;
        if (lgn < 4 || (lgn & 15) == 0)
            LOGI("LGLUE match #%d: %.1f ms (%d x %d kpts -> %d matches, CPU fp32)",
                 lgn, ms, n0, n1, nm);
        lgn++;
    }
    return nm;
}
