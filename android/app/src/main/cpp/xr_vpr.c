/* xr_vpr.c — see xr_vpr.h. Two inference paths behind one contract:
 *
 *   NPU — the EigenPlaces backbone as an AoT EPContext .onnx on QNN's HTP.
 *         The context is Hexagon-arch specific and so is its graph IO: the
 *         v81 (SM8850) build is NATIVE fp16 end to end, the v68 (SM8350)
 *         build is QAIRT-native A8W8 with u8 IO. Neither dtype is assumed —
 *         both are read off the session at bring-up.
 *   CPU — the fp32 .onnx on ORT's CPU EP (the benchmark container path, and
 *         the fallback whenever the QNN EP / context is unavailable: non-qnn
 *         builds, non-QC SoCs, an arch-mismatched context).
 *
 * Either way the caller gets the same L2-normalized float embedding, so the
 * map layer cannot tell them apart. One session for the process; nothing
 * links libonnxruntime (dlopen'd, as xr_xfeat does). */
#include "xr_vpr.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/system_properties.h>

#include <android/log.h>

#include "ort/onnxruntime_c_api.h"
#include "xr_liteanystereo.h"   /* XR_NPU_GATE — process-wide HTP serializer */
#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

/* __fp16 is an ARM storage type (arm64 always has it under NDK 21); the
 * container's x86 build only ever sees the fp32 CPU model, so the half
 * conversions compile on the device targets alone. */
#ifdef __ANDROID__
#define VPR_HAS_FP16 1
#endif

/* u8 quantization of the v68 EigenPlaces context's IO (ctx_eigen_a8
 * meta.json). Input scale 0.998046875 with offset 0, so the raw 8-bit gray
 * frame IS the quantized tensor — worst case a 0.5 gray-level bias at 255,
 * far below the quantizer's own noise (the same property the XFeat A8W8
 * context relies on). The embedding dequantizes as (q + offset) * 0.00131859828
 * with QNN's stored offset -130; the scale is a positive constant that cancels
 * in the L2 renormalization below, so only the zero point is applied. */
#define VPR_Q_ZERO 130

/* u16 quantization of the A16W8 v68 context (ctx_eigen_a16w8 meta.json), the
 * variant actually proven on the 888. Input scale 0.0038910505827516317 with
 * offset 0, i.e. q = gray / scale = gray * 257 exactly (255 * 257 = 65535, so
 * the 8-bit range maps onto the full u16 span without clipping). The embedding
 * dequantizes as (q + offset) * 5.14680868946e-06 with QNN's stored offset
 * -33346; as on the u8 path the positive scale cancels in the renormalization,
 * leaving only the zero point. */
#define VPR_Q_ZERO16 33346
#define VPR_Q_IN16   257

static struct {
    const OrtApi *api;
    void *dl;                         /* libonnxruntime handle (RTLD_LOCAL) */
    OrtEnv *env;
    OrtSession *session;
    OrtMemoryInfo *meminfo;
    OrtRunOptions *run_opts;          /* per-run DSP clock vote; NULL on CPU */
    char in_name[64];
    char out_name[64];
    char path[512];
    char npu_path[512];
    atomic_int ready;
    int ort_up;                       /* api/env/meminfo initialized */
    int failed;                       /* permanent: don't retry every frame */
    int dim;                          /* model output dim, set at bring-up */
    int on_cuda;                      /* CUDA EP actually active */
    int on_npu;                       /* session lives on the QNN HTP */
    ONNXTensorElementDataType in_ty;  /* graph IO dtypes, read off the session */
    ONNXTensorElementDataType out_ty;
    float input[XR_OH * XR_OW];       /* map thread only */
    uint16_t input_q16[XR_OH * XR_OW]; /* u16-IO staging (A16W8 v68 context) */
#ifdef VPR_HAS_FP16
    __fp16 input_h[XR_OH * XR_OW];    /* fp16-IO staging (v81 context) */
#endif
} V;

/* 1 = the active session runs on the CUDA EP (bench telemetry: a silent
 * CPU fallback converts a run into the map-density-starved mode) */
int xr_vpr_on_cuda(void) { return V.on_cuda; }

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

void xr_vpr_set_npu_model(const char *epctx_path) {
    if (!epctx_path || !epctx_path[0]) return;
    strncpy(V.npu_path, epctx_path, sizeof V.npu_path - 1);
}

int xr_vpr_ready(void) {
    return atomic_load_explicit(&V.ready, memory_order_acquire);
}

int xr_vpr_dim(void) {
    return atomic_load_explicit(&V.ready, memory_order_acquire) ? V.dim : 0;
}

/* ---------------- shared ORT bring-up ---------------- */

static int vpr_ensure_ort(void) {
    if (V.ort_up) return 1;
    V.dl = dlopen("libonnxruntime.so", RTLD_NOW | RTLD_LOCAL);
    if (!V.dl) {
        LOGI("VPR: libonnxruntime.so not present (%s) — retrieval off",
             dlerror());
        return 0;
    }
    const OrtApiBase *(*get_base)(void) =
        (const OrtApiBase *(*)(void))dlsym(V.dl, "OrtGetApiBase");
    if (!get_base) return 0;
    V.api = get_base()->GetApi(ORT_API_VERSION);
    if (!V.api) return 0;
    if (!ort_ok(V.api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "xrvpr", &V.env),
                "CreateEnv"))
        return 0;
    if (!ort_ok(V.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault,
                                           &V.meminfo), "CreateCpuMemoryInfo"))
        return 0;
    V.ort_up = 1;
    return 1;
}

static const char *vpr_ty_name(ONNXTensorElementDataType t) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "fp32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "fp16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "u8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return "u16";
    default:                                    return "unsupported";
    }
}

static int vpr_ty_supported(ONNXTensorElementDataType t) {
    if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ||
        t == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
        return 1;
#ifdef VPR_HAS_FP16
    if (t == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) return 1;
#endif
    return 0;
}

/* IO names, graph dtypes and the embedding dimension, all read off the live
 * session — the CPU model, the v68 quantized context and the v81 fp16 context
 * present three different contracts and one binary drives all of them.
 * Rejecting an unsupported dtype here beats reinterpreting the tensor and
 * shipping silently wrong cosines into the map. */
static int vpr_bind_io(const char *what) {
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

    OrtTypeInfo *ti = NULL;
    const OrtTensorTypeAndShapeInfo *tsi = NULL;
    V.in_ty = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (!ort_ok(V.api->SessionGetInputTypeInfo(V.session, 0, &ti),
                "InputTypeInfo"))
        return 0;
    V.api->CastTypeInfoToTensorInfo(ti, &tsi);
    if (tsi) V.api->GetTensorElementType(tsi, &V.in_ty);
    V.api->ReleaseTypeInfo(ti);

    /* the embedding dimension comes from the output shape [1, D] */
    ti = NULL;
    tsi = NULL;
    int64_t d[2] = { 0, 0 };
    size_t nd = 0;
    V.out_ty = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (!ort_ok(V.api->SessionGetOutputTypeInfo(V.session, 0, &ti),
                "OutputTypeInfo"))
        return 0;
    V.api->CastTypeInfoToTensorInfo(ti, &tsi);
    if (tsi) {
        V.api->GetDimensionsCount(tsi, &nd);
        V.api->GetDimensions(tsi, d, nd < 2 ? nd : 2);
        V.api->GetTensorElementType(tsi, &V.out_ty);
    }
    V.api->ReleaseTypeInfo(ti);

    if (!vpr_ty_supported(V.in_ty) || !vpr_ty_supported(V.out_ty)) {
        LOGE("VPR %s: graph IO dtypes in=%s out=%s not handled", what,
             vpr_ty_name(V.in_ty), vpr_ty_name(V.out_ty));
        return 0;
    }
    V.dim = (int)d[1];
    if (V.dim <= 0 || V.dim > XR_VPR_MAX_DIM) {
        LOGE("VPR %s: unusable embedding dim %d (max %d)", what, V.dim,
             XR_VPR_MAX_DIM);
        return 0;
    }
    return 1;
}

/* ---------------- inference (shared by both paths) ---------------- */

/* Stage the gray frame in whatever dtype the graph takes. u8 needs no copy at
 * all: with input scale ~1.0 / offset 0 the frame already IS the quantized
 * tensor. */
static OrtValue *vpr_make_input(const uint8_t *img) {
    const int plane = XR_OH * XR_OW;
    const int64_t shape[4] = { 1, 1, XR_OH, XR_OW };
    OrtValue *in = NULL;
    if (V.in_ty == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        if (!ort_ok(V.api->CreateTensorWithDataAsOrtValue(
                        V.meminfo, (void *)img, (size_t)plane, shape, 4,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, &in),
                    "CreateTensor u8"))
            return NULL;
        return in;
    }
    if (V.in_ty == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
        for (int i = 0; i < plane; i++)
            V.input_q16[i] = (uint16_t)(img[i] * VPR_Q_IN16);
        if (!ort_ok(V.api->CreateTensorWithDataAsOrtValue(
                        V.meminfo, V.input_q16, sizeof V.input_q16, shape, 4,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, &in),
                    "CreateTensor u16"))
            return NULL;
        return in;
    }
#ifdef VPR_HAS_FP16
    if (V.in_ty == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        for (int i = 0; i < plane; i++) V.input_h[i] = (__fp16)img[i];
        if (!ort_ok(V.api->CreateTensorWithDataAsOrtValue(
                        V.meminfo, V.input_h, sizeof V.input_h, shape, 4,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &in),
                    "CreateTensor fp16"))
            return NULL;
        return in;
    }
#endif
    for (int i = 0; i < plane; i++) V.input[i] = (float)img[i];
    if (!ort_ok(V.api->CreateTensorWithDataAsOrtValue(
                    V.meminfo, V.input, sizeof V.input, shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in), "CreateTensor"))
        return NULL;
    return in;
}

/* Convert/dequantize the output tensor into emb and L2-normalize it. Every
 * path renormalizes: the graph already emits a unit vector, but xr_map
 * compares embeddings by plain dot product, so the stored vectors have to be
 * true unit vectors under quantization error, fp16 rounding and numeric
 * drift alike. */
static int vpr_read_emb(OrtValue *ov, float emb[XR_VPR_MAX_DIM]) {
    void *p = NULL;
    if (!ort_ok(V.api->GetTensorMutableData(ov, &p), "out data") || !p)
        return 0;
    float n = 0;
    if (V.out_ty == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        const uint8_t *q = (const uint8_t *)p;
        for (int i = 0; i < V.dim; i++) {
            float v = (float)((int)q[i] - VPR_Q_ZERO);
            emb[i] = v;
            n += v * v;
        }
    }
    else if (V.out_ty == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16) {
        const uint16_t *q = (const uint16_t *)p;
        for (int i = 0; i < V.dim; i++) {
            float v = (float)((int)q[i] - VPR_Q_ZERO16);
            emb[i] = v;
            n += v * v;
        }
    }
#ifdef VPR_HAS_FP16
    else if (V.out_ty == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const __fp16 *h = (const __fp16 *)p;
        for (int i = 0; i < V.dim; i++) {
            float v = (float)h[i];
            emb[i] = v;
            n += v * v;
        }
    }
#endif
    else {
        const float *e = (const float *)p;
        for (int i = 0; i < V.dim; i++) {
            emb[i] = e[i];
            n += e[i] * e[i];
        }
    }
    /* Pre-normalization norm on the first embeds: on the u8 path it must land
     * near 1/dequant_scale (~758 for the A8W8 context) — an order-of-magnitude
     * miss is a wrong zero point, which cosine similarity would otherwise hide
     * as uniformly high scores. */
    static int nlog;
    if (nlog < 3) {
        LOGI("VPR embed #%d: raw norm %.3f (%s out, %d-D)", nlog, sqrtf(n),
             vpr_ty_name(V.out_ty), V.dim);
        nlog++;
    }
    n = n > 1e-12f ? 1.0f / sqrtf(n) : 0.0f;
    for (int i = 0; i < V.dim; i++) emb[i] *= n;
    return 1;
}

static int vpr_run(const uint8_t *img, float emb[XR_VPR_MAX_DIM]) {
    OrtValue *in = vpr_make_input(img);
    if (!in) return -1;
    const char *ins[1] = { V.in_name };
    const char *outs[1] = { V.out_name };
    OrtValue *ov = NULL;
    /* HTP runs never overlap another client's (see XR_NPU_GATE): the v68 has
     * no graph preemption and the per-run clock votes would interleave. */
    if (V.on_npu) pthread_mutex_lock(&XR_NPU_GATE);
    OrtStatus *st = V.api->Run(V.session, V.run_opts, ins,
                               (const OrtValue *const *)&in, 1, outs, 1, &ov);
    if (V.on_npu) pthread_mutex_unlock(&XR_NPU_GATE);
    V.api->ReleaseValue(in);
    if (!ort_ok(st, "Run")) return -1;
    int ok = vpr_read_emb(ov, emb);
    V.api->ReleaseValue(ov);
    return ok ? V.dim : -1;
}

/* ---------------- NPU path ---------------- */

static void lc(char *s) { for (; *s; s++) if (*s >= 'A' && *s <= 'Z') *s += 32; }

static int vpr_is_qcom(void) {
    char soc[PROP_VALUE_MAX] = {0}, b[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.manufacturer", soc);
    __system_property_get("ro.board.platform", b);
    lc(soc); lc(b);
    return strstr(soc, "qti") || strstr(soc, "qualcomm") ||
           strstr(b, "kona") || strstr(b, "lahaina") ||
           strstr(b, "taro") || strstr(b, "kalama") ||
           strstr(b, "pineapple") || strstr(b, "sm8") ? 1 : 0;
}

/* Hexagon v81 (SM8850 / 8 Elite Gen 5) vs everything else. Same keys the
 * Kotlin staging autoswitch matches on, so the arch that picked the asset and
 * the arch that decides the power vote can never disagree. */
static int vpr_is_v81(void) {
    char soc[PROP_VALUE_MAX] = {0}, b[PROP_VALUE_MAX] = {0},
         pb[PROP_VALUE_MAX] = {0};
    __system_property_get("ro.soc.model", soc);
    __system_property_get("ro.board.platform", b);
    __system_property_get("ro.product.board", pb);
    lc(soc); lc(b); lc(pb);
    return strstr(soc, "sm8850") || strstr(b, "canoe") ||
           strstr(pb, "canoe") ? 1 : 0;
}

/* Prepend our native-lib dir to the loader paths so QNN's HTP can reach the
 * Hexagon (libcdsprpc + DSP skel). ADSP_LIBRARY_PATH is SEMICOLON-delimited
 * (FastRPC convention). Mirrors xfeat_fastrpc_env; all three modules may run
 * this — prepending the same dir twice is harmless. */
static void vpr_fastrpc_env(char *libdir, size_t ln) {
    libdir[0] = 0;
    Dl_info info;
    if (dladdr((void *)vpr_fastrpc_env, &info) && info.dli_fname) {
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

/* One inference on a black frame: this is where the HTP finalizes the graph,
 * and the only cheap place a dtype mismatch or a rejected clock vote can still
 * surface. Failing here gives the context up for the CPU model instead of
 * losing every keyframe's embedding later. */
static int vpr_warmup(void) {
    uint8_t *img = calloc((size_t)XR_OH * XR_OW, 1);
    float *emb = calloc(XR_VPR_MAX_DIM, sizeof(float));
    int ok = img && emb ? vpr_run(img, emb) > 0 : 0;
    free(img);
    free(emb);
    return ok;
}

static int vpr_try_init_npu(void) {
    if (!V.npu_path[0] || !vpr_is_qcom()) return 0;

    char libdir[512];
    vpr_fastrpc_env(libdir, sizeof libdir);
    char htp[560];
    if (libdir[0]) snprintf(htp, sizeof htp, "%s/libQnnHtp.so", libdir);
    else           snprintf(htp, sizeof htp, "libQnnHtp.so");

    OrtSessionOptions *opts = NULL;
    if (!ort_ok(V.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    V.api->SetSessionLogSeverityLevel(opts, 2);
    V.api->SetIntraOpNumThreads(opts, 1);
    V.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
    /* ONE strict attempt: a single EPContext node carrying the arch-matched
     * HTP context (v81 native fp16 / v68 QAIRT-native A8W8). All-or-nothing on
     * the Hexagon — there are no boundary ops to split — so failure drops
     * cleanly to the CPU model. */
    const char *kf[] = { "backend_path", "offload_graph_io_quantization" };
    const char *vf[] = { htp, "0" };
    OrtStatus *st = V.api->SessionOptionsAppendExecutionProvider(
        opts, "QNN", kf, vf, 2);
    if (st) {
        LOGI("VPR NPU: QNN EP unavailable (%s) — CPU model",
             V.api->GetErrorMessage(st));
        V.api->ReleaseStatus(st);
        V.api->ReleaseSessionOptions(opts);
        return 0;
    }
    ort_ok(V.api->AddSessionConfigEntry(
               opts, "session.disable_cpu_ep_fallback", "1"),
           "DisableCpuFallback");
    pthread_mutex_lock(&XR_NPU_GATE);      /* context deserialize hits the HTP */
    st = V.api->CreateSession(V.env, V.npu_path, opts, &V.session);
    pthread_mutex_unlock(&XR_NPU_GATE);
    V.api->ReleaseSessionOptions(opts);
    if (st) {
        LOGE("VPR NPU: HTP did not take the context (%s) — CPU model",
             V.api->GetErrorMessage(st));
        V.api->ReleaseStatus(st);
        V.session = NULL;
        return 0;
    }
    V.on_npu = 1;                          /* gates the runs below on the HTP */
    if (!vpr_bind_io("NPU")) {
        V.api->ReleaseSession(V.session);
        V.session = NULL;
        V.on_npu = 0;
        return 0;
    }

    /* Per-run DSP clock vote. v81's unsigned protection domain ACCEPTS votes
     * and badly needs one — saturated default DCVS runs several times slower
     * than burst + a 100us RPC latency floor. The v68 gets the sustained vote
     * the XFeat/LAS2 paths already carry there. A PD that refuses votes fails
     * the whole Run, so a failed warmup retries vote-free before the context
     * is given up. */
    const int v81 = vpr_is_v81();
    int voted = 0;
    if (ort_ok(V.api->CreateRunOptions(&V.run_opts), "CreateRunOptions")) {
        if (v81)
            voted = ort_ok(V.api->AddRunConfigEntry(
                               V.run_opts, "qnn.htp_perf_mode", "burst"),
                           "RunCfg perf_mode") &&
                    ort_ok(V.api->AddRunConfigEntry(
                               V.run_opts, "qnn.rpc_control_latency", "100"),
                           "RunCfg rpc_latency");
        else
            voted = ort_ok(V.api->AddRunConfigEntry(
                               V.run_opts, "qnn.htp_perf_mode",
                               "sustained_high_performance"),
                           "RunCfg perf_mode") &&
                    ort_ok(V.api->AddRunConfigEntry(
                               V.run_opts, "qnn.htp_perf_mode_post_run",
                               "default"),
                           "RunCfg perf_post");
    }
    int ok = vpr_warmup();
    if (!ok && voted) {
        LOGI("VPR NPU: warmup failed under the clock vote — retrying vote-free");
        V.api->ReleaseRunOptions(V.run_opts);
        V.run_opts = NULL;
        voted = 0;
        ok = vpr_warmup();
    }
    if (!ok) {
        LOGE("VPR NPU: warmup inference failed — CPU model");
        if (V.run_opts) {
            V.api->ReleaseRunOptions(V.run_opts);
            V.run_opts = NULL;
        }
        V.api->ReleaseSession(V.session);
        V.session = NULL;
        V.on_npu = 0;
        return 0;
    }
    LOGI("VPR NPU ready: %s (HTP %s, IO %s->%s, %s)", V.npu_path,
         v81 ? "v81" : "v68", vpr_ty_name(V.in_ty), vpr_ty_name(V.out_ty),
         voted ? (v81 ? "burst vote" : "sustained vote") : "no clock vote");
    return 1;
}

/* ---------------- CPU path ---------------- */

static int vpr_try_init_cpu(void) {
    if (!V.path[0]) return 0;
    OrtSessionOptions *opts = NULL;
    if (!ort_ok(V.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
        return 0;
    V.api->SetIntraOpNumThreads(opts, 2);
    V.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
#ifndef __ANDROID__
    /* Opt-in CUDA EP for host/container replay (XR_ORT_CUDA=<device_id>).
     * Resolved via dlsym so the same source runs against CPU-only ORT.
     * Heavy VPR models (MegaLoc 915MB fp32, ~229ms/embed on CPU) drop to
     * ~10ms on GPU, restoring realistic retrieval cadence in the bench. */
    {
        const char *cud = getenv("XR_ORT_CUDA");
        if (cud && *cud) {
            /* look up on the dlopen handle: the lib is RTLD_LOCAL, so its
             * symbols are invisible to RTLD_DEFAULT */
            OrtStatus *(*cuda_ep)(OrtSessionOptions *, int) =
                (OrtStatus * (*)(OrtSessionOptions *, int))
                dlsym(V.dl, "OrtSessionOptionsAppendExecutionProvider_CUDA");
            if (cuda_ep) {
                OrtStatus *st = cuda_ep(opts, atoi(cud));
                if (st) {
                    LOGE("VPR: CUDA EP failed (%s), staying on CPU",
                         V.api->GetErrorMessage(st));
                    V.api->ReleaseStatus(st);
                } else {
                    LOGI("VPR: CUDA EP enabled (device %s)", cud);
                    V.on_cuda = 1;
                }
            } else {
                LOGI("VPR: CUDA EP symbol absent (CPU-only ORT), staying on CPU");
            }
        }
    }
#endif
    if (!ort_ok(V.api->CreateSession(V.env, V.path, opts, &V.session),
                "CreateSession")) {
        /* CUDA-EP init can fail TRANSIENTLY (GPU OOM under wide parallel
         * benches — corridor4 lost its whole run's retrieval to one 536MB
         * allocation). Retry the same model on plain CPU options before
         * giving up: slow embeds beat a permanently blind map. */
        V.api->ReleaseSessionOptions(opts);
        opts = NULL;
        if (!ort_ok(V.api->CreateSessionOptions(&opts), "CreateSessionOptions"))
            return 0;
        V.api->SetIntraOpNumThreads(opts, 2);
        V.api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
        if (!ort_ok(V.api->CreateSession(V.env, V.path, opts, &V.session),
                    "CreateSession (CPU retry)")) {
            V.api->ReleaseSessionOptions(opts);
            V.session = NULL;
            return 0;
        }
        if (V.on_cuda) {
            V.on_cuda = 0;
            LOGE("VPR: GPU session failed — running on CPU instead (~20x slower "
                 "embeds: map density WILL degrade)");
        }
    }
    V.api->ReleaseSessionOptions(opts);
    if (!vpr_bind_io("CPU")) {
        V.api->ReleaseSession(V.session);
        V.session = NULL;
        V.on_cuda = 0;
        return 0;
    }
    return 1;
}

static int vpr_try_init(void) {
    if (!vpr_ensure_ort()) return 0;
    /* NPU first; when the HTP takes the context the CPU session is skipped
     * entirely. Both slots usually hold the SAME staged path (the arch-matched
     * EPContext), which the CPU EP cannot execute anyway — so a declined
     * context leaves retrieval on the map's spatial gate, exactly as an
     * absent asset does. */
    if (!vpr_try_init_npu() && !vpr_try_init_cpu()) return 0;
    atomic_store_explicit(&V.ready, 1, memory_order_release);
    LOGI("VPR ready: %s (%s, %d-D, in=%s out=%s, IO %s->%s)",
         V.on_npu ? V.npu_path : V.path,
         V.on_npu ? "QNN HTP" : (V.on_cuda ? "CUDA EP" : "CPU EP"),
         V.dim, V.in_name, V.out_name,
         vpr_ty_name(V.in_ty), vpr_ty_name(V.out_ty));
    return 1;
}

int xr_vpr_embed(const uint8_t *img, float emb[XR_VPR_MAX_DIM]) {
    if (!atomic_load_explicit(&V.ready, memory_order_acquire)) {
        if (V.failed || (!V.path[0] && !V.npu_path[0])) return -1;
        if (!vpr_try_init()) {         /* map thread; one attempt only */
            V.failed = 1;
            return -1;
        }
    }
    return vpr_run(img, emb);
}
