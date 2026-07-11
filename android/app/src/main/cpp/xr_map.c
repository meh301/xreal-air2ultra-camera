/* xr_map.c — see xr_map.h.
 *
 * Fallback descriptors: compact ORB-style — FAST-9 corners with grid NMS,
 * intensity-centroid orientation, rotated 256-bit BRIEF from a fixed
 * seeded pattern (self-consistent within this system; no OpenCV).
 * Primary descriptors: XFeat via xr_xfeat (int8-quantized 64-D, cosine ~=
 * dot/16129; NEON sdot when available). Brute force matching is fine at
 * session scale — the koide3/GLIM lesson applied here is the bounded
 * incremental store, not fancy search.
 */
#include "xr_map.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#include <android/log.h>

#include "xr_xfeat.h"
#include "xreal_core.h"

#if defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define MAP_SDOT 1
#endif

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

#define KF_DIST_M 0.30f            /* motion gates */
#define KF_ANGLE_COS 0.99144f      /* cos(15/2 deg) on the quat dot */
#define KF_NEAR_M 2.0f             /* proximity that keeps a keyframe fresh */

/* mini-ORB */
#define FAST_THRESH 18
#define NMS_GRID 24
#define MARGIN 20
#define ORB_MAX_DIST 60            /* Hamming acceptance */
#define ORB_MARGIN 10

/* XFeat int8 cosine: dot of two L2-normalized-then-*127 vectors */
#define XF_MIN_DOT 13000           /* ~cos 0.81 */
#define XF_MARGIN 1200

#define CAND_MIN_MATCHES 22
#define CAND_SKIP_RECENT 5

/* Store gates, from the shake-flood logcat: a flailing pose re-passes
 * the motion gate every few frames and stored ~140 zero-landmark
 * keyframes in 16 s — junk that can never be verified against, which
 * evicted the good map at the rolling cap. A keyframe earns storage by
 * carrying geometry, at a bounded rate. */
#define STORE_MIN_LM 20            /* a sparse frame is not worth a keyframe */
#define STORE_MIN_INTERVAL_NS 350000000ull

/* The correction is a RECOVERY from significant drift, not a continuous
 * clamp: only snap when the VIO has strayed meaningfully from the map,
 * and then not again for a cooldown (repeated micro-snaps read as
 * jitter). Below the deviation gate the VIO is trusted as-is. */
#define SNAP_MIN_M 0.30f
#define SNAP_MIN_ANG_RAD 0.14f     /* ~8 deg */
#define SNAP_COOLDOWN_NS 1500000000ull
/* the matched place must be an ESTABLISHED cluster (this many covisible
 * keyframes contributed pooled points) — a lone shake-spawned keyframe
 * or a two-frame junk map is rejected */
#define COVIS_MIN_KF 2
/* loop SEARCH is the heavy work (descriptor extraction + match against
 * every keyframe + PnP RANSAC). Un-throttled it ran back-to-back on
 * every offer — 15-20x/s in a loop-rich scene — and starved Basalt's VIO
 * threads (IMU queue overflow -> divergence -> "flying off"). ~3 Hz is
 * ample for relocalization; storage keeps its own (faster) cadence. */
#define LOOP_SEARCH_INTERVAL_NS 300000000ull
/* stationary reloc cadence: when the motion gate blocks, a query-only
 * offer still goes through this often — matching (against the SAME
 * old-keyframe set the moving queries use) but never storing. Standing
 * still and staring at a known scene must be able to heal the pose. */
#define QUERY_INTERVAL_NS 2500000000ull

enum { DESC_ORB = 0, DESC_XFEAT = 1 };

typedef struct {
    uint64_t ts;
    uint64_t last_used;            /* rolling eviction: refreshed by loop
                                      matches and spatial proximity */
    float q[4], p[3];              /* ODOM pose at capture (immutable — the
                                      graph's odometry measurements) */
    float qc[4], pc[3];            /* pose-graph CORRECTED pose (session) */
    int desc_type;
    int n_kp;
    float kp_uv[XR_MAP_KP_PER_KF][2];
    int8_t desc[XR_MAP_KP_PER_KF][64];   /* ORB uses the first 32 bytes */
    int lm_of_kp[XR_MAP_KP_PER_KF];      /* kp -> landmark index, or -1 */
    int n_lm;
    int32_t lm_id[XR_MAP_KP_PER_KF];
    float lm_xyz[XR_MAP_KP_PER_KF][3];
    float lm_uv[XR_MAP_KP_PER_KF][2];
} xr_kf;

static xr_kf KF[XR_MAP_MAX_KF];        /* .bss */
static int KF_N;
static atomic_int KF_COUNT_PUB;
static struct { uint64_t a, b; int matches; int have; } LAST_CAND;
/* last verification attempt, for the on-screen panel: no more guessing
 * which stage blocked a snap */
enum { VOUT_NONE = 0, VOUT_BELOW_BAR = 1, VOUT_FEW_PAIRS = 2,
       VOUT_FEW_INLIERS = 3, VOUT_CAPPED = 4, VOUT_APPLIED = 5 };
static struct { int pairs, inliers, outcome; } VER_LAST;
static struct {
    int count; float pos[3]; int matches;
    int n_lm;                            /* matched kf's stored landmarks — */
    float lm[XR_MAP_KP_PER_KF][3];       /* the AR loop/reloc flash */
} LOOP_STATS;

static pthread_mutex_t MAP_LOCK = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t MAP_COND = PTHREAD_COND_INITIALIZER;
static struct {                        /* worker -> map thread mailbox */
    int full;
    int query_only;                    /* stationary reloc query: no store */
    uint64_t ts;
    float q[4], p[3];
    uint8_t img[XR_OW * XR_OH];
    int n_lm;
    int32_t lm_id[XR_MAP_KP_PER_KF];
    float lm_xyz[XR_MAP_KP_PER_KF][3];
    float lm_uv[XR_MAP_KP_PER_KF][2];
} MBOX;
static struct { float q[4], p[3]; int have; } LAST_POSE;
static uint64_t LAST_ACCEPT_NS;        /* stationary query cadence anchor */
static uint64_t LAST_STORE_NS;         /* keyframe store rate limit */
static uint64_t LAST_SNAP_NS;          /* correction cooldown anchor */
static char MODEL_PATH[512];
static pthread_once_t THREAD_ONCE = PTHREAD_ONCE_INIT;
static atomic_int MAPPING = 1;
static atomic_int RECOVERY = 1;        /* live-pose snap (loop closure of
                                          the MAP itself is always on) */
static atomic_int USE_XFEAT = 0;       /* runtime descriptor selector */

/* Live session correction D = C_newest ∘ O_newest⁻¹ (map -> odom pattern):
 * composes onto every odom-frame quantity leaving the SLAM worker. Updated
 * by verified loop closures; identity until the first one. */
static struct { float q[4]; float p[3]; int gen; } CORR = { .q = {1, 0, 0, 0} };

static void *xfeat_preload_thread(void *arg) {
    (void)arg;
    if (MODEL_PATH[0]) xr_xfeat_init(MODEL_PATH);   /* dlopen ORT + load model */
    return NULL;
}

void xr_map_set_model(const char *onnx_path) {
    if (!onnx_path) return;
    strncpy(MODEL_PATH, onnx_path, sizeof MODEL_PATH - 1);
    /* eagerly load ORT + the model off the UI thread so xfeat_ready() is
     * accurate the moment the user flips the descriptor toggle (the lazy
     * map-thread init made the button label race and read "not ready") */
    pthread_t t;
    if (pthread_create(&t, NULL, xfeat_preload_thread, NULL) == 0)
        pthread_detach(t);
}

void xr_map_set_mapping(int on) {
    atomic_store(&MAPPING, on ? 1 : 0);
    LOGI("session map: %s", on ? "MAPPING" : "LOCALIZATION-ONLY (frozen)");
}

void xr_map_set_recovery(int on) {
    atomic_store(&RECOVERY, on ? 1 : 0);
    LOGI("session map: loop recovery %s",
         on ? "ON (verified closures snap the live pose)"
            : "OFF (map keeps closing loops internally; the live pose is "
              "odometry-continuous — the GNSS-fusion mode)");
}

/* Switch the keyframe descriptor (0 = mini-ORB, 1 = XFeat). ORB and
 * XFeat keyframes cannot match each other, so the store is cleared on a
 * real change — the map rebuilds with the selected descriptor. */
void xr_map_set_use_xfeat(int on) {
    int want = on ? 1 : 0;
    if (atomic_exchange(&USE_XFEAT, want) == want) return;
    pthread_mutex_lock(&MAP_LOCK);
    KF_N = 0;
    atomic_store(&KF_COUNT_PUB, 0);
    LAST_STORE_NS = 0;
    pthread_mutex_unlock(&MAP_LOCK);
    LOGI("session map: descriptor -> %s (keyframes cleared, rebuilding)",
         want ? "XFeat" : "mini-ORB");
}

/* Whether the XFeat model + ONNX Runtime are actually loaded (so the UI
 * can tell "XFeat requested" from "XFeat running"). */
int xr_map_xfeat_ready(void) {
    return MODEL_PATH[0] != 0 && xr_xfeat_available();
}

void xr_map_reset(void) {
    pthread_mutex_lock(&MAP_LOCK);
    KF_N = 0;
    LAST_CAND.have = 0;
    LAST_POSE.have = 0;
    MBOX.full = 0;
    LAST_ACCEPT_NS = 0;
    LAST_STORE_NS = 0;
    LAST_SNAP_NS = 0;
    memset(&LOOP_STATS, 0, sizeof LOOP_STATS);
    memset(&VER_LAST, 0, sizeof VER_LAST);
    CORR.q[0] = 1; CORR.q[1] = CORR.q[2] = CORR.q[3] = 0;
    CORR.p[0] = CORR.p[1] = CORR.p[2] = 0;
    CORR.gen++;                    /* consumers re-sync their frame */
    atomic_store(&KF_COUNT_PUB, 0);
    pthread_mutex_unlock(&MAP_LOCK);
}

int xr_map_get_correction(float q[4], float p[3]) {
    pthread_mutex_lock(&MAP_LOCK);
    memcpy(q, CORR.q, sizeof CORR.q);
    memcpy(p, CORR.p, sizeof CORR.p);
    int g = CORR.gen;
    pthread_mutex_unlock(&MAP_LOCK);
    return g;
}

/* ---- small quaternion / rotation kit (Hamilton wxyz, R row-major) ------------- */

static void qmul(const float a[4], const float b[4], float o[4]) {
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static void qconj(const float a[4], float o[4]) {
    o[0] = a[0]; o[1] = -a[1]; o[2] = -a[2]; o[3] = -a[3];
}

static void q2R(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z);
    R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z);     R[4] = 1 - 2 * (x * x + z * z);
    R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y);     R[7] = 2 * (y * z + w * x);
    R[8] = 1 - 2 * (x * x + y * y);
}

static void R2q(const float R[9], float q[4]) {
    float tr = R[0] + R[4] + R[8];
    if (tr > 0) {
        float s = sqrtf(tr + 1.0f) * 2;
        q[0] = 0.25f * s;
        q[1] = (R[7] - R[5]) / s;
        q[2] = (R[2] - R[6]) / s;
        q[3] = (R[3] - R[1]) / s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        float s = sqrtf(1.0f + R[0] - R[4] - R[8]) * 2;
        q[0] = (R[7] - R[5]) / s;
        q[1] = 0.25f * s;
        q[2] = (R[1] + R[3]) / s;
        q[3] = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        float s = sqrtf(1.0f + R[4] - R[0] - R[8]) * 2;
        q[0] = (R[2] - R[6]) / s;
        q[1] = (R[1] + R[3]) / s;
        q[2] = 0.25f * s;
        q[3] = (R[5] + R[7]) / s;
    } else {
        float s = sqrtf(1.0f + R[8] - R[0] - R[4]) * 2;
        q[0] = (R[3] - R[1]) / s;
        q[1] = (R[2] + R[6]) / s;
        q[2] = (R[5] + R[7]) / s;
        q[3] = 0.25f * s;
    }
    float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (int i = 0; i < 4; i++) q[i] /= n;
}

static void qrotv(const float q[4], const float v[3], float o[3]) {
    float R[9];
    q2R(q, R);
    o[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    o[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    o[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

/* rotation vector (axis*angle) <-> quaternion */
static void rv_from_q(const float q[4], float rv[3]) {
    float w = q[0] >= 0 ? q[0] : -q[0];
    float s = q[0] >= 0 ? 1.0f : -1.0f;
    float vn = sqrtf(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (vn < 1e-9f) { rv[0] = rv[1] = rv[2] = 0; return; }
    float ang = 2.0f * atan2f(vn, w);
    float k = s * ang / vn;
    rv[0] = q[1] * k; rv[1] = q[2] * k; rv[2] = q[3] * k;
}

static void q_from_rv(const float rv[3], float q[4]) {
    float ang = sqrtf(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]);
    if (ang < 1e-9f) { q[0] = 1; q[1] = q[2] = q[3] = 0; return; }
    float s = sinf(ang * 0.5f) / ang;
    q[0] = cosf(ang * 0.5f);
    q[1] = rv[0] * s; q[2] = rv[1] * s; q[3] = rv[2] * s;
}

/* pose composition o = a ∘ b on (q, p) pairs: R_o = R_a R_b,
 * p_o = R_a p_b + p_a */
static void pose_compose(const float qa[4], const float pa[3],
                         const float qb[4], const float pb[3],
                         float qo[4], float po[3]) {
    float t[3];
    qrotv(qa, pb, t);
    qmul(qa, qb, qo);
    po[0] = t[0] + pa[0]; po[1] = t[1] + pa[1]; po[2] = t[2] + pa[2];
}

/* o = a⁻¹ */
static void pose_invert(const float qa[4], const float pa[3],
                        float qo[4], float po[3]) {
    float t[3];
    qconj(qa, qo);
    qrotv(qo, pa, t);
    po[0] = -t[0]; po[1] = -t[1]; po[2] = -t[2];
}

int xr_map_verify_stats(int *pairs, int *inliers) {
    pthread_mutex_lock(&MAP_LOCK);
    *pairs = VER_LAST.pairs;
    *inliers = VER_LAST.inliers;
    int out = VER_LAST.outcome;
    pthread_mutex_unlock(&MAP_LOCK);
    return out;
}

int xr_map_loop_stats(int *count, float pos[3], int *matches) {
    pthread_mutex_lock(&MAP_LOCK);
    *count = LOOP_STATS.count;
    memcpy(pos, LOOP_STATS.pos, sizeof LOOP_STATS.pos);
    *matches = LOOP_STATS.matches;
    int have = LOOP_STATS.count > 0;
    pthread_mutex_unlock(&MAP_LOCK);
    return have;
}

int xr_map_loop_points(float *xyz, int max) {
    pthread_mutex_lock(&MAP_LOCK);
    int n = LOOP_STATS.n_lm;
    if (n > max) n = max;
    if (n > 0) memcpy(xyz, LOOP_STATS.lm, sizeof(float) * 3u * (size_t)n);
    pthread_mutex_unlock(&MAP_LOCK);
    return n;
}

int xr_map_num_keyframes(void) {
    return atomic_load(&KF_COUNT_PUB);
}

int xr_map_last_candidate(uint64_t *ts_a, uint64_t *ts_b, int *matches) {
    pthread_mutex_lock(&MAP_LOCK);
    int have = LAST_CAND.have;
    if (have) {
        *ts_a = LAST_CAND.a;
        *ts_b = LAST_CAND.b;
        *matches = LAST_CAND.matches;
    }
    pthread_mutex_unlock(&MAP_LOCK);
    return have;
}

/* ---- mini-ORB fallback --------------------------------------------------------- */

static int8_t PAT[256][4];
static int pat_ready;

static void pat_init(void) {
    if (pat_ready) return;
    uint32_t s = 0xC0FFEE01u;
    for (int i = 0; i < 256; i++)
        for (int k = 0; k < 4; k++) {
            s = s * 1664525u + 1013904223u;
            PAT[i][k] = (int8_t)((int)(s >> 24) % 14 * ((s >> 23 & 1) ? 1 : -1));
        }
    pat_ready = 1;
}

static const int8_t CIRC[16][2] = {
    {0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},
    {0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}
};

static int fast_score(const uint8_t *img, int x, int y) {
    int c = img[y * XR_OW + x];
    int b = 0, d = 0;
    for (int k = 0; k < 16; k += 4) {
        int v = img[(y + CIRC[k][1]) * XR_OW + x + CIRC[k][0]];
        if (v > c + FAST_THRESH) b++;
        else if (v < c - FAST_THRESH) d++;
    }
    if (b < 3 && d < 3) return 0;
    int ring[32];
    for (int k = 0; k < 16; k++)
        ring[k] = ring[k + 16] =
            (int)img[(y + CIRC[k][1]) * XR_OW + x + CIRC[k][0]] - c;
    int run_b = 0, run_d = 0, best = 0;
    for (int k = 0; k < 32; k++) {
        run_b = ring[k] > FAST_THRESH ? run_b + 1 : 0;
        run_d = ring[k] < -FAST_THRESH ? run_d + 1 : 0;
        if (run_b >= 9 || run_d >= 9) {
            int s = 0;
            for (int j = 0; j < 16; j++) s += abs(ring[j]);
            if (s > best) best = s;
        }
    }
    return best;
}

/* rotated-BRIEF descriptor with centroid orientation, at (bx, by) */
static void orb_describe(const uint8_t *img, int bx, int by, int8_t *desc) {
    int m10 = 0, m01 = 0;
    for (int dy = -7; dy <= 7; dy++)
        for (int dx = -7; dx <= 7; dx++) {
            int v = img[(by + dy) * XR_OW + bx + dx];
            m10 += dx * v;
            m01 += dy * v;
        }
    float ang = atan2f((float)m01, (float)m10);
    float ca = cosf(ang), sa = sinf(ang);
    uint64_t d[4] = { 0, 0, 0, 0 };
    for (int b = 0; b < 256; b++) {
        int ax = (int)lroundf(ca * PAT[b][0] - sa * PAT[b][1]);
        int ay = (int)lroundf(sa * PAT[b][0] + ca * PAT[b][1]);
        int bx2 = (int)lroundf(ca * PAT[b][2] - sa * PAT[b][3]);
        int by2 = (int)lroundf(sa * PAT[b][2] + ca * PAT[b][3]);
        if (img[(by + ay) * XR_OW + bx + ax] <
            img[(by + by2) * XR_OW + bx + bx2])
            d[b >> 6] |= 1ull << (b & 63);
    }
    memcpy(desc, d, sizeof d);
}

static void orb_extract(const uint8_t *img, xr_kf *kf) {
    pat_init();
    kf->n_kp = 0;
    kf->desc_type = DESC_ORB;
    /* descriptors anchored AT the VIO landmarks first: a descriptor match
     * then IS a 3D-3D landmark correspondence — what the loop verification
     * and pose graph consume. Basalt picks corners, so the patches are
     * descriptor-worthy by construction. */
    for (int i = 0; i < kf->n_lm && kf->n_kp < XR_MAP_KP_PER_KF; i++) {
        int x = (int)lroundf(kf->lm_uv[i][0]);
        int y = (int)lroundf(kf->lm_uv[i][1]);
        if (x < MARGIN || x >= XR_OW - MARGIN ||
            y < MARGIN || y >= XR_OH - MARGIN)
            continue;
        int j = kf->n_kp++;
        kf->kp_uv[j][0] = (float)x;
        kf->kp_uv[j][1] = (float)y;
        kf->lm_of_kp[j] = i;
        orb_describe(img, x, y, kf->desc[j]);
    }
    /* then FAST-grid corners for place-recognition coverage (no 3D) */
    enum { GX = (XR_OW - 2 * MARGIN) / NMS_GRID,
           GY = (XR_OH - 2 * MARGIN) / NMS_GRID };
    int n_anchored = kf->n_kp;
    for (int gy = 0; gy < GY && kf->n_kp < XR_MAP_KP_PER_KF; gy++)
        for (int gx = 0; gx < GX && kf->n_kp < XR_MAP_KP_PER_KF; gx++) {
            int bs = 0, bx = -1, by = -1;
            for (int y = MARGIN + gy * NMS_GRID;
                 y < MARGIN + (gy + 1) * NMS_GRID; y += 2)
                for (int x = MARGIN + gx * NMS_GRID;
                     x < MARGIN + (gx + 1) * NMS_GRID; x += 2) {
                    int s = fast_score(img, x, y);
                    if (s > bs) { bs = s; bx = x; by = y; }
                }
            if (bs <= 0) continue;
            int dup = 0;                   /* skip near a landmark anchor */
            for (int k = 0; k < n_anchored; k++) {
                float du = kf->kp_uv[k][0] - (float)bx;
                float dv = kf->kp_uv[k][1] - (float)by;
                if (du * du + dv * dv < 36.0f) { dup = 1; break; }
            }
            if (dup) continue;
            int i = kf->n_kp++;
            kf->kp_uv[i][0] = (float)bx;
            kf->kp_uv[i][1] = (float)by;
            kf->lm_of_kp[i] = -1;
            orb_describe(img, bx, by, kf->desc[i]);
        }
}

/* ---- matching ------------------------------------------------------------------- */

static inline int hamming256(const int8_t *a, const int8_t *b) {
    const uint64_t *ua = (const uint64_t *)a, *ub = (const uint64_t *)b;
    return __builtin_popcountll(ua[0] ^ ub[0]) +
           __builtin_popcountll(ua[1] ^ ub[1]) +
           __builtin_popcountll(ua[2] ^ ub[2]) +
           __builtin_popcountll(ua[3] ^ ub[3]);
}

static inline int dot64_i8(const int8_t *a, const int8_t *b) {
#ifdef MAP_SDOT
    int32x4_t acc = vdupq_n_s32(0);
    for (int k = 0; k < 64; k += 16)
        acc = vdotq_s32(acc, vld1q_s8(a + k), vld1q_s8(b + k));
    return vaddvq_s32(acc);
#else
    int s = 0;
    for (int k = 0; k < 64; k++) s += (int)a[k] * (int)b[k];
    return s;
#endif
}

/* ratio-tested matching over the FIRST na/nb keypoints of each frame;
 * when pairs != NULL, the kp index pairs of up to max_pairs matches are
 * stored as well. Returns the total match count. Keypoints are ordered
 * landmark-anchored first, so limiting na/nb to the anchored counts
 * matches ONLY 3D-carrying corners — without the FAST fill-ins stealing
 * best/second-best in the ratio test (which starved the verification of
 * 3D pairs and silently blocked every snap). */
static int match_pairs_lim(const xr_kf *a, int na, const xr_kf *b, int nb,
                           int (*pairs)[2], int max_pairs) {
    if (a->desc_type != b->desc_type) return 0;
    int n = 0;
    if (a->desc_type == DESC_XFEAT) {
        for (int i = 0; i < na; i++) {
            int best = -32768 * 64, second = best, jb = -1;
            for (int j = 0; j < nb; j++) {
                int d = dot64_i8(a->desc[i], b->desc[j]);
                if (d > best) { second = best; best = d; jb = j; }
                else if (d > second) second = d;
            }
            if (best >= XF_MIN_DOT && best - XF_MARGIN >= second) {
                if (pairs && n < max_pairs) {
                    pairs[n][0] = i;
                    pairs[n][1] = jb;
                }
                n++;
            }
        }
    } else {
        for (int i = 0; i < na; i++) {
            int best = 999, second = 999, jb = -1;
            for (int j = 0; j < nb; j++) {
                int d = hamming256(a->desc[i], b->desc[j]);
                if (d < best) { second = best; best = d; jb = j; }
                else if (d < second) second = d;
            }
            if (best <= ORB_MAX_DIST && best + ORB_MARGIN <= second) {
                if (pairs && n < max_pairs) {
                    pairs[n][0] = i;
                    pairs[n][1] = jb;
                }
                n++;
            }
        }
    }
    return n;
}

static int match_pairs(const xr_kf *a, const xr_kf *b,
                       int (*pairs)[2], int max_pairs) {
    return match_pairs_lim(a, a->n_kp, b, b->n_kp, pairs, max_pairs);
}

static int match_count(const xr_kf *a, const xr_kf *b) {
    return match_pairs(a, b, NULL, 0);
}

/* leading keypoints are the landmark-anchored ones (extraction order) */
static int anchored_count(const xr_kf *k) {
    int n = 0;
    while (n < k->n_kp && k->lm_of_kp[n] >= 0) n++;
    return n;
}

/* ---- loop verification: gravity-aligned PnP relocalization ---------------------- */

/* The ORB-SLAM relocalization pattern: the MAP supplies the 3D (the
 * stored keyframe's landmarks), the query supplies ONLY descriptors and
 * pixels — a query frame's own fresh triangulations are garbage in
 * exactly the situations relocalization exists for. Both odom worlds
 * are gravity-aligned (z-up) and pitch/roll are gravity-observable, so
 * the unknown reduces to yaw + translation: TWO correspondences solve
 * it in closed form, RANSAC picks the consensus, a linear pass refines
 * the camera center and yaw. */
#define VER_MIN_PAIRS 8            /* 2D-3D matches needed to try */
/* bearing inlier tolerance ~4 deg. 2.5 deg was too tight: mini-ORB
 * keypoint noise (~0.5 deg) PLUS the map point's own inverse-depth
 * position error (a near point off by the 0.35 m cache tolerance is
 * degrees of bearing from a metre away) left genuine revisits one or two
 * inliers short of the floor (the on-device "20 pairs, 7 inliers" near
 * miss). The absolute floor (VER_MIN_PAIRS inliers) still rejects noise. */
#define VER_COS_TOL 0.99756f       /* ~4 deg */
#define VER_MIN_RANGE_M 0.3f
#define VER_MAX_RANGE_M 6.0f       /* near points have trustier depth */
#define VER_ITERS 120
#define VER_MAX_T_M 3.5f           /* a shake can drift more than 2.5 m */
#define VER_MAX_ANG_RAD 0.61f      /* ~35 deg */

/* overwhelming place-recognition evidence permits a bigger snap: after a
 * violent shake the pose error can exceed the normal caps (especially in
 * yaw), while the map is CERTAIN it is the same scene */
#define VER_STRONG_MATCHES 33      /* ~1.5x the candidate threshold */
#define VER_STRONG_INLIERS 11
#define VER_JUMP_T_M 8.0f
#define VER_JUMP_ANG_RAD 1.4f      /* ~80 deg */

/* left-camera geometry, wired in from the SLAM bridge */
static struct {
    int have;
    int (*unproject)(float u, float v, float ray_cam[3]);
    float R_ic[9], p_ic[3];        /* camera -> IMU/body */
} GEOM;

void xr_map_set_geom(int (*unproject)(float, float, float[3]),
                     const float R_ic[9], const float p_ic[3]) {
    GEOM.unproject = unproject;
    memcpy(GEOM.R_ic, R_ic, sizeof GEOM.R_ic);
    memcpy(GEOM.p_ic, p_ic, sizeof GEOM.p_ic);
    GEOM.have = 1;
    LOGI("session map: PnP geometry wired (kb4 + extrinsics)");
}

/* count bearing inliers of the model (yaw, C) */
static int pnp2_inliers(const float (*s)[3], const float (*P)[3], int n,
                        float yaw, const float C[3]) {
    float cy = cosf(yaw), sy = sinf(yaw);
    int cnt = 0;
    for (int m = 0; m < n; m++) {
        float qx = P[m][0] - C[0], qy = P[m][1] - C[1], qz = P[m][2] - C[2];
        float nq = sqrtf(qx * qx + qy * qy + qz * qz);
        if (nq < VER_MIN_RANGE_M) continue;
        float rx = cy * s[m][0] - sy * s[m][1];
        float ry = sy * s[m][0] + cy * s[m][1];
        float dot = (qx * rx + qy * ry + qz * s[m][2]) / nq;
        if (dot > VER_COS_TOL) cnt++;
    }
    return cnt;
}

/* Gravity-aligned 2-point PnP RANSAC. s[i] = unit bearing of query kp i
 * rotated into the CURRENT-odom world (pre-yaw-correction), P[i] = the
 * matched map point (kf-odom world). Solves Rz(dyaw) and the camera
 * center C so that P[i] - C is parallel to Rz*s[i]. Returns the inlier
 * count (0 = no acceptable model). */
static int pnp2_ransac(const float (*s)[3], const float (*P)[3], int n,
                       float Rz_out[9], float C_out[3]) {
    if (n < 2) return 0;
    int best_in = 0;
    float b_yaw = 0, bC[3] = { 0, 0, 0 };
    uint32_t seed = 0x51ED270Bu ^ (uint32_t)n;
    for (int it = 0; it < VER_ITERS; it++) {
        seed = seed * 1664525u + 1013904223u;
        int i0 = (int)(seed % (uint32_t)n);
        seed = seed * 1664525u + 1013904223u;
        int i1 = (int)(seed % (uint32_t)n);
        if (i0 == i1) continue;
        float dx = P[i0][0] - P[i1][0];
        float dy = P[i0][1] - P[i1][1];
        float dz = P[i0][2] - P[i1][2];
        float Q = dx * dx + dy * dy;           /* |P01_xy|^2 */
        float s0z = s[i0][2], s1z = s[i1][2];
        if (fabsf(s0z) < 1e-3f) continue;      /* degenerate for the z eq */
        /* d0 = alpha + beta*d1 from the z (gravity) equation; the
         * horizontal-norm equation is then a quadratic in d1 (Rz drops
         * out of both: it preserves z and horizontal norms) */
        float A0 = s[i0][0] * s[i0][0] + s[i0][1] * s[i0][1];
        float A1 = s[i1][0] * s[i1][0] + s[i1][1] * s[i1][1];
        float Bd = s[i0][0] * s[i1][0] + s[i0][1] * s[i1][1];
        float alpha = dz / s0z, beta = s1z / s0z;
        float qa2 = beta * beta * A0 - 2 * beta * Bd + A1;
        float qb = 2 * alpha * beta * A0 - 2 * alpha * Bd;
        float qc = alpha * alpha * A0 - Q;
        float roots[2];
        int nroots = 0;
        if (fabsf(qa2) < 1e-9f) {
            if (fabsf(qb) > 1e-9f) roots[nroots++] = -qc / qb;
        } else {
            float disc = qb * qb - 4 * qa2 * qc;
            if (disc < 0) continue;
            float sq = sqrtf(disc);
            roots[nroots++] = (-qb + sq) / (2 * qa2);
            roots[nroots++] = (-qb - sq) / (2 * qa2);
        }
        for (int r = 0; r < nroots; r++) {
            float d1 = roots[r];
            float d0 = alpha + beta * d1;
            if (d0 < VER_MIN_RANGE_M || d1 < VER_MIN_RANGE_M ||
                d0 > 40.0f || d1 > 40.0f)
                continue;
            /* yaw aligns the horizontal pair vector */
            float ux = d0 * s[i0][0] - d1 * s[i1][0];
            float uy = d0 * s[i0][1] - d1 * s[i1][1];
            if (ux * ux + uy * uy < 1e-6f) continue;
            float yaw = atan2f(ux * dy - uy * dx, ux * dx + uy * dy);
            float cy = cosf(yaw), sy = sinf(yaw);
            float C[3] = {
                P[i0][0] - d0 * (cy * s[i0][0] - sy * s[i0][1]),
                P[i0][1] - d0 * (sy * s[i0][0] + cy * s[i0][1]),
                P[i0][2] - d0 * s0z,
            };
            int in = pnp2_inliers(s, P, n, yaw, C);
            if (in > best_in) {
                best_in = in;
                b_yaw = yaw;
                memcpy(bC, C, sizeof bC);
            }
        }
    }
    if (best_in < 3) return 0;

    /* refine: two rounds of (inliers -> linear camera center -> weighted
     * mean yaw residual) */
    float yaw = b_yaw, C[3];
    memcpy(C, bC, sizeof C);
    for (int round = 0; round < 2; round++) {
        float cy = cosf(yaw), sy = sinf(yaw);
        float M[9] = { 0 }, b[3] = { 0 };
        float dsum = 0, wsum = 0;
        int cnt = 0;
        for (int m = 0; m < n; m++) {
            float qx = P[m][0] - C[0], qy = P[m][1] - C[1],
                  qz = P[m][2] - C[2];
            float nq = sqrtf(qx * qx + qy * qy + qz * qz);
            if (nq < VER_MIN_RANGE_M) continue;
            float rx = cy * s[m][0] - sy * s[m][1];
            float ry = sy * s[m][0] + cy * s[m][1];
            float rz = s[m][2];
            float dot = (qx * rx + qy * ry + qz * rz) / nq;
            if (dot <= VER_COS_TOL) continue;
            cnt++;
            /* sum (I - r r^T): rays constrain C perpendicular to them */
            float II[9] = { 1 - rx * rx, -rx * ry, -rx * rz,
                            -rx * ry, 1 - ry * ry, -ry * rz,
                            -rx * rz, -ry * rz, 1 - rz * rz };
            for (int k = 0; k < 9; k++) M[k] += II[k];
            b[0] += II[0] * P[m][0] + II[1] * P[m][1] + II[2] * P[m][2];
            b[1] += II[3] * P[m][0] + II[4] * P[m][1] + II[5] * P[m][2];
            b[2] += II[6] * P[m][0] + II[7] * P[m][1] + II[8] * P[m][2];
            /* yaw residual, weighted by horizontal reach */
            float cross = rx * qy - ry * qx;
            float dotxy = rx * qx + ry * qy;
            float w = sqrtf(qx * qx + qy * qy);
            dsum += w * atan2f(cross, dotxy);
            wsum += w;
        }
        if (cnt < 3) break;
        float det = M[0] * (M[4] * M[8] - M[5] * M[7]) -
                    M[1] * (M[3] * M[8] - M[5] * M[6]) +
                    M[2] * (M[3] * M[7] - M[4] * M[6]);
        if (fabsf(det) > 1e-6f) {
            C[0] = ((M[4] * M[8] - M[5] * M[7]) * b[0] +
                    (M[2] * M[7] - M[1] * M[8]) * b[1] +
                    (M[1] * M[5] - M[2] * M[4]) * b[2]) / det;
            C[1] = ((M[5] * M[6] - M[3] * M[8]) * b[0] +
                    (M[0] * M[8] - M[2] * M[6]) * b[1] +
                    (M[2] * M[3] - M[0] * M[5]) * b[2]) / det;
            C[2] = ((M[3] * M[7] - M[4] * M[6]) * b[0] +
                    (M[1] * M[6] - M[0] * M[7]) * b[1] +
                    (M[0] * M[4] - M[1] * M[3]) * b[2]) / det;
        }
        if (wsum > 1e-3f) yaw += dsum / wsum;
    }
    int in = pnp2_inliers(s, P, n, yaw, C);
    if (in < best_in) {                        /* refinement went sour */
        yaw = b_yaw;
        memcpy(C, bC, sizeof C);
        in = best_in;
    }
    float cy = cosf(yaw), sy = sinf(yaw);
    Rz_out[0] = cy; Rz_out[1] = -sy; Rz_out[2] = 0;
    Rz_out[3] = sy; Rz_out[4] = cy;  Rz_out[5] = 0;
    Rz_out[6] = 0;  Rz_out[7] = 0;   Rz_out[8] = 1;
    memcpy(C_out, C, 3 * sizeof(float));
    return in;
}

/* ---- covisibility-pooled relocalization ----------------------------------------- */

#define COVIS_R_M 3.0f             /* neighbours within this join the pool */
#define COVIS_MAX_PAIRS (XR_MAP_KP_PER_KF * 3)

/* Relocalize the query against the matched keyframe AND its spatial
 * neighbours pooled together (the covisibility idea, cheap: one extra
 * descriptor match per nearby keyframe). Every physical landmark
 * contributes ONE correspondence — its session-frame position averaged
 * across the keyframes that saw it, which cancels the single-view
 * inverse-depth noise that starved the per-keyframe PnP. Fills D
 * (odom -> session) and the inlier counts; returns 1 on a solved pose. */
static int reloc_pnp(const xr_kf *w, int best_i, float Dq[4], float Dp[3],
                     int *out_n3, int *out_nin, int *out_covis) {
    *out_n3 = 0;
    *out_nin = 0;
    *out_covis = 0;
    if (!GEOM.have) return 0;

    static float Sw[COVIS_MAX_PAIRS][3];   /* query bearing, current-odom */
    static float Pw[COVIS_MAX_PAIRS][3];   /* map point, session frame */
    static int32_t pid[COVIS_MAX_PAIRS];   /* physical landmark id */
    static int pcnt[COVIS_MAX_PAIRS];
    static int prs[XR_MAP_KP_PER_KF][2];
    int n = 0;

    float Rq[9];
    q2R(w->q, Rq);                          /* body -> current odom */

    for (int s = 0; s < KF_N; s++) {
        if (s != best_i) {                  /* covisible neighbourhood */
            float dx = KF[s].p[0] - KF[best_i].p[0];
            float dy = KF[s].p[1] - KF[best_i].p[1];
            float dz = KF[s].p[2] - KF[best_i].p[2];
            if (dx * dx + dy * dy + dz * dz > COVIS_R_M * COVIS_R_M) continue;
            if (KF[s].desc_type != w->desc_type) continue;
        }
        /* D_kf = C_kf ∘ O_kf⁻¹ maps this kf's landmarks into session */
        float qoi[4], poi[3], qd[4], pd[3];
        pose_invert(KF[s].q, KF[s].p, qoi, poi);
        pose_compose(KF[s].qc, KF[s].pc, qoi, poi, qd, pd);

        int nb = KF[s].desc_type == DESC_ORB ? anchored_count(&KF[s])
                                             : KF[s].n_kp;
        int np = match_pairs_lim(w, w->n_kp, &KF[s], nb, prs,
                                 XR_MAP_KP_PER_KF);
        int contributed = 0;
        for (int m = 0; m < np; m++) {
            int lk = KF[s].lm_of_kp[prs[m][1]];
            if (lk < 0) continue;
            float dk2 = 0;
            for (int c = 0; c < 3; c++) {
                float d = KF[s].lm_xyz[lk][c] - KF[s].p[c];
                dk2 += d * d;
            }
            if (dk2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M) continue;
            contributed = 1;

            float ps[3];                    /* landmark -> session */
            qrotv(qd, KF[s].lm_xyz[lk], ps);
            ps[0] += pd[0]; ps[1] += pd[1]; ps[2] += pd[2];

            int32_t id = KF[s].lm_id[lk];
            int found = -1;
            for (int k = 0; k < n; k++)
                if (pid[k] == id) { found = k; break; }
            if (found >= 0) {               /* average this landmark */
                float c1 = (float)pcnt[found];
                for (int c = 0; c < 3; c++)
                    Pw[found][c] = (Pw[found][c] * c1 + ps[c]) / (c1 + 1);
                pcnt[found]++;
                continue;
            }
            if (n >= COVIS_MAX_PAIRS) continue;
            float rc[3], rb[3];
            if (GEOM.unproject(w->kp_uv[prs[m][0]][0],
                               w->kp_uv[prs[m][0]][1], rc))
                continue;
            rb[0] = GEOM.R_ic[0] * rc[0] + GEOM.R_ic[1] * rc[1] +
                    GEOM.R_ic[2] * rc[2];
            rb[1] = GEOM.R_ic[3] * rc[0] + GEOM.R_ic[4] * rc[1] +
                    GEOM.R_ic[5] * rc[2];
            rb[2] = GEOM.R_ic[6] * rc[0] + GEOM.R_ic[7] * rc[1] +
                    GEOM.R_ic[8] * rc[2];
            Sw[n][0] = Rq[0] * rb[0] + Rq[1] * rb[1] + Rq[2] * rb[2];
            Sw[n][1] = Rq[3] * rb[0] + Rq[4] * rb[1] + Rq[5] * rb[2];
            Sw[n][2] = Rq[6] * rb[0] + Rq[7] * rb[1] + Rq[8] * rb[2];
            memcpy(Pw[n], ps, sizeof ps);
            pid[n] = id;
            pcnt[n] = 1;
            n++;
        }
        if (contributed) (*out_covis)++;
    }
    *out_n3 = n;
    if (n < VER_MIN_PAIRS) return 0;

    float Rz[9], C[3];
    int nin = pnp2_ransac(Sw, Pw, n, Rz, C);
    *out_nin = nin;
    if (nin < VER_MIN_PAIRS || nin * 100 < 33 * n) return 0;

    /* D (odom -> session): rotation = the yaw Rz; translation places the
     * query body at the recovered camera centre C (minus the lever arm) */
    R2q(Rz, Dq);
    float qsb[4], t3[3], body_s[3];
    qmul(Dq, w->q, qsb);                    /* body -> session */
    qrotv(qsb, GEOM.p_ic, t3);
    body_s[0] = C[0] - t3[0];
    body_s[1] = C[1] - t3[1];
    body_s[2] = C[2] - t3[2];
    qrotv(Dq, w->p, t3);
    Dp[0] = body_s[0] - t3[0];
    Dp[1] = body_s[1] - t3[1];
    Dp[2] = body_s[2] - t3[2];
    return 1;
}

/* ---- map thread ----------------------------------------------------------------- */

static void process_keyframe(void) {
    /* snapshot the mailbox without holding the lock during the heavy work */
    static xr_kf work;                      /* map thread only */
    static uint8_t img[XR_OW * XR_OH];
    pthread_mutex_lock(&MAP_LOCK);
    int q_only = MBOX.query_only;
    work.ts = MBOX.ts;
    memcpy(work.q, MBOX.q, sizeof work.q);
    memcpy(work.p, MBOX.p, sizeof work.p);
    memcpy(img, MBOX.img, sizeof img);
    work.n_lm = MBOX.n_lm;
    memcpy(work.lm_id, MBOX.lm_id, sizeof(int32_t) * (size_t)work.n_lm);
    memcpy(work.lm_xyz, MBOX.lm_xyz, sizeof(float) * 3 * (size_t)work.n_lm);
    memcpy(work.lm_uv, MBOX.lm_uv, sizeof(float) * 2 * (size_t)work.n_lm);
    MBOX.full = 0;
    pthread_mutex_unlock(&MAP_LOCK);

    int mapping = atomic_load(&MAPPING);
    /* rate gates (map-thread-only timing). When this offer will neither
     * search for loops nor store a keyframe, skip ALL the heavy work
     * (descriptor extraction + matching + PnP). This is what keeps the
     * map thread from monopolising a core and starving Basalt. */
    static uint64_t last_search_ns;
    int do_search = last_search_ns == 0 ||
                    work.ts - last_search_ns >= LOOP_SEARCH_INTERVAL_NS;
    int may_store = mapping && !q_only && work.n_lm >= STORE_MIN_LM &&
                    work.ts - LAST_STORE_NS >= STORE_MIN_INTERVAL_NS;
    if (!do_search && !may_store)
        return;
    if (do_search) last_search_ns = work.ts;

    /* descriptors: XFeat when selected AND available, mini-ORB otherwise */
    if (atomic_load(&USE_XFEAT) && MODEL_PATH[0]) xr_xfeat_init(MODEL_PATH);
    if (atomic_load(&USE_XFEAT) && xr_xfeat_available()) {
        int n = xr_xfeat_extract(img, work.kp_uv, work.desc,
                                 XR_MAP_KP_PER_KF);
        if (n >= 0) {
            work.n_kp = n;
            work.desc_type = DESC_XFEAT;
            /* associate XFeat keypoints to landmarks by proximity (the
             * ORB path anchors descriptors AT the landmarks instead) */
            for (int j = 0; j < work.n_kp; j++) {
                work.lm_of_kp[j] = -1;
                float bd = 16.0f;              /* 4 px */
                for (int i = 0; i < work.n_lm; i++) {
                    float du = work.lm_uv[i][0] - work.kp_uv[j][0];
                    float dv = work.lm_uv[i][1] - work.kp_uv[j][1];
                    float d = du * du + dv * dv;
                    if (d < bd) { bd = d; work.lm_of_kp[j] = i; }
                }
            }
        } else {
            orb_extract(img, &work);
        }
    } else {
        orb_extract(img, &work);
    }

    pthread_mutex_lock(&MAP_LOCK);
    memcpy(LAST_POSE.q, work.q, sizeof LAST_POSE.q);
    memcpy(LAST_POSE.p, work.p, sizeof LAST_POSE.p);
    LAST_POSE.have = 1;

    /* spatial recency: keyframes near the current position stay fresh, so
     * living in the same space never rolls its map away */
    for (int i = 0; i < KF_N; i++) {
        float dx = work.p[0] - KF[i].p[0], dy = work.p[1] - KF[i].p[1],
              dz = work.p[2] - KF[i].p[2];
        if (dx * dx + dy * dy + dz * dz < KF_NEAR_M * KF_NEAR_M)
            KF[i].last_used = work.ts;
    }

    int j_corrected = 0;      /* verified closure filled work.qc/pc */
    /* loop/reloc candidates: match the CURRENT view against the store
     * (runs in both modes — in localization-only this IS the reloc query).
     * In mapping mode skip the most recent keyframes (always similar).
     * Throttled (do_search) so it can't monopolise the CPU. */
    if (do_search) {
    int lim = KF_N - (mapping ? CAND_SKIP_RECENT : 0);
    int best_m = 0, best_i = -1;
    for (int i = 0; i < lim; i++) {
        int m = match_count(&work, &KF[i]);
        if (m > best_m) { best_m = m; best_i = i; }
    }
    if (best_i >= 0 && best_m < CAND_MIN_MATCHES) {
        /* below the candidate bar: say so occasionally, so a "nothing
         * happens" report can distinguish no-match from no-verify */
        static uint64_t nolog_ts;              /* map thread only */
        VER_LAST.pairs = 0;
        VER_LAST.inliers = 0;
        VER_LAST.outcome = VOUT_BELOW_BAR;
        if (work.ts - nolog_ts > 5000000000ull) {
            nolog_ts = work.ts;
            LOGI("session map: best %d matches (kf#%d, %d stored) — below "
                 "the %d candidate bar",
                 best_m, best_i, KF_N, CAND_MIN_MATCHES);
        }
    }
    if (best_i >= 0 && best_m >= CAND_MIN_MATCHES) {
        KF[best_i].last_used = work.ts;    /* matched = useful */
        LAST_CAND.a = work.ts;
        LAST_CAND.b = KF[best_i].ts;
        LAST_CAND.matches = best_m;
        LAST_CAND.have = 1;
        LOOP_STATS.count++;
        LOOP_STATS.matches = best_m;

        /* covisibility-pooled PnP relocalization -> D (odom -> session) */
        int n3 = 0, nin = 0, covis = 0;
        float Dq[4], Dp[3];
        int ok = reloc_pnp(&work, best_i, Dq, Dp, &n3, &nin, &covis);
        VER_LAST.pairs = n3;
        VER_LAST.inliers = nin;
        if (!ok || covis < COVIS_MIN_KF) {
            /* not verified, or an isolated (junk / shake-spawned) place */
            VER_LAST.outcome = (ok || n3 >= VER_MIN_PAIRS)
                                   ? VOUT_FEW_INLIERS : VOUT_FEW_PAIRS;
            LOGI("session map: %s kf#%d (%d matches, %s) unverified "
                 "(%d pairs, %d inliers, %d covis kfs)",
                 mapping ? "LOOP CANDIDATE vs" : "RELOC MATCH vs", best_i,
                 best_m, work.desc_type == DESC_XFEAT ? "xfeat" : "orb",
                 n3, nin, covis);
        } else {
            /* deviation = how far the VIO has strayed from the map (D vs
             * the live correction, at the current pose) */
            float ns[3], os[3];
            qrotv(Dq, work.p, ns);
            qrotv(CORR.q, work.p, os);
            float st2 = 0;
            for (int c = 0; c < 3; c++) {
                float d = (ns[c] + Dp[c]) - (os[c] + CORR.p[c]);
                st2 += d * d;
            }
            float qci[4], qe[4], rvv[3];
            qconj(CORR.q, qci);
            qmul(Dq, qci, qe);
            rv_from_q(qe, rvv);
            float sang = sqrtf(rvv[0] * rvv[0] + rvv[1] * rvv[1] +
                               rvv[2] * rvv[2]);
            float dev = sqrtf(st2);
            int strong = best_m >= VER_STRONG_MATCHES &&
                         nin >= VER_STRONG_INLIERS && nin * 100 >= 50 * n3;
            float mxt = strong ? VER_JUMP_T_M : VER_MAX_T_M;
            float mxa = strong ? VER_JUMP_ANG_RAD : VER_MAX_ANG_RAD;
            int significant = dev > SNAP_MIN_M || sang > SNAP_MIN_ANG_RAD;
            int cooled = work.ts - LAST_SNAP_NS > SNAP_COOLDOWN_NS;
            VER_LAST.outcome = VOUT_APPLIED;   /* verified = tracking the map */
            if (dev > mxt || sang > mxa) {
                VER_LAST.outcome = VOUT_CAPPED;
                LOGI("session map: kf#%d PnP GOOD (%d/%d inliers, %d covis) "
                     "but |t|=%.2fm ang=%.0fdeg exceeds caps — likely a "
                     "wrong-place match, ignored",
                     best_i, nin, n3, covis, (double)dev,
                     (double)(sang * 57.3f));
            } else if (significant && cooled && atomic_load(&RECOVERY)) {
                /* genuine drift recovery: snap the pose back to the map */
                memcpy(CORR.q, Dq, sizeof CORR.q);
                memcpy(CORR.p, Dp, sizeof CORR.p);
                CORR.gen++;
                LAST_SNAP_NS = work.ts;
                pose_compose(Dq, Dp, work.q, work.p, work.qc, work.pc);
                j_corrected = 1;
                LOGI("session map: %s kf#%d RECOVERY SNAP %.2fm %.0fdeg "
                     "(%d/%d inliers, %d covis, gen %d)",
                     mapping ? "LOOP" : "RELOC", best_i, (double)dev,
                     (double)(sang * 57.3f), nin, n3, covis, CORR.gen);
            }
            /* below the deviation gate: the VIO agrees with the map, do
             * nothing (this is a return-on-drift recovery, not a clamp) */
        }

        /* AR flash + panel marker: matched kf's landmarks in session */
        {
            float qoi[4], poi[3], qdi[4], pdi[3];
            pose_invert(KF[best_i].q, KF[best_i].p, qoi, poi);
            pose_compose(KF[best_i].qc, KF[best_i].pc, qoi, poi, qdi, pdi);
            memcpy(LOOP_STATS.pos, KF[best_i].pc, sizeof LOOP_STATS.pos);
            LOOP_STATS.n_lm = KF[best_i].n_lm;
            for (int i = 0; i < KF[best_i].n_lm; i++) {
                float t[3];
                qrotv(qdi, KF[best_i].lm_xyz[i], t);
                LOOP_STATS.lm[i][0] = t[0] + pdi[0];
                LOOP_STATS.lm[i][1] = t[1] + pdi[1];
                LOOP_STATS.lm[i][2] = t[2] + pdi[2];
            }
        }
    }
    }   /* end throttled loop search (do_search) */

    /* store (mapping mode only; never for a stationary query — those are
     * matching-only; only frames that carry verifiable geometry, at a
     * bounded rate), rolling cap: evict least-recently-useful */
    if (mapping && !q_only && work.n_lm >= STORE_MIN_LM &&
        work.ts - LAST_STORE_NS >= STORE_MIN_INTERVAL_NS) {
        LAST_STORE_NS = work.ts;
        if (KF_N == XR_MAP_MAX_KF) {
            int victim = 0;
            for (int i = 1; i < KF_N; i++)
                if (KF[i].last_used < KF[victim].last_used) victim = i;
            memmove(&KF[victim], &KF[victim + 1],
                    sizeof(xr_kf) * (size_t)(KF_N - 1 - victim));
            KF_N--;
        }
        work.last_used = work.ts;
        if (!j_corrected)     /* corrected = D_cur ∘ O_j */
            pose_compose(CORR.q, CORR.p, work.q, work.p, work.qc, work.pc);
        KF[KF_N] = work;
        KF_N++;
        atomic_store(&KF_COUNT_PUB, KF_N);
        LOGI("session map: kf#%d stored (%d landmarks, %d kps)",
             KF_N - 1, work.n_lm, work.n_kp);
    }
    pthread_mutex_unlock(&MAP_LOCK);
}

static void *map_thread(void *arg) {
    (void)arg;
    setpriority(PRIO_PROCESS, (id_t)gettid(), 19);   /* never outrank VIO */
    pthread_mutex_lock(&MAP_LOCK);
    for (;;) {
        while (!MBOX.full) pthread_cond_wait(&MAP_COND, &MAP_LOCK);
        pthread_mutex_unlock(&MAP_LOCK);
        process_keyframe();
        pthread_mutex_lock(&MAP_LOCK);
    }
    return NULL;
}

static void thread_start(void) {
    pthread_t t;
    pthread_create(&t, NULL, map_thread, NULL);
    pthread_detach(t);
}

void xr_map_offer(const float q[4], const float p[3], uint64_t ts_ns,
                  const uint8_t *img,
                  const int32_t *lm_id, const float (*lm_xyz)[3],
                  const float (*lm_uv)[2], int n_lm) {
    pthread_once(&THREAD_ONCE, thread_start);
    pthread_mutex_lock(&MAP_LOCK);
    if (MBOX.full) {                       /* map thread busy: drop */
        pthread_mutex_unlock(&MAP_LOCK);
        return;
    }
    int query_only = 0;
    if (LAST_POSE.have) {
        float dx = p[0] - LAST_POSE.p[0], dy = p[1] - LAST_POSE.p[1],
              dz = p[2] - LAST_POSE.p[2];
        float qd = fabsf(q[0] * LAST_POSE.q[0] + q[1] * LAST_POSE.q[1] +
                         q[2] * LAST_POSE.q[2] + q[3] * LAST_POSE.q[3]);
        if (dx * dx + dy * dy + dz * dz < KF_DIST_M * KF_DIST_M &&
            qd > KF_ANGLE_COS) {
            /* stationary: keep the reloc query alive on a slow cadence */
            if (ts_ns - LAST_ACCEPT_NS < QUERY_INTERVAL_NS) {
                pthread_mutex_unlock(&MAP_LOCK);
                return;
            }
            query_only = 1;
        }
    }
    LAST_ACCEPT_NS = ts_ns;
    MBOX.query_only = query_only;
    MBOX.ts = ts_ns;
    memcpy(MBOX.q, q, sizeof MBOX.q);
    memcpy(MBOX.p, p, sizeof MBOX.p);
    memcpy(MBOX.img, img, sizeof MBOX.img);
    if (n_lm > XR_MAP_KP_PER_KF) n_lm = XR_MAP_KP_PER_KF;
    MBOX.n_lm = n_lm;
    memcpy(MBOX.lm_id, lm_id, sizeof(int32_t) * (size_t)n_lm);
    memcpy(MBOX.lm_xyz, lm_xyz, sizeof(float) * 3 * (size_t)n_lm);
    memcpy(MBOX.lm_uv, lm_uv, sizeof(float) * 2 * (size_t)n_lm);
    MBOX.full = 1;
    /* provisional gate anchor so a slow map thread doesn't cause a burst
     * of near-identical offers */
    memcpy(LAST_POSE.q, q, sizeof LAST_POSE.q);
    memcpy(LAST_POSE.p, p, sizeof LAST_POSE.p);
    LAST_POSE.have = 1;
    pthread_cond_signal(&MAP_COND);
    pthread_mutex_unlock(&MAP_LOCK);
}
