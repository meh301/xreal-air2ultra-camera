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
static char MODEL_PATH[512];
static pthread_once_t THREAD_ONCE = PTHREAD_ONCE_INIT;
static atomic_int MAPPING = 1;
static atomic_int RECOVERY = 1;        /* live-pose snap (loop closure of
                                          the MAP itself is always on) */

/* Live session correction D = C_newest ∘ O_newest⁻¹ (map -> odom pattern):
 * composes onto every odom-frame quantity leaving the SLAM worker. Updated
 * by verified loop closures; identity until the first one. */
static struct { float q[4]; float p[3]; int gen; } CORR = { .q = {1, 0, 0, 0} };

void xr_map_set_model(const char *onnx_path) {
    if (!onnx_path) return;
    strncpy(MODEL_PATH, onnx_path, sizeof MODEL_PATH - 1);
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

void xr_map_reset(void) {
    pthread_mutex_lock(&MAP_LOCK);
    KF_N = 0;
    LAST_CAND.have = 0;
    LAST_POSE.have = 0;
    MBOX.full = 0;
    LAST_ACCEPT_NS = 0;
    memset(&LOOP_STATS, 0, sizeof LOOP_STATS);
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

/* ratio-tested matching; when pairs != NULL, the kp index pairs of up to
 * max_pairs matches are stored as well. Returns the total match count. */
static int match_pairs(const xr_kf *a, const xr_kf *b,
                       int (*pairs)[2], int max_pairs) {
    if (a->desc_type != b->desc_type) return 0;
    int n = 0;
    if (a->desc_type == DESC_XFEAT) {
        for (int i = 0; i < a->n_kp; i++) {
            int best = -32768 * 64, second = best, jb = -1;
            for (int j = 0; j < b->n_kp; j++) {
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
        for (int i = 0; i < a->n_kp; i++) {
            int best = 999, second = 999, jb = -1;
            for (int j = 0; j < b->n_kp; j++) {
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

static int match_count(const xr_kf *a, const xr_kf *b) {
    return match_pairs(a, b, NULL, 0);
}

/* ---- loop verification: rigid 3D-3D alignment ----------------------------------- */

#define VER_MIN_3D 8               /* landmark pairs needed to try */
#define VER_INLIER_M 0.12f
#define VER_MAX_RANGE_M 8.0f       /* inverse-depth points are noisy far out */
#define VER_ITERS 200
#define VER_MAX_T_M 2.5f           /* wilder alignment = wrong place */
#define VER_MAX_ANG_RAD 0.44f      /* ~25 deg */

/* overwhelming place-recognition evidence permits a bigger snap: after a
 * violent shake the pose error can exceed the normal caps (especially in
 * yaw), while the map is CERTAIN it is the same scene — the post-shake
 * "it knows where it is but does not snap" failure */
#define VER_STRONG_MATCHES 33      /* ~1.5x the candidate threshold */
#define VER_STRONG_INLIERS 12
#define VER_JUMP_T_M 6.0f
#define VER_JUMP_ANG_RAD 1.05f     /* ~60 deg */

/* alignment A = (R, t) with pk ≈ R*pq + t: RANSAC over point triads, then
 * a Horn (quaternion) refit on the inliers via power iteration. Returns
 * the inlier count (0 = no acceptable model). */
static int align_ransac(const float (*pq)[3], const float (*pk)[3], int n,
                        float R_out[9], float t_out[3]) {
    if (n < 3) return 0;
    int best_in = 0;
    float bR[9], bt[3];
    uint32_t seed = 0x9E3779B9u ^ (uint32_t)n;
    for (int it = 0; it < VER_ITERS; it++) {
        seed = seed * 1664525u + 1013904223u;
        int i0 = (int)(seed % (uint32_t)n);
        seed = seed * 1664525u + 1013904223u;
        int i1 = (int)(seed % (uint32_t)n);
        seed = seed * 1664525u + 1013904223u;
        int i2 = (int)(seed % (uint32_t)n);
        if (i0 == i1 || i1 == i2 || i0 == i2) continue;
        /* orthonormal triads over both triangles -> closed-form R */
        float Bq[9], Bk[9];
        const float (*P)[3];
        float *B;
        int deg = 0;
        for (int s = 0; s < 2; s++) {
            P = s ? pk : pq;
            B = s ? Bk : Bq;
            float e1[3] = { P[i1][0] - P[i0][0], P[i1][1] - P[i0][1],
                            P[i1][2] - P[i0][2] };
            float e2[3] = { P[i2][0] - P[i0][0], P[i2][1] - P[i0][1],
                            P[i2][2] - P[i0][2] };
            float l1 = sqrtf(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
            if (l1 < 0.05f) { deg = 1; break; }
            e1[0] /= l1; e1[1] /= l1; e1[2] /= l1;
            float nx = e1[1] * e2[2] - e1[2] * e2[1];
            float ny = e1[2] * e2[0] - e1[0] * e2[2];
            float nz = e1[0] * e2[1] - e1[1] * e2[0];
            float ln = sqrtf(nx * nx + ny * ny + nz * nz);
            if (ln < 0.02f) { deg = 1; break; }   /* collinear */
            nx /= ln; ny /= ln; nz /= ln;
            float u2x = ny * e1[2] - nz * e1[1];
            float u2y = nz * e1[0] - nx * e1[2];
            float u2z = nx * e1[1] - ny * e1[0];
            B[0] = e1[0]; B[3] = e1[1]; B[6] = e1[2];   /* columns */
            B[1] = u2x;   B[4] = u2y;   B[7] = u2z;
            B[2] = nx;    B[5] = ny;    B[8] = nz;
        }
        if (deg) continue;
        float R[9];                                   /* R = Bk * Bq^T */
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                R[r * 3 + c] = Bk[r * 3] * Bq[c * 3] +
                               Bk[r * 3 + 1] * Bq[c * 3 + 1] +
                               Bk[r * 3 + 2] * Bq[c * 3 + 2];
        float t[3];
        for (int c = 0; c < 3; c++)
            t[c] = pk[i0][c] - (R[c * 3] * pq[i0][0] +
                                R[c * 3 + 1] * pq[i0][1] +
                                R[c * 3 + 2] * pq[i0][2]);
        int in = 0;
        for (int m = 0; m < n; m++) {
            float e = 0;
            for (int c = 0; c < 3; c++) {
                float d = pk[m][c] - (R[c * 3] * pq[m][0] +
                                      R[c * 3 + 1] * pq[m][1] +
                                      R[c * 3 + 2] * pq[m][2] + t[c]);
                e += d * d;
            }
            if (e < VER_INLIER_M * VER_INLIER_M) in++;
        }
        if (in > best_in) {
            best_in = in;
            memcpy(bR, R, sizeof bR);
            memcpy(bt, t, sizeof bt);
        }
    }
    if (best_in < 3) return 0;

    /* Horn refit on the inliers of the best model */
    float cq[3] = { 0, 0, 0 }, ck[3] = { 0, 0, 0 };
    int in = 0;
    static int keep[XR_MAP_KP_PER_KF];               /* map thread only */
    for (int m = 0; m < n; m++) {
        float e = 0;
        for (int c = 0; c < 3; c++) {
            float d = pk[m][c] - (bR[c * 3] * pq[m][0] +
                                  bR[c * 3 + 1] * pq[m][1] +
                                  bR[c * 3 + 2] * pq[m][2] + bt[c]);
            e += d * d;
        }
        if (e < VER_INLIER_M * VER_INLIER_M) {
            keep[in++] = m;
            for (int c = 0; c < 3; c++) {
                cq[c] += pq[m][c];
                ck[c] += pk[m][c];
            }
        }
    }
    for (int c = 0; c < 3; c++) { cq[c] /= (float)in; ck[c] /= (float)in; }
    float S[9] = { 0 };                               /* Σ q_c k_c^T */
    for (int m = 0; m < in; m++) {
        float aq[3], ak[3];
        for (int c = 0; c < 3; c++) {
            aq[c] = pq[keep[m]][c] - cq[c];
            ak[c] = pk[keep[m]][c] - ck[c];
        }
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                S[r * 3 + c] += aq[r] * ak[c];
    }
    float N[16] = {
        S[0] + S[4] + S[8], S[5] - S[7],        S[6] - S[2],        S[1] - S[3],
        S[5] - S[7],        S[0] - S[4] - S[8], S[1] + S[3],        S[2] + S[6],
        S[6] - S[2],        S[1] + S[3],       -S[0] + S[4] - S[8], S[5] + S[7],
        S[1] - S[3],        S[2] + S[6],        S[5] + S[7],       -S[0] - S[4] + S[8]
    };
    float fro = 0;
    for (int i = 0; i < 16; i++) fro += N[i] * N[i];
    float shift = sqrtf(fro) + 1e-6f;                 /* all eigenvalues > 0 */
    N[0] += shift; N[5] += shift; N[10] += shift; N[15] += shift;
    /* skewed start: a u-turn alignment (w = 0) is orthogonal to (1,0,0,0)
     * and pure power iteration would never leave that subspace */
    float v[4] = { 1, 0.02f, 0.017f, 0.013f };
    for (int it = 0; it < 80; it++) {
        float u[4];
        for (int r = 0; r < 4; r++)
            u[r] = N[r * 4] * v[0] + N[r * 4 + 1] * v[1] +
                   N[r * 4 + 2] * v[2] + N[r * 4 + 3] * v[3];
        float l = sqrtf(u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + u[3] * u[3]);
        if (l < 1e-12f) break;
        for (int r = 0; r < 4; r++) v[r] = u[r] / l;
    }
    q2R(v, R_out);                                    /* q rotates q -> k */
    for (int c = 0; c < 3; c++)
        t_out[c] = ck[c] - (R_out[c * 3] * cq[0] + R_out[c * 3 + 1] * cq[1] +
                            R_out[c * 3 + 2] * cq[2]);
    /* recount with the refined model */
    int in2 = 0;
    for (int m = 0; m < n; m++) {
        float e = 0;
        for (int c = 0; c < 3; c++) {
            float d = pk[m][c] - (R_out[c * 3] * pq[m][0] +
                                  R_out[c * 3 + 1] * pq[m][1] +
                                  R_out[c * 3 + 2] * pq[m][2] + t_out[c]);
            e += d * d;
        }
        if (e < VER_INLIER_M * VER_INLIER_M) in2++;
    }
    if (in2 < in) {                                   /* refit made it worse */
        memcpy(R_out, bR, sizeof bR);
        memcpy(t_out, bt, sizeof bt);
        return in;
    }
    return in2;
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

    /* descriptors: XFeat when available, mini-ORB otherwise */
    if (MODEL_PATH[0]) xr_xfeat_init(MODEL_PATH);
    if (xr_xfeat_available()) {
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

    int mapping = atomic_load(&MAPPING);
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

    /* loop/reloc candidates: match the CURRENT view against the store
     * (runs in both modes — in localization-only this IS the reloc query).
     * In mapping mode skip the most recent keyframes (always similar). */
    int lim = KF_N - (mapping ? CAND_SKIP_RECENT : 0);
    int best_m = 0, best_i = -1;
    for (int i = 0; i < lim; i++) {
        int m = match_count(&work, &KF[i]);
        if (m > best_m) { best_m = m; best_i = i; }
    }
    int j_corrected = 0;      /* verified closure filled work.qc/pc */
    if (best_i >= 0 && best_m >= CAND_MIN_MATCHES) {
        KF[best_i].last_used = work.ts;    /* matched = useful */
        LAST_CAND.a = work.ts;
        LAST_CAND.b = KF[best_i].ts;
        LAST_CAND.matches = best_m;
        LAST_CAND.have = 1;
        LOOP_STATS.count++;
        LOOP_STATS.matches = best_m;

        /* ---- geometric verification: rigid 3D-3D alignment over the
         * matched landmark pairs. A verified alignment A (query-odom ->
         * kf-capture-odom coords) becomes a pose-graph constraint. */
        int verified = 0, n3 = 0, nin = 0;
        float qa[4], ta[3];
        {
            static int prs[XR_MAP_KP_PER_KF][2];          /* map thread */
            static float Pq[XR_MAP_KP_PER_KF][3];
            static float Pk[XR_MAP_KP_PER_KF][3];
            int np = match_pairs(&work, &KF[best_i], prs, XR_MAP_KP_PER_KF);
            for (int m = 0; m < np; m++) {
                int lq = work.lm_of_kp[prs[m][0]];
                int lk = KF[best_i].lm_of_kp[prs[m][1]];
                if (lq < 0 || lk < 0) continue;
                float dq2 = 0, dk2 = 0;
                for (int c = 0; c < 3; c++) {
                    float a = work.lm_xyz[lq][c] - work.p[c];
                    float b = KF[best_i].lm_xyz[lk][c] - KF[best_i].p[c];
                    dq2 += a * a;
                    dk2 += b * b;
                }
                if (dq2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M ||
                    dk2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M)
                    continue;
                memcpy(Pq[n3], work.lm_xyz[lq], sizeof Pq[0]);
                memcpy(Pk[n3], KF[best_i].lm_xyz[lk], sizeof Pk[0]);
                n3++;
            }
            if (n3 >= VER_MIN_3D) {
                float Ra[9];
                nin = align_ransac(Pq, Pk, n3, Ra, ta);
                if (nin >= VER_MIN_3D && nin * 100 >= 35 * n3) {
                    float rv[3];
                    R2q(Ra, qa);
                    rv_from_q(qa, rv);
                    float ang = sqrtf(rv[0] * rv[0] + rv[1] * rv[1] +
                                      rv[2] * rv[2]);
                    float tn2 = ta[0] * ta[0] + ta[1] * ta[1] + ta[2] * ta[2];
                    /* overwhelming evidence -> the caps open up so a
                     * post-shake error can snap out in one closure */
                    int strong = best_m >= VER_STRONG_MATCHES &&
                                 nin >= VER_STRONG_INLIERS &&
                                 nin * 100 >= 45 * n3;
                    float mxt = strong ? VER_JUMP_T_M : VER_MAX_T_M;
                    float mxa = strong ? VER_JUMP_ANG_RAD : VER_MAX_ANG_RAD;
                    if (ang < mxa && tn2 < mxt * mxt)
                        verified = 1;
                    else
                        LOGI("session map: kf#%d alignment GOOD (%d/%d "
                             "inliers, %d matches) but |t|=%.2fm "
                             "ang=%.0fdeg exceeds the snap caps",
                             best_i, nin, n3, best_m,
                             sqrt((double)tn2), (double)(ang * 57.3f));
                }
            }
        }

        if (verified) {
            /* D_i = C_i ∘ O_i⁻¹ (node i's point-level correction);
             * session pose of the query: C_j = D_i ∘ A ∘ O_j */
            float qoi[4], poi[3], qdi[4], pdi[3];
            pose_invert(KF[best_i].q, KF[best_i].p, qoi, poi);
            pose_compose(KF[best_i].qc, KF[best_i].pc, qoi, poi, qdi, pdi);
            float qda[4], pda[3];                     /* D_i ∘ A */
            pose_compose(qdi, pdi, qa, ta, qda, pda);
            float qcj[4], pcj[3];                     /* C_j */
            pose_compose(qda, pda, work.q, work.p, qcj, pcj);

            if (mapping && KF_N > best_i + 1) {
                /* distribute the change over the chain (i .. newest] by
                 * cumulative odom path length — drift accrues with
                 * distance. Node i itself is the trusted anchor. */
                float qce[4], pce[3];                 /* current estimate */
                pose_compose(CORR.q, CORR.p, work.q, work.p, qce, pce);
                float qinv[4], pinv[3], qe[4], pe[3]; /* E = C_j ∘ est⁻¹ */
                pose_invert(qce, pce, qinv, pinv);
                pose_compose(qcj, pcj, qinv, pinv, qe, pe);
                float rve[3];
                rv_from_q(qe, rve);
                float total = 0;
                for (int k = best_i; k < KF_N - 1; k++) {
                    float d = 0;
                    for (int c = 0; c < 3; c++) {
                        float x = KF[k + 1].p[c] - KF[k].p[c];
                        d += x * x;
                    }
                    total += sqrtf(d);
                }
                float dlast = 0;
                for (int c = 0; c < 3; c++) {
                    float x = work.p[c] - KF[KF_N - 1].p[c];
                    dlast += x * x;
                }
                total += sqrtf(dlast);
                float cum = 0;
                for (int k = best_i + 1; k < KF_N; k++) {
                    float d = 0;
                    for (int c = 0; c < 3; c++) {
                        float x = KF[k].p[c] - KF[k - 1].p[c];
                        d += x * x;
                    }
                    cum += sqrtf(d);
                    float f = total > 0.01f
                                  ? cum / total
                                  : (float)(k - best_i) / (float)(KF_N - best_i);
                    float rvf[3] = { rve[0] * f, rve[1] * f, rve[2] * f };
                    float qef[4], pef[3] = { pe[0] * f, pe[1] * f, pe[2] * f };
                    q_from_rv(rvf, qef);
                    float qn[4], pn[3];               /* E^f ∘ C_k */
                    pose_compose(qef, pef, KF[k].qc, KF[k].pc, qn, pn);
                    memcpy(KF[k].qc, qn, sizeof qn);
                    memcpy(KF[k].pc, pn, sizeof pn);
                }
            }
            /* publish the live correction D = C_j ∘ O_j⁻¹ (in loc-only
             * mode this IS the relocalization of the displayed pose) —
             * unless recovery is toggled off (GNSS-fusion mode): the map
             * stays self-consistent, the live pose stays continuous */
            if (atomic_load(&RECOVERY)) {
                float qoj[4], poj[3];
                pose_invert(work.q, work.p, qoj, poj);
                pose_compose(qcj, pcj, qoj, poj, CORR.q, CORR.p);
                CORR.gen++;
            }
            memcpy(work.qc, qcj, sizeof work.qc);
            memcpy(work.pc, pcj, sizeof work.pc);
            j_corrected = 1;
            LOGI("session map: LOOP VERIFIED kf#%d: %d/%d 3D inliers, "
                 "|t|=%.2fm%s",
                 best_i, nin, n3,
                 sqrt((double)(ta[0] * ta[0] + ta[1] * ta[1] + ta[2] * ta[2])),
                 atomic_load(&RECOVERY) ? ", pose snapped"
                                        : " (recovery off: map-only)");
        } else {
            LOGI("session map: %s kf#%d (%d matches, %s, dt=%.1fs) "
                 "unverified (%d 3D pairs, %d inliers)",
                 mapping ? "LOOP CANDIDATE vs" : "RELOC MATCH vs", best_i,
                 best_m, work.desc_type == DESC_XFEAT ? "xfeat" : "orb",
                 (double)(work.ts - KF[best_i].ts) / 1e9, n3, nin);
        }

        /* AR flash + panel marker, in the SESSION frame: stored landmarks
         * through D_i = C_i ∘ O_i⁻¹ */
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

    /* store (mapping mode only; never for a stationary query — those are
     * matching-only), rolling cap: evict least-recently-useful */
    if (mapping && !q_only) {
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
    }
    pthread_mutex_unlock(&MAP_LOCK);
}

static void *map_thread(void *arg) {
    (void)arg;
    setpriority(PRIO_PROCESS, (id_t)gettid(), 15);   /* lowest of our threads */
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
