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

enum { DESC_ORB = 0, DESC_XFEAT = 1 };

typedef struct {
    uint64_t ts;
    uint64_t last_used;            /* rolling eviction: refreshed by loop
                                      matches and spatial proximity */
    float q[4], p[3];              /* odom pose at capture */
    int desc_type;
    int n_kp;
    float kp_uv[XR_MAP_KP_PER_KF][2];
    int8_t desc[XR_MAP_KP_PER_KF][64];   /* ORB uses the first 32 bytes */
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
    uint64_t ts;
    float q[4], p[3];
    uint8_t img[XR_OW * XR_OH];
    int n_lm;
    int32_t lm_id[XR_MAP_KP_PER_KF];
    float lm_xyz[XR_MAP_KP_PER_KF][3];
    float lm_uv[XR_MAP_KP_PER_KF][2];
} MBOX;
static struct { float q[4], p[3]; int have; } LAST_POSE;
static char MODEL_PATH[512];
static pthread_once_t THREAD_ONCE = PTHREAD_ONCE_INIT;
static atomic_int MAPPING = 1;

void xr_map_set_model(const char *onnx_path) {
    if (!onnx_path) return;
    strncpy(MODEL_PATH, onnx_path, sizeof MODEL_PATH - 1);
}

void xr_map_set_mapping(int on) {
    atomic_store(&MAPPING, on ? 1 : 0);
    LOGI("session map: %s", on ? "MAPPING" : "LOCALIZATION-ONLY (frozen)");
}

void xr_map_reset(void) {
    pthread_mutex_lock(&MAP_LOCK);
    KF_N = 0;
    LAST_CAND.have = 0;
    LAST_POSE.have = 0;
    MBOX.full = 0;
    memset(&LOOP_STATS, 0, sizeof LOOP_STATS);
    atomic_store(&KF_COUNT_PUB, 0);
    pthread_mutex_unlock(&MAP_LOCK);
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

static void orb_extract(const uint8_t *img, xr_kf *kf) {
    pat_init();
    kf->n_kp = 0;
    kf->desc_type = DESC_ORB;
    enum { GX = (XR_OW - 2 * MARGIN) / NMS_GRID,
           GY = (XR_OH - 2 * MARGIN) / NMS_GRID };
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
            int m10 = 0, m01 = 0;
            for (int dy = -7; dy <= 7; dy++)
                for (int dx = -7; dx <= 7; dx++) {
                    int v = img[(by + dy) * XR_OW + bx + dx];
                    m10 += dx * v;
                    m01 += dy * v;
                }
            float ang = atan2f((float)m01, (float)m10);
            float ca = cosf(ang), sa = sinf(ang);
            int i = kf->n_kp++;
            kf->kp_uv[i][0] = (float)bx;
            kf->kp_uv[i][1] = (float)by;
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
            memcpy(kf->desc[i], d, sizeof d);
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

static int match_count(const xr_kf *a, const xr_kf *b) {
    if (a->desc_type != b->desc_type) return 0;
    int n = 0;
    if (a->desc_type == DESC_XFEAT) {
        for (int i = 0; i < a->n_kp; i++) {
            int best = -32768 * 64, second = best;
            for (int j = 0; j < b->n_kp; j++) {
                int d = dot64_i8(a->desc[i], b->desc[j]);
                if (d > best) { second = best; best = d; }
                else if (d > second) second = d;
            }
            if (best >= XF_MIN_DOT && best - XF_MARGIN >= second) n++;
        }
    } else {
        for (int i = 0; i < a->n_kp; i++) {
            int best = 999, second = 999;
            for (int j = 0; j < b->n_kp; j++) {
                int d = hamming256(a->desc[i], b->desc[j]);
                if (d < best) { second = best; best = d; }
                else if (d < second) second = d;
            }
            if (best <= ORB_MAX_DIST && best + ORB_MARGIN <= second) n++;
        }
    }
    return n;
}

/* ---- map thread ----------------------------------------------------------------- */

static void process_keyframe(void) {
    /* snapshot the mailbox without holding the lock during the heavy work */
    static xr_kf work;                      /* map thread only */
    static uint8_t img[XR_OW * XR_OH];
    pthread_mutex_lock(&MAP_LOCK);
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
    if (best_i >= 0 && best_m >= CAND_MIN_MATCHES) {
        KF[best_i].last_used = work.ts;    /* matched = useful */
        LAST_CAND.a = work.ts;
        LAST_CAND.b = KF[best_i].ts;
        LAST_CAND.matches = best_m;
        LAST_CAND.have = 1;
        LOOP_STATS.count++;
        memcpy(LOOP_STATS.pos, KF[best_i].p, sizeof LOOP_STATS.pos);
        LOOP_STATS.matches = best_m;
        LOOP_STATS.n_lm = KF[best_i].n_lm;
        memcpy(LOOP_STATS.lm, KF[best_i].lm_xyz,
               sizeof(float) * 3u * (size_t)KF[best_i].n_lm);
        LOGI("session map: %s kf#%d (%d matches, %s, dt=%.1fs)",
             mapping ? "LOOP CANDIDATE vs" : "RELOC MATCH vs", best_i,
             best_m, work.desc_type == DESC_XFEAT ? "xfeat" : "orb",
             (double)(work.ts - KF[best_i].ts) / 1e9);
    }

    /* store (mapping mode only), rolling cap: evict least-recently-useful */
    if (mapping) {
        if (KF_N == XR_MAP_MAX_KF) {
            int victim = 0;
            for (int i = 1; i < KF_N; i++)
                if (KF[i].last_used < KF[victim].last_used) victim = i;
            memmove(&KF[victim], &KF[victim + 1],
                    sizeof(xr_kf) * (size_t)(KF_N - 1 - victim));
            KF_N--;
        }
        work.last_used = work.ts;
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
    if (LAST_POSE.have) {
        float dx = p[0] - LAST_POSE.p[0], dy = p[1] - LAST_POSE.p[1],
              dz = p[2] - LAST_POSE.p[2];
        float qd = fabsf(q[0] * LAST_POSE.q[0] + q[1] * LAST_POSE.q[1] +
                         q[2] * LAST_POSE.q[2] + q[3] * LAST_POSE.q[3]);
        if (dx * dx + dy * dy + dz * dz < KF_DIST_M * KF_DIST_M &&
            qd > KF_ANGLE_COS) {
            pthread_mutex_unlock(&MAP_LOCK);
            return;
        }
    }
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
