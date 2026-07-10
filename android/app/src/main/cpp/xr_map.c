/* xr_map.c — see xr_map.h.
 *
 * Descriptors: compact ORB-style — FAST-9 corners with grid NMS, intensity-
 * centroid orientation, rotated 256-bit BRIEF from a fixed seeded pattern.
 * Self-consistent within this system (session + fog use the same code), no
 * OpenCV compatibility needed. Matching = Hamming over 4x u64 (one popcount
 * chain per pair; brute force is fine at session scale — the koide3/GLIM
 * lesson applied here is the bounded incremental store, not fancy search).
 */
#include "xr_map.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#include <android/log.h>

#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

#define KF_DIST_M 0.30f            /* motion gates */
#define KF_ANGLE_COS 0.99144f      /* cos(15/2 deg) on the quat dot */
#define FAST_THRESH 18
#define NMS_GRID 24                /* px cell for corner NMS */
#define MARGIN 20                  /* keep patches inside the image */
#define MATCH_MAX_DIST 60          /* Hamming acceptance */
#define MATCH_MARGIN 10            /* best must beat 2nd by this */
#define CAND_MIN_MATCHES 25
#define CAND_SKIP_RECENT 5         /* neighbours always match; skip them */

typedef struct {
    uint64_t ts;
    float q[4], p[3];              /* odom pose at capture */
    int n_kp;
    uint16_t kp_u[XR_MAP_KP_PER_KF], kp_v[XR_MAP_KP_PER_KF];
    uint64_t desc[XR_MAP_KP_PER_KF][4];
    int n_lm;
    int32_t lm_id[XR_MAP_KP_PER_KF];
    float lm_xyz[XR_MAP_KP_PER_KF][3];
    float lm_uv[XR_MAP_KP_PER_KF][2];
} xr_kf;

static xr_kf KF[XR_MAP_MAX_KF];    /* .bss (zero static) */
static int KF_N;                   /* live count (compacted array) */
static int KF_SEQ;                 /* total ever stored */
static atomic_int KF_COUNT_PUB;
static struct { uint64_t a, b; int matches; int have; } LAST_CAND;

/* 256 BRIEF pairs in [-13, 13]^2, fixed seeded LCG — regenerated
 * identically everywhere this code runs */
static int8_t PAT[256][4];
static int pat_ready;

static void pat_init(void) {
    if (pat_ready) return;
    uint32_t s = 0xC0FFEE01u;
    for (int i = 0; i < 256; i++) {
        for (int k = 0; k < 4; k++) {
            s = s * 1664525u + 1013904223u;
            PAT[i][k] = (int8_t)((int)(s >> 24) % 14 * ((s >> 23 & 1) ? 1 : -1));
        }
    }
    pat_ready = 1;
}

void xr_map_reset(void) {
    KF_N = 0;
    KF_SEQ = 0;
    atomic_store(&KF_COUNT_PUB, 0);
    LAST_CAND.have = 0;
}

int xr_map_num_keyframes(void) {
    return atomic_load(&KF_COUNT_PUB);
}

int xr_map_last_candidate(uint64_t *ts_a, uint64_t *ts_b, int *matches) {
    if (!LAST_CAND.have) return 0;
    *ts_a = LAST_CAND.a;
    *ts_b = LAST_CAND.b;
    *matches = LAST_CAND.matches;
    return 1;
}

void xr_map_tick(uint64_t now_ts_ns, uint64_t timeout_ns) {
    int w = 0;
    for (int i = 0; i < KF_N; i++) {
        if (now_ts_ns - KF[i].ts <= timeout_ns) {
            if (w != i) KF[w] = KF[i];
            w++;
        }
    }
    if (w != KF_N) {
        KF_N = w;
        atomic_store(&KF_COUNT_PUB, w);
    }
}

/* ---- compact ORB ------------------------------------------------------------- */

static const int8_t CIRC[16][2] = {
    {0,-3},{1,-3},{2,-2},{3,-1},{3,0},{3,1},{2,2},{1,3},
    {0,3},{-1,3},{-2,2},{-3,1},{-3,0},{-3,-1},{-2,-2},{-1,-3}
};

static int fast_score(const uint8_t *img, int x, int y) {
    int c = img[y * XR_OW + x];
    /* quick reject on the compass points */
    int b = 0, d = 0;
    for (int k = 0; k < 16; k += 4) {
        int v = img[(y + CIRC[k][1]) * XR_OW + x + CIRC[k][0]];
        if (v > c + FAST_THRESH) b++;
        else if (v < c - FAST_THRESH) d++;
    }
    if (b < 3 && d < 3) return 0;
    /* contiguous arc of 9 */
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
    enum { GX = (XR_OW - 2 * MARGIN) / NMS_GRID,
           GY = (XR_OH - 2 * MARGIN) / NMS_GRID };
    for (int gy = 0; gy < GY && kf->n_kp < XR_MAP_KP_PER_KF; gy++) {
        for (int gx = 0; gx < GX && kf->n_kp < XR_MAP_KP_PER_KF; gx++) {
            int bs = 0, bx = -1, by = -1;
            for (int y = MARGIN + gy * NMS_GRID;
                 y < MARGIN + (gy + 1) * NMS_GRID; y += 2) {
                for (int x = MARGIN + gx * NMS_GRID;
                     x < MARGIN + (gx + 1) * NMS_GRID; x += 2) {
                    int s = fast_score(img, x, y);
                    if (s > bs) { bs = s; bx = x; by = y; }
                }
            }
            if (bs <= 0) continue;
            /* intensity-centroid orientation over 15x15 */
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
            kf->kp_u[i] = (uint16_t)bx;
            kf->kp_v[i] = (uint16_t)by;
            uint64_t d[4] = { 0, 0, 0, 0 };
            for (int b = 0; b < 256; b++) {
                int ax = (int)lroundf(ca * PAT[b][0] - sa * PAT[b][1]);
                int ay = (int)lroundf(sa * PAT[b][0] + ca * PAT[b][1]);
                int bx2 = (int)lroundf(ca * PAT[b][2] - sa * PAT[b][3]);
                int by2 = (int)lroundf(sa * PAT[b][2] + ca * PAT[b][3]);
                int va = img[(by + ay) * XR_OW + bx + ax];
                int vb = img[(by + by2) * XR_OW + bx + bx2];
                if (va < vb) d[b >> 6] |= 1ull << (b & 63);
            }
            memcpy(kf->desc[i], d, sizeof d);
        }
    }
}

static inline int hamming256(const uint64_t a[4], const uint64_t b[4]) {
    return __builtin_popcountll(a[0] ^ b[0]) +
           __builtin_popcountll(a[1] ^ b[1]) +
           __builtin_popcountll(a[2] ^ b[2]) +
           __builtin_popcountll(a[3] ^ b[3]);
}

static int match_count(const xr_kf *a, const xr_kf *b) {
    int n = 0;
    for (int i = 0; i < a->n_kp; i++) {
        int best = 999, second = 999;
        for (int j = 0; j < b->n_kp; j++) {
            int d = hamming256(a->desc[i], b->desc[j]);
            if (d < best) { second = best; best = d; }
            else if (d < second) second = d;
        }
        if (best <= MATCH_MAX_DIST && best + MATCH_MARGIN <= second) n++;
    }
    return n;
}

int xr_map_maybe_keyframe(const float q[4], const float p[3], uint64_t ts_ns,
                          const uint8_t *img,
                          const int32_t *lm_id, const float (*lm_xyz)[3],
                          const float (*lm_uv)[2], int n_lm) {
    if (KF_N > 0) {
        const xr_kf *last = &KF[KF_N - 1];
        float dx = p[0] - last->p[0], dy = p[1] - last->p[1],
              dz = p[2] - last->p[2];
        float qd = fabsf(q[0] * last->q[0] + q[1] * last->q[1] +
                         q[2] * last->q[2] + q[3] * last->q[3]);
        if (dx * dx + dy * dy + dz * dz < KF_DIST_M * KF_DIST_M &&
            qd > KF_ANGLE_COS)
            return 0;
    }
    if (KF_N == XR_MAP_MAX_KF) {           /* cap: drop the oldest */
        memmove(&KF[0], &KF[1], sizeof(xr_kf) * (XR_MAP_MAX_KF - 1));
        KF_N--;
    }
    xr_kf *kf = &KF[KF_N];
    kf->ts = ts_ns;
    memcpy(kf->q, q, sizeof kf->q);
    memcpy(kf->p, p, sizeof kf->p);
    orb_extract(img, kf);
    kf->n_lm = n_lm > XR_MAP_KP_PER_KF ? XR_MAP_KP_PER_KF : n_lm;
    for (int i = 0; i < kf->n_lm; i++) {
        kf->lm_id[i] = lm_id[i];
        memcpy(kf->lm_xyz[i], lm_xyz[i], sizeof kf->lm_xyz[i]);
        memcpy(kf->lm_uv[i], lm_uv[i], sizeof kf->lm_uv[i]);
    }
    KF_N++;
    KF_SEQ++;
    atomic_store(&KF_COUNT_PUB, KF_N);

    /* loop/reloc candidates: descriptor stage (geometric verification and
     * the T_session<-odom correction are the next layer) */
    int best_m = 0, best_i = -1;
    for (int i = 0; i < KF_N - 1 - CAND_SKIP_RECENT; i++) {
        int m = match_count(kf, &KF[i]);
        if (m > best_m) { best_m = m; best_i = i; }
    }
    if (best_i >= 0 && best_m >= CAND_MIN_MATCHES) {
        LAST_CAND.a = kf->ts;
        LAST_CAND.b = KF[best_i].ts;
        LAST_CAND.matches = best_m;
        LAST_CAND.have = 1;
        LOGI("session map: LOOP CANDIDATE kf#%d <-> kf#%d (%d descriptor "
             "matches, dt=%.1fs)", KF_N - 1, best_i, best_m,
             (double)(kf->ts - KF[best_i].ts) / 1e9);
    }
    return 1;
}
