/* xr_map.c — see xr_map.h.
 *
 * Handcrafted descriptors (no inference): BAD/TEBLID — FAST-9 corners with
 * grid NMS + intensity-centroid orientation, then a 256-bit LEARNED
 * box-average-difference descriptor off an integral image (Suarez et al.,
 * RA-L 2021; params in xr_teblid_params.h). Far more robust than the old
 * mini-ORB BRIEF on these grainy sensors, still binary (Hamming), still CPU.
 * Learned descriptors: XFeat via xr_xfeat (int8-quantized 64-D, cosine ~=
 * dot/16129; NEON sdot when available) — heavier (ONNX inference), used when
 * BAD is not enough. Brute force matching is fine at session scale — the
 * koide3/GLIM lesson applied here is the bounded incremental store.
 */
#include "xr_map.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <android/log.h>

#include "xr_vpr.h"
#include "xr_xfeat.h"
#include "xr_teblid_params.h"
#include "xreal_core.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#define MAP_SDOT 1
#endif

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

#define KF_DIST_M 0.30f            /* motion gates */
#define KF_ANGLE_COS 0.99144f      /* cos(15/2 deg) on the quat dot */
#define KF_NEAR_M 2.0f             /* proximity that keeps a keyframe fresh */

/* FAST-9 detector (shared by both descriptors) */
#define FAST_THRESH 18
#define NMS_GRID 24
#define MARGIN 20
#define BAD_MAX_DIST 60            /* Hamming acceptance (256-bit TEBLID) */
#define BAD_MARGIN 10

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
/* the matched place must be an ESTABLISHED cluster: this many DISTINCT
 * keyframes must each supply a geometric INLIER (not just a raw match) —
 * a lone shake-spawned keyframe or a two-frame junk cluster is rejected */
#define COVIS_MIN_KF 3
/* a proposed correction is applied only after a SECOND verified query
 * within this window agrees on the pose — one confident-but-wrong PnP
 * frame never snaps on its own */
#define CONFIRM_WINDOW_NS 4000000000ull
#define CONFIRM_DP_M 0.25f
#define CONFIRM_DA_RAD 0.09f       /* ~5 deg */
/* loop SEARCH is the heavy work (descriptor extraction + match against
 * every keyframe + PnP RANSAC). Un-throttled it ran back-to-back on
 * every offer — 15-20x/s in a loop-rich scene — and starved Basalt's VIO
 * threads (IMU queue overflow -> divergence -> "flying off"). The
 * event-driven cadence (LOOP_SEARCH_HEALTHY_NS while tracking, every offer
 * when LOST) lives with the recovery constants below; storage keeps its
 * own (faster) cadence. */
/* stationary reloc cadence: when the motion gate blocks, a query-only
 * offer still goes through this often — matching (against the SAME
 * old-keyframe set the moving queries use) but never storing. Standing
 * still and staring at a known scene must be able to heal the pose. */
#define QUERY_INTERVAL_NS 2500000000ull

/* Confirmed-recovery lifecycle. A shake/loss FREEZES keyframe storage and
 * stays frozen until a relocalization is verified (RECOVERED) — NOT a
 * fixed shake timer: a stable-but-WRONG post-shake odometry must never be
 * allowed to quietly lay down keyframes in the wrong frame. If no map is
 * ever matched (a genuinely new area), storage resumes after a give-up. */
enum { REC_HEALTHY = 0, REC_LOST = 1, REC_RECOVERED = 2 };
#define REC_MIN_MAP 12                 /* fewer keyframes than this = no map to be "lost" from (just a brief shake freeze) */
#define REC_STABLE_NS 1200000000ull    /* healthy this long after a recovery -> normal cadence */
/* loop SEARCH cadence is event-driven AND descriptor-aware: FASTER when
 * LOST (quick recovery), and faster in BAD mode than XFeat mode because
 * BAD/TEBLID has no inference cost — only the (shared) brute-force match +
 * top-K PnP, so we can afford to run it more often. XFeat mode stays
 * backed off (its inference is tens of ms; running it back-to-back on the
 * low-priority map thread starves the VIO — the "flying off" failure).
 * BAD healthy ~3 Hz matches the known-good mini-ORB rate. */
#define LOOP_SEARCH_HEALTHY_BAD_NS   300000000ull
#define LOOP_SEARCH_HEALTHY_XFEAT_NS 700000000ull
#define LOOP_SEARCH_LOST_BAD_NS      150000000ull
#define LOOP_SEARCH_LOST_XFEAT_NS    250000000ull
#define RELOC_TOPK 3                   /* candidate clusters verified when relocalizing (repetitive scenes hand raw scoring to the wrong keyframe) */
/* VPR retrieval (xr_vpr): appearance embedding pre-ranking of closure /
 * relocalization candidates. Active once a model is registered and frames
 * embed; only the top VPR_SHORTLIST keyframes by cosine similarity get the
 * per-keypoint descriptor scoring. Keyframes stored before the model came
 * up (has_emb=0) stay always-eligible so a mid-session rollout can't hide
 * part of the map. */
#define VPR_SHORTLIST 12
#define VPR_MIN_SIM 0.25f
#define VPR_MIN_KF (VPR_SHORTLIST + 4)  /* smaller maps: full scan is cheap */
/* HEALTHY loop-detection coarse gate: only full-match keyframes whose
 * SESSION position is within this radius of the corrected pose (a revisit
 * is where you physically are). A FULL sweep every Nth healthy search
 * still catches a large-residual-drift loop the gate would miss. LOST
 * recovery ignores the gate (recall must not regress). This cuts the
 * dominant per-search matching from O(all keyframes) toward O(local). */
#define SHORTLIST_R_M 10.0f
#define FULL_SWEEP_EVERY 8
/* displayed keyframe-derived cloud cap — a generous safety ceiling, not a
 * budget: the map is really bounded by the 200-keyframe rolling cap, and
 * drawing points is cheap (GL_POINTS, ~linear). This shows essentially the
 * whole session map rather than an arbitrary window. */
#define CLOUD_MAX 8192

enum { DESC_BAD = 0, DESC_XFEAT = 1 };

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
    /* One descriptor type per keyframe (desc_type). Overlaid so BAD/TEBLID
     * packs at a 32-byte stride — two descriptors per cache line in the hot
     * match scan — instead of the old 64-byte slots it half-filled. */
    union {
        int8_t bad[XR_MAP_KP_PER_KF][32];    /* TEBLID/BAD: 256-bit Hamming */
        int8_t xfeat[XR_MAP_KP_PER_KF][64];  /* XFeat: int8 64-D */
    } desc;
    int lm_of_kp[XR_MAP_KP_PER_KF];      /* kp -> landmark index, or -1 */
    int n_lm;
    int32_t lm_id[XR_MAP_KP_PER_KF];
    float lm_xyz[XR_MAP_KP_PER_KF][3];
    float lm_uv[XR_MAP_KP_PER_KF][2];
    /* place-recognition embedding (xr_vpr), L2-normalized; the retrieval
     * pre-rank's dot products are cosines. Model-version-locked: emb_dim
     * is the active model's dimension (512 EigenPlaces .. 8448 MegaLoc);
     * only same-dim embeddings compare. */
    int has_emb;
    int emb_dim;
    float emb[XR_VPR_MAX_DIM];
} xr_kf;

static xr_kf KF[XR_MAP_MAX_KF];        /* .bss — physical slot pool */
static int KF_N;                       /* active keyframe count */
/* Stable-slot indirection. KF[] is a pool: a keyframe keeps its slot for life
 * (stable IDs for future cross-references, and eviction no longer memmoves
 * ~4 MB of structs). KFO[k] is the physical slot of the k-th keyframe in
 * insertion-time order — exactly what the old contiguous KF[k] meant — so
 * every reader goes through KFA(k). KF_FREE is a stack of unused slots. */
static int KFO[XR_MAP_MAX_KF];         /* time order -> physical slot */
static int KF_FREE[XR_MAP_MAX_KF];     /* free-slot stack */
static int KF_FREE_N;                  /* free-slot count */
#define KFA(i) KF[KFO[i]]              /* the i-th keyframe in time order */

/* Empty the map: every slot free, no keyframes. Run at load and on every
 * clear (reset / descriptor switch) so KF_FREE is always valid before a
 * store. */
static void kf_slots_reset(void) {
    KF_N = 0;
    KF_FREE_N = XR_MAP_MAX_KF;
    for (int i = 0; i < XR_MAP_MAX_KF; i++) KF_FREE[i] = i;
}
__attribute__((constructor)) static void kf_slots_ctor(void) { kf_slots_reset(); }

static atomic_int KF_COUNT_PUB;
static struct { uint64_t a, b; int matches; int have; } LAST_CAND;
/* last verification attempt, for the on-screen panel: no more guessing
 * which stage blocked a snap */
enum { VOUT_NONE = 0, VOUT_BELOW_BAR = 1, VOUT_FEW_PAIRS = 2,
       VOUT_FEW_INLIERS = 3, VOUT_CAPPED = 4, VOUT_APPLIED = 5,
       VOUT_GATED = 6,        /* verified, but no significant drift / cooldown */
       VOUT_PENDING = 7 };    /* verified, awaiting a confirming 2nd frame */
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
static atomic_int SHAKING;             /* raw shake flag from the IMU thread */
static atomic_int REC_STATE;           /* REC_HEALTHY / REC_LOST / REC_RECOVERED */
/* bumped whenever another thread clears the store (reset / descriptor
 * switch). The map thread captures it before its LOCK-FREE candidate
 * search and re-checks after taking the lock: a mismatch means the map
 * changed under the search, so its (now stale) result is discarded. */
static atomic_uint RESET_GEN;
static uint64_t RECOVERED_NS;          /* when the last recovery landed */
static struct {                        /* correction awaiting confirmation */
    int have;
    uint64_t ts;
    float q[4], p[3];                  /* candidate D = odom -> session */
} PENDING_D;
static char MODEL_PATH[512];
static pthread_once_t THREAD_ONCE = PTHREAD_ONCE_INIT;
static atomic_int MAPPING = 1;
static atomic_int RECOVERY = 1;        /* live-pose snap (loop closure of
                                          the MAP itself is always on) */
static atomic_int USE_XFEAT = 0;       /* runtime descriptor selector */

/* TWO corrections, because a closure has two jobs (the review's point).
 * CORR = the MAP correction: the frame keyframes are inserted in and the
 * displayed cloud lives in. It ALWAYS updates on a confirmed closure so
 * new nodes attach in the healed frame. LIVE = the DISPLAY correction that
 * xr_map_get_correction hands the SLAM worker for the shown pose: it also
 * updates when recovery is ON (pose follows the healed map), but with
 * recovery OFF it stays put (odometry-continuous — the GNSS-fusion mode,
 * where an external reference places the pose while the map self-heals).
 * With recovery ON the two are identical, so the tested path is unchanged.
 * Both are D = C ∘ O⁻¹ (map->odom pattern), identity until the first
 * closure. */
static struct { float q[4]; float p[3]; int gen; } CORR = { .q = {1, 0, 0, 0} };
static struct { float q[4]; float p[3]; int gen; } LIVE = { .q = {1, 0, 0, 0} };

/* the authoritative displayed map: the session-frame point cloud DERIVED
 * from the keyframe graph (every landmark placed through its owning
 * keyframe's corrected pose). Rebuilt whenever the graph changes — a
 * closure that deforms the keyframes deforms this cloud with them. */
static struct { float xyz[CLOUD_MAX][3]; int n; } CLOUD;
static int CLOUD_DIRTY;
/* Bumped every time CLOUD is rebuilt (store / deform / evict). A reader
 * passes back the generation it last copied; xr_map_get_cloud returns -1
 * without taking the lock when nothing changed, so the VIO- and UI-rate
 * pollers stop copying an unchanged 96 KB cloud under MAP_LOCK. */
static atomic_uint CLOUD_GEN;

/* Loop-search telemetry (written under MAP_LOCK in the write phase, read by
 * xr_map_perf). Tells us how well the spatial gate prunes and where the map
 * thread spends its budget: how many keyframes were actually match-scored,
 * how long the lock-free match+PnP took, and how long the brief write phase
 * then waited for the lock the VIO worker also holds. */
static struct {
    int searched;         /* keyframes match_count'd in the last search */
    int candidates;       /* clusters that reached geometric verification */
    unsigned match_us;    /* lock-free match + top-K PnP wall time */
    unsigned lock_us;     /* write-phase wait for MAP_LOCK */
} PERF;

static inline uint64_t map_mono_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

static void *xfeat_preload_thread(void *arg) {
    (void)arg;
    /* ORT's intra-op pool threads are created inside CreateSession and
     * inherit THIS thread's nice value. Left at the default 0 they outrank
     * the VIO workers during every relocalization inference burst — the
     * exact window pose is most fragile — defeating the map thread's own
     * nice-19 demotion. Demote first so the pool lands at 19 too. */
    setpriority(PRIO_PROCESS, (id_t)gettid(), 19);
    if (MODEL_PATH[0]) xr_xfeat_init(MODEL_PATH);   /* dlopen ORT + load model */
    return NULL;
}

/* Register the VPR (place recognition) model — retrieval pre-ranking
 * activates once frames embed successfully; without it the map keeps the
 * brute-scan + spatial-gate behavior. */
void xr_map_set_vpr_model(const char *onnx_path) {
    xr_vpr_set_model(onnx_path);
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
    if (on) {
        /* catch the display frame up to the map frame immediately, so
         * closures that landed while recovery was OFF are reflected at
         * once instead of waiting for the next one */
        pthread_mutex_lock(&MAP_LOCK);
        memcpy(LIVE.q, CORR.q, sizeof LIVE.q);
        memcpy(LIVE.p, CORR.p, sizeof LIVE.p);
        LIVE.gen++;
        pthread_mutex_unlock(&MAP_LOCK);
    }
    LOGI("session map: loop recovery %s",
         on ? "ON (verified closures snap the live pose)"
            : "OFF (map keeps closing loops internally; the live pose is "
              "odometry-continuous — the GNSS-fusion mode)");
}

/* Note violent motion (shake/loss) from the IMU thread. This drives the
 * confirmed-recovery lifecycle (below): storage freezes and stays frozen
 * until a relocalization is VERIFIED, not merely until the shake stops.
 * Relocalization queries keep running throughout. */
void xr_map_freeze_storage(int shaking) {
    atomic_store(&SHAKING, shaking ? 1 : 0);
}

/* Recovery lifecycle for the panel: 0 healthy, 1 lost/relocalizing,
 * 2 recovered (map just healed, storage resuming). */
int xr_map_recovery_state(void) {
    return atomic_load(&REC_STATE);
}

/* Switch the keyframe descriptor (0 = BAD/TEBLID, 1 = XFeat). BAD and
 * XFeat keyframes cannot match each other, so the store is cleared on a
 * real change — the map rebuilds with the selected descriptor. Only ever
 * called from the UI/JNI thread. Returns the descriptor actually in effect
 * afterwards (0 = BAD/TEBLID, 1 = XFeat) so the UI can reflect a rejected
 * switch instead of desyncing its label. */
int xr_map_set_use_xfeat(int on) {
    int want = on ? 1 : 0;
    if (atomic_load(&USE_XFEAT) == want) return want;   /* no change */
    if (want) {
        /* Load XFeat NOW, before committing the switch. If ONNX Runtime or the
         * model isn't there — always the case in a lean build — reject the
         * switch: don't wipe a working BAD/TEBLID map for a descriptor that
         * would only fall back to BAD anyway. Loading here rather than lazily
         * on the map thread also guarantees XFeat is ready before the first
         * keyframe, so a half-loaded switch can't store BAD keyframes into an
         * otherwise-XFeat map. */
        if (!MODEL_PATH[0] || !xr_xfeat_init(MODEL_PATH)) {
            LOGI("session map: XFeat unavailable — staying on BAD/TEBLID");
            return 0;                          /* rejected: still BAD/TEBLID */
        }
    }
    atomic_store(&USE_XFEAT, want);
    pthread_mutex_lock(&MAP_LOCK);
    kf_slots_reset();
    atomic_store(&KF_COUNT_PUB, 0);
    atomic_fetch_add(&RESET_GEN, 1);       /* invalidate any in-flight search */
    LAST_STORE_NS = 0;
    CLOUD.n = 0;                            /* the old-descriptor cloud is void */
    CLOUD_DIRTY = 0;
    atomic_fetch_add(&CLOUD_GEN, 1);       /* readers drop it; empty until rebuilt */
    /* This is a NEW-MAP transition. If we were LOST, the map that reloc would
     * recover against is now gone, and LOST only exits via a verified reloc —
     * so leaving REC_STATE=LOST would strand us forever on the empty map. Reset
     * the recovery lifecycle to HEALTHY and clear the pending/verify/candidate
     * state. SHAKING is left intact on purpose: storage stays frozen until the
     * physical shake actually ends. */
    atomic_store(&REC_STATE, REC_HEALTHY);
    PENDING_D.have = 0;
    RECOVERED_NS = 0;
    LAST_CAND.have = 0;
    LAST_POSE.have = 0;
    LAST_ACCEPT_NS = 0;
    LAST_SNAP_NS = 0;
    memset(&LOOP_STATS, 0, sizeof LOOP_STATS);
    memset(&VER_LAST, 0, sizeof VER_LAST);
    pthread_mutex_unlock(&MAP_LOCK);
    LOGI("session map: descriptor -> %s (keyframes cleared, rebuilding)",
         want ? "XFeat" : "BAD/TEBLID");
    return want;
}

/* Whether the XFeat model + ONNX Runtime are actually loaded (so the UI
 * can tell "XFeat requested" from "XFeat running"). */
int xr_map_xfeat_ready(void) {
    return MODEL_PATH[0] != 0 && xr_xfeat_available();
}

void xr_map_reset(void) {
    pthread_mutex_lock(&MAP_LOCK);
    kf_slots_reset();
    atomic_fetch_add(&RESET_GEN, 1);       /* invalidate any in-flight search */
    LAST_CAND.have = 0;
    LAST_POSE.have = 0;
    MBOX.full = 0;
    LAST_ACCEPT_NS = 0;
    LAST_STORE_NS = 0;
    LAST_SNAP_NS = 0;
    PENDING_D.have = 0;
    RECOVERED_NS = 0;
    atomic_store(&SHAKING, 0);
    atomic_store(&REC_STATE, REC_HEALTHY);
    CLOUD.n = 0;
    CLOUD_DIRTY = 0;
    atomic_fetch_add(&CLOUD_GEN, 1);       /* readers must drop the old cloud */
    memset(&LOOP_STATS, 0, sizeof LOOP_STATS);
    memset(&VER_LAST, 0, sizeof VER_LAST);
    CORR.q[0] = 1; CORR.q[1] = CORR.q[2] = CORR.q[3] = 0;
    CORR.p[0] = CORR.p[1] = CORR.p[2] = 0;
    CORR.gen++;                    /* consumers re-sync their frame */
    LIVE.q[0] = 1; LIVE.q[1] = LIVE.q[2] = LIVE.q[3] = 0;
    LIVE.p[0] = LIVE.p[1] = LIVE.p[2] = 0;
    LIVE.gen++;
    atomic_store(&KF_COUNT_PUB, 0);
    pthread_mutex_unlock(&MAP_LOCK);
}

int xr_map_get_correction(float q[4], float p[3]) {
    pthread_mutex_lock(&MAP_LOCK);
    memcpy(q, LIVE.q, sizeof LIVE.q);   /* the DISPLAY correction */
    memcpy(p, LIVE.p, sizeof LIVE.p);
    int g = LIVE.gen;
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

/* ---- pose graph: anchor-local landmarks + 4-DoF chain deformation -------------- */

/* A keyframe's own correction D_j = C_j ∘ O_j⁻¹ (odom -> session): the
 * transform carrying its IMMUTABLE odom pose to its graph-optimized
 * session pose. Its landmarks are stored in odom and ride on D_j, so
 * moving C_j moves the keyframe's entire point cloud — the landmarks are
 * anchor-local for free, no per-point rewrite. */
static void kf_corr(const xr_kf *k, float dq[4], float dp[3]) {
    float qoi[4], poi[3];
    pose_invert(k->q, k->p, qoi, poi);
    pose_compose(k->qc, k->pc, qoi, poi, dq, dp);
}

/* interpolate a correction a fraction t (0..1) from (qa,pa) to (qb,pb):
 * SLERP the (gravity-aligned yaw) rotation, LERP the translation */
static void corr_interp(const float qa[4], const float pa[3],
                        const float qb[4], const float pb[3], float t,
                        float qo[4], float po[3]) {
    float d = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
    float b[4];
    if (d < 0) { for (int i = 0; i < 4; i++) b[i] = -qb[i]; d = -d; }
    else memcpy(b, qb, sizeof b);
    if (d > 0.9995f) {
        for (int i = 0; i < 4; i++) qo[i] = qa[i] + t * (b[i] - qa[i]);
    } else {
        float ang = acosf(d), s = sinf(ang);
        float wa = sinf((1 - t) * ang) / s, wb = sinf(t * ang) / s;
        for (int i = 0; i < 4; i++) qo[i] = wa * qa[i] + wb * b[i];
    }
    float n = sqrtf(qo[0] * qo[0] + qo[1] * qo[1] + qo[2] * qo[2] +
                    qo[3] * qo[3]);
    for (int i = 0; i < 4; i++) qo[i] /= n;
    for (int i = 0; i < 3; i++) po[i] = pa[i] + t * (pb[i] - pa[i]);
}

/* Heal ACCUMULATED DRIFT (a healthy loop closure only — NOT post-loss
 * recovery). When the recent stored trajectory drifted and then closed a
 * loop, the matched anchor and everything BEFORE it are trusted and hold
 * still; the drifted tail (anchor -> newest) is pulled toward D in
 * proportion to its PATH LENGTH from the anchor. Each keyframe's landmarks
 * follow, so the tail DEFORMS onto the reference. O(chain), 4-DoF (gravity
 * fixes roll/pitch). MAP_LOCK held. `qsp` = the query's SESSION position.
 *
 * The path is measured in the SESSION frame (pc), not raw odom: if Basalt
 * re-initialised during the session, consecutive keyframes can span an odom
 * discontinuity, and an odom-frame path length would be a huge bogus jump
 * that corrupts the distribution. The session frame is continuous. Within a
 * single odom epoch the two are near-identical (the correction is locally
 * rigid), so this only changes behaviour across a discontinuity — for the
 * better.
 *
 * This must NOT run for post-shake relocalization: storage is frozen
 * through a discontinuity, so the stored map is already correct and only
 * the live odom frame needs registering — deforming a correct map corrupts
 * it. The caller gates on REC_HEALTHY. */
static void graph_deform(int anchor, const float qsp[3],
                         const float Dq[4], const float Dp[3]) {
    if (anchor < 0 || anchor >= KF_N) return;
    static float cum[XR_MAP_MAX_KF];       /* session path length from anchor */
    float total = 0;
    cum[anchor] = 0;
    for (int j = anchor + 1; j < KF_N; j++) {
        float dx = KFA(j).pc[0] - KFA(j - 1).pc[0];
        float dy = KFA(j).pc[1] - KFA(j - 1).pc[1];
        float dz = KFA(j).pc[2] - KFA(j - 1).pc[2];
        total += sqrtf(dx * dx + dy * dy + dz * dz);
        cum[j] = total;
    }
    /* the correction D was estimated for the CURRENT query, not the newest
     * stored keyframe. Extend the path to the query (in session coords) so
     * the newest keyframe gets a fraction < 1 (the drift between it and the
     * query rides on the live pose via CORR); otherwise the newest stored
     * portion is over-deformed. */
    if (KF_N > anchor + 1) {
        float dx = qsp[0] - KFA(KF_N - 1).pc[0];
        float dy = qsp[1] - KFA(KF_N - 1).pc[1];
        float dz = qsp[2] - KFA(KF_N - 1).pc[2];
        total += sqrtf(dx * dx + dy * dy + dz * dz);
    }
    for (int j = anchor + 1; j < KF_N; j++) {
        float a = total > 1e-3f ? cum[j] / total : 1.0f;
        float dq_old[4], dp_old[3], dq_new[4], dp_new[3];
        kf_corr(&KFA(j), dq_old, dp_old);  /* this keyframe's current corr */
        corr_interp(dq_old, dp_old, Dq, Dp, a, dq_new, dp_new);
        pose_compose(dq_new, dp_new, KFA(j).q, KFA(j).p, KFA(j).qc, KFA(j).pc);
    }
}

/* Rebuild the authoritative display cloud from the keyframe graph: every
 * landmark placed through its owning keyframe's CORRECTED pose, deduped by
 * landmark id (newest keyframe wins — freshest pose). When the graph
 * deforms, this deforms with it. MAP_LOCK held. */
static void cloud_rebuild(void) {
    static int32_t seen[8192];             /* id hash set, map thread only */
    memset(seen, 0xFF, sizeof seen);       /* -1 = empty */
    int n = 0;
    for (int k = KF_N - 1; k >= 0 && n < CLOUD_MAX; k--) {
        float dq[4], dp[3];
        kf_corr(&KFA(k), dq, dp);
        for (int i = 0; i < KFA(k).n_lm && n < CLOUD_MAX; i++) {
            int32_t id = KFA(k).lm_id[i];
            uint32_t h = ((uint32_t)id * 2654435761u) & 8191u;
            int dup = 0;
            for (int s = 0; s < 32; s++) {
                uint32_t j = (h + (uint32_t)s) & 8191u;
                if (seen[j] == id) { dup = 1; break; }
                if (seen[j] == -1) { seen[j] = id; break; }
            }
            if (dup) continue;
            float t[3];
            qrotv(dq, KFA(k).lm_xyz[i], t);
            CLOUD.xyz[n][0] = t[0] + dp[0];
            CLOUD.xyz[n][1] = t[1] + dp[1];
            CLOUD.xyz[n][2] = t[2] + dp[2];
            n++;
        }
    }
    CLOUD.n = n;
    CLOUD_DIRTY = 0;
    atomic_fetch_add(&CLOUD_GEN, 1);
}

/* Snapshot the keyframe-derived cloud (session frame) for the AR/3D view.
 * This REPLACES the old flat cloud that the JNI layer rigidly shifted by
 * the correction — that shift moved everything together and healed
 * nothing; the map now lives in the keyframes and heals when they do. */
int xr_map_get_cloud(float *xyz, int max, unsigned *inout_gen) {
    /* Unlocked fast path: if the caller already holds the current generation
     * the cloud is byte-for-byte unchanged, so skip both the lock and the
     * copy and let the caller reuse the buffer it already has. */
    if (inout_gen && atomic_load(&CLOUD_GEN) == *inout_gen) return -1;
    pthread_mutex_lock(&MAP_LOCK);
    int n = CLOUD.n;
    if (n > max) n = max;
    if (n > 0) memcpy(xyz, CLOUD.xyz, sizeof(float) * 3u * (size_t)n);
    if (inout_gen) *inout_gen = atomic_load(&CLOUD_GEN);
    pthread_mutex_unlock(&MAP_LOCK);
    return n;
}

int xr_map_verify_stats(int *pairs, int *inliers) {
    pthread_mutex_lock(&MAP_LOCK);
    *pairs = VER_LAST.pairs;
    *inliers = VER_LAST.inliers;
    int out = VER_LAST.outcome;
    pthread_mutex_unlock(&MAP_LOCK);
    return out;
}

/* Last loop-search cost: keyframes match-scored, clusters verified, match+PnP
 * microseconds, and write-phase lock wait. For the health line / benchmarking.
 * PERF is written by the map thread under MAP_LOCK; take the lock here too —
 * a concurrent plain read/write is a data race regardless of whether a torn
 * value would be "harmless", and this is polled only every ~10 s. */
void xr_map_perf(int *searched, int *candidates, int *match_us, int *lock_us) {
    pthread_mutex_lock(&MAP_LOCK);
    if (searched) *searched = PERF.searched;
    if (candidates) *candidates = PERF.candidates;
    if (match_us) *match_us = (int)PERF.match_us;
    if (lock_us) *lock_us = (int)PERF.lock_us;
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

/* ---- BAD / TEBLID descriptor (integral-image box differences) ------------------ */

/* the 32x32 training patch maps ~1:1 to image pixels — our detector is
 * single-scale so keypoints carry no size to scale by */
#define BAD_SCALE 1.0f

static int32_t INTEG[(XR_OH + 1) * (XR_OW + 1)];   /* map-thread scratch (.bss) */

static void build_integral(const uint8_t *img) {
    const int IW = XR_OW + 1;
    for (int x = 0; x <= XR_OW; x++) INTEG[x] = 0;
    for (int y = 0; y < XR_OH; y++) {
        int32_t *row = INTEG + (y + 1) * IW;
        const int32_t *prev = INTEG + y * IW;
        const uint8_t *irow = img + y * XR_OW;
        int32_t s = 0;
        row[0] = 0;
        for (int x = 0; x < XR_OW; x++) {
            s += irow[x];
            row[x + 1] = prev[x + 1] + s;
        }
    }
}

/* mean intensity of the (2r+1)-square box centred at (cx,cy), clamped to
 * the frame (matches OpenCV's computeABWLResponse) */
static inline float box_avg(int cx, int cy, int r) {
    const int IW = XR_OW + 1, IH = XR_OH + 1;
    int x1 = cx - r, x2 = cx + r + 1, y1 = cy - r, y2 = cy + r + 1;
    if (x1 < 0) x1 = 0; else if (x1 > IW - 2) x1 = IW - 2;
    if (x2 < 1) x2 = 1; else if (x2 > IW - 1) x2 = IW - 1;
    if (y1 < 0) y1 = 0; else if (y1 > IH - 2) y1 = IH - 2;
    if (y2 < 1) y2 = 1; else if (y2 > IH - 1) y2 = IH - 1;
    int A = INTEG[y1 * IW + x1], B = INTEG[y1 * IW + x2];
    int C = INTEG[y2 * IW + x1], D = INTEG[y2 * IW + x2];
    return (float)(A + D - B - C) / (float)((y2 - y1) * (x2 - x1));
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

/* TEBLID-256 descriptor at (cx, cy): intensity-centroid orientation, then
 * the 256 learned box-average differences read off the integral image.
 * `img` supplies the small centroid window; INTEG the box sums. */
static void teblid_describe(const uint8_t *img, int cx, int cy, int8_t *desc) {
    int m10 = 0, m01 = 0;
    for (int dy = -7; dy <= 7; dy++)
        for (int dx = -7; dx <= 7; dx++) {
            int v = img[(cy + dy) * XR_OW + cx + dx];
            m10 += dx * v;
            m01 += dy * v;
        }
    float ang = atan2f((float)m01, (float)m10);
    float ca = cosf(ang) * BAD_SCALE, sa = sinf(ang) * BAD_SCALE;
    uint8_t d[32];
    memset(d, 0, sizeof d);
    for (int i = 0; i < 256; i++) {
        const bad_wl *w = &BAD_WL_256[i];
        int px1 = w->x1 - 16, py1 = w->y1 - 16;   /* patch coords, centred */
        int px2 = w->x2 - 16, py2 = w->y2 - 16;
        int x1 = cx + (int)lroundf(ca * px1 - sa * py1);
        int y1 = cy + (int)lroundf(sa * px1 + ca * py1);
        int x2 = cx + (int)lroundf(ca * px2 - sa * py2);
        int y2 = cy + (int)lroundf(sa * px2 + ca * py2);
        int r = (int)lroundf(BAD_SCALE * w->boxRadius);
        if (box_avg(x1, y1, r) - box_avg(x2, y2, r) <= w->th)
            d[i >> 3] |= (uint8_t)(1u << (7 - (i & 7)));
    }
    memcpy(desc, d, 32);
}

static void bad_extract(const uint8_t *img, xr_kf *kf) {
    build_integral(img);
    kf->n_kp = 0;
    kf->desc_type = DESC_BAD;
    /* descriptors anchored AT the VIO landmarks first: a descriptor match
     * then IS a landmark correspondence — what the loop verification and
     * pose graph consume. Basalt picks corners, so the patches are
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
        teblid_describe(img, x, y, kf->desc.bad[j]);
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
            teblid_describe(img, bx, by, kf->desc.bad[i]);
        }
}

/* ---- matching ------------------------------------------------------------------- */

static inline int hamming256(const int8_t *a, const int8_t *b) {
#if defined(__ARM_NEON)
    /* 256-bit XOR -> per-byte population count -> widening pairwise
     * reduction. vld1q loads are alignment-agnostic, which also removes the
     * formal unaligned / strict-aliasing UB of the old uint64_t* cast
     * (relevant while armeabi-v7a is a target). Portable aarch64 + v7a. */
    uint8x16_t x0 = veorq_u8(vld1q_u8((const uint8_t *)a),
                             vld1q_u8((const uint8_t *)b));
    uint8x16_t x1 = veorq_u8(vld1q_u8((const uint8_t *)a + 16),
                             vld1q_u8((const uint8_t *)b + 16));
    uint8x16_t c = vaddq_u8(vcntq_u8(x0), vcntq_u8(x1));
    uint64x2_t s = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(c)));
    return (int)(vgetq_lane_u64(s, 0) + vgetq_lane_u64(s, 1));
#else
    uint64_t ua[4], ub[4];                 /* memcpy: no aliasing/alignment UB */
    memcpy(ua, a, 32);
    memcpy(ub, b, 32);
    return __builtin_popcountll(ua[0] ^ ub[0]) +
           __builtin_popcountll(ua[1] ^ ub[1]) +
           __builtin_popcountll(ua[2] ^ ub[2]) +
           __builtin_popcountll(ua[3] ^ ub[3]);
#endif
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
                int d = dot64_i8(a->desc.xfeat[i], b->desc.xfeat[j]);
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
                int d = hamming256(a->desc.bad[i], b->desc.bad[j]);
                if (d < best) { second = best; best = d; jb = j; }
                else if (d < second) second = d;
            }
            if (best <= BAD_MAX_DIST && best + BAD_MARGIN <= second) {
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
/* bearing inlier tolerance ~4 deg. 2.5 deg was too tight: FAST keypoint
 * noise (~0.5 deg) PLUS the map point's own inverse-depth
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
    /* squared bearing test: dot/|q| > TOL  <=>  dot>0 && dot^2 > TOL^2*|q|^2
     * (|r|=1). Exact same result as the divide, minus a sqrt + a division
     * per point — this is the RANSAC inner loop, called VER_ITERS x n. */
    const float tol2 = VER_COS_TOL * VER_COS_TOL;
    const float minr2 = VER_MIN_RANGE_M * VER_MIN_RANGE_M;
    int cnt = 0;
    for (int m = 0; m < n; m++) {
        float qx = P[m][0] - C[0], qy = P[m][1] - C[1], qz = P[m][2] - C[2];
        float nq2 = qx * qx + qy * qy + qz * qz;
        if (nq2 < minr2) continue;
        float rx = cy * s[m][0] - sy * s[m][1];
        float ry = sy * s[m][0] + cy * s[m][1];
        float raw = qx * rx + qy * ry + qz * s[m][2];
        if (raw > 0 && raw * raw > tol2 * nq2) cnt++;
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
#define COVIS_MAX_KF 16            /* pooled keyframes: best match + nearest */
#define COVIS_MAX_CAND (XR_MAP_KP_PER_KF * 4)
#define COVIS_AVG_TOL_M 0.5f       /* drop observations this far from the
                                      descriptor-best view before averaging */

/* one 2D-3D correspondence candidate for the greedy assignment */
struct rc_cand {
    int qkp;                       /* query keypoint index */
    int32_t id;                    /* map landmark id */
    int kf;                        /* source keyframe */
    int cost;                      /* descriptor distance, LOWER = better */
    float ps[3];                   /* landmark in the session frame */
};

static int rc_cmp(const void *a, const void *b) {
    return ((const struct rc_cand *)a)->cost - ((const struct rc_cand *)b)->cost;
}

/* covisible-neighbour sort key: pooled keyframes are added nearest-first,
 * so the bounded candidate array is filled with the most relevant views */
struct rc_nb { int s; float d2; };
static int rc_nb_cmp(const void *a, const void *b) {
    float da = ((const struct rc_nb *)a)->d2, db = ((const struct rc_nb *)b)->d2;
    return (da > db) - (da < db);
}

/* Relocalize the query against the matched keyframe AND its spatial
 * neighbours pooled together (covisibility). Matching is one-to-one and
 * best-first: every query feature and every physical landmark is used at
 * most once, so a repeated map descriptor cannot inflate a false
 * consensus (the "confidently wrong" failure). Verification support is
 * counted by GEOMETRIC INLIERS across DISTINCT keyframes, not raw
 * matches. Fills D (odom -> session), the pair/inlier counts, and the
 * inlier-backed covisible-keyframe count; returns 1 on a solved pose. */
static int reloc_pnp(const xr_kf *w, int best_i, int nkf, float Dq[4], float Dp[3],
                     int *out_n3, int *out_nin, int *out_covis) {
    *out_n3 = *out_nin = *out_covis = 0;
    if (!GEOM.have) return 0;
    int bad = w->desc_type == DESC_BAD;

    /* 1a. covisible keyframe pool: the winning keyframe FIRST, then its
     * nearest same-descriptor neighbours within COVIS_R (distance-sorted).
     * Leading with best_i means a dense area can never crowd the actual
     * place winner out of the bounded candidate array. */
    int cov[COVIS_MAX_KF]; int ncov = 0;
    cov[ncov++] = best_i;
    static struct rc_nb nbr[XR_MAP_MAX_KF];
    int nnb = 0;
    for (int s = 0; s < nkf; s++) {    /* SESSION-frame proximity (pc): pool
                                          physically-adjacent keyframes even
                                          across drifted odom epochs */
        if (s == best_i || KFA(s).desc_type != w->desc_type) continue;
        float dx = KFA(s).pc[0] - KFA(best_i).pc[0];
        float dy = KFA(s).pc[1] - KFA(best_i).pc[1];
        float dz = KFA(s).pc[2] - KFA(best_i).pc[2];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > COVIS_R_M * COVIS_R_M) continue;
        nbr[nnb].s = s; nbr[nnb].d2 = d2; nnb++;
    }
    qsort(nbr, (size_t)nnb, sizeof nbr[0], rc_nb_cmp);
    for (int k = 0; k < nnb && ncov < COVIS_MAX_KF; k++) cov[ncov++] = nbr[k].s;

    /* cache each pooled keyframe's landmark -> session transform */
    static float covqd[COVIS_MAX_KF][4], covpd[COVIS_MAX_KF][3];
    for (int c = 0; c < ncov; c++) {
        float qoi[4], poi[3];
        pose_invert(KFA(cov[c]).q, KFA(cov[c]).p, qoi, poi);
        pose_compose(KFA(cov[c]).qc, KFA(cov[c]).pc, qoi, poi, covqd[c], covpd[c]);
    }

    /* 1b. candidate correspondences (query kp -> map landmark) with
     * descriptor cost; best_i's matches enter the pool first */
    static struct rc_cand cand[COVIS_MAX_CAND];
    int nc = 0;
    for (int c = 0; c < ncov && nc < COVIS_MAX_CAND; c++) {
        int s = cov[c];
        int nkp = bad ? anchored_count(&KFA(s)) : KFA(s).n_kp;
        for (int i = 0; i < w->n_kp && nc < COVIS_MAX_CAND; i++) {
            int best = bad ? 999 : -(1 << 30), second = best, bj = -1;
            for (int j = 0; j < nkp; j++) {
                if (bad) {
                    int d = hamming256(w->desc.bad[i], KFA(s).desc.bad[j]);
                    if (d < best) { second = best; best = d; bj = j; }
                    else if (d < second) second = d;
                } else {
                    int d = dot64_i8(w->desc.xfeat[i], KFA(s).desc.xfeat[j]);
                    if (d > best) { second = best; best = d; bj = j; }
                    else if (d > second) second = d;
                }
            }
            if (bj < 0) continue;
            int accept = bad
                ? (best <= BAD_MAX_DIST && best + BAD_MARGIN <= second)
                : (best >= XF_MIN_DOT && best - XF_MARGIN >= second);
            if (!accept) continue;
            int lk = KFA(s).lm_of_kp[bj];
            if (lk < 0) continue;
            float dk2 = 0;
            for (int cc = 0; cc < 3; cc++) {
                float d = KFA(s).lm_xyz[lk][cc] - KFA(s).p[cc];
                dk2 += d * d;
            }
            if (dk2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M) continue;
            float ps[3];
            qrotv(covqd[c], KFA(s).lm_xyz[lk], ps);
            cand[nc].qkp = i;
            cand[nc].id = KFA(s).lm_id[lk];
            cand[nc].kf = s;
            cand[nc].cost = bad ? best : ((1 << 30) - best);
            cand[nc].ps[0] = ps[0] + covpd[c][0];
            cand[nc].ps[1] = ps[1] + covpd[c][1];
            cand[nc].ps[2] = ps[2] + covpd[c][2];
            nc++;
        }
    }
    if (nc < VER_MIN_PAIRS) return 0;

    /* 2. greedy one-to-one: best cost first; each query kp and each
     * landmark id assigned at most once */
    qsort(cand, (size_t)nc, sizeof cand[0], rc_cmp);
    static float Sw[COVIS_MAX_CAND][3], Pw[COVIS_MAX_CAND][3];
    static int32_t Pid[COVIS_MAX_CAND];
    static int32_t used_id[COVIS_MAX_CAND];
    static char qused[XR_MAP_KP_PER_KF];
    memset(qused, 0, sizeof qused);
    int n = 0, nid = 0;
    float Rq[9];
    q2R(w->q, Rq);
    for (int k = 0; k < nc && n < COVIS_MAX_CAND; k++) {
        int i = cand[k].qkp;
        if (qused[i]) continue;
        int dup = 0;
        for (int u = 0; u < nid; u++)
            if (used_id[u] == cand[k].id) { dup = 1; break; }
        if (dup) continue;
        float rc[3], rb[3];
        if (GEOM.unproject(w->kp_uv[i][0], w->kp_uv[i][1], rc)) continue;
        rb[0] = GEOM.R_ic[0] * rc[0] + GEOM.R_ic[1] * rc[1] +
                GEOM.R_ic[2] * rc[2];
        rb[1] = GEOM.R_ic[3] * rc[0] + GEOM.R_ic[4] * rc[1] +
                GEOM.R_ic[5] * rc[2];
        rb[2] = GEOM.R_ic[6] * rc[0] + GEOM.R_ic[7] * rc[1] +
                GEOM.R_ic[8] * rc[2];
        Sw[n][0] = Rq[0] * rb[0] + Rq[1] * rb[1] + Rq[2] * rb[2];
        Sw[n][1] = Rq[3] * rb[0] + Rq[4] * rb[1] + Rq[5] * rb[2];
        Sw[n][2] = Rq[6] * rb[0] + Rq[7] * rb[1] + Rq[8] * rb[2];
        memcpy(Pw[n], cand[k].ps, sizeof Pw[0]);
        Pid[n] = cand[k].id;
        qused[i] = 1;
        used_id[nid++] = cand[k].id;
        n++;
    }
    *out_n3 = n;
    if (n < VER_MIN_PAIRS) return 0;

    /* 2b. robust landmark geometry + observation sets. The greedy step
     * kept whichever single observation had the best DESCRIPTOR cost, but
     * descriptor quality is not depth quality — that lone view may carry
     * the noisiest inverse depth. Re-derive each assigned landmark's
     * session position as the mean of every pooled keyframe that stores it
     * (views past COVIS_AVG_TOL from the descriptor-best one dropped as
     * outliers), and record the SET of keyframes that observe it. That
     * observation set — not the single descriptor-winner — is the real
     * covisibility signal: an established map point is seen by many
     * keyframes, a shake-spawned junk point by one. */
    static uint32_t obsmask[COVIS_MAX_CAND];
    for (int m = 0; m < n; m++) {
        float ref[3] = { Pw[m][0], Pw[m][1], Pw[m][2] };
        float acc[3] = { 0, 0, 0 };
        int cnt = 0;
        uint32_t mask = 0;
        for (int c = 0; c < ncov; c++) {
            int s = cov[c];
            for (int l = 0; l < KFA(s).n_lm; l++) {
                if (KFA(s).lm_id[l] != Pid[m]) continue;
                float dk2 = 0;
                for (int cc = 0; cc < 3; cc++) {
                    float d = KFA(s).lm_xyz[l][cc] - KFA(s).p[cc];
                    dk2 += d * d;
                }
                if (dk2 <= VER_MAX_RANGE_M * VER_MAX_RANGE_M) {
                    float ps[3];
                    qrotv(covqd[c], KFA(s).lm_xyz[l], ps);
                    ps[0] += covpd[c][0];
                    ps[1] += covpd[c][1];
                    ps[2] += covpd[c][2];
                    float ex = ps[0] - ref[0], ey = ps[1] - ref[1],
                          ez = ps[2] - ref[2];
                    if (ex * ex + ey * ey + ez * ez <=
                        COVIS_AVG_TOL_M * COVIS_AVG_TOL_M) {
                        /* only a GEOMETRICALLY-CONSISTENT observation counts
                         * as covisibility support (the review's point): a
                         * keyframe whose stored position disagrees > tol is
                         * not evidence the place is the same one */
                        mask |= (uint32_t)1 << c;
                        acc[0] += ps[0]; acc[1] += ps[1]; acc[2] += ps[2];
                        cnt++;
                    }
                }
                break;  /* one observation of this landmark per keyframe */
            }
        }
        if (cnt > 0) {
            Pw[m][0] = acc[0] / (float)cnt;
            Pw[m][1] = acc[1] / (float)cnt;
            Pw[m][2] = acc[2] / (float)cnt;
        }
        obsmask[m] = mask;
    }

    float Rz[9], C[3];
    int nin = pnp2_ransac(Sw, Pw, n, Rz, C);
    *out_nin = nin;
    if (nin < VER_MIN_PAIRS || nin * 100 < 33 * n) return 0;

    /* 3. inlier-backed covisibility: the union of observation sets over the
     * geometric INLIERS — how many DISTINCT pooled keyframes actually back
     * the solved pose (Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]). A lone junk
     * cluster collapses to one or two bits and is rejected upstream. */
    float cy = Rz[0], sy = Rz[3];
    uint32_t covmask = 0;
    for (int m = 0; m < n; m++) {
        float qx = Pw[m][0] - C[0], qy = Pw[m][1] - C[1], qz = Pw[m][2] - C[2];
        float nq = sqrtf(qx * qx + qy * qy + qz * qz);
        if (nq < VER_MIN_RANGE_M) continue;
        float rx = cy * Sw[m][0] - sy * Sw[m][1];
        float ry = sy * Sw[m][0] + cy * Sw[m][1];
        float dot = (qx * rx + qy * ry + qz * Sw[m][2]) / nq;
        if (dot <= VER_COS_TOL) continue;
        covmask |= obsmask[m];
    }
    *out_covis = __builtin_popcount(covmask);

    /* 4. D (odom -> session): rotation = Rz; translation places the query
     * body at the recovered camera centre C (minus the lever arm) */
    R2q(Rz, Dq);
    float qsb[4], t3[3], body_s[3];
    qmul(Dq, w->q, qsb);
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
    /* Capture the store's generation, count and correction AT DEQUEUE, under
     * the lock — the coherent state this (possibly stale) offer belongs to.
     * The lock-free search below reads only these snapshots, never KF_N /
     * CORR directly (those are written by a concurrent reset / descriptor
     * switch — reading them unlocked would be a data race). If RESET_GEN has
     * moved by the time we take the lock to publish, the whole result is
     * abandoned — no stale correction AND no stale/wrong-descriptor store. */
    unsigned rgen0 = atomic_load(&RESET_GEN);
    int kfn = KF_N;
    uint64_t snap_recovered_ns = RECOVERED_NS;   /* reset writes these under */
    uint64_t snap_last_store_ns = LAST_STORE_NS;  /* the lock; snapshot here */
    float snap_corr_q[4], snap_corr_p[3];
    memcpy(snap_corr_q, CORR.q, sizeof snap_corr_q);
    memcpy(snap_corr_p, CORR.p, sizeof snap_corr_p);
    pthread_mutex_unlock(&MAP_LOCK);

    int mapping = atomic_load(&MAPPING);

    /* confirmed-recovery lifecycle. A shake marks us LOST and FREEZES
     * storage. We stay LOST — relocalizing against the (now frozen,
     * still-correct) map — until a closure is VERIFIED (RECOVERED, storage
     * resumes). There is deliberately NO time-based give-up: a stable-but-
     * WRONG post-shake odometry must never quietly resume laying keyframes
     * in an unregistered frame. Mapping a genuinely NEW area after an
     * unrecoverable loss needs a reset for now (proper new-submap handling
     * is the pending same-session-submap work). */
    int shaking = atomic_load(&SHAKING);
    int rstate_prev = atomic_load(&REC_STATE);
    int rstate = rstate_prev;
    if (shaking && kfn >= REC_MIN_MAP) {
        rstate = REC_LOST;
    } else if (rstate == REC_RECOVERED) {
        if (work.ts - snap_recovered_ns > REC_STABLE_NS) rstate = REC_HEALTHY;
    }
    /* REC_LOST is left ONLY by a confirmed recovery (in the apply branch).
     * Do NOT publish REC_STATE here. A concurrent reset / descriptor switch
     * may have advanced RESET_GEN and set HEALTHY on a now-empty map; storing
     * our (pre-reset) rstate unguarded would clobber that back to LOST, and
     * with no map to relocalize against we would be stranded LOST. Publication
     * happens only once the generation is revalidated under MAP_LOCK — in the
     * write phase, and in the early-return path below for a bare transition. */
    int lost = rstate == REC_LOST;
    /* a shake with too small a map to be "lost" from still briefly freezes */
    int shake_freeze = shaking && kfn < REC_MIN_MAP;

    /* rate gates (map-thread-only timing). Skip ALL the heavy work when
     * this offer will neither search nor store — what keeps the map thread
     * from starving Basalt. Search cadence is event-driven and
     * descriptor-aware (see the constants): faster when LOST, and faster in
     * cheap BAD mode than in XFeat mode. */
    static uint64_t last_search_ns;
    static unsigned last_search_cost_us;   /* previous search's measured cost */
    int xf = atomic_load(&USE_XFEAT) && xr_xfeat_available();
    uint64_t search_iv = lost
        ? (xf ? LOOP_SEARCH_LOST_XFEAT_NS : LOOP_SEARCH_LOST_BAD_NS)
        : (xf ? LOOP_SEARCH_HEALTHY_XFEAT_NS : LOOP_SEARCH_HEALTHY_BAD_NS);
    /* LOST cost-cap: LOST always full-scans, and as the map grows that scan
     * (+ top-K PnP) reaches 20-50 ms — at the fixed 150/250 ms cadence the
     * map thread becomes a 15-30% duty burner exactly while pose is most
     * fragile. Cap search duty at ~25%: never search more often than 4x the
     * previous search's measured cost. Small maps are unaffected. */
    if (lost && (uint64_t)last_search_cost_us * 4000ull > search_iv)
        search_iv = (uint64_t)last_search_cost_us * 4000ull;  /* us->ns, x4 */
    int do_search = last_search_ns == 0 || q_only ||
                    work.ts - last_search_ns >= search_iv;
    int may_store = mapping && !q_only && !lost && !shake_freeze &&
                    work.n_lm >= STORE_MIN_LM &&
                    work.ts - snap_last_store_ns >= STORE_MIN_INTERVAL_NS;
    if (!do_search && !may_store) {
        /* nothing heavy to do this pass, but a state TRANSITION (e.g. a shake
         * flipping HEALTHY->LOST on a pass that neither searches nor stores)
         * must still persist — gen-safely, and only when it actually changed
         * so the common no-work path stays off the lock. */
        if (rstate != rstate_prev) {
            pthread_mutex_lock(&MAP_LOCK);
            if (atomic_load(&RESET_GEN) == rgen0)
                atomic_store(&REC_STATE, rstate);
            pthread_mutex_unlock(&MAP_LOCK);
        }
        return;
    }
    if (do_search) last_search_ns = work.ts;

    /* descriptors: XFeat when selected AND available, BAD/TEBLID otherwise */
    if (atomic_load(&USE_XFEAT) && MODEL_PATH[0]) xr_xfeat_init(MODEL_PATH);
    if (atomic_load(&USE_XFEAT) && xr_xfeat_available()) {
        int n = xr_xfeat_extract(img, work.kp_uv, work.desc.xfeat,
                                 XR_MAP_KP_PER_KF);
        if (n >= 0) {
            work.n_kp = n;
            work.desc_type = DESC_XFEAT;
            /* associate XFeat keypoints to landmarks by proximity (the
             * BAD path anchors descriptors AT the landmarks instead) */
            for (int j = 0; j < work.n_kp; j++) {
                work.lm_of_kp[j] = -1;
                /* association radius 8 px (was 4): XFeat's learned detector
                 * rarely fires within 4 px of the VIO's FAST corners — on
                 * 512x512 fisheye the 4 px join starved PnP verification to
                 * 0-12 usable 2D-3D pairs (bench matrix 1, corridor/room
                 * data) while matching itself outperformed BAD. 8 px ~ 1 deg
                 * at these focals, well inside VER_COS_TOL (~4 deg). */
                float bd = 64.0f;              /* 8 px squared */
                for (int i = 0; i < work.n_lm; i++) {
                    float du = work.lm_uv[i][0] - work.kp_uv[j][0];
                    float dv = work.lm_uv[i][1] - work.kp_uv[j][1];
                    float d = du * du + dv * dv;
                    if (d < bd) { bd = d; work.lm_of_kp[j] = i; }
                }
            }
        } else {
            bad_extract(img, &work);
        }
    } else {
        bad_extract(img, &work);
    }

    /* place-recognition embedding for retrieval pre-ranking. Cheap no-op
     * until a VPR model is registered (and permanently off after a failed
     * bring-up). Runs on this (map) thread at keyframe/search rate. */
    work.emb_dim = xr_vpr_embed(img, work.emb);
    work.has_emb = work.emb_dim > 0;

    /* ---- candidate search + geometric verification: LOCK-FREE. Only the
     * map thread WRITES the keyframe store; a concurrent reset / descriptor
     * switch clears KF_N (frees the slots) and bumps RESET_GEN, but never
     * touches KF[] contents or the KFO order array, so reading them here is
     * safe; KF_N and CORR ARE written by reset, so we use the DEQUEUE-time
     * snapshots (kfn,
     * snap_corr) instead of reading them unlocked. THIS is the
     * priority-inversion fix: the tens-of-ms brute-force match + top-K PnP
     * no longer runs under the lock the VIO worker needs. */
    int did_search = 0;
    int cand_i[RELOC_TOPK], cand_m[RELOC_TOPK], ncand = 0;
    int raw_best_m = 0, raw_best_i = -1;
    int best_i = -1, best_m = 0, best_n3 = 0, best_nin = 0, best_covis = 0;
    int any_pairs = 0;
    int searched = 0;                  /* keyframes actually match-scored */
    unsigned match_us = 0;             /* telemetry: match+PnP wall time */
    int use_vpr = 0;                   /* retrieval pre-rank active */
    float vpr_top = 0;                 /* best cosine this search (ledger) */
    float bDq[4], bDp[3];
    if (do_search) {
    did_search = 1;
    uint64_t t_match0 = map_mono_us();
    /* mapping skips the freshest keyframes (trivial self-matches); a
     * relocalization query (stationary, or LOST) searches EVERYTHING —
     * the pre-loss keyframes are the best recovery anchors */
    int relocalizing = q_only || lost;
    int lim = (mapping && !relocalizing) ? kfn - CAND_SKIP_RECENT : kfn;
    if (lim > kfn) lim = kfn;
    if (lim < 0) lim = 0;

    /* rank the top-K matching keyframes. K is adaptive: 1 while tracking
     * is healthy (cheap), RELOC_TOPK when relocalizing — a repetitive
     * scene can hand raw scoring to the wrong keyframe, so several clusters
     * each get a geometric try and the strongest wins. */
    int K = relocalizing ? RELOC_TOPK : 1;
    /* coarse spatial gate for HEALTHY loop-detection (skipped when
     * relocalizing — LOST always full-scans). Every FULL_SWEEP_EVERY-th
     * healthy search is a full sweep so a large-drift loop outside the gate
     * is still caught within ~a couple seconds. CORR is the map frame and
     * is written only by this (map) thread, so reading it lock-free here is
     * safe. */
    static int healthy_search_n;
    int gate = 0;
    float wsp[3] = { 0, 0, 0 };
    if (!relocalizing) {
        int sweep = (++healthy_search_n % FULL_SWEEP_EVERY) == 0;
        gate = !sweep;
        if (gate) {
            qrotv(snap_corr_q, work.p, wsp);
            wsp[0] += snap_corr_p[0]; wsp[1] += snap_corr_p[1];
            wsp[2] += snap_corr_p[2];
        }
    }
    /* VPR pre-rank: when the query and store carry embeddings, appearance
     * similarity (one 512-D dot per keyframe) selects a short list and ONLY
     * those get descriptor-scored. Replaces the spatial gate — appearance
     * needs no trusted pose, which is exactly what LOST lacks — and full
     * recall is preserved because every keyframe is ranked, not windowed. */
    static uint8_t vpr_pick[XR_MAP_MAX_KF];    /* map thread only */
    use_vpr = work.has_emb && lim > VPR_MIN_KF;
    if (use_vpr) {
        memset(vpr_pick, 0, (size_t)lim);
        float bs[VPR_SHORTLIST];
        int bi[VPR_SHORTLIST], bn = 0;
        for (int i = 0; i < lim; i++) {
            /* pre-VPR or other-model keyframes stay always-eligible */
            if (!KFA(i).has_emb || KFA(i).emb_dim != work.emb_dim) {
                vpr_pick[i] = 1;
                continue;
            }
            const float *a = work.emb, *b = KFA(i).emb;
            float s = 0;
            for (int d = 0; d < work.emb_dim; d++) s += a[d] * b[d];
            if (s > vpr_top) vpr_top = s;
            if (s < VPR_MIN_SIM) continue;
            int pos = bn < VPR_SHORTLIST
                          ? bn : (s > bs[VPR_SHORTLIST - 1] ? VPR_SHORTLIST - 1
                                                            : -1);
            if (pos < 0) continue;
            if (bn < VPR_SHORTLIST) bn++;
            while (pos > 0 && bs[pos - 1] < s) {
                bs[pos] = bs[pos - 1];
                bi[pos] = bi[pos - 1];
                pos--;
            }
            bs[pos] = s;
            bi[pos] = i;
        }
        for (int t = 0; t < bn; t++) vpr_pick[bi[t]] = 1;
        gate = 0;                      /* retrieval supersedes the gate */
    }
    for (int i = 0; i < lim; i++) {
        if (use_vpr && !vpr_pick[i]) continue;
        if (gate) {
            float gx = wsp[0] - KFA(i).pc[0], gy = wsp[1] - KFA(i).pc[1],
                  gz = wsp[2] - KFA(i).pc[2];
            if (gx * gx + gy * gy + gz * gz > SHORTLIST_R_M * SHORTLIST_R_M)
                continue;
        }
        searched++;
        int m = match_count(&work, &KFA(i));
        if (m > raw_best_m) { raw_best_m = m; raw_best_i = i; }
        if (m < CAND_MIN_MATCHES) continue;
        /* distinct CLUSTERS, not just distinct keyframes: reloc_pnp already
         * pools everything within COVIS_R of a candidate, so two candidates
         * that close together are the same cluster and waste a slot (the
         * review's point). Keep only the strongest keyframe per
         * neighbourhood, so the K slots cover K different places. */
        int near = -1;
        for (int t = 0; t < ncand; t++) {   /* SESSION-frame distance (pc) */
            float dx = KFA(i).pc[0] - KFA(cand_i[t]).pc[0];
            float dy = KFA(i).pc[1] - KFA(cand_i[t]).pc[1];
            float dz = KFA(i).pc[2] - KFA(cand_i[t]).pc[2];
            if (dx * dx + dy * dy + dz * dz < COVIS_R_M * COVIS_R_M) {
                near = t; break;
            }
        }
        if (near >= 0) {
            if (m <= cand_m[near]) continue;   /* weaker in-cluster: drop */
            for (int t = near; t < ncand - 1; t++) {   /* evict, then re-insert */
                cand_m[t] = cand_m[t + 1];
                cand_i[t] = cand_i[t + 1];
            }
            ncand--;
        }
        int pos = ncand < K ? ncand : (m > cand_m[K - 1] ? K - 1 : -1);
        if (pos < 0) continue;
        if (ncand < K) ncand++;
        while (pos > 0 && cand_m[pos - 1] < m) {
            cand_m[pos] = cand_m[pos - 1];
            cand_i[pos] = cand_i[pos - 1];
            pos--;
        }
        cand_m[pos] = m;
        cand_i[pos] = i;
    }

    /* verify every candidate cluster; keep the geometrically strongest
     * (most inliers, covisibility-backed) — still lock-free */
    for (int c = 0; c < ncand; c++) {
        int cn3 = 0, cnin = 0, ccov = 0;
        float cDq[4], cDp[3];
        int ok = reloc_pnp(&work, cand_i[c], kfn, cDq, cDp, &cn3, &cnin, &ccov);
        if (cn3 > any_pairs) any_pairs = cn3;
        if (ok && ccov >= COVIS_MIN_KF && cnin > best_nin) {
            best_i = cand_i[c]; best_m = cand_m[c];
            best_n3 = cn3; best_nin = cnin; best_covis = ccov;
            memcpy(bDq, cDq, sizeof bDq);
            memcpy(bDp, cDp, sizeof bDp);
        }
    }
    match_us = (unsigned)(map_mono_us() - t_match0);
    /* closure ledger: one line per search so retrieval recall/precision is
     * measurable offline (the benchmark GT-labels candidates post-hoc) */
    if (use_vpr)
        LOGI("session map: LEDGER q=%llu vprtop=%.3f searched=%d cand=%d "
             "bestm=%d n3=%d nin=%d lost=%d",
             (unsigned long long)work.ts, (double)vpr_top, searched, ncand,
             raw_best_m, any_pairs, best_nin, lost);
    }   /* end LOCK-FREE candidate search */

    /* ---- write / publish: LOCKED, but BRIEF (no match, no PnP). The VIO
     * worker's fast-path calls (offer / get_correction / get_cloud) now
     * contend only with this, never the heavy search — the priority
     * inversion is gone. */
    uint64_t t_lock0 = map_mono_us();
    pthread_mutex_lock(&MAP_LOCK);
    unsigned lock_us = (unsigned)(map_mono_us() - t_lock0);
    if (atomic_load(&RESET_GEN) != rgen0) {
        /* reset / descriptor switch happened while we processed this
         * (pre-change) offer: abandon it ENTIRELY — no stale correction,
         * and (the fix the audit caught) no stale / wrong-descriptor
         * keyframe inserted below, which would otherwise immediately
         * repopulate the freshly-cleared map. */
        pthread_mutex_unlock(&MAP_LOCK);
        return;
    }
    /* generation-safe recovery-state publication (see the note at the rstate
     * computation). The apply branch may still override this to REC_RECOVERED
     * on a confirmed closure, under this same lock hold. */
    atomic_store(&REC_STATE, rstate);
    if (did_search) {
        PERF.searched = searched;
        PERF.candidates = ncand;
        PERF.match_us = match_us;
        PERF.lock_us = lock_us;
        last_search_cost_us = match_us;   /* feeds the LOST cadence cost-cap */
    }
    memcpy(LAST_POSE.q, work.q, sizeof LAST_POSE.q);
    memcpy(LAST_POSE.p, work.p, sizeof LAST_POSE.p);
    LAST_POSE.have = 1;

    /* spatial recency: keyframes near the current position stay fresh, so
     * living in the same space never rolls its map away. Compared in the
     * SESSION frame (corrected) — a physically-near keyframe from a
     * different drifted odom epoch must still refresh. */
    float wsp2[3];
    qrotv(snap_corr_q, work.p, wsp2);
    wsp2[0] += snap_corr_p[0]; wsp2[1] += snap_corr_p[1]; wsp2[2] += snap_corr_p[2];
    for (int i = 0; i < KF_N; i++) {
        float dx = wsp2[0] - KFA(i).pc[0], dy = wsp2[1] - KFA(i).pc[1],
              dz = wsp2[2] - KFA(i).pc[2];
        if (dx * dx + dy * dy + dz * dz < KF_NEAR_M * KF_NEAR_M)
            KFA(i).last_used = work.ts;
    }

    /* apply the search result */
    if (did_search) {
    if (ncand == 0 && raw_best_i >= 0) {
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
                 raw_best_m, raw_best_i, KF_N, CAND_MIN_MATCHES);
        }
    } else if (ncand > 0) {
        LAST_CAND.a = work.ts;
        LAST_CAND.b = KFA(cand_i[0]).ts;
        LAST_CAND.matches = cand_m[0];
        LAST_CAND.have = 1;

        if (best_i < 0) {
            /* candidates matched but none verified into an inlier-backed
             * cluster of >= COVIS_MIN_KF keyframes — junk / shake-spawned /
             * a repetitive wrong place */
            VER_LAST.pairs = any_pairs;
            VER_LAST.inliers = 0;
            VER_LAST.outcome = any_pairs >= VER_MIN_PAIRS
                                   ? VOUT_FEW_INLIERS : VOUT_FEW_PAIRS;
            PENDING_D.have = 0;                /* break any confirmation run */
            LOGI("session map: %s %d cluster(s) best %d matches (%s) — "
                 "unverified (%d pairs)", mapping ? "LOOP" : "RELOC", ncand,
                 cand_m[0], work.desc_type == DESC_XFEAT ? "xfeat" : "bad",
                 any_pairs);
        } else {
            int n3 = best_n3, nin = best_nin, covis = best_covis;
            float Dq[4], Dp[3];
            memcpy(Dq, bDq, sizeof Dq);
            memcpy(Dp, bDp, sizeof Dp);
            VER_LAST.pairs = n3;
            VER_LAST.inliers = nin;
            KFA(best_i).last_used = work.ts;   /* geometrically useful */
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

            /* does this pose agree with a pending one? (2nd-frame confirm) */
            int confirmed = 0;
            if (PENDING_D.have && work.ts > PENDING_D.ts &&
                work.ts - PENDING_D.ts < CONFIRM_WINDOW_NS) {
                float pns[3], dp2 = 0;
                qrotv(PENDING_D.q, work.p, pns);
                for (int c = 0; c < 3; c++) {
                    float d = (ns[c] + Dp[c]) - (pns[c] + PENDING_D.p[c]);
                    dp2 += d * d;
                }
                float pqi[4], pqe[4], prv[3];
                qconj(PENDING_D.q, pqi);
                qmul(Dq, pqi, pqe);
                rv_from_q(pqe, prv);
                float da = sqrtf(prv[0] * prv[0] + prv[1] * prv[1] +
                                 prv[2] * prv[2]);
                confirmed = dp2 < CONFIRM_DP_M * CONFIRM_DP_M &&
                            da < CONFIRM_DA_RAD;
            }

            /* A verified alignment is worth pursuing when: we are LOST (ANY
             * alignment is a recovery candidate — even a small deviation,
             * from a shake that didn't actually disturb Basalt), OR we are
             * healthy and the VIO has drifted SIGNIFICANTLY from the map and
             * the cooldown has passed. Otherwise the VIO already agrees. */
            int worth = lost || (significant && cooled);

            if (dev > mxt || sang > mxa) {
                VER_LAST.outcome = VOUT_CAPPED;
                PENDING_D.have = 0;
                LOGI("session map: kf#%d PnP GOOD (%d/%d inliers, %d covis) "
                     "but |t|=%.2fm ang=%.0fdeg exceeds caps — wrong-place "
                     "match, ignored", best_i, nin, n3, covis, (double)dev,
                     (double)(sang * 57.3f));
            } else if (!worth) {
                /* healthy and the VIO agrees with the map (or we just
                 * snapped): do nothing — a recovery, not a continuous clamp */
                VER_LAST.outcome = VOUT_GATED;
                PENDING_D.have = 0;
            } else if (!confirmed) {
                /* one good frame is not enough — wait for a 2nd that agrees */
                VER_LAST.outcome = VOUT_PENDING;
                PENDING_D.have = 1;
                PENDING_D.ts = work.ts;
                memcpy(PENDING_D.q, Dq, sizeof PENDING_D.q);
                memcpy(PENDING_D.p, Dp, sizeof PENDING_D.p);
                LOGI("session map: kf#%d %s %.2fm %.0fdeg (%d/%d inliers, "
                     "%d covis) — awaiting a confirming frame",
                     best_i, lost ? "reloc" : "loop", (double)dev,
                     (double)(sang * 57.3f), nin, n3, covis);
            } else {
                /* CONFIRMED. The MAP correction ALWAYS advances so new
                 * keyframes attach in the healed frame; the DISPLAY
                 * correction only snaps when recovery is on. */
                int snap = atomic_load(&RECOVERY);
                if (lost) {
                    /* POST-LOSS RECOVERY. Storage was frozen through the
                     * discontinuity, so the stored map is already correct —
                     * only REGISTER the new live odom frame back to it. Do
                     * NOT deform the reference (that would corrupt a correct
                     * map — the review's critical finding). */
                    atomic_store(&REC_STATE, REC_RECOVERED);
                    RECOVERED_NS = work.ts;
                    LOGI("session map: kf#%d RELOC RECOVERED %.2fm %.0fdeg "
                         "(%d/%d inliers, %d covis) — live frame registered, "
                         "stored map held fixed%s", best_i, (double)dev,
                         (double)(sang * 57.3f), nin, n3, covis,
                         snap ? " + pose snapped" : "");
                } else {
                    /* HEALTHY accumulated-drift closure: DEFORM the drifted
                     * tail onto the reference (real co-localization). Pass
                     * the query's SESSION position (CORR is still the
                     * pre-closure correction here) so the path weighting is
                     * discontinuity-safe. */
                    float qsp[3];
                    qrotv(CORR.q, work.p, qsp);
                    qsp[0] += CORR.p[0]; qsp[1] += CORR.p[1]; qsp[2] += CORR.p[2];
                    graph_deform(best_i, qsp, Dq, Dp);
                    CLOUD_DIRTY = 1;
                    LOGI("session map: LOOP kf#%d CLOSURE %.2fm %.0fdeg "
                         "(%d/%d inliers, %d covis) — map deformed%s", best_i,
                         (double)dev, (double)(sang * 57.3f), nin, n3, covis,
                         snap ? " + pose snapped" : "");
                }
                memcpy(CORR.q, Dq, sizeof CORR.q);
                memcpy(CORR.p, Dp, sizeof CORR.p);
                CORR.gen++;
                LAST_SNAP_NS = work.ts;
                if (snap) {
                    memcpy(LIVE.q, Dq, sizeof LIVE.q);
                    memcpy(LIVE.p, Dp, sizeof LIVE.p);
                    LIVE.gen++;
                }
                PENDING_D.have = 0;
                LOOP_STATS.count++;
                LOOP_STATS.matches = best_m;
                VER_LAST.outcome = VOUT_APPLIED;
            }

            /* AR flash + panel marker: the winner's landmarks in session */
            {
                float qoi[4], poi[3], qdi[4], pdi[3];
                pose_invert(KFA(best_i).q, KFA(best_i).p, qoi, poi);
                pose_compose(KFA(best_i).qc, KFA(best_i).pc, qoi, poi, qdi, pdi);
                memcpy(LOOP_STATS.pos, KFA(best_i).pc, sizeof LOOP_STATS.pos);
                LOOP_STATS.n_lm = KFA(best_i).n_lm;
                for (int i = 0; i < KFA(best_i).n_lm; i++) {
                    float t[3];
                    qrotv(qdi, KFA(best_i).lm_xyz[i], t);
                    LOOP_STATS.lm[i][0] = t[0] + pdi[0];
                    LOOP_STATS.lm[i][1] = t[1] + pdi[1];
                    LOOP_STATS.lm[i][2] = t[2] + pdi[2];
                }
            }
        }
    }
    }   /* end apply (verified result, under the lock) */

    /* store (mapping mode only; never for a stationary query — those are
     * matching-only; only frames that carry verifiable geometry, at a
     * bounded rate), rolling cap: evict least-recently-useful. Reuse the
     * SAME may_store gate computed above — it already carries the LOST
     * check, so a search-only pass during a shake/loss can never fall
     * through and insert the contaminated frame. */
    if (may_store) {
        LAST_STORE_NS = work.ts;
        if (KF_N == XR_MAP_MAX_KF) {
            /* evict the least-recently-useful keyframe: free its SLOT and drop
             * it from the time order. Only the small order array shifts (a few
             * hundred ints), never the ~4 MB of keyframe structs. */
            int vpos = 0;
            for (int i = 1; i < KF_N; i++)
                if (KFA(i).last_used < KFA(vpos).last_used) vpos = i;
            KF_FREE[KF_FREE_N++] = KFO[vpos];      /* slot returns to the pool */
            memmove(&KFO[vpos], &KFO[vpos + 1],
                    sizeof(int) * (size_t)(KF_N - 1 - vpos));
            KF_N--;
        }
        work.last_used = work.ts;
        /* corrected session pose = current global correction ∘ odom. A
         * confirmed closure this same pass already deformed the tail and
         * (if recovering) updated CORR, so the new tip lands consistent. */
        pose_compose(CORR.q, CORR.p, work.q, work.p, work.qc, work.pc);
        int slot = KF_FREE[--KF_FREE_N];           /* a stable slot for life */
        KF[slot] = work;
        KFO[KF_N] = slot;                          /* append in time order */
        KF_N++;
        atomic_store(&KF_COUNT_PUB, KF_N);
        CLOUD_DIRTY = 1;
        LOGI("session map: kf#%d stored (%d landmarks, %d kps)",
             KF_N - 1, work.n_lm, work.n_kp);
    }
    /* refresh the authoritative display cloud whenever the graph changed
     * (a store, an eviction, or a closure-driven deformation) */
    if (CLOUD_DIRTY) cloud_rebuild();
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
