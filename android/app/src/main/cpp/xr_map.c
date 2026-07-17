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

#include "xr_slam.h"
#include "xr_lighterglue.h"
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
/* Tunables below are #ifndef-guarded so the bench can sweep them via
 * compiler -D without touching source (make EXTRA="-DSNAP_MIN_M=0.15f"). */
#ifndef SNAP_MIN_M
/* 0.50 beat 0.30 in the first REAL sweep (the historic "0.50" sweep was
 * identical binaries — EXTRA was unwired): hunt-subset map median 18.2 vs
 * 26.8 cm, corridor1 30->10, corridor5 32->20, magistrale2 55->45; only
 * corridor2 regressed (+7). Small spaces flat. */
#define SNAP_MIN_M 0.50f
#endif
#ifndef SNAP_MIN_ANG_RAD
#define SNAP_MIN_ANG_RAD 0.14f     /* ~8 deg */
#endif
#ifndef SNAP_COOLDOWN_NS
#define SNAP_COOLDOWN_NS 1500000000ull
#endif
/* the matched place must be an ESTABLISHED cluster: this many DISTINCT
 * keyframes must each supply a geometric INLIER (not just a raw match) —
 * a lone shake-spawned keyframe or a two-frame junk cluster is rejected */
#define COVIS_MIN_KF 3
/* a proposed correction is applied only after a SECOND verified query
 * within this window agrees on the pose — one confident-but-wrong PnP
 * frame never snaps on its own */
/* Confirmation window must span at least two retrieval queries. Heavy VPR
 * embeddings throttle query cadence (MegaLoc ~229 ms/embed on CPU => one
 * query per ~6 s in replay; big models on NPU can be slow too) — at 4 s the
 * window could not contain two MegaLoc queries, structurally zeroing its
 * loop closures. 12 s keeps the 2-frame agreement requirement meaningful
 * while surviving slow-embed cadences. */
#define CONFIRM_WINDOW_NS 12000000000ull
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
#ifndef VPR_SHORTLIST
#define VPR_SHORTLIST 12
#endif
#define VPR_MIN_SIM 0.25f
#define VPR_MIN_KF (VPR_SHORTLIST + 4)  /* smaller maps: full scan is cheap */
/* HEALTHY loop-detection coarse gate: only full-match keyframes whose
 * SESSION position is within this radius of the corrected pose (a revisit
 * is where you physically are). A FULL sweep every Nth healthy search
 * still catches a large-residual-drift loop the gate would miss. LOST
 * recovery ignores the gate (recall must not regress). This cuts the
 * dominant per-search matching from O(all keyframes) toward O(local). */
#define SHORTLIST_R_M 10.0f
#ifndef FULL_SWEEP_EVERY
#define FULL_SWEEP_EVERY 8
#endif
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
    /* same-session submap id: 0 = the primary (session-frame-registered)
     * map; >0 = a segment started after an unrecoverable loss, registered
     * only to its own odom continuation until a cross-segment verification
     * WELDS it into the primary (Atlas-style merge). */
    int seg;
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

/* ---- same-session submaps. An unrecoverable loss no longer freezes the
 * map forever (the 5588-keyframe field freeze): once relocalization has had
 * SEG_OPEN_NS of LOST to try and the physical shake is over, a NEW segment
 * opens (CUR_SEG++) and mapping resumes. Segment keyframes are registered
 * only to their own odom continuation — internally consistent, offset from
 * the primary by the unknown discontinuity — until a cross-segment
 * verified+confirmed closure WELDS the frames together rigidly
 * (Atlas-style merge). Session-frame (pc) distances are meaningless ACROSS
 * segments, so pooling / clustering / gating are all seg-guarded below. */
#ifndef SEG_OPEN_NS
#define SEG_OPEN_NS 10000000000ull     /* LOST this long (shake over) -> new submap */
#endif
#ifndef COVK_CELL_M
#define COVK_CELL_M 1.5f               /* XR_COVKEEP spatial cell (m) */
#endif
static int CUR_SEG;                    /* active segment; 0 = primary. Map
                                          thread writes under MAP_LOCK */
static uint64_t LOST_SINCE_NS;         /* map thread only: when LOST began */

/* XR_SEQVOTE: decaying per-SLOT retrieval votes across consecutive
 * relocalizing searches (SeqSLAM idea): a place that keeps scoring 0.35
 * for ten frames should beat a single 0.5 spike — repetitive corridors
 * fail single-frame appearance but survive temporal consistency. */
#ifndef SEQV_W
#define SEQV_W 0.5f
#define SEQV_DECAY 0.75f
#endif
static float SEQV[XR_MAP_MAX_KF];      /* physical-slot indexed, map thread */
/* XR_DESPERATE: after this long LOST (or for one-shot probes), widen the
 * retrieval shortlist and lower the similarity floor — nothing to lose. */
#ifndef DESPERATE_AFTER_NS
#define DESPERATE_AFTER_NS 5000000000ull
#define DESPERATE_MIN_SIM 0.15f
#endif
#define VPR_SHORTLIST_MAX 32
/* XR_ROTSTORE: allow an early store on a heading change > ~25 deg —
 * time-gated storage under-samples turns, which is exactly the reverse-
 * viewpoint coverage the reloc bestm~0 mode is missing. */
#ifndef ROTSTORE_RAD
#define ROTSTORE_RAD 0.44f
#endif
static float LAST_STORE_Q[4];          /* map thread only */

/* ---- lifetime landmark descriptor bank (XR_LMDESC). A direct-mapped
 * CACHE of the freshest descriptor per landmark id, across every keyframe
 * that ever observed it — so 2D-3D association stops depending on which
 * single keyframe retrieval happens to surface (the reloc bestm~0
 * coverage mode's matching-side half), and the direct relocalization
 * channel below can match query descriptors straight against landmarks.
 * Entries reference their newest observer (physical slot + landmark
 * index); validity is re-checked at use time (slots get reused by
 * eviction). Written only by the map thread under MAP_LOCK; read only by
 * the map thread — no locking subtleties. */
#ifndef LMB_MAX
#define LMB_MAX 8192                   /* direct-mapped: collisions overwrite */
#endif
typedef struct {
    int32_t id;
    int8_t desc[64];                   /* 32 used for BAD, 64 for XFeat */
    uint8_t desc_type;
    uint8_t li;                        /* landmark index in the owner kf */
    int16_t slot;                      /* newest observer (PHYSICAL slot) */
    uint16_t nobs;                     /* 0 = empty */
    uint64_t ts;
} lmb_ent;
static lmb_ent LMB[LMB_MAX];
static int LMB_LIVE;                   /* populated entries (map thread) */
/* Reset must not memset the bank from a JNI thread while the map thread's
 * lock-free search scans it (review finding): resets bump the epoch and
 * the MAP THREAD clears the bank lazily at its next dequeue. */
static atomic_uint LMB_EPOCH;
static inline uint32_t lmb_slot_of(int32_t id) {
    return ((uint32_t)id * 2654435761u) % LMB_MAX;
}

/* Empty the map: every slot free, no keyframes. Run at load and on every
 * clear (reset / descriptor switch) so KF_FREE is always valid before a
 * store. */
static void kf_slots_reset(void) {
    KF_N = 0;
    KF_FREE_N = XR_MAP_MAX_KF;
    for (int i = 0; i < XR_MAP_MAX_KF; i++) KF_FREE[i] = i;
    CUR_SEG = 0;
    atomic_fetch_add(&LMB_EPOCH, 1);   /* bank voided: map thread clears it */
    memset(SEQV, 0, sizeof SEQV);
    memset(LAST_STORE_Q, 0, sizeof LAST_STORE_Q);
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
    int probe;                         /* reloc-benchmark probe (no apply) */
    int bfirst, blast;                 /* XR_BURSTPNP burst framing */
    float bp[3];                       /* burst-relative VIO position */
    uint64_t ts;
    float q[4], p[3];
    uint8_t img[XR_OW * XR_OH];
    uint8_t img2[XR_OW * XR_OH];       /* XR_DEPTHFILL: right eye (opt) */
    int have2;
    int n_lm;
    int32_t lm_id[XR_MAP_KP_PER_KF];
    float lm_xyz[XR_MAP_KP_PER_KF][3];
    float lm_uv[XR_MAP_KP_PER_KF][2];
} MBOX;
/* XR_BURSTPNP accumulator (map thread only): matches from every frame of
 * the wake-up burst, in a SHARED gravity-aligned odom frame, with
 * per-match camera-origin offsets. */
#define BURST_MAX 512
#define BURST_EXPORT_MAX 128           /* per-frame contribution cap */
static struct {
    int active, n, nframes;
    float S[BURST_MAX][3];             /* query bearings (burst odom frame) */
    float P[BURST_MAX][3];             /* matched landmarks (session frame) */
    float O[BURST_MAX][3];             /* camera-center offset vs frame 0 */
    uint8_t fid[BURST_MAX];            /* source frame within the burst */
} BURST;
static struct { int ok, nin; float Dq[4], Dp[3]; } BSOLVE; /* map thread */
/* Reloc-benchmark probe result mailbox (guarded by MAP_LOCK/MAP_COND).
 * A probe runs the full retrieval+PnP pipeline on a bare image with an
 * identity odom pose, so the verified alignment D IS the query's pose in
 * the SESSION frame. Nothing is applied or stored. */
static struct {
    int done, ok, inliers, kf;
    float q[4], p[3];                  /* session-frame pose of the query */
} PROBE_RES;
static int PROBE_REQ;                  /* map thread only (set at dequeue) */
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

/* Register the LighterGlue matcher — XFeat-descriptor verification
 * switches from greedy NN+margin to learned matching (reloc-recall
 * feature; retrieval ranking stays NN). No-op for BAD/TEBLID maps. */
void xr_map_set_lglue_model(const char *onnx_path) {
    xr_lglue_set_model(onnx_path);
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

/* Correction RAMP (XR_RAMP env): glide the DISPLAY correction to a new
 * target over RAMP_MS instead of stepping it — approximates OKVIS2's
 * optimization-spread corrections for the output pose (causal ATE stops
 * paying the step penalty; AR display stops visibly snapping). The map
 * (CORR, keyframes, cloud) still updates instantly — only the emitted
 * pose is smoothed. */
#ifndef RAMP_MS
#define RAMP_MS 500.0f
#endif
static int ramp_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_RAMP");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: correction RAMP ON (%.0f ms)", (double)RAMP_MS);
    }
    return v;
}

int xr_map_get_correction(float q[4], float p[3]) {
    pthread_mutex_lock(&MAP_LOCK);
    memcpy(q, LIVE.q, sizeof LIVE.q);   /* the DISPLAY correction */
    memcpy(p, LIVE.p, sizeof LIVE.p);
    int g = LIVE.gen;
    pthread_mutex_unlock(&MAP_LOCK);
    if (ramp_on()) {
        static struct {
            int gen;                     /* target gen being ramped to */
            float q0[4], p0[3];          /* ramp start (previous output) */
            float qo[4], po[3];          /* last emitted output */
            struct timespec t0;
            int have;
        } R;
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (!R.have) {
            /* first call: seed START and OUTPUT from the live target and
             * emit it unramped. (q0/p0 left zero here NaN'd the first pose:
             * a=0 blend of a zero quaternion normalizes 0 -> inf -> NaN.) */
            memcpy(R.q0, q, 16); memcpy(R.p0, p, 12);
            memcpy(R.qo, q, 16); memcpy(R.po, p, 12);
            R.gen = g; R.have = 1;
            R.t0.tv_sec = now.tv_sec - 1; R.t0.tv_nsec = now.tv_nsec;
        }
        if (g != R.gen) {                /* new target: ramp from last output */
            memcpy(R.q0, R.qo, 16); memcpy(R.p0, R.po, 12);
            R.gen = g; R.t0 = now;
        }
        float dt_ms = (float)(now.tv_sec - R.t0.tv_sec) * 1000.f +
                      (float)(now.tv_nsec - R.t0.tv_nsec) * 1e-6f;
        float a = dt_ms >= RAMP_MS ? 1.f : dt_ms / RAMP_MS;
        if (a < 1.f) {
            float dot = R.q0[0]*q[0] + R.q0[1]*q[1] + R.q0[2]*q[2] + R.q0[3]*q[3];
            float sgn = dot < 0 ? -1.f : 1.f, n2 = 0;
            for (int c = 0; c < 4; c++) {
                q[c] = (1 - a) * R.q0[c] * sgn + a * q[c];
                n2 += q[c] * q[c];
            }
            n2 = 1.f / sqrtf(n2);
            for (int c = 0; c < 4; c++) q[c] *= n2;
            for (int c = 0; c < 3; c++) p[c] = (1 - a) * R.p0[c] + a * p[c];
        }
        memcpy(R.qo, q, 16); memcpy(R.po, p, 12);
    }
    return g;
}

/* XR_MAPSEED (stage-3-lite map->VIO coupling): when the live pose sits
 * near a stored keyframe in the session frame, hand back that keyframe's
 * landmark pixel uvs so the caller can seed the VIO's optical-flow
 * detector (xr_slam_seed_keypoints) — the tracker re-acquires the SAME
 * physical corners the map already triangulated, densifying
 * re-observations of mapped structure exactly where closures verify.
 * Full stage 3 (landmark 3D priors in the estimator) rides on top later;
 * this needs no camera projection model at all. */
#ifndef MAPSEED_R_M
#define MAPSEED_R_M 0.6f
#endif
#ifndef MAPSEED_ANG_RAD
#define MAPSEED_ANG_RAD 0.45f          /* ~25 deg */
#endif
static int mapseed_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_MAPSEED");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: MAP-SEED (reseed mapped corners) ON");
    }
    return v;
}
static void qmul(const float a[4], const float b[4], float o[4]);
static void qconj(const float a[4], float o[4]);
static void qrotv(const float q[4], const float v[3], float o[3]);
static void rv_from_q(const float q[4], float rv[3]);

int xr_map_get_reseed(const float q[4], const float p[3],
                      float (*uv)[2], int max) {
    if (!mapseed_on() || max <= 0) return 0;
    pthread_mutex_lock(&MAP_LOCK);
    /* live session pose = CORR ∘ odom */
    float sq[4], sp[3], t[3];
    qmul(CORR.q, q, sq);
    qrotv(CORR.q, p, t);
    sp[0] = t[0] + CORR.p[0]; sp[1] = t[1] + CORR.p[1];
    sp[2] = t[2] + CORR.p[2];
    int best = -1;
    float bd2 = MAPSEED_R_M * MAPSEED_R_M;
    for (int i = 0; i < KF_N; i++) {
        if (KFA(i).seg != CUR_SEG) continue;
        float dx = sp[0] - KFA(i).pc[0], dy = sp[1] - KFA(i).pc[1],
              dz = sp[2] - KFA(i).pc[2];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 >= bd2) continue;
        float qi[4], qe[4], rv[3];
        qconj(KFA(i).qc, qi);
        qmul(qi, sq, qe);
        rv_from_q(qe, rv);
        if (rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2] >
            MAPSEED_ANG_RAD * MAPSEED_ANG_RAD)
            continue;
        bd2 = d2;
        best = i;
    }
    int n = 0;
    if (best >= 0) {
        n = KFA(best).n_lm;
        if (n > max) n = max;
        memcpy(uv, KFA(best).lm_uv, sizeof(float) * 2 * (size_t)n);
    }
    pthread_mutex_unlock(&MAP_LOCK);
    return n;
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

/* --- map->VIO tight coupling (XR_TIGHT env, bench A/B) ------------------
 * Instead of stepping CORR / deforming on a verified alignment, hand the
 * residual E = CORR^-1 * D to the VIO optimizer as a weak unary pose prior
 * (target = E * T_newest). The estimator arbitrates the pull against IMU
 * and vision factors, spreading the correction smoothly; our subsequent
 * deviation measurements shrink as it absorbs. */
#define TIGHT tight_mode()
static int tight_mode(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_TIGHT");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: TIGHT map->VIO coupling ON");
    }
    return v;
}
/* All TIGHT_* tunables are #ifndef-guarded for bench -D sweeps (a bare
 * define silently OVERRODE -DTIGHT_MAX_DEV_M=... — later definition wins —
 * which invalidated a fastbench round; never leave sweepables unguarded). */
#ifndef TIGHT_SIGMA_T
#define TIGHT_SIGMA_T 0.07f
#endif
#ifndef TIGHT_SIGMA_R
#define TIGHT_SIGMA_R 0.035f            /* ~2 deg */
#endif
#ifndef TIGHT_EXPIRY_NS
#define TIGHT_EXPIRY_NS 700000000ull    /* prior lives 0.7 s of frame time */
#endif
/* Hybrid split (fleet v7 evidence): weak priors absorb SMALL corrections
 * perfectly (EuRoC/rooms reached VIO parity) but cannot move meter-scale
 * corridor drift (long lost its 14cm gains, map==VIO). Above this deviation
 * a confirmed closure takes the classic snap+deform path even in TIGHT.
 * 0.0 = confirmed closures ALWAYS snap; priors remain sub-gate only. */
#ifndef TIGHT_MAX_DEV_M
#define TIGHT_MAX_DEV_M 0.60f
#endif
#ifndef TIGHT_MAX_DEV_ANG
#define TIGHT_MAX_DEV_ANG 0.35f         /* ~20 deg */
#endif
/* Sub-gate priors only against GENUINE REVISITS (fleet v9 verdict): an
 * agreement with a recently-stored keyframe carries the same drift as the
 * live pose — the prior then glues the VIO to its own drift and prevents
 * the big correcting closure from ever crossing the snap gate (long lost
 * 31→43 across every always-on tight config). Old-keyframe agreement is
 * real loop information. */
#ifndef TIGHT_REVISIT_NS
#define TIGHT_REVISIT_NS 30000000000ull /* 30 s */
#endif

/* --- reactivation-lite (XR_REACT env; OKVIS2's segment-reactivation analog)
 * After a verified alignment, keep verifying against THAT keyframe at a
 * fast cadence instead of re-running retrieval — "tracking against the
 * map" while inside mapped space. Each success refreshes the anchor and
 * (under XR_TIGHT + revisit gate) feeds the optimizer priors; repeated
 * failures drop back to normal retrieval. Cost: one match+PnP against a
 * single keyframe per pass — the cheapest query the layer can run. */
#define REACT react_mode_on()
static int react_mode_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_REACT");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: REACTIVATION-LITE ON");
    }
    return v;
}
#ifndef REACT_IV_NS
#define REACT_IV_NS 400000000ull        /* verify cadence inside mapped space */
#endif
#ifndef REACT_WINDOW_NS
#define REACT_WINDOW_NS 15000000000ull  /* anchor expires w/o a success */
#endif
#ifndef REACT_MAX_FAILS
#define REACT_MAX_FAILS 3
#endif
static struct { int kf; uint64_t ts; int fails; } REACT_A = { -1, 0, 0 };

/* --- confidence-weighted correction (XR_CONFW env) -----------------------
 * The EuRoC finding: a geometrically-correct closure onto a WEAKER map
 * anchor moves a good VIO toward map error. Weight the applied correction
 * by verification strength (inlier ratio as the confidence proxy): strong
 * alignments apply fully, marginal ones fractionally — the map keeps its
 * evidence, the pose keeps its stability. LOST recovery always snaps. */
static int confw_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_CONFW");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: CONFIDENCE-WEIGHTED corrections ON");
    }
    return v;
}

/* A/B iteration flags (all default OFF; see bench/ITERATIONS.md):
 *   XR_COVKEEP  — coverage-aware eviction (viewpoint diversity)
 *   XR_SEGQUIET — no display snaps from within-segment closures while an
 *                 unwelded submap is active
 *   XR_PGO      — Gauss-Seidel pose-graph relaxation instead of the
 *                 path-weighted graph_deform
 *   XR_LMDESC   — lifetime landmark descriptor bank + direct 2D-3D
 *                 relocalization channel
 *   XR_REACT2   — reactivation anchor as an ADDITIONAL candidate (never
 *                 pins/replaces the main scan like XR_REACT does) */
static int covkeep_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_COVKEEP");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: COVERAGE-AWARE eviction ON");
    }
    return v;
}
static int segquiet_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_SEGQUIET");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: SEGMENT-QUIET display ON");
    }
    return v;
}
static int pgo_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_PGO");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: POSE-GRAPH relaxation ON");
    }
    return v;
}
static int lmdesc_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_LMDESC");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: LANDMARK DESCRIPTOR BANK ON");
    }
    return v;
}
static int react2_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_REACT2");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: REACTIVATION-AS-CANDIDATE ON");
    }
    return v;
}
static int relocsweep_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_RELOCSWEEP");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: RELOC FULL-SWEEP ON (no shortlist when lost)");
    }
    return v;
}
static int storeguard_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_STOREGUARD");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: STORE-GUARD ON (no embeds on store-only passes)");
    }
    return v;
}
static int tightsub_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_TIGHTSUB");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: TIGHT SUB-GATE confirmations ON");
    }
    return v;
}
static int seqvote_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_SEQVOTE");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: SEQUENCE VOTING (reloc retrieval) ON");
    }
    return v;
}
static int desperate_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_DESPERATE");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: DESPERATION WIDENING (long-LOST retrieval) ON");
    }
    return v;
}
static int rotstore_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_ROTSTORE");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: ROTATION-GATED storage ON");
    }
    return v;
}
/* XR_LMFACT — stage-3 map->VIO coupling: after a verified closure /
 * relocalization, feed the RE-OBSERVED MAP LANDMARKS (the PnP inlier 2D-3D
 * pairs) into the estimator as reprojection factors with FIXED 3D. Unlike
 * the pose-prior channel, arbitration is PER POINT inside the optimizer, so
 * it is safe at any revisit age — the mechanism OKVIS2 uses to push error
 * below VIO drift on room-scale revisits the 30s revisit-age gate excludes.
 * Needs the extended libbasalt (vit_tracker_xreal_landmark_factors); on a
 * stock build the post is a no-op and behavior is bit-identical. */
static int lmfact_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_LMFACT");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: LANDMARK-FACTOR map->VIO coupling ON");
    }
    return v;
}
/* XR_FARBEAR — bearing-only FAR landmarks in relocalization PnP. Matched
 * keypoints whose landmark never triangulated (or sits past the range
 * gate) are exactly the distant structure that dominates outdoor scenes;
 * the drive funnel showed healthy descriptor matching collapsing at the
 * 3D gate (bestm 42 -> n3 7-21). Their owner-keyframe ray is a world
 * DIRECTION: parallax-free at far range, it cannot range the camera but
 * it votes on YAW — the exact unknown the 4-DOF solver estimates. Far
 * pairs add yaw hypotheses + inlier votes; translation stays near-only. */
static int farbear_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_FARBEAR");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: FAR-BEARING reloc PnP ON");
    }
    return v;
}
#ifndef LMFACT_MAX
#define LMFACT_MAX 32               /* factors per frame (basalt-side cap) */
#endif
#ifndef LMFACT_SIGMA_PX
#define LMFACT_SIGMA_PX 2.0f        /* reprojection measurement std [px] */
#endif
/* XR_LMF_SIGMA overrides the measurement std at runtime — the weight is a
 * per-call channel argument, so strength sweeps need no basalt rebuild.
 * Smaller sigma = stronger pull toward the map's 3D. */
static float lmf_sigma(void) {
    static float v = -1.0f;
    if (v < 0) {
        const char *e = getenv("XR_LMF_SIGMA");
        v = (e && *e) ? (float)atof(e) : LMFACT_SIGMA_PX;
        if (v <= 0.01f || v > 50.0f) v = LMFACT_SIGMA_PX;
        if (v != LMFACT_SIGMA_PX)
            LOGI("session map: LMFACT sigma override %.2f px", (double)v);
    }
    return v;
}
#ifndef CONFW_MIN_W
#define CONFW_MIN_W 0.4f       /* weakest accepted alignment applies 40% */
#endif
#ifndef CONFW_FULL_RATIO
#define CONFW_FULL_RATIO 0.65f /* inlier ratio at which weight reaches 1 */
#endif
static void tight_post_prior(const float Dq[4], const float Dp[3],
                             uint64_t ts) {
    float ci[4], Eq[4], Ep[3], d[3];
    qconj(CORR.q, ci);
    qmul(ci, Dq, Eq);                       /* E.q = CORR.q^-1 * D.q */
    d[0] = Dp[0] - CORR.p[0];
    d[1] = Dp[1] - CORR.p[1];
    d[2] = Dp[2] - CORR.p[2];
    qrotv(ci, d, Ep);                       /* E.p = R(q^-1)(D.p - CORR.p) */
    float q_xyzw[4] = { Eq[1], Eq[2], Eq[3], Eq[0] };   /* wxyz -> xyzw */
    xr_slam_pose_prior(q_xyzw, Ep, TIGHT_SIGMA_T, TIGHT_SIGMA_R,
                       ts + TIGHT_EXPIRY_NS);
}

/* XR_LMFACT posting: hand the verification's inlier 2D-3D pairs to the
 * estimator as fixed-3D reprojection factors on the query frame. The
 * landmark positions are SESSION-frame (what PnP consumed); the estimator
 * lives in the ODOM frame (session = CORR ∘ odom), so transform each point
 * back via CORR⁻¹. With p_o = CORR⁻¹(p_s) the camera-frame geometry at the
 * pose-prior target T' = CORR⁻¹∘D∘T is IDENTICAL to the session-frame
 * geometry the verification proved — the factors pull the odom state toward
 * exactly the same place, but per-point and optimizer-arbitrated.
 * Called under MAP_LOCK (CORR stable); map thread only. */
static void lmfact_post(const float (*uv)[2], const float (*ps)[3], int n,
                        uint64_t ts) {
    if (n <= 0) return;
    if (n > LMFACT_MAX) n = LMFACT_MAX;
    float ci[4];
    qconj(CORR.q, ci);
    static float fuv[LMFACT_MAX * 2], fxyz[LMFACT_MAX * 3]; /* map thread */
    for (int m = 0; m < n; m++) {
        float d[3], po[3];
        d[0] = ps[m][0] - CORR.p[0];
        d[1] = ps[m][1] - CORR.p[1];
        d[2] = ps[m][2] - CORR.p[2];
        qrotv(ci, d, po);              /* odom = R(CORR.q⁻¹)(session - CORR.p) */
        fuv[2 * m] = uv[m][0];
        fuv[2 * m + 1] = uv[m][1];
        fxyz[3 * m] = po[0];
        fxyz[3 * m + 1] = po[1];
        fxyz[3 * m + 2] = po[2];
    }
    if (xr_slam_landmark_factors(ts, 0, fuv, fxyz, n, lmf_sigma()) > 0)
        LOGI("session map: LMFACT posted %d landmark factors", n);
}

/* XR_LMTRACK — CONTINUOUS landmark-factor coverage. LMFACT alone posts
 * only at closure instants; between them the estimator is back on raw
 * VIO. Hold the last applied closure's inlier landmarks (descriptor +
 * session 3D) for a short window and re-match them against every later
 * search frame's keypoints — each hit re-posts factors for THAT frame,
 * so the map keeps pulling while the scene stays in view (the sliding
 * window renews on every successful re-match). Map thread only. */
#define LMT_WINDOW_NS 2000000000ull    /* 2 s since last successful match */
#define LMT_MIN_MATCH 10
static struct {
    int n;
    uint64_t until_ns;
    int8_t desc[LMFACT_MAX][64];
    float ps[LMFACT_MAX][3];           /* session frame */
} LMT;
/* XR_TRUSTVPR — while relocalizing, HIGH-confidence retrieval earns a
 * verification slot regardless of the greedy-NN prematch count. The
 * funnel audit: 49-56% of corridor probe frames died at bestm<24 while
 * verification passed 90%+ once a candidate existed — the weak matcher
 * was gatekeeping the strong one (LighterGlue never saw those frames). */
#define TRUSTVPR_MIN 0.75f
static int trustvpr_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_TRUSTVPR");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: TRUST-VPR candidate rider ON");
    }
    return v;
}
static int burstpnp_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_BURSTPNP");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: BURST-PNP joint wake-up solve ON");
    }
    return v;
}
/* XR_DEPTHFILL — stereo-depth landmark backfill at keyframe store. The
 * outdoor funnel dies at n3: matched keypoints whose landmark never
 * triangulated (VIO's instantaneous baseline is centimetres; the stereo
 * baseline is ~30 cm and ranges to tens of metres). At ~1 Hz store
 * cadence, per-KEYPOINT epipolar ZNCC on the rectified pair (no dense
 * map) backfills 3D for keypoints the VIO left dark — they become
 * first-class landmarks for reloc, the bank, and the factor channel.
 * Only active when the harness registered a rectified stereo geometry. */
static struct { float fx, base; int set; } STEREOG;
void xr_map_set_stereo(float fx, float baseline_m) {
    if (fx > 1.0f && baseline_m > 1e-3f) {
        STEREOG.fx = fx;
        STEREOG.base = baseline_m;
        STEREOG.set = 1;
        LOGI("session map: stereo geom fx=%.1f base=%.3fm", (double)fx,
             (double)baseline_m);
    }
}
static int depthfill_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_DEPTHFILL");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: STEREO DEPTH-FILL ON");
    }
    return v;
}
static int lmtrack_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_LMTRACK");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: LMTRACK continuous factor coverage ON");
    }
    return v;
}
/* capture the applied closure's inlier set (uv values are exact copies of
 * w->kp_uv entries, so equality recovers the keypoint index -> desc) */
static void lmt_capture(const xr_kf *w, const float (*uv)[2],
                        const float (*ps)[3], int n, uint64_t ts) {
    if (w->desc_type == DESC_BAD) return;
    LMT.n = 0;
    for (int m = 0; m < n && LMT.n < LMFACT_MAX; m++) {
        for (int i = 0; i < w->n_kp; i++) {
            if (w->kp_uv[i][0] == uv[m][0] && w->kp_uv[i][1] == uv[m][1]) {
                memcpy(LMT.desc[LMT.n], w->desc.xfeat[i], 64);
                memcpy(LMT.ps[LMT.n], ps[m], sizeof LMT.ps[0]);
                LMT.n++;
                break;
            }
        }
    }
    LMT.until_ns = ts + LMT_WINDOW_NS;
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
    /* segment guard (review finding): unwelded-segment keyframes in the
     * tail live in a DIFFERENT session frame — their pc must neither be
     * warped toward D nor inflate the path length across the boundary.
     * Distances accumulate between consecutive SAME-segment keyframes;
     * other-segment keyframes are marked skipped. */
    int aseg = KFA(anchor).seg, prev = anchor;
    for (int j = anchor + 1; j < KF_N; j++) {
        if (KFA(j).seg != aseg) { cum[j] = -1.f; continue; }
        float dx = KFA(j).pc[0] - KFA(prev).pc[0];
        float dy = KFA(j).pc[1] - KFA(prev).pc[1];
        float dz = KFA(j).pc[2] - KFA(prev).pc[2];
        total += sqrtf(dx * dx + dy * dy + dz * dz);
        cum[j] = total;
        prev = j;
    }
    /* the correction D was estimated for the CURRENT query, not the newest
     * stored keyframe. Extend the path to the query (in session coords) so
     * the newest keyframe gets a fraction < 1 (the drift between it and the
     * query rides on the live pose via CORR); otherwise the newest stored
     * portion is over-deformed. */
    if (prev > anchor) {
        float dx = qsp[0] - KFA(prev).pc[0];
        float dy = qsp[1] - KFA(prev).pc[1];
        float dz = qsp[2] - KFA(prev).pc[2];
        total += sqrtf(dx * dx + dy * dy + dz * dz);
    }
    for (int j = anchor + 1; j < KF_N; j++) {
        if (cum[j] < 0) continue;          /* other segment: untouched */
        float a = total > 1e-3f ? cum[j] / total : 1.0f;
        float dq_old[4], dp_old[3], dq_new[4], dp_new[3];
        kf_corr(&KFA(j), dq_old, dp_old);  /* this keyframe's current corr */
        corr_interp(dq_old, dp_old, Dq, Dp, a, dq_new, dp_new);
        pose_compose(dq_new, dp_new, KFA(j).q, KFA(j).p, KFA(j).qc, KFA(j).pc);
    }
}

/* XR_PGO: Gauss-Seidel pose-graph relaxation over the drifted tail —
 * the principled replacement for graph_deform's path-length-weighted
 * interpolation. Nodes = keyframes anchor..KF_N-1 (session poses qc/pc)
 * plus a virtual query node; edges = consecutive ODOM relatives (skipped
 * across >10s gaps — submap breaks have no valid odometry) and the
 * closure measurement D∘O_q pinning the query node. Alternating-direction
 * sweeps propagate the correction both ways along the chain; each node
 * settles at the weighted blend of its neighbours' predictions, so
 * multiple constraints distribute consistently instead of by the single
 * path heuristic. Anchor stays fixed (the reference side of the loop).
 * MAP_LOCK held; ~200 nodes x 12 sweeps = microseconds. */
#ifndef PGO_SWEEPS
#define PGO_SWEEPS 12
#endif
#ifndef PGO_W_CLOSE
#define PGO_W_CLOSE 4.0f              /* closure prior vs odom edge weight */
#endif
#define PGO_GAP_NS 10000000000ull     /* no odom edge across this gap */

/* XR_PGO4DOF — restrict pose-graph deformation updates to yaw+translation
 * (roll/pitch are gravity-observable and must not bend under closures;
 * the VINS-Mono 4-DOF pose-graph discipline). */
static int pgo4dof_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_PGO4DOF");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: 4-DOF pose-graph projection ON");
    }
    return v;
}

static void pgo_blend(const float qa[4], const float pa[3], float wa,
                      const float qb[4], const float pb[3], float wb,
                      float qo[4], float po[3]) {
    float s = 1.0f / (wa + wb);
    float dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3];
    float sgn = dot < 0 ? -1.f : 1.f, n2 = 0;
    for (int c = 0; c < 4; c++) {
        qo[c] = (wa * qa[c] + wb * sgn * qb[c]) * s;
        n2 += qo[c] * qo[c];
    }
    n2 = 1.0f / sqrtf(n2);
    for (int c = 0; c < 4; c++) qo[c] *= n2;
    for (int c = 0; c < 3; c++) po[c] = (wa * pa[c] + wb * pb[c]) * s;
}

static void pgo_deform(int anchor, const float wq[4], const float wp[3],
                       const float Dq[4], const float Dp[3]) {
    if (anchor < 0 || anchor >= KF_N) return;
    int n = KF_N - anchor;               /* keyframe nodes */
    if (n < 2) return;
    /* odom relatives Z_i: node i -> i+1 (last edge: last kf -> query) */
    static float Zq[XR_MAP_MAX_KF + 1][4], Zp[XR_MAP_MAX_KF + 1][3];
    static uint8_t Zok[XR_MAP_MAX_KF + 1];
    for (int i = 0; i < n; i++) {
        const float *qa = KFA(anchor + i).q, *pa = KFA(anchor + i).p;
        const float *qb, *pb;
        uint64_t dt;
        if (i < n - 1) {
            qb = KFA(anchor + i + 1).q; pb = KFA(anchor + i + 1).p;
            dt = KFA(anchor + i + 1).ts - KFA(anchor + i).ts;
        } else {
            qb = wq; pb = wp;
            dt = 0;                       /* query follows the last kf */
        }
        Zok[i] = dt < PGO_GAP_NS;
        float qai[4], t[3];
        qconj(qa, qai);
        qmul(qai, qb, Zq[i]);
        t[0] = pb[0] - pa[0]; t[1] = pb[1] - pa[1]; t[2] = pb[2] - pa[2];
        qrotv(qai, t, Zp[i]);
    }
    /* states: X[0..n-1] = keyframes (init current session pose),
     * X[n] = query (init pre-closure CORR∘odom; prior = D∘odom) */
    static float Xq[XR_MAP_MAX_KF + 1][4], Xp[XR_MAP_MAX_KF + 1][3];
    for (int i = 0; i < n; i++) {
        memcpy(Xq[i], KFA(anchor + i).qc, sizeof Xq[i]);
        memcpy(Xp[i], KFA(anchor + i).pc, sizeof Xp[i]);
    }
    float Tq[4], Tp[3];                  /* query prior (the closure) */
    pose_compose(Dq, Dp, wq, wp, Tq, Tp);
    pose_compose(CORR.q, CORR.p, wq, wp, Xq[n], Xp[n]);
    for (int s = 0; s < PGO_SWEEPS; s++) {
        int fwd = (s & 1) == 0;
        for (int k = 1; k <= n; k++) {
            int i = fwd ? k : n - k + 1;           /* 1..n or n..1 */
            float lq[4], lp[3], rq[4], rp[3];
            int has_l = Zok[i - 1], has_r = 0;
            if (has_l)                    /* prediction from the left */
                pose_compose(Xq[i - 1], Xp[i - 1], Zq[i - 1], Zp[i - 1],
                             lq, lp);
            if (i < n && Zok[i]) {        /* prediction from the right */
                float ziq[4], zip[3];
                pose_invert(Zq[i], Zp[i], ziq, zip);
                pose_compose(Xq[i + 1], Xp[i + 1], ziq, zip, rq, rp);
                has_r = 1;
            }
            if (i == n) {                 /* query: closure prior is the
                                             right-side constraint */
                memcpy(rq, Tq, sizeof rq);
                memcpy(rp, Tp, sizeof rp);
                has_r = 1;
            }
            if (has_l && has_r)
                pgo_blend(lq, lp, 1.0f,
                          rq, rp, i == n ? PGO_W_CLOSE : 1.0f,
                          Xq[i], Xp[i]);
            else if (has_l) { memcpy(Xq[i], lq, 16); memcpy(Xp[i], lp, 12); }
            else if (has_r) { memcpy(Xq[i], rq, 16); memcpy(Xp[i], rp, 12); }
        }
    }
    for (int i = 1; i < n; i++) {        /* write back (anchor unchanged) */
        if (pgo4dof_on()) {
            /* XR_PGO4DOF: gravity makes roll/pitch OBSERVABLE — closure
             * error lives in yaw+translation only (VINS-Mono's 4-DOF pose
             * graph). Quaternion blending can leak correction into
             * roll/pitch; project the update onto the closest pure-yaw
             * world rotation of the ORIGINAL attitude instead. */
            float qi[4], dq[4];
            qconj(KFA(anchor + i).qc, qi);
            qmul(Xq[i], qi, dq);               /* world-frame left delta */
            /* closest Rz: yaw = atan2(2(w z), w^2 - z^2 + ...) — from the
             * delta's horizontal rotation block (Frobenius-optimal) */
            float w = dq[0], z = dq[3];
            float nrm = sqrtf(w * w + z * z);
            if (nrm > 1e-6f) {
                float dqz[4] = { w / nrm, 0, 0, z / nrm };
                qmul(dqz, KFA(anchor + i).qc, Xq[i]);
            }
        }
        memcpy(KFA(anchor + i).qc, Xq[i], sizeof Xq[i]);
        memcpy(KFA(anchor + i).pc, Xp[i], sizeof Xp[i]);
    }
}

/* XR_EDGEGRAPH — closure-edge MEMORY + whole-chain relaxation. The
 * per-sequence audit against OKVIS2+LC showed corridors 1-4 lost 2-7x:
 * they keep every closure as a graph edge and re-optimize the whole
 * trajectory on each revisit; we applied a closure and FORGOT it. Every
 * accepted closure now leaves a persistent edge (matched keyframe ->
 * newest keyframe, measured relative session pose), and relaxation runs
 * over the ENTIRE chain with ALL remembered edges — not just the anchor
 * tail with the newest edge. Edge admission stays conservative (only
 * verified closures get here) and each edge is down-weighted by Dynamic
 * Covariance Scaling on its current residual, so one bad edge cannot
 * bend the chain. Microseconds at 200 kf x 12 sweeps. Slots are
 * validated by timestamp (slot reuse after eviction would silently
 * re-target an edge). */
#define EG_MAX 64
#define EG_DCS_PHI 0.25f               /* DCS kernel, metres^2 */
static struct {
    int16_t slot_ref, slot_tip;
    uint64_t ts_ref, ts_tip;           /* validity: slot-reuse detection */
    float Zq[4], Zp[3];                /* measured tip pose in ref frame */
    float w;
} EG[EG_MAX];
static int EG_N;                       /* map thread only */
static int edgegraph_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_EDGEGRAPH");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: CLOSURE-EDGE GRAPH ON");
    }
    return v;
}

/* admit an edge from an accepted closure: ref = the matched keyframe
 * (order index), tip = the newest stored keyframe; Z carries the verified
 * query pose back to the tip through raw odometry (drift-free over the
 * seconds separating them). MAP_LOCK held. */
static void eg_admit(int ref_ord, const float wq[4], const float wp[3],
                     const float Dq[4], const float Dp[3], int nin) {
    if (KF_N < 2 || ref_ord < 0 || ref_ord >= KF_N) return;
    int tip_ord = KF_N - 1;
    if (tip_ord == ref_ord) return;
    xr_kf *ref = &KFA(ref_ord), *tip = &KFA(tip_ord);
    if (ref->seg != tip->seg) return;   /* cross-seg belongs to WELD */
    float Tq[4], Tp[3];
    pose_compose(Dq, Dp, wq, wp, Tq, Tp);
    float qi[4], pi[3], Rq[4], Rp[3];
    pose_invert(tip->q, tip->p, qi, pi);
    pose_compose(qi, pi, wq, wp, Rq, Rp);
    float ri[4], rp2[3], Ttq[4], Ttp[3];
    pose_invert(Rq, Rp, ri, rp2);
    pose_compose(Tq, Tp, ri, rp2, Ttq, Ttp);
    float rqi[4], rpi[3], Zq[4], Zp[3];
    pose_invert(ref->qc, ref->pc, rqi, rpi);
    pose_compose(rqi, rpi, Ttq, Ttp, Zq, Zp);
    int slot = -1;
    for (int e = 0; e < EG_N; e++)
        if (EG[e].slot_ref == KFO[ref_ord] && EG[e].slot_tip == KFO[tip_ord])
            slot = e;
    if (slot < 0) {
        if (EG_N < EG_MAX) slot = EG_N++;
        else {
            slot = 0;
            for (int e = 1; e < EG_N; e++)
                if (EG[e].w < EG[slot].w) slot = e;
        }
    }
    EG[slot].slot_ref = (int16_t)KFO[ref_ord];
    EG[slot].slot_tip = (int16_t)KFO[tip_ord];
    EG[slot].ts_ref = ref->ts;
    EG[slot].ts_tip = tip->ts;
    memcpy(EG[slot].Zq, Zq, sizeof Zq);
    memcpy(EG[slot].Zp, Zp, sizeof Zp);
    EG[slot].w = nin >= 24 ? 1.0f : (float)nin / 24.0f;
}

/* whole-chain relaxation with all live edges (MAP_LOCK held) */
static void eg_relax(void) {
    int n = KF_N;
    if (n < 3 || EG_N == 0) return;
    static int ord_of[XR_MAP_MAX_KF];
    for (int i = 0; i < XR_MAP_MAX_KF; i++) ord_of[i] = -1;
    for (int k = 0; k < n; k++) ord_of[KFO[k]] = k;
    static int e_ref[EG_MAX], e_tip[EG_MAX];
    static float e_w[EG_MAX];
    int ne = 0;
    for (int e = 0; e < EG_N; e++) {
        int ro = EG[e].slot_ref >= 0 ? ord_of[EG[e].slot_ref] : -1;
        int to = EG[e].slot_tip >= 0 ? ord_of[EG[e].slot_tip] : -1;
        if (ro < 0 || to < 0) continue;
        if (KFA(ro).ts != EG[e].ts_ref || KFA(to).ts != EG[e].ts_tip)
            continue;                   /* slot reused: edge is dead */
        float pred_q[4], pred_p[3];
        pose_compose(KFA(ro).qc, KFA(ro).pc, EG[e].Zq, EG[e].Zp,
                     pred_q, pred_p);
        float dx = pred_p[0] - KFA(to).pc[0];
        float dy = pred_p[1] - KFA(to).pc[1];
        float dz = pred_p[2] - KFA(to).pc[2];
        float r2 = dx * dx + dy * dy + dz * dz;
        float s = 2.0f * EG_DCS_PHI / (EG_DCS_PHI + r2);
        if (s > 1.0f) s = 1.0f;
        e_ref[ne] = ro; e_tip[ne] = to;
        e_w[ne] = EG[e].w * s * s;
        ne++;
    }
    if (!ne) return;
    static float Zq[XR_MAP_MAX_KF][4], Zp[XR_MAP_MAX_KF][3];
    static uint8_t Zok[XR_MAP_MAX_KF];
    static float Xq[XR_MAP_MAX_KF][4], Xp[XR_MAP_MAX_KF][3];
    for (int i = 0; i < n; i++) {
        memcpy(Xq[i], KFA(i).qc, sizeof Xq[i]);
        memcpy(Xp[i], KFA(i).pc, sizeof Xp[i]);
        if (i < n - 1) {
            uint64_t dt = KFA(i + 1).ts - KFA(i).ts;
            Zok[i] = dt < PGO_GAP_NS && KFA(i + 1).seg == KFA(i).seg;
            float qai[4], t[3];
            qconj(KFA(i).q, qai);
            qmul(qai, KFA(i + 1).q, Zq[i]);
            t[0] = KFA(i + 1).p[0] - KFA(i).p[0];
            t[1] = KFA(i + 1).p[1] - KFA(i).p[1];
            t[2] = KFA(i + 1).p[2] - KFA(i).p[2];
            qrotv(qai, t, Zp[i]);
        } else Zok[i] = 0;
    }
    for (int s = 0; s < PGO_SWEEPS; s++) {
        int fwd = (s & 1) == 0;
        for (int k = 1; k < n; k++) {
            int i = fwd ? k : n - k;
            float aq[4], ap[3];
            float wsum = 0;
            int have = 0;
            if (Zok[i - 1]) {
                pose_compose(Xq[i - 1], Xp[i - 1], Zq[i - 1], Zp[i - 1],
                             aq, ap);
                wsum = 1.0f; have = 1;
            }
            if (i < n - 1 && Zok[i]) {
                float ziq[4], zip[3], bq[4], bp[3];
                pose_invert(Zq[i], Zp[i], ziq, zip);
                pose_compose(Xq[i + 1], Xp[i + 1], ziq, zip, bq, bp);
                if (have) pgo_blend(aq, ap, wsum, bq, bp, 1.0f, aq, ap);
                else { memcpy(aq, bq, 16); memcpy(ap, bp, 12); }
                wsum += 1.0f; have = 1;
            }
            for (int e = 0; e < ne; e++) {
                float bq[4], bp[3];
                float wgt = PGO_W_CLOSE * e_w[e];
                if (e_tip[e] == i) {
                    pose_compose(Xq[e_ref[e]], Xp[e_ref[e]],
                                 EG[e].Zq, EG[e].Zp, bq, bp);
                } else if (e_ref[e] == i) {
                    float ziq[4], zip[3];
                    pose_invert(EG[e].Zq, EG[e].Zp, ziq, zip);
                    pose_compose(Xq[e_tip[e]], Xp[e_tip[e]], ziq, zip,
                                 bq, bp);
                } else continue;
                if (have) pgo_blend(aq, ap, wsum, bq, bp, wgt, aq, ap);
                else { memcpy(aq, bq, 16); memcpy(ap, bp, 12); }
                wsum += wgt; have = 1;
            }
            if (have) { memcpy(Xq[i], aq, 16); memcpy(Xp[i], ap, 12); }
        }
    }
    for (int i = 1; i < n; i++) {
        if (pgo4dof_on()) {
            float qi2[4], dq[4];
            qconj(KFA(i).qc, qi2);
            qmul(Xq[i], qi2, dq);
            float w = dq[0], z = dq[3];
            float nrm = sqrtf(w * w + z * z);
            if (nrm > 1e-6f) {
                float dqz[4] = { w / nrm, 0, 0, z / nrm };
                qmul(dqz, KFA(i).qc, Xq[i]);
            }
        }
        memcpy(KFA(i).qc, Xq[i], sizeof Xq[i]);
        memcpy(KFA(i).pc, Xp[i], sizeof Xp[i]);
    }
    CLOUD_DIRTY = 1;
}


/* Submap WELD: rigidly re-register every keyframe of segment `from` into
 * the session frame of segment `to` (E = session_to <- session_from), and
 * merge the ids. Rigid on purpose — the two segments have NO shared odom
 * path across the discontinuity, so a graph_deform-style path-weighted
 * blend has nothing sound to interpolate along; within-segment drift keeps
 * healing through the normal closure path afterwards. The odom poses (q,p)
 * are left alone: they are only ever used relative to qc/pc through the
 * per-keyframe landmark transform (kf_corr / reloc_pnp), which this update
 * keeps exact. MAP_LOCK held. */
static int seg_weld(int from, int to, const float Eq[4], const float Ep[3]) {
    int n = 0;
    for (int i = 0; i < KF_N; i++) {
        if (KFA(i).seg != from) continue;
        float q2[4], p2[3];
        pose_compose(Eq, Ep, KFA(i).qc, KFA(i).pc, q2, p2);
        memcpy(KFA(i).qc, q2, sizeof q2);
        memcpy(KFA(i).pc, p2, sizeof p2);
        KFA(i).seg = to;
        n++;
    }
    return n;
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

/* XR_FARBEAR: far-pair yaw agreement. A far pair is (query bearing s,
 * owner-ray world direction w), both unit, both gravity-aligned. Yaw only
 * spins the horizontal components, so an inlier must agree in z outright
 * AND horizontally once rotated. Tighter than VER_COS_TOL — a direction
 * carries no range to absorb error. */
#define FARB_COS_TOL 0.9986f            /* ~3 deg */
#define FARB_Z_TOL 0.05f
static int pnp2_far_inliers(const float (*sf)[3], const float (*wf)[3],
                            int nf, float yaw) {
    float cy = cosf(yaw), sy = sinf(yaw);
    int cnt = 0;
    for (int m = 0; m < nf; m++) {
        if (fabsf(sf[m][2] - wf[m][2]) > FARB_Z_TOL) continue;
        float rx = cy * sf[m][0] - sy * sf[m][1];
        float ry = sy * sf[m][0] + cy * sf[m][1];
        if (rx * wf[m][0] + ry * wf[m][1] + sf[m][2] * wf[m][2] >
            FARB_COS_TOL)
            cnt++;
    }
    return cnt;
}

/* hypothesis ranges must admit the compiled reloc range (drives run 60 m) */
#define PNP2_MAX_D (VER_MAX_RANGE_M > 40.0f ? VER_MAX_RANGE_M : 40.0f)

/* Gravity-aligned 2-point PnP RANSAC. s[i] = unit bearing of query kp i
 * rotated into the CURRENT-odom world (pre-yaw-correction), P[i] = the
 * matched map point (kf-odom world). Solves Rz(dyaw) and the camera
 * center C so that P[i] - C is parallel to Rz*s[i]. sf/wf/nf = OPTIONAL
 * far bearing pairs (XR_FARBEAR): extra yaw hypotheses + inlier votes;
 * they never touch translation. Returns near+far inliers (0 = no model). */
static int pnp2_ransac(const float (*s)[3], const float (*P)[3], int n,
                       const float (*sf)[3], const float (*wf)[3], int nf,
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
                d0 > PNP2_MAX_D || d1 > PNP2_MAX_D)
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
            int in = pnp2_inliers(s, P, n, yaw, C) +
                     pnp2_far_inliers(sf, wf, nf, yaw);
            if (in > best_in) {
                best_in = in;
                b_yaw = yaw;
                memcpy(bC, C, sizeof bC);
            }
        }
    }
    /* XR_FARBEAR hypotheses: one far pair fixes yaw outright; C then
     * follows LINEARLY from every near ray at that yaw (sum of (I-rr^T)
     * perpendicular constraints — no C needed up front). Rescues the
     * many-far/few-near regime where 2-near sampling rarely draws a
     * clean pair. */
    for (int m = 0; m < nf; m++) {
        if (fabsf(sf[m][2] - wf[m][2]) > FARB_Z_TOL) continue;
        float hs = sf[m][0] * sf[m][0] + sf[m][1] * sf[m][1];
        if (hs < 1e-4f) continue;              /* vertical ray: no yaw info */
        float yaw = atan2f(wf[m][1], wf[m][0]) - atan2f(sf[m][1], sf[m][0]);
        float cy = cosf(yaw), sy = sinf(yaw);
        float M[9] = { 0 }, b[3] = { 0 };
        for (int k = 0; k < n; k++) {
            float rx = cy * s[k][0] - sy * s[k][1];
            float ry = sy * s[k][0] + cy * s[k][1];
            float rz = s[k][2];
            float II[9] = { 1 - rx * rx, -rx * ry, -rx * rz,
                            -rx * ry, 1 - ry * ry, -ry * rz,
                            -rx * rz, -ry * rz, 1 - rz * rz };
            for (int t = 0; t < 9; t++) M[t] += II[t];
            b[0] += II[0] * P[k][0] + II[1] * P[k][1] + II[2] * P[k][2];
            b[1] += II[3] * P[k][0] + II[4] * P[k][1] + II[5] * P[k][2];
            b[2] += II[6] * P[k][0] + II[7] * P[k][1] + II[8] * P[k][2];
        }
        float det = M[0] * (M[4] * M[8] - M[5] * M[7]) -
                    M[1] * (M[3] * M[8] - M[5] * M[6]) +
                    M[2] * (M[3] * M[7] - M[4] * M[6]);
        if (fabsf(det) < 1e-6f) continue;
        float C[3] = {
            ((M[4] * M[8] - M[5] * M[7]) * b[0] +
             (M[2] * M[7] - M[1] * M[8]) * b[1] +
             (M[1] * M[5] - M[2] * M[4]) * b[2]) / det,
            ((M[5] * M[6] - M[3] * M[8]) * b[0] +
             (M[0] * M[8] - M[2] * M[6]) * b[1] +
             (M[2] * M[3] - M[0] * M[5]) * b[2]) / det,
            ((M[3] * M[7] - M[4] * M[6]) * b[0] +
             (M[1] * M[6] - M[0] * M[7]) * b[1] +
             (M[0] * M[4] - M[1] * M[3]) * b[2]) / det,
        };
        int in = pnp2_inliers(s, P, n, yaw, C) +
                 pnp2_far_inliers(sf, wf, nf, yaw);
        if (in > best_in) {
            best_in = in;
            b_yaw = yaw;
            memcpy(bC, C, sizeof bC);
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
        /* far pairs vote on yaw with unit weight (a direction has no
         * range to weight by; z-consistency is their admission test) */
        {
            float cyf = cosf(yaw), syf = sinf(yaw);
            for (int m = 0; m < nf; m++) {
                if (fabsf(sf[m][2] - wf[m][2]) > FARB_Z_TOL) continue;
                float rx = cyf * sf[m][0] - syf * sf[m][1];
                float ry = syf * sf[m][0] + cyf * sf[m][1];
                float cross = rx * wf[m][1] - ry * wf[m][0];
                float dotxy = rx * wf[m][0] + ry * wf[m][1];
                if (dotxy < 0) continue;       /* opposite hemisphere */
                dsum += atan2f(cross, dotxy);
                wsum += 1.0f;
            }
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
    int in = pnp2_inliers(s, P, n, yaw, C) +
             pnp2_far_inliers(sf, wf, nf, yaw);
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

/* XR_FARBEAR: a descriptor match dropped at the 3D gates, kept as a
 * bearing pair candidate (owner keypoint ray = world direction) */
#define FARB_MAX 96
struct rc_far {
    int qkp;                       /* query keypoint index */
    int16_t kf;                    /* owner keyframe slot */
    int16_t kp;                    /* owner keypoint index */
    int16_t c;                     /* covis-pool index (covqd/covpd) */
    int cost;                      /* descriptor distance, LOWER = better */
};
static int rc_far_cmp(const void *a, const void *b) {
    return ((const struct rc_far *)a)->cost - ((const struct rc_far *)b)->cost;
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
 * inlier-backed covisible-keyframe count; returns 1 on a solved pose.
 * XR_LMFACT extension: when out_iuv/out_ips/out_nl are non-NULL, also
 * emits up to LMFACT_MAX geometric-inlier correspondences — query pixel
 * (out_iuv) + matched landmark SESSION-frame 3D (out_ips) — for the
 * stage-3 landmark-factor channel. Callers not needing them pass NULL. */
static int reloc_pnp(const xr_kf *w, int best_i, int nkf, float Dq[4], float Dp[3],
                     int *out_n3, int *out_nin, int *out_covis,
                     float (*out_iuv)[2], float (*out_ips)[3], int *out_nl,
                     float (*out_bS)[3], float (*out_bP)[3], int *out_bn) {
    *out_n3 = *out_nin = *out_covis = 0;
    if (out_nl) *out_nl = 0;
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
        if (s == best_i || KFA(s).desc_type != w->desc_type ||
            KFA(s).seg != KFA(best_i).seg)   /* pc-space is per-segment */
            continue;
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
     * descriptor cost; best_i's matches enter the pool first.
     * XFeat + LighterGlue (when registered): learned attention matching
     * over the two keypoint SETS replaces greedy NN+margin — recovers the
     * repetitive-structure matches NN loses (the corridor reloc-recall
     * axis). Runs only here (few candidates), never in the retrieval
     * scan. Falls back to NN on any inference failure. */
    static struct rc_cand cand[COVIS_MAX_CAND];
    int nc = 0;
    /* XR_FARBEAR side pool: matches the 3D gates reject (never-triangulated
     * or out-of-range landmarks — distant structure). Bearing-only. */
    static struct rc_far fcand[FARB_MAX];
    int nfc = 0;
    int want_far = farbear_on();
    int use_lg = !bad && xr_lglue_wanted();
    for (int c = 0; c < ncov && nc < COVIS_MAX_CAND; c++) {
        int s = cov[c];
        /* LG budget: learned matching for the CANDIDATE keyframe only
         * (cov[0] leads the pool by construction); pooled neighbours keep
         * greedy NN. Running LG per pooled keyframe was 8x the inference
         * per cluster and starved the map thread (room1 stores 225 -> 40)
         * for marginal extra pairs — the candidate dominates the
         * correspondence set, neighbours mostly widen the landmark pool. */
        if (use_lg && c == 0) {
            static int li0[XR_LGLUE_N], li1[XR_LGLUE_N];   /* map thread */
            static float lsc[XR_LGLUE_N];
            int nm = xr_lglue_match(w->kp_uv, w->desc.xfeat, w->n_kp,
                                    KFA(s).kp_uv, KFA(s).desc.xfeat,
                                    KFA(s).n_kp, (float)XR_OW, (float)XR_OH,
                                    li0, li1, lsc, XR_LGLUE_N);
            if (nm >= 0) {
                for (int m = 0; m < nm && nc < COVIS_MAX_CAND; m++) {
                    int i = li0[m];
                    int lk = KFA(s).lm_of_kp[li1[m]];
                    int lgcost = (int)((1.0f - lsc[m]) * 1000.0f);
                    if (lk < 0) {                  /* kp without a landmark */
                        if (want_far && nfc < FARB_MAX) {
                            fcand[nfc].qkp = i;
                            fcand[nfc].kf = (int16_t)s;
                            fcand[nfc].kp = (int16_t)li1[m];
                            fcand[nfc].c = (int16_t)c;
                            fcand[nfc].cost = lgcost;
                            nfc++;
                        }
                        continue;
                    }
                    float dk2 = 0;
                    for (int cc = 0; cc < 3; cc++) {
                        float dd = KFA(s).lm_xyz[lk][cc] - KFA(s).p[cc];
                        dk2 += dd * dd;
                    }
                    if (dk2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M) {
                        if (want_far && nfc < FARB_MAX) {
                            fcand[nfc].qkp = i;
                            fcand[nfc].kf = (int16_t)s;
                            fcand[nfc].kp = (int16_t)li1[m];
                            fcand[nfc].c = (int16_t)c;
                            fcand[nfc].cost = lgcost;
                            nfc++;
                        }
                        continue;
                    }
                    float ps[3];
                    qrotv(covqd[c], KFA(s).lm_xyz[lk], ps);
                    cand[nc].qkp = i;
                    cand[nc].id = KFA(s).lm_id[lk];
                    cand[nc].kf = s;
                    /* lower = better, like the NN costs */
                    cand[nc].cost = lgcost;
                    cand[nc].ps[0] = ps[0] + covpd[c][0];
                    cand[nc].ps[1] = ps[1] + covpd[c][1];
                    cand[nc].ps[2] = ps[2] + covpd[c][2];
                    nc++;
                }
                continue;                          /* this keyframe done */
            }
            use_lg = 0;                /* bring-up failed: NN for the rest */
        }
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
            int nncost = bad ? best : ((1 << 30) - best);
            int lk = KFA(s).lm_of_kp[bj];
            if (lk < 0) {
                if (want_far && nfc < FARB_MAX) {
                    fcand[nfc].qkp = i;
                    fcand[nfc].kf = (int16_t)s;
                    fcand[nfc].kp = (int16_t)bj;
                    fcand[nfc].c = (int16_t)c;
                    fcand[nfc].cost = nncost;
                    nfc++;
                }
                continue;
            }
            float dk2 = 0;
            for (int cc = 0; cc < 3; cc++) {
                float d = KFA(s).lm_xyz[lk][cc] - KFA(s).p[cc];
                dk2 += d * d;
            }
            if (dk2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M) {
                if (want_far && nfc < FARB_MAX) {
                    fcand[nfc].qkp = i;
                    fcand[nfc].kf = (int16_t)s;
                    fcand[nfc].kp = (int16_t)bj;
                    fcand[nfc].c = (int16_t)c;
                    fcand[nfc].cost = nncost;
                    nfc++;
                }
                continue;
            }
            float ps[3];
            qrotv(covqd[c], KFA(s).lm_xyz[lk], ps);
            cand[nc].qkp = i;
            cand[nc].id = KFA(s).lm_id[lk];
            cand[nc].kf = s;
            cand[nc].cost = nncost;
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
    static float Quv[COVIS_MAX_CAND][2];      /* query pixel per pair (lmfact) */
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
        Quv[n][0] = w->kp_uv[i][0];
        Quv[n][1] = w->kp_uv[i][1];
        Pid[n] = cand[k].id;
        qused[i] = 1;
        used_id[nid++] = cand[k].id;
        n++;
    }
    *out_n3 = n;
    /* XR_BURSTPNP: export the ASSIGNED match set (pre-gate — the joint
     * burst solve re-arbitrates across frames, so even a frame too weak
     * to verify alone contributes) */
    if (out_bS && out_bP && out_bn) {
        int bn = n < BURST_EXPORT_MAX ? n : BURST_EXPORT_MAX;
        memcpy(out_bS, Sw, sizeof(float) * 3 * (size_t)bn);
        memcpy(out_bP, Pw, sizeof(float) * 3 * (size_t)bn);
        *out_bn = bn;
    }
    if (n < VER_MIN_PAIRS) return 0;

    /* 2c. XR_FARBEAR: greedy one-to-one over the far pool, AFTER the near
     * assignment so a 3D-backed match always wins its query keypoint.
     * Each far pair becomes (query bearing, owner-ray session direction) —
     * the owner keypoint unprojects in the owner camera, rotates through
     * the owner's stored RAW orientation, then through the same
     * raw->session alignment its landmarks would use (rotation only:
     * directions have no translation). */
    static float Sf[FARB_MAX][3], Wf[FARB_MAX][3];
    int nfar = 0;
    if (want_far && nfc > 0) {
        qsort(fcand, (size_t)nfc, sizeof fcand[0], rc_far_cmp);
        for (int k = 0; k < nfc && nfar < FARB_MAX; k++) {
            int i = fcand[k].qkp;
            if (qused[i]) continue;
            int s2 = fcand[k].kf;
            float rc[3], rb[3];
            if (GEOM.unproject(w->kp_uv[i][0], w->kp_uv[i][1], rc)) continue;
            rb[0] = GEOM.R_ic[0] * rc[0] + GEOM.R_ic[1] * rc[1] +
                    GEOM.R_ic[2] * rc[2];
            rb[1] = GEOM.R_ic[3] * rc[0] + GEOM.R_ic[4] * rc[1] +
                    GEOM.R_ic[5] * rc[2];
            rb[2] = GEOM.R_ic[6] * rc[0] + GEOM.R_ic[7] * rc[1] +
                    GEOM.R_ic[8] * rc[2];
            Sf[nfar][0] = Rq[0] * rb[0] + Rq[1] * rb[1] + Rq[2] * rb[2];
            Sf[nfar][1] = Rq[3] * rb[0] + Rq[4] * rb[1] + Rq[5] * rb[2];
            Sf[nfar][2] = Rq[6] * rb[0] + Rq[7] * rb[1] + Rq[8] * rb[2];
            if (GEOM.unproject(KFA(s2).kp_uv[fcand[k].kp][0],
                               KFA(s2).kp_uv[fcand[k].kp][1], rc))
                continue;
            rb[0] = GEOM.R_ic[0] * rc[0] + GEOM.R_ic[1] * rc[1] +
                    GEOM.R_ic[2] * rc[2];
            rb[1] = GEOM.R_ic[3] * rc[0] + GEOM.R_ic[4] * rc[1] +
                    GEOM.R_ic[5] * rc[2];
            rb[2] = GEOM.R_ic[6] * rc[0] + GEOM.R_ic[7] * rc[1] +
                    GEOM.R_ic[8] * rc[2];
            float Rk[9], ro[3];
            q2R(KFA(s2).q, Rk);
            ro[0] = Rk[0] * rb[0] + Rk[1] * rb[1] + Rk[2] * rb[2];
            ro[1] = Rk[3] * rb[0] + Rk[4] * rb[1] + Rk[5] * rb[2];
            ro[2] = Rk[6] * rb[0] + Rk[7] * rb[1] + Rk[8] * rb[2];
            qrotv(covqd[fcand[k].c], ro, Wf[nfar]);
            qused[i] = 1;
            nfar++;
        }
    }

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
    /* far votes join nin (evidence is evidence) but the 33% consensus
     * ratio stays NEAR-only: bearing pairs never ranged anything, so they
     * must not dilute the fraction of 3D matches the pose explains. */
    int nin = pnp2_ransac(Sw, Pw, n, Sf, Wf, nfar, Rz, C);
    *out_nin = nin;
    int near_in = nin - (nfar > 0 ? pnp2_far_inliers(Sf, Wf, nfar,
                             atan2f(Rz[3], Rz[0])) : 0);
    if (nin < VER_MIN_PAIRS || near_in * 100 < 33 * n) return 0;

    /* 3. inlier-backed covisibility: the union of observation sets over the
     * geometric INLIERS — how many DISTINCT pooled keyframes actually back
     * the solved pose (Rz = [[cy,-sy,0],[sy,cy,0],[0,0,1]]). A lone junk
     * cluster collapses to one or two bits and is rejected upstream. */
    float cy = Rz[0], sy = Rz[3];
    uint32_t covmask = 0;
    int nl = 0;
    for (int m = 0; m < n; m++) {
        float qx = Pw[m][0] - C[0], qy = Pw[m][1] - C[1], qz = Pw[m][2] - C[2];
        float nq = sqrtf(qx * qx + qy * qy + qz * qz);
        if (nq < VER_MIN_RANGE_M) continue;
        float rx = cy * Sw[m][0] - sy * Sw[m][1];
        float ry = sy * Sw[m][0] + cy * Sw[m][1];
        float dot = (qx * rx + qy * ry + qz * Sw[m][2]) / nq;
        if (dot <= VER_COS_TOL) continue;
        covmask |= obsmask[m];
        if (out_iuv && out_ips && nl < LMFACT_MAX) {   /* XR_LMFACT pairs */
            out_iuv[nl][0] = Quv[m][0];
            out_iuv[nl][1] = Quv[m][1];
            memcpy(out_ips[nl], Pw[m], sizeof out_ips[0]);
            nl++;
        }
    }
    if (out_nl) *out_nl = nl;
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

/* XR_LOCALBA — windowed structure refinement after a verified closure.
 * The pose graph (above) moves keyframes rigidly; landmark 3D stays
 * whatever the VIO first triangulated, and each keyframe keeps its OWN
 * copy — the multi-view spread the reloc averaging step merely tolerates.
 * This is the BA-quality axis where OKVIS2+LC's ~1.2 cm rooms live.
 *
 * Bounded resection-intersection over the closure's covis window:
 *   intersect: per landmark (seen by >=2 window kfs), Gauss-Newton on the
 *     session-frame point under BEARING residuals r = (I - bb^T) v / |v|
 *     (no forward camera model needed — b is the stored kp's unprojected
 *     ray in the keyframe's session orientation);
 *   resect: per non-anchor keyframe, 4-DOF (yaw + t) Gauss-Newton against
 *     the refined points (gravity axes stay untouched).
 * Alternating LBA_ROUNDS; Huber on the residual angle. Writes back the
 * refined session poses AND per-keyframe landmark copies (re-expressed
 * through each kf's raw->session alignment, so every view finally agrees
 * on where the point is). Anchor = the closure-verified keyframe.
 * MAP_LOCK held; window <= 16 kf x 160 lm x 6 obs — microseconds-scale. */
#define LBA_MAX_LM 160
#define LBA_MAX_OBS 6
#define LBA_ROUNDS 3
#define LBA_HUBER 0.008f              /* rad, ~0.46 deg */
#define LBA_OUTLIER 0.05f             /* rad: obs beyond this never votes */
static uint64_t LBA_LAST_NS;          /* rate limit (map thread only) */
static int localba_on(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("XR_LOCALBA");
        v = (e && *e && *e != '0') ? 1 : 0;
        if (v) LOGI("session map: LOCAL-BA structure refinement ON");
    }
    return v;
}

static void local_ba(int center) {
    if (center < 0 || center >= KF_N || !GEOM.have) return;
    /* window: center + nearest same-segment neighbours (session space) */
    int win[COVIS_MAX_KF]; int nw = 0;
    win[nw++] = center;
    static struct rc_nb lnb[XR_MAP_MAX_KF];
    int nn = 0;
    for (int s = 0; s < KF_N; s++) {
        if (s == center || KFA(s).seg != KFA(center).seg) continue;
        float dx = KFA(s).pc[0] - KFA(center).pc[0];
        float dy = KFA(s).pc[1] - KFA(center).pc[1];
        float dz = KFA(s).pc[2] - KFA(center).pc[2];
        float d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > COVIS_R_M * COVIS_R_M) continue;
        lnb[nn].s = s; lnb[nn].d2 = d2; nn++;
    }
    qsort(lnb, (size_t)nn, sizeof lnb[0], rc_nb_cmp);
    for (int k = 0; k < nn && nw < COVIS_MAX_KF; k++) win[nw++] = lnb[k].s;
    if (nw < 2) return;

    /* per-window-kf pose state + raw->session alignment cache */
    float R[COVIS_MAX_KF][9], O[COVIS_MAX_KF][3];
    float alq[COVIS_MAX_KF][4], alp[COVIS_MAX_KF][3];
    for (int k = 0; k < nw; k++) {
        xr_kf *f = &KFA(win[k]);
        q2R(f->qc, R[k]);
        float t[3];
        qrotv(f->qc, GEOM.p_ic, t);
        O[k][0] = f->pc[0] + t[0];
        O[k][1] = f->pc[1] + t[1];
        O[k][2] = f->pc[2] + t[2];
        float qoi[4], poi[3];
        pose_invert(f->q, f->p, qoi, poi);
        pose_compose(f->qc, f->pc, qoi, poi, alq[k], alp[k]);
    }

    /* gather shared landmarks: id -> obs list (window kf, BODY-frame ray) */
    static int32_t Lid[LBA_MAX_LM];
    static float LX[LBA_MAX_LM][3];               /* session estimate */
    static uint8_t Lobs_k[LBA_MAX_LM][LBA_MAX_OBS];
    static float Lobs_rb[LBA_MAX_LM][LBA_MAX_OBS][3];
    static uint8_t Lnobs[LBA_MAX_LM];
    int nl = 0;
    for (int k = 0; k < nw; k++) {
        xr_kf *f = &KFA(win[k]);
        for (int j = 0; j < f->n_kp; j++) {
            int l = f->lm_of_kp[j];
            if (l < 0) continue;
            float rng2 = 0;
            for (int c = 0; c < 3; c++) {
                float d = f->lm_xyz[l][c] - f->p[c];
                rng2 += d * d;
            }
            if (rng2 < VER_MIN_RANGE_M * VER_MIN_RANGE_M ||
                rng2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M)
                continue;
            int li = -1;
            for (int t = 0; t < nl; t++)
                if (Lid[t] == f->lm_id[l]) { li = t; break; }
            if (li < 0) {
                if (nl >= LBA_MAX_LM) continue;
                li = nl++;
                Lid[li] = f->lm_id[l];
                Lnobs[li] = 0;
                float ps[3];
                qrotv(alq[k], f->lm_xyz[l], ps);
                LX[li][0] = ps[0] + alp[k][0];
                LX[li][1] = ps[1] + alp[k][1];
                LX[li][2] = ps[2] + alp[k][2];
            }
            if (Lnobs[li] >= LBA_MAX_OBS) continue;
            float rc[3], rb[3];
            if (GEOM.unproject(f->kp_uv[j][0], f->kp_uv[j][1], rc)) continue;
            rb[0] = GEOM.R_ic[0] * rc[0] + GEOM.R_ic[1] * rc[1] +
                    GEOM.R_ic[2] * rc[2];
            rb[1] = GEOM.R_ic[3] * rc[0] + GEOM.R_ic[4] * rc[1] +
                    GEOM.R_ic[5] * rc[2];
            rb[2] = GEOM.R_ic[6] * rc[0] + GEOM.R_ic[7] * rc[1] +
                    GEOM.R_ic[8] * rc[2];
            Lobs_k[li][Lnobs[li]] = (uint8_t)k;
            memcpy(Lobs_rb[li][Lnobs[li]], rb, sizeof rb);
            Lnobs[li]++;
        }
    }
    int nshared = 0;
    for (int t = 0; t < nl; t++) if (Lnobs[t] >= 2) nshared++;
    if (nshared < 8) return;                       /* nothing to refine */

    for (int round = 0; round < LBA_ROUNDS; round++) {
        /* -- intersect: refine each shared landmark, poses fixed -- */
        for (int t = 0; t < nl; t++) {
            if (Lnobs[t] < 2) continue;
            for (int it = 0; it < 2; it++) {
                float H[9] = { 0 }, g[3] = { 0 };
                int used = 0;
                for (int o = 0; o < Lnobs[t]; o++) {
                    int k = Lobs_k[t][o];
                    float b[3];
                    b[0] = R[k][0] * Lobs_rb[t][o][0] + R[k][1] * Lobs_rb[t][o][1] + R[k][2] * Lobs_rb[t][o][2];
                    b[1] = R[k][3] * Lobs_rb[t][o][0] + R[k][4] * Lobs_rb[t][o][1] + R[k][5] * Lobs_rb[t][o][2];
                    b[2] = R[k][6] * Lobs_rb[t][o][0] + R[k][7] * Lobs_rb[t][o][1] + R[k][8] * Lobs_rb[t][o][2];
                    float v[3] = { LX[t][0] - O[k][0], LX[t][1] - O[k][1],
                                   LX[t][2] - O[k][2] };
                    float d = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                    if (d < VER_MIN_RANGE_M) continue;
                    float bv = (b[0] * v[0] + b[1] * v[1] + b[2] * v[2]) / d;
                    if (bv < 0) continue;          /* behind the camera */
                    float r3[3], rn = 0;
                    for (int c = 0; c < 3; c++) {
                        r3[c] = v[c] / d - bv * b[c];
                        rn += r3[c] * r3[c];
                    }
                    rn = sqrtf(rn);
                    if (rn > LBA_OUTLIER) continue;
                    float w = rn <= LBA_HUBER ? 1.0f : LBA_HUBER / rn;
                    /* J = (I - bb^T)/d; the projector is idempotent, so
                     * J^T J = (I - bb^T)/d^2 and J^T r = r/d (r _|_ b) */
                    for (int a = 0; a < 3; a++)
                        for (int c2 = 0; c2 < 3; c2++) {
                            float JtJ = ((a == c2 ? 1.0f : 0.0f) -
                                         b[a] * b[c2]) / (d * d);
                            H[a * 3 + c2] += w * JtJ;
                        }
                    for (int a = 0; a < 3; a++)
                        g[a] += w * r3[a] / d;
                    used++;
                }
                if (used < 2) break;
                /* damp + solve 3x3 (H is PSD projector sum; add LM mu) */
                float mu = 1e-4f;
                H[0] += mu; H[4] += mu; H[8] += mu;
                float det = H[0] * (H[4] * H[8] - H[5] * H[7]) -
                            H[1] * (H[3] * H[8] - H[5] * H[6]) +
                            H[2] * (H[3] * H[7] - H[4] * H[6]);
                if (fabsf(det) < 1e-12f) break;
                float dx[3] = {
                    ((H[4] * H[8] - H[5] * H[7]) * g[0] +
                     (H[2] * H[7] - H[1] * H[8]) * g[1] +
                     (H[1] * H[5] - H[2] * H[4]) * g[2]) / det,
                    ((H[5] * H[6] - H[3] * H[8]) * g[0] +
                     (H[0] * H[8] - H[2] * H[6]) * g[1] +
                     (H[2] * H[3] - H[0] * H[5]) * g[2]) / det,
                    ((H[3] * H[7] - H[4] * H[6]) * g[0] +
                     (H[1] * H[6] - H[0] * H[7]) * g[1] +
                     (H[0] * H[4] - H[1] * H[3]) * g[2]) / det,
                };
                float step2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
                if (step2 > 0.25f) break;          /* >0.5 m: distrust */
                LX[t][0] -= dx[0];
                LX[t][1] -= dx[1];
                LX[t][2] -= dx[2];
            }
        }
        /* -- resect: refine each NON-ANCHOR pose (yaw + t), points fixed -- */
        for (int k = 1; k < nw; k++) {
            for (int it = 0; it < 2; it++) {
                float H[16] = { 0 }, g[4] = { 0 };
                int used = 0;
                for (int t = 0; t < nl; t++) {
                    if (Lnobs[t] < 2) continue;
                    for (int o = 0; o < Lnobs[t]; o++) {
                        if (Lobs_k[t][o] != k) continue;
                        float b[3];
                        b[0] = R[k][0] * Lobs_rb[t][o][0] + R[k][1] * Lobs_rb[t][o][1] + R[k][2] * Lobs_rb[t][o][2];
                        b[1] = R[k][3] * Lobs_rb[t][o][0] + R[k][4] * Lobs_rb[t][o][1] + R[k][5] * Lobs_rb[t][o][2];
                        b[2] = R[k][6] * Lobs_rb[t][o][0] + R[k][7] * Lobs_rb[t][o][1] + R[k][8] * Lobs_rb[t][o][2];
                        float v[3] = { LX[t][0] - O[k][0], LX[t][1] - O[k][1],
                                       LX[t][2] - O[k][2] };
                        float d = sqrtf(v[0] * v[0] + v[1] * v[1] +
                                        v[2] * v[2]);
                        if (d < VER_MIN_RANGE_M) continue;
                        float bv = (b[0] * v[0] + b[1] * v[1] + b[2] * v[2]) / d;
                        if (bv < 0) continue;
                        float r3[3], rn = 0;
                        for (int c = 0; c < 3; c++) {
                            r3[c] = v[c] / d - bv * b[c];
                            rn += r3[c] * r3[c];
                        }
                        rn = sqrtf(rn);
                        if (rn > LBA_OUTLIER) continue;
                        float w = rn <= LBA_HUBER ? 1.0f : LBA_HUBER / rn;
                        /* J columns: dyaw (b spins: db = z x b, O spins:
                         * dO = z x (O - pc) — dominate via the b term; the
                         * O lever is p_ic-short, keep it anyway), t (dO=dt).
                         * dr/db = -(bv*I + b v^T/d), dr/dO = -(I-bb^T)/d */
                        float zb[3] = { -b[1], b[0], 0.0f };
                        xr_kf *f = &KFA(win[k]);
                        float lever[3] = { O[k][0] - f->pc[0],
                                           O[k][1] - f->pc[1],
                                           O[k][2] - f->pc[2] };
                        float zO[3] = { -lever[1], lever[0], 0.0f };
                        float J[3][4];
                        for (int a = 0; a < 3; a++) {
                            /* dr/dyaw = dr/db * zb + dr/dO * zO */
                            float drb = -(bv * zb[a] +
                                          b[a] * (zb[0] * v[0] + zb[1] * v[1] +
                                                  zb[2] * v[2]) / d);
                            float drO = 0;
                            for (int c2 = 0; c2 < 3; c2++)
                                drO += -((a == c2 ? 1.0f : 0.0f) -
                                         b[a] * b[c2]) / d * zO[c2];
                            J[a][0] = drb + drO;
                            for (int c2 = 0; c2 < 3; c2++)
                                J[a][1 + c2] = -((a == c2 ? 1.0f : 0.0f) -
                                                 b[a] * b[c2]) / d;
                        }
                        for (int a2 = 0; a2 < 4; a2++) {
                            for (int b2 = 0; b2 < 4; b2++) {
                                float acc = 0;
                                for (int c2 = 0; c2 < 3; c2++)
                                    acc += J[c2][a2] * J[c2][b2];
                                H[a2 * 4 + b2] += w * acc;
                            }
                            float acc = 0;
                            for (int c2 = 0; c2 < 3; c2++)
                                acc += J[c2][a2] * r3[c2];
                            g[a2] += w * acc;
                        }
                        used++;
                    }
                }
                if (used < 6) break;
                float mu = 1e-4f;
                for (int a = 0; a < 4; a++) H[a * 4 + a] += mu;
                /* 4x4 gaussian elimination */
                float A[4][5];
                for (int a = 0; a < 4; a++) {
                    for (int b2 = 0; b2 < 4; b2++) A[a][b2] = H[a * 4 + b2];
                    A[a][4] = -g[a];
                }
                int sing = 0;
                for (int col = 0; col < 4 && !sing; col++) {
                    int piv = col;
                    for (int rr = col + 1; rr < 4; rr++)
                        if (fabsf(A[rr][col]) > fabsf(A[piv][col])) piv = rr;
                    if (fabsf(A[piv][col]) < 1e-10f) { sing = 1; break; }
                    if (piv != col)
                        for (int cc = 0; cc < 5; cc++) {
                            float tmp = A[col][cc];
                            A[col][cc] = A[piv][cc];
                            A[piv][cc] = tmp;
                        }
                    for (int rr = col + 1; rr < 4; rr++) {
                        float fkt = A[rr][col] / A[col][col];
                        for (int cc = col; cc < 5; cc++)
                            A[rr][cc] -= fkt * A[col][cc];
                    }
                }
                if (sing) break;
                float u[4];
                for (int a = 3; a >= 0; a--) {
                    u[a] = A[a][4];
                    for (int cc = a + 1; cc < 4; cc++)
                        u[a] -= A[a][cc] * u[cc];
                    u[a] /= A[a][a];
                }
                if (fabsf(u[0]) > 0.05f ||
                    u[1] * u[1] + u[2] * u[2] + u[3] * u[3] > 0.04f)
                    break;                          /* implausible step */
                /* apply: yaw about world z (left), then translation */
                xr_kf *f = &KFA(win[k]);
                float dq[4] = { cosf(u[0] * 0.5f), 0, 0, sinf(u[0] * 0.5f) };
                float qn[4];
                qmul(dq, f->qc, qn);
                memcpy(f->qc, qn, sizeof qn);
                f->pc[0] += u[1];
                f->pc[1] += u[2];
                f->pc[2] += u[3];
                /* refresh caches for this kf */
                q2R(f->qc, R[k]);
                float t2[3];
                qrotv(f->qc, GEOM.p_ic, t2);
                O[k][0] = f->pc[0] + t2[0];
                O[k][1] = f->pc[1] + t2[1];
                O[k][2] = f->pc[2] + t2[2];
                float qoi[4], poi[3];
                pose_invert(f->q, f->p, qoi, poi);
                pose_compose(f->qc, f->pc, qoi, poi, alq[k], alp[k]);
            }
        }
    }

    /* write back: every window copy of a refined landmark re-expressed
     * through its keyframe's (updated) raw->session alignment — the views
     * now agree on the point */
    int nwrote = 0;
    for (int k = 0; k < nw; k++) {
        xr_kf *f = &KFA(win[k]);
        float qi[4], t[3];
        qconj(alq[k], qi);
        for (int l = 0; l < f->n_lm; l++) {
            for (int tt = 0; tt < nl; tt++) {
                if (Lnobs[tt] < 2 || Lid[tt] != f->lm_id[l]) continue;
                t[0] = LX[tt][0] - alp[k][0];
                t[1] = LX[tt][1] - alp[k][1];
                t[2] = LX[tt][2] - alp[k][2];
                qrotv(qi, t, f->lm_xyz[l]);
                nwrote++;
                break;
            }
        }
    }
    LOGI("session map: LOCALBA kf#%d window=%d shared_lm=%d copies=%d",
         center, nw, nshared, nwrote);
}

/* ---- XR_BURSTPNP: offset-aware joint solve --------------------------------------
 * Twin of pnp2_ransac for burst fusion: match m's ray leaves the camera
 * at C + Rz*O[m] (per-frame origin offsets in the pre-yaw frame). Kept as
 * a SEPARATE function so the validated single-frame path stays untouched.
 * Hypothesis pairs are drawn within one frame (equal offsets — the
 * closed form needs a common origin), then re-based to the burst origin. */
static int pnp2_inliers_ofs(const float (*s)[3], const float (*P)[3],
                            const float (*O)[3], int n, float yaw,
                            const float C[3]) {
    float cy = cosf(yaw), sy = sinf(yaw);
    const float tol2 = VER_COS_TOL * VER_COS_TOL;
    const float minr2 = VER_MIN_RANGE_M * VER_MIN_RANGE_M;
    int cnt = 0;
    for (int m = 0; m < n; m++) {
        float ox = cy * O[m][0] - sy * O[m][1];
        float oy = sy * O[m][0] + cy * O[m][1];
        float qx = P[m][0] - C[0] - ox;
        float qy = P[m][1] - C[1] - oy;
        float qz = P[m][2] - C[2] - O[m][2];
        float nq2 = qx * qx + qy * qy + qz * qz;
        if (nq2 < minr2) continue;
        float rx = cy * s[m][0] - sy * s[m][1];
        float ry = sy * s[m][0] + cy * s[m][1];
        float raw = qx * rx + qy * ry + qz * s[m][2];
        if (raw > 0 && raw * raw > tol2 * nq2) cnt++;
    }
    return cnt;
}

static int pnp2_ransac_burst(const float (*s)[3], const float (*P)[3],
                             const float (*O)[3], const uint8_t *fid, int n,
                             float Rz_out[9], float C_out[3]) {
    if (n < 2) return 0;
    int best_in = 0;
    float b_yaw = 0, bC[3] = { 0, 0, 0 };
    uint32_t seed = 0xB0857u ^ (uint32_t)n;
    for (int it = 0; it < VER_ITERS * 2; it++) {   /* bigger pool: 2x tries */
        seed = seed * 1664525u + 1013904223u;
        int i0 = (int)(seed % (uint32_t)n);
        seed = seed * 1664525u + 1013904223u;
        int i1 = (int)(seed % (uint32_t)n);
        if (i0 == i1 || fid[i0] != fid[i1]) continue;  /* common origin only */
        float dx = P[i0][0] - P[i1][0];
        float dy = P[i0][1] - P[i1][1];
        float dz = P[i0][2] - P[i1][2];
        float Q = dx * dx + dy * dy;
        float s0z = s[i0][2], s1z = s[i1][2];
        if (fabsf(s0z) < 1e-3f) continue;
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
                d0 > PNP2_MAX_D || d1 > PNP2_MAX_D)
                continue;
            float ux = d0 * s[i0][0] - d1 * s[i1][0];
            float uy = d0 * s[i0][1] - d1 * s[i1][1];
            if (ux * ux + uy * uy < 1e-6f) continue;
            float yaw = atan2f(ux * dy - uy * dx, ux * dx + uy * dy);
            float cy = cosf(yaw), sy = sinf(yaw);
            /* pair-frame camera centre, then re-base to the burst origin */
            float C[3] = {
                P[i0][0] - d0 * (cy * s[i0][0] - sy * s[i0][1]) -
                    (cy * O[i0][0] - sy * O[i0][1]),
                P[i0][1] - d0 * (sy * s[i0][0] + cy * s[i0][1]) -
                    (sy * O[i0][0] + cy * O[i0][1]),
                P[i0][2] - d0 * s0z - O[i0][2],
            };
            int in = pnp2_inliers_ofs(s, P, O, n, yaw, C);
            if (in > best_in) {
                best_in = in;
                b_yaw = yaw;
                memcpy(bC, C, sizeof bC);
            }
        }
    }
    if (best_in < 3) return 0;
    /* refine: (inliers -> linear centre with offset-adjusted rhs ->
     * weighted mean yaw residual), two rounds */
    float yaw = b_yaw, C[3];
    memcpy(C, bC, sizeof C);
    for (int round = 0; round < 2; round++) {
        float cy = cosf(yaw), sy = sinf(yaw);
        float M[9] = { 0 }, b[3] = { 0 };
        float dsum = 0, wsum = 0;
        int cnt = 0;
        for (int m = 0; m < n; m++) {
            float ox = cy * O[m][0] - sy * O[m][1];
            float oy = sy * O[m][0] + cy * O[m][1];
            float qx = P[m][0] - C[0] - ox, qy = P[m][1] - C[1] - oy,
                  qz = P[m][2] - C[2] - O[m][2];
            float nq = sqrtf(qx * qx + qy * qy + qz * qz);
            if (nq < VER_MIN_RANGE_M) continue;
            float rx = cy * s[m][0] - sy * s[m][1];
            float ry = sy * s[m][0] + cy * s[m][1];
            float rz = s[m][2];
            float dot = (qx * rx + qy * ry + qz * rz) / nq;
            if (dot <= VER_COS_TOL) continue;
            cnt++;
            float II[9] = { 1 - rx * rx, -rx * ry, -rx * rz,
                            -rx * ry, 1 - ry * ry, -ry * rz,
                            -rx * rz, -ry * rz, 1 - rz * rz };
            /* rays constrain C + Rz*O[m]: move the offset into the rhs */
            float Px = P[m][0] - ox, Py = P[m][1] - oy, Pz = P[m][2] - O[m][2];
            for (int k = 0; k < 9; k++) M[k] += II[k];
            b[0] += II[0] * Px + II[1] * Py + II[2] * Pz;
            b[1] += II[3] * Px + II[4] * Py + II[5] * Pz;
            b[2] += II[6] * Px + II[7] * Py + II[8] * Pz;
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
    int in = pnp2_inliers_ofs(s, P, O, n, yaw, C);
    if (in < best_in) {
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

#define DF_PATCH 4                 /* 9x9 ZNCC window */
#define DF_MAX_DISP 128
#define DF_MIN_Z 2.0f              /* VIO owns the near field */
#define DF_MIN_ZNCC 0.78f
#define DF_MARGIN 0.06f
static uint32_t DF_SYNTH_CTR;      /* map thread only */
static int df_backfill(xr_kf *w, const uint8_t *L, const uint8_t *Rt) {
    if (!STEREOG.set || !GEOM.have) return 0;
    float maxz = VER_MAX_RANGE_M;
    float mind = STEREOG.fx * STEREOG.base / maxz;
    float maxd = STEREOG.fx * STEREOG.base / DF_MIN_Z;
    if (maxd > DF_MAX_DISP) maxd = DF_MAX_DISP;
    if (mind < 1.0f) mind = 1.0f;
    int added = 0;
    float Rq[9];
    q2R(w->q, Rq);
    for (int j = 0; j < w->n_kp && w->n_lm < XR_MAP_KP_PER_KF; j++) {
        if (w->lm_of_kp[j] >= 0) continue;
        int u = (int)(w->kp_uv[j][0] + 0.5f);
        int v = (int)(w->kp_uv[j][1] + 0.5f);
        if (u < DF_PATCH + (int)mind + 1 || u >= XR_OW - DF_PATCH ||
            v < DF_PATCH || v >= XR_OH - DF_PATCH)
            continue;
        /* clamp the search to what THIS column can see instead of
         * rejecting every keypoint left of the global max disparity */
        float kmaxd = maxd;
        if ((float)(u - DF_PATCH) < kmaxd) kmaxd = (float)(u - DF_PATCH);
        /* left patch stats (9x9) */
        float lsum = 0, lsum2 = 0;
        for (int dy = -DF_PATCH; dy <= DF_PATCH; dy++)
            for (int dx = -DF_PATCH; dx <= DF_PATCH; dx++) {
                float px = (float)L[(v + dy) * XR_OW + u + dx];
                lsum += px;
                lsum2 += px * px;
            }
        const float NPX = (float)((2 * DF_PATCH + 1) * (2 * DF_PATCH + 1));
        float lmean = lsum / NPX;
        float lvar = lsum2 / NPX - lmean * lmean;
        if (lvar < 25.0f) continue;            /* flat patch: unmatchable */
        float lstd = sqrtf(lvar);
        float best = -2, second = -2;
        int bd = -1;
        for (int d = (int)mind; d <= (int)kmaxd; d++) {
            int ru = u - d;
            float rsum = 0, rsum2 = 0, dot = 0;
            for (int dy = -DF_PATCH; dy <= DF_PATCH; dy++)
                for (int dx = -DF_PATCH; dx <= DF_PATCH; dx++) {
                    float a = (float)L[(v + dy) * XR_OW + u + dx];
                    float b = (float)Rt[(v + dy) * XR_OW + ru + dx];
                    rsum += b;
                    rsum2 += b * b;
                    dot += a * b;
                }
            float rmean = rsum / NPX;
            float rvar = rsum2 / NPX - rmean * rmean;
            if (rvar < 16.0f) continue;
            float z = (dot / NPX - lmean * rmean) / (lstd * sqrtf(rvar));
            if (z > best) { second = best; best = z; bd = d; }
            else if (z > second) second = z;
        }
        if (bd < 0 || best < DF_MIN_ZNCC || best - second < DF_MARGIN)
            continue;
        float dsub = (float)bd;                 /* TODO subpixel: adequate
                                                 * for landmark seeding */
        float z = STEREOG.fx * STEREOG.base / dsub;
        float rc[3];
        if (GEOM.unproject(w->kp_uv[j][0], w->kp_uv[j][1], rc)) continue;
        if (rc[2] < 0.1f) continue;
        float s = z / rc[2];
        float pc[3] = { rc[0] * s, rc[1] * s, rc[2] * s };
        float pb[3] = {
            GEOM.R_ic[0] * pc[0] + GEOM.R_ic[1] * pc[1] +
                GEOM.R_ic[2] * pc[2] + GEOM.p_ic[0],
            GEOM.R_ic[3] * pc[0] + GEOM.R_ic[4] * pc[1] +
                GEOM.R_ic[5] * pc[2] + GEOM.p_ic[1],
            GEOM.R_ic[6] * pc[0] + GEOM.R_ic[7] * pc[1] +
                GEOM.R_ic[8] * pc[2] + GEOM.p_ic[2],
        };
        int li = w->n_lm++;
        w->lm_id[li] = (int32_t)(0x40000000u | (DF_SYNTH_CTR++ & 0x3FFFFFFFu));
        w->lm_xyz[li][0] = Rq[0] * pb[0] + Rq[1] * pb[1] + Rq[2] * pb[2] + w->p[0];
        w->lm_xyz[li][1] = Rq[3] * pb[0] + Rq[4] * pb[1] + Rq[5] * pb[2] + w->p[1];
        w->lm_xyz[li][2] = Rq[6] * pb[0] + Rq[7] * pb[1] + Rq[8] * pb[2] + w->p[2];
        w->lm_of_kp[j] = li;
        added++;
    }
    return added;
}

/* XR_LMDESC direct relocalization: match the query's descriptors straight
 * against the lifetime landmark bank (no keyframe retrieval at all) and
 * PnP the pairs. Runs ONLY when the normal retrieval channel produced no
 * verified candidate while relocalizing — the coverage-failure fallback:
 * a landmark seen from anywhere is matchable, no matter which keyframe
 * retrieval would have surfaced. Covis proxy = distinct LIVE owner
 * keyframes among the geometric inliers. */
static int lmb_reloc(const xr_kf *w, int nkf, float Dq[4], float Dp[3],
                     int *out_n3, int *out_nin, int *out_covis,
                     int *out_best) {
    *out_n3 = *out_nin = *out_covis = 0;
    *out_best = -1;
    if (!GEOM.have || w->n_kp <= 0) return 0;
    if (LMB_LIVE < VER_MIN_PAIRS) return 0;   /* bank too thin to solve */
    int bad = w->desc_type == DESC_BAD;
    static int ord_of[XR_MAP_MAX_KF];      /* slot -> time-order, live only */
    for (int i = 0; i < XR_MAP_MAX_KF; i++) ord_of[i] = -1;
    /* nkf = the DEQUEUE snapshot, never live KF_N (snapshot discipline —
     * a concurrent reset zeroes KF_N mid-scan; review finding) */
    for (int k = 0; k < nkf; k++) ord_of[KFO[k]] = k;

    static float Sw[COVIS_MAX_CAND][3], Pw[COVIS_MAX_CAND][3];
    static int16_t owner[COVIS_MAX_CAND], oseg[COVIS_MAX_CAND];
    static uint8_t used[LMB_MAX / 8];      /* one query kp per bank entry */
    memset(used, 0, sizeof used);
    int n = 0;
    float Rq[9];
    q2R(w->q, Rq);
    for (int i = 0; i < w->n_kp && n < COVIS_MAX_CAND; i++) {
        int best = bad ? 999 : -(1 << 30), second = best, bt = -1;
        for (int t = 0; t < LMB_MAX; t++) {
            const lmb_ent *e = &LMB[t];
            if (e->nobs == 0 || e->desc_type != (uint8_t)w->desc_type)
                continue;
            if (used[t >> 3] & (1u << (t & 7))) continue;
            if (bad) {
                int d = hamming256(w->desc.bad[i], e->desc);
                if (d < best) { second = best; best = d; bt = t; }
                else if (d < second) second = d;
            } else {
                int d = dot64_i8(w->desc.xfeat[i], e->desc);
                if (d > best) { second = best; best = d; bt = t; }
                else if (d > second) second = d;
            }
        }
        if (bt < 0) continue;
        int accept = bad
            ? (best <= BAD_MAX_DIST && best + BAD_MARGIN <= second)
            : (best >= XF_MIN_DOT && best - XF_MARGIN >= second);
        if (!accept) continue;
        const lmb_ent *e = &LMB[bt];
        int oo = ord_of[e->slot];
        if (oo < 0) continue;              /* owner evicted: 3D untrusted */
        const xr_kf *kf = &KF[e->slot];
        if ((int)e->li >= kf->n_lm) continue;   /* slot reused by a smaller
                                                   frame: tail is stale */
        if (kf->lm_id[e->li] != e->id) continue;      /* slot reused */
        float dk2 = 0;
        for (int c = 0; c < 3; c++) {
            float d = kf->lm_xyz[e->li][c] - kf->p[c];
            dk2 += d * d;
        }
        if (dk2 > VER_MAX_RANGE_M * VER_MAX_RANGE_M) continue;
        /* landmark -> session frame via the owner's live correction */
        float qoi[4], poi[3], qd[4], pd[3], ps[3];
        pose_invert(kf->q, kf->p, qoi, poi);
        pose_compose(kf->qc, kf->pc, qoi, poi, qd, pd);
        qrotv(qd, kf->lm_xyz[e->li], ps);
        /* query bearing in the (gravity-true) odom frame */
        float rc[3], rb[3];
        if (GEOM.unproject(w->kp_uv[i][0], w->kp_uv[i][1], rc)) continue;
        rb[0] = GEOM.R_ic[0]*rc[0] + GEOM.R_ic[1]*rc[1] + GEOM.R_ic[2]*rc[2];
        rb[1] = GEOM.R_ic[3]*rc[0] + GEOM.R_ic[4]*rc[1] + GEOM.R_ic[5]*rc[2];
        rb[2] = GEOM.R_ic[6]*rc[0] + GEOM.R_ic[7]*rc[1] + GEOM.R_ic[8]*rc[2];
        Sw[n][0] = Rq[0]*rb[0] + Rq[1]*rb[1] + Rq[2]*rb[2];
        Sw[n][1] = Rq[3]*rb[0] + Rq[4]*rb[1] + Rq[5]*rb[2];
        Sw[n][2] = Rq[6]*rb[0] + Rq[7]*rb[1] + Rq[8]*rb[2];
        Pw[n][0] = ps[0] + pd[0];
        Pw[n][1] = ps[1] + pd[1];
        Pw[n][2] = ps[2] + pd[2];
        owner[n] = e->slot;
        oseg[n] = (int16_t)kf->seg;
        used[bt >> 3] |= (uint8_t)(1u << (bt & 7));
        n++;
    }
    /* frame consistency: unwelded segments live in DIFFERENT session
     * frames — one PnP over mixed frames is meaningless. Keep only the
     * DOMINANT segment's pairs (review finding). */
    if (n > 0) {
        int16_t segs[16];
        int cnt[16], ns = 0;
        for (int m = 0; m < n; m++) {
            int f = -1;
            for (int u = 0; u < ns; u++)
                if (segs[u] == oseg[m]) { f = u; break; }
            if (f < 0 && ns < 16) { segs[ns] = oseg[m]; cnt[ns] = 0; f = ns++; }
            if (f >= 0) cnt[f]++;
        }
        int bs = 0;
        for (int u = 1; u < ns; u++) if (cnt[u] > cnt[bs]) bs = u;
        if (ns > 1) {
            int m2 = 0;
            for (int m = 0; m < n; m++) {
                if (oseg[m] != segs[bs]) continue;
                memcpy(Sw[m2], Sw[m], sizeof Sw[0]);
                memcpy(Pw[m2], Pw[m], sizeof Pw[0]);
                owner[m2] = owner[m];
                oseg[m2] = oseg[m];
                m2++;
            }
            n = m2;
        }
    }
    *out_n3 = n;
    if (n < VER_MIN_PAIRS) return 0;
    float Rz[9], C[3];
    /* bank entries always carry 3D — no far set on this path */
    int nin = pnp2_ransac(Sw, Pw, n, NULL, NULL, 0, Rz, C);
    *out_nin = nin;
    if (nin < VER_MIN_PAIRS || nin * 100 < 33 * n) return 0;
    /* covis proxy + dominant owner over the geometric inliers */
    float cy = Rz[0], sy = Rz[3];
    static int16_t seen_own[COVIS_MAX_CAND];
    static int own_cnt[COVIS_MAX_CAND];
    int nown = 0;
    for (int m = 0; m < n; m++) {
        float qx = Pw[m][0]-C[0], qy = Pw[m][1]-C[1], qz = Pw[m][2]-C[2];
        float nq = sqrtf(qx*qx + qy*qy + qz*qz);
        if (nq < VER_MIN_RANGE_M) continue;
        float rx = cy*Sw[m][0] - sy*Sw[m][1];
        float ry = sy*Sw[m][0] + cy*Sw[m][1];
        if ((qx*rx + qy*ry + qz*Sw[m][2]) / nq <= VER_COS_TOL) continue;
        int f = -1;
        for (int u = 0; u < nown; u++)
            if (seen_own[u] == owner[m]) { f = u; break; }
        if (f < 0) { seen_own[nown] = owner[m]; own_cnt[nown] = 0; f = nown++; }
        own_cnt[f]++;
    }
    *out_covis = nown;
    int bo = 0;
    for (int u = 1; u < nown; u++) if (own_cnt[u] > own_cnt[bo]) bo = u;
    if (nown == 0) return 0;
    *out_best = ord_of[seen_own[bo]];
    /* D (odom -> session), same composition as reloc_pnp step 4 */
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
    /* lazy bank clear (see LMB_EPOCH): map thread only, under the lock */
    {
        static unsigned lmb_seen;          /* map thread only */
        unsigned ep = atomic_load(&LMB_EPOCH);
        if (lmb_seen != ep) {
            memset(LMB, 0, sizeof LMB);
            LMB_LIVE = 0;
            lmb_seen = ep;
        }
    }
    PROBE_REQ = MBOX.probe;
    MBOX.probe = 0;
    int burst_first = MBOX.bfirst, burst_last = MBOX.blast;
    float burst_ofs[3];
    memcpy(burst_ofs, MBOX.bp, sizeof burst_ofs);
    MBOX.bfirst = MBOX.blast = 0;
    int q_only = MBOX.query_only || PROBE_REQ;
    work.ts = MBOX.ts;
    memcpy(work.q, MBOX.q, sizeof work.q);
    memcpy(work.p, MBOX.p, sizeof work.p);
    memcpy(img, MBOX.img, sizeof img);
    static uint8_t img2[XR_OW * XR_OH];
    int have_img2 = MBOX.have2;
    if (have_img2) memcpy(img2, MBOX.img2, sizeof img2);
    MBOX.have2 = 0;
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
    int snap_seg = CUR_SEG;                      /* reset zeroes it (review) */
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
     * resumes), OR relocalization has had SEG_OPEN_NS with no luck and the
     * shake is over: then a NEW SUBMAP opens and mapping resumes there.
     * Post-shake odometry is never trusted back into the OLD frame without
     * a verified closure — the new segment's keyframes carry their own seg
     * id and stay quarantined from the primary until a cross-segment
     * closure welds them. */
    int shaking = atomic_load(&SHAKING);
    int rstate_prev = atomic_load(&REC_STATE);
    int rstate = rstate_prev;
    int open_seg = 0;
    if (shaking && kfn >= REC_MIN_MAP) {
        rstate = REC_LOST;
    } else if (rstate == REC_RECOVERED) {
        if (work.ts - snap_recovered_ns > REC_STABLE_NS) rstate = REC_HEALTHY;
    }
    if (rstate == REC_LOST) {
        if (!LOST_SINCE_NS) {
            LOST_SINCE_NS = work.ts;
            LOGI("session map: LOST (%d keyframes stored) — relocalizing; "
                 "submap opens in %.0fs if unrecovered", kfn,
                 (double)(SEG_OPEN_NS * 1e-9));
        } else if (!shaking && work.ts - LOST_SINCE_NS > SEG_OPEN_NS) {
            /* the give-up-into-a-new-submap transition. The CUR_SEG bump
             * itself happens gen-guarded under MAP_LOCK below — never here. */
            open_seg = 1;
            rstate = REC_HEALTHY;
            LOST_SINCE_NS = 0;
        }
    } else {
        LOST_SINCE_NS = 0;
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
    /* duty cap now applies to EVERY cadence, not just LOST (review: the
     * healthy cadence was ungoverned while pass costs — extract + embed +
     * LG — starve offer processing and collapse map density) */
    if ((uint64_t)last_search_cost_us * 4000ull > search_iv)
        search_iv = (uint64_t)last_search_cost_us * 4000ull;  /* us->ns, x4 */
    /* reactivation-lite: while an anchor is fresh, verify against it at a
     * fast cadence (single-kf query — negligible cost) */
    /* REACT2 standalone works (anchor machinery lives under either flag);
     * probes are EXCLUDED — a pinned scan would test one keyframe instead
     * of the map, and the shared probe timestamp defeats the fail-out
     * (review findings #13/#15). */
    int react_active = (REACT || react2_on()) && !lost && !PROBE_REQ &&
                       REACT_A.kf >= 0 &&
                       work.ts - REACT_A.ts < REACT_WINDOW_NS;
    if (react_active && search_iv > REACT_IV_NS) search_iv = REACT_IV_NS;
    int do_search = last_search_ns == 0 || q_only ||
                    work.ts - last_search_ns >= search_iv;
    /* XR_ROTSTORE: heading change > ~25 deg unlocks an early store (at a
     * third of the interval) — turns get viewpoint coverage */
    int rot_kick = 0;
    if (rotstore_on() && LAST_STORE_Q[0] != 0.f &&
        work.ts - snap_last_store_ns >= STORE_MIN_INTERVAL_NS / 3) {
        float qi[4], qe[4], rv[3];
        qconj(LAST_STORE_Q, qi);
        qmul(qi, work.q, qe);
        rv_from_q(qe, rv);
        rot_kick = rv[0]*rv[0] + rv[1]*rv[1] + rv[2]*rv[2] >
                   ROTSTORE_RAD * ROTSTORE_RAD;
    }
    int may_store = mapping && !q_only && !lost && !shake_freeze &&
                    work.n_lm >= STORE_MIN_LM &&
                    (work.ts - snap_last_store_ns >= STORE_MIN_INTERVAL_NS ||
                     rot_kick);
    if (!do_search && !may_store) {
        /* nothing heavy to do this pass, but a state TRANSITION (e.g. a shake
         * flipping HEALTHY->LOST on a pass that neither searches nor stores)
         * must still persist — gen-safely, and only when it actually changed
         * so the common no-work path stays off the lock. */
        if (rstate != rstate_prev || open_seg) {
            pthread_mutex_lock(&MAP_LOCK);
            if (atomic_load(&RESET_GEN) == rgen0) {
                if (open_seg) {
                    CUR_SEG++;
                    LOGI("session map: no verified recovery in %.0fs — "
                         "SUBMAP seg=%d opened, mapping resumes (weld on "
                         "cross-segment closure)",
                         (double)(SEG_OPEN_NS * 1e-9), CUR_SEG);
                }
                atomic_store(&REC_STATE, rstate);
            }
            pthread_mutex_unlock(&MAP_LOCK);
        }
        return;
    }
    if (do_search) last_search_ns = work.ts;
    uint64_t t_pass0 = map_mono_us();      /* FULL pass cost (extract+embed+
                                              search) feeds the duty cap */

    /* descriptors: XFeat when selected AND available, BAD/TEBLID otherwise */
    if (atomic_load(&USE_XFEAT) && MODEL_PATH[0]) xr_xfeat_init(MODEL_PATH);
    if (atomic_load(&USE_XFEAT) && xr_xfeat_available()) {
        /* landmark ANCHORING (dense paths): reserve budget so every VIO
         * landmark gets a descriptor sampled AT its exact uv — the 2D-3D
         * stage stops being starved by the proximity join below (probe
         * ledgers: bestm 48-109 image matches collapsing to n3 0-36
         * landmark-backed pairs = the corridor reloc killer, the MOO07
         * xfeat regression, and the rooms-xfeat 13 cm fleet outlier). */
        int anchors = xr_xfeat_can_sample() ? work.n_lm : 0;
        if (anchors > XR_MAP_KP_PER_KF / 2) anchors = XR_MAP_KP_PER_KF / 2;
        static int8_t adesc[XR_MAP_KP_PER_KF][64];   /* map thread only */
        int anchored = 0;
        /* extract + anchor-sample under ONE xr_xfeat lock hold: a separate
         * sample call could race the XR_SEED thread's extract and store
         * descriptors from the WRONG frame's dense map (review finding) */
        int n = xr_xfeat_extract_anchored(img, work.kp_uv, work.desc.xfeat,
                                          XR_MAP_KP_PER_KF - anchors,
                                          (const float (*)[2])work.lm_uv,
                                          anchors, adesc, &anchored);
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
            /* append the anchored descriptors: exact uv, exact landmark */
            if (anchored == anchors && anchors > 0) {
                memcpy(&work.desc.xfeat[work.n_kp], adesc,
                       sizeof(int8_t) * 64 * (size_t)anchors);
                for (int i = 0; i < anchors; i++) {
                    work.kp_uv[work.n_kp + i][0] = work.lm_uv[i][0];
                    work.kp_uv[work.n_kp + i][1] = work.lm_uv[i][1];
                    work.lm_of_kp[work.n_kp + i] = i;
                }
                work.n_kp += anchors;
            }
        } else {
            bad_extract(img, &work);
        }
    } else {
        bad_extract(img, &work);
    }

    /* place-recognition embedding for retrieval pre-ranking. Cheap no-op
     * until a VPR model is registered (and permanently off after a failed
     * bring-up). Runs on this (map) thread at keyframe/search rate.
     * XR_STOREGUARD: store-only passes skip the embed — map-thread
     * throughput IS map density (stores collapse when passes run long);
     * an embedding-less keyframe stays always-eligible in retrieval. */
    if (!storeguard_on() || do_search) {
        work.emb_dim = xr_vpr_embed(img, work.emb);
        work.has_emb = work.emb_dim > 0;
    } else {
        work.emb_dim = 0;
        work.has_emb = 0;
    }

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
    static float burst_exp_S[BURST_EXPORT_MAX][3],   /* map thread only */
                 burst_exp_P[BURST_EXPORT_MAX][3];
    int burst_exp_n = 0;
    int searched = 0;                  /* keyframes actually match-scored */
    unsigned match_us = 0;             /* telemetry: match+PnP wall time */
    int use_vpr = 0;                   /* retrieval pre-rank active */
    float vpr_top = 0;                 /* best cosine this search (ledger) */
    int vpr_top_i = -1;                /* its keyframe (XR_TRUSTVPR rider) */
    float bDq[4], bDp[3];
    /* XR_LMFACT: the WINNING candidate's inlier 2D-3D pairs (query pixel +
     * session-frame landmark), kept for the apply branch (map thread only) */
    static float lf_uv[LMFACT_MAX][2], lf_ps[LMFACT_MAX][3];
    int lf_n = 0;
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
    int gate = 0, sweep = 0;
    float wsp[3] = { 0, 0, 0 };
    if (!relocalizing) {
        sweep = (++healthy_search_n % FULL_SWEEP_EVERY) == 0;
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
    /* reactivation-lite: pin the scan to the fresh anchor — no retrieval,
     * no gate; one match+PnP against a known-good keyframe.
     * XR_REACT2 instead keeps the FULL scan and only force-includes the
     * anchor as an extra candidate afterwards — the v12 fastbench showed
     * pinning starves distant-loop retrieval (corridor2 29->50). */
    int scan_lo = 0, scan_hi = lim;
    if (react_active && REACT_A.kf < lim && !react2_on()) {
        scan_lo = REACT_A.kf;
        scan_hi = REACT_A.kf + 1;
        gate = 0;
    }
    use_vpr = work.has_emb && lim > VPR_MIN_KF && !(scan_hi - scan_lo == 1);
    /* XR_RELOCSWEEP: while RELOCALIZING, the shortlist is a liability —
     * MegaLoc aliases repetitive corridors and can push the TRUE revisit
     * out of the top-12, and unlike healthy loop search there is no
     * periodic full sweep here. Measured: retrieval-blind corridor1
     * probes brute-scanned to 23-37%% recall; the shortlist got 3%%.
     * Full descriptor scan instead (the LOST cost-cap paces cadence). */
    if (relocsweep_on() && relocalizing) use_vpr = 0;
    if (use_vpr) {
        memset(vpr_pick, 0, (size_t)lim);
        /* XR_DESPERATE: long-LOST (or one-shot probes) widen the shortlist
         * and lower the similarity floor — every extra candidate is a PnP
         * try we can afford when the alternative is staying lost */
        int wide = desperate_on() &&
                   (q_only || (lost && LOST_SINCE_NS &&
                               work.ts - LOST_SINCE_NS > DESPERATE_AFTER_NS));
        const int shortn = wide ? VPR_SHORTLIST_MAX : VPR_SHORTLIST;
        const float minsim = wide ? DESPERATE_MIN_SIM : VPR_MIN_SIM;
        float bs[VPR_SHORTLIST_MAX];
        int bi[VPR_SHORTLIST_MAX], bn = 0;
        for (int i = 0; i < lim; i++) {
            /* pre-VPR or other-model keyframes stay always-eligible */
            if (!KFA(i).has_emb || KFA(i).emb_dim != work.emb_dim) {
                vpr_pick[i] = 1;
                continue;
            }
            const float *a = work.emb, *b = KFA(i).emb;
            float s = 0;
            for (int d = 0; d < work.emb_dim; d++) s += a[d] * b[d];
            if (s > vpr_top) { vpr_top = s; vpr_top_i = i; }
            /* XR_SEQVOTE: rank by score + decayed history (use the OLD
             * vote, then fold this search's score in) */
            if (seqvote_on() && relocalizing) {
                int sl = KFO[i];
                float sv = SEQV[sl];
                SEQV[sl] = sv * SEQV_DECAY + s;
                s += SEQV_W * sv;
            }
            if (s < minsim) continue;
            int pos = bn < shortn
                          ? bn : (s > bs[shortn - 1] ? shortn - 1
                                                     : -1);
            if (pos < 0) continue;
            if (bn < shortn) bn++;
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
        /* Periodic full-recall sweep, same cadence the non-VPR path gets:
         * appearance embeddings alias on repetitive structure (corridor
         * segments all "look alike"), pushing the TRUE revisit out of the
         * shortlist — matrix v4 measured plain descriptor scan beating both
         * VPR arms on TUM-VI long for exactly this reason. The sweep
         * restores worst-case recall at the cost the BAD arm already pays. */
        if (sweep) use_vpr = 0;
    }
    for (int i = scan_lo; i < scan_hi; i++) {
        if (use_vpr && !vpr_pick[i]) continue;
        /* the spatial gate only means anything within OUR segment; other
         * segments' pc live in a different frame — leave them always
         * eligible, they are exactly the weld opportunities. (Only bites
         * while unwelded segments exist; the VPR shortlist supersedes the
         * gate anyway on the main arms.) */
        if (gate && KFA(i).seg == snap_seg) {
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
        for (int t = 0; t < ncand; t++) {   /* SESSION-frame distance (pc);
                                               only comparable within a
                                               segment — different segments
                                               are always distinct clusters */
            if (KFA(i).seg != KFA(cand_i[t]).seg) continue;
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

    /* XR_TRUSTVPR rider: retrieval is our strongest signal — when it is
     * confident, the top keyframe goes to LighterGlue+PnP even though the
     * NN prematch could not scrape CAND_MIN_MATCHES from this viewpoint */
    if (trustvpr_on() && relocalizing && use_vpr && vpr_top >= TRUSTVPR_MIN &&
        vpr_top_i >= 0) {
        int have = 0;
        for (int t2 = 0; t2 < ncand; t2++)
            if (cand_i[t2] == vpr_top_i) { have = 1; break; }
        if (!have && ncand < RELOC_TOPK) {
            cand_i[ncand] = vpr_top_i;
            cand_m[ncand] = CAND_MIN_MATCHES;  /* nominal score */
            ncand++;
        }
    }

    /* XR_REACT2: the fresh anchor rides along as an EXTRA candidate —
     * full retrieval above is untouched, so distant loops keep their
     * shot while the anchor gets its cheap fast-cadence verify. */
    if (react2_on() && react_active && REACT_A.kf < lim) {
        int have = 0;
        for (int t = 0; t < ncand; t++)
            if (cand_i[t] == REACT_A.kf) { have = 1; break; }
        if (!have && ncand < RELOC_TOPK) {
            cand_i[ncand] = REACT_A.kf;
            cand_m[ncand] = CAND_MIN_MATCHES;      /* nominal score */
            ncand++;
        }
    }

    /* verify every candidate cluster; keep the geometrically strongest
     * (most inliers, covisibility-backed) — still lock-free */
    for (int c = 0; c < ncand; c++) {
        int cn3 = 0, cnin = 0, ccov = 0;
        float cDq[4], cDp[3];
        static float cuv[LMFACT_MAX][2], cps[LMFACT_MAX][3]; /* map thread */
        int cnl = 0;
        int want_lf = lmfact_on();
        static float ebS[BURST_EXPORT_MAX][3], ebP[BURST_EXPORT_MAX][3];
        int ebn = 0;
        int want_burst = PROBE_REQ && burstpnp_on();
        int ok = reloc_pnp(&work, cand_i[c], kfn, cDq, cDp, &cn3, &cnin, &ccov,
                           want_lf ? cuv : NULL, want_lf ? cps : NULL,
                           want_lf ? &cnl : NULL,
                           want_burst ? ebS : NULL, want_burst ? ebP : NULL,
                           want_burst ? &ebn : NULL);
        /* keep the RICHEST candidate's matches for the burst accumulator */
        if (want_burst && ebn > burst_exp_n) {
            memcpy(burst_exp_S, ebS, sizeof(float) * 3 * (size_t)ebn);
            memcpy(burst_exp_P, ebP, sizeof(float) * 3 * (size_t)ebn);
            burst_exp_n = ebn;
        }
        if (cn3 > any_pairs) any_pairs = cn3;
        if (ok && ccov >= COVIS_MIN_KF && cnin > best_nin) {
            best_i = cand_i[c]; best_m = cand_m[c];
            best_n3 = cn3; best_nin = cnin; best_covis = ccov;
            memcpy(bDq, cDq, sizeof bDq);
            memcpy(bDp, cDp, sizeof bDp);
            if (want_lf) {                 /* keep the winner's pairs */
                memcpy(lf_uv, cuv, sizeof lf_uv);
                memcpy(lf_ps, cps, sizeof lf_ps);
                lf_n = cnl;
            }
        }
    }
    /* XR_LMDESC fallback: retrieval surfaced nothing verifiable while
     * relocalizing — go straight at the landmark bank (coverage-
     * independent). Result feeds the SAME apply chain below. */
    if (lmdesc_on() && relocalizing && best_i < 0) {
        int bi2, n32, nin2, cov2;
        float dq2[4], dp2[3];
        if (lmb_reloc(&work, kfn, dq2, dp2, &n32, &nin2, &cov2, &bi2) &&
            bi2 >= 0) {
            best_i = bi2;
            best_n3 = n32;
            best_nin = nin2;
            best_covis = cov2;
            lf_n = 0;    /* pairs (if any) came from a different solve */
            memcpy(bDq, dq2, sizeof bDq);
            memcpy(bDp, dp2, sizeof bDp);
            if (n32 > any_pairs) any_pairs = n32;
            if (ncand == 0) {              /* apply branch needs a candidate */
                cand_i[0] = bi2;
                cand_m[0] = CAND_MIN_MATCHES;
                ncand = 1;
            }
            /* LMDESC carries no image-match evidence of its own — never
             * unlock the strong-jump caps with an unrelated cluster's
             * match count (review finding) */
            best_m = CAND_MIN_MATCHES;
            LOGI("session map: LMIDX direct reloc — %d pairs, %d inliers, "
                 "%d live owners (kf#%d)", n32, nin2, cov2, bi2);
        }
    }
    /* XR_BURSTPNP: accumulate this frame's matches; joint-solve on last */
    BSOLVE.ok = 0;
    if (PROBE_REQ && burstpnp_on()) {
        if (burst_first) { BURST.active = 1; BURST.n = 0; BURST.nframes = 0; }
        if (BURST.active && burst_exp_n > 0 && BURST.nframes < 255) {
            for (int m = 0; m < burst_exp_n && BURST.n < BURST_MAX; m++) {
                memcpy(BURST.S[BURST.n], burst_exp_S[m], sizeof BURST.S[0]);
                memcpy(BURST.P[BURST.n], burst_exp_P[m], sizeof BURST.P[0]);
                memcpy(BURST.O[BURST.n], burst_ofs, sizeof BURST.O[0]);
                BURST.fid[BURST.n] = (uint8_t)BURST.nframes;
                BURST.n++;
            }
            BURST.nframes++;
        }
        if (burst_last && BURST.active) {
            if (BURST.nframes >= 2 && BURST.n >= VER_MIN_PAIRS) {
                float Rz[9], C0[3];
                int jin = pnp2_ransac_burst(BURST.S, BURST.P, BURST.O,
                                            BURST.fid, BURST.n, Rz, C0);
                /* pooled multi-frame consensus: absolute floor + a softer
                 * ratio (single-frame noise is diluted across the pool) */
                if (jin >= VER_MIN_PAIRS && jin * 100 >= 25 * BURST.n) {
                    float Cl[3] = {   /* last frame's camera centre */
                        C0[0] + Rz[0] * burst_ofs[0] + Rz[1] * burst_ofs[1],
                        C0[1] + Rz[3] * burst_ofs[0] + Rz[4] * burst_ofs[1],
                        C0[2] + burst_ofs[2],
                    };
                    R2q(Rz, BSOLVE.Dq);
                    float qsb[4], t3[3], body_s[3];
                    qmul(BSOLVE.Dq, work.q, qsb);
                    qrotv(qsb, GEOM.p_ic, t3);
                    body_s[0] = Cl[0] - t3[0];
                    body_s[1] = Cl[1] - t3[1];
                    body_s[2] = Cl[2] - t3[2];
                    qrotv(BSOLVE.Dq, work.p, t3);
                    BSOLVE.Dp[0] = body_s[0] - t3[0];
                    BSOLVE.Dp[1] = body_s[1] - t3[1];
                    BSOLVE.Dp[2] = body_s[2] - t3[2];
                    BSOLVE.nin = jin;
                    BSOLVE.ok = 1;
                    LOGI("session map: BURSTPNP joint solve %d/%d inliers "
                         "over %d frames", jin, BURST.n, BURST.nframes);
                }
            }
            BURST.active = 0;
        }
    }
    match_us = (unsigned)(map_mono_us() - t_match0);
    /* closure ledger: one line per search so retrieval recall/precision is
     * measurable offline (the benchmark GT-labels candidates post-hoc) */
    /* sweep-mode searches (XR_RELOCSWEEP) and probes must stay measurable
     * too — the drive-reloc 0/30 was invisible because this line was
     * vpr-gated. Extra fields go at the END (parsers key on the prefix). */
    if (use_vpr || q_only || lost)
        LOGI("session map: LEDGER q=%llu vprtop=%.3f searched=%d cand=%d "
             "bestm=%d n3=%d nin=%d lost=%d cov=%d%s%s",
             (unsigned long long)work.ts, (double)vpr_top, searched, ncand,
             raw_best_m, any_pairs, best_nin, lost, best_covis,
             use_vpr ? "" : " sweep", PROBE_REQ ? " probe" : "");
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
    if (open_seg) {
        CUR_SEG++;
        LOGI("session map: no verified recovery in %.0fs — SUBMAP seg=%d "
             "opened, mapping resumes (weld on cross-segment closure)",
             (double)(SEG_OPEN_NS * 1e-9), CUR_SEG);
    }
    atomic_store(&REC_STATE, rstate);
    if (did_search) {
        PERF.searched = searched;
        PERF.candidates = ncand;
        PERF.match_us = match_us;
        PERF.lock_us = lock_us;
        last_search_cost_us =             /* feeds the cadence duty cap */
            (unsigned)(map_mono_us() - t_pass0);
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
        if (KFA(i).seg != CUR_SEG) continue;   /* pc comparable per-segment */
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
            /* cross-segment match: the verified D registers our live odom
             * into the MATCHED segment's frame. Deviation-vs-CORR and the
             * jump caps are meaningless across segments (the inter-segment
             * offset is arbitrary by construction) — the 2-frame
             * confirmation is the entire gate, and the payoff is a WELD. */
            int xseg = KFA(best_i).seg != CUR_SEG;
            VER_LAST.pairs = n3;
            VER_LAST.inliers = nin;
            KFA(best_i).last_used = work.ts;   /* geometrically useful */
            if ((REACT || react2_on()) && !lost && !PROBE_REQ) {
                REACT_A.kf = best_i;           /* refresh reactivation anchor */
                REACT_A.ts = work.ts;
                REACT_A.fails = 0;
            }
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
            /* Strong single-frame confirm — LOST RECOVERY ONLY. Recovery
             * benefits from fast re-anchoring and any alignment beats being
             * lost; healthy tracking must keep the 2-frame agreement, or
             * transient deviation spikes snap a good VIO onto stale map
             * error (EuRoC: 12 such snaps turned 6.5cm ATE into 20cm+). */
            if (!confirmed && lost && !xseg && nin >= 2 * VER_STRONG_INLIERS &&
                nin * 100 >= 60 * n3 && covis >= 12)
                confirmed = 1;   /* !xseg: a weld rewrites stored keyframe
                                  * poses irreversibly — always 2-frame */

            /* A verified alignment is worth pursuing when: we are LOST (ANY
             * alignment is a recovery candidate — even a small deviation,
             * from a shake that didn't actually disturb Basalt), OR we are
             * healthy and the VIO has drifted SIGNIFICANTLY from the map and
             * the cooldown has passed. Otherwise the VIO already agrees. */
            int worth = lost || xseg || (significant && cooled);

            if (PROBE_REQ) {
                /* reloc-benchmark probe: record the verified alignment (the
                 * query's session-frame pose, since the probe's odom pose is
                 * identity) and change NOTHING — no apply, no pending.
                 * NOTE: this whole section already runs under MAP_LOCK —
                 * re-locking here self-deadlocked the first live test. */
                PROBE_RES.ok = 1;
                PROBE_RES.inliers = nin;
                PROBE_RES.kf = best_i;
                /* session ORIENTATION of the query = D ∘ odom-quat (the
                 * gravity prior made work.q non-identity; returning raw D
                 * was only correct for identity probes) */
                qmul(Dq, work.q, PROBE_RES.q);
                /* session position = D o (odom pose); classic probes have
                 * p = 0 so this reduces to Dp, but burst frames carry the
                 * intra-burst VIO position */
                {
                    float t3[3];
                    qrotv(Dq, work.p, t3);
                    PROBE_RES.p[0] = Dp[0] + t3[0];
                    PROBE_RES.p[1] = Dp[1] + t3[1];
                    PROBE_RES.p[2] = Dp[2] + t3[2];
                }
            } else if (!xseg && (dev > mxt || sang > mxa)) {
                VER_LAST.outcome = VOUT_CAPPED;
                PENDING_D.have = 0;
                LOGI("session map: kf#%d PnP GOOD (%d/%d inliers, %d covis) "
                     "but |t|=%.2fm ang=%.0fdeg exceeds caps — wrong-place "
                     "match, ignored", best_i, nin, n3, covis, (double)dev,
                     (double)(sang * 57.3f));
            } else if (!worth) {
                /* healthy and the VIO agrees with the map (or we just
                 * snapped): do nothing — a recovery, not a continuous clamp.
                 * KEEP any pending candidate (sub-gate flicker must not kill
                 * the 2-frame confirmation); it expires via CONFIRM_WINDOW.
                 * NOTE a continuous sub-gate servo was tried here and pulled
                 * EuRoC map ATE 12→40 cm: with a biased map ANY steady pull
                 * toward it compounds error. Needs map-vs-VIO confidence
                 * weighting first (see bench notes).
                 * TIGHT mode is different: the pull goes INTO the VIO
                 * optimizer as a weak prior, arbitrated against IMU and
                 * vision factors — post it when this verified frame agrees
                 * with the previous one (2-frame consistency) AND the
                 * matched keyframe is a genuine revisit (see
                 * TIGHT_REVISIT_NS). */
                if (TIGHT && confirmed &&
                    work.ts - KFA(best_i).ts > TIGHT_REVISIT_NS)
                    tight_post_prior(Dq, Dp, work.ts);
                /* XR_LMFACT: sub-gate closures also feed the landmark-
                 * factor channel — NO revisit-age gate: a factor from a
                 * recent (self-drift-correlated) keyframe is arbitrated
                 * per point by the optimizer instead of gluing the whole
                 * pose like the prior did (the v9 failure), and room-scale
                 * revisits younger than 30s are exactly what the prior
                 * channel could never absorb. */
                if (lmfact_on() && lf_n > 0) {
                    lmfact_post(lf_uv, lf_ps, lf_n, work.ts);
                    if (lmtrack_on())
                        lmt_capture(&work, lf_uv, lf_ps, lf_n, work.ts);
                }
                /* XR_EDGEGRAPH: sub-gate closures are verified geometry
                 * too — remember them as edges and relax the chain at a
                 * bounded cadence (this is the corridors 1-4 lever) */
                if (edgegraph_on()) {
                    static uint64_t EG_LAST_NS;   /* map thread only */
                    eg_admit(best_i, work.q, work.p, Dq, Dp, nin);
                    if (work.ts - EG_LAST_NS > 1000000000ull) {
                        EG_LAST_NS = work.ts;
                        eg_relax();
                    }
                }
                /* XR_LOCALBA: a VERIFIED sub-gate closure is the rooms
                 * regime — refine the window even though no correction
                 * applies (rate-limited; structure is what improves). */
                if (localba_on() && work.ts - LBA_LAST_NS > 1000000000ull) {
                    LBA_LAST_NS = work.ts;
                    local_ba(best_i);
                }
                /* XR_TIGHTSUB: arm the 2-frame confirmation for SUB-GATE
                 * closures too. Without this, PENDING is only ever set
                 * above the 0.5m gate, so `confirmed` can never become
                 * true down here and the revisit-aged tight posting above
                 * is dead code on low-drift sequences — the whole reason
                 * rooms/EuRoC/MSD sit at exact VIO parity while OKVIS2+LC
                 * absorbs sub-gate corrections continuously. The optimizer
                 * arbitrates the prior; the revisit-age gate keeps the v9
                 * self-drift-teaching failure out. */
                if (tightsub_on() && TIGHT && !confirmed) {
                    PENDING_D.have = 1;
                    PENDING_D.ts = work.ts;
                    memcpy(PENDING_D.q, Dq, sizeof PENDING_D.q);
                    memcpy(PENDING_D.p, Dp, sizeof PENDING_D.p);
                }
                VER_LAST.outcome = VOUT_GATED;
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
                /* XR_SEGQUIET: while an unwelded submap is active, its
                 * registration is arbitrary — a within-segment display
                 * snap just moves the output around in the GT/primary
                 * frame (MH01 kidnap: 133->181 cm transient). Keep the
                 * map-internal healing, skip the display step; the WELD
                 * is the one snap that means something. */
                if (segquiet_on() && CUR_SEG != 0 && !xseg && !lost) snap = 0;
                /* (!lost: a verified post-loss RECOVERY inside a submap
                 * removes real coasted-IMU error — suppressing THAT snap
                 * leaves the live pose permanently wrong; SEGQUIET only
                 * quiets healthy within-submap drift snaps.) */
                int tight_applied = 0;
                /* confidence weighting: blend the correction toward the
                 * current CORR by verification strength (healthy only —
                 * recovery must snap fully) */
                if (confw_on() && !lost && !xseg) {   /* a weld needs exact D */
                    float ratio = n3 > 0 ? (float)nin / (float)n3 : 0.f;
                    float w = CONFW_MIN_W + (1.f - CONFW_MIN_W) *
                              (ratio >= CONFW_FULL_RATIO
                                   ? 1.f : ratio / CONFW_FULL_RATIO);
                    if (w < 1.f) {
                        float dot = CORR.q[0]*Dq[0] + CORR.q[1]*Dq[1] +
                                    CORR.q[2]*Dq[2] + CORR.q[3]*Dq[3];
                        float sgn = dot < 0 ? -1.f : 1.f, n2 = 0;
                        for (int c = 0; c < 4; c++) {
                            Dq[c] = (1 - w) * CORR.q[c] * sgn + w * Dq[c];
                            n2 += Dq[c] * Dq[c];
                        }
                        n2 = 1.f / sqrtf(n2);
                        for (int c = 0; c < 4; c++) Dq[c] *= n2;
                        for (int c = 0; c < 3; c++)
                            Dp[c] = (1 - w) * CORR.p[c] + w * Dp[c];
                    }
                }
                if (xseg) {
                    /* CROSS-SEGMENT WELD (Atlas-style merge): two confirmed
                     * frames agree on where the live pose sits inside the
                     * matched segment — fuse the two rigid bodies into one
                     * frame. Which side moves is an age rule: the older
                     * segment (ultimately the primary) is the canonical
                     * frame, so welding never yanks the established map. */
                    int src = KFA(best_i).seg;     /* matched segment */
                    float Eq[4], Ep[3], iq[4], ip[3];
                    int moved, dst, from;
                    if (src < CUR_SEG) {
                        /* matched segment is OLDER: adopt ITS frame. Move
                         * our whole segment there via E = D ∘ CORR⁻¹, then
                         * re-register the live odom with CORR = D (the
                         * shared tail below). */
                        dst = src;
                        from = CUR_SEG;
                        pose_invert(CORR.q, CORR.p, iq, ip);
                        pose_compose(Dq, Dp, iq, ip, Eq, Ep);
                        moved = seg_weld(from, dst, Eq, Ep);
                        CUR_SEG = dst;
                    } else if (!lost) {
                        /* matched segment is YOUNGER (an orphan from a later
                         * loss episode, revisited from an older frame): pull
                         * IT onto us via E = CORR ∘ D⁻¹. Our own
                         * registration is already right — neutralize the
                         * tail's CORR step. */
                        dst = CUR_SEG;
                        from = src;
                        pose_invert(Dq, Dp, iq, ip);
                        pose_compose(CORR.q, CORR.p, iq, ip, Eq, Ep);
                        moved = seg_weld(from, dst, Eq, Ep);
                        memcpy(Dq, CORR.q, sizeof Dq);
                        memcpy(Dp, CORR.p, sizeof Dp);
                    } else {
                        /* LOST + younger orphan match: CORR is the STALE
                         * pre-loss registration — using it to move the
                         * orphan would corrupt it and fake a recovery
                         * (review finding). Recover INTO the orphan's frame
                         * instead: no keyframes move, CUR_SEG adopts the
                         * orphan, the tail's CORR <- D registers us there.
                         * A later old-segment match welds everything. */
                        dst = src;
                        from = src;
                        moved = 0;
                        CUR_SEG = src;
                    }
                    CLOUD_DIRTY = 1;
                    if (lost) {                    /* a weld IS a recovery */
                        atomic_store(&REC_STATE, REC_RECOVERED);
                        RECOVERED_NS = work.ts;
                    }
                    LOGI("session map: kf#%d SUBMAP WELD seg=%d -> seg=%d "
                         "(%d keyframes re-registered, offset %.2fm %.0fdeg, "
                         "%d/%d inliers, %d covis)%s", best_i, from, dst,
                         moved, (double)dev, (double)(sang * 57.3f), nin, n3,
                         covis, snap ? " + pose snapped" : "");
                } else if (lost) {
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
                } else if (TIGHT && dev <= TIGHT_MAX_DEV_M &&
                           sang <= TIGHT_MAX_DEV_ANG) {
                    /* TIGHT: hand the confirmed closure to the VIO optimizer
                     * as a weak prior instead of stepping CORR + deforming.
                     * The estimator spreads the correction against IMU and
                     * vision factors; our deviation measurements shrink as
                     * it absorbs, so the map layer converges without ever
                     * discontinuity-stepping the output (OKVIS2 property). */
                    tight_post_prior(Dq, Dp, work.ts);
                    /* XR_LMFACT: the confirmed closure's inlier landmarks
                     * ride along as fixed-3D reprojection factors — the
                     * per-point channel the optimizer arbitrates (CORR is
                     * NOT stepped in this branch, so the CORR⁻¹ transform
                     * inside matches the prior's E = CORR⁻¹∘D exactly). */
                    if (lmfact_on() && lf_n > 0) {
                        lmfact_post(lf_uv, lf_ps, lf_n, work.ts);
                        if (lmtrack_on())
                            lmt_capture(&work, lf_uv, lf_ps, lf_n, work.ts);
                    }
                    if (edgegraph_on()) {
                        eg_admit(best_i, work.q, work.p, Dq, Dp, nin);
                        eg_relax();
                    }
                    tight_applied = 1;
                    LAST_SNAP_NS = work.ts;
                    PENDING_D.have = 0;
                    LOOP_STATS.count++;
                    LOOP_STATS.matches = best_m;
                    VER_LAST.outcome = VOUT_APPLIED;
                    LOGI("session map: LOOP kf#%d TIGHT-PRIOR %.2fm %.0fdeg "
                         "(%d/%d inliers, %d covis) — posted to VIO", best_i,
                         (double)dev, (double)(sang * 57.3f), nin, n3, covis);
                } else {
                    /* HEALTHY accumulated-drift closure: DEFORM the drifted
                     * tail onto the reference (real co-localization). Pass
                     * the query's SESSION position (CORR is still the
                     * pre-closure correction here) so the path weighting is
                     * discontinuity-safe. */
                    float qsp[3];
                    qrotv(CORR.q, work.p, qsp);
                    qsp[0] += CORR.p[0]; qsp[1] += CORR.p[1]; qsp[2] += CORR.p[2];
                    if (pgo_on()) pgo_deform(best_i, work.q, work.p, Dq, Dp);
                    else          graph_deform(best_i, qsp, Dq, Dp);
                    if (edgegraph_on()) {
                        eg_admit(best_i, work.q, work.p, Dq, Dp, nin);
                        eg_relax();
                    }
                    CLOUD_DIRTY = 1;
                    LOGI("session map: LOOP kf#%d CLOSURE %.2fm %.0fdeg "
                         "(%d/%d inliers, %d covis) — map deformed%s", best_i,
                         (double)dev, (double)(sang * 57.3f), nin, n3, covis,
                         snap ? " + pose snapped" : "");
                }
                /* XR_LOCALBA: an APPLIED closure just re-anchored this
                 * neighbourhood — refine its structure while the consensus
                 * is fresh (both the tight and the deform branch land here) */
                if (localba_on()) {
                    LBA_LAST_NS = work.ts;
                    local_ba(best_i);
                }
                if (!tight_applied) {
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

    /* XR_BURSTPNP: the joint verdict rides over the last frame's probe
     * result — a burst whose frames all failed alone can still verify
     * as one consensus. */
    if (PROBE_REQ && BSOLVE.ok &&
        (!PROBE_RES.ok || BSOLVE.nin > PROBE_RES.inliers)) {
        PROBE_RES.ok = 1;
        PROBE_RES.inliers = BSOLVE.nin;
        qmul(BSOLVE.Dq, work.q, PROBE_RES.q);
        float bt3[3];
        qrotv(BSOLVE.Dq, work.p, bt3);
        PROBE_RES.p[0] = BSOLVE.Dp[0] + bt3[0];
        PROBE_RES.p[1] = BSOLVE.Dp[1] + bt3[1];
        PROBE_RES.p[2] = BSOLVE.Dp[2] + bt3[2];
    }

    /* XR_LMTRACK re-post: while the last closure's landmarks stay in view,
     * every search frame re-matches them and posts fresh factors — the
     * estimator keeps being pulled BETWEEN closures, not only at them.
     * Probes excluded (synthetic queries must not extend the window). */
    if (lmtrack_on() && lmfact_on() && LMT.n > 0 && !PROBE_REQ &&
        work.desc_type != DESC_BAD && work.ts < LMT.until_ns) {
        static float tuv[LMFACT_MAX][2], tps[LMFACT_MAX][3]; /* map thread */
        int tn = 0;
        for (int m = 0; m < LMT.n && tn < LMFACT_MAX; m++) {
            int best = -(1 << 30), second = best, bi = -1;
            for (int i = 0; i < work.n_kp; i++) {
                int d = dot64_i8(LMT.desc[m], work.desc.xfeat[i]);
                if (d > best) { second = best; best = d; bi = i; }
                else if (d > second) second = d;
            }
            if (bi < 0 || best < XF_MIN_DOT || best - XF_MARGIN < second)
                continue;
            tuv[tn][0] = work.kp_uv[bi][0];
            tuv[tn][1] = work.kp_uv[bi][1];
            memcpy(tps[tn], LMT.ps[m], sizeof tps[0]);
            tn++;
        }
        if (tn >= LMT_MIN_MATCH) {
            lmfact_post(tuv, tps, tn, work.ts);
            LMT.until_ns = work.ts + LMT_WINDOW_NS;   /* still seen: slide */
        } else if (tn * 2 < LMT_MIN_MATCH) {
            LMT.n = 0;                                 /* scene moved on */
        }
    }

    /* reactivation-lite bookkeeping: a pinned pass that did NOT refresh the
     * anchor is a failure — after a few, drop back to normal retrieval */
    if (react_active && REACT_A.kf >= 0 && REACT_A.ts != work.ts) {
        if (++REACT_A.fails >= REACT_MAX_FAILS) {
            REACT_A.kf = -1;
            REACT_A.fails = 0;
        }
    }

    /* store (mapping mode only; never for a stationary query — those are
     * matching-only; only frames that carry verifiable geometry, at a
     * bounded rate), rolling cap: evict least-recently-useful. Reuse the
     * SAME may_store gate computed above — it already carries the LOST
     * check, so a search-only pass during a shake/loss can never fall
     * through and insert the contaminated frame. */
    if (may_store) {
        LAST_STORE_NS = work.ts;
        memcpy(LAST_STORE_Q, work.q, sizeof LAST_STORE_Q);
        if (KF_N == XR_MAP_MAX_KF) {
            /* evict the least-recently-useful keyframe: free its SLOT and drop
             * it from the time order. Only the small order array shifts (a few
             * hundred ints), never the ~4 MB of keyframe structs. */
            int vpos = 0;
            if (covkeep_on()) {
                /* XR_COVKEEP: viewpoint-diversity-aware eviction. Redundancy
                 * of a keyframe = how many others share its spatial cell AND
                 * yaw quadrant (same segment — pc is per-segment); evict the
                 * most redundant, ties to least-recently-useful. A corridor
                 * walked once in each direction keeps BOTH headings alive —
                 * the reloc bestm~0 coverage mode's storage-side half. */
                static int cx[XR_MAP_MAX_KF], cy[XR_MAP_MAX_KF],
                           cz[XR_MAP_MAX_KF], yb[XR_MAP_MAX_KF];
                for (int i = 0; i < KF_N; i++) {
                    cx[i] = (int)floorf(KFA(i).pc[0] / COVK_CELL_M);
                    cy[i] = (int)floorf(KFA(i).pc[1] / COVK_CELL_M);
                    cz[i] = (int)floorf(KFA(i).pc[2] / (2.f * COVK_CELL_M));
                    float fwd[3], zax[3] = { 0, 0, 1 };
                    qrotv(KFA(i).qc, zax, fwd);
                    float yaw = atan2f(fwd[1], fwd[0]);
                    yb[i] = ((int)floorf((yaw + 3.14159265f) /
                                         1.5707963f)) & 3;
                }
                int best_red = -1;
                for (int i = 0; i < KF_N; i++) {
                    int red = 0;
                    for (int j = 0; j < KF_N; j++)
                        if (KFA(j).seg == KFA(i).seg && cx[j] == cx[i] &&
                            cy[j] == cy[i] && cz[j] == cz[i] && yb[j] == yb[i])
                            red++;
                    if (red > best_red ||
                        (red == best_red &&
                         KFA(i).last_used < KFA(vpos).last_used)) {
                        best_red = red;
                        vpos = i;
                    }
                }
            } else {
            for (int i = 1; i < KF_N; i++)
                if (KFA(i).last_used < KFA(vpos).last_used) vpos = i;
            }
            SEQV[KFO[vpos]] = 0;                   /* vote dies with the slot */
            KF_FREE[KF_FREE_N++] = KFO[vpos];      /* slot returns to the pool */
            memmove(&KFO[vpos], &KFO[vpos + 1],
                    sizeof(int) * (size_t)(KF_N - 1 - vpos));
            KF_N--;
            /* REACT_A.kf is a TIME-ORDER index: remap across the shift
             * (review finding — a stale index pins verification to the
             * wrong keyframe after eviction) */
            if (REACT_A.kf == vpos) { REACT_A.kf = -1; REACT_A.fails = 0; }
            else if (REACT_A.kf > vpos) REACT_A.kf--;
        }
        /* XR_DEPTHFILL: give landmark-less keypoints stereo 3D before
         * the frame is immortalised (bank + reloc + factors all inherit) */
        if (depthfill_on() && have_img2) {
            int nadd = df_backfill(&work, img, img2);
            if (nadd)
                LOGI("session map: DEPTHFILL +%d stereo landmarks (%d total)",
                     nadd, work.n_lm);
        }
        work.last_used = work.ts;
        /* corrected session pose = current global correction ∘ odom. A
         * confirmed closure this same pass already deformed the tail and
         * (if recovering) updated CORR, so the new tip lands consistent. */
        pose_compose(CORR.q, CORR.p, work.q, work.p, work.qc, work.pc);
        work.seg = CUR_SEG;                        /* tag the live segment */
        int slot = KF_FREE[--KF_FREE_N];           /* a stable slot for life */
        KF[slot] = work;
        KFO[KF_N] = slot;                          /* append in time order */
        KF_N++;
        if (lmdesc_on()) {
            /* lifetime bank upsert: freshest descriptor per landmark id.
             * Anchored keypoints are appended LAST, so when both a maxima
             * kp and an anchor observe a landmark, the anchor's exact-uv
             * descriptor wins. */
            for (int j = 0; j < work.n_kp; j++) {
                int lk = work.lm_of_kp[j];
                if (lk < 0) continue;
                lmb_ent *e = &LMB[lmb_slot_of(work.lm_id[lk])];
                if (e->nobs == 0 || e->id != work.lm_id[lk]) {
                    if (e->nobs == 0) LMB_LIVE++;  /* fresh slot */
                    e->id = work.lm_id[lk];        /* new / collision evict */
                    e->nobs = 0;
                }
                memcpy(e->desc, work.desc_type == DESC_XFEAT
                                    ? (const void *)work.desc.xfeat[j]
                                    : (const void *)work.desc.bad[j],
                       work.desc_type == DESC_XFEAT ? 64 : 32);
                e->desc_type = (uint8_t)work.desc_type;
                e->slot = (int16_t)slot;
                e->li = (uint8_t)lk;
                if (e->nobs < 65535) e->nobs++;
                e->ts = work.ts;
            }
        }
        atomic_store(&KF_COUNT_PUB, KF_N);
        CLOUD_DIRTY = 1;
        LOGI("session map: kf#%d stored (%d landmarks, %d kps, seg=%d)",
             KF_N - 1, work.n_lm, work.n_kp, CUR_SEG);
    }
    /* refresh the authoritative display cloud whenever the graph changed
     * (a store, an eviction, or a closure-driven deformation) */
    if (CLOUD_DIRTY) cloud_rebuild();
    pthread_mutex_unlock(&MAP_LOCK);
}

static void thread_start(void);        /* fwd: used by xr_map_probe below */

static void *map_thread(void *arg) {
    (void)arg;
    setpriority(PRIO_PROCESS, (id_t)gettid(), 19);   /* never outrank VIO */
    pthread_mutex_lock(&MAP_LOCK);
    for (;;) {
        while (!MBOX.full) pthread_cond_wait(&MAP_COND, &MAP_LOCK);
        pthread_mutex_unlock(&MAP_LOCK);
        process_keyframe();
        pthread_mutex_lock(&MAP_LOCK);
        if (PROBE_REQ) {               /* finalize probe on EVERY exit path */
            PROBE_REQ = 0;
            PROBE_RES.done = 1;        /* .ok filled by verify, else 0 */
        }
        /* ALWAYS wake waiters: a probe submitter may be sleeping on
         * "mailbox full" from a preceding regular offer — without this
         * broadcast that wait is a lost-wakeup deadlock (probe hung the
         * first reloc test for an hour at 0% CPU). */
        pthread_cond_broadcast(&MAP_COND);
    }
    return NULL;
}

/* Reloc-benchmark probe: run retrieval + PnP verification for a bare image
 * against the CURRENT map with an identity odom pose; returns 1 with the
 * query's session-frame pose on a verified match, 0 otherwise. Blocking
 * (worst case one VPR embed + full descriptor scan). Bench/test use. */
int xr_map_probe(const uint8_t *img, const float grav_q[4], float out_q[4],
                 float out_p[3], int *out_inliers) {
    pthread_once(&THREAD_ONCE, thread_start);
    pthread_mutex_lock(&MAP_LOCK);
    while (MBOX.full) pthread_cond_wait(&MAP_COND, &MAP_LOCK);
    MBOX.query_only = 1;
    MBOX.probe = 1;
    MBOX.ts = LAST_ACCEPT_NS + 1;
    /* gravity-consistent orientation (see the header note): the 4-DOF PnP
     * trusts roll/pitch from here; identity on a tilted frame kills it */
    if (grav_q) memcpy(MBOX.q, grav_q, sizeof MBOX.q);
    else { MBOX.q[0] = 1; MBOX.q[1] = MBOX.q[2] = MBOX.q[3] = 0; }
    MBOX.p[0] = MBOX.p[1] = MBOX.p[2] = 0;
    memcpy(MBOX.img, img, sizeof MBOX.img);
    MBOX.n_lm = 0;
    PROBE_RES.done = 0;
    PROBE_RES.ok = 0;
    MBOX.full = 1;
    pthread_cond_broadcast(&MAP_COND);
    while (!PROBE_RES.done) pthread_cond_wait(&MAP_COND, &MAP_LOCK);
    int ok = PROBE_RES.ok;
    if (ok) {
        if (out_q) memcpy(out_q, PROBE_RES.q, sizeof PROBE_RES.q);
        if (out_p) memcpy(out_p, PROBE_RES.p, sizeof PROBE_RES.p);
        if (out_inliers) *out_inliers = PROBE_RES.inliers;
    }
    pthread_mutex_unlock(&MAP_LOCK);
    return ok;
}

int xr_map_probe_burst(const uint8_t *img, const float grav_q[4],
                       const float rel_p[3], int first, int last,
                       float out_q[4], float out_p[3], int *out_inliers) {
    pthread_once(&THREAD_ONCE, thread_start);
    pthread_mutex_lock(&MAP_LOCK);
    while (MBOX.full) pthread_cond_wait(&MAP_COND, &MAP_LOCK);
    MBOX.query_only = 1;
    MBOX.probe = 1;
    MBOX.bfirst = first ? 1 : 0;
    MBOX.blast = last ? 1 : 0;
    if (rel_p) memcpy(MBOX.bp, rel_p, sizeof MBOX.bp);
    else MBOX.bp[0] = MBOX.bp[1] = MBOX.bp[2] = 0;
    MBOX.ts = LAST_ACCEPT_NS + 1;
    if (grav_q) memcpy(MBOX.q, grav_q, sizeof MBOX.q);
    else { MBOX.q[0] = 1; MBOX.q[1] = MBOX.q[2] = MBOX.q[3] = 0; }
    /* the burst frame's odom pose IS its intra-burst VIO position */
    memcpy(MBOX.p, MBOX.bp, sizeof MBOX.p);
    memcpy(MBOX.img, img, sizeof MBOX.img);
    MBOX.n_lm = 0;
    PROBE_RES.done = 0;
    PROBE_RES.ok = 0;
    MBOX.full = 1;
    pthread_cond_broadcast(&MAP_COND);
    while (!PROBE_RES.done) pthread_cond_wait(&MAP_COND, &MAP_LOCK);
    int ok = PROBE_RES.ok;
    if (ok) {
        if (out_q) memcpy(out_q, PROBE_RES.q, sizeof PROBE_RES.q);
        if (out_p) memcpy(out_p, PROBE_RES.p, sizeof PROBE_RES.p);
        if (out_inliers) *out_inliers = PROBE_RES.inliers;
    }
    pthread_mutex_unlock(&MAP_LOCK);
    return ok;
}

static void thread_start(void) {
    pthread_t t;
    pthread_create(&t, NULL, map_thread, NULL);
    pthread_detach(t);
}

void xr_map_offer2(const float q[4], const float p[3], uint64_t ts_ns,
                   const uint8_t *img, const uint8_t *img_right,
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
    if (img_right) {
        memcpy(MBOX.img2, img_right, sizeof MBOX.img2);
        MBOX.have2 = 1;
    } else {
        MBOX.have2 = 0;
    }
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

/* legacy single-image offer (the app path) — no stereo backfill */
void xr_map_offer(const float q[4], const float p[3], uint64_t ts_ns,
                  const uint8_t *img,
                  const int32_t *lm_id, const float (*lm_xyz)[3],
                  const float (*lm_uv)[2], int n_lm) {
    xr_map_offer2(q, p, ts_ns, img, NULL, lm_id, lm_xyz, lm_uv, n_lm);

}
