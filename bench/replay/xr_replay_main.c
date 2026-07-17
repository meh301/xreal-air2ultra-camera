/* xr_replay_main.c — headless dataset replay through the production stack:
 * replay pack (see bench/host/README.md) -> xr_slam (Basalt VIT) -> landmark
 * confirm cache -> xr_map (loop closure / healing), emitting causal per-frame
 * trajectories in TUM format:
 *   <out>_vio.tum   raw Basalt odometry        (the "stock Basalt" baseline)
 *   <out>_map.tum   session pose = D o odom    (ours: VIO + loop closure)
 * One run produces both. Replicates the slam_worker order from xreal_jni.c:
 * push_pair -> poll -> confirm-cache -> xr_map_offer -> get_correction ->
 * compose. Build per dataset resolution: -DXR_OW=<W> -DXR_OH=<H>.
 *
 * Usage: xr_replay --pack <dir> --out <prefix> [--toml <basalt.toml>]
 *        [--inflight N] [--fast] [--no-map] [--xfeat <model.onnx>]
 */
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "xr_map.h"
#include "xr_slam.h"
#include "xr_vpr.h"
#include "xr_xfeat.h"
#include "xreal_core.h"

/* _exit, not exit: an error mid-run must not hang in the same static
 * destructors the success path skips (see the end of main). */
#define DIE(...) do { fprintf(stderr, "xr_replay: " __VA_ARGS__); \
                      fprintf(stderr, "\n"); fflush(NULL); _exit(1); } while (0)

/* ---------------- pack readers ---------------- */

typedef struct { int W, H, n_frames; double fps, imu_hz; } pack_meta;

static void read_meta(const char *dir, pack_meta *m) {
    char path[1024];
    snprintf(path, sizeof path, "%s/meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) DIE("missing %s", path);
    char k[64], v[256];
    memset(m, 0, sizeof *m);
    while (fscanf(f, " %63[^=\n]=%255s", k, v) == 2) {
        if (!strcmp(k, "W")) m->W = atoi(v);
        else if (!strcmp(k, "H")) m->H = atoi(v);
        else if (!strcmp(k, "fps")) m->fps = atof(v);
        else if (!strcmp(k, "imu_hz")) m->imu_hz = atof(v);
        else if (!strcmp(k, "n_frames")) m->n_frames = atoi(v);
    }
    fclose(f);
    if (m->W != XR_OW || m->H != XR_OH)
        DIE("pack is %dx%d but this binary is built for %dx%d "
            "(rebuild with -DXR_OW=%d -DXR_OH=%d)",
            m->W, m->H, XR_OW, XR_OH, m->W, m->H);
    if (m->imu_hz <= 0 || m->fps <= 0 || m->n_frames <= 0)
        DIE("meta.txt missing fps/imu_hz/n_frames");
}

static void read_cam(FILE *f, const char *side, int model,
                     xr_slam_cam_raw *c, const pack_meta *m) {
    char key[64];
    char want[64];
    int ndist = model == XR_SLAM_CAM_RT8 ? 9 : 4;
    snprintf(want, sizeof want, "%s_pinhole", side);
    if (fscanf(f, "%63s %f %f %f %f", key, &c->fx, &c->fy, &c->cx, &c->cy) != 5
        || strcmp(key, want))
        DIE("calib.txt: expected '%s'", want);
    if (fscanf(f, "%63s", key) != 1) DIE("calib.txt: expected %s_dist", side);
    for (int i = 0; i < ndist; i++)
        if (fscanf(f, "%f", &c->k[i]) != 1)
            DIE("calib.txt: %s_dist needs %d values", side, ndist);
    if (fscanf(f, "%63s %f %f %f %f", key, &c->q_xyzw[0], &c->q_xyzw[1],
               &c->q_xyzw[2], &c->q_xyzw[3]) != 5)
        DIE("calib.txt: expected %s_q_xyzw", side);
    if (fscanf(f, "%63s %f %f %f", key, &c->p[0], &c->p[1], &c->p[2]) != 4)
        DIE("calib.txt: expected %s_p", side);
    c->model = model;
    c->width = m->W;
    c->height = m->H;
    c->fps = m->fps;
}

static void read_calib(const char *dir, const pack_meta *m,
                       xr_slam_cam_raw *L, xr_slam_cam_raw *R,
                       float noises[4], int *have_noises) {
    char path[1024];
    snprintf(path, sizeof path, "%s/calib.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) DIE("missing %s", path);
    char key[64], model[64];
    if (fscanf(f, "%63s %63s", key, model) != 2 || strcmp(key, "model"))
        DIE("calib.txt: expected 'model ...'");
    int cm;
    if (!strcmp(model, "kb4")) cm = XR_SLAM_CAM_KB4;
    else if (!strcmp(model, "rt8")) cm = XR_SLAM_CAM_RT8;
    else DIE("calib.txt: unknown model '%s'", model);
    read_cam(f, "left", cm, L, m);
    read_cam(f, "right", cm, R, m);
    *have_noises = fscanf(f, "%63s %f %f %f %f", key, &noises[0], &noises[1],
                          &noises[2], &noises[3]) == 5 &&
                   !strcmp(key, "noises");
    fclose(f);
}

/* ---------------- small quaternion helpers (Hamilton wxyz) ---------------- */

static void quat_mul_wxyz(const float a[4], const float b[4], float o[4]) {
    o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

static void wxyz_to_rot(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
}

static void m3v(const float R[9], const float v[3], float o[3]) {
    o[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    o[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    o[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

/* ---------------- frame ring (pose ts -> left image for xr_map) ---------- */

#define RING_N 64
static struct {
    uint64_t ts[RING_N];
    uint8_t *img[RING_N];   /* left frames, XR_OW*XR_OH each */
    uint8_t *img2[RING_N];  /* right frames (XR_DEPTHFILL) */
    int head;
} RING;

static void ring_put(uint64_t ts, const uint8_t *left) {
    int i = RING.head;
    memcpy(RING.img[i], left, (size_t)XR_OW * XR_OH);
    memcpy(RING.img2[i], left + (size_t)XR_OW * XR_OH,
           (size_t)XR_OW * XR_OH);   /* frames.raw stores L then R */
    RING.ts[i] = ts;
    RING.head = (i + 1) % RING_N;
}

static const uint8_t *ring_get(uint64_t ts) {
    for (int i = 0; i < RING_N; i++)
        if (RING.ts[i] == ts) return RING.img[i];
    return NULL;
}
static const uint8_t *ring_get2(uint64_t ts) {
    for (int i = 0; i < RING_N; i++)
        if (RING.ts[i] == ts) return RING.img2[i];
    return NULL;
}

/* ---------------- per-pose processing ---------------- */

/* landmark confirm cache, ported from slam_worker (xreal_jni.c:1081-1140):
 * a 3D position is only trusted after TWO exports that agree (<0.35 m) —
 * single gate-passing values of later-divergent depths starve RANSAC. */
static struct {
    int32_t id;
    float xyz[3];
    uint64_t ts;
    int hits;
} LMC[1024];

typedef struct {
    FILE *f_vio, *f_map;
    int use_map;
    int geom_wired;
    long n_poses;
    long n_offers;
} pose_ctx;

/* --kidnap: absolute time (s) until which on_pose must NOT clear the
 * SHAKING flag, so the post-blackout offers latch LOST. -1 = inactive. */
static double KD_NOCLEAR_UNTIL = -1;

/* latest VIO pose, for XR_MAPSEED's near-keyframe query (the seed block
 * runs before the frame's own pose exists; one frame stale is fine) */
static float LASTP_Q[4] = { 1, 0, 0, 0 }, LASTP_P[3];

static void tum_line(FILE *f, uint64_t ts_ns, const float q_wxyz[4],
                     const float p[3]) {
    fprintf(f, "%.9f %.6f %.6f %.6f %.7f %.7f %.7f %.7f\n",
            ts_ns / 1e9, p[0], p[1], p[2],
            q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]);
}

static void on_pose(const xr_slam_state *st_in, void *user) {
    pose_ctx *ctx = user;
    xr_slam_state *st = (xr_slam_state *)st_in;   /* scratch, mutable */
    ctx->n_poses++;
    memcpy(LASTP_Q, st->q, sizeof LASTP_Q);
    memcpy(LASTP_P, st->p, sizeof LASTP_P);
    tum_line(ctx->f_vio, st->ts, st->q, st->p);
    if (!ctx->use_map) return;

    if (!ctx->geom_wired) {
        float Ric[9], pic[3];
        if (xr_slam_cam0_geom(Ric, pic) == 0) {
            xr_map_set_geom(xr_slam_unproject0, Ric, pic);
            ctx->geom_wired = 1;
        }
    }

    /* confirm cache update */
    for (int i = 0; i < st->n_landmarks; i++) {
        uint32_t h = ((uint32_t)st->lm_id[i] * 2654435761u) & 1023u;
        int slot = (int)h, oldest = (int)h;
        for (int k = 0; k < 16; k++) {
            uint32_t j = (h + k) & 1023u;
            if (!LMC[j].ts || LMC[j].id == st->lm_id[i]) { slot = (int)j; break; }
            if (LMC[j].ts < LMC[oldest].ts) oldest = (int)j;
            slot = oldest;
        }
        if (LMC[slot].ts && LMC[slot].id == st->lm_id[i]) {
            float dx = LMC[slot].xyz[0] - st->lm_xyz[i][0];
            float dy = LMC[slot].xyz[1] - st->lm_xyz[i][1];
            float dz = LMC[slot].xyz[2] - st->lm_xyz[i][2];
            if (dx * dx + dy * dy + dz * dz < 0.35f * 0.35f) {
                if (LMC[slot].hits < 100) LMC[slot].hits++;
            } else {
                LMC[slot].hits = 1;
            }
        } else {
            LMC[slot].id = st->lm_id[i];
            LMC[slot].hits = 1;
        }
        memcpy(LMC[slot].xyz, st->lm_xyz[i], sizeof LMC[slot].xyz);
        LMC[slot].ts = st->ts;
    }
    static int32_t off_id[XR_SLAM_MAX_FEATURES];
    static float off_xyz[XR_SLAM_MAX_FEATURES][3];
    static float off_uv[XR_SLAM_MAX_FEATURES][2];
    int off_n = 0;
    for (int i = 0; i < st->n_features; i++) {
        uint32_t h = ((uint32_t)st->feat_id[i] * 2654435761u) & 1023u;
        for (int k = 0; k < 16; k++) {
            uint32_t j = (h + k) & 1023u;
            if (!LMC[j].ts) break;
            if (LMC[j].id == st->feat_id[i]) {
                if (LMC[j].hits >= 2 && st->ts - LMC[j].ts < 2000000000ull) {
                    off_id[off_n] = st->feat_id[i];
                    memcpy(off_xyz[off_n], LMC[j].xyz, sizeof off_xyz[0]);
                    off_uv[off_n][0] = st->feat_uv[i][0];
                    off_uv[off_n][1] = st->feat_uv[i][1];
                    off_n++;
                }
                break;
            }
        }
    }

    const uint8_t *img = ring_get(st->ts);
    if (img) {
        /* kidnap grace: keep SHAKING visible to the first post-blackout
         * offers so the map thread actually latches LOST (this clear
         * otherwise races the latch away before any offer processes) */
        if ((double)st->ts * 1e-9 >= KD_NOCLEAR_UNTIL)
            xr_map_freeze_storage(0);
        xr_map_offer2(st->q, st->p, st->ts, img, ring_get2(st->ts),
                      off_id, off_xyz, off_uv, off_n);
        ctx->n_offers++;
    }

    /* session correction (identity until the first verified closure) */
    float cq[4], cp[3], CR[9], t3[3], qs[4];
    xr_map_get_correction(cq, cp);
    wxyz_to_rot(cq, CR);
    quat_mul_wxyz(cq, st->q, qs);
    m3v(CR, st->p, t3);
    float p2[3] = { t3[0] + cp[0], t3[1] + cp[1], t3[2] + cp[2] };
    tum_line(ctx->f_map, st->ts, qs, p2);
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    const char *pack = NULL, *out = NULL, *toml = NULL, *xfeat = NULL;
    const char *vpr = NULL, *lglue = NULL;
    int inflight = 6, fast = 0, use_map = 1, reloc_n = 0, reloc_clip = 1;
    double kd_t0 = -1, kd_dur = 0;
    int seed_frontend = 0;   /* set after args: XR_SEED env + xfeat model */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--toml") && i + 1 < argc) toml = argv[++i];
        else if (!strcmp(argv[i], "--xfeat") && i + 1 < argc) xfeat = argv[++i];
        else if (!strcmp(argv[i], "--vpr") && i + 1 < argc) vpr = argv[++i];
        else if (!strcmp(argv[i], "--lglue") && i + 1 < argc) lglue = argv[++i];
        else if (!strcmp(argv[i], "--inflight") && i + 1 < argc) inflight = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fast")) fast = 1;
        else if (!strcmp(argv[i], "--no-map")) use_map = 0;
        else if (!strcmp(argv[i], "--reloc") && i + 1 < argc) reloc_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reloc-clip") && i + 1 < argc) reloc_clip = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kidnap") && i + 1 < argc) {
            /* t0,dur (s): drop ALL camera frames in [t0,t0+dur) while IMU
             * continues, with SHAKING held through the window + a short
             * grace — the synthetic twin of the field failure (pocketed
             * device / violent shake): VIO coasts blind on IMU, comes back
             * with a drifted frame, the map must latch LOST, then either
             * reloc-recover or open a submap and weld later. */
            if (sscanf(argv[++i], "%lf,%lf", &kd_t0, &kd_dur) != 2)
                DIE("--kidnap wants t0,dur seconds");
        }
        else DIE("unknown arg '%s'", argv[i]);
    }
    if (!pack || !out) DIE("usage: xr_replay --pack <dir> --out <prefix> "
                           "[--toml f] [--inflight N] [--fast] [--no-map] "
                           "[--xfeat model.onnx] [--vpr model.onnx]");

    pack_meta meta;
    read_meta(pack, &meta);
    xr_slam_cam_raw L, R;
    float noises[4];
    int have_noises = 0;
    read_calib(pack, &meta, &L, &R, noises, &have_noises);

    char path[1024];
    snprintf(path, sizeof path, "%s/imu.bin", pack);
    FILE *f_imu = fopen(path, "rb");
    if (!f_imu) DIE("missing %s", path);
    snprintf(path, sizeof path, "%s/frames.csv", pack);
    FILE *f_idx = fopen(path, "r");
    if (!f_idx) DIE("missing %s", path);
    snprintf(path, sizeof path, "%s/frames.raw", pack);
    FILE *f_raw = fopen(path, "rb");
    if (!f_raw) DIE("missing %s", path);

    for (int i = 0; i < RING_N; i++) {
        RING.img[i] = malloc((size_t)XR_OW * XR_OH);
        RING.img2[i] = malloc((size_t)XR_OW * XR_OH);
        if (!RING.img[i] || !RING.img2[i]) DIE("oom");
    }
    /* XR_DEPTHFILL: register rectified stereo geometry when the pack is
     * distortion-free (drives). fx + |left_p - right_p| from calib.txt. */
    {
        char cpath[1024];
        snprintf(cpath, sizeof cpath, "%s/calib.txt", pack);
        FILE *cf = fopen(cpath, "r");
        if (cf) {
            char line[256];
            double fx = 0, lp[3] = {0}, rp[3] = {0}, dmax = 0;
            int have_l = 0, have_r = 0;
            while (fgets(line, sizeof line, cf)) {
                double a, b, c, d;
                if (sscanf(line, "left_pinhole %lf %lf %lf %lf",
                           &a, &b, &c, &d) == 4) fx = a;
                else if (sscanf(line, "left_dist %lf %lf %lf %lf",
                                &a, &b, &c, &d) == 4) {
                    if (fabs(a) > dmax) dmax = fabs(a);
                    if (fabs(b) > dmax) dmax = fabs(b);
                    if (fabs(c) > dmax) dmax = fabs(c);
                    if (fabs(d) > dmax) dmax = fabs(d);
                } else if (sscanf(line, "left_p %lf %lf %lf",
                                  &a, &b, &c) == 3) {
                    lp[0] = a; lp[1] = b; lp[2] = c; have_l = 1;
                } else if (sscanf(line, "right_p %lf %lf %lf",
                                  &a, &b, &c) == 3) {
                    rp[0] = a; rp[1] = b; rp[2] = c; have_r = 1;
                }
            }
            fclose(cf);
            if (fx > 1 && have_l && have_r && dmax < 1e-9) {
                double bl = sqrt((lp[0]-rp[0])*(lp[0]-rp[0]) +
                                 (lp[1]-rp[1])*(lp[1]-rp[1]) +
                                 (lp[2]-rp[2])*(lp[2]-rp[2]));
                xr_map_set_stereo((float)fx, (float)bl);
            }
        }
    }
    uint8_t *fr = malloc((size_t)2 * XR_OW * XR_OH);
    if (!fr) DIE("oom");

    pose_ctx ctx = { 0 };
    ctx.use_map = use_map;
    char fname[1100];
    snprintf(fname, sizeof fname, "%s_vio.tum", out);
    ctx.f_vio = fopen(fname, "w");
    snprintf(fname, sizeof fname, "%s_map.tum", out);
    ctx.f_map = fopen(fname, "w");
    if (!ctx.f_vio || !ctx.f_map) DIE("cannot open outputs '%s_*.tum'", out);

    if (toml) xr_slam_set_config(toml);
    if (use_map) {
        xr_map_set_mapping(1);
        xr_map_set_recovery(1);
        if (xfeat) {
            xr_map_set_model(xfeat);
            xr_map_set_use_xfeat(1);
        }
        if (vpr) xr_map_set_vpr_model(vpr);
        if (lglue) xr_map_set_lglue_model(lglue);
    }
    /* v9 detector unification: XFeat maxima seed Basalt corner detection
     * (needs --xfeat for the model AND XR_SEED=1; extension is dlsym'd so
     * stock basalt libs simply ignore the seeds) */
    {
        const char *se = getenv("XR_SEED");
        if (se && *se && *se != '0' && xfeat) {
            seed_frontend = 1;
            fprintf(stderr, "xr_replay: XFeat frontend seeding ON\n");
        }
    }
    if (xr_slam_start_raw(&L, &R, meta.imu_hz,
                          have_noises ? noises : NULL) != 0)
        DIE("Basalt failed to start (libbasalt.so on LD_LIBRARY_PATH?)");

    /* interleaved feed in timestamp order */
    struct { int64_t ts; float g[3], a[3]; } imu;
    int have_imu = fread(&imu, 32, 1, f_imu) == 1;
    xr_slam_state scratch;
    long frames_pushed = 0, frames_skipped = 0, imu_pushed = 0;
    uint64_t f_ts;
    long f_i;
    long line = 0;
    while (fscanf(f_idx, "%" SCNu64 ",%ld\n", &f_ts, &f_i) == 2) {
        while (have_imu && (uint64_t)imu.ts <= f_ts) {
            xr_slam_push_imu((uint64_t)imu.ts, imu.g, imu.a);
            imu_pushed++;
            have_imu = fread(&imu, 32, 1, f_imu) == 1;
        }
        if (fread(fr, (size_t)2 * XR_OW * XR_OH, 1, f_raw) != 1)
            DIE("frames.raw truncated at pair %ld", line);
        /* --kidnap blackout window (see the flag parse) */
        static uint64_t kd_ts0;
        static int kd_was, kd_dropped;
        if (!kd_ts0) kd_ts0 = f_ts;
        double kd_rel = (double)(f_ts - kd_ts0) * 1e-9;
        int kd_in = kd_t0 >= 0 && kd_rel >= kd_t0 && kd_rel < kd_t0 + kd_dur;
        if (kd_in && !kd_was) {
            xr_map_freeze_storage(1);
            KD_NOCLEAR_UNTIL = (double)f_ts * 1e-9 + kd_dur + 0.7;
            printf("KIDNAP begin rel=%.1fs (blackout %.1fs)\n", kd_rel, kd_dur);
        }
        if (!kd_in && kd_was)
            printf("KIDNAP end rel=%.1fs dropped=%d\n", kd_rel, kd_dropped);
        kd_was = kd_in;
        if (kd_in) {
            /* blank the pair, do NOT drop it: a pocketed camera sees
             * darkness, and Basalt's bounded IMU queue DEADLOCKS if frames
             * stop while IMU flows (first kidnap run hung here). Black
             * frames kill every KLT track — vision gone, estimator coasts
             * on IMU — while the pipeline keeps draining. */
            memset(fr, 8, (size_t)2 * XR_OW * XR_OH);
            kd_dropped++;
        }
        if (imu_pushed >= 20) {   /* mirrors xr_slam_start_raw's small gate */
            ring_put(f_ts, fr);
            /* v9 detector unification (XR_SEED env): XFeat maxima seed
             * Basalt's corner detection; KLT still does the tracking. Must
             * be posted BEFORE the frame so addPoints finds them. */
            if (seed_frontend) {
                static float suv[512][2];
                static int8_t sdesc[512][64];
                int sn = xr_xfeat_extract(fr, suv, sdesc, 512);
                if (sn < 0) sn = 0;
                /* XR_MAPSEED: append the nearest mapped keyframe's
                 * landmark uvs — KLT re-acquires the map's own corners */
                sn += xr_map_get_reseed(LASTP_Q, LASTP_P, &suv[sn],
                                        512 - sn);
                if (sn > 0)
                    xr_slam_seed_keypoints(f_ts, 0, &suv[0][0], sn);
            }
            xr_slam_push_pair(fr, fr + (size_t)XR_OW * XR_OH, f_ts);
            frames_pushed++;
        } else {
            frames_skipped++;   /* IMU warmup window */
        }
        xr_slam_poll_each(on_pose, &ctx, &scratch);
        if (!fast) {           /* observed backpressure: don't outrun VIO */
            int spins = 0;
            while (frames_pushed - ctx.n_poses > inflight && spins++ < 2500) {
                usleep(2000);
                xr_slam_poll_each(on_pose, &ctx, &scratch);
            }
        }
        line++;
    }
    while (have_imu) {          /* tail IMU */
        xr_slam_push_imu((uint64_t)imu.ts, imu.g, imu.a);
        have_imu = fread(&imu, 32, 1, f_imu) == 1;
    }
    long quiet = 0, last = ctx.n_poses;
    while (quiet < 20) {        /* drain until 2 s with no new poses */
        usleep(100000);
        xr_slam_poll_each(on_pose, &ctx, &scratch);
        quiet = ctx.n_poses == last ? quiet + 1 : 0;
        last = ctx.n_poses;
    }
    xr_slam_stop();

    int kf = use_map ? xr_map_num_keyframes() : 0;
    int loops = 0, matches = 0;
    float lp[3] = { 0 };
    if (use_map) xr_map_loop_stats(&loops, lp, &matches);
    int vpairs = 0, vinl = 0;
    if (use_map) xr_map_verify_stats(&vpairs, &vinl);
    fprintf(stderr,
            "xr_replay done: vpr_ep=%s pairs=%ld pushed=%ld warmup_skip=%ld imu=%ld "
            "poses=%ld offers=%ld completion=%.1f%% kf=%d loops=%d "
            "last_verify=%d/%d\n",
            vpr ? (xr_vpr_on_cuda() ? "cuda" : "cpu") : "off",
            line, frames_pushed, frames_skipped, imu_pushed, ctx.n_poses,
            ctx.n_offers, 100.0 * ctx.n_poses / (line ? line : 1), kf, loops,
            vinl, vpairs);
    fclose(ctx.f_vio);
    fclose(ctx.f_map);

    /* ---- relocalization benchmark (--reloc N): probe N random frames of
     * this sequence against the finished map. Success = verified PnP
     * (recall); error = probe's session pose vs the run's own map-track
     * pose at that frame (self-consistent, GT-free: measures re-entry
     * precision against the map the user would actually resume into). */
    if (reloc_n > 0 && use_map) {
        char mpath[1024];
        snprintf(mpath, sizeof mpath, "%s_map.tum", out);
        FILE *fm = fopen(mpath, "r");
        static double mts[120000];
        static float mp[120000][3];
        long mn = 0;
        if (fm) {
            double t, x, y, z, qx, qy, qz, qw;
            while (mn < 120000 &&
                   fscanf(fm, "%lf %lf %lf %lf %lf %lf %lf %lf",
                          &t, &x, &y, &z, &qx, &qy, &qz, &qw) == 8) {
                mts[mn] = t;
                mp[mn][0] = (float)x; mp[mn][1] = (float)y; mp[mn][2] = (float)z;
                mn++;
            }
            fclose(fm);
        }
        /* the VIO track's odom orientations: the probe's gravity prior.
         * The map's PnP verifier is 4-DOF (yaw+translation, roll/pitch
         * trusted from the IMU) — a real kidnapped device still knows
         * gravity from its accelerometer, so each probe carries the odom
         * orientation of its frame. Yaw is solved, never trusted. */
        snprintf(mpath, sizeof mpath, "%s_vio.tum", out);
        FILE *fv = fopen(mpath, "r");
        static double vts[120000];
        static float vq[120000][4];        /* Hamilton wxyz */
        static float vp[120000][3];        /* odom position (burst rel_p) */
        long vn = 0;
        if (fv) {
            double t, x, y, z, qx, qy, qz, qw;
            while (vn < 120000 &&
                   fscanf(fv, "%lf %lf %lf %lf %lf %lf %lf %lf",
                          &t, &x, &y, &z, &qx, &qy, &qz, &qw) == 8) {
                vts[vn] = t;
                vq[vn][0] = (float)qw; vq[vn][1] = (float)qx;
                vq[vn][2] = (float)qy; vq[vn][3] = (float)qz;
                vp[vn][0] = (float)x; vp[vn][1] = (float)y;
                vp[vn][2] = (float)z;
                vn++;
            }
            fclose(fv);
        }
        /* XR_BURSTPNP: fuse the clip into one joint solve (needs the vio
         * track for intra-burst relatives) */
        int use_burst = 0;
        { const char *e = getenv("XR_BURSTPNP");
          use_burst = (e && *e && *e != '0') ? 1 : 0; }
        rewind(f_idx);
        static uint64_t fts[120000];
        long fn = 0;
        char lbuf[256];
        while (fn < 120000 && fgets(lbuf, sizeof lbuf, f_idx))
            if (sscanf(lbuf, "%llu", (unsigned long long *)&fts[fn]) == 1) fn++;
        uint8_t *pimg = malloc((size_t)XR_OW * XR_OH);
        float errs[4096];
        int ok_n = 0, err_n = 0;
        if (reloc_clip < 1) reloc_clip = 1;
        if (fn < 1) {
            fprintf(stderr, "xr_replay: no frames indexed — skipping --reloc\n");
            reloc_n = 0;
        } else if (reloc_clip > (int)fn) {
            fprintf(stderr, "xr_replay: --reloc-clip %d > %ld frames; clamped\n",
                    reloc_clip, fn);
            reloc_clip = (int)fn;
        }
        srand(12345);   /* seeded: reproducible probe set */
        for (int k = 0; k < reloc_n && k < 4096; k++) {
            long fi0 = (long)((double)rand() / RAND_MAX *
                              (double)(fn - reloc_clip));
            /* --reloc-clip K: a probe is K CONSECUTIVE frames (a waking
             * device sees a stream, not a snapshot); the clip verifies if
             * ANY frame does, and the best-inlier frame's pose lands.
             * fi/pq/pp/inl below carry the winner. K=1 = the old probe. */
            float pq[4], pp[3];
            int inl = 0, ok = 0;
            long fi = fi0;
            for (int c = 0; c < reloc_clip; c++) {
                long fc = fi0 + c;
                fseeko(f_raw, (off_t)fc * XR_OW * XR_OH * 2, SEEK_SET);
                if (fread(pimg, 1, (size_t)XR_OW * XR_OH, f_raw) !=
                    (size_t)XR_OW * XR_OH) continue;
                float cq[4], cp[3];
                int cinl = 0;
                const float *gq = NULL;
                long bv = -1;
                if (vn) {                  /* nearest odom quat = gravity */
                    double t = (double)fts[fc] * 1e-9;
                    bv = 0;
                    for (long j = 1; j < vn; j++)
                        if (fabs(vts[j] - t) < fabs(vts[bv] - t)) bv = j;
                    gq = vq[bv];
                }
                int cok;
                static long bv0;           /* clip start's vio index */
                if (use_burst && reloc_clip > 1 && bv >= 0) {
                    if (c == 0) bv0 = bv;
                    float rp[3] = { vp[bv][0] - vp[bv0][0],
                                    vp[bv][1] - vp[bv0][1],
                                    vp[bv][2] - vp[bv0][2] };
                    cok = xr_map_probe_burst(pimg, gq, rp, c == 0,
                                             c == reloc_clip - 1,
                                             cq, cp, &cinl);
                } else {
                    cok = xr_map_probe(pimg, gq, cq, cp, &cinl);
                }
                if (cok && cinl > inl) {
                    ok = 1;
                    inl = cinl;
                    fi = fc;
                    memcpy(pq, cq, sizeof cq);
                    memcpy(pp, cp, sizeof cp);
                }
            }
            float err = -1.f;
            if (ok && mn) {
                double t = (double)fts[fi] * 1e-9;
                long best = 0;
                for (long j = 1; j < mn; j++)
                    if (fabs(mts[j] - t) < fabs(mts[best] - t)) best = j;
                float dx = pp[0] - mp[best][0], dy = pp[1] - mp[best][1],
                      dz = pp[2] - mp[best][2];
                err = sqrtf(dx * dx + dy * dy + dz * dz);
                errs[err_n++] = err;
                ok_n++;
            }
            /* expected = the run's own session pose at that frame;
             * landed = the probe's verified session pose (if any) —
             * both emitted so the site can plot expected->landed */
            {
                double t = (double)fts[fi] * 1e-9;
                long best = 0;
                for (long j = 1; j < mn; j++)
                    if (fabs(mts[j] - t) < fabs(mts[best] - t)) best = j;
                printf("RELOC k=%d frame=%ld ok=%d inl=%d err_m=%.3f "
                       "exp=%.3f,%.3f,%.3f got=%.3f,%.3f,%.3f\n",
                       k, fi, ok, inl, (double)err,
                       mn ? (double)mp[best][0] : 0.0,
                       mn ? (double)mp[best][1] : 0.0,
                       mn ? (double)mp[best][2] : 0.0,
                       ok ? (double)pp[0] : 0.0, ok ? (double)pp[1] : 0.0,
                       ok ? (double)pp[2] : 0.0);
            }
        }
        /* summary: recall + median error + recall@0.25m/@0.10m */
        int u25 = 0, u10 = 0;
        for (int i = 0; i < err_n; i++) {
            if (errs[i] < 0.25f) u25++;
            if (errs[i] < 0.10f) u10++;
        }
        for (int i = 0; i < err_n; i++)          /* insertion sort (small) */
            for (int j = i; j > 0 && errs[j] < errs[j - 1]; j--) {
                float tswap = errs[j]; errs[j] = errs[j - 1]; errs[j - 1] = tswap;
            }
        printf("RELOC-SUMMARY n=%d verified=%d recall=%.3f "
               "r@25cm=%.3f r@10cm=%.3f med_err_m=%.3f clip=%d\n",
               reloc_n, ok_n, (double)ok_n / reloc_n,
               (double)u25 / reloc_n, (double)u10 / reloc_n,
               err_n ? (double)errs[err_n / 2] : -1.0, reloc_clip);
        free(pimg);
    }

    fclose(f_imu);
    fclose(f_idx);
    fclose(f_raw);
    /* Every output is closed by here. Skip the C++ static destructors on
     * the way out: ORT's CUDA EP teardown and basalt/TBB worker threads
     * can deadlock in __run_exit_handlers, leaving finished runs hanging
     * at 0% CPU (the fleet's pkill-discipline pain, and the reason
     * `timeout` wrappers were load-bearing). Nothing after main() needs
     * to run — flush and leave. */
    fflush(NULL);
    _exit(0);
}
