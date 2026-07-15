/* xr_slam.c — see xr_slam.h. */
#include "xr_slam.h"

#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <android/log.h>

#include "vit_interface.h"
#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static struct {
    void *dl;
    PFN_vit_api_get_version get_version;
    PFN_vit_tracker_create create;
    PFN_vit_tracker_destroy destroy;
    PFN_vit_tracker_enable_extension enable_ext;
    PFN_vit_tracker_start start;
    PFN_vit_tracker_stop stop;
    PFN_vit_tracker_reset reset;
    PFN_vit_tracker_push_imu_sample push_imu;
    PFN_vit_tracker_push_img_sample push_img;
    PFN_vit_tracker_add_imu_calibration add_imu_calib;
    PFN_vit_tracker_add_camera_calibration add_cam_calib;
    PFN_vit_tracker_pop_pose pop_pose;
    PFN_vit_pose_destroy pose_destroy;
    PFN_vit_pose_get_data pose_data;
    PFN_vit_pose_get_features pose_features;

    vit_tracker_t *tracker;
    atomic_int running;
    atomic_int imu_pushed;  /* samples since start: frames wait for history */
    float kb4[2][8];        /* per cam: fx fy cx cy k1..k4 (for unproject) */
    float R_ic[2][9];       /* camera -> IMU rotation, row-major */
    float p_ic[2][3];       /* camera position in the IMU frame */
    uint64_t last_pair_ts;
    char config_path[512];
} B;

void xr_slam_set_config(const char *unified_config_path) {
    if (!unified_config_path) {
        B.config_path[0] = 0;
        return;
    }
    strncpy(B.config_path, unified_config_path, sizeof B.config_path - 1);
    B.config_path[sizeof B.config_path - 1] = 0;
}

/* the VIO initializes its world from the first accel samples: give it some
 * IMU history before the first frame arrives */
#define IMU_WARMUP_SAMPLES 300

/* Basalt reports through std::cout/cerr, which Android swallows. Redirect
 * both into logcat (tag "basalt") so config echoes, queue warnings and
 * assertion messages are visible. */
static void *stdio_pump(void *arg) {
    int fd = (int)(intptr_t)arg;
    char buf[512];
    for (;;) {
        ssize_t n = read(fd, buf, sizeof buf - 1);
        if (n <= 0) break;
        buf[n] = 0;
        __android_log_write(ANDROID_LOG_INFO, "basalt", buf);
    }
    return NULL;
}

static void redirect_stdio(void) {
    static int done;
    if (done) return;
    done = 1;
    int p[2];
    if (pipe(p) != 0) return;
    dup2(p[1], 1);
    dup2(p[1], 2);
    close(p[1]);
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    pthread_t t;
    pthread_create(&t, NULL, stdio_pump, (void *)(intptr_t)p[0]);
    pthread_detach(t);
}

int xr_slam_load(void) {
    static int attempted;
    if (B.dl) return 1;
    if (attempted) return 0;
    attempted = 1;
    redirect_stdio();
    B.dl = dlopen("libbasalt.so", RTLD_NOW | RTLD_LOCAL);
    if (!B.dl) {
        LOGI("Basalt backend not present (%s) — using built-in tracker", dlerror());
        return 0;
    }
#define SYM(field, name) \
    do { \
        *(void **)&B.field = dlsym(B.dl, name); \
        if (!B.field) { \
            LOGE("Basalt: missing symbol %s", name); \
            dlclose(B.dl); \
            B.dl = NULL; \
            return 0; \
        } \
    } while (0)
    SYM(get_version, "vit_api_get_version");
    SYM(create, "vit_tracker_create");
    SYM(destroy, "vit_tracker_destroy");
    SYM(enable_ext, "vit_tracker_enable_extension");
    SYM(start, "vit_tracker_start");
    SYM(stop, "vit_tracker_stop");
    SYM(reset, "vit_tracker_reset");
    SYM(push_imu, "vit_tracker_push_imu_sample");
    SYM(push_img, "vit_tracker_push_img_sample");
    SYM(add_imu_calib, "vit_tracker_add_imu_calibration");
    SYM(add_cam_calib, "vit_tracker_add_camera_calibration");
    SYM(pop_pose, "vit_tracker_pop_pose");
    SYM(pose_destroy, "vit_pose_destroy");
    SYM(pose_data, "vit_pose_get_data");
    SYM(pose_features, "vit_pose_get_features");
#undef SYM
    uint32_t maj = 0, min = 0, pat = 0;
    B.get_version(&maj, &min, &pat);
    LOGI("Basalt VIT backend loaded, interface %u.%u.%u", maj, min, pat);
    return 1;
}

/* fit the kb4 radial polynomial r/f = theta + k1 th^3 + k2 th^5 + k3 th^7
 * + k4 th^9 to the fisheye624 theta-polynomial by linear least squares over
 * the usable field of view (tangential/thin-prism terms are ~0 and dropped) */
static void fit_kb4(const xr_eye_calib *c, float out[8]) {
    double AtA[16] = { 0 }, Atb[4] = { 0 };
    const int N = 64;
    const double theta_max = 1.30;             /* ~74 deg half-angle */
    for (int i = 1; i <= N; i++) {
        double th = theta_max * i / N;
        double t2 = th * th;
        /* fisheye624 radial: th * (1 + kc0 t2 + ... + kc5 t2^6) */
        double poly = 1.0 + t2 * (c->kc[0] + t2 * (c->kc[1] + t2 * (c->kc[2] +
                      t2 * (c->kc[3] + t2 * (c->kc[4] + t2 * c->kc[5])))));
        double r = th * poly;
        double basis[4] = { th * t2, th * t2 * t2, th * t2 * t2 * t2,
                            th * t2 * t2 * t2 * t2 };
        double resid = r - th;
        for (int a = 0; a < 4; a++) {
            Atb[a] += basis[a] * resid;
            for (int b = 0; b < 4; b++) AtA[a * 4 + b] += basis[a] * basis[b];
        }
    }
    /* solve the 4x4 normal equations (Gauss elimination, well-conditioned) */
    double k[4] = { 0 };
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(AtA[r * 4 + col]) > fabs(AtA[piv * 4 + col])) piv = r;
        for (int cc2 = 0; cc2 < 4; cc2++) {
            double t = AtA[col * 4 + cc2];
            AtA[col * 4 + cc2] = AtA[piv * 4 + cc2];
            AtA[piv * 4 + cc2] = t;
        }
        double tb = Atb[col]; Atb[col] = Atb[piv]; Atb[piv] = tb;
        for (int r = col + 1; r < 4; r++) {
            double f = AtA[r * 4 + col] / AtA[col * 4 + col];
            for (int cc2 = col; cc2 < 4; cc2++) AtA[r * 4 + cc2] -= f * AtA[col * 4 + cc2];
            Atb[r] -= f * Atb[col];
        }
    }
    for (int r = 3; r >= 0; r--) {
        double s = Atb[r];
        for (int cc2 = r + 1; cc2 < 4; cc2++) s -= AtA[r * 4 + cc2] * k[cc2];
        k[r] = s / AtA[r * 4 + r];
    }
    out[0] = c->fc[0]; out[1] = c->fc[1];
    out[2] = c->cc[0]; out[3] = c->cc[1];
    for (int i = 0; i < 4; i++) out[4 + i] = (float)k[i];
}

/* Hamilton rotation matrix from xyzw with optional conjugation — must match
 * quat_to_rot in xreal_align.c (R maps camera -> IMU for variant 2). */
static void quat_to_rot_xyzw(const float q[4], int conj, float R[9]) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    float n = sqrtf(x * x + y * y + z * z + w * w);
    x /= n; y /= n; z /= n; w /= n;
    if (conj) { x = -x; y = -y; z = -z; }
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
}

static void cam_calibration(const xr_eye_calib *c, int variant, uint32_t index,
                            vit_camera_calibration_t *out) {
    memset(out, 0, sizeof *out);
    out->camera_index = index;
    out->width = XR_OW;
    out->height = XR_OH;
    out->frequency = 30.0;
    fit_kb4(c, B.kb4[index]);
    out->fx = B.kb4[index][0];
    out->fy = B.kb4[index][1];
    out->cx = B.kb4[index][2];
    out->cy = B.kb4[index][3];
    out->model = VIT_CAMERA_DISTORTION_KB4;
    out->distortion_count = 4;
    for (int i = 0; i < 4; i++) out->distortion[i] = B.kb4[index][4 + i];

    /* T_imu_cam: R maps camera -> IMU (xr_align_project uses its transpose
     * to go IMU -> camera), translation is the camera position in the IMU
     * frame straight from imu_p_cam */
    float R[9];
    quat_to_rot_xyzw(c->q_cam, (variant >> 1) & 1, R);
    memcpy(B.R_ic[index], R, sizeof R);
    memcpy(B.p_ic[index], c->p_cam, sizeof B.p_ic[index]);
    const double t[3] = { c->p_cam[0], c->p_cam[1], c->p_cam[2] };
    for (int r = 0; r < 3; r++) {
        for (int cc2 = 0; cc2 < 3; cc2++) out->transform[r * 4 + cc2] = R[r * 3 + cc2];
        out->transform[r * 4 + 3] = t[r];
    }
    out->transform[15] = 1.0;
}

int xr_slam_start(const xr_eye_calib *left, const xr_eye_calib *right,
                  int variant, const float gyro_bias[3],
                  const float accel_bias[3], const float noises[4]) {
    if (!xr_slam_load()) return -1;
    if (B.tracker) return 0;

    vit_config_t cfg = { .file = B.config_path[0] ? B.config_path : NULL,
                         .cam_count = 2, .imu_count = 1, .show_ui = false };
    if (cfg.file) LOGI("Basalt: using config %s", cfg.file);
    if (B.create(&cfg, &B.tracker) != VIT_SUCCESS || !B.tracker) {
        LOGE("Basalt: tracker create failed");
        B.tracker = NULL;
        return -1;
    }
    B.enable_ext(B.tracker, VIT_TRACKER_EXTENSION_POSE_FEATURES, true);

    /* cameras: VIT index 0 = left */
    vit_camera_calibration_t cc;
    cam_calibration(left, variant, 0, &cc);
    B.add_cam_calib(B.tracker, &cc);
    cam_calibration(right, variant, 1, &cc);
    B.add_cam_calib(B.tracker, &cc);
    LOGI("Basalt: kb4 left fx=%.1f k=[%.4f %.4f %.4f %.4f]",
         B.kb4[0][0], B.kb4[0][4], B.kb4[0][5], B.kb4[0][6], B.kb4[0][7]);

    /* IMU. The raw stream is UNCORRECTED (the AHRS does its own bias
     * capture for the same reason), so the factory biases must go in as
     * offsets — a 0.9 deg/s gyro bias fed as zero makes the estimator
     * fight the cameras and diverge. offset ADDs to raw: offset = -bias. */
    vit_imu_calibration_t ic;
    memset(&ic, 0, sizeof ic);
    ic.imu_index = 0;
    ic.frequency = 1000.0;
    /* {gyro_noise, gyro_bias_rw, accel_noise, accel_bias_rw} */
    float nz[4] = { 0.00035f, 0.0001f, 0.00667f, 0.001f };  /* defaults */
    if (noises) memcpy(nz, noises, sizeof nz);
    for (int i = 0; i < 3; i++) {
        ic.accel.transform[i * 3 + i] = 1.0;
        ic.gyro.transform[i * 3 + i] = 1.0;
        if (gyro_bias) ic.gyro.offset[i] = -(double)gyro_bias[i];
        if (accel_bias) ic.accel.offset[i] = -(double)accel_bias[i];
        ic.gyro.noise_std[i] = nz[0];       /* rad/s  / sqrt(Hz) */
        ic.gyro.bias_std[i] = nz[1];
        ic.accel.noise_std[i] = nz[2];      /* m/s^2 / sqrt(Hz) */
        ic.accel.bias_std[i] = nz[3];
    }
    B.add_imu_calib(B.tracker, &ic);
    LOGI("Basalt: gyro bias [%.4f %.4f %.4f] rad/s, noises [%g %g %g %g]",
         gyro_bias ? gyro_bias[0] : 0.0f, gyro_bias ? gyro_bias[1] : 0.0f,
         gyro_bias ? gyro_bias[2] : 0.0f,
         (double)nz[0], (double)nz[1], (double)nz[2], (double)nz[3]);

    if (B.start(B.tracker) != VIT_SUCCESS) {
        LOGE("Basalt: tracker start failed");
        B.destroy(B.tracker);
        B.tracker = NULL;
        return -1;
    }
    atomic_store(&B.imu_pushed, 0);
    B.last_pair_ts = 0;
    atomic_store(&B.running, 1);
    LOGI("Basalt VIO started (2 cams, 1 kHz IMU) — hold still for init");
    return 0;
}

void xr_slam_stop(void) {
    if (!B.tracker) return;
    atomic_store(&B.running, 0);
    B.stop(B.tracker);
    B.destroy(B.tracker);
    B.tracker = NULL;
}

void xr_slam_reset(void) {
    if (B.tracker && atomic_load(&B.running)) B.reset(B.tracker);
}

int xr_slam_running(void) {
    return atomic_load(&B.running);
}

static void kb4_unproject(const float k[8], float u, float v, float ray[3]);

int xr_slam_unproject0(float u, float v, float ray_cam[3]) {
    if (B.kb4[0][0] == 0) return -1;           /* not configured yet */
    kb4_unproject(B.kb4[0], u, v, ray_cam);
    return 0;
}

int xr_slam_cam0_geom(float R_ic[9], float p_ic[3]) {
    if (B.kb4[0][0] == 0) return -1;
    memcpy(R_ic, B.R_ic[0], 9 * sizeof(float));
    memcpy(p_ic, B.p_ic[0], 3 * sizeof(float));
    return 0;
}

void xr_slam_push_imu(uint64_t ts_ns, const float gyro_dps[3],
                      const float accel_g[3]) {
    if (!atomic_load(&B.running)) return;
    const float D2R = (float)M_PI / 180.0f;
    const float G = 9.80665f;
    vit_imu_sample_t s = {
        .timestamp = (int64_t)ts_ns,
        .ax = accel_g[0] * G, .ay = accel_g[1] * G, .az = accel_g[2] * G,
        .wx = gyro_dps[0] * D2R, .wy = gyro_dps[1] * D2R, .wz = gyro_dps[2] * D2R,
    };
    B.push_imu(B.tracker, &s);
    if (atomic_fetch_add(&B.imu_pushed, 1) == IMU_WARMUP_SAMPLES)
        LOGI("Basalt: accel at init (%.2f, %.2f, %.2f) m/s^2 — expect "
             "(0, -9.8, 0) when worn level", s.ax, s.ay, s.az);
}

void xr_slam_push_pair(const uint8_t *left, const uint8_t *right,
                       uint64_t ts_ns) {
    if (!atomic_load(&B.running) || ts_ns == 0) return;
    /* the VIO derives its gravity/world at the first frame from the IMU
     * pushed so far — don't hand it frames before there is history */
    if (atomic_load(&B.imu_pushed) < IMU_WARMUP_SAMPLES) return;
    if (ts_ns <= B.last_pair_ts) return;       /* must increase monotonically */
    B.last_pair_ts = ts_ns;
    vit_img_sample_t s = {
        .cam_index = 0,
        .timestamp = (int64_t)ts_ns,
        .data = (uint8_t *)left,
        .width = XR_OW, .height = XR_OH,
        .stride = XR_OW, .size = XR_OW * XR_OH,
        .format = VIT_IMAGE_FORMAT_L8,
        .mask_count = 0, .masks = NULL,
    };
    B.push_img(B.tracker, &s);                 /* copies synchronously */
    s.cam_index = 1;
    s.data = (uint8_t *)right;
    B.push_img(B.tracker, &s);
}

/* kb4 unprojection: solve r(theta) = theta + k1 th^3 + ... = r_obs by
 * Newton iteration, then lift to a unit-ish ray */
static void kb4_unproject(const float k[8], float u, float v, float ray[3]) {
    float mx = (u - k[2]) / k[0];
    float my = (v - k[3]) / k[1];
    float r = sqrtf(mx * mx + my * my);
    float th = r;                              /* good initial guess */
    for (int i = 0; i < 5; i++) {
        float t2 = th * th;
        float f = th * (1 + t2 * (k[4] + t2 * (k[5] + t2 * (k[6] + t2 * k[7])))) - r;
        float df = 1 + t2 * (3 * k[4] + t2 * (5 * k[5] + t2 * (7 * k[6] + t2 * 9 * k[7])));
        th -= f / df;
    }
    float s = r > 1e-9f ? sinf(th) / r : 0.0f;
    ray[0] = mx * s;
    ray[1] = my * s;
    ray[2] = cosf(th);
}

/* unpack one popped pose into xr_slam_state (shared by the newest-only
 * poll and the benchmark's every-pose drain); destroys the pose. */
static int fill_state(vit_pose_t *newest, xr_slam_state *out) {
    vit_pose_data_t d;
    if (B.pose_data(newest, &d) != VIT_SUCCESS) {
        B.pose_destroy(newest);
        return 0;
    }
    out->ts = (uint64_t)d.timestamp;
    out->q[0] = d.ow; out->q[1] = d.ox; out->q[2] = d.oy; out->q[3] = d.oz;
    out->p[0] = d.px; out->p[1] = d.py; out->p[2] = d.pz;
    out->v[0] = d.vx; out->v[1] = d.vy; out->v[2] = d.vz;

    /* body -> world rotation of this pose, for landmark reconstruction */
    float Rw[9];
    {
        float x = out->q[1], y = out->q[2], z = out->q[3], w = out->q[0];
        Rw[0] = 1 - 2 * (y * y + z * z); Rw[1] = 2 * (x * y - w * z); Rw[2] = 2 * (x * z + w * y);
        Rw[3] = 2 * (x * y + w * z); Rw[4] = 1 - 2 * (x * x + z * z); Rw[5] = 2 * (y * z - w * x);
        Rw[6] = 2 * (x * z - w * y); Rw[7] = 2 * (y * z + w * x); Rw[8] = 1 - 2 * (x * x + y * y);
    }

    out->n_features = 0;
    out->n_landmarks = 0;
    vit_pose_features_t feats;
    if (B.pose_features(newest, 0, &feats) == VIT_SUCCESS) {
        int n = (int)feats.count;
        if (n > XR_SLAM_MAX_FEATURES) n = XR_SLAM_MAX_FEATURES;
        for (int i = 0; i < n; i++) {
            float u = feats.features[i].u, v = feats.features[i].v;
            float invd = feats.features[i].depth;      /* INVERSE distance */
            out->feat_uv[i][0] = u;
            out->feat_uv[i][1] = v;
            out->feat_id[i] = (int32_t)feats.features[i].id;
            float rc[3];
            kb4_unproject(B.kb4[0], u, v, rc);         /* unit ray, camera */
            const float *R = B.R_ic[0];
            float ri[3] = {                            /* ray in IMU frame */
                R[0] * rc[0] + R[1] * rc[1] + R[2] * rc[2],
                R[3] * rc[0] + R[4] * rc[1] + R[5] * rc[2],
                R[6] * rc[0] + R[7] * rc[1] + R[8] * rc[2],
            };
            memcpy(out->feat_ray[i], ri, sizeof ri);

            /* landmark: camera point at distance 1/invd -> IMU -> world.
             * Low-parallax landmarks triangulate to garbage distances, so
             * the map only takes 5 cm .. 30 m (the live overlays still
             * show every feature). */
            if (invd > 0.0333f && invd < 20.0f) {
                float dist = 1.0f / invd;
                float pi[3] = { ri[0] * dist + B.p_ic[0][0],
                                ri[1] * dist + B.p_ic[0][1],
                                ri[2] * dist + B.p_ic[0][2] };
                int k = out->n_landmarks;
                out->lm_id[k] = (int32_t)feats.features[i].id;
                out->lm_xyz[k][0] = Rw[0] * pi[0] + Rw[1] * pi[1] + Rw[2] * pi[2] + out->p[0];
                out->lm_xyz[k][1] = Rw[3] * pi[0] + Rw[4] * pi[1] + Rw[5] * pi[2] + out->p[1];
                out->lm_xyz[k][2] = Rw[6] * pi[0] + Rw[7] * pi[1] + Rw[8] * pi[2] + out->p[2];
                out->lm_uv[k][0] = u;
                out->lm_uv[k][1] = v;
                out->n_landmarks = k + 1;
            }
        }
        out->n_features = n;
    }
    out->n_features_r = 0;
    if (B.pose_features(newest, 1, &feats) == VIT_SUCCESS) {
        int n = (int)feats.count;
        if (n > XR_SLAM_MAX_FEATURES) n = XR_SLAM_MAX_FEATURES;
        for (int i = 0; i < n; i++) {
            out->feat_uv_r[i][0] = feats.features[i].u;
            out->feat_uv_r[i][1] = feats.features[i].v;
        }
        out->n_features_r = n;
    }
    B.pose_destroy(newest);
    return 1;
}

int xr_slam_poll(xr_slam_state *out) {
    if (!atomic_load(&B.running)) return 0;
    vit_pose_t *newest = NULL;
    for (;;) {
        vit_pose_t *p = NULL;
        if (B.pop_pose(B.tracker, &p) != VIT_SUCCESS || !p) break;
        if (newest) B.pose_destroy(newest);
        newest = p;
    }
    if (!newest) return 0;
    return fill_state(newest, out);
}

int xr_slam_poll_each(void (*cb)(const xr_slam_state *st, void *user),
                      void *user, xr_slam_state *state) {
    if (!atomic_load(&B.running)) return 0;
    int n = 0;
    for (;;) {
        vit_pose_t *p = NULL;
        if (B.pop_pose(B.tracker, &p) != VIT_SUCCESS || !p) break;
        if (fill_state(p, state)) {   /* destroys p */
            cb(state, user);
            n++;
        }
    }
    return n;
}

int xr_slam_start_raw(const xr_slam_cam_raw *left, const xr_slam_cam_raw *right,
                      double imu_hz, const float noises[4]) {
    if (!xr_slam_load()) return -1;
    if (B.tracker) return 0;
    if (left->width != XR_OW || left->height != XR_OH ||
        right->width != XR_OW || right->height != XR_OH) {
        LOGE("Basalt raw: calib %dx%d but this build is compiled for %dx%d — "
             "rebuild with -DXR_OW/-DXR_OH", left->width, left->height,
             XR_OW, XR_OH);
        return -1;
    }

    vit_config_t cfg = { .file = B.config_path[0] ? B.config_path : NULL,
                         .cam_count = 2, .imu_count = 1, .show_ui = false };
    if (cfg.file) LOGI("Basalt: using config %s", cfg.file);
    if (B.create(&cfg, &B.tracker) != VIT_SUCCESS || !B.tracker) {
        LOGE("Basalt: tracker create failed");
        B.tracker = NULL;
        return -1;
    }
    B.enable_ext(B.tracker, VIT_TRACKER_EXTENSION_POSE_FEATURES, true);

    for (int i = 0; i < 2; i++) {
        const xr_slam_cam_raw *c = i ? right : left;
        vit_camera_calibration_t cc;
        memset(&cc, 0, sizeof cc);
        cc.camera_index = (uint32_t)i;
        cc.width = c->width;
        cc.height = c->height;
        cc.frequency = c->fps;
        cc.fx = c->fx; cc.fy = c->fy; cc.cx = c->cx; cc.cy = c->cy;
        cc.model = VIT_CAMERA_DISTORTION_KB4;
        cc.distortion_count = 4;
        for (int k = 0; k < 4; k++) cc.distortion[k] = c->k[k];
        B.kb4[i][0] = c->fx; B.kb4[i][1] = c->fy;
        B.kb4[i][2] = c->cx; B.kb4[i][3] = c->cy;
        for (int k = 0; k < 4; k++) B.kb4[i][4 + k] = c->k[k];
        float R[9];
        quat_to_rot_xyzw(c->q_xyzw, 0, R);
        memcpy(B.R_ic[i], R, sizeof R);
        memcpy(B.p_ic[i], c->p, sizeof B.p_ic[i]);
        for (int r = 0; r < 3; r++) {
            for (int cc2 = 0; cc2 < 3; cc2++)
                cc.transform[r * 4 + cc2] = R[r * 3 + cc2];
            cc.transform[r * 4 + 3] = c->p[r];
        }
        cc.transform[15] = 1.0;
        B.add_cam_calib(B.tracker, &cc);
    }
    LOGI("Basalt raw: kb4 left fx=%.1f k=[%.4f %.4f %.4f %.4f] imu %.0f Hz",
         B.kb4[0][0], B.kb4[0][4], B.kb4[0][5], B.kb4[0][6], B.kb4[0][7],
         imu_hz);

    vit_imu_calibration_t ic;
    memset(&ic, 0, sizeof ic);
    ic.imu_index = 0;
    ic.frequency = imu_hz;
    float nz[4] = { 0.00035f, 0.0001f, 0.00667f, 0.001f };
    if (noises) memcpy(nz, noises, sizeof nz);
    for (int i = 0; i < 3; i++) {
        ic.accel.transform[i * 3 + i] = 1.0;
        ic.gyro.transform[i * 3 + i] = 1.0;
        ic.gyro.noise_std[i] = nz[0];
        ic.gyro.bias_std[i] = nz[1];
        ic.accel.noise_std[i] = nz[2];
        ic.accel.bias_std[i] = nz[3];
    }
    B.add_imu_calib(B.tracker, &ic);

    if (B.start(B.tracker) != VIT_SUCCESS) {
        LOGE("Basalt: tracker start failed");
        B.destroy(B.tracker);
        B.tracker = NULL;
        return -1;
    }
    atomic_store(&B.imu_pushed, 0);
    B.last_pair_ts = 0;
    atomic_store(&B.running, 1);
    LOGI("Basalt VIO started (raw calib, %d cams, %.0f Hz IMU)", 2, imu_hz);
    return 0;
}
