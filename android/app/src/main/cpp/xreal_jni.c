/* xreal_jni.c — Android glue: wrap the USB fd from UsbDeviceConnection with
 * libusb/libuvc, descramble + clean each frame on the UVC callback thread,
 * and hand composed RGBA stereo frames to Kotlin on demand.
 *
 * Threading model: the UVC callback thread owns all image processing and
 * writes the composed frame under a mutex; the Kotlin UI thread polls
 * nativeGrabFrame() which copies it out under the same mutex.
 */
#include <jni.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <libusb.h>
#include <libuvc/libuvc.h>

#include "xr_map.h"
#include "xr_slam.h"
#include "xr_stereo.h"
#include "xr_track.h"
#include "xreal_align.h"
#include "xreal_core.h"
#include "xreal_gles.h"
#include "xreal_imu.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MAX_W (2 * XR_W)     /* scrambled view: 1280x480 */
#define MAX_H XR_OH          /* clean view:      960x640 */

/* 0 = IMU-only diagnostic (Basalt never started, pose = AHRS). The IMU
 * pose was signed off on-device with the ground-truth remap; Basalt now
 * receives frames, extrinsics and inertial data in one consistent frame. */
#define XR_ENABLE_BASALT 1

static struct {
    uvc_context_t *ctx;
    uvc_device_handle_t *devh;
    int streaming;

    xr_order order;
    xr_cleaner cleaners[2];
    uint8_t frame[XR_FRAME_BYTES];          /* working copy of the UVC frame */
    uint8_t clean[2][XR_OW * XR_OH];        /* equalized, for viewing */
    uint8_t clean_raw[2][XR_OW * XR_OH];    /* non-equalized, for tracking */
    int have[2];

    /* SLAM worker (research app): rectify + track + depth at sensor rate.
     * The tracker is the live front-end stand-in; Basalt plugs in behind
     * the same data flow (see docs/VSLAM.md). */
    pthread_t slam_thread;
    atomic_int slam_running;
    pthread_mutex_t slam_lock;
    pthread_cond_t slam_cond;
    uint32_t pair_seq;
    uint64_t pair_ts, slam_prev_ts;
    xr_track track;
    int stereo_ready;
    atomic_int depth_on;
    atomic_int show_pts;
    uint8_t slam_in[2][XR_OW * XR_OH];      /* worker's copy of clean_raw */
    uint8_t disp[XS_W * XS_H];              /* published disparity */
    float pts_pane[XR_SLAM_MAX_FEATURES][2]; /* dots for the phone pane (cam px) */
    int pts_n;
    int pts_src;                            /* 1 = Basalt features, 0 = fallback */
    float pts_pane_r[XR_SLAM_MAX_FEATURES][2]; /* dots for the right pane */
    int pts_n_r;
    int track_count;
    float depth_ms;
    atomic_int pane_mode;                   /* 0 = left|depth, 1 = left|right */

    int map_n;
    uint32_t map_stamp;
    xr_slam_state vio;                      /* newest Basalt state */
    int vio_fresh;

    pthread_mutex_t lock;
    uint8_t rgba[MAX_W * MAX_H * 4];
    int cw, ch, counter;
    uint32_t seq, grabbed;

    /* factory calibration blob (fetched over the IMU command channel) and
     * the world-aligned per-eye passthrough state built from it; glasses
     * output itself goes through the GLES renderer (xreal_gles.c) */
    uint8_t *config;
    uint32_t config_len;
    xr_eye_calib eye_calib[2];        /* [0]=left eye, [1]=right eye */
    float imu_gyro_bias[3];           /* factory IMU calibration (rad/s, */
    float imu_accel_bias[3];          /* m/s^2, noise densities) */
    float imu_noises[4];
    int imu_calib_have;
    atomic_int align_have;            /* calibration params received */
    atomic_int align_variant;

    int fps_count;
    int64_t fps_t0;
    int fps_x10;

    atomic_int swap_eyes;       /* swap the glasses' eyes (debug) */
    atomic_int stereo_mode;     /* 1: per-eye stereo display + aligned warp;
                                   0: mirror display + plain framebuffer */

    /* IMU (HID interface 2 over the same libusb handle) */
    libusb_device_handle *usb;
    int imu_ep_in;
    int imu_ep_out;             /* interrupt OUT for commands, 0 if absent */

    /* MCU channel (HID interface 0): display mode switching */
    int mcu_claimed;
    int mcu_ep_in, mcu_ep_out;
    pthread_t imu_thread;
    atomic_int imu_running;
    pthread_mutex_t imu_lock;
    xr_imu_sample imu_latest;
    float imu_q[4];
    int imu_has_q;
    uint32_t imu_seq, imu_grabbed;
    int imu_count, imu_rate;
    int64_t imu_t0;
    xr_ahrs ahrs;
    /* sample ring so the UI can drain the full 1 kHz stream in batches */
    xr_imu_sample ring[1024];
    uint32_t ring_head, ring_tail;          /* head = write, tail = read */

    /* pose history for the timewarp: (ts, quaternion) at 1 kHz, 0.25 s deep */
    struct { uint64_t ts; float q[4]; } qhist[256];
    uint32_t qhist_n;

    /* raw-stream (pre-remap) gyro bias capture: one log line to check the
     * channel signs against the factory calibration numbers */
    double raw_gbias[3];
    int raw_gbias_n;
} S = { .lock = PTHREAD_MUTEX_INITIALIZER,
        .imu_lock = PTHREAD_MUTEX_INITIALIZER,
        .slam_lock = PTHREAD_MUTEX_INITIALIZER,
        .slam_cond = PTHREAD_COND_INITIALIZER,
        .align_variant = XR_ALIGN_VARIANT_DEFAULT,
        .stereo_mode = 1, .depth_on = 1, .show_pts = 1 };

/* ~14 MB of rectification maps + SGM scratch. Kept outside S: S has nonzero
 * initializers, so its members live in .data and would bloat the .so by
 * this much per ABI; a plain zero static stays in .bss. */
static xr_stereo ST;

/* same .bss rule: the Basalt feed frames and the accumulated landmark map
 * (open-addressed on landmark id, stamp 0 = empty slot; guarded by S.lock) */
static uint8_t SLAM_FEED[2][XR_OW * XR_OH];   /* contrast-stretched frames */
static struct {
    int32_t id;
    float x, y, z;
    uint32_t stamp;
    uint16_t seen;      /* observations: one-shot triangulations are noise */
    uint16_t pad_;
} MAP_PT[4096];

static int64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* disparity (quarter-pixel) -> color ramp: far/small = deep blue,
 * near/large = red; 0 (invalid) = black */
static void disp_color(uint8_t d, uint8_t px[4]) {
    if (!d) { px[0] = px[1] = px[2] = 0; px[3] = 0xFF; return; }
    float t = (float)d / (float)XS_MAX_OUT;
    if (t > 1.0f) t = 1.0f;
    float r, g, b;
    if (t < 0.33f)      { float u = t / 0.33f;          r = 0;      g = u;          b = 1; }
    else if (t < 0.66f) { float u = (t - 0.33f) / 0.33f; r = u;      g = 1;          b = 1 - u; }
    else                { float u = (t - 0.66f) / 0.34f; r = 1;      g = 1 - u;      b = 0; }
    px[0] = (uint8_t)(r * 255); px[1] = (uint8_t)(g * 255);
    px[2] = (uint8_t)(b * 255); px[3] = 0xFF;
}

/* Research-app phone view: left pane = left camera + tracked features,
 * right pane = the disparity map colorized and upscaled 2x (240x320 ->
 * 480x640). The glasses passthrough below is untouched by this. */
static void compose(int counter, uint64_t exposure_ts) {
    int swap = atomic_load(&S.swap_eyes);
    pthread_mutex_lock(&S.lock);
    S.cw = 2 * XR_OW;
    S.ch = XR_OH;
    S.counter = counter;

    /* left pane: equalized left camera (cam1) */
    for (int y = 0; y < XR_OH; y++) {
        uint8_t *dst = S.rgba + (size_t)y * S.cw * 4;
        const uint8_t *s = S.clean[1] + y * XR_OW;
        for (int x = 0; x < XR_OW; x++) {
            uint8_t v = s[x];
            dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = 0xFF;
            dst += 4;
        }
    }
    /* tracked features: green = Basalt's own optical-flow keypoints,
     * orange = the built-in fallback tracker (Basalt not loaded) */
    if (atomic_load(&S.show_pts)) {
        uint8_t cr = S.pts_src ? 0x00 : 0xFF;
        uint8_t cg = S.pts_src ? 0xFF : 0xAA;
        uint8_t cb = S.pts_src ? 0x50 : 0x00;
        for (int i = 0; i < S.pts_n; i++) {
            int px = (int)S.pts_pane[i][0], py = (int)S.pts_pane[i][1];
            for (int dy = -1; dy <= 1; dy++) {
                int y = py + dy;
                if (y < 0 || y >= XR_OH) continue;
                for (int dx = -1; dx <= 1; dx++) {
                    int x = px + dx;
                    if (x < 0 || x >= XR_OW) continue;
                    uint8_t *d = S.rgba + ((size_t)y * S.cw + x) * 4;
                    d[0] = cr; d[1] = cg; d[2] = cb; d[3] = 0xFF;
                }
            }
        }
    }
    /* right pane: the right camera, or colorized depth */
    if (atomic_load(&S.pane_mode) == 1) {
        for (int y = 0; y < XR_OH; y++) {
            uint8_t *dst = S.rgba + ((size_t)y * S.cw + XR_OW) * 4;
            const uint8_t *sr = S.clean[0] + y * XR_OW;
            for (int x = 0; x < XR_OW; x++) {
                uint8_t v = sr[x];
                dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = 0xFF;
                dst += 4;
            }
        }
        /* the same landmarks as observed by the right camera */
        if (atomic_load(&S.show_pts) && S.pts_src) {
            for (int i = 0; i < S.pts_n_r; i++) {
                int px = (int)S.pts_pane_r[i][0], py = (int)S.pts_pane_r[i][1];
                for (int dy = -1; dy <= 1; dy++) {
                    int y = py + dy;
                    if (y < 0 || y >= XR_OH) continue;
                    for (int dx = -1; dx <= 1; dx++) {
                        int x = px + dx;
                        if (x < 0 || x >= XR_OW) continue;
                        uint8_t *d = S.rgba + ((size_t)y * S.cw + XR_OW + x) * 4;
                        d[0] = 0; d[1] = 0xFF; d[2] = 0x50; d[3] = 0xFF;
                    }
                }
            }
        }
    } else if (atomic_load(&S.depth_on)) {
        for (int y = 0; y < XR_OH; y++) {
            uint8_t *dst = S.rgba + ((size_t)y * S.cw + XR_OW) * 4;
            const uint8_t *dr = S.disp + (y >> 1) * XS_W;
            for (int x = 0; x < XR_OW; x++) {
                disp_color(dr[x >> 1], dst);
                dst += 4;
            }
        }
    } else {
        for (int y = 0; y < XR_OH; y++) {
            uint8_t *dst = S.rgba + ((size_t)y * S.cw + XR_OW) * 4;
            for (int x = 0; x < XR_OW; x++) {
                dst[0] = dst[1] = dst[2] = 24; dst[3] = 0xFF;
                dst += 4;
            }
        }
    }
    S.seq++;
    int cw = S.cw, ch = S.ch;
    pthread_mutex_unlock(&S.lock);

    /* glasses output through the GLES renderer (front-buffer when the device
     * supports it). All source buffers are written only by this thread. */
    if (atomic_load(&S.stereo_mode) && atomic_load(&S.align_have)) {
        /* left eye view samples cam1, the physical left camera */
        xr_gles_submit_eyes(S.clean[swap ? 0 : 1], S.clean[swap ? 1 : 0],
                            exposure_ts);
    } else {
        xr_gles_submit_pair(S.rgba, cw, ch);
    }
}

static void frame_cb(uvc_frame_t *frame, void *user) {
    (void)user;
    if (frame->data_bytes < XR_FRAME_BYTES) return;   /* startup runts */
    memcpy(S.frame, frame->data, XR_FRAME_BYTES);

    if (S.order == XR_ORDER_UNKNOWN) {
        S.order = xr_classify(S.frame);
        if (S.order == XR_ORDER_UNKNOWN) return;      /* black startup frame,
            or unsupported old firmware - update the glasses at
            https://ota.xreal.com/ultra-update?version=1 */
        LOGI("stream fingerprinted: byte order %s",
             S.order == XR_ORDER_SWAPPED ? "swapped (fixing)" : "ok");
    }
    if (S.order == XR_ORDER_SWAPPED) xr_unswap16(S.frame, XR_FRAME_BYTES);

    /* skip all-black frames right after startup (same test as the viewers) */
    int64_t sum = 0;
    for (int i = 0; i < XR_IMG_BYTES; i += 7) sum += S.frame[i];
    if (sum / (XR_IMG_BYTES / 7) < 5) return;

    int cam = xr_cam(S.frame);
    static uint8_t dscr[XR_OW * XR_OH];               /* uvc thread only */
    xr_descramble(S.frame, cam, dscr);
    /* clean without equalization for the trackers (per-frame global HE
     * flickers), then an equalized copy for humans */
    xr_clean(&S.cleaners[cam], dscr, S.clean_raw[cam], 0);
    xr_equalize(S.clean_raw[cam], S.clean[cam], XR_OW * XR_OH);
    /* Basalt feed conditioning, two flicker-free stages:
     *
     * 1. VIGNETTE COMPENSATION — the fisheye's radial falloff violates the
     *    brightness-constancy assumption of Basalt's patch tracker at the
     *    periphery (TUM photometric-calibration lineage). The profile is
     *    estimated online: EMA'd mean intensity per radial bin around the
     *    calibrated principal point; with head motion the scene averages
     *    out and the stable radial ratio IS the vignette. Gain capped 3.5x.
     * 2. CONTRAST STRETCH between slowly-EMA'd 2%/98% percentiles — lifts
     *    the grainy low-contrast frames for the FAST detector without
     *    per-frame equalization's brightness flicker. */
    {
        static uint8_t rbin[2][XR_OW * XR_OH];   /* radial bin per pixel */
        static int rbin_ready[2];
        static float bin_mean[2][64];
        static int bin_warm[2];
        static uint16_t gainq[2][64];            /* gain * 256 */
        static int gain_ready[2];
        const uint8_t *in = S.clean_raw[cam];

        if (!rbin_ready[cam] && atomic_load(&S.align_have)) {
            /* eye_calib[0] holds the LEFT camera (= cam1) */
            const float *cc = S.eye_calib[cam == 1 ? 0 : 1].cc;
            for (int y = 0; y < XR_OH; y++)
                for (int x = 0; x < XR_OW; x++) {
                    float r = hypotf((float)x - cc[0], (float)y - cc[1]);
                    int b = (int)(r * (63.0f / 400.0f));
                    rbin[cam][y * XR_OW + x] = (uint8_t)(b > 63 ? 63 : b);
                }
            for (int b = 0; b < 64; b++) gainq[cam][b] = 256;
            rbin_ready[cam] = 1;
        }
        if (rbin_ready[cam]) {
            float sum[64] = { 0 };
            int cnt[64] = { 0 };
            for (int i = 0; i < XR_OW * XR_OH; i += 4) {
                sum[rbin[cam][i]] += in[i];
                cnt[rbin[cam][i]]++;
            }
            float a = bin_warm[cam] < 60 ? 0.15f : 0.02f;
            for (int b = 0; b < 64; b++) {
                if (cnt[b] < 32) continue;
                float m = sum[b] / (float)cnt[b];
                bin_mean[cam][b] = bin_mean[cam][b] == 0.0f
                                       ? m
                                       : bin_mean[cam][b] + a * (m - bin_mean[cam][b]);
            }
            if (++bin_warm[cam] >= 60) {
                float ref = 0;
                int nref = 0;
                for (int b = 2; b <= 6; b++)
                    if (bin_mean[cam][b] > 1) { ref += bin_mean[cam][b]; nref++; }
                if (nref) {
                    ref /= (float)nref;
                    for (int b = 0; b < 64; b++) {
                        float g = bin_mean[cam][b] > 4.0f
                                      ? ref / bin_mean[cam][b] : 1.0f;
                        if (g < 1.0f) g = 1.0f;
                        if (g > 3.5f) g = 3.5f;
                        gainq[cam][b] = (uint16_t)(g * 256.0f);
                    }
                    gain_ready[cam] = 1;
                }
            }
        }

        static float lo_f[2], hi_f[2];
        static int have_stretch[2];
        int32_t hist[256] = { 0 };
        for (int i = 0; i < XR_OW * XR_OH; i += 2) hist[in[i]]++;
        int32_t tail = (XR_OW * XR_OH / 2) / 50;      /* 2% each side */
        int32_t acc = 0;
        int lo = 0, hi = 255;
        for (int i = 0; i < 256; i++) { acc += hist[i]; if (acc >= tail) { lo = i; break; } }
        acc = 0;
        for (int i = 255; i >= 0; i--) { acc += hist[i]; if (acc >= tail) { hi = i; break; } }
        if (hi - lo < 8) hi = lo + 8;
        if (!have_stretch[cam]) {
            lo_f[cam] = (float)lo; hi_f[cam] = (float)hi; have_stretch[cam] = 1;
        } else {
            lo_f[cam] += 0.05f * ((float)lo - lo_f[cam]);
            hi_f[cam] += 0.05f * ((float)hi - hi_f[cam]);
        }
        float scale = 240.0f / (hi_f[cam] - lo_f[cam]);
        uint8_t lut[256];
        for (int i = 0; i < 256; i++) {
            float v = ((float)i - lo_f[cam]) * scale + 8.0f;
            lut[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
        uint8_t *out = SLAM_FEED[cam];
        if (gain_ready[cam]) {
            const uint8_t *rb = rbin[cam];
            const uint16_t *gq = gainq[cam];
            for (int i = 0; i < XR_OW * XR_OH; i++) {
                uint32_t v = ((uint32_t)in[i] * gq[rb[i]]) >> 8;
                out[i] = lut[v > 255 ? 255 : v];
            }
        } else {
            for (int i = 0; i < XR_OW * XR_OH; i++) out[i] = lut[in[i]];
        }
    }
    S.have[cam] = 1;

    S.fps_count++;
    int64_t t = now_ms();
    if (S.fps_t0 == 0) S.fps_t0 = t;
    if (t - S.fps_t0 >= 1000) {
        S.fps_x10 = (int)(S.fps_count * 10000 / (t - S.fps_t0));
        S.fps_count = 0;
        S.fps_t0 = t;
    }

    if (S.have[0] && S.have[1]) {
        /* unwrap the camera's u32 ns exposure stamp against the IMU's u64
         * clock (same timebase, see docs/PROTOCOL.md "Clock domains") */
        uint64_t t_exp = 0;
        pthread_mutex_lock(&S.imu_lock);
        uint64_t t_imu = S.imu_latest.ts_ns;
        pthread_mutex_unlock(&S.imu_lock);
        if (t_imu)
            t_exp = t_imu -
                    ((t_imu - (uint64_t)xr_exposure_ts(S.frame)) & 0xFFFFFFFFull);
        compose(xr_counter(S.frame), t_exp);

        /* feed Basalt directly from this thread (it copies synchronously;
         * VIT wants cam0=left first, then cam1, same timestamp) */
        xr_slam_push_pair(SLAM_FEED[1], SLAM_FEED[0], t_exp);

        /* hand the pair to the SLAM worker (it snapshots the buffers) */
        pthread_mutex_lock(&S.slam_lock);
        S.pair_seq++;
        S.pair_ts = t_exp;
        pthread_cond_signal(&S.slam_cond);
        pthread_mutex_unlock(&S.slam_lock);
    }
}

/* ---- timewarp pose delta ------------------------------------------------------ */

/* Hamilton [w,x,y,z] (the AHRS convention) -> rotation matrix, row-major. */
static void wxyz_to_rot(const float q[4], float R[9]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
}

/* Soft deadband on the warp angle: at rest the AHRS wobbles by fractions of
 * a degree, which would jitter the whole view; subtracting the deadband
 * makes stillness exactly still, and a 0.1-degree lag during real motion is
 * imperceptible. */
#define TW_DEADBAND_RAD (0.1f * (float)M_PI / 180.0f)

/* quaternion nearest (<= ts, clamped) from the 1 kHz history; ts == 0 means
 * the newest sample. Returns 0, or -1 when no usable history exists. */
static int qhist_get(uint64_t ts, float q[4]) {
    pthread_mutex_lock(&S.imu_lock);
    uint32_t n = S.qhist_n;
    if (n < 2) {
        pthread_mutex_unlock(&S.imu_lock);
        return -1;
    }
    uint32_t best = n - 1;
    if (ts != 0) {
        uint32_t depth = n < 256 ? n : 256;
        for (uint32_t k = 0; k < depth; k++) {
            uint32_t i = n - 1 - k;
            if (S.qhist[i % 256].ts <= ts) { best = i; break; }
            best = i;                          /* clamp to oldest available */
        }
    }
    memcpy(q, S.qhist[best % 256].q, 4 * sizeof(float));
    pthread_mutex_unlock(&S.imu_lock);
    return 0;
}

/* delta quaternion qd = conj(qa) * qb (Hamilton wxyz), w kept positive:
 * R(qd) = R(qa)^T * R(qb). */
static void quat_delta(const float qa[4], const float qb[4], float qd[4]) {
    qd[0] = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
    qd[1] = qa[0] * qb[1] - qa[1] * qb[0] - qa[2] * qb[3] + qa[3] * qb[2];
    qd[2] = qa[0] * qb[2] + qa[1] * qb[3] - qa[2] * qb[0] - qa[3] * qb[1];
    qd[3] = qa[0] * qb[3] - qa[1] * qb[2] + qa[2] * qb[1] - qa[3] * qb[0];
    if (qd[0] < 0) { qd[0] = -qd[0]; qd[1] = -qd[1]; qd[2] = -qd[2]; qd[3] = -qd[3]; }
}

/* raw rotation of the IMU frame between two history times (no deadband) —
 * used for the tracker's search-window prediction. ts_b == 0 -> now. */
static int pose_delta_between(uint64_t ts_a, uint64_t ts_b, float dR[9]) {
    float qa[4], qb[4], qd[4];
    if (qhist_get(ts_a, qa) || qhist_get(ts_b, qb)) return -1;
    quat_delta(qa, qb, qd);
    wxyz_to_rot(qd, dR);
    return 0;
}

/* dR = R(q_exposure)^T * R(q_now) with the timewarp deadband: how a fixed
 * world direction moved in the IMU frame since the camera exposure. Drift
 * cancels over this short window, so the AHRS absolute error is irrelevant. */
static int pose_delta(uint64_t ts_exposure, float dR[9]) {
    float qe[4], qn[4], qd[4];
    if (qhist_get(ts_exposure, qe) || qhist_get(0, qn)) return -1;
    quat_delta(qe, qn, qd);

    float vn = sqrtf(qd[1] * qd[1] + qd[2] * qd[2] + qd[3] * qd[3]);
    float angle = 2.0f * atan2f(vn, qd[0]);
    if (angle <= TW_DEADBAND_RAD || vn < 1e-9f) {
        memset(dR, 0, 9 * sizeof(float));
        dR[0] = dR[4] = dR[8] = 1.0f;          /* still = exactly still */
        return 0;
    }
    float soft = angle - TW_DEADBAND_RAD;
    float s = sinf(soft * 0.5f) / vn;
    qd[0] = cosf(soft * 0.5f);
    qd[1] *= s; qd[2] *= s; qd[3] *= s;
    wxyz_to_rot(qd, dR);
    return 0;
}

/* ---- SLAM worker: rectify + track + depth at sensor rate --------------------- */

static void *slam_worker(void *arg) {
    (void)arg;
    /* below the UVC/render threads: depth is the first thing to yield when
     * cores get scarce (SD888-class), capture and present never stutter */
    setpriority(PRIO_PROCESS, (id_t)gettid(), 10);
    uint32_t done_seq = 0;
    for (;;) {
        pthread_mutex_lock(&S.slam_lock);
        while (atomic_load(&S.slam_running) && S.pair_seq == done_seq)
            pthread_cond_wait(&S.slam_cond, &S.slam_lock);
        if (!atomic_load(&S.slam_running)) {
            pthread_mutex_unlock(&S.slam_lock);
            return NULL;
        }
        done_seq = S.pair_seq;           /* always the newest pair, no queue */
        uint64_t ts = S.pair_ts;
        pthread_mutex_unlock(&S.slam_lock);

        if (!S.stereo_ready) {
            if (!atomic_load(&S.align_have)) continue;
            float bx = S.eye_calib[1].p_cam[0] - S.eye_calib[0].p_cam[0];
            float by = S.eye_calib[1].p_cam[1] - S.eye_calib[0].p_cam[1];
            float bz = S.eye_calib[1].p_cam[2] - S.eye_calib[0].p_cam[2];
            if (bx * bx + by * by + bz * bz < 1e-4f) continue;   /* no p_cam */
            xr_stereo_init(&ST, &S.eye_calib[0], &S.eye_calib[1],
                           S.eye_calib[0].p_cam, S.eye_calib[1].p_cam,
                           atomic_load(&S.align_variant));
            xr_track_reset(&S.track);
            S.stereo_ready = 1;
            xr_gles_set_rect(ST.R_rect_imu, ST.f_rect,
                             (XS_W - 1) * 0.5f, (XS_H - 1) * 0.5f);
            LOGI("stereo rectification ready: baseline %.1f cm, f_rect %.0f px",
                 (double)ST.baseline_m * 100.0, (double)ST.f_rect);
        }

        /* snapshot the tracker inputs; written by the UVC thread, and a rare
         * mid-write copy only smears one frame during fast motion */
        memcpy(S.slam_in[0], S.clean_raw[1], sizeof S.slam_in[0]); /* left  = cam1 */
        memcpy(S.slam_in[1], S.clean_raw[0], sizeof S.slam_in[1]); /* right = cam0 */

        xr_stereo_rectify(&ST, 0, S.slam_in[0]);

        if (xr_slam_running()) {
            /* Basalt is the front end: publish its pose and features (the
             * feature u,v are per-camera pixels — the panes are 1:1) */
            static xr_slam_state st;               /* worker thread only */
            if (xr_slam_poll(&st)) {
                float rays[XR_SLAM_MAX_FEATURES * 3];
                int nr = st.n_features;
                pthread_mutex_lock(&S.lock);
                S.vio = st;
                S.vio_fresh = 1;
                S.pts_src = 1;
                S.pts_n = nr;
                for (int i = 0; i < nr; i++) {
                    S.pts_pane[i][0] = st.feat_uv[i][0];
                    S.pts_pane[i][1] = st.feat_uv[i][1];
                    memcpy(&rays[i * 3], st.feat_ray[i], 3 * sizeof(float));
                }
                S.pts_n_r = st.n_features_r;
                for (int i = 0; i < st.n_features_r; i++) {
                    S.pts_pane_r[i][0] = st.feat_uv_r[i][0];
                    S.pts_pane_r[i][1] = st.feat_uv_r[i][1];
                }
                S.track_count = nr;
                /* fold this pose's landmarks into the accumulated map */
                S.map_stamp++;
                for (int i = 0; i < st.n_landmarks; i++) {
                    uint32_t h = ((uint32_t)st.lm_id[i] * 2654435761u) & 4095u;
                    int slot = -1, oldest = -1;
                    uint32_t oldest_stamp = 0xFFFFFFFFu;
                    for (int k = 0; k < 32; k++) {
                        uint32_t j = (h + k) & 4095u;
                        if (MAP_PT[j].stamp == 0) { slot = (int)j; break; }
                        if (MAP_PT[j].id == st.lm_id[i]) { slot = (int)j; break; }
                        if (MAP_PT[j].stamp < oldest_stamp) {
                            oldest_stamp = MAP_PT[j].stamp;
                            oldest = (int)j;
                        }
                    }
                    if (slot < 0) slot = oldest;   /* neighborhood full: evict */
                    if (MAP_PT[slot].stamp != 0 &&
                        MAP_PT[slot].id == st.lm_id[i]) {
                        /* re-observed: blend toward the newer (better-
                         * converged) estimate instead of keeping the first */
                        MAP_PT[slot].x = 0.7f * MAP_PT[slot].x + 0.3f * st.lm_xyz[i][0];
                        MAP_PT[slot].y = 0.7f * MAP_PT[slot].y + 0.3f * st.lm_xyz[i][1];
                        MAP_PT[slot].z = 0.7f * MAP_PT[slot].z + 0.3f * st.lm_xyz[i][2];
                        if (MAP_PT[slot].seen < 65535) MAP_PT[slot].seen++;
                    } else {
                        if (MAP_PT[slot].stamp == 0) S.map_n++;
                        MAP_PT[slot].id = st.lm_id[i];
                        MAP_PT[slot].x = st.lm_xyz[i][0];
                        MAP_PT[slot].y = st.lm_xyz[i][1];
                        MAP_PT[slot].z = st.lm_xyz[i][2];
                        MAP_PT[slot].seen = 1;
                    }
                    MAP_PT[slot].stamp = S.map_stamp;
                }
                /* AR eye mode: snapshot the gated landmark map + this pose
                 * so the renderer can draw world-anchored points */
                static float map_snap[XR_GLES_MAX_MAP * 3]; /* worker only */
                int mn = 0;
                for (int i = 0; i < 4096 && mn < XR_GLES_MAX_MAP; i++) {
                    if (!MAP_PT[i].stamp || MAP_PT[i].seen < 3) continue;
                    map_snap[mn * 3] = MAP_PT[i].x;
                    map_snap[mn * 3 + 1] = MAP_PT[i].y;
                    map_snap[mn * 3 + 2] = MAP_PT[i].z;
                    mn++;
                }
                pthread_mutex_unlock(&S.lock);
                xr_gles_set_points(rays, atomic_load(&S.show_pts) ? nr : 0,
                                   st.ts);
                float Rw[9];
                wxyz_to_rot(st.q, Rw);
                xr_gles_set_map(map_snap, mn, Rw, st.p, st.ts);

                /* session-map layer: motion-gated keyframes on the same
                 * conditioned feed Basalt sees; heavy work (XFeat/ORB +
                 * matching) runs on the dedicated map thread */
                xr_map_offer(st.q, st.p, st.ts, SLAM_FEED[1],
                             st.lm_id, st.lm_xyz, st.lm_uv, st.n_landmarks);
            }
        } else if (!xr_slam_load()) {
            /* fallback front end (ONLY when libbasalt.so is absent, e.g.
             * armeabi-v7a): the built-in tracker on the rectified left
             * image, with a gyro-predicted search window. With Basalt
             * present the dots are always its own features — nothing is
             * shown while it starts up. */
            float pdx = 0.0f, pdy = 0.0f;
            float dR[9];
            if (S.slam_prev_ts && ts &&
                pose_delta_between(S.slam_prev_ts, ts, dR) == 0) {
                const float *Rr = ST.R_rect_imu;
                float rz[3] = { Rr[2], Rr[5], Rr[8] }; /* rect z in IMU frame */
                float rn[3] = {                        /* ray now = dR^T rz */
                    dR[0] * rz[0] + dR[3] * rz[1] + dR[6] * rz[2],
                    dR[1] * rz[0] + dR[4] * rz[1] + dR[7] * rz[2],
                    dR[2] * rz[0] + dR[5] * rz[1] + dR[8] * rz[2],
                };
                float rr[3] = {                        /* back into rect frame */
                    Rr[0] * rn[0] + Rr[3] * rn[1] + Rr[6] * rn[2],
                    Rr[1] * rn[0] + Rr[4] * rn[1] + Rr[7] * rn[2],
                    Rr[2] * rn[0] + Rr[5] * rn[1] + Rr[8] * rn[2],
                };
                if (rr[2] > 0.1f) {
                    pdx = ST.f_rect * rr[0] / rr[2];
                    pdy = ST.f_rect * rr[1] / rr[2];
                }
            }
            S.slam_prev_ts = ts;

            int n = xr_track_step(&S.track, ST.rect[0], pdx, pdy);

            float rays[XT_MAX * 3];
            int nr = 0;
            const float cx = (XS_W - 1) * 0.5f, cy = (XS_H - 1) * 0.5f;
            pthread_mutex_lock(&S.lock);
            S.pts_src = 0;
            S.pts_n = 0;
            for (int i = 0; i < XT_MAX; i++) {
                /* fresh seeds (age < 2) flicker; only show survivors */
                if (!S.track.pt[i].alive || S.track.pt[i].age < 2) continue;
                int rx = (int)S.track.pt[i].x, ry = (int)S.track.pt[i].y;
                if (rx < 0 || rx >= XS_W || ry < 0 || ry >= XS_H) continue;
                int32_t m = ST.map[0][ry * XS_W + rx];
                if (m >= 0) {
                    S.pts_pane[S.pts_n][0] = (float)(m % XR_OW);
                    S.pts_pane[S.pts_n][1] = (float)(m / XR_OW);
                    S.pts_n++;
                }
                const float *Rr = ST.R_rect_imu;
                float v[3] = { (S.track.pt[i].x - cx) / ST.f_rect,
                               (S.track.pt[i].y - cy) / ST.f_rect, 1.0f };
                rays[nr * 3 + 0] = Rr[0] * v[0] + Rr[1] * v[1] + Rr[2] * v[2];
                rays[nr * 3 + 1] = Rr[3] * v[0] + Rr[4] * v[1] + Rr[5] * v[2];
                rays[nr * 3 + 2] = Rr[6] * v[0] + Rr[7] * v[1] + Rr[8] * v[2];
                nr++;
            }
            S.track_count = n;
            pthread_mutex_unlock(&S.lock);
            xr_gles_set_points(rays, atomic_load(&S.show_pts) ? nr : 0, ts);
        }

        /* stereo depth (SGM) at every 2nd pair (15 Hz) by default — the
         * Snapdragon 888 needs the headroom for Basalt + future AR work.
         * Only runs per-pair when a pass fits in half a frame budget. */
        static uint8_t disp_local[XS_W * XS_H];        /* slam thread only */
        static uint32_t depth_tick;
        static int depth_stride = 2;
        if (atomic_load(&S.depth_on)) {
            if (depth_tick++ % depth_stride == 0) {
                xr_stereo_rectify(&ST, 1, S.slam_in[1]);
                int64_t t0 = now_ms();
                xr_stereo_depth(&ST, disp_local);
                float depth_ms = (float)(now_ms() - t0);
                depth_stride = depth_ms > 13.0f ? 2 : 1;
                pthread_mutex_lock(&S.lock);
                memcpy(S.disp, disp_local, sizeof S.disp);
                S.depth_ms = depth_ms;
                pthread_mutex_unlock(&S.lock);

                /* colorized copy for the glasses' depth passthrough (black
                 * border so out-of-image samples clamp to black) */
                static uint8_t drgba[XS_W * XS_H * 4];   /* slam thread only */
                for (int y = 0; y < XS_H; y++) {
                    int border_y = y == 0 || y == XS_H - 1;
                    for (int x = 0; x < XS_W; x++) {
                        uint8_t *px = drgba + ((size_t)y * XS_W + x) * 4;
                        if (border_y || x == 0 || x == XS_W - 1) {
                            px[0] = px[1] = px[2] = 0; px[3] = 0xFF;
                        } else {
                            disp_color(disp_local[y * XS_W + x], px);
                        }
                    }
                }
                xr_gles_submit_depth(drgba, XS_W, XS_H);
            }
        } else {
            pthread_mutex_lock(&S.lock);
            memset(S.disp, 0, sizeof S.disp);
            S.depth_ms = 0;
            pthread_mutex_unlock(&S.lock);
        }
    }
}

static void slam_start(void) {
    if (atomic_load(&S.slam_running)) return;
    S.pair_seq = 0;
    S.slam_prev_ts = 0;
    S.stereo_ready = 0;              /* rebuild with whatever calib arrives */
    S.pts_n = 0;
    S.track_count = 0;
    S.depth_ms = 0;
    memset(S.disp, 0, sizeof S.disp);
    xr_track_reset(&S.track);
    atomic_store(&S.slam_running, 1);
    pthread_create(&S.slam_thread, NULL, slam_worker, NULL);
}

static void slam_stop(void) {
    if (!atomic_load(&S.slam_running)) return;
    pthread_mutex_lock(&S.slam_lock);
    atomic_store(&S.slam_running, 0);
    pthread_cond_broadcast(&S.slam_cond);
    pthread_mutex_unlock(&S.slam_lock);
    pthread_join(S.slam_thread, NULL);
}

/* ---- IMU: enable + drain thread --------------------------------------------- */

/* ---- MCU channel: put the glasses into per-eye stereo mode ------------------ */

static void find_interface_eps(int ifnum, int *ep_in, int *ep_out) {
    *ep_in = *ep_out = 0;
    struct libusb_config_descriptor *cfg;
    if (libusb_get_active_config_descriptor(libusb_get_device(S.usb), &cfg) != 0)
        return;
    for (int i = 0; i < cfg->bNumInterfaces; i++) {
        const struct libusb_interface_descriptor *alt =
            &cfg->interface[i].altsetting[0];
        if (alt->bInterfaceNumber != ifnum) continue;
        for (int e = 0; e < alt->bNumEndpoints; e++) {
            uint8_t addr = alt->endpoint[e].bEndpointAddress;
            if (addr & LIBUSB_ENDPOINT_IN) *ep_in = addr;
            else *ep_out = addr;
        }
    }
    libusb_free_config_descriptor(cfg);
}

/* Set the glasses' display mode (XR_DISPLAY_*). Stereo is what makes the
 * left half of the frame reach only the left eye - without it both eyes see
 * the whole side-by-side canvas, exactly what AR apps switch on themselves.
 * Returns 0 on acked success, 1 on sent-without-ack, -1 on rejection/error. */
static int mcu_set_display_mode(uint8_t mode) {
    if (!S.mcu_claimed) {
        if (libusb_claim_interface(S.usb, XR_MCU_INTERFACE) != 0) {
            LOGE("MCU: could not claim interface %d, display mode unchanged",
                 XR_MCU_INTERFACE);
            return -1;
        }
        S.mcu_claimed = 1;
        find_interface_eps(XR_MCU_INTERFACE, &S.mcu_ep_in, &S.mcu_ep_out);
        LOGI("MCU interface endpoints: in=0x%02x out=0x%02x",
             S.mcu_ep_in, S.mcu_ep_out);
    }
    uint8_t pkt[XR_MCU_CMD_LEN];
    xr_mcu_command(pkt, 0x08, &mode, 1);
    int rc;
    if (S.mcu_ep_out) {
        int sent = 0;
        rc = libusb_interrupt_transfer(S.usb, S.mcu_ep_out, pkt,
                                       XR_MCU_CMD_LEN, &sent, 1000);
    } else {
        rc = libusb_control_transfer(S.usb, 0x21, 0x09, 0x0200,
                                     XR_MCU_INTERFACE, pkt, XR_MCU_CMD_LEN, 1000);
    }
    if (rc < 0) { LOGE("display mode %d: send failed rc=%d", mode, rc); return -1; }
    /* read the ack (cmd echo; first data byte 0 = success) */
    if (S.mcu_ep_in) {
        uint8_t buf[XR_MCU_CMD_LEN];
        for (int tries = 0; tries < 20; tries++) {
            int got = 0;
            if (libusb_interrupt_transfer(S.usb, S.mcu_ep_in, buf, sizeof buf,
                                          &got, 300) != 0 || got < 23)
                break;
            if (buf[0] == 0xFD && buf[15] == 0x08 && buf[16] == 0x00) {
                LOGI("display mode %d: %s", mode,
                     buf[22] == 0 ? "ok" : "rejected");
                return buf[22] == 0 ? 0 : -1;
            }
        }
    }
    LOGI("display mode %d: sent (no ack read)", mode);
    return 1;
}

/* Enter stereo. SBS 60 Hz is this hardware's limit (mode 9 / 90 Hz never
 * engages on-device), so go straight there - one clean renegotiation. */
static void mcu_enter_stereo(void) {
    mcu_set_display_mode(XR_DISPLAY_SBS_60);
}

static int imu_send_cmd(uint8_t cmd, const uint8_t *data, size_t n) {
    uint8_t pkt[XR_IMU_CMD_LEN];
    xr_imu_command(pkt, cmd, data, n);
    if (S.imu_ep_out) {
        int sent = 0;
        return libusb_interrupt_transfer(S.usb, S.imu_ep_out, pkt,
                                         XR_IMU_CMD_LEN, &sent, 1000);
    }
    return libusb_control_transfer(S.usb, 0x21, 0x09, 0x0200, XR_IMU_INTERFACE,
                                   pkt, XR_IMU_CMD_LEN, 1000);
}

/* One synchronous command/reply on the IMU channel (stream must be quiet or
 * merely interleaved - stream reports are skipped). Returns the payload
 * length, or -1. */
static int imu_cmd_sync(uint8_t cmd, const uint8_t *data, size_t n,
                        uint8_t *reply, size_t cap) {
    if (imu_send_cmd(cmd, data, n) < 0) return -1;
    uint8_t buf[XR_IMU_REPORT];
    for (int tries = 0; tries < 100; tries++) {
        int got = 0;
        int rc = libusb_interrupt_transfer(S.usb, S.imu_ep_in, buf, sizeof buf,
                                           &got, 300);
        if (rc == LIBUSB_ERROR_TIMEOUT) return -1;
        if (rc < 0 || got < 8) continue;
        if (buf[0] != 0xAA || buf[7] != cmd) continue;   /* stream/other */
        int len = (buf[5] | buf[6] << 8) - 3;
        if (len < 0) len = 0;
        if ((size_t)len > cap) len = (int)cap;
        if (len > got - 8) len = got - 8;
        memcpy(reply, buf + 8, (size_t)len);
        return len;
    }
    return -1;
}

/* Fetch the factory calibration JSON (cmds 0x14/0x15) while the stream is
 * off; it powers the world-aligned passthrough. Failure is non-fatal. */
static void imu_fetch_config(void) {
    free(S.config);
    S.config = NULL;
    S.config_len = 0;
    uint8_t r[XR_IMU_REPORT];
    if (imu_cmd_sync(0x14, NULL, 0, r, sizeof r) < 4) {
        LOGE("calibration: length query failed (aligned passthrough off)");
        return;
    }
    uint32_t total = (uint32_t)r[0] | (uint32_t)r[1] << 8 |
                     (uint32_t)r[2] << 16 | (uint32_t)r[3] << 24;
    if (total == 0 || total > 1 << 20) return;
    uint8_t *cfg = malloc(total);
    uint32_t got = 0;
    while (got < total) {
        int pl = imu_cmd_sync(0x15, NULL, 0, r, sizeof r);
        if (pl <= 0) { free(cfg); LOGE("calibration: read stalled at %u/%u",
                                       got, total); return; }
        if ((uint32_t)pl > total - got) pl = (int)(total - got);
        memcpy(cfg + got, r, (size_t)pl);
        got += (uint32_t)pl;
    }
    S.config = cfg;
    S.config_len = total;
    LOGI("calibration fetched: %u bytes", total);
}

static int imu_enable(void) {
    uint8_t cmd[XR_IMU_CMD_LEN], on = 1;
    xr_imu_command(cmd, 0x19, &on, 1);
    int rc;
    if (S.imu_ep_out) {
        /* prefer the interrupt OUT pipe — it is what OS HID drivers use, and
         * hammering ep0 with SET_REPORT wedged the whole device on a phone */
        int sent = 0;
        rc = libusb_interrupt_transfer(S.usb, S.imu_ep_out, cmd, XR_IMU_CMD_LEN,
                                       &sent, 1000);
        LOGI("IMU enable via interrupt-out 0x%02x: rc=%d sent=%d",
             S.imu_ep_out, rc, sent);
    } else {
        rc = libusb_control_transfer(S.usb, 0x21, 0x09, 0x0200, XR_IMU_INTERFACE,
                                     cmd, XR_IMU_CMD_LEN, 1000);
        LOGI("IMU enable via SET_REPORT: rc=%d", rc);
    }
    return rc;
}

static void *imu_worker(void *arg) {
    (void)arg;
    uint8_t buf[XR_IMU_REPORT];
    int stalls = 0, reenabled = 0;
    while (atomic_load(&S.imu_running)) {
        int got = 0;
        int rc = libusb_interrupt_transfer(S.usb, S.imu_ep_in, buf, sizeof buf,
                                           &got, 500);
        if (rc == LIBUSB_ERROR_TIMEOUT || (rc == 0 && got == 0)) {
            stalls++;
            /* one cautious re-enable, then give up quietly — never risk
             * disturbing the camera stream with repeated commands */
            if (stalls == 3 && !reenabled) { reenabled = 1; imu_enable(); }
            if (stalls >= 10) {
                LOGE("IMU stream silent, giving up (camera unaffected)");
                atomic_store(&S.imu_running, 0);
                return NULL;
            }
            continue;
        }
        if (rc < 0) { LOGE("IMU transfer failed: %d", rc); break; }
        xr_imu_sample s;
        if (xr_imu_parse(buf, (size_t)got, &s) != 0) continue;
        stalls = 0;

        /* forensics: average the RAW (pre-remap) gyro over the first second
         * at rest. Expected on this unit: about (-0.94, -0.55, -0.01) deg/s
         * = the factory bias (0.85, -0.12, 0.53) pulled back through the
         * chip->factory map below (x/z temperature-dependent). */
        if (S.raw_gbias_n < 1000) {
            for (int i = 0; i < 3; i++) S.raw_gbias[i] += s.gyro_dps[i];
            if (++S.raw_gbias_n == 1000)
                LOGI("raw gyro bias (chip frame, deg/s): (%+.3f, %+.3f, %+.3f)",
                     S.raw_gbias[0] / 1000.0, S.raw_gbias[1] / 1000.0,
                     S.raw_gbias[2] / 1000.0);
        }

        /* Raw chip frame -> factory calibration frame.
         *
         * Chip frame, pinned by physical ground truth (xreal_imu_axistest:
         * yaw-CCW -> z+, front-up -> x-, right-side-down -> y-; rest accel
         * +1 g on z; raw bias (-0.94,-0.55,-0.01) deg/s matches the factory
         * file through this map): x = LEFT, y = BACK, z = UP, right-handed,
         * gyro and accel healthy and mutually consistent.
         *
         * The factory frame (imu_q_cam / imu_p_cam / biases, ~= the display
         * frame) is x = right, y = down, z = forward, so ONE proper
         * rotation maps BOTH sensors:
         *
         *     v_f = ( -vx, -vz, -vy )
         *
         * (The earlier per-sensor sign hacks came from misreading the pose
         * view: watching the frustum from its FRONT flips apparent yaw and
         * roll while pitch stays invariant — hence the axis labels in
         * PoseMapView.) */
        {
            float t;
            s.gyro_dps[0] = -s.gyro_dps[0];
            t = s.gyro_dps[1];
            s.gyro_dps[1] = -s.gyro_dps[2];
            s.gyro_dps[2] = -t;
            s.accel_g[0] = -s.accel_g[0];
            t = s.accel_g[1];
            s.accel_g[1] = -s.accel_g[2];
            s.accel_g[2] = -t;
        }

        int has_q = xr_ahrs_feed(&S.ahrs, &s);
        xr_slam_push_imu(s.ts_ns, s.gyro_dps, s.accel_g);

        S.imu_count++;
        int64_t t = now_ms();
        if (S.imu_t0 == 0) S.imu_t0 = t;
        if (t - S.imu_t0 >= 1000) {
            S.imu_rate = (int)(S.imu_count * 1000 / (t - S.imu_t0));
            S.imu_count = 0;
            S.imu_t0 = t;
        }

        pthread_mutex_lock(&S.imu_lock);
        S.imu_latest = s;
        if (has_q) {
            memcpy(S.imu_q, S.ahrs.q, sizeof S.imu_q);
            uint32_t qi = S.qhist_n % 256;
            S.qhist[qi].ts = s.ts_ns;
            memcpy(S.qhist[qi].q, S.ahrs.q, sizeof S.qhist[qi].q);
            S.qhist_n++;
        }
        S.imu_has_q = has_q;
        S.imu_seq++;
        S.ring[S.ring_head % 1024] = s;
        S.ring_head++;
        if (S.ring_head - S.ring_tail > 1024)
            S.ring_tail = S.ring_head - 1024;   /* overflow: drop oldest */
        pthread_mutex_unlock(&S.imu_lock);
    }
    return NULL;
}

static int imu_started;   /* thread created + interface claimed this session */

static void imu_start(void) {
    S.usb = uvc_get_libusb_handle(S.devh);
    S.imu_ep_in = S.imu_ep_out = 0;
    imu_started = 0;
    S.raw_gbias[0] = S.raw_gbias[1] = S.raw_gbias[2] = 0.0;
    S.raw_gbias_n = 0;
    libusb_set_auto_detach_kernel_driver(S.usb, 1);
    int rc = libusb_claim_interface(S.usb, XR_IMU_INTERFACE);
    if (rc != 0) {
        LOGE("IMU: could not claim interface %d: rc=%d (camera unaffected)",
             XR_IMU_INTERFACE, rc);
        return;
    }
    find_interface_eps(XR_IMU_INTERFACE, &S.imu_ep_in, &S.imu_ep_out);
    LOGI("IMU interface %d endpoints: in=0x%02x out=0x%02x",
         XR_IMU_INTERFACE, S.imu_ep_in, S.imu_ep_out);
    if (!S.imu_ep_in) {
        LOGE("IMU: no interrupt-in endpoint on interface %d", XR_IMU_INTERFACE);
        libusb_release_interface(S.usb, XR_IMU_INTERFACE);
        return;
    }
    /* quiet the channel, pull the factory calibration, then start streaming */
    uint8_t off = 0;
    imu_cmd_sync(0x19, &off, 1, (uint8_t[8]){0}, 8);
    imu_fetch_config();

    xr_ahrs_init(&S.ahrs);
    S.imu_seq = S.imu_grabbed = 0;
    S.imu_has_q = 0;
    S.imu_count = 0; S.imu_t0 = 0; S.imu_rate = 0;
    S.ring_head = S.ring_tail = 0;
    S.qhist_n = 0;
    imu_enable();
    atomic_store(&S.imu_running, 1);
    pthread_create(&S.imu_thread, NULL, imu_worker, NULL);
    imu_started = 1;
}

static void imu_stop(void) {
    if (imu_started) {
        atomic_store(&S.imu_running, 0);
        pthread_join(S.imu_thread, NULL);   /* fine if the worker already left */
        libusb_release_interface(S.usb, XR_IMU_INTERFACE);
        imu_started = 0;
    }
    S.usb = NULL;
}

JNIEXPORT jint JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeStart(JNIEnv *env, jclass cls, jint fd) {
    (void)env; (void)cls;
    if (S.streaming) return 0;
    xr_init();
    S.order = XR_ORDER_UNKNOWN;
    S.have[0] = S.have[1] = 0;
    S.seq = S.grabbed = 0;
    S.fps_count = 0; S.fps_t0 = 0; S.fps_x10 = 0;
    xr_cleaner_reset(&S.cleaners[0]);
    xr_cleaner_reset(&S.cleaners[1]);

    /* Android blocks usbfs device discovery; the fd comes from Java instead */
    libusb_set_option(NULL, LIBUSB_OPTION_NO_DEVICE_DISCOVERY, NULL);

    uvc_error_t res = uvc_init(&S.ctx, NULL);
    if (res < 0) { LOGE("uvc_init: %s", uvc_strerror(res)); return res; }

    res = uvc_wrap(fd, S.ctx, &S.devh);
    if (res < 0) {
        LOGE("uvc_wrap: %s", uvc_strerror(res));
        uvc_exit(S.ctx); S.ctx = NULL;
        return res;
    }

    /* per-eye stereo: without this the glasses mirror the whole frame into
     * both eyes and no side-by-side content can fuse */
    S.usb = uvc_get_libusb_handle(S.devh);
    libusb_set_auto_detach_kernel_driver(S.usb, 1);
    if (atomic_load(&S.stereo_mode)) mcu_enter_stereo();
    else mcu_set_display_mode(XR_DISPLAY_MIRROR);

    /* the stream advertises itself as 640x241 "YUY2"; trust the descriptor
     * rather than hardcoding, and accept any mode of the right byte size */
    int w = 640, h = 241, fps = 60;
    for (const uvc_format_desc_t *fmt = uvc_get_format_descs(S.devh);
         fmt; fmt = fmt->next) {
        if (fmt->bDescriptorSubtype != UVC_VS_FORMAT_UNCOMPRESSED) continue;
        for (const struct uvc_frame_desc *fr = fmt->frame_descs; fr; fr = fr->next) {
            if ((int)fr->wWidth * fr->wHeight * 2 == XR_FRAME_BYTES) {
                w = fr->wWidth;
                h = fr->wHeight;
                if (fr->dwDefaultFrameInterval > 0)
                    fps = (int)(10000000 / fr->dwDefaultFrameInterval);
            }
        }
    }
    LOGI("negotiating %dx%d @ %d fps", w, h, fps);

    uvc_stream_ctrl_t ctrl;
    res = uvc_get_stream_ctrl_format_size(S.devh, &ctrl, UVC_FRAME_FORMAT_ANY, w, h, fps);
    if (res < 0) {
        LOGE("get_stream_ctrl: %s", uvc_strerror(res));
    } else {
        res = uvc_start_streaming(S.devh, &ctrl, frame_cb, NULL, 0);
        if (res < 0) LOGE("start_streaming: %s", uvc_strerror(res));
    }
    if (res < 0) {
        uvc_close(S.devh); S.devh = NULL;
        uvc_exit(S.ctx); S.ctx = NULL;
        return res;
    }
    S.streaming = 1;
    LOGI("streaming started");
    imu_start();   /* best effort — camera works without it */
    xr_gles_set_pose_fn(pose_delta);   /* timewarp pose source */
    slam_start();
    return 0;
}

/* Enable/disable the IMU timewarp on the glasses renderer (default on). */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetTimewarp(JNIEnv *env, jclass cls,
                                                           jboolean on) {
    (void)env; (void)cls;
    xr_gles_set_timewarp(on ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeStop(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    if (!S.streaming) return;
    S.streaming = 0;
    slam_stop();   /* before the IMU: it reads the pose history */
    if (S.mcu_claimed) {
        mcu_set_display_mode(XR_DISPLAY_MIRROR);   /* leave the glasses as found */
        libusb_release_interface(S.usb, XR_MCU_INTERFACE);
        S.mcu_claimed = 0;
    }
    imu_stop();
    uvc_stop_streaming(S.devh);
    xr_slam_stop();   /* after both producers (UVC + IMU threads) are gone */
    S.vio_fresh = 0;
    uvc_close(S.devh); S.devh = NULL;
    uvc_exit(S.ctx); S.ctx = NULL;
    LOGI("streaming stopped");
}

/* Copy the newest IMU state into `buf` (a direct ByteBuffer, >= 56 bytes,
 * native byte order): u64 ts_ns | f32 gyro_dps[3] | f32 accel_g[3] |
 * f32 quat_wxyz[4] | f32 rate_hz | u32 has_quat. Returns JNI_FALSE when
 * nothing new arrived since the last call. */
JNIEXPORT jboolean JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeGrabImu(JNIEnv *env, jclass cls,
                                                       jobject buf) {
    (void)cls;
    uint8_t *dst = (*env)->GetDirectBufferAddress(env, buf);
    if (!dst || (*env)->GetDirectBufferCapacity(env, buf) < 56) return JNI_FALSE;
    jboolean fresh = JNI_FALSE;
    pthread_mutex_lock(&S.imu_lock);
    if (S.imu_seq != S.imu_grabbed) {
        S.imu_grabbed = S.imu_seq;
        float rate = S.imu_rate;
        uint32_t has_q = (uint32_t)S.imu_has_q;
        memcpy(dst, &S.imu_latest.ts_ns, 8);
        memcpy(dst + 8, S.imu_latest.gyro_dps, 12);
        memcpy(dst + 20, S.imu_latest.accel_g, 12);
        memcpy(dst + 32, S.imu_q, 16);
        memcpy(dst + 48, &rate, 4);
        memcpy(dst + 52, &has_q, 4);
        fresh = JNI_TRUE;
    }
    pthread_mutex_unlock(&S.imu_lock);
    return fresh;
}

/* Drain pending IMU samples into `buf` (direct ByteBuffer, native order),
 * 32 bytes each: u64 ts_ns | f32 gyro_dps[3] | f32 accel_g[3].
 * Returns the number of samples written (0 if none pending). Lets the UI
 * pull the full 1 kHz stream in ~33 ms batches for graphing. */
JNIEXPORT jint JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeGrabImuBatch(JNIEnv *env, jclass cls,
                                                            jobject buf) {
    (void)cls;
    uint8_t *dst = (*env)->GetDirectBufferAddress(env, buf);
    jlong cap = (*env)->GetDirectBufferCapacity(env, buf);
    if (!dst || cap < 32) return 0;
    jint n = 0;
    pthread_mutex_lock(&S.imu_lock);
    while (S.ring_tail != S.ring_head && (jlong)(n + 1) * 32 <= cap) {
        const xr_imu_sample *s = &S.ring[S.ring_tail % 1024];
        memcpy(dst, &s->ts_ns, 8);
        memcpy(dst + 8, s->gyro_dps, 12);
        memcpy(dst + 20, s->accel_g, 12);
        dst += 32;
        S.ring_tail++;
        n++;
    }
    pthread_mutex_unlock(&S.imu_lock);
    return n;
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetSwap(JNIEnv *env, jclass cls,
                                                       jboolean swap) {
    (void)env; (void)cls;
    atomic_store(&S.swap_eyes, swap ? 1 : 0);
}

/* Reset the SLAM system: Basalt state (pose back to origin), the
 * accumulated landmark map, and the fallback tracker. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSlamReset(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    xr_slam_reset();
    xr_map_reset();
    pthread_mutex_lock(&S.lock);
    xr_track_reset(&S.track);      /* worker touches it only between locks */
    S.pts_n = 0;
    S.pts_n_r = 0;
    S.track_count = 0;
    S.vio_fresh = 0;
    memset(MAP_PT, 0, sizeof MAP_PT);
    S.map_n = 0;
    S.map_stamp = 0;
    pthread_mutex_unlock(&S.lock);
    S.slam_prev_ts = 0;
    xr_gles_set_points(NULL, 0, 0);
    xr_gles_set_map(NULL, 0, NULL, NULL, 0);
    LOGI("SLAM reset");
}

/* Copy the accumulated landmark map into `buf` (direct ByteBuffer, native
 * order): u32 count, then count x 3 f32 world xyz. Returns the count. */
JNIEXPORT jint JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeGrabMap(JNIEnv *env, jclass cls,
                                                       jobject buf) {
    (void)cls;
    uint8_t *dst = (*env)->GetDirectBufferAddress(env, buf);
    jlong cap = (*env)->GetDirectBufferCapacity(env, buf);
    if (!dst || cap < 4) return 0;
    int max = (int)((cap - 4) / 12);
    int n = 0;
    pthread_mutex_lock(&S.lock);
    for (int i = 0; i < 4096 && n < max; i++) {
        /* one- and two-shot landmarks are mostly bad triangulations */
        if (!MAP_PT[i].stamp || MAP_PT[i].seen < 3) continue;
        memcpy(dst + 4 + (size_t)n * 12, &MAP_PT[i].x, 12);
        n++;
    }
    pthread_mutex_unlock(&S.lock);
    uint32_t un = (uint32_t)n;
    memcpy(dst, &un, 4);
    return n;
}

/* Show/hide the tracked features (phone pane dots + glasses overlay). */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetShowPoints(JNIEnv *env, jclass cls,
                                                             jboolean on) {
    (void)env; (void)cls;
    atomic_store(&S.show_pts, on ? 1 : 0);
    xr_gles_set_show_points(on ? 1 : 0);
    if (!on) xr_gles_set_points(NULL, 0, 0);
}

/* Enable/disable stereo depth computation (the tracker keeps running). */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetDepth(JNIEnv *env, jclass cls,
                                                        jboolean on) {
    (void)env; (void)cls;
    atomic_store(&S.depth_on, on ? 1 : 0);
}

/* Glasses eye-view mode: 0 = camera passthrough, 1 = depth passthrough
 * (world-aligned per eye), 2 = AR (tracked points only), 3 = off. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetEyeMode(JNIEnv *env, jclass cls,
                                                          jint mode) {
    (void)env; (void)cls;
    xr_gles_set_eye_mode((int)mode);
    if (mode == XR_EYE_DEPTH) atomic_store(&S.depth_on, 1);
}

/* Phone pane layout: 0 = left camera | depth, 1 = left | right cameras. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetPaneMode(JNIEnv *env, jclass cls,
                                                           jint mode) {
    (void)env; (void)cls;
    atomic_store(&S.pane_mode, mode & 1);
}

/* Path of the unified Basalt config (written by the app into its files
 * dir; sets num-threads and the VioConfig). Call before streaming starts. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetSlamConfig(JNIEnv *env, jclass cls,
                                                             jstring path) {
    (void)cls;
    const char *p = path ? (*env)->GetStringUTFChars(env, path, NULL) : NULL;
    xr_slam_set_config(p);
    if (p) (*env)->ReleaseStringUTFChars(env, path, p);
}

/* Path of the staged XFeat ONNX model for session-map descriptors. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetXfeatModel(JNIEnv *env, jclass cls,
                                                             jstring path) {
    (void)cls;
    const char *p = path ? (*env)->GetStringUTFChars(env, path, NULL) : NULL;
    xr_map_set_model(p);
    if (p) (*env)->ReleaseStringUTFChars(env, path, p);
}

/* Copy the newest pose/SLAM state into `buf` (direct ByteBuffer, >= 40 bytes,
 * native order): f32 quat_wxyz[4] | f32 pos_m[3] | i32 tracked | f32 depth_ms
 * | u32 flags (bit0 depth on, bit1 rectification ready, bit2 orientation
 * valid, bit3 Basalt VIO live). With Basalt live the full 6-DoF pose comes
 * from the VIO; otherwise the AHRS supplies orientation only.
 * Returns JNI_FALSE when no orientation exists yet. */
JNIEXPORT jboolean JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeGrabPose(JNIEnv *env, jclass cls,
                                                        jobject buf) {
    (void)cls;
    uint8_t *dst = (*env)->GetDirectBufferAddress(env, buf);
    if (!dst || (*env)->GetDirectBufferCapacity(env, buf) < 40) return JNI_FALSE;
    float q[4] = { 1, 0, 0, 0 }, p[3] = { 0, 0, 0 };
    int has_q = 0, basalt = 0;

    pthread_mutex_lock(&S.lock);
    int32_t tracked = S.track_count;
    int32_t tracked_r = S.pts_n_r;
    float depth_ms = S.depth_ms;
    if (S.vio_fresh && xr_slam_running()) {
        memcpy(q, S.vio.q, sizeof q);
        memcpy(p, S.vio.p, sizeof p);
        has_q = 1;
        basalt = 1;
    }
    pthread_mutex_unlock(&S.lock);

    if (!basalt) {
        pthread_mutex_lock(&S.imu_lock);
        has_q = S.imu_has_q;
        if (has_q) memcpy(q, S.imu_q, sizeof q);
        pthread_mutex_unlock(&S.imu_lock);
    }
    uint32_t flags = (atomic_load(&S.depth_on) ? 1u : 0u) |
                     (S.stereo_ready ? 2u : 0u) |
                     (has_q ? 4u : 0u) |
                     (basalt ? 8u : 0u);
    memcpy(dst, q, 16);
    memcpy(dst + 16, p, 12);
    memcpy(dst + 28, &tracked, 4);
    memcpy(dst + 32, &depth_ms, 4);
    memcpy(dst + 36, &flags, 4);
    if ((*env)->GetDirectBufferCapacity(env, buf) >= 48) {
        int32_t kf = xr_map_num_keyframes();
        memcpy(dst + 40, &kf, 4);
        memcpy(dst + 44, &tracked_r, 4);   /* right-camera observations */
    }
    return has_q ? JNI_TRUE : JNI_FALSE;
}

/* The factory calibration JSON fetched at start, or null. */
JNIEXPORT jbyteArray JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeGetConfig(JNIEnv *env, jclass cls) {
    (void)cls;
    if (!S.config || !S.config_len) return NULL;
    jbyteArray out = (*env)->NewByteArray(env, (jsize)S.config_len);
    if (out)
        (*env)->SetByteArrayRegion(env, out, 0, (jsize)S.config_len,
                                   (const jbyte *)S.config);
    return out;
}

/* 82 floats: per eye (left, then right): K[9], q_display[4], q_cam[4],
 * fc[2], cc[2], kc[12], p_cam[3] (= 72), then gyro_bias[3], accel_bias[3],
 * imu_noises[4]. Enables the world-aligned passthrough, the stereo
 * rectification (p_cam gives the baseline) and the Basalt VIO. Shorter
 * arrays (66/72 floats) still enable the earlier subsets. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetAlignment(JNIEnv *env, jclass cls,
                                                            jfloatArray arr) {
    (void)cls;
    jsize len = (*env)->GetArrayLength(env, arr);
    if (len < 66) return;
    int stride = len >= 72 ? 36 : 33;
    float f[82];
    (*env)->GetFloatArrayRegion(env, arr, 0, len >= 82 ? 82 : stride * 2, f);
    if (len >= 82) {
        memcpy(S.imu_gyro_bias, f + 72, 3 * sizeof(float));
        memcpy(S.imu_accel_bias, f + 75, 3 * sizeof(float));
        memcpy(S.imu_noises, f + 78, 4 * sizeof(float));
        S.imu_calib_have = 1;
    }
    for (int e = 0; e < 2; e++) {
        const float *p = f + e * stride;
        xr_eye_calib *c = &S.eye_calib[e];
        memcpy(c->K, p, 9 * sizeof(float));
        memcpy(c->q_disp, p + 9, 4 * sizeof(float));
        memcpy(c->q_cam, p + 13, 4 * sizeof(float));
        memcpy(c->fc, p + 17, 2 * sizeof(float));
        memcpy(c->cc, p + 19, 2 * sizeof(float));
        memcpy(c->kc, p + 21, 12 * sizeof(float));
        if (stride == 36) memcpy(c->p_cam, p + 33, 3 * sizeof(float));
        else memset(c->p_cam, 0, sizeof c->p_cam);
    }
    xr_gles_set_alignment(S.eye_calib, atomic_load(&S.align_variant));
    atomic_store(&S.align_have, 1);
    LOGI("alignment calibration set (%d floats)", (int)len);

    /* with camera positions available, bring up the Basalt backend */
#if XR_ENABLE_BASALT
    float bx = S.eye_calib[1].p_cam[0] - S.eye_calib[0].p_cam[0];
    float by = S.eye_calib[1].p_cam[1] - S.eye_calib[0].p_cam[1];
    float bz = S.eye_calib[1].p_cam[2] - S.eye_calib[0].p_cam[2];
    if (bx * bx + by * by + bz * bz > 1e-4f)
        xr_slam_start(&S.eye_calib[0], &S.eye_calib[1],
                      atomic_load(&S.align_variant),
                      S.imu_calib_have ? S.imu_gyro_bias : NULL,
                      S.imu_calib_have ? S.imu_accel_bias : NULL,
                      S.imu_calib_have ? S.imu_noises : NULL);
#endif
}

/* Toggle the glasses between per-eye stereo with the calibrated aligned warp
 * (true) and plain mirror mode showing the raw framebuffer (false). Switches
 * the display mode over the MCU channel when streaming. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetStereoMode(JNIEnv *env, jclass cls,
                                                             jboolean stereo) {
    (void)env; (void)cls;
    atomic_store(&S.stereo_mode, stereo ? 1 : 0);
    if (S.streaming && S.usb) {
        if (stereo) mcu_enter_stereo();
        else mcu_set_display_mode(XR_DISPLAY_MIRROR);
    }
}

/* Set the rotation-convention variant (0..3); the verified default is 2. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetAlignVariant(JNIEnv *env, jclass cls,
                                                               jint variant) {
    (void)env; (void)cls;
    atomic_store(&S.align_variant, variant & 3);
    if (atomic_load(&S.align_have))
        xr_gles_set_alignment(S.eye_calib, variant & 3);
}

/* Attach/detach the glasses' Surface; the GLES render thread takes it from
 * here. Pass null before the surface is destroyed. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetSurface(JNIEnv *env, jclass cls,
                                                          jobject surface) {
    (void)cls;
    ANativeWindow *nw = surface ? ANativeWindow_fromSurface(env, surface) : NULL;
    xr_gles_set_window(nw);
    LOGI("passthrough surface %s", nw ? "attached" : "detached");
}

/* Copy the latest composed RGBA frame into `buf` (a direct ByteBuffer of at
 * least MAX_W*MAX_H*4 bytes). Returns 0 if nothing new since the last call,
 * else (width << 48) | (height << 32) | (fps*10 << 16) | pair counter. */
JNIEXPORT jlong JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeGrabFrame(JNIEnv *env, jclass cls,
                                                         jobject buf) {
    (void)cls;
    uint8_t *dst = (*env)->GetDirectBufferAddress(env, buf);
    jlong cap = (*env)->GetDirectBufferCapacity(env, buf);
    if (!dst) return 0;
    jlong ret = 0;
    pthread_mutex_lock(&S.lock);
    if (S.seq != S.grabbed && S.cw && cap >= (jlong)S.cw * S.ch * 4) {
        memcpy(dst, S.rgba, (size_t)S.cw * S.ch * 4);
        S.grabbed = S.seq;
        ret = ((jlong)S.cw << 48) | ((jlong)S.ch << 32) |
              ((jlong)(S.fps_x10 & 0xFFFF) << 16) | (S.counter & 0xFF);
    }
    pthread_mutex_unlock(&S.lock);
    return ret;
}
