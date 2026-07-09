/* xreal_jni.c — Android glue: wrap the USB fd from UsbDeviceConnection with
 * libusb/libuvc, descramble + clean each frame on the UVC callback thread,
 * and hand composed RGBA stereo frames to Kotlin on demand.
 *
 * Threading model: the UVC callback thread owns all image processing and
 * writes the composed frame under a mutex; the Kotlin UI thread polls
 * nativeGrabFrame() which copies it out under the same mutex.
 */
#include <jni.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <android/log.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <libusb.h>
#include <libuvc/libuvc.h>

#include "xreal_align.h"
#include "xreal_core.h"
#include "xreal_gles.h"
#include "xreal_imu.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define MAX_W (2 * XR_W)     /* scrambled view: 1280x480 */
#define MAX_H XR_OH          /* clean view:      960x640 */

static struct {
    uvc_context_t *ctx;
    uvc_device_handle_t *devh;
    int streaming;

    xr_order order;
    xr_cleaner cleaners[2];
    uint8_t frame[XR_FRAME_BYTES];          /* working copy of the UVC frame */
    uint8_t clean[2][XR_OW * XR_OH];
    uint8_t raweq[2][XR_W * XR_H_IMG];
    int have[2];

    atomic_int show_clean;

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
    atomic_int align_have;            /* calibration params received */
    atomic_int align_variant;

    int fps_count;
    int64_t fps_t0;
    int fps_x10;

    atomic_int mirror;          /* horizontal per-eye flip of the composed view */
    atomic_int swap_eyes;       /* swap the two panes (debug) */
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
} S = { .show_clean = 1, .lock = PTHREAD_MUTEX_INITIALIZER,
        .imu_lock = PTHREAD_MUTEX_INITIALIZER,
        .align_variant = XR_ALIGN_VARIANT_DEFAULT,
        .stereo_mode = 1 };

static int64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void compose(int counter) {
    int clean = atomic_load(&S.show_clean);
    int mirror = atomic_load(&S.mirror);
    int swap = atomic_load(&S.swap_eyes);
    int sw = clean ? XR_OW : XR_W;
    int sh = clean ? XR_OH : XR_H_IMG;
    pthread_mutex_lock(&S.lock);
    S.cw = 2 * sw;
    S.ch = sh;
    S.counter = counter;
    for (int cam = 0; cam < 2; cam++) {
        /* cam1 is the physical LEFT camera (verified on-device): left pane */
        int pane = (cam == 1) ? 0 : 1;
        if (swap) pane = 1 - pane;
        const uint8_t *src = clean ? S.clean[cam] : S.raweq[cam];
        for (int y = 0; y < sh; y++) {
            uint8_t *dst = S.rgba + (y * S.cw + pane * sw) * 4;
            const uint8_t *s = src + y * sw;
            for (int x = 0; x < sw; x++) {
                uint8_t v = s[mirror ? sw - 1 - x : x];
                dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = 0xFF;
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
        xr_gles_submit_eyes(S.clean[swap ? 0 : 1], S.clean[swap ? 1 : 0]);
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
    xr_clean(&S.cleaners[cam], dscr, S.clean[cam]);
    xr_equalize(S.frame, S.raweq[cam], XR_W * XR_H_IMG);
    S.have[cam] = 1;

    S.fps_count++;
    int64_t t = now_ms();
    if (S.fps_t0 == 0) S.fps_t0 = t;
    if (t - S.fps_t0 >= 1000) {
        S.fps_x10 = (int)(S.fps_count * 10000 / (t - S.fps_t0));
        S.fps_count = 0;
        S.fps_t0 = t;
    }

    if (S.have[0] && S.have[1]) compose(xr_counter(S.frame));
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

/* Enter stereo at the highest refresh the glasses accept (less vsync wait
 * = lower passthrough latency), falling back to SBS 60 Hz. */
static void mcu_enter_stereo(void) {
    if (mcu_set_display_mode(XR_DISPLAY_SBS_90) < 0)
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
        int has_q = xr_ahrs_feed(&S.ahrs, &s);

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
        if (has_q) memcpy(S.imu_q, S.ahrs.q, sizeof S.imu_q);
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
    return 0;
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeStop(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    if (!S.streaming) return;
    S.streaming = 0;
    if (S.mcu_claimed) {
        mcu_set_display_mode(XR_DISPLAY_MIRROR);   /* leave the glasses as found */
        libusb_release_interface(S.usb, XR_MCU_INTERFACE);
        S.mcu_claimed = 0;
    }
    imu_stop();
    uvc_stop_streaming(S.devh);
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
Java_org_air2ultra_stereocam_XrealNative_nativeSetClean(JNIEnv *env, jclass cls,
                                                        jboolean clean) {
    (void)env; (void)cls;
    atomic_store(&S.show_clean, clean ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetMirror(JNIEnv *env, jclass cls,
                                                         jboolean mirror) {
    (void)env; (void)cls;
    atomic_store(&S.mirror, mirror ? 1 : 0);
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetSwap(JNIEnv *env, jclass cls,
                                                       jboolean swap) {
    (void)env; (void)cls;
    atomic_store(&S.swap_eyes, swap ? 1 : 0);
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

/* 66 floats: per eye (left, then right): K[9], q_display[4], q_cam[4],
 * fc[2], cc[2], kc[12]. Enables the world-aligned passthrough. */
JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetAlignment(JNIEnv *env, jclass cls,
                                                            jfloatArray arr) {
    (void)cls;
    if ((*env)->GetArrayLength(env, arr) < 66) return;
    float f[66];
    (*env)->GetFloatArrayRegion(env, arr, 0, 66, f);
    for (int e = 0; e < 2; e++) {
        const float *p = f + e * 33;
        xr_eye_calib *c = &S.eye_calib[e];
        memcpy(c->K, p, 9 * sizeof(float));
        memcpy(c->q_disp, p + 9, 4 * sizeof(float));
        memcpy(c->q_cam, p + 13, 4 * sizeof(float));
        memcpy(c->fc, p + 17, 2 * sizeof(float));
        memcpy(c->cc, p + 19, 2 * sizeof(float));
        memcpy(c->kc, p + 21, 12 * sizeof(float));
    }
    xr_gles_set_alignment(S.eye_calib, atomic_load(&S.align_variant));
    atomic_store(&S.align_have, 1);
    LOGI("alignment calibration set");
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
