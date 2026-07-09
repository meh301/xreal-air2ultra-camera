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
#include <libusb.h>
#include <libuvc/libuvc.h>

#include "xreal_core.h"

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
    xr_dialect dialect;
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

    int fps_count;
    int64_t fps_t0;
    int fps_x10;
} S = { .show_clean = 1, .lock = PTHREAD_MUTEX_INITIALIZER };

static int64_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void compose(int counter) {
    int clean = atomic_load(&S.show_clean);
    int sw = clean ? XR_OW : XR_W;
    int sh = clean ? XR_OH : XR_H_IMG;
    pthread_mutex_lock(&S.lock);
    S.cw = 2 * sw;
    S.ch = sh;
    S.counter = counter;
    for (int cam = 0; cam < 2; cam++) {
        const uint8_t *src = clean ? S.clean[cam] : S.raweq[cam];
        for (int y = 0; y < sh; y++) {
            uint8_t *dst = S.rgba + (y * S.cw + cam * sw) * 4;
            const uint8_t *s = src + y * sw;
            for (int x = 0; x < sw; x++) {
                uint8_t v = s[x];
                dst[0] = v; dst[1] = v; dst[2] = v; dst[3] = 0xFF;
                dst += 4;
            }
        }
    }
    S.seq++;
    pthread_mutex_unlock(&S.lock);
}

static void frame_cb(uvc_frame_t *frame, void *user) {
    (void)user;
    if (frame->data_bytes < XR_FRAME_BYTES) return;   /* startup runts */
    memcpy(S.frame, frame->data, XR_FRAME_BYTES);

    if (S.order == XR_ORDER_UNKNOWN) {
        S.order = xr_classify(S.frame, &S.dialect);
        if (S.order == XR_ORDER_UNKNOWN) return;      /* black startup frame */
        LOGI("stream fingerprinted: dialect %c, byte order %s",
             S.dialect == XR_DIALECT_B ? 'B' : 'A',
             S.order == XR_ORDER_SWAPPED ? "swapped (fixing)" : "ok");
    }
    if (S.order == XR_ORDER_SWAPPED) xr_unswap16(S.frame, XR_FRAME_BYTES);

    /* skip all-black frames right after startup (same test as the viewers) */
    int64_t sum = 0;
    for (int i = 0; i < XR_IMG_BYTES; i += 7) sum += S.frame[i];
    if (sum / (XR_IMG_BYTES / 7) < 5) return;

    int cam = xr_cam(S.frame, S.dialect);
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

    if (S.have[0] && S.have[1]) compose(xr_counter(S.frame, S.dialect));
}

JNIEXPORT jint JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeStart(JNIEnv *env, jclass cls, jint fd) {
    (void)env; (void)cls;
    if (S.streaming) return 0;
    xr_init();
    S.order = XR_ORDER_UNKNOWN;
    S.dialect = XR_DIALECT_UNKNOWN;
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
    return 0;
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeStop(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    if (!S.streaming) return;
    S.streaming = 0;
    uvc_stop_streaming(S.devh);
    uvc_close(S.devh); S.devh = NULL;
    uvc_exit(S.ctx); S.ctx = NULL;
    LOGI("streaming stopped");
}

JNIEXPORT void JNICALL
Java_org_air2ultra_stereocam_XrealNative_nativeSetClean(JNIEnv *env, jclass cls,
                                                        jboolean clean) {
    (void)env; (void)cls;
    atomic_store(&S.show_clean, clean ? 1 : 0);
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
