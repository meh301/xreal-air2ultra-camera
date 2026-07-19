/* xreal_core.h — portable C99 core for the XREAL Air 2 Ultra tracking-camera
 * stream: telemetry fingerprint, block descramble, FPN/banding cleanup.
 * Same algorithms as src/preview_clean.swift and python/xreal_uvc.py;
 * see docs/PROTOCOL.md for the frame layout.
 */
#ifndef XREAL_CORE_H
#define XREAL_CORE_H

#include <stddef.h>
#include <stdint.h>

enum {
    XR_W = 640, XR_H_IMG = 480, XR_H_FULL = 482,
    XR_META_ROW = 480,
    XR_CTR_COL = 18, XR_CAM_COL = 59,       /* telemetry: pair counter, camera bit */
    XR_FRAME_BYTES = XR_W * XR_H_FULL,      /* 308480 */
    XR_IMG_BYTES = XR_W * XR_H_IMG,         /* 307200 */
    XR_NB = 128, XR_BS = 2400               /* scramble blocks */
};

/* Working image size fed to SLAM/map (descrambled portrait on-device).
 * Overridable at compile time (-DXR_OW=... -DXR_OH=...) so the benchmark
 * replay harness can rebuild the stack at dataset resolutions (EuRoC
 * 752x480, TUM-VI 512x512, MSD 640x480/960x960); all downstream buffers
 * (xr_slam frame pushes, xr_map integral/keyframe arrays) size from these. */
#ifndef XR_OW
#define XR_OW 480
#endif
#ifndef XR_OH
#define XR_OH 640
#endif

/* Byte order of the delivered stream. The fourcc is fake, so a UVC stack may
 * hand us the bytes of each 16-bit "YUV" pair in either order; constant
 * telemetry bytes reveal which. Requires the current glasses firmware
 * (MCU 12.1.00.498+; https://ota.xreal.com/ultra-update?version=1). */
typedef enum {
    XR_ORDER_UNKNOWN = -1,
    XR_ORDER_OK = 0,
    XR_ORDER_SWAPPED = 1
} xr_order;

void xr_init(void);                            /* build LUTs; call once */
xr_order xr_classify(const uint8_t *flat);     /* fingerprint XR_FRAME_BYTES */
void xr_unswap16(uint8_t *flat, size_t n);     /* undo the pairwise swap */

/* camera index (0 = the camera using the right-LUT; telemetry bit 1) */
static inline int xr_cam(const uint8_t *flat)
{ return 1 - (flat[XR_META_ROW * XR_W + XR_CAM_COL] & 1); }
/* stereo pair counter, shared by both frames of a pair */
static inline int xr_counter(const uint8_t *flat)
{ return flat[XR_META_ROW * XR_W + XR_CTR_COL]; }
/* u32 exposure timestamp [ns], low 32 bits of the IMU clock */
static inline uint32_t xr_exposure_ts(const uint8_t *flat) {
    const uint8_t *t = flat + XR_META_ROW * XR_W;
    return (uint32_t)t[0] | (uint32_t)t[1] << 8 |
           (uint32_t)t[2] << 16 | (uint32_t)t[3] << 24;
}

/* Descramble the 307200 image bytes of camera `cam` (cam0 = right LUT) into a
 * 480x640 portrait raster. Returns 0, or -1 if phase sync failed. */
int xr_descramble(const uint8_t *img, int cam, uint8_t *out);

void xr_equalize(const uint8_t *in, uint8_t *out, int n);
/* As xr_equalize, but also copies the 256-bin intensity histogram it builds
 * into hist_out (may be NULL). Lets a caller that needs the same histogram
 * skip a second full-image pass. */
void xr_equalize_h(const uint8_t *in, uint8_t *out, int n, int32_t *hist_out);

/* Online column-FPN + row-banding removal (one instance per camera).
 * NOT thread-safe: all cleaners share static scratch buffers, call them from
 * a single thread (the UVC callback thread). */
typedef struct {
    float stripe[XR_OW];
    int have_stripe;
    int frame_count;   /* the static stripe is re-estimated every 3rd frame */
} xr_cleaner;

void xr_cleaner_reset(xr_cleaner *c);
/* do_equalize: per-frame global histogram equalization is right for human
 * viewing but flickers brightness frame to frame, which hurts feature
 * matching — pass 0 when the output feeds a tracker. */
void xr_clean(xr_cleaner *c, const uint8_t *in, uint8_t *out,
              int do_equalize); /* 480x640 */

#endif
