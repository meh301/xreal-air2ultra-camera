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
    XR_META_ROW = 480, XR_PAD_ROW = 481, XR_PAD_VAL = 0x5C,
    XR_CTR_COL = 19, XR_CAM_COL = 58,
    XR_FRAME_BYTES = XR_W * XR_H_FULL,      /* 308480 */
    XR_IMG_BYTES = XR_W * XR_H_IMG,         /* 307200 */
    XR_NB = 128, XR_BS = 2400,              /* scramble blocks */
    XR_OW = 480, XR_OH = 640                /* descrambled portrait size */
};

/* Byte order of the delivered stream. The fourcc is fake, so a UVC stack may
 * hand us the bytes of each 16-bit "YUV" pair in either order; constant
 * telemetry bytes reveal which. */
typedef enum {
    XR_ORDER_UNKNOWN = -1,
    XR_ORDER_OK = 0,
    XR_ORDER_SWAPPED = 1
} xr_order;

/* Firmware telemetry dialect (see docs/PROTOCOL.md "Telemetry row"):
 * A: counter col 19, camera id 0x20/0x21 at col 58, 0xAD,0xDA markers.
 * B: counter col 18, camera bit at col 59, dims marker at cols 51-56. */
typedef enum {
    XR_DIALECT_UNKNOWN = -1,
    XR_DIALECT_A = 0,
    XR_DIALECT_B = 1
} xr_dialect;

void xr_init(void);                            /* build LUTs; call once */
/* fingerprint one XR_FRAME_BYTES frame; sets *dialect on success */
xr_order xr_classify(const uint8_t *flat, xr_dialect *dialect);
void xr_unswap16(uint8_t *flat, size_t n);     /* undo the pairwise swap */

/* camera index (0 = the camera using the right-LUT) and stereo pair counter */
int xr_cam(const uint8_t *flat, xr_dialect d);
int xr_counter(const uint8_t *flat, xr_dialect d);

/* Descramble the 307200 image bytes of camera `cam` (cam0 = right LUT) into a
 * 480x640 portrait raster. Returns 0, or -1 if phase sync failed. */
int xr_descramble(const uint8_t *img, int cam, uint8_t *out);

void xr_equalize(const uint8_t *in, uint8_t *out, int n);

/* Online column-FPN + row-banding removal (one instance per camera).
 * NOT thread-safe: all cleaners share static scratch buffers, call them from
 * a single thread (the UVC callback thread). */
typedef struct {
    float stripe[XR_OW];
    int have_stripe;
} xr_cleaner;

void xr_cleaner_reset(xr_cleaner *c);
void xr_clean(xr_cleaner *c, const uint8_t *in, uint8_t *out); /* 480x640 */

#endif
