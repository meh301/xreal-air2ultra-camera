/* xreal_core.c — see xreal_core.h. Ported from src/preview_clean.swift. */
#include "xreal_core.h"

#include <string.h>

/* block reorder table, discovered by github.com/mazeasdamien/myXreal */
static const uint8_t REORDER[XR_NB] = {
    119, 54, 21, 0, 108, 22, 51, 63, 93, 99, 67, 7, 32, 112, 52, 43,
    14, 35, 75, 116, 64, 71, 44, 89, 18, 88, 26, 61, 70, 56, 90, 79,
    87, 120, 81, 101, 121, 17, 72, 31, 53, 124, 127, 113, 111, 36, 48,
    19, 37, 83, 126, 74, 109, 5, 84, 41, 76, 30, 110, 29, 12, 115, 28,
    102, 105, 62, 103, 20, 3, 68, 49, 77, 117, 125, 106, 60, 69, 98, 9,
    16, 78, 47, 40, 2, 118, 34, 13, 50, 46, 80, 85, 66, 42, 123, 122,
    96, 11, 25, 97, 39, 6, 86, 1, 8, 82, 92, 59, 104, 24, 15, 73, 65,
    38, 58, 10, 23, 33, 55, 57, 107, 100, 94, 27, 95, 45, 91, 4, 114,
};

static uint8_t INV_REORDER[XR_NB];
static int32_t LUT[2][XR_IMG_BYTES];   /* [cam][stream byte] -> output index */

void xr_init(void) {
    for (int i = 0; i < XR_NB; i++)
        INV_REORDER[REORDER[i]] = (uint8_t)i;
    /* byte i of the descrambled stream is pixel i%640 of sensor line i/640;
     * sensor lines are columns of the upright portrait image (sensors sit
     * rotated 90 deg, the two cameras 180 deg opposed). Horizontal sense
     * verified against the real scene on-device (the original mapping was
     * mirrored). */
    for (int32_t i = 0; i < XR_IMG_BYTES; i++) {
        int32_t r = i / XR_W, c = i % XR_W;
        LUT[0][i] = c * XR_OW + (XR_H_IMG - 1 - r);                  /* cam0 */
        LUT[1][i] = (XR_W - 1 - c) * XR_OW + r;                      /* cam1 */
    }
}

/* fingerprint: frame dimensions 640,480,640 as LE u16 at telemetry cols
 * 51-56, in native and 16-bit-pair-swapped byte order */
static const uint8_t MARK_NATIVE[8]  = {0x00, 0x80, 0x02, 0xE0, 0x01, 0x80, 0x02, 0x00};
static const uint8_t MARK_SWAPPED[8] = {0x80, 0x00, 0xE0, 0x02, 0x80, 0x01, 0x00, 0x02};

xr_order xr_classify(const uint8_t *flat) {
    const uint8_t *t = flat + XR_META_ROW * XR_W;
    if (!memcmp(t + 50, MARK_NATIVE, 8))  return XR_ORDER_OK;
    if (!memcmp(t + 50, MARK_SWAPPED, 8)) return XR_ORDER_SWAPPED;
    return XR_ORDER_UNKNOWN;
}

void xr_unswap16(uint8_t *flat, size_t n) {
    for (size_t i = 0; i + 1 < n; i += 2) {
        uint8_t a = flat[i];
        flat[i] = flat[i + 1];
        flat[i + 1] = a;
    }
}

int xr_descramble(const uint8_t *img, int cam, uint8_t *out) {
    /* sync: logical block 0 starts in the fisheye's black border, so the raw
     * block whose first 128 bytes sum lowest is REORDER[align] */
    int32_t best = INT32_MAX;
    int min_block = 0;
    for (int b = 0; b < XR_NB; b++) {
        const uint8_t *p = img + b * XR_BS;
        int32_t s = 0;
        for (int k = 0; k < 128; k++) s += p[k];
        if (s < best) { best = s; min_block = b; }
    }
    int align = INV_REORDER[min_block];
    const int32_t *lut = LUT[cam & 1];
    for (int t = 0; t < XR_NB; t++) {
        const uint8_t *src = img + REORDER[(align + t) & (XR_NB - 1)] * XR_BS;
        const int32_t *l = lut + t * XR_BS;
        for (int k = 0; k < XR_BS; k++) out[l[k]] = src[k];
    }
    return 0;
}

void xr_equalize(const uint8_t *in, uint8_t *out, int n) {
    int32_t hist[256] = {0}, cdf[256];
    for (int i = 0; i < n; i++) hist[in[i]]++;
    int32_t acc = 0;
    for (int i = 0; i < 256; i++) { acc += hist[i]; cdf[i] = acc; }
    int32_t cdfmin = 0;
    for (int i = 0; i < 256; i++) if (cdf[i]) { cdfmin = cdf[i]; break; }
    int64_t denom = n - cdfmin; if (denom < 1) denom = 1;
    uint8_t lut[256];
    for (int i = 0; i < 256; i++) {
        int64_t v = (int64_t)(cdf[i] - cdfmin) * 255 / denom;
        lut[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    for (int i = 0; i < n; i++) out[i] = lut[in[i]];
}

/* ---- cleanup ---------------------------------------------------------------
 * Column FPN (vertical stripes, static per camera): median of a horizontal
 * high-pass, accumulated across frames with an EMA. Row banding (temporal):
 * removed per frame the same way along rows. Box means use clamped windows
 * via prefix sums, exactly like the Swift viewer. */

#define R 15  /* high-pass box radius */

static float g_f[XR_OW * XR_OH];   /* shared scratch: single-thread use only */
static float g_hp[XR_OW * XR_OH];

static float select_kth(float *a, int n, int k) {
    /* Hoare quickselect; a is scratch and gets reordered */
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        float pivot = a[k];
        int i = lo, j = hi;
        do {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i <= j) {
                float t = a[i]; a[i] = a[j]; a[j] = t;
                i++; j--;
            }
        } while (i <= j);
        if (j < k) lo = i;
        if (k < i) hi = j;
    }
    return a[k];
}

void xr_cleaner_reset(xr_cleaner *c) {
    c->have_stripe = 0;
    c->frame_count = 0;
    memset(c->stripe, 0, sizeof c->stripe);
}

void xr_clean(xr_cleaner *c, const uint8_t *in, uint8_t *out,
              int do_equalize) {
    float pref[XR_OH + 1];   /* fits the longer axis */
    float buf[XR_OH];

    for (int i = 0; i < XR_OW * XR_OH; i++) g_f[i] = in[i];

    /* column FPN is static: re-estimate the EMA'd stripe only every 3rd
     * frame (it converges after ~15 estimates and then barely moves) */
    if (!c->have_stripe || c->frame_count % 3 == 0) {
        /* horizontal high-pass */
        for (int y = 0; y < XR_OH; y++) {
            float *row = g_f + y * XR_OW, *hp = g_hp + y * XR_OW;
            pref[0] = 0;
            for (int x = 0; x < XR_OW; x++) pref[x + 1] = pref[x] + row[x];
            for (int x = 0; x < XR_OW; x++) {
                int lo = x - R < 0 ? 0 : x - R;
                int hi = x + R > XR_OW - 1 ? XR_OW - 1 : x + R;
                hp[x] = row[x] - (pref[hi + 1] - pref[lo]) / (float)(hi - lo + 1);
            }
        }
        /* column median -> EMA stripe */
        for (int x = 0; x < XR_OW; x++) {
            for (int y = 0; y < XR_OH; y++) buf[y] = g_hp[y * XR_OW + x];
            float cur = select_kth(buf, XR_OH, XR_OH / 2);
            c->stripe[x] = c->have_stripe ? 0.95f * c->stripe[x] + 0.05f * cur : cur;
        }
        c->have_stripe = 1;
    }
    c->frame_count++;
    for (int y = 0; y < XR_OH; y++) {
        float *row = g_f + y * XR_OW;
        for (int x = 0; x < XR_OW; x++) row[x] -= c->stripe[x];
    }

    /* vertical high-pass, per-frame row median subtraction */
    for (int x = 0; x < XR_OW; x++) {
        pref[0] = 0;
        for (int y = 0; y < XR_OH; y++) pref[y + 1] = pref[y] + g_f[y * XR_OW + x];
        for (int y = 0; y < XR_OH; y++) {
            int lo = y - R < 0 ? 0 : y - R;
            int hi = y + R > XR_OH - 1 ? XR_OH - 1 : y + R;
            g_hp[y * XR_OW + x] = g_f[y * XR_OW + x] -
                                  (pref[hi + 1] - pref[lo]) / (float)(hi - lo + 1);
        }
    }
    for (int y = 0; y < XR_OH; y++) {
        memcpy(buf, g_hp + y * XR_OW, XR_OW * sizeof(float));
        float m = select_kth(buf, XR_OW, XR_OW / 2);
        float *row = g_f + y * XR_OW;
        for (int x = 0; x < XR_OW; x++) row[x] -= m;
    }

    for (int i = 0; i < XR_OW * XR_OH; i++) {
        float v = g_f[i];
        out[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    if (do_equalize) xr_equalize(out, out, XR_OW * XR_OH);
}
