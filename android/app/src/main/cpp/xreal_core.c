/* xreal_core.c — see xreal_core.h. Ported from src/preview_clean.swift. */
#include "xreal_core.h"

#include <string.h>

#if defined(__aarch64__)
#include <arm_neon.h>   /* hot loops vectorized below; scalar fallback kept.
                         * Row width 480 and totals are multiples of 16, so
                         * no tail handling is needed anywhere. */
#endif

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
    size_t i = 0;
#if defined(__aarch64__)
    for (; i + 16 <= n; i += 16)
        vst1q_u8(flat + i, vrev16q_u8(vld1q_u8(flat + i)));
#endif
    for (; i + 1 < n; i += 2) {
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

/* hist_out (256 entries, optional) receives the intensity histogram this
 * function has to build anyway. The frame path needs the same histogram of the
 * same buffer a few lines later for its 2%/98% contrast stretch; handing this
 * one over saves a second full-image pass per callback at 60 Hz, and makes
 * those percentiles exact instead of 2x-subsampled. */
void xr_equalize_h(const uint8_t *in, uint8_t *out, int n, int32_t *hist_out) {
    int32_t hist[256] = {0}, cdf[256];
    for (int i = 0; i < n; i++) hist[in[i]]++;
    if (hist_out) memcpy(hist_out, hist, sizeof hist);
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

void xr_equalize(const uint8_t *in, uint8_t *out, int n) {
    xr_equalize_h(in, out, n, NULL);
}

/* ---- cleanup ---------------------------------------------------------------
 * Column FPN (vertical stripes, static per camera): median of a horizontal
 * high-pass, accumulated across frames with an EMA. Row banding (temporal):
 * removed per frame the same way along rows. Box means use clamped windows
 * via prefix sums, exactly like the Swift viewer. */

#define R 15  /* high-pass box radius */

static float g_f[XR_OW * XR_OH];   /* shared scratch: single-thread use only */
static float g_hp[XR_OW * XR_OH];
static float g_cp[(XR_OH + 1) * XR_OW];  /* column-prefix scratch for the
                                          * cache-friendly vertical high-pass */
static float g_hpT[XR_OW * XR_OH];       /* blocked transpose of g_hp for the
                                          * cache-friendly column median */

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

#if defined(__aarch64__)
    for (int i = 0; i < XR_OW * XR_OH; i += 16) {   /* u8 -> f32, 16 px/iter */
        uint8x16_t b = vld1q_u8(in + i);
        uint16x8_t lo = vmovl_u8(vget_low_u8(b)), hi = vmovl_u8(vget_high_u8(b));
        vst1q_f32(g_f + i,      vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))));
        vst1q_f32(g_f + i + 4,  vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo))));
        vst1q_f32(g_f + i + 8,  vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))));
        vst1q_f32(g_f + i + 12, vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi))));
    }
#else
    for (int i = 0; i < XR_OW * XR_OH; i++) g_f[i] = in[i];
#endif

    /* column FPN is static: re-estimate the EMA'd stripe only every 3rd frame,
     * and FREEZE once converged — it converges after ~15 estimates and then
     * barely moves, yet the estimator (high-pass + 1.2 MB transpose + 480
     * column medians, ~1 ms) used to rerun 20x/s per camera forever. 24
     * estimates (72 frames) of margin, re-armed by xr_cleaner_reset. */
    if (!c->have_stripe || (c->frame_count < 72 && c->frame_count % 3 == 0)) {
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
        /* column median -> EMA stripe. Gathering each column straight from
         * g_hp strides by XR_OW (cache miss per read); instead transpose g_hp
         * in cache-friendly tiles once, then each column is contiguous for the
         * median. Same medians, far fewer misses. */
        enum { CT = 32 };                         /* transpose tile */
        for (int y0 = 0; y0 < XR_OH; y0 += CT) {
            int y1 = y0 + CT < XR_OH ? y0 + CT : XR_OH;
            for (int x0 = 0; x0 < XR_OW; x0 += CT) {
                int x1 = x0 + CT < XR_OW ? x0 + CT : XR_OW;
                for (int y = y0; y < y1; y++)
                    for (int x = x0; x < x1; x++)
                        g_hpT[(size_t)x * XR_OH + y] = g_hp[(size_t)y * XR_OW + x];
            }
        }
        for (int x = 0; x < XR_OW; x++) {
            float cur = select_kth(g_hpT + (size_t)x * XR_OH, XR_OH, XR_OH / 2);
            c->stripe[x] = c->have_stripe ? 0.95f * c->stripe[x] + 0.05f * cur : cur;
        }
        c->have_stripe = 1;
    }
    c->frame_count++;
    for (int y = 0; y < XR_OH; y++) {
        float *row = g_f + y * XR_OW;
#if defined(__aarch64__)
        for (int x = 0; x < XR_OW; x += 4)
            vst1q_f32(row + x, vsubq_f32(vld1q_f32(row + x),
                                         vld1q_f32(c->stripe + x)));
#else
        for (int x = 0; x < XR_OW; x++) row[x] -= c->stripe[x];
#endif
    }

    /* vertical high-pass. The naive form loops columns-outer and strides down
     * each column of g_f (1920-byte step -> a cache miss per access, every
     * frame). Reformulated as two ROW-sequential passes over a full
     * column-prefix buffer g_cp: g_cp[y+1][x] = g_cp[y][x] + g_f[y][x], then
     * the windowed box-sum reads two whole prefix rows (g_cp[hi+1], g_cp[lo]).
     * The per-column accumulation order in y is unchanged, so the result is
     * bit-identical to the old loop. */
    for (int x = 0; x < XR_OW; x++) g_cp[x] = 0.0f;          /* prefix row 0 */
    for (int y = 0; y < XR_OH; y++) {
        const float *row = g_f + (size_t)y * XR_OW;
        const float *pp = g_cp + (size_t)y * XR_OW;          /* g_cp[y]   */
        float *pc = g_cp + (size_t)(y + 1) * XR_OW;          /* g_cp[y+1] */
#if defined(__aarch64__)
        for (int x = 0; x < XR_OW; x += 4)                   /* columns are
            independent: lanes keep the per-column accumulation order, so the
            sums stay bit-identical to the scalar loop */
            vst1q_f32(pc + x, vaddq_f32(vld1q_f32(pp + x), vld1q_f32(row + x)));
#else
        for (int x = 0; x < XR_OW; x++) pc[x] = pp[x] + row[x];
#endif
    }
    for (int y = 0; y < XR_OH; y++) {
        int lo = y - R < 0 ? 0 : y - R;
        int hi = y + R > XR_OH - 1 ? XR_OH - 1 : y + R;
        const float *plo = g_cp + (size_t)lo * XR_OW;
        const float *phi = g_cp + (size_t)(hi + 1) * XR_OW;
        const float *row = g_f + (size_t)y * XR_OW;
        float *hp = g_hp + (size_t)y * XR_OW;
        float denom = (float)(hi - lo + 1);
#if defined(__aarch64__)
        float32x4_t vd = vdupq_n_f32(denom);                 /* true division
            (vdivq), not reciprocal-multiply: bit-identical to scalar */
        for (int x = 0; x < XR_OW; x += 4)
            vst1q_f32(hp + x,
                      vsubq_f32(vld1q_f32(row + x),
                                vdivq_f32(vsubq_f32(vld1q_f32(phi + x),
                                                    vld1q_f32(plo + x)), vd)));
#else
        for (int x = 0; x < XR_OW; x++)
            hp[x] = row[x] - (phi[x] - plo[x]) / denom;
#endif
    }
    for (int y = 0; y < XR_OH; y++) {
        memcpy(buf, g_hp + y * XR_OW, XR_OW * sizeof(float));
        float m = select_kth(buf, XR_OW, XR_OW / 2);
        float *row = g_f + y * XR_OW;
#if defined(__aarch64__)
        float32x4_t vm = vdupq_n_f32(m);
        for (int x = 0; x < XR_OW; x += 4)
            vst1q_f32(row + x, vsubq_f32(vld1q_f32(row + x), vm));
#else
        for (int x = 0; x < XR_OW; x++) row[x] -= m;
#endif
    }

#if defined(__aarch64__)
    for (int i = 0; i < XR_OW * XR_OH; i += 16) {  /* clamp+narrow, 16 px/iter:
            vcvtq_u32_f32 truncates toward zero exactly like the C cast */
        float32x4_t z = vdupq_n_f32(0.0f), m255 = vdupq_n_f32(255.0f);
        uint32x4_t a = vcvtq_u32_f32(vminq_f32(vmaxq_f32(vld1q_f32(g_f + i),      z), m255));
        uint32x4_t b = vcvtq_u32_f32(vminq_f32(vmaxq_f32(vld1q_f32(g_f + i + 4),  z), m255));
        uint32x4_t c2 = vcvtq_u32_f32(vminq_f32(vmaxq_f32(vld1q_f32(g_f + i + 8), z), m255));
        uint32x4_t d = vcvtq_u32_f32(vminq_f32(vmaxq_f32(vld1q_f32(g_f + i + 12), z), m255));
        uint16x8_t lo16 = vcombine_u16(vmovn_u32(a), vmovn_u32(b));
        uint16x8_t hi16 = vcombine_u16(vmovn_u32(c2), vmovn_u32(d));
        vst1q_u8(out + i, vcombine_u8(vmovn_u16(lo16), vmovn_u16(hi16)));
    }
#else
    for (int i = 0; i < XR_OW * XR_OH; i++) {
        float v = g_f[i];
        out[i] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
#endif
    if (do_equalize) xr_equalize(out, out, XR_OW * XR_OH);
}
