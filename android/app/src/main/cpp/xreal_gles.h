/* xreal_gles.h — minimum-latency renderer for the glasses' Surface.
 *
 * A persistent render thread owns an EGL/GLES2 context and, when the device
 * supports EGL_KHR_mutable_render_buffer, switches the window surface into
 * SINGLE-buffer (front-buffer) mode — frames are drawn straight into the
 * buffer being scanned out, VR-compositor style, skipping the SurfaceFlinger
 * queue entirely (worth ~a display frame of latency; tearing is possible and
 * accepted). Falls back to ordinary double-buffered EGL when unavailable.
 *
 * Two draw modes:
 *   - aligned stereo: one distortion mesh per eye (built from the factory
 *     calibration via xr_align_uv), sampling that eye's camera texture with
 *     bilinear filtering — sharper than the old CPU nearest-neighbor warp.
 *   - plain pair: the composed side-by-side RGBA framebuffer as one quad.
 *
 * All entry points are thread-safe; rendering happens on the internal thread.
 */
#ifndef XREAL_GLES_H
#define XREAL_GLES_H

#include <stdint.h>

#include <android/native_window.h>

#include "xreal_align.h"

/* Hand over the glasses' window (ownership of the reference transfers; pass
 * NULL on surface destruction). Starts the render thread on first use. */
void xr_gles_set_window(ANativeWindow *win);

/* Stage the per-eye calibration for the distortion meshes. */
void xr_gles_set_alignment(const xr_eye_calib eyes[2], int variant);

/* Submit the two descrambled+cleaned 480x640 eye images (left, right);
 * copied internally, wakes the render thread. */
void xr_gles_submit_eyes(const uint8_t *left, const uint8_t *right);

/* Submit the plain composed RGBA pair (w x h) instead. */
void xr_gles_submit_pair(const uint8_t *rgba, int w, int h);

#endif
