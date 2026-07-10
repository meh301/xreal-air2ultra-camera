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

/* Submit the two descrambled+cleaned 480x640 eye images (left, right) with
 * their exposure timestamp on the IMU clock (0 = unknown, disables warping
 * for this frame); copied internally, wakes the render thread. */
void xr_gles_submit_eyes(const uint8_t *left, const uint8_t *right,
                         uint64_t exposure_ts_ns);

/* Submit the plain composed RGBA pair (w x h) instead. */
void xr_gles_submit_pair(const uint8_t *rgba, int w, int h);

/* IMU timewarp: fn(ts_exposure, dR) fills the 3x3 rotation of the IMU frame
 * from the exposure time to now (row-major, IMU-frame vectors), returning 0,
 * or nonzero when no pose is available. The renderer counter-rotates the
 * displayed rays by it every present, and re-presents the newest frame at
 * display rate between camera frames (asynchronous timewarp). */
void xr_gles_set_pose_fn(int (*fn)(uint64_t ts_exposure_ns, float dR[9]));

/* Enable/disable the timewarp at runtime (default on). */
void xr_gles_set_timewarp(int on);

/* Stage tracked points / landmarks as IMU-frame rays (n x 3 floats) with
 * the exposure timestamp of the frame they were measured in (IMU clock;
 * 0 = unknown). The overlay is timewarped by ITS OWN pose delta, so points
 * from an older frame than the displayed image still land world-locked.
 * n = 0 clears. */
enum { XR_GLES_MAX_POINTS = 256 };
void xr_gles_set_points(const float *rays_imu, int n, uint64_t exposure_ts_ns);

/* Stage the accumulated landmark map for the AR eye mode: world-frame
 * points plus the reference pose they should be viewed from (body->world
 * rotation R, position p, at IMU-clock ts). Every present re-orients by
 * the 1 kHz pose delta and renders with full 6-DoF parallax — SLAM drift
 * becomes directly visible against the real world. */
enum { XR_GLES_MAX_MAP = 4096 };
void xr_gles_set_map(const float *xyz_world, int n, const float R_base[9],
                     const float p_base[3], uint64_t ts_ns);

/* Show/hide the point overlay (default on; drawn in every eye mode
 * except OFF). */
void xr_gles_set_show_points(int on);

/* What the glasses show in the aligned stereo mode. */
enum {
    XR_EYE_CAM = 0,     /* camera passthrough (default) */
    XR_EYE_DEPTH = 1,   /* colorized stereo depth, world-aligned per eye */
    XR_EYE_AR = 2,      /* black background: tracked points only */
    XR_EYE_OFF = 3      /* nothing */
};
void xr_gles_set_eye_mode(int mode);

/* Rectified-frame parameters for the depth passthrough: R maps rect->IMU
 * (row-major, columns = the rect axes), pinhole f/cx/cy in rect pixels.
 * Set once when the rectification is built. */
void xr_gles_set_rect(const float R_rect_imu[9], float f, float cx, float cy);

/* Submit the colorized depth image (RGBA, w x h, up to 480x640); copied
 * internally. The border pixels should be black — samples that leave the
 * image clamp to them. */
void xr_gles_submit_depth(const uint8_t *rgba, int w, int h);

#endif
