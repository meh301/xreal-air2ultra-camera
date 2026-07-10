/* xreal_align.c — see xreal_align.h. Mirrors python/xreal_align.py. */
#include "xreal_align.h"

#include <math.h>

#include "xreal_core.h"

/* Hamilton rotation matrix from xyzw; conj flips the vector part (this is
 * how the JPL-vs-Hamilton ambiguity is cycled at runtime). */
static void quat_to_rot(const float q[4], int conj, float R[9]) {
    float x = q[0], y = q[1], z = q[2], w = q[3];
    float n = sqrtf(x * x + y * y + z * z + w * w);
    x /= n; y /= n; z /= n; w /= n;
    if (conj) { x = -x; y = -y; z = -z; }
    R[0] = 1 - 2 * (y * y + z * z); R[1] = 2 * (x * y - w * z); R[2] = 2 * (x * z + w * y);
    R[3] = 2 * (x * y + w * z); R[4] = 1 - 2 * (x * x + z * z); R[5] = 2 * (y * z - w * x);
    R[6] = 2 * (x * z - w * y); R[7] = 2 * (y * z + w * x); R[8] = 1 - 2 * (x * x + y * y);
}

static void fisheye624_project(const float p[3], const float fc[2],
                               const float cc[2], const float kc[12],
                               float *u, float *v) {
    float r = hypotf(p[0], p[1]);
    float theta = atan2f(r, p[2]);
    float t2 = theta * theta;
    float poly = 1 + t2 * (kc[0] + t2 * (kc[1] + t2 * (kc[2] + t2 * (
        kc[3] + t2 * (kc[4] + t2 * kc[5])))));
    float thd = theta * poly;
    float xr = r > 1e-9f ? thd * p[0] / r : 0.0f;
    float yr = r > 1e-9f ? thd * p[1] / r : 0.0f;
    float r2 = xr * xr + yr * yr;
    float p0 = kc[6], p1 = kc[7];
    float dx = 2 * p0 * xr * yr + p1 * (r2 + 2 * xr * xr) + kc[8] * r2 + kc[9] * r2 * r2;
    float dy = p0 * (r2 + 2 * yr * yr) + 2 * p1 * xr * yr + kc[10] * r2 + kc[11] * r2 * r2;
    *u = fc[0] * (xr + dx) + cc[0];
    *v = fc[1] * (yr + dy) + cc[1];
}

void xr_align_ray(const xr_eye_calib *eye, int variant,
                  float u_disp, float v_disp, float ray_imu[3]) {
    float Rd[9];
    quat_to_rot(eye->q_disp, variant & 1, Rd);
    float ud = (u_disp - eye->K[2]) / eye->K[0];
    float vd = (v_disp - eye->K[5]) / eye->K[4];
    /* ray in display frame -> IMU frame (Rd maps display->imu) */
    ray_imu[0] = Rd[0] * ud + Rd[1] * vd + Rd[2];
    ray_imu[1] = Rd[3] * ud + Rd[4] * vd + Rd[5];
    ray_imu[2] = Rd[6] * ud + Rd[7] * vd + Rd[8];
}

int xr_align_project(const xr_eye_calib *eye, int variant,
                     const float ray_imu[3], float *u_cam, float *v_cam) {
    float Rc[9];
    quat_to_rot(eye->q_cam, (variant >> 1) & 1, Rc);
    /* IMU -> camera: Rc maps cam->imu, so multiply by its transpose */
    float p[3] = {
        Rc[0] * ray_imu[0] + Rc[3] * ray_imu[1] + Rc[6] * ray_imu[2],
        Rc[1] * ray_imu[0] + Rc[4] * ray_imu[1] + Rc[7] * ray_imu[2],
        Rc[2] * ray_imu[0] + Rc[5] * ray_imu[1] + Rc[8] * ray_imu[2],
    };
    if (p[2] <= 1e-6f) return -1;
    fisheye624_project(p, eye->fc, eye->cc, eye->kc, u_cam, v_cam);
    return 0;
}

int xr_align_uv(const xr_eye_calib *eye, int variant,
                float u_disp, float v_disp, float *u_cam, float *v_cam) {
    float ray[3];
    xr_align_ray(eye, variant, u_disp, v_disp, ray);
    return xr_align_project(eye, variant, ray, u_cam, v_cam);
}

int xr_align_ray_to_display(const xr_eye_calib *eye, int variant,
                            const float ray_imu[3], float *u_disp, float *v_disp) {
    float Rd[9];
    quat_to_rot(eye->q_disp, variant & 1, Rd);
    /* Rd maps display->imu; transpose takes the ray into the display frame */
    float x = Rd[0] * ray_imu[0] + Rd[3] * ray_imu[1] + Rd[6] * ray_imu[2];
    float y = Rd[1] * ray_imu[0] + Rd[4] * ray_imu[1] + Rd[7] * ray_imu[2];
    float z = Rd[2] * ray_imu[0] + Rd[5] * ray_imu[1] + Rd[8] * ray_imu[2];
    if (z <= 1e-6f) return -1;
    *u_disp = eye->K[0] * (x / z) + eye->K[2];
    *v_disp = eye->K[4] * (y / z) + eye->K[5];
    return 0;
}

void xr_align_build(const xr_eye_calib *eye, int variant, int w, int h,
                    int full_w, int full_h, int32_t *out_idx) {
    float su = (float)full_w / w, sv = (float)full_h / h;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float u, v;
            int32_t idx = -1;
            if (xr_align_uv(eye, variant, (x + 0.5f) * su - 0.5f,
                            (y + 0.5f) * sv - 0.5f, &u, &v) == 0) {
                int ui = (int)lroundf(u), vi = (int)lroundf(v);
                if (ui >= 0 && ui < XR_OW && vi >= 0 && vi < XR_OH)
                    idx = vi * XR_OW + ui;
            }
            out_idx[(size_t)y * w + x] = idx;
        }
    }
}
