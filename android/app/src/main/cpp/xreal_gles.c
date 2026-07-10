/* xreal_gles.c — see xreal_gles.h. */
#include "xreal_gles.h"

#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <android/log.h>

#include "xreal_core.h"

#define TAG "xrealcam"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#ifndef EGL_MUTABLE_RENDER_BUFFER_BIT_KHR
#define EGL_MUTABLE_RENDER_BUFFER_BIT_KHR 0x1000
#endif
#ifndef EGL_SINGLE_BUFFER
#define EGL_SINGLE_BUFFER 0x3085
#endif
#ifndef EGL_FRONT_BUFFER_AUTO_REFRESH_ANDROID
#define EGL_FRONT_BUFFER_AUTO_REFRESH_ANDROID 0x314C
#endif

enum { GRID_X = 64, GRID_Y = 36 };                 /* distortion mesh cells */
enum { VERTS = (GRID_X + 1) * (GRID_Y + 1), IDX = GRID_X * GRID_Y * 6 };
enum { MODE_NONE = 0, MODE_EYES = 1, MODE_PAIR = 2 };

static struct {
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int thread_running;

    /* commands / staged data (producer side) */
    ANativeWindow *pending_win;
    int win_changed;
    xr_eye_calib eyes_calib[2];
    int calib_variant;
    int calib_staged;
    int calib_valid;     /* calibration persists: meshes (re)build whenever a
                            surface exists — the surface often arrives AFTER
                            the calibration (the 3D-mode switch creates the
                            presentation display) */
    uint8_t eye_img[2][XR_OW * XR_OH];
    uint8_t *pair_img;                              /* MAX pair RGBA */
    int pair_w, pair_h;
    int frame_mode;                                 /* MODE_* of newest frame */
    uint32_t frame_seq;
    int64_t submit_ms;
    uint64_t frame_ts;                              /* exposure, IMU clock ns */
    int (*pose_fn)(uint64_t, float[9]);             /* timewarp pose delta */
    int timewarp;
    float pt_rays[XR_GLES_MAX_POINTS * 3];          /* overlay, IMU-frame */
    int pt_count;
    uint64_t pt_ts;                                 /* their exposure time */
    int show_points;
    int eye_mode;                                   /* XR_EYE_* */

    /* depth passthrough: rectified-frame geometry + colorized image */
    float rect_R[9], rect_f, rect_cx, rect_cy;
    int rect_valid;
    uint8_t depth_img[480 * 640 * 4];
    int depth_w, depth_h;
    int depth_fresh;
    int depth_tex_w, depth_tex_h;

    /* GL state (render thread only) */
    EGLDisplay dpy;
    EGLContext ctx;
    EGLSurface surf;
    ANativeWindow *win;
    int single_buffer;
    GLuint prog, tex_eye[2], tex_pair, tex_depth;
    GLint loc_pos, loc_uv, loc_tex;
    GLuint prog_pt;
    GLint loc_pt_pos;
    GLuint vbo_pt;
    GLuint vbo_eye[2], vbo_quad, ibo;
    int mesh_ready, pair_tex_w, pair_tex_h;
    float mesh_pos[2][VERTS * 2];                   /* static NDC positions */
    float mesh_ray[2][VERTS * 3];                   /* static IMU-frame rays */
    float mesh_vtx[VERTS * 4];                      /* per-present scratch */

    /* stats */
    int frames;
    double render_ms_acc, wait_ms_acc;
    int64_t stat_t0;
} G = { .lock = PTHREAD_MUTEX_INITIALIZER, .cond = PTHREAD_COND_INITIALIZER,
        .calib_variant = XR_ALIGN_VARIANT_DEFAULT, .timewarp = 1,
        .show_points = 1, .eye_mode = XR_EYE_CAM };

static int64_t now_ms64(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* ---- GL helpers (render thread) ---------------------------------------------- */

static const char *VS =
    "attribute vec2 aPos; attribute vec2 aUV; varying vec2 vUV;\n"
    "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); vUV = aUV; }\n";
static const char *FS =
    "precision mediump float; varying vec2 vUV; uniform sampler2D uTex;\n"
    "void main(){ gl_FragColor = texture2D(uTex, vUV); }\n";
static const char *VS_PT =
    "attribute vec2 aPos;\n"
    "void main(){ gl_Position = vec4(aPos, 0.0, 1.0); gl_PointSize = 7.0; }\n";
static const char *FS_PT =
    "precision mediump float;\n"
    "void main(){ gl_FragColor = vec4(0.0, 1.0, 0.4, 1.0); }\n";

static GLuint compile(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[256];
        glGetShaderInfoLog(s, sizeof log, NULL, log);
        LOGE("shader compile: %s", log);
    }
    return s;
}

static int gl_objects_init(void) {
    GLuint vs = compile(GL_VERTEX_SHADER, VS);
    GLuint fs = compile(GL_FRAGMENT_SHADER, FS);
    G.prog = glCreateProgram();
    glAttachShader(G.prog, vs);
    glAttachShader(G.prog, fs);
    glLinkProgram(G.prog);
    GLint ok = 0;
    glGetProgramiv(G.prog, GL_LINK_STATUS, &ok);
    if (!ok) { LOGE("program link failed"); return -1; }
    G.loc_pos = glGetAttribLocation(G.prog, "aPos");
    G.loc_uv = glGetAttribLocation(G.prog, "aUV");
    G.loc_tex = glGetUniformLocation(G.prog, "uTex");

    glGenTextures(2, G.tex_eye);
    for (int e = 0; e < 2; e++) {
        glBindTexture(GL_TEXTURE_2D, G.tex_eye[e]);
        /* nearest on purpose: at this upscale factor crisp camera-pixel
         * blocks look better through the glasses than bilinear smear */
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, XR_OW, XR_OH, 0,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);
    }
    glGenTextures(1, &G.tex_pair);
    glBindTexture(GL_TEXTURE_2D, G.tex_pair);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &G.tex_depth);
    glBindTexture(GL_TEXTURE_2D, G.tex_depth);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    GLuint vsp = compile(GL_VERTEX_SHADER, VS_PT);
    GLuint fsp = compile(GL_FRAGMENT_SHADER, FS_PT);
    G.prog_pt = glCreateProgram();
    glAttachShader(G.prog_pt, vsp);
    glAttachShader(G.prog_pt, fsp);
    glLinkProgram(G.prog_pt);
    G.loc_pt_pos = glGetAttribLocation(G.prog_pt, "aPos");
    glGenBuffers(1, &G.vbo_pt);

    glGenBuffers(2, G.vbo_eye);
    glGenBuffers(1, &G.vbo_quad);
    glGenBuffers(1, &G.ibo);

    /* shared triangle indices for the distortion grids */
    GLushort *idx = malloc(sizeof(GLushort) * IDX);
    int k = 0;
    for (int y = 0; y < GRID_Y; y++)
        for (int x = 0; x < GRID_X; x++) {
            GLushort a = (GLushort)(y * (GRID_X + 1) + x);
            GLushort b = (GLushort)(a + 1);
            GLushort c = (GLushort)(a + GRID_X + 1);
            GLushort d = (GLushort)(c + 1);
            idx[k++] = a; idx[k++] = c; idx[k++] = b;
            idx[k++] = b; idx[k++] = c; idx[k++] = d;
        }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, G.ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * IDX, idx,
                 GL_STATIC_DRAW);
    free(idx);

    /* identity quad for the plain-pair mode (triangle strip) */
    const GLfloat quad[] = { -1, -1, 0, 1,   1, -1, 1, 1,
                             -1,  1, 0, 0,   1,  1, 1, 0 };
    glBindBuffer(GL_ARRAY_BUFFER, G.vbo_quad);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    return 0;
}

/* Distortion mesh statics from the calibration: NDC positions over the
 * eye's viewport and the IMU-frame ray of every vertex. UVs are computed
 * per present (the timewarp rotates the rays before projection). */
static void build_meshes(void) {
    for (int e = 0; e < 2; e++) {
        for (int gy = 0; gy <= GRID_Y; gy++) {
            for (int gx = 0; gx <= GRID_X; gx++) {
                float fx = (float)gx / GRID_X, fy = (float)gy / GRID_Y;
                size_t i = (size_t)gy * (GRID_X + 1) + gx;
                G.mesh_pos[e][i * 2] = -1.0f + 2.0f * fx;   /* NDC x */
                G.mesh_pos[e][i * 2 + 1] = 1.0f - 2.0f * fy; /* top = +1 */
                xr_align_ray(&G.eyes_calib[e], G.calib_variant,
                             fx * 1920.0f, fy * 1080.0f, &G.mesh_ray[e][i * 3]);
            }
        }
    }
    G.mesh_ready = 1;
    LOGI("distortion meshes built (variant %d)", G.calib_variant);
}

/* Fill one eye's interleaved vertex buffer, rotating the rays by dR (the
 * IMU-frame rotation from exposure to now) before projection. depth_mode
 * projects through the rectified pinhole (the depth image's frame) instead
 * of that eye's fisheye camera. */
static void update_eye_vbo(int e, const float dR[9], int depth_mode) {
    for (size_t i = 0; i < VERTS; i++) {
        const float *r = &G.mesh_ray[e][i * 3];
        float w[3];
        if (dR) {
            w[0] = dR[0] * r[0] + dR[1] * r[1] + dR[2] * r[2];
            w[1] = dR[3] * r[0] + dR[4] * r[1] + dR[5] * r[2];
            w[2] = dR[6] * r[0] + dR[7] * r[1] + dR[8] * r[2];
        } else {
            w[0] = r[0]; w[1] = r[1]; w[2] = r[2];
        }
        float u = 0, v = 0;
        if (depth_mode) {
            /* IMU ray -> rect frame (columns of rect_R are the rect axes) */
            const float *R = G.rect_R;
            float rx = R[0] * w[0] + R[3] * w[1] + R[6] * w[2];
            float ry = R[1] * w[0] + R[4] * w[1] + R[7] * w[2];
            float rz = R[2] * w[0] + R[5] * w[1] + R[8] * w[2];
            if (rz > 1e-3f) {
                u = (G.rect_f * rx / rz + G.rect_cx) / (float)G.depth_w;
                v = (G.rect_f * ry / rz + G.rect_cy) / (float)G.depth_h;
            } else {
                u = -1.0f; v = -1.0f;          /* behind: clamps to border */
            }
            G.mesh_vtx[i * 4 + 2] = u;
            G.mesh_vtx[i * 4 + 3] = v;
        } else {
            xr_align_project(&G.eyes_calib[e], G.calib_variant, w, &u, &v);
            G.mesh_vtx[i * 4 + 2] = u / (float)XR_OW;
            G.mesh_vtx[i * 4 + 3] = v / (float)XR_OH;
        }
        G.mesh_vtx[i * 4] = G.mesh_pos[e][i * 2];
        G.mesh_vtx[i * 4 + 1] = G.mesh_pos[e][i * 2 + 1];
    }
    glBindBuffer(GL_ARRAY_BUFFER, G.vbo_eye[e]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * VERTS * 4, G.mesh_vtx,
                 GL_STREAM_DRAW);
}

static void egl_teardown_surface(void) {
    if (G.dpy != EGL_NO_DISPLAY)
        eglMakeCurrent(G.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (G.surf != EGL_NO_SURFACE) {
        eglDestroySurface(G.dpy, G.surf);
        G.surf = EGL_NO_SURFACE;
    }
    if (G.win) {
        ANativeWindow_release(G.win);
        G.win = NULL;
    }
}

static int egl_bind_window(ANativeWindow *win) {
    if (G.dpy == EGL_NO_DISPLAY) {
        G.dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (!eglInitialize(G.dpy, NULL, NULL)) { LOGE("eglInitialize failed"); return -1; }
    }
    const char *exts = eglQueryString(G.dpy, EGL_EXTENSIONS);
    int mutable_rb = exts && strstr(exts, "EGL_KHR_mutable_render_buffer") != NULL;

    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT |
                          (mutable_rb ? EGL_MUTABLE_RENDER_BUFFER_BIT_KHR : 0),
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_NONE
    };
    EGLConfig cfg;
    EGLint ncfg = 0;
    if (!eglChooseConfig(G.dpy, attribs, &cfg, 1, &ncfg) || ncfg == 0) {
        if (mutable_rb) {           /* retry without the front-buffer bit */
            mutable_rb = 0;
            attribs[3] = EGL_WINDOW_BIT;
            if (!eglChooseConfig(G.dpy, attribs, &cfg, 1, &ncfg) || ncfg == 0) {
                LOGE("eglChooseConfig failed");
                return -1;
            }
        } else { LOGE("eglChooseConfig failed"); return -1; }
    }
    if (G.ctx == EGL_NO_CONTEXT) {
        const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
        G.ctx = eglCreateContext(G.dpy, cfg, EGL_NO_CONTEXT, ctx_attribs);
        if (G.ctx == EGL_NO_CONTEXT) { LOGE("eglCreateContext failed"); return -1; }
    }
    G.surf = eglCreateWindowSurface(G.dpy, cfg, win, NULL);
    if (G.surf == EGL_NO_SURFACE) { LOGE("eglCreateWindowSurface failed"); return -1; }
    if (!eglMakeCurrent(G.dpy, G.surf, G.surf, G.ctx)) {
        LOGE("eglMakeCurrent failed");
        eglDestroySurface(G.dpy, G.surf);
        G.surf = EGL_NO_SURFACE;
        return -1;
    }
    eglSwapInterval(G.dpy, 0);

    /* Front-buffer mode needs BOTH extensions: mutable_render_buffer to
     * switch the surface, and front_buffer_auto_refresh so the compositor
     * keeps scanning the shared buffer (without it exactly one frame shows
     * and the layer freezes). The mode change latches on the next
     * eglSwapBuffers; afterwards frames are just flushed. */
    G.single_buffer = 0;
    int auto_refresh = exts &&
        strstr(exts, "EGL_ANDROID_front_buffer_auto_refresh") != NULL;
    if (mutable_rb && auto_refresh &&
        eglSurfaceAttrib(G.dpy, G.surf, EGL_RENDER_BUFFER, EGL_SINGLE_BUFFER) &&
        eglSurfaceAttrib(G.dpy, G.surf, EGL_FRONT_BUFFER_AUTO_REFRESH_ANDROID,
                         EGL_TRUE)) {
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        eglSwapBuffers(G.dpy, G.surf);          /* latch single-buffer mode */
        EGLint rb = 0;
        eglQuerySurface(G.dpy, G.surf, EGL_RENDER_BUFFER, &rb);
        G.single_buffer = rb == EGL_SINGLE_BUFFER;
    }
    LOGI("glasses renderer: GLES2, %s", G.single_buffer
         ? "FRONT-buffer (minimum latency, tearing possible)"
         : "double-buffered (front-buffer mode unavailable)");

    if (!G.prog && gl_objects_init() != 0) return -1;
    G.win = win;
    return 0;
}

static void render_frame(int mode, int fresh, const float *dR,
                         const float *dR_pts) {
    EGLint w = 0, h = 0;
    eglQuerySurface(G.dpy, G.surf, EGL_WIDTH, &w);
    eglQuerySurface(G.dpy, G.surf, EGL_HEIGHT, &h);
    glUseProgram(G.prog);
    glUniform1i(G.loc_tex, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, G.ibo);

    if (mode == MODE_EYES) {
        int eye_mode = G.eye_mode;
        /* depth mode needs the rectification and at least one depth image */
        if (eye_mode == XR_EYE_DEPTH && (!G.rect_valid || G.depth_w == 0))
            eye_mode = XR_EYE_AR;
        int draw_mesh = eye_mode == XR_EYE_CAM || eye_mode == XR_EYE_DEPTH;
        if (!draw_mesh) {
            /* AR/off: point overlay (or nothing) over black */
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
        }
        if (eye_mode == XR_EYE_DEPTH) {
            glBindTexture(GL_TEXTURE_2D, G.tex_depth);
            if (G.depth_tex_w != G.depth_w || G.depth_tex_h != G.depth_h) {
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, G.depth_w, G.depth_h,
                             0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
                G.depth_tex_w = G.depth_w;
                G.depth_tex_h = G.depth_h;
                G.depth_fresh = 1;
            }
            if (G.depth_fresh) {
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G.depth_w, G.depth_h,
                                GL_RGBA, GL_UNSIGNED_BYTE, G.depth_img);
                G.depth_fresh = 0;
            }
        }
        for (int e = 0; e < 2; e++) {
            if (draw_mesh) {
                glUseProgram(G.prog);
                if (eye_mode == XR_EYE_CAM) {
                    glBindTexture(GL_TEXTURE_2D, G.tex_eye[e]);
                    if (fresh)
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, XR_OW, XR_OH,
                                        GL_LUMINANCE, GL_UNSIGNED_BYTE,
                                        G.eye_img[e]);
                } else {
                    glBindTexture(GL_TEXTURE_2D, G.tex_depth);
                }
                update_eye_vbo(e, dR, eye_mode == XR_EYE_DEPTH);
            }
            glViewport(e * w / 2, 0, w / 2, h);
            if (draw_mesh) {
            glVertexAttribPointer((GLuint)G.loc_pos, 2, GL_FLOAT, GL_FALSE,
                                  4 * sizeof(GLfloat), (void *)0);
            glVertexAttribPointer((GLuint)G.loc_uv, 2, GL_FLOAT, GL_FALSE,
                                  4 * sizeof(GLfloat),
                                  (void *)(2 * sizeof(GLfloat)));
            glEnableVertexAttribArray((GLuint)G.loc_pos);
            glEnableVertexAttribArray((GLuint)G.loc_uv);
            glDrawElements(GL_TRIANGLES, IDX, GL_UNSIGNED_SHORT, (void *)0);
            }

            /* tracked-point overlay, timewarped with its OWN pose delta:
             * the points usually come from an older frame than the image */
            if (G.show_points && eye_mode != XR_EYE_OFF && G.pt_count > 0) {
                const float *pw = dR_pts ? dR_pts : dR;
                float ndc[XR_GLES_MAX_POINTS * 2];
                int n = 0;
                for (int i = 0; i < G.pt_count; i++) {
                    const float *r = &G.pt_rays[i * 3];
                    float wr[3];
                    if (pw) {
                        wr[0] = pw[0] * r[0] + pw[1] * r[1] + pw[2] * r[2];
                        wr[1] = pw[3] * r[0] + pw[4] * r[1] + pw[5] * r[2];
                        wr[2] = pw[6] * r[0] + pw[7] * r[1] + pw[8] * r[2];
                    } else {
                        wr[0] = r[0]; wr[1] = r[1]; wr[2] = r[2];
                    }
                    float u, v;
                    if (xr_align_ray_to_display(&G.eyes_calib[e],
                                                G.calib_variant, wr, &u, &v))
                        continue;
                    if (u < 0 || u > 1920 || v < 0 || v > 1080) continue;
                    ndc[n * 2] = u / 960.0f - 1.0f;
                    ndc[n * 2 + 1] = 1.0f - v / 540.0f;
                    n++;
                }
                if (n > 0) {
                    glUseProgram(G.prog_pt);
                    glBindBuffer(GL_ARRAY_BUFFER, G.vbo_pt);
                    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * n * 2, ndc,
                                 GL_STREAM_DRAW);
                    glVertexAttribPointer((GLuint)G.loc_pt_pos, 2, GL_FLOAT,
                                          GL_FALSE, 0, (void *)0);
                    glEnableVertexAttribArray((GLuint)G.loc_pt_pos);
                    glDrawArrays(GL_POINTS, 0, n);
                }
            }
        }
    } else {
        glBindTexture(GL_TEXTURE_2D, G.tex_pair);
        if (G.pair_tex_w != G.pair_w || G.pair_tex_h != G.pair_h) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, G.pair_w, G.pair_h, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE, NULL);
            G.pair_tex_w = G.pair_w;
            G.pair_tex_h = G.pair_h;
        }
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, G.pair_w, G.pair_h,
                        GL_RGBA, GL_UNSIGNED_BYTE, G.pair_img);
        glViewport(0, 0, w, h);
        glBindBuffer(GL_ARRAY_BUFFER, G.vbo_quad);
        glVertexAttribPointer((GLuint)G.loc_pos, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(GLfloat), (void *)0);
        glVertexAttribPointer((GLuint)G.loc_uv, 2, GL_FLOAT, GL_FALSE,
                              4 * sizeof(GLfloat), (void *)(2 * sizeof(GLfloat)));
        glEnableVertexAttribArray((GLuint)G.loc_pos);
        glEnableVertexAttribArray((GLuint)G.loc_uv);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }
    if (G.single_buffer) glFlush();          /* front buffer: no swap needed */
    else eglSwapBuffers(G.dpy, G.surf);
}

static void *render_thread(void *arg) {
    (void)arg;
    uint32_t done_seq = 0;
    int warps = 0;
    for (;;) {
        pthread_mutex_lock(&G.lock);
        /* asynchronous timewarp: once an eyes frame is on screen, wake at
         * display cadence and re-present it with the freshest pose even
         * when no new camera frame arrived */
        int can_atw = G.timewarp && G.pose_fn && G.mesh_ready &&
                      G.frame_mode == MODE_EYES && G.frame_ts != 0 &&
                      G.surf != EGL_NO_SURFACE && done_seq != 0;
        if (can_atw) {
            struct timespec until;
            clock_gettime(CLOCK_REALTIME, &until);
            /* the glasses' SBS mode scans out at 60 Hz — re-warping faster
             * than the panel refresh only burns a big core (mesh reproject
             * per present) and heats the SoC into throttling */
            until.tv_nsec += 16 * 1000000L;
            if (until.tv_nsec >= 1000000000L) {
                until.tv_sec++;
                until.tv_nsec -= 1000000000L;
            }
            while (!G.win_changed && G.frame_seq == done_seq && !G.calib_staged)
                if (pthread_cond_timedwait(&G.cond, &G.lock, &until) == ETIMEDOUT)
                    break;
        } else {
            while (!G.win_changed && G.frame_seq == done_seq && !G.calib_staged)
                pthread_cond_wait(&G.cond, &G.lock);
        }

        ANativeWindow *new_win = NULL;
        int win_changed = G.win_changed;
        if (win_changed) {
            new_win = G.pending_win;
            G.pending_win = NULL;
            G.win_changed = 0;
        }
        if (G.calib_staged) {
            G.calib_staged = 0;
            G.calib_valid = 1;
            G.mesh_ready = 0;                 /* (re)build below */
        }
        int mode = G.frame_mode;
        uint32_t seq = G.frame_seq;
        int64_t submitted = G.submit_ms;
        uint64_t frame_ts = G.frame_ts;
        uint64_t pt_ts = G.pt_ts;
        int tw = G.timewarp;
        int (*pose_fn)(uint64_t, float[9]) = G.pose_fn;
        pthread_mutex_unlock(&G.lock);

        if (win_changed) {
            egl_teardown_surface();
            if (new_win && egl_bind_window(new_win) != 0) {
                LOGE("glasses renderer unavailable on this surface");
                ANativeWindow_release(new_win);
            }
        }
        /* meshes need both a GL surface and the calibration, which arrive in
         * either order (the 3D-mode switch creates the display late) */
        if (G.surf != EGL_NO_SURFACE && G.calib_valid && !G.mesh_ready)
            build_meshes();

        int fresh = seq != done_seq;
        int drawable = G.surf != EGL_NO_SURFACE && mode != MODE_NONE &&
                       (mode == MODE_EYES ? G.mesh_ready : G.pair_w > 0);
        if (!drawable) {
            if (fresh) done_seq = seq;         /* nothing to draw with yet */
            continue;
        }
        if (!fresh && !can_atw) continue;      /* woken for win/calib only */

        float dR[9], dRp[9];
        const float *pdR = NULL, *pdRp = NULL;
        if (mode == MODE_EYES && tw && pose_fn && frame_ts &&
            pose_fn(frame_ts, dR) == 0)
            pdR = dR;
        if (mode == MODE_EYES && tw && pose_fn && pt_ts &&
            pose_fn(pt_ts, dRp) == 0)
            pdRp = dRp;

        int64_t t0 = now_ms64();
        render_frame(mode, fresh, pdR, pdRp);
        int64_t t1 = now_ms64();
        done_seq = seq;
        if (fresh) {
            G.frames++;
            G.wait_ms_acc += (double)(t0 - submitted);
        } else {
            warps++;
        }
        G.render_ms_acc += (double)(t1 - t0);
        if (G.stat_t0 == 0) G.stat_t0 = t1;
        if (t1 - G.stat_t0 >= 1000) {
            int total = G.frames + warps;
            LOGI("glasses present: %d cam + %d warp fps, wait %.1f ms, "
                 "render %.1f ms (%s%s)",
                 G.frames, warps,
                 G.frames ? G.wait_ms_acc / G.frames : 0.0,
                 total ? G.render_ms_acc / total : 0.0,
                 G.single_buffer ? "front-buffer" : "double-buffered",
                 pdR ? ", timewarp" : "");
            G.frames = 0;
            warps = 0;
            G.render_ms_acc = G.wait_ms_acc = 0;
            G.stat_t0 = t1;
        }
    }
    return NULL;
}

static void ensure_thread(void) {
    if (!G.thread_running) {
        G.thread_running = 1;
        pthread_create(&G.thread, NULL, render_thread, NULL);
    }
}

/* ---- producer API ------------------------------------------------------------- */

void xr_gles_set_window(ANativeWindow *win) {
    pthread_mutex_lock(&G.lock);
    ensure_thread();
    if (G.pending_win) ANativeWindow_release(G.pending_win);
    G.pending_win = win;
    G.win_changed = 1;
    pthread_cond_signal(&G.cond);
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_alignment(const xr_eye_calib eyes[2], int variant) {
    pthread_mutex_lock(&G.lock);
    ensure_thread();
    memcpy(G.eyes_calib, eyes, sizeof G.eyes_calib);
    G.calib_variant = variant;
    G.calib_staged = 1;
    pthread_cond_signal(&G.cond);
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_submit_eyes(const uint8_t *left, const uint8_t *right,
                         uint64_t exposure_ts_ns) {
    pthread_mutex_lock(&G.lock);
    ensure_thread();
    memcpy(G.eye_img[0], left, sizeof G.eye_img[0]);
    memcpy(G.eye_img[1], right, sizeof G.eye_img[1]);
    G.frame_mode = MODE_EYES;
    G.frame_ts = exposure_ts_ns;
    G.frame_seq++;
    G.submit_ms = now_ms64();
    pthread_cond_signal(&G.cond);
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_pose_fn(int (*fn)(uint64_t, float[9])) {
    pthread_mutex_lock(&G.lock);
    G.pose_fn = fn;
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_timewarp(int on) {
    pthread_mutex_lock(&G.lock);
    G.timewarp = on ? 1 : 0;
    pthread_cond_signal(&G.cond);
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_points(const float *rays_imu, int n, uint64_t exposure_ts_ns) {
    if (n > XR_GLES_MAX_POINTS) n = XR_GLES_MAX_POINTS;
    pthread_mutex_lock(&G.lock);
    /* the render thread reads the rays outside the lock (like the eye
     * images); rays are written before the count, so a mid-present update
     * at worst draws one frame of slightly stale dots */
    if (n > 0) memcpy(G.pt_rays, rays_imu, sizeof(float) * (size_t)n * 3);
    G.pt_count = n;
    G.pt_ts = exposure_ts_ns;
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_show_points(int on) {
    pthread_mutex_lock(&G.lock);
    G.show_points = on ? 1 : 0;
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_eye_mode(int mode) {
    pthread_mutex_lock(&G.lock);
    G.eye_mode = mode & 3;
    pthread_cond_signal(&G.cond);      /* re-present with the new look */
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_set_rect(const float R_rect_imu[9], float f, float cx, float cy) {
    pthread_mutex_lock(&G.lock);
    memcpy(G.rect_R, R_rect_imu, sizeof G.rect_R);
    G.rect_f = f;
    G.rect_cx = cx;
    G.rect_cy = cy;
    G.rect_valid = 1;
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_submit_depth(const uint8_t *rgba, int w, int h) {
    if (w <= 0 || h <= 0 || w > 480 || h > 640) return;
    pthread_mutex_lock(&G.lock);
    memcpy(G.depth_img, rgba, (size_t)w * h * 4);
    G.depth_w = w;
    G.depth_h = h;
    G.depth_fresh = 1;
    pthread_cond_signal(&G.cond);
    pthread_mutex_unlock(&G.lock);
}

void xr_gles_submit_pair(const uint8_t *rgba, int w, int h) {
    pthread_mutex_lock(&G.lock);
    ensure_thread();
    size_t need = (size_t)w * h * 4;
    if (!G.pair_img) G.pair_img = malloc((size_t)2 * XR_W * XR_H_IMG * 4);
    if (G.pair_img && need <= (size_t)2 * XR_W * XR_H_IMG * 4) {
        memcpy(G.pair_img, rgba, need);
        G.pair_w = w;
        G.pair_h = h;
        G.frame_mode = MODE_PAIR;
        G.frame_seq++;
        G.submit_ms = now_ms64();
        pthread_cond_signal(&G.cond);
    }
    pthread_mutex_unlock(&G.lock);
}
