/* xreal_gles.c — see xreal_gles.h. */
#include "xreal_gles.h"

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
    uint8_t eye_img[2][XR_OW * XR_OH];
    uint8_t *pair_img;                              /* MAX pair RGBA */
    int pair_w, pair_h;
    int frame_mode;                                 /* MODE_* of newest frame */
    uint32_t frame_seq;
    int64_t submit_ms;

    /* GL state (render thread only) */
    EGLDisplay dpy;
    EGLContext ctx;
    EGLSurface surf;
    ANativeWindow *win;
    int single_buffer;
    GLuint prog, tex_eye[2], tex_pair;
    GLint loc_pos, loc_uv, loc_tex;
    GLuint vbo_eye[2], vbo_quad, ibo;
    int mesh_ready, pair_tex_w, pair_tex_h;

    /* stats */
    int frames;
    double render_ms_acc, wait_ms_acc;
    int64_t stat_t0;
} G = { .lock = PTHREAD_MUTEX_INITIALIZER, .cond = PTHREAD_COND_INITIALIZER,
        .calib_variant = XR_ALIGN_VARIANT_DEFAULT };

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
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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

/* Distortion meshes from the calibration: NDC positions over the eye's
 * viewport, UVs into the 480x640 camera texture. */
static void build_meshes(void) {
    GLfloat *v = malloc(sizeof(GLfloat) * VERTS * 4);
    for (int e = 0; e < 2; e++) {
        for (int gy = 0; gy <= GRID_Y; gy++) {
            for (int gx = 0; gx <= GRID_X; gx++) {
                float fx = (float)gx / GRID_X, fy = (float)gy / GRID_Y;
                float u, vv;
                if (xr_align_uv(&G.eyes_calib[e], G.calib_variant,
                                fx * 1920.0f, fy * 1080.0f, &u, &vv) != 0) {
                    u = vv = 0;
                }
                GLfloat *p = v + ((size_t)gy * (GRID_X + 1) + gx) * 4;
                p[0] = -1.0f + 2.0f * fx;          /* NDC x */
                p[1] = 1.0f - 2.0f * fy;           /* NDC y (display top = +1) */
                p[2] = u / (float)XR_OW;           /* camera texture UV */
                p[3] = vv / (float)XR_OH;
            }
        }
        glBindBuffer(GL_ARRAY_BUFFER, G.vbo_eye[e]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * VERTS * 4, v,
                     GL_STATIC_DRAW);
    }
    free(v);
    G.mesh_ready = 1;
    LOGI("distortion meshes built (variant %d)", G.calib_variant);
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

    G.single_buffer = 0;
    if (mutable_rb &&
        eglSurfaceAttrib(G.dpy, G.surf, EGL_RENDER_BUFFER, EGL_SINGLE_BUFFER)) {
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

static void render_frame(int mode) {
    EGLint w = 0, h = 0;
    eglQuerySurface(G.dpy, G.surf, EGL_WIDTH, &w);
    eglQuerySurface(G.dpy, G.surf, EGL_HEIGHT, &h);
    glUseProgram(G.prog);
    glUniform1i(G.loc_tex, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, G.ibo);

    if (mode == MODE_EYES && G.mesh_ready) {
        for (int e = 0; e < 2; e++) {
            glBindTexture(GL_TEXTURE_2D, G.tex_eye[e]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, XR_OW, XR_OH,
                            GL_LUMINANCE, GL_UNSIGNED_BYTE, G.eye_img[e]);
            glViewport(e * w / 2, 0, w / 2, h);
            glBindBuffer(GL_ARRAY_BUFFER, G.vbo_eye[e]);
            glVertexAttribPointer((GLuint)G.loc_pos, 2, GL_FLOAT, GL_FALSE,
                                  4 * sizeof(GLfloat), (void *)0);
            glVertexAttribPointer((GLuint)G.loc_uv, 2, GL_FLOAT, GL_FALSE,
                                  4 * sizeof(GLfloat),
                                  (void *)(2 * sizeof(GLfloat)));
            glEnableVertexAttribArray((GLuint)G.loc_pos);
            glEnableVertexAttribArray((GLuint)G.loc_uv);
            glDrawElements(GL_TRIANGLES, IDX, GL_UNSIGNED_SHORT, (void *)0);
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
    eglSwapBuffers(G.dpy, G.surf);   /* single-buffer: acts as a flush */
    if (G.single_buffer) glFinish();
}

static void *render_thread(void *arg) {
    (void)arg;
    uint32_t done_seq = 0;
    for (;;) {
        pthread_mutex_lock(&G.lock);
        while (!G.win_changed && G.frame_seq == done_seq && !G.calib_staged)
            pthread_cond_wait(&G.cond, &G.lock);

        ANativeWindow *new_win = NULL;
        int win_changed = G.win_changed;
        if (win_changed) {
            new_win = G.pending_win;
            G.pending_win = NULL;
            G.win_changed = 0;
        }
        int rebuild = G.calib_staged;
        G.calib_staged = 0;
        int mode = G.frame_mode;
        uint32_t seq = G.frame_seq;
        int64_t submitted = G.submit_ms;
        pthread_mutex_unlock(&G.lock);

        if (win_changed) {
            egl_teardown_surface();
            if (new_win && egl_bind_window(new_win) != 0) {
                LOGE("glasses renderer unavailable on this surface");
                ANativeWindow_release(new_win);
            }
        }
        if (rebuild && G.surf != EGL_NO_SURFACE) build_meshes();
        else if (rebuild) G.mesh_ready = 0;   /* build after surface arrives */

        if (seq != done_seq && G.surf != EGL_NO_SURFACE && mode != MODE_NONE) {
            int64_t t0 = now_ms64();
            render_frame(mode);
            int64_t t1 = now_ms64();
            done_seq = seq;
            G.frames++;
            G.render_ms_acc += (double)(t1 - t0);
            G.wait_ms_acc += (double)(t0 - submitted);
            if (G.stat_t0 == 0) G.stat_t0 = t1;
            if (t1 - G.stat_t0 >= 1000) {
                LOGI("glasses present: %d fps, wait %.1f ms, render %.1f ms (%s)",
                     G.frames, G.wait_ms_acc / G.frames,
                     G.render_ms_acc / G.frames,
                     G.single_buffer ? "front-buffer" : "double-buffered");
                G.frames = 0;
                G.render_ms_acc = G.wait_ms_acc = 0;
                G.stat_t0 = t1;
            }
        } else if (seq != done_seq) {
            done_seq = seq;                    /* nothing to draw on */
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

void xr_gles_submit_eyes(const uint8_t *left, const uint8_t *right) {
    pthread_mutex_lock(&G.lock);
    ensure_thread();
    memcpy(G.eye_img[0], left, sizeof G.eye_img[0]);
    memcpy(G.eye_img[1], right, sizeof G.eye_img[1]);
    G.frame_mode = MODE_EYES;
    G.frame_seq++;
    G.submit_ms = now_ms64();
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
