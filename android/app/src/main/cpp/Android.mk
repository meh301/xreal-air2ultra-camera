# ndk-build makefile: libusb (shared, LGPL — kept as its own .so) + libuvc
# (static, BSD) + our JNI library. libusb/libuvc sources are cloned into
# third_party/ by android/fetch_deps.ps1 / fetch_deps.sh — run one of those
# once before building.

LOCAL_PATH := $(call my-dir)
# libusb.mk reassigns LOCAL_PATH via my-dir when included, so keep our own
# copy and include that file only at the very end
XREAL_LOCAL_PATH := $(LOCAL_PATH)

ifeq ($(wildcard $(LOCAL_PATH)/third_party/libusb/libusb/libusb.h),)
$(error third_party sources missing — run android/fetch_deps.ps1 (Windows) or android/fetch_deps.sh first)
endif

# ---- libuvc.a ----------------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_MODULE := uvc
UVC_SRC := third_party/libuvc/src
LOCAL_SRC_FILES := \
    $(UVC_SRC)/ctrl.c \
    $(UVC_SRC)/ctrl-gen.c \
    $(UVC_SRC)/device.c \
    $(UVC_SRC)/diag.c \
    $(UVC_SRC)/frame.c \
    $(UVC_SRC)/init.c \
    $(UVC_SRC)/misc.c \
    $(UVC_SRC)/stream.c
# frame-mjpeg.c is omitted: the XREAL stream is uncompressed, no libjpeg needed
LOCAL_C_INCLUDES := \
    $(LOCAL_PATH)/third_party/libuvc/include \
    $(LOCAL_PATH)/libuvc_config \
    $(LOCAL_PATH)/third_party/libusb/libusb
LOCAL_EXPORT_C_INCLUDES := \
    $(LOCAL_PATH)/third_party/libuvc/include \
    $(LOCAL_PATH)/libuvc_config
LOCAL_CFLAGS := -O2 -Wno-unused-parameter
LOCAL_SHARED_LIBRARIES := libusb1.0
include $(BUILD_STATIC_LIBRARY)

# ---- libxrealcam.so (JNI) ------------------------------------------------------
include $(CLEAR_VARS)
LOCAL_PATH := $(XREAL_LOCAL_PATH)
LOCAL_MODULE := xrealcam
LOCAL_SRC_FILES := xr_map.c xr_slam.c xr_stereo.c xr_track.c xreal_align.c \
    xreal_core.c xreal_gles.c xreal_imu.c xreal_jni.c
LOCAL_STATIC_LIBRARIES := uvc
LOCAL_SHARED_LIBRARIES := libusb1.0
# vit_interface.h comes from the basalt clone (fetch_deps / build_basalt);
# libbasalt.so itself is dlopen'd at runtime, never linked
LOCAL_C_INCLUDES := $(LOCAL_PATH)/third_party/basalt/thirdparty/vit
LOCAL_CFLAGS := -O3 -std=c11 -Wall
LOCAL_ARM_NEON := true      # armeabi-v7a; arm64 has NEON unconditionally
LOCAL_LDLIBS := -llog -landroid -lEGL -lGLESv2 -ldl
include $(BUILD_SHARED_LIBRARY)

# ---- libusb1.0.so (upstream-maintained module definition; include last, it
# ---- clobbers LOCAL_PATH) ------------------------------------------------------
include $(XREAL_LOCAL_PATH)/third_party/libusb/android/jni/libusb.mk
