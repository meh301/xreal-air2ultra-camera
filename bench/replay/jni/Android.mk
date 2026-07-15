# Standalone ndk-build project for the on-device replay binary (arm64).
# Built OUTSIDE the app's gradle build:
#   ndk-build -C bench/replay NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=jni/Android.mk \
#             XR_OW=<W> XR_OH=<H>
# (bench/build_replay.ps1 wraps this per dataset resolution.)
LOCAL_PATH := $(call my-dir)
CPP := $(LOCAL_PATH)/../../../android/app/src/main/cpp

include $(CLEAR_VARS)
LOCAL_MODULE := xr_replay
LOCAL_SRC_FILES := ../xr_replay_main.c \
    ../../../android/app/src/main/cpp/xr_slam.c \
    ../../../android/app/src/main/cpp/xr_map.c \
    ../../../android/app/src/main/cpp/xr_xfeat.c \
    ../../../android/app/src/main/cpp/xr_liteanystereo.c
LOCAL_C_INCLUDES := $(CPP) $(CPP)/third_party/basalt/thirdparty/vit
LOCAL_CFLAGS := -O3 -std=c11 -Wall -DXR_OW=$(XR_OW) -DXR_OH=$(XR_OH)
LOCAL_LDLIBS := -llog -ldl
include $(BUILD_EXECUTABLE)
