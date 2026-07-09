/* Hand-written stand-in for the CMake-generated libuvc_config.h (we build
 * libuvc v0.0.7 with ndk-build instead of CMake). LIBUVC_HAS_JPEG stays
 * undefined: the XREAL stream is uncompressed, so frame-mjpeg.c is not built.
 */
#ifndef LIBUVC_CONFIG_H
#define LIBUVC_CONFIG_H

#define LIBUVC_VERSION_MAJOR 0
#define LIBUVC_VERSION_MINOR 0
#define LIBUVC_VERSION_PATCH 7
#define LIBUVC_VERSION_STR "0.0.7"
#define LIBUVC_VERSION_INT                      \
  ((LIBUVC_VERSION_MAJOR << 16) |               \
   (LIBUVC_VERSION_MINOR << 8) |                \
   (LIBUVC_VERSION_PATCH))

/** @brief Test whether libuvc is new enough */
#define LIBUVC_VERSION_GTE(major, minor, patch) \
  (LIBUVC_VERSION_INT >= (((major) << 16) | ((minor) << 8) | (patch)))

/* #undef LIBUVC_HAS_JPEG */

#endif
