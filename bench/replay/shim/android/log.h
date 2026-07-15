/* android/log.h shim for non-Android (container/desktop) builds of the
 * replay harness: logcat calls become stderr lines. Only what our sources
 * use is provided. */
#ifndef BENCH_ANDROID_LOG_SHIM_H
#define BENCH_ANDROID_LOG_SHIM_H

#include <stdio.h>

typedef enum {
    ANDROID_LOG_UNKNOWN = 0, ANDROID_LOG_DEFAULT, ANDROID_LOG_VERBOSE,
    ANDROID_LOG_DEBUG, ANDROID_LOG_INFO, ANDROID_LOG_WARN,
    ANDROID_LOG_ERROR, ANDROID_LOG_FATAL, ANDROID_LOG_SILENT
} android_LogPriority;

#define __android_log_print(prio, tag, ...) \
    (fprintf(stderr, "[%s] ", (tag)), fprintf(stderr, __VA_ARGS__), \
     fprintf(stderr, "\n"), 1)

static inline int __android_log_write(int prio, const char *tag,
                                      const char *text) {
    (void)prio;
    return fprintf(stderr, "[%s] %s", tag, text);
}

#endif
