/* sys/system_properties.h shim for non-Android builds: no properties ->
 * xr_xfeat's SoC detection reports non-Qualcomm, so the NPU path stays off
 * and XFeat runs on the ORT CPU EP (matching the app's CPU fallback). */
#ifndef BENCH_SYSPROP_SHIM_H
#define BENCH_SYSPROP_SHIM_H

#define PROP_VALUE_MAX 92

static inline int __system_property_get(const char *name, char *value) {
    (void)name;
    value[0] = 0;
    return 0;
}

#endif
