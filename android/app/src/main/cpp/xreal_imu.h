/* xreal_imu.h — IMU packet parsing + host-side attitude for the Air 2 Ultra.
 * The glasses stream raw gyro/accel only (1 kHz, HID interface 2, see
 * docs/PROTOCOL.md); orientation is fused here with a 6-axis Madgwick filter.
 */
#ifndef XREAL_IMU_H
#define XREAL_IMU_H

#include <stddef.h>
#include <stdint.h>

enum { XR_IMU_INTERFACE = 2, XR_IMU_REPORT = 512, XR_IMU_CMD_LEN = 64 };

typedef struct {
    uint64_t ts_ns;
    uint32_t counter;      /* 1024 Hz ODR tick counter */
    uint16_t temp_raw;
    float gyro_dps[3];
    float accel_g[3];
} xr_imu_sample;

/* Parse one 01-02 stream report; returns 0 on success, -1 otherwise. */
int xr_imu_parse(const uint8_t *report, size_t len, xr_imu_sample *out);

/* Build a 64-byte 0xAA-framed command for the IMU HID channel
 * (cmd 0x19 data 01/00 = stream on/off). */
void xr_imu_command(uint8_t out[XR_IMU_CMD_LEN], uint8_t cmd,
                    const uint8_t *data, size_t n);

/* Build a 64-byte 0xFD-framed command for the MCU HID channel (interface 0):
 * cmd 0x08 data {mode} sets the display mode - 1 = mirror, 3 = SBS 60 Hz
 * (per-eye stereo, needed for passthrough), 4 = SBS 72 Hz, 9 = SBS 90 Hz. */
enum { XR_MCU_INTERFACE = 0, XR_MCU_CMD_LEN = 64 };
enum { XR_DISPLAY_MIRROR = 1, XR_DISPLAY_SBS_60 = 3, XR_DISPLAY_SBS_72 = 4,
       XR_DISPLAY_SBS_90 = 9 };
void xr_mcu_command(uint8_t out[XR_MCU_CMD_LEN], uint16_t cmd,
                    const uint8_t *data, size_t n);

/* 6-axis Madgwick AHRS + automatic gyro-bias capture (first ~1000 samples
 * are averaged as the bias, then tracking starts with an aggressive beta
 * that settles after one second). q is sensor->earth (w, x, y, z). */
typedef struct {
    float q[4];
    float beta;
    float bias[3];
    double bias_acc[3];
    int n;                 /* samples fed */
    uint64_t last_ts;
} xr_ahrs;

void xr_ahrs_init(xr_ahrs *a);
/* Feed one sample; returns 1 once q is valid (bias capture done). */
int xr_ahrs_feed(xr_ahrs *a, const xr_imu_sample *s);

#endif
