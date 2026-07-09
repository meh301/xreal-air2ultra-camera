/* xreal_imu.c — see xreal_imu.h. Mirrors python/xreal_imu.py + xreal_ahrs.py. */
#include "xreal_imu.h"

#include <math.h>
#include <string.h>

/* ---- little-endian readers ---------------------------------------------- */
static uint16_t rd16(const uint8_t *b) { return (uint16_t)(b[0] | b[1] << 8); }
static uint32_t rd32(const uint8_t *b) {
    return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 |
           (uint32_t)b[3] << 24;
}
static uint64_t rd64(const uint8_t *b) {
    return (uint64_t)rd32(b) | (uint64_t)rd32(b + 4) << 32;
}
static int32_t rd24s(const uint8_t *b) {
    int32_t v = (int32_t)b[0] | (int32_t)b[1] << 8 | (int32_t)b[2] << 16;
    return (v & 0x800000) ? v - 0x1000000 : v;
}

int xr_imu_parse(const uint8_t *r, size_t len, xr_imu_sample *out) {
    if (len < 58 || r[0] != 0x01 || r[1] != 0x02) return -1;
    out->temp_raw = rd16(r + 2);
    out->ts_ns = rd64(r + 4);
    float gm = rd16(r + 12), gd = (float)rd32(r + 14);
    float am = rd16(r + 27), ad = (float)rd32(r + 29);
    if (gd <= 0 || ad <= 0) return -1;
    for (int i = 0; i < 3; i++) {
        out->gyro_dps[i] = rd24s(r + 18 + 3 * i) * gm / gd;
        out->accel_g[i] = rd24s(r + 33 + 3 * i) * am / ad;
    }
    out->counter = rd32(r + 54);
    return 0;
}

/* ---- 0xAA command framing (crc32 = zlib polynomial) ----------------------- */
static uint32_t crc_table[256];
static void crc_init(void) {
    if (crc_table[1]) return;
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
}
static uint32_t crc32z(const uint8_t *buf, size_t n) {
    crc_init();
    uint32_t r = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) r = (r >> 8) ^ crc_table[(r ^ buf[i]) & 0xFF];
    return r ^ 0xFFFFFFFFu;
}

void xr_imu_command(uint8_t out[XR_IMU_CMD_LEN], uint8_t cmd,
                    const uint8_t *data, size_t n) {
    memset(out, 0, XR_IMU_CMD_LEN);
    uint16_t length = (uint16_t)(n + 3);
    out[0] = 0xAA;
    out[5] = (uint8_t)(length & 0xFF);
    out[6] = (uint8_t)(length >> 8);
    out[7] = cmd;
    if (n) memcpy(out + 8, data, n);
    uint32_t crc = crc32z(out + 5, length);
    out[1] = (uint8_t)crc;
    out[2] = (uint8_t)(crc >> 8);
    out[3] = (uint8_t)(crc >> 16);
    out[4] = (uint8_t)(crc >> 24);
}

void xr_mcu_command(uint8_t out[XR_MCU_CMD_LEN], uint16_t cmd,
                    const uint8_t *data, size_t n) {
    /* fd | crc32 | len u16 (= n+17) | request_id u32 | timestamp u32 |
     * cmd u16 | reserved[5] | data (docs/PROTOCOL.md, "Vendor HID protocol") */
    memset(out, 0, XR_MCU_CMD_LEN);
    uint16_t length = (uint16_t)(n + 17);
    out[0] = 0xFD;
    out[5] = (uint8_t)(length & 0xFF);
    out[6] = (uint8_t)(length >> 8);
    out[7] = 0x37; out[8] = 0x13;              /* request_id 0x1337 */
    out[15] = (uint8_t)(cmd & 0xFF);
    out[16] = (uint8_t)(cmd >> 8);
    if (n) memcpy(out + 22, data, n);
    uint32_t crc = crc32z(out + 5, length);
    out[1] = (uint8_t)crc;
    out[2] = (uint8_t)(crc >> 8);
    out[3] = (uint8_t)(crc >> 16);
    out[4] = (uint8_t)(crc >> 24);
}

/* ---- Madgwick 6-axis (port of the reference IMU update) ------------------- */
void xr_ahrs_init(xr_ahrs *a) {
    memset(a, 0, sizeof *a);
    a->q[0] = 1.0f;
    a->beta = 0.5f;        /* aggressive until converged */
}

static void madgwick_update(xr_ahrs *m, float gx, float gy, float gz,
                            float ax, float ay, float az, float dt) {
    float q1 = m->q[0], q2 = m->q[1], q3 = m->q[2], q4 = m->q[3];

    float qd1 = 0.5f * (-q2 * gx - q3 * gy - q4 * gz);
    float qd2 = 0.5f * (q1 * gx + q3 * gz - q4 * gy);
    float qd3 = 0.5f * (q1 * gy - q2 * gz + q4 * gx);
    float qd4 = 0.5f * (q1 * gz + q2 * gy - q3 * gx);

    float norm = sqrtf(ax * ax + ay * ay + az * az);
    if (norm > 0.0f) {
        ax /= norm; ay /= norm; az /= norm;
        float _2q1 = 2 * q1, _2q2 = 2 * q2, _2q3 = 2 * q3, _2q4 = 2 * q4;
        float _4q1 = 4 * q1, _4q2 = 4 * q2, _4q3 = 4 * q3;
        float _8q2 = 8 * q2, _8q3 = 8 * q3;
        float q1q1 = q1 * q1, q2q2 = q2 * q2, q3q3 = q3 * q3, q4q4 = q4 * q4;

        float s1 = _4q1 * q3q3 + _2q3 * ax + _4q1 * q2q2 - _2q2 * ay;
        float s2 = _4q2 * q4q4 - _2q4 * ax + 4 * q1q1 * q2 - _2q1 * ay - _4q2
                   + _8q2 * q2q2 + _8q2 * q3q3 + _4q2 * az;
        float s3 = 4 * q1q1 * q3 + _2q1 * ax + _4q3 * q4q4 - _2q4 * ay - _4q3
                   + _8q3 * q2q2 + _8q3 * q3q3 + _4q3 * az;
        float s4 = 4 * q2q2 * q4 - _2q2 * ax + 4 * q3q3 * q4 - _2q3 * ay;
        float n = sqrtf(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4);
        if (n > 0.0f) {
            float b = m->beta / n;
            qd1 -= b * s1; qd2 -= b * s2; qd3 -= b * s3; qd4 -= b * s4;
        }
    }
    q1 += qd1 * dt; q2 += qd2 * dt; q3 += qd3 * dt; q4 += qd4 * dt;
    float n = sqrtf(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4);
    if (n <= 0.0f) n = 1.0f;
    m->q[0] = q1 / n; m->q[1] = q2 / n; m->q[2] = q3 / n; m->q[3] = q4 / n;
}

int xr_ahrs_feed(xr_ahrs *a, const xr_imu_sample *s) {
    enum { CAPTURE_N = 1000, SETTLE_N = 2000 };
    if (a->n < CAPTURE_N) {                       /* bias capture phase */
        for (int i = 0; i < 3; i++) a->bias_acc[i] += s->gyro_dps[i];
        if (++a->n == CAPTURE_N) {
            for (int i = 0; i < 3; i++)
                a->bias[i] = (float)(a->bias_acc[i] / CAPTURE_N);
            a->last_ts = s->ts_ns;
        }
        return 0;
    }
    float dt = (s->ts_ns - a->last_ts) / 1e9f;
    a->last_ts = s->ts_ns;
    if (dt < 0.0f) dt = 0.0f;
    if (dt > 0.005f) dt = 0.005f;
    if (++a->n == SETTLE_N) a->beta = 0.02f;      /* converged: settle down —
        kept low so accel noise doesn't wobble the timewarp at rest */
    float d2r = (float)(M_PI / 180.0);
    madgwick_update(a,
                    (s->gyro_dps[0] - a->bias[0]) * d2r,
                    (s->gyro_dps[1] - a->bias[1]) * d2r,
                    (s->gyro_dps[2] - a->bias[2]) * d2r,
                    s->accel_g[0], s->accel_g[1], s->accel_g[2], dt);
    return 1;
}
