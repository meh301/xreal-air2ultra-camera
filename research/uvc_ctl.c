// UVC control get/set over libusb for the XREAL cameras.
// cc uvc_ctl.c -o uvc_ctl -I/opt/homebrew/include -L/opt/homebrew/lib -lusb-1.0
// usage:
//   uvc_ctl get <unitId> <cs> <ifnum> <len>
//   uvc_ctl set <unitId> <cs> <ifnum> <hexdata>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libusb-1.0/libusb.h>

#define GET_CUR 0x81
#define GET_MIN 0x82
#define GET_MAX 0x83
#define GET_RES 0x84
#define GET_INFO 0x86
#define GET_DEF 0x87
#define SET_CUR 0x01

static libusb_device_handle *h;

static int ctl(int req, int cs, int unit, int ifnum, unsigned char *data, int len) {
    int type = (req == SET_CUR) ? (LIBUSB_ENDPOINT_OUT | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE)
                                : (LIBUSB_ENDPOINT_IN  | LIBUSB_REQUEST_TYPE_CLASS | LIBUSB_RECIPIENT_INTERFACE);
    return libusb_control_transfer(h, type, req, cs << 8, (unit << 8) | ifnum, data, len, 1000);
}

int main(int argc, char **argv) {
    if (argc < 5) { printf("usage: get/set <unit> <cs> <ifnum> ...\n"); return 1; }
    libusb_context *ctx; libusb_init(&ctx);
    h = libusb_open_device_with_vid_pid(ctx, 0x3318, 0x0426);
    if (!h) { printf("open fail\n"); return 1; }
    // don't claim (macOS UVC driver owns streaming); control transfers target ep0
    int unit = strtol(argv[2], 0, 0), cs = strtol(argv[3], 0, 0), ifn = strtol(argv[4], 0, 0);
    unsigned char buf[64] = {0};

    if (!strcmp(argv[1], "get")) {
        int len = argc > 5 ? atoi(argv[5]) : 2;
        struct { const char *n; int r; } reqs[] = {{"CUR",GET_CUR},{"MIN",GET_MIN},{"MAX",GET_MAX},{"RES",GET_RES},{"DEF",GET_DEF},{"INFO",GET_INFO}};
        for (int k = 0; k < 6; k++) {
            memset(buf, 0, sizeof buf);
            int rl = (reqs[k].r == GET_INFO) ? 1 : len;
            int n = ctl(reqs[k].r, cs, unit, ifn, buf, rl);
            printf("%s: ", reqs[k].n);
            if (n < 0) printf("err %s\n", libusb_error_name(n));
            else { for (int i = 0; i < n; i++) printf("%02x ", buf[i]); printf("\n"); }
        }
    } else { // set
        unsigned char data[64]; int len = 0;
        const char *hex = argv[5];
        for (int i = 0; hex[i] && hex[i+1]; i += 2) { unsigned v; sscanf(hex+i, "%2x", &v); data[len++] = v; }
        int n = ctl(SET_CUR, cs, unit, ifn, data, len);
        printf("SET_CUR unit=%d cs=0x%x if=%d data=%s -> %s\n", unit, cs, ifn, hex, n < 0 ? libusb_error_name(n) : "ok");
    }
    libusb_close(h); libusb_exit(ctx);
    return 0;
}
