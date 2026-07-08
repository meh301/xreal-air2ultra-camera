// Dump the full raw USB configuration descriptor (incl. UVC class-specific descriptors).
// cc get_uvc_desc.c -o get_uvc_desc -I/opt/homebrew/include -L/opt/homebrew/lib -lusb-1.0
#include <stdio.h>
#include <libusb-1.0/libusb.h>

int main(void) {
    libusb_context *ctx = NULL;
    if (libusb_init(&ctx) < 0) { printf("init fail\n"); return 1; }
    libusb_device_handle *h = libusb_open_device_with_vid_pid(ctx, 0x3318, 0x0426);
    if (!h) { printf("open fail (device busy or no permission)\n"); libusb_exit(ctx); return 1; }

    unsigned char buf[4096];
    // GET_DESCRIPTOR: type=CONFIG(0x02), index 0
    int n = libusb_control_transfer(h,
        LIBUSB_ENDPOINT_IN | LIBUSB_REQUEST_TYPE_STANDARD | LIBUSB_RECIPIENT_DEVICE,
        LIBUSB_REQUEST_GET_DESCRIPTOR, (0x02 << 8) | 0, 0, buf, sizeof(buf), 1000);
    if (n < 0) { printf("control transfer fail: %s\n", libusb_error_name(n)); libusb_close(h); libusb_exit(ctx); return 1; }
    printf("CONFIG_DESCRIPTOR_LEN %d\n", n);
    for (int i = 0; i < n; i++) printf("%02x", buf[i]);
    printf("\n");
    libusb_close(h); libusb_exit(ctx);
    return 0;
}
