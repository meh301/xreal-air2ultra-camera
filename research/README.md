# Research tools

Small tools used while reverse-engineering the device. Not needed for normal use;
kept for anyone who wants to dig further (see [docs/PROTOCOL.md](../docs/PROTOCOL.md)).

| Tool | Purpose |
|------|---------|
| `capture.swift` | Minimal raw UVC frame dumper (`frame_N.raw`, 640×482 bytes). |
| `hid_cmd.swift` | Send a vendor HID MCU command (0xFD protocol, CRC32) and print replies. |
| `hid_read.swift` | Listen on the vendor HID interfaces. |
| `cam_probe.swift` | Hold the HID channel open while watching camera statistics (for command scanning). |
| `controls.swift` | Dump/set AVFoundation exposure-related properties for the device. |
| `lighttest.swift` | Light-response test (cover lens / point at light, log image means). |
| `uvc_ctl.c` | Get/set UVC controls via libusb control transfers: `uvc_ctl set <unit> <cs> <ifnum> <hex>`. |
| `get_uvc_desc.c` | Dump the USB configuration descriptor (UVC unit topology). |
| `expsweep.sh` | Sweep ExposureTimeAbs while capturing. |

C tools build with libusb: `cc uvc_ctl.c -o uvc_ctl $(pkg-config --cflags --libs libusb-1.0)`.
Swift tools: `swiftc <name>.swift -o <name>`.
