# Builds the native macOS tools. On Windows/Linux use the Python tools in
# python/ instead (no build step), and android/ for the Android app.
ifneq ($(shell uname),Darwin)
$(error The Swift tools are macOS-only. Use python/preview_clean.py (Windows/Linux) or android/ — see README)
endif

SWIFTC = swiftc -O

all: preview_clean xreal_cam xreal_imu enumerate

preview_clean: src/preview_clean.swift
	$(SWIFTC) $< -o $@ -framework AVFoundation -framework AppKit

xreal_cam: src/xreal_cam.swift
	$(SWIFTC) $< -o $@

xreal_imu: src/xreal_imu.swift
	$(SWIFTC) $< -o $@ -framework IOKit -framework CoreFoundation

enumerate: src/enumerate.swift
	$(SWIFTC) $< -o $@

clean:
	rm -f preview_clean xreal_cam xreal_imu enumerate

.PHONY: all clean
