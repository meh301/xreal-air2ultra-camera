SWIFTC = swiftc -O

all: preview_clean xreal_cam enumerate

preview_clean: src/preview_clean.swift
	$(SWIFTC) $< -o $@ -framework AVFoundation -framework AppKit

xreal_cam: src/xreal_cam.swift
	$(SWIFTC) $< -o $@

enumerate: src/enumerate.swift
	$(SWIFTC) $< -o $@

clean:
	rm -f preview_clean xreal_cam enumerate

.PHONY: all clean
