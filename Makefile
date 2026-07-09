SWIFTC = swiftc -O

all: preview_clean glasses_passthrough xreal_cam enumerate

preview_clean: src/preview_clean.swift
	$(SWIFTC) $< -o $@ -framework AVFoundation -framework AppKit

glasses_passthrough: src/glasses_passthrough.swift
	$(SWIFTC) $< -o $@ -framework AVFoundation -framework AppKit

xreal_cam: src/xreal_cam.swift
	$(SWIFTC) $< -o $@

enumerate: src/enumerate.swift
	$(SWIFTC) $< -o $@

clean:
	rm -f preview_clean glasses_passthrough xreal_cam enumerate

.PHONY: all clean
