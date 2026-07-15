#!/usr/bin/env python3
"""Bench-variant Basalt patch: ONLY patch #1 from android/patch_basalt.py
(--cam-calib optional, calibration stays programmatic). Patches #2/#3
(non-blocking drop-on-backpressure pushes) are deliberately OMITTED: the
offline replay harness WANTS stock blocking queues — natural flow control,
as-fast-as-possible feeding, zero drops, deterministic completeness. This
also makes the VIO baseline a faithful reproduction of the Basalt lineage
used by the MSD (IROS 2025) and VOCA (arXiv 2607.00189) papers.

Usage: patch_basalt_bench.py <path-to-basalt-clone>
"""

import sys
from pathlib import Path

REPLACEMENTS = [
    ('used for simulation.")->required();',
     'used for simulation.");'),
    ('    load_calibration_data(cam_calib_path);',
     '    if (!cam_calib_path.empty()) load_calibration_data(cam_calib_path);'),
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    vt = Path(sys.argv[1]) / "src/vit/vit_tracker.cpp"
    src = vt.read_text(encoding="utf-8")
    changed = 0
    for old, new in REPLACEMENTS:
        if new in src:
            continue
        if old not in src:
            print(f"patch_basalt_bench: TARGET NOT FOUND:\n{old}\n")
            return 1
        src = src.replace(old, new, 1)
        changed += 1
    if changed:
        vt.write_text(src, encoding="utf-8", newline="\n")
    print(f"patch_basalt_bench: {changed} applied, "
          f"{len(REPLACEMENTS) - changed} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
