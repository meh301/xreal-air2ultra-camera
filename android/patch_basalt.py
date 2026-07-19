#!/usr/bin/env python3
"""Local patches to the basalt clone (idempotent; run by build_basalt.ps1).

1. --cam-calib becomes optional so a unified config file (num-threads,
   VioConfig path) can be used together with PROGRAMMATIC calibration.
2. The VIT push entry points become NON-BLOCKING: basalt's bounded queues
   otherwise block the caller when the pipeline runs behind — which stalls
   the app's UVC callback / IMU threads and hard-freezes everything.
   Realtime capture must drop, not wait.
3. stop() drains queues before pushing the shutdown sentinels for the same
   reason.
4. XR_OUTFILT ship (audit finding #3, benchmark-verified 2026-07-19): re-enable
   the dormant stock filterOutliers(3.0 px, min 2 obs) after convergence,
   before marginalize() — deletes gross/dynamic-object outliers the Huber cost
   only softens (drive3 VIO H -60% / V -27%, clean fleet-wide). Default-ON;
   XR_OUTFILT=0 disables for A/B. TRIPAR was tested and NOT shipped.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent / "app/src/main/cpp/third_party/basalt"
VT = _ROOT / "src/vit/vit_tracker.cpp"
SKV = _ROOT / "src/vi_estimator/sqrt_keypoint_vio.cpp"

REPLACEMENTS = [
    # 1. cam-calib optional
    ('used for simulation.")->required();',
     'used for simulation.");'),
    ('    load_calibration_data(cam_calib_path);',
     '    if (!cam_calib_path.empty()) load_calibration_data(cam_calib_path);'),
    # 2. non-blocking pushes (drop under backpressure)
    ('    imu_data_queue->push(data);\n'
     '    opt_flow_ptr->input_imu_queue.push(data);',
     '    if (!imu_data_queue->try_push(data)) {\n'
     '      static int dropped_vio = 0;\n'
     '      if (++dropped_vio % 1000 == 1)\n'
     '        std::cout << "[vit] VIO imu queue full: " << dropped_vio << " samples dropped\\n";\n'
     '    }\n'
     '    if (!opt_flow_ptr->input_imu_queue.try_push(data)) {\n'
     '      static int dropped_of = 0;\n'
     '      if (++dropped_of % 1000 == 1)\n'
     '        std::cout << "[vit] OF imu queue full: " << dropped_of << " samples dropped\\n";\n'
     '    }'),
    ('      image_data_queue->push(partial_frame);',
     '      if (!image_data_queue->try_push(partial_frame)) {\n'
     '        static int dropped_frames = 0;\n'
     '        if (++dropped_frames % 30 == 1)\n'
     '          std::cout << "[vit] frontend busy: " << dropped_frames << " frame bundles dropped\\n";\n'
     '      }'),
    # 3. non-blocking shutdown sentinels
    ('    image_data_queue->push(nullptr);\n'
     '    imu_data_queue->push(nullptr);\n'
     '    opt_flow_ptr->input_imu_queue.push(nullptr);',
     '    while (!image_data_queue->try_push(nullptr)) {\n'
     '      OpticalFlowInput::Ptr _d;\n'
     '      image_data_queue->try_pop(_d);\n'
     '    }\n'
     '    while (!imu_data_queue->try_push(nullptr)) {\n'
     '      ImuData<double>::Ptr _d;\n'
     '      imu_data_queue->try_pop(_d);\n'
     '    }\n'
     '    while (!opt_flow_ptr->input_imu_queue.try_push(nullptr)) {\n'
     '      ImuData<double>::Ptr _d;\n'
     '      opt_flow_ptr->input_imu_queue.try_pop(_d);\n'
     '    }'),
]


SKV_REPLACEMENTS = [
    # cstdlib for std::getenv (the XR_OUTFILT ship gate)
    ("#include <memory>\n\n#include <tbb/blocked_range.h>",
     "#include <memory>\n#include <cstdlib>  // XREAL: std::getenv for XR_OUTFILT ship gate\n\n#include <tbb/blocked_range.h>"),
    # re-enable the dormant hard outlier filter after convergence, before marg
    ("    // TODO: call filterOutliers at least once (also for CG version)\n\n"
     "    stats_all_.merge_all(stats);",
     "    // XREAL SHIP (audit #3, benchmark-verified): hard reprojection-outlier\n"
     "    // rejection after convergence, before marginalize(). Deletes gross /\n"
     "    // dynamic-object outliers the Huber cost only softens (drive3 VIO H\n"
     "    // -60% / V -27%). Default-ON; XR_OUTFILT=0 disables for A/B.\n"
     "    {\n"
     "      static const bool xr_outfilt = [] {\n"
     "        const char* e = std::getenv(\"XR_OUTFILT\");\n"
     "        return !(e && *e == '0');\n"
     "      }();\n"
     "      if (xr_outfilt && converged) this->filterOutliers(Scalar(3.0), 2);\n"
     "    }\n\n"
     "    stats_all_.merge_all(stats);"),
]


def _apply(path, replacements):
    """Apply (old,new) replacements to a file idempotently. Returns count or None on error."""
    if not path.exists():
        print(f"patch_basalt: FILE MISSING: {path}")
        return None
    src = path.read_text(encoding="utf-8")
    changed = 0
    for old, new in replacements:
        if new in src:
            continue                       # already applied
        if old not in src:
            print(f"patch_basalt: TARGET NOT FOUND in {path.name}:\n{old}\n")
            return None
        src = src.replace(old, new, 1)
        changed += 1
    if changed:
        path.write_text(src, encoding="utf-8", newline="\n")
    return changed


def main():
    total = 0
    for path, reps in ((VT, REPLACEMENTS), (SKV, SKV_REPLACEMENTS)):
        n = _apply(path, reps)
        if n is None:
            return 1
        total += n
        print(f"patch_basalt: {path.name}: {n} applied, {len(reps) - n} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
