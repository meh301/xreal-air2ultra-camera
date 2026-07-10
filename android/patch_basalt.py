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
"""

import sys
from pathlib import Path

VT = Path(__file__).resolve().parent / \
    "app/src/main/cpp/third_party/basalt/src/vit/vit_tracker.cpp"

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


def main():
    src = VT.read_text(encoding="utf-8")
    changed = 0
    for old, new in REPLACEMENTS:
        if new in src:
            continue                       # already applied
        if old not in src:
            print(f"patch_basalt: TARGET NOT FOUND:\n{old}\n")
            return 1
        src = src.replace(old, new, 1)
        changed += 1
    if changed:
        VT.write_text(src, encoding="utf-8", newline="\n")
    print(f"patch_basalt: {changed} patch(es) applied, "
          f"{len(REPLACEMENTS) - changed} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
