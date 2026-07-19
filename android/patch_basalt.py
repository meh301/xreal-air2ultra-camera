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

import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE / "app/src/main/cpp/third_party/basalt"
VT = _ROOT / "src/vit/vit_tracker.cpp"
SKV = _ROOT / "src/vi_estimator/sqrt_keypoint_vio.cpp"
VTH = _ROOT / "include/basalt/vit/vit_tracker.hpp"

# 5. The estimator that produced the published benchmark numbers. The map->VIO
#    coupling (XR_TIGHT prior, XR_LMFACT factors, XR_LMINJ injection, XR_LMTRACK,
#    XR_LMMARG, XR_DELAYMARG re-advance) lives in the estimator, and the upstream
#    clone has none of it — a stock build silently produces a libbasalt.so whose
#    vit_tracker_xreal_* symbols do not exist, so every gate is a no-op no matter
#    what the environment says. These three files are copied verbatim from the
#    benchmark containers (/root/xreal/bench/container/basalt-linux) so a fresh
#    clone reproduces the benchmarked estimator rather than a bare one. They are
#    installed BEFORE the string patches below, which then re-apply XR_OUTFILT on
#    top (the containers do not carry OUTFILT — it was a separate A/B and is our
#    ship decision, default-on, disabled with XR_OUTFILT=0).
_OVERLAY = {
    "vio_estimator.h":      _ROOT / "include/basalt/vi_estimator/vio_estimator.h",
    "sqrt_keypoint_vio.h":  _ROOT / "include/basalt/vi_estimator/sqrt_keypoint_vio.h",
    "sqrt_keypoint_vio.cpp": SKV,
}

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


VTH_REPLACEMENTS = [
    # stage-8 / stage-12 Tracker method declarations (the clone has neither)
    ("  void xreal_seed_keypoints(int64_t t_ns, int cam, const float* uv, int n);",
     "  void xreal_seed_keypoints(int64_t t_ns, int cam, const float* uv, int n);\n\n"
     "  /* XREAL stage-8: inject verified closure landmarks into the live lmdb as\n"
     "   * optimizable landmarks with streaming observations (XR_LMINJ). */\n"
     "  void xreal_inject_landmarks(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world,\n"
     "                              const int32_t *ids, int n);\n\n"
     "  /* XREAL stage-12: arm a delayed-marg re-advance on a closure trigger. */\n"
     "  void xreal_readvance();"),
]

VT_XREAL_REPLACEMENTS = [
    # the two Tracker methods the overlay estimator exposes
    ("}  // namespace basalt::vit_implementation",
     "void Tracker::xreal_inject_landmarks(int64_t t_ns, int cam_id, const float *uv, const float *xyz_world,\n"
     "                                     const int32_t *ids, int n) {\n"
     "  if (!impl_ || !impl_->vio) return;\n"
     "  impl_->vio->setXrInjectLandmarks(t_ns, cam_id, uv, xyz_world, ids, n);\n"
     "}\n\n"
     "void Tracker::xreal_readvance() {\n"
     "  if (!impl_ || !impl_->vio) return;\n"
     "  impl_->vio->setXrReadvance();\n"
     "}\n\n"
     "}  // namespace basalt::vit_implementation"),
    # and their dlsym-reachable C entry points (xr_slam.c resolves all five)
    ("/* XREAL extension: reachable via dlsym, no header coupling required. */",
     "/* XREAL stage-8 extension: closure-landmark injection into the lmdb. */\n"
     "extern \"C\" vit_result_t vit_tracker_xreal_inject_landmarks(vit_tracker_t *tracker, int64_t t_ns, int32_t cam_id,\n"
     "                                                           const float *uv, const float *xyz_world,\n"
     "                                                           const int32_t *ids, int32_t n) {\n"
     "  if (!tracker || !uv || !xyz_world || !ids || n <= 0) return VIT_ERROR_INVALID_VALUE;\n"
     "  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));\n"
     "  t->xreal_inject_landmarks(t_ns, cam_id, uv, xyz_world, ids, n);\n"
     "  return VIT_SUCCESS;\n"
     "}\n\n"
     "/* XREAL stage-12 extension: closure-triggered re-advance. */\n"
     "extern \"C\" vit_result_t vit_tracker_xreal_readvance(vit_tracker_t *tracker) {\n"
     "  if (!tracker) return VIT_ERROR_INVALID_VALUE;\n"
     "  auto *t = static_cast<basalt::vit_implementation::Tracker *>(static_cast<vit::Tracker *>(tracker));\n"
     "  t->xreal_readvance();\n"
     "  return VIT_SUCCESS;\n"
     "}\n\n"
     "/* XREAL extension: reachable via dlsym, no header coupling required. */"),
]


def _install_overlay():
    """Copy the benchmarked estimator over the clone's. Idempotent by content."""
    src_dir = _HERE / "basalt_overlay"
    n = 0
    for name, dst in _OVERLAY.items():
        src = src_dir / name
        if not src.exists():
            print(f"patch_basalt: OVERLAY MISSING: {src}")
            return None
        if not dst.exists():
            print(f"patch_basalt: TARGET MISSING: {dst}")
            return None
        if src.read_bytes() != dst.read_bytes():
            shutil.copyfile(src, dst)
            n += 1
    print(f"patch_basalt: estimator overlay: {n} installed, {len(_OVERLAY) - n} current")
    return n


def main():
    if _install_overlay() is None:
        return 1
    total = 0
    for path, reps in ((VTH, VTH_REPLACEMENTS), (VT, VT_XREAL_REPLACEMENTS),
                       (VT, REPLACEMENTS), (SKV, SKV_REPLACEMENTS)):
        n = _apply(path, reps)
        if n is None:
            return 1
        total += n
        print(f"patch_basalt: {path.name}: {n} applied, {len(reps) - n} already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
