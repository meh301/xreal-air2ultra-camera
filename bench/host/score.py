#!/usr/bin/env python
"""Score an estimated trajectory against ground truth (VOCA conventions).

Usage:
    score.py <est.tum> <gt.tum> --dataset euroc|tumvi|msd [--json out.json]

Metrics (via the `evo` library, Umeyama alignment):
  ATE  = APE translation RMSE in metres after SE(3) alignment, NO scale
         (equivalent to `evo_ape tum gt est -a`).
  RTE  = RPE translation RMSE, --delta 6 --delta_unit f, consecutive pairs
         (NOT --all_pairs), reported in centimetres (m * 100).
  Divergence: ATE > 10 m (1 m for msd) or RTE > 10 cm  =>  both marked inf.
  completion% = est poses within the GT time span / expected frame count,
         where expected = gt_span_seconds * fps + 1 and fps is the dataset
         nominal rate (euroc/tumvi 20, msd 30; override with --fps).

Exit code 0 for any successfully computed score (diverged included);
nonzero only for operational errors (missing/corrupt files, too few poses).
"""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

from pack_common import (ATE_DIVERGE_M, DATASET_FPS, DATASETS,
                         RTE_DIVERGE_CM, die, warn)

RTE_DELTA_FRAMES = 6
DEFAULT_T_MAX_DIFF = 0.01   # s, association tolerance (evo default)


def _load_tum(path, what):
    from evo.tools import file_interface
    p = Path(path)
    if not p.exists():
        die(f"{what} file not found: {p}")
    if p.stat().st_size == 0:
        return None   # empty estimate = total failure, handled by caller
    try:
        traj = file_interface.read_tum_trajectory_file(str(p))
    except Exception as e:
        die(f"failed to parse {what} {p}: {e}")
    if traj.num_poses == 0:
        return None
    return traj


def score(est_path, gt_path, dataset, fps=None, t_max_diff=DEFAULT_T_MAX_DIFF):
    from evo.core import metrics, sync

    if dataset not in DATASETS:
        die(f"--dataset must be one of {DATASETS}")
    fps = float(fps) if fps else DATASET_FPS[dataset]
    ate_thresh_m = ATE_DIVERGE_M[dataset]

    gt = _load_tum(gt_path, "ground truth")
    if gt is None:
        die(f"ground truth {gt_path} is empty")
    est = _load_tum(est_path, "estimate")

    gt_span = float(gt.timestamps[-1] - gt.timestamps[0])
    expected_frames = max(1, int(round(gt_span * fps)) + 1)

    result = {
        "dataset": dataset,
        "est": str(est_path), "gt": str(gt_path),
        "n_gt": int(gt.num_poses),
        "fps_assumed": fps,
        "expected_frames": expected_frames,
        "rte_delta_frames": RTE_DELTA_FRAMES,
        "ate_diverge_m": ate_thresh_m, "rte_diverge_cm": RTE_DIVERGE_CM,
    }

    if est is None:
        result.update(n_est=0, completion_pct=0.0, diverged=True,
                      diverge_reason="empty estimate",
                      ate_m=math.inf, rte_cm=math.inf,
                      ate_raw_m=None, rte_raw_cm=None, n_matched=0)
        return result

    # completion: est poses inside the GT span (half-frame slack at the ends)
    slack = 0.5 / fps
    lo, hi = gt.timestamps[0] - slack, gt.timestamps[-1] + slack
    n_in_span = int(((est.timestamps >= lo) & (est.timestamps <= hi)).sum())
    completion = min(100.0, 100.0 * n_in_span / expected_frames)
    result.update(n_est=int(est.num_poses), completion_pct=completion)

    try:
        gt_s, est_s = sync.associate_trajectories(gt, est,
                                                  max_diff=t_max_diff)
    except Exception as e:
        die(f"timestamp association failed (est/gt time bases disjoint?): {e}")
    result["n_matched"] = int(est_s.num_poses)
    if est_s.num_poses < RTE_DELTA_FRAMES + 2:
        result.update(diverged=True,
                      diverge_reason=f"only {est_s.num_poses} matched poses",
                      ate_m=math.inf, rte_cm=math.inf,
                      ate_raw_m=None, rte_raw_cm=None)
        return result

    # ATE: Umeyama SE(3) alignment, no scale (evo_ape -a)
    est_aligned = copy.deepcopy(est_s)
    est_aligned.align(gt_s, correct_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((gt_s, est_aligned))
    ate_m = float(ape.get_statistic(metrics.StatisticsType.rmse))

    # RTE: evo_rpe --delta 6 --delta_unit f --pose_relation trans_part
    # (no --all_pairs => consecutive non-overlapping delta pairs)
    rpe = metrics.RPE(metrics.PoseRelation.translation_part,
                      delta=RTE_DELTA_FRAMES, delta_unit=metrics.Unit.frames,
                      all_pairs=False)
    rpe.process_data((gt_s, est_s))
    rte_cm = float(rpe.get_statistic(metrics.StatisticsType.rmse)) * 100.0

    # Divergence is ATE-only. The RTE>10cm gate was formally rejected in the
    # ledger (it marked healthy near-threshold runs inf and diverged from the
    # site's score_one, so fastbench and site verdicts were incomparable);
    # RTE stays a reported metric but never gates. (Forensic review 2026-07-19,
    # numbers finding: "two divergent scorers".)
    diverged = (ate_m > ate_thresh_m)
    reason = ""
    if diverged:
        reason = f"ATE {ate_m:.3f} m > {ate_thresh_m} m"
    result.update(diverged=diverged, diverge_reason=reason,
                  ate_raw_m=ate_m, rte_raw_cm=rte_cm,
                  ate_m=math.inf if diverged else ate_m,
                  rte_cm=math.inf if diverged else rte_cm)
    return result


def human_line(r):
    if r["diverged"]:
        raw = ""
        if r.get("ate_raw_m") is not None:
            raw = f" (raw ATE {r['ate_raw_m']:.3f} m, RTE {r['rte_raw_cm']:.2f} cm)"
        return (f"ATE inf | RTE inf | completion {r['completion_pct']:.1f}% | "
                f"DIVERGED: {r['diverge_reason']}{raw}")
    return (f"ATE {r['ate_m']:.4f} m | RTE {r['rte_cm']:.3f} cm | "
            f"completion {r['completion_pct']:.1f}% | OK "
            f"({r['n_matched']}/{r['n_est']} poses matched)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("est", help="estimated trajectory, TUM format")
    ap.add_argument("gt", help="ground truth trajectory, TUM format")
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--json", default=None, help="write result JSON here")
    ap.add_argument("--fps", type=float, default=None,
                    help="override nominal fps for completion%%")
    ap.add_argument("--t-max-diff", type=float, default=DEFAULT_T_MAX_DIFF,
                    help="association tolerance, seconds")
    args = ap.parse_args()

    try:
        import evo  # noqa: F401
    except ImportError:
        die("evo is not installed in this python -- run: "
            f"{sys.executable} -m pip install evo")

    r = score(args.est, args.gt, args.dataset, fps=args.fps,
              t_max_diff=args.t_max_diff)
    print(human_line(r))
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        # json has no inf: serialize as the string "inf"
        ser = {k: ("inf" if isinstance(v, float) and math.isinf(v) else v)
               for k, v in r.items()}
        out.write_text(json.dumps(ser, indent=2), encoding="ascii")


if __name__ == "__main__":
    main()
