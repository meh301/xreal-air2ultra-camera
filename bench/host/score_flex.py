#!/usr/bin/env python3
"""Score a results directory with arbitrary arm names (score_parallel's
regex hardcoded the matrix-1 arms). Handles both single- and double-
underscore <seq>[_]_<arm>_r<N>_<track>.tum naming, prints per-seq medians
per arm/track plus the EVALUATION.md arm-level aggregate (median over
sequences), and dumps raw per-run scores as JSON next to the results."""
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
import subprocess

RES = Path(sys.argv[1])
ARMS = (sys.argv[2].split(",") if len(sys.argv) > 2 else
        ["bad", "xdlg6", "fullm", "full", "lfbase", "lfonly", "lfts",
         "sn30", "sn50"])
PACKS = [Path(r"F:\slam_bench\packs\tumvi"), Path(r"F:\slam_bench\packs\msd"),
         Path(r"F:\slam_bench\packs\euroc")]
PY = sys.executable
SCORE = Path(__file__).parent / "score.py"
# longest-first so 'fullm' wins over 'full'
PAT = re.compile(r"(.+?)_{1,2}(" +
                 "|".join(sorted(ARMS, key=len, reverse=True)) +
                 r")_r(\d+)_(vio|map)\.tum$")


def gt_for(seq):
    for root in PACKS:
        p = root / seq / "gt.tum"
        if p.exists():
            return p, ("msd" if seq.startswith(("MOO", "MI", "MG"))
                       else "euroc" if seq.startswith(("MH_", "V1_", "V2_"))
                       else "tumvi")
    return None, None


def score_one(args):
    fname, gt, ds = args
    out = subprocess.run([PY, str(SCORE), fname, gt, "--dataset", ds],
                         capture_output=True, text=True)
    m = re.search(r"ATE ([\d.]+|inf) m \| RTE ([\d.]+|inf) cm", out.stdout)
    if not m:
        return fname, None, None
    ate = float(m.group(1)) * 100 if m.group(1) != "inf" else float("inf")
    rte = float(m.group(2)) if m.group(2) != "inf" else float("inf")
    return fname, ate, rte


if __name__ == "__main__":
    jobs, meta = [], {}
    for f in sorted(RES.glob("*.tum")):
        m = PAT.match(f.name)
        if not m:
            continue
        seq, arm, track = m.group(1).rstrip("_"), m.group(2), m.group(4)
        gt, ds = gt_for(seq)
        if not gt:
            print(f"!! no gt for {seq}", file=sys.stderr)
            continue
        meta[str(f)] = (seq, arm, track)
        jobs.append((str(f), str(gt), ds))
    print(f"scoring {len(jobs)} trajectories, 12-wide", file=sys.stderr)
    rows = defaultdict(lambda: defaultdict(list))
    raw = []
    with ProcessPoolExecutor(max_workers=12) as ex:
        for fname, ate, rte in ex.map(score_one, jobs):
            if ate is None:
                print(f"!! {Path(fname).name} failed", file=sys.stderr)
                continue
            seq, arm, track = meta[fname]
            rows[seq][(arm, track)].append((ate, rte))
            raw.append({"seq": seq, "arm": arm, "track": track,
                        "ate_cm": ate, "rte_cm": rte})
    print(f"{'sequence':34s} {'arm':7s} {'track':5s} {'runs':>4s} "
          f"{'ATE med cm':>10s} {'RTE med cm':>10s}")
    for seq in sorted(rows):
        for (arm, track), vals in sorted(rows[seq].items()):
            print(f"{seq:34s} {arm:7s} {track:5s} {len(vals):4d} "
                  f"{median(a for a, _ in vals):10.2f} "
                  f"{median(r for _, r in vals):10.3f}")
    # arm-level: median over per-seq medians (EVALUATION.md error rule)
    print("\narm aggregate (median over per-seq medians, map track):")
    agg = defaultdict(list)
    for seq in rows:
        for (arm, track), vals in rows[seq].items():
            if track == "map":
                agg[arm].append(median(a for a, _ in vals))
    for arm in sorted(agg):
        n_div = sum(1 for a in agg[arm] if a == float("inf"))
        print(f"  {arm:7s} n_seq={len(agg[arm]):3d} "
              f"med={median(agg[arm]):8.2f} cm diverged_seqs={n_div}")
    (RES / "scores.json").write_text(json.dumps(raw, indent=1))
