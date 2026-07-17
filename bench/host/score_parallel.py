#!/usr/bin/env python3
"""Parallel wrapper around score_batch's per-file scoring: a process pool of
score.py invocations (12-wide), then the same median aggregation. Output is
identical to score_batch.py; wall time ~12x better on many-run matrices."""
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median

RES = Path(sys.argv[1] if len(sys.argv) > 1 else r"F:\slam_bench\results\matrix1")
PACKS = [Path(r"F:\slam_bench\packs\tumvi"), Path(r"F:\slam_bench\packs\msd"),
         Path(r"F:\slam_bench\packs\euroc")]
PY = sys.executable
SCORE = Path(__file__).parent / "score.py"

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
    jobs = []
    meta = {}
    for f in sorted(RES.glob("*.tum")):
        m = re.match(r"(.+)_(bad|xfeat|vpr|xvpr)_r(\d+)_(vio|map)\.tum$", f.name)
        if not m:
            continue
        seq, arm, track = m.group(1), m.group(2), m.group(4)
        gt, ds = gt_for(seq)
        if not gt:
            continue
        meta[str(f)] = (seq, arm, track)
        jobs.append((str(f), str(gt), ds))
    print(f"scoring {len(jobs)} trajectories, 12-wide", file=sys.stderr)
    rows = defaultdict(lambda: defaultdict(list))
    with ProcessPoolExecutor(max_workers=12) as ex:
        for fname, ate, rte in ex.map(score_one, jobs):
            if ate is None:
                print(f"!! {Path(fname).name} failed", file=sys.stderr)
                continue
            seq, arm, track = meta[fname]
            rows[seq][(arm, track)].append((ate, rte))
    print(f"{'sequence':34s} {'arm':6s} {'track':5s} {'runs':>4s} "
          f"{'ATE med cm':>10s} {'RTE med cm':>10s}")
    for seq in sorted(rows):
        for (arm, track), vals in sorted(rows[seq].items()):
            print(f"{seq:34s} {arm:6s} {track:5s} {len(vals):4d} "
                  f"{median(a for a, _ in vals):10.2f} "
                  f"{median(r for _, r in vals):10.3f}")
