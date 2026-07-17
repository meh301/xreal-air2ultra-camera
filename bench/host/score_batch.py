#!/usr/bin/env python3
"""Score a directory of batch trajectories (<seq>_<arm>_r<N>_{vio,map}.tum)
against local pack gt.tum files; print per-sequence medians per arm/track."""
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

RES = Path(sys.argv[1] if len(sys.argv) > 1 else r"F:\slam_bench\results\batch")
PACKS = [Path(r"F:\slam_bench\packs\tumvi"), Path(r"F:\slam_bench\packs\msd"),
         Path(r"F:\slam_bench\packs\euroc")]
PY = sys.executable
SCORE = Path(__file__).parent / "score.py"

def gt_for(seq):
    for root in PACKS:
        p = root / seq / "gt.tum"
        if p.exists():
            return p, ("msd" if "MOO" in seq or "MI" in seq or "MG" in seq
                       else "tumvi" if "room" in seq or "corridor" in seq or
                            "magistrale" in seq or "slides" in seq else "euroc")
    return None, None

rows = defaultdict(lambda: defaultdict(list))   # rows[seq][(arm,track)] = [(ate,rte)]
for f in sorted(RES.glob("*.tum")):
    m = re.match(r"(.+)_(bad|xfeat|vpr|xvpr)_r(\d+)_(vio|map)\.tum$", f.name)
    if not m:
        continue
    seq, arm, run, track = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    gt, ds = gt_for(seq)
    if not gt:
        print(f"!! no gt for {seq}")
        continue
    out = subprocess.run([PY, str(SCORE), str(f), str(gt), "--dataset", ds],
                         capture_output=True, text=True)
    mm = re.search(r"ATE ([\d.]+|inf) m \| RTE ([\d.]+|inf) cm", out.stdout)
    if not mm:
        print(f"!! score failed {f.name}: {out.stdout.strip()[:90]}")
        continue
    ate = float(mm.group(1)) * 100 if mm.group(1) != "inf" else float("inf")
    rte = float(mm.group(2)) if mm.group(2) != "inf" else float("inf")
    rows[seq][(arm, track)].append((ate, rte))

print(f"\n{'sequence':34s} {'arm':6s} {'track':5s} {'runs':>4s} "
      f"{'ATE med cm':>10s} {'RTE med cm':>10s}")
for seq in sorted(rows):
    for (arm, track), vals in sorted(rows[seq].items()):
        ates = [a for a, _ in vals]
        rtes = [r for _, r in vals]
        print(f"{seq:34s} {arm:6s} {track:5s} {len(vals):4d} "
              f"{median(ates):10.2f} {median(rtes):10.3f}")
