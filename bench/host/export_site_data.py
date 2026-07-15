#!/usr/bin/env python3
"""Export benchmark data for bench/site: run-level scores, baselines,
closure-ledger stats, and downsampled trajectories.

  python export_site_data.py --results <tum_dir> [<tum_dir> ...]
      [--baselines <dir>] [--logs <dir> ...] [--traj] --out <site>/public/data
"""
import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

PACKS = [Path(r"F:\slam_bench\packs\tumvi"), Path(r"F:\slam_bench\packs\msd"),
         Path(r"F:\slam_bench\packs\euroc")]
PY = sys.executable
SCORE = Path(__file__).parent / "score.py"

def group_of(seq):
    if seq.startswith(("MH_", "V1_", "V2_")): return "euroc"
    if seq.startswith(("MOO", "MIO", "MIP", "MGO")): return "msd"
    if "room" in seq: return "rooms"
    return "long"

def gt_for(seq):
    for root in PACKS:
        p = root / seq / "gt.tum"
        if p.exists():
            return p, ("msd" if group_of(seq) == "msd"
                       else "euroc" if group_of(seq) == "euroc" else "tumvi")
    return None, None

def score_one(args):
    fname, gt, ds = args
    out = subprocess.run([PY, str(SCORE), fname, gt, "--dataset", ds],
                         capture_output=True, text=True)
    m = re.search(r"ATE ([\d.]+|inf) m \| RTE ([\d.]+|inf) cm \| "
                  r"completion ([\d.]+)%", out.stdout)
    if not m:
        return fname, None, None, None
    ate = float(m.group(1)) * 100 if m.group(1) != "inf" else None
    rte = float(m.group(2)) if m.group(2) != "inf" else None
    return fname, ate, rte, float(m.group(3))

def collect_runs(dirs, name_re, cache=None):
    jobs, meta, rows = [], {}, []
    for d in dirs:
        for f in sorted(Path(d).glob("*.tum")):
            m = name_re.match(f.name)
            if not m:
                continue
            seq = m.group("seq")
            gt, ds = gt_for(seq)
            if not gt:
                continue
            key = f"{f.name}|{f.stat().st_size}"
            if cache is not None and key in cache:
                rows.append(cache[key])
                continue
            meta[str(f)] = (m.groupdict(), key)
            jobs.append((str(f), str(gt), ds))
    with ProcessPoolExecutor(max_workers=12) as ex:
        for fname, ate, rte, comp in ex.map(score_one, jobs):
            g, key = meta[fname]
            row = {
                "seq": g["seq"], "group": group_of(g["seq"]),
                "arm": g.get("arm") or g.get("sys"),
                "run": int(g["run"]) if g.get("run") else 1,
                "track": g.get("track") or ("lc" + g.get("lc", "0")),
                "ate_cm": round(ate, 3) if ate is not None else None,
                "rte_cm": round(rte, 3) if rte is not None else None,
                "completion": comp,
            }
            rows.append(row)
            if cache is not None:
                cache[key] = row
    return rows

def parse_ledgers(log_dirs):
    out = defaultdict(lambda: {"searches": 0, "cand": 0, "top": []})
    pat = re.compile(r"LEDGER q=\d+ vprtop=([\d.]+) searched=(\d+) cand=(\d+)")
    for d in log_dirs:
        for f in Path(d).glob("*.log"):
            m = re.match(r"(.+)_(bad|vpr|megaloc|xfeat|xvpr|xmegaloc)_r(\d+)", f.stem)
            if not m:
                continue
            key = f"{m.group(1)}|{m.group(2)}"
            try:
                txt = f.read_text(errors="replace")
            except OSError:
                continue
            for lm in pat.finditer(txt):
                out[key]["searches"] += 1
                out[key]["cand"] += int(lm.group(3))
                out[key]["top"].append(float(lm.group(1)))
    return [{"seq": k.split("|")[0], "arm": k.split("|")[1],
             "searches": v["searches"], "candidates": v["cand"],
             "vprtop_mean": round(sum(v["top"]) / len(v["top"]), 3) if v["top"] else None}
            for k, v in out.items()]

def downsample_traj(results_dirs, out_dir, max_pts=1500):
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    seen = set()
    for d in results_dirs:
        for f in sorted(Path(d).glob("*_r1_map.tum")):
            m = re.match(r"(.+)_(bad|vpr|megaloc|xfeat|xvpr|xmegaloc)_r1_map", f.stem)
            if not m:
                continue
            seq, arm = m.group(1), m.group(2)
            if (seq, arm) in seen:
                continue
            seen.add((seq, arm))
            gt, _ = gt_for(seq)
            if not gt:
                continue
            try:
                est = np.loadtxt(f)
                gta = np.loadtxt(gt)
            except Exception:
                continue
            if est.ndim != 2 or len(est) < 10:
                continue
            # associate + SE3-Umeyama align est to GT for honest 3D overlay
            gt_t, gt_p = gta[:, 0], gta[:, 1:4]
            t, p = est[:, 0], est[:, 1:4]
            i = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1)
            ok = np.abs(gt_t[i] - t) < 0.02
            A, B = p[ok], gt_p[i][ok]
            if len(A) < 10:
                continue
            ca, cb = A.mean(0), B.mean(0)
            Hm = (A - ca).T @ (B - cb)
            U, S, Vt = np.linalg.svd(Hm)
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            Aal = (R @ (A - ca).T).T + cb
            err = np.linalg.norm(Aal - B, axis=1) * 100.0
            ts = t[ok] - t[ok][0]
            step = max(1, len(Aal) // max_pts)
            gstep = max(1, len(gt_p) // max_pts)
            r3 = lambda a: [[round(float(v), 3) for v in row] for row in a]
            (out_dir / f"{seq}_{arm}.json").write_text(json.dumps({
                "est": r3(Aal[::step]), "gt": r3(gt_p[::gstep]),
                "err": [round(float(e), 1) for e in err[::step]],
                "t": [round(float(x), 2) for x in ts[::step]],
            }))
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=[])
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--traj", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ours_re = re.compile(
        r"(?P<seq>.+)_(?P<arm>bad|vpr|megaloc|xfeat|xvpr|xmegaloc)"
        r"_r(?P<run>\d+)_(?P<track>vio|map)\.tum$")
    base_re = re.compile(
        r"(?P<seq>.+)_(?P<sys>okvis2|orb3|openvins)_lc(?P<lc>[01])\.tum$")

    cache_f = out / ".score_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}

    if a.results:
        rows = collect_runs(a.results, ours_re, cache)
        (out / "results.json").write_text(json.dumps(
            {"generated": datetime.now(timezone.utc).isoformat(), "runs": rows}))
        print(f"results.json: {len(rows)} rows")
    if a.baselines:
        rows = collect_runs(a.baselines, base_re, cache)
        (out / "baselines.json").write_text(json.dumps({"runs": rows}))
        print(f"baselines.json: {len(rows)} rows")
    cache_f.write_text(json.dumps(cache))
    if a.logs:
        led = parse_ledgers(a.logs)
        (out / "ledger.json").write_text(json.dumps(led))
        print(f"ledger.json: {len(led)} entries")
    if a.traj and a.results:
        n = downsample_traj(a.results, out / "traj")
        print(f"traj: {n} files")
    meta = {"generated": datetime.now(timezone.utc).isoformat(),
            "protocol": "causal (pose at first estimate), SE3-Umeyama ATE, "
                        "RTE delta=6 frames, medians over runs"}
    (out / "meta.json").write_text(json.dumps(meta))
    print("meta.json written")

if __name__ == "__main__":
    main()
