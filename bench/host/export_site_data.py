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

"""In-process scorer replicating score.py's protocol exactly (validated
against evo on the full matrix; see fast-scorer commit). The old path spawned
a fresh python+evo subprocess PER trajectory (~12 s of import overhead for
~10 ms of math) — scoring took longer than the benchmarks themselves."""
DATASET_FPS = {"euroc": 20.0, "tumvi": 20.0, "msd": 30.0}
ATE_DIVERGE_M = {"euroc": 10.0, "tumvi": 10.0, "msd": 1.0}
RTE_DIVERGE_CM = 10.0
RTE_DELTA = 6
T_MAX_DIFF = 0.01

_GT_CACHE = {}

def _load_gt(path):
    import numpy as np
    if path not in _GT_CACHE:
        a = np.loadtxt(path)
        _GT_CACHE[path] = a if a.ndim == 2 else a.reshape(1, -1)
    return _GT_CACHE[path]

def _quat_to_R(q):
    """(N,4) qx qy qz qw -> (N,3,3), normalized."""
    import numpy as np
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def score_one(args):
    import numpy as np
    fname, gt_path, ds = args
    fps = DATASET_FPS[ds]
    try:
        gta = _load_gt(gt_path)
        est = np.loadtxt(fname)
    except Exception:
        return fname, None, None, None
    if est.ndim != 2 or len(est) < RTE_DELTA + 2:
        return fname, None, None, 0.0
    gt_t = gta[:, 0]
    gt_span = gt_t[-1] - gt_t[0]
    expected = max(1, int(round(gt_span * fps)) + 1)
    t = est[:, 0]
    slack = 0.5 / fps
    completion = min(100.0, 100.0 * ((t >= gt_t[0] - slack) & (t <= gt_t[-1] + slack)).sum() / expected)
    # associate: nearest GT stamp within tolerance, unique GT indices
    idx = np.clip(np.searchsorted(gt_t, t), 1, len(gt_t) - 1)
    idx = np.where(np.abs(gt_t[idx - 1] - t) < np.abs(gt_t[idx] - t), idx - 1, idx)
    ok = np.abs(gt_t[idx] - t) <= T_MAX_DIFF
    # dedup: keep closest est per GT index
    order = np.argsort(np.abs(gt_t[idx] - t), kind="stable")
    keep = np.zeros(len(t), bool); used = set()
    for k in order:
        if ok[k] and idx[k] not in used:
            used.add(idx[k]); keep[k] = True
    sel = np.where(keep)[0]
    sel = sel[np.argsort(t[sel])]
    if len(sel) < RTE_DELTA + 2:
        return fname, None, None, round(float(completion), 1)
    E_t, G_t = est[sel, 1:4], gta[idx[sel], 1:4]
    E_q, G_q = est[sel, 4:8], gta[idx[sel], 4:8]
    # ATE: Umeyama SE(3), no scale
    ca, cb = E_t.mean(0), G_t.mean(0)
    H = (E_t - ca).T @ (G_t - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    ate_m = float(np.sqrt((np.linalg.norm((R @ (E_t - ca).T).T + cb - G_t, axis=1) ** 2).mean()))
    # RTE: delta=6 frames, consecutive non-overlapping pairs, trans part of
    # relative error E = inv(rel_gt) @ rel_est. Pairs whose matched stamps
    # span a GT hole (e.g. TUM-VI corridor's 247 s no-mocap stretch) measure
    # relative motion over unobserved time — drop any pair wider than 4x the
    # nominal delta duration.
    ii = np.arange(0, len(sel) - RTE_DELTA, RTE_DELTA)
    jj = ii + RTE_DELTA
    ts = t[sel]
    span_ok = (ts[jj] - ts[ii]) <= (RTE_DELTA / fps) * 4.0
    ii, jj = ii[span_ok], jj[span_ok]
    if not len(ii):
        return fname, None, None, round(float(completion), 1)
    Re, Rg = _quat_to_R(E_q), _quat_to_R(G_q)
    def rel_trans(Rm, tm):
        Ri = np.transpose(Rm[ii], (0, 2, 1))
        rR = Ri @ Rm[jj]
        rt = np.einsum("nij,nj->ni", Ri, tm[jj] - tm[ii])
        return rR, rt
    eR, et2 = rel_trans(Re, E_t)
    gR, gt2 = rel_trans(Rg, G_t)
    gRi = np.transpose(gR, (0, 2, 1))
    err_t = np.einsum("nij,nj->ni", gRi, et2 - gt2)
    rte_cm = float(np.sqrt((np.linalg.norm(err_t, axis=1) ** 2).mean())) * 100.0
    # Divergence gates on ATE ONLY. An RTE gate structurally kills causal
    # loop-closure systems: a correction re-anchor is one giant frame-pair
    # step (ORB-SLAM3+LC lost 10/11 EuRoC runs to it while whole-run ATE was
    # 0.4-8.7 m). RTE is still computed and reported — snaps show up there.
    if ate_m > ATE_DIVERGE_M[ds]:
        return fname, None, None, round(float(completion), 1)
    return fname, ate_m * 100.0, rte_cm, round(float(completion), 1)

def collect_runs(dirs, name_re, cache=None, progress=None):
    """progress = (out_path, cache_path) -> write results.json + cache
    incrementally every ~40 newly-scored rows so the live site fills in as
    scoring proceeds instead of only at the end."""
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
    n = 0
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
            n += 1
            if progress and n % 40 == 0:
                out_path, cache_path, gen = progress
                out_path.write_text(json.dumps({"generated": gen, "runs": rows}))
                cache_path.write_text(json.dumps(cache))
                print(f"  ...{n}/{len(jobs)} scored", flush=True)
    return rows

def parse_reloc(reloc_dirs, out):
    """Parse reloc-benchmark logs (<seq>__<arm>.log with RELOC k= lines and
    RELOC-SUMMARY) into reloc.json for the site's Reloc tab."""
    pk = re.compile(r"RELOC k=(\d+) frame=(\d+) ok=(\d) inl=(\d+) err_m=([-\d.]+)"
                    r"(?: exp=([-\d.]+),([-\d.]+),([-\d.]+) got=([-\d.]+),([-\d.]+),([-\d.]+))?")
    ps = re.compile(r"RELOC-SUMMARY n=(\d+) verified=(\d+) recall=([\d.]+) "
                    r"r@25cm=([\d.]+) r@10cm=([\d.]+) med_err_m=([-\d.]+)")
    entries = []
    for d in reloc_dirs:
        for f in sorted(Path(d).glob("*__*.log")):
            seq, arm = f.stem.split("__", 1)
            txt = f.read_text(errors="replace")
            probes = []
            for m in pk.finditer(txt):
                p = {"ok": int(m.group(3)), "inl": int(m.group(4)),
                     "err": float(m.group(5))}
                if m.group(6) is not None:
                    p["exp"] = [float(m.group(i)) for i in (6, 7, 8)]
                    p["got"] = [float(m.group(i)) for i in (9, 10, 11)]
                probes.append(p)
            s = ps.search(txt)
            if not s:
                continue
            entry = {"seq": seq, "arm": arm,
                     "n": int(s.group(1)), "verified": int(s.group(2)),
                     "recall": float(s.group(3)), "r25": float(s.group(4)),
                     "r10": float(s.group(5)), "med": float(s.group(6)),
                     "probes": probes}
            # SESSION-frame trajectory (same frame as exp/got — the aligned
            # traj jsons are in the GT frame and MUST NOT be mixed in)
            mt = f.with_name(f.stem + "_map.tum")
            if mt.exists():
                import numpy as np
                a = np.loadtxt(mt)
                if a.ndim == 2 and len(a) > 2:
                    step = max(1, len(a) // 800)
                    entry["traj"] = [[round(float(v), 3) for v in row]
                                     for row in a[::step, 1:4]]
            entries.append(entry)
    # dedupe by (seq, arm): LAST wins, so later --reloc dirs override
    # earlier ones (e.g. clean reruns replacing degraded-map runs)
    dedup = {}
    for e in entries:
        dedup[(e["seq"], e["arm"])] = e
    entries = list(dedup.values())
    (out / "reloc.json").write_text(json.dumps({"entries": entries}))
    print(f"reloc.json: {len(entries)} seq-arm entries")
    return len(entries)

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

def downsample_traj(results_dirs, out_dir, max_pts=1500, baseline_dirs=()):
    """Emit <seq>_<key>.json for every plot the viewer offers:
    our arms r1 map track (key=<arm>), r1 vio track (key=<arm>-vio, bad only
    would be redundant per arm — VIO differs only by descriptor so bad/xfeat),
    and baseline systems (key=okvis2lc0 etc.)."""
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    seen = set()
    jobs = []
    for d in results_dirs:
        for f in sorted(Path(d).glob("*_r1_map.tum")):
            m = re.match(r"(.+)_(bad|vpr|megaloc|xfeat|xvpr|xmegaloc)_r1_map", f.stem)
            if m:
                jobs.append((f, m.group(1), m.group(2)))
        # VIO track: one per descriptor family (bad, xfeat) — vpr/megaloc VIO
        # is identical to bad's (map layer has no feedback into VIO)
        for f in sorted(Path(d).glob("*_r1_vio.tum")):
            m = re.match(r"(.+)_(bad|xfeat)_r1_vio", f.stem)
            if m:
                jobs.append((f, m.group(1), m.group(2) + "-vio"))
    for d in baseline_dirs:
        for f in sorted(Path(d).glob("*.tum")):
            m = re.match(r"(.+)_(okvis2|orb3|openvins)_lc([01])$", f.stem)
            if m:
                jobs.append((f, m.group(1), f"{m.group(2)}lc{m.group(3)}"))
    for f, seq, arm in jobs:
        if True:
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
            # Associate est frames to GT within tolerance. Datasets like
            # TUM-VI corridor/magistrale have mocap GT only at the start/end
            # (big time gaps in between) — align on matched frames only, but
            # export the FULL continuous estimate so the 3D path is complete;
            # error is defined only where GT exists (null elsewhere), and GT
            # is broken across its own gaps so nothing is drawn through them.
            gt_t, gt_p = gta[:, 0], gta[:, 1:4]
            t, p = est[:, 0], est[:, 1:4]
            i = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1)
            ok = np.abs(gt_t[i] - t) < 0.03
            if ok.sum() < 10:
                continue
            A, B = p[ok], gt_p[i][ok]
            ca, cb = A.mean(0), B.mean(0)
            Hm = (A - ca).T @ (B - cb)
            U, Sv, Vt = np.linalg.svd(Hm)
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
            full = (R @ (p - ca).T).T + cb          # full aligned estimate
            errf = np.full(len(p), np.nan)
            errf[ok] = np.linalg.norm(full[ok] - B, axis=1) * 100.0
            ts = t - t[0]
            step = max(1, len(p) // max_pts)
            r3 = lambda a: [[round(float(v), 3) for v in row] for row in a]
            # GT downsampled, with null breaks across time gaps > 2 s
            gstep = max(1, len(gt_p) // max_pts)
            gtd_t, gtd_p = gt_t[::gstep], gt_p[::gstep]
            gt_out = []
            for k in range(len(gtd_p)):
                if k and gtd_t[k] - gtd_t[k - 1] > 2.0:
                    gt_out.append(None)
                gt_out.append([round(float(v), 3) for v in gtd_p[k]])
            ed = errf[::step]
            (out_dir / f"{seq}_{arm}.json").write_text(json.dumps({
                "est": r3(full[::step]), "gt": gt_out,
                "err": [None if np.isnan(e) else round(float(e), 1) for e in ed],
                "t": [round(float(x), 2) for x in ts[::step]],
            }))
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=[])
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--logs", nargs="*", default=[])
    ap.add_argument("--reloc", nargs="*", default=[],
                    help="dirs of reloc logs (<seq>__<arm>.log)")
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

    gen = datetime.now(timezone.utc).isoformat()
    if a.results:
        rows = collect_runs(a.results, ours_re, cache,
                            progress=(out / "results.json", cache_f, gen))
        (out / "results.json").write_text(json.dumps({"generated": gen, "runs": rows}))
        print(f"results.json: {len(rows)} rows")
    if a.baselines:
        rows = collect_runs(a.baselines, base_re, cache)
        (out / "baselines.json").write_text(json.dumps({"runs": rows}))
        print(f"baselines.json: {len(rows)} rows")
    cache_f.write_text(json.dumps(cache))
    if a.reloc:
        parse_reloc(a.reloc, out)
    if a.logs:
        led = parse_ledgers(a.logs)
        (out / "ledger.json").write_text(json.dumps(led))
        print(f"ledger.json: {len(led)} entries")
    if a.traj and a.results:
        n = downsample_traj(a.results, out / "traj", baseline_dirs=a.baselines)
        print(f"traj: {n} files")
    meta = {"generated": datetime.now(timezone.utc).isoformat(),
            "protocol": "causal (pose at first estimate), SE3-Umeyama ATE, "
                        "RTE delta=6 frames, medians over runs"}
    (out / "meta.json").write_text(json.dumps(meta))
    print("meta.json written")

if __name__ == "__main__":
    main()
