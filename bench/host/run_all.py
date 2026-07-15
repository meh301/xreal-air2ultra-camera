#!/usr/bin/env python
"""Benchmark orchestrator: prep -> (compress) -> replay -> score -> table.

Usage:
    run_all.py [--root F:\\slam_bench] [--packs DIR] [--datasets euroc,tumvi,msd]
               [--compress] [--bitrate 500k] [--only SUBSTR]
               [--replay-cmd "TEMPLATE {pack} {est}"] [--results results.md]

Layout assumed under --root:
    <root>/euroc/*.zip           (or extracted dirs containing mav0/)
    <root>/tumvi/*.tar
    <root>/msd/**/MOO*.zip       (MOO12 split parts .z01/.z02 auto-joined)

Packs are written to <packs>/<dataset>/<seq> (and <seq>_voca with --compress).
A pack is considered done when its meta.txt exists (meta is written last).

Replay: the C replay harness is NOT wired yet. Either
  * pass --replay-cmd, a shell template run per sequence with {pack} and
    {est} substituted (e.g. "replay.exe {pack} {est}"), or
  * run the harness yourself and drop est.tum into each pack dir; re-running
    run_all.py will pick existing est.tum files up and score them.
Sequences without an estimate are listed as `pending` and excluded from
the aggregate rows.

Aggregation (VOCA conventions): SR = non-diverged / attempted;
AVG over non-diverged sequences only ("exclude any-failed");
MED over all attempted sequences with failures counted as inf.
"""

import argparse
import math
import statistics
import subprocess
import sys
from pathlib import Path

from pack_common import DATASETS, die, info, read_meta, warn
import prep_dataset
import compress_voca
import score as score_mod


def discover_inputs(root, dataset):
    """Archives + already-extracted sequence dirs for one dataset."""
    base = Path(root) / dataset
    if not base.is_dir():
        warn(f"{base} does not exist -- skipping {dataset}")
        return []
    pats = {"euroc": ["*.zip"], "tumvi": ["*.tar"], "msd": ["**/*.zip"]}
    inputs = []
    for pat in pats[dataset]:
        for p in sorted(base.glob(pat)):
            if p.suffix.lower() == ".zip" and p.stem.endswith((".z01", ".z02")):
                continue  # split parts ride along with their .zip
            # skip archives whose extraction dir is already marked done;
            # discover_sequences will reuse it anyway, this just avoids
            # re-listing both the archive and its extracted dir
            inputs.append(p)
    # extracted dirs that contain sequences but no sibling archive
    for d in sorted(base.iterdir()):
        if d.is_dir() and not any(d == i.parent / i.stem for i in inputs):
            if (d / "mav0").is_dir() or any(d.rglob("mav0")):
                inputs.append(d)
    return inputs


def run_replay(pack_dir, est_path, replay_cmd):
    """Obtain est.tum for a pack. Returns True if an estimate exists.

    TODO(bench): wire the C replay harness here. Two intended modes:
      * local container:  e.g.
          docker run --rm -v <pack>:/pack -v <out>:/out slam-replay \\
              /pack /out/est.tum
      * on-device (adb):
          adb push <pack> /data/local/tmp/pack
          adb shell /data/local/tmp/replay /data/local/tmp/pack \\
              /data/local/tmp/est.tum
          adb pull /data/local/tmp/est.tum <est_path>
    Until then, --replay-cmd gives a generic escape hatch and pre-existing
    est.tum files are picked up as-is.
    """
    est_path = Path(est_path)
    if est_path.exists() and est_path.stat().st_size > 0:
        info(f"    est.tum already present: {est_path}")
        return True
    if replay_cmd:
        cmd = replay_cmd.format(pack=str(pack_dir), est=str(est_path))
        info(f"    replay: {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            warn(f"replay command failed (exit {r.returncode}): "
                 f"{(r.stderr or r.stdout or '').strip()[-500:]}")
            return False
        return est_path.exists() and est_path.stat().st_size > 0
    info("    no replay harness wired (see run_replay TODO) -- pending")
    return False


def fmt(v, prec=3):
    if v is None:
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.{prec}f}"


def aggregate(rows):
    """rows: list of dicts with keys seq/dataset/status/ate/rte/compl."""
    attempted = [r for r in rows if r["status"] in ("ok", "diverged")]
    ok = [r for r in attempted if r["status"] == "ok"]
    agg = {"attempted": len(attempted), "ok": len(ok),
           "pending": len(rows) - len(attempted)}
    agg["sr_pct"] = 100.0 * len(ok) / len(attempted) if attempted else None

    def col(rs, k):
        return [r[k] for r in rs if r[k] is not None]

    for k in ("ate", "rte", "compl"):
        vals_ok = col(ok, k)
        agg[f"avg_{k}"] = sum(vals_ok) / len(vals_ok) if vals_ok else None
        vals_all = [(math.inf if r["status"] == "diverged" and k != "compl"
                     else r[k]) for r in attempted if r[k] is not None
                    or r["status"] == "diverged"]
        vals_all = [v for v in vals_all if v is not None]
        agg[f"med_{k}"] = statistics.median(vals_all) if vals_all else None
    return agg


def make_markdown(rows, agg, compressed):
    lines = ["# SLAM benchmark results", "",
             f"Pack variant: {'VOCA-compressed' if compressed else 'raw'}", "",
             "| dataset | sequence | ATE (m) | RTE (cm) | compl (%) | status |",
             "|---|---|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['seq']} | {fmt(r['ate'], 4)} | "
                     f"{fmt(r['rte'], 3)} | {fmt(r['compl'], 1)} | "
                     f"{r['status']} |")
    lines += ["",
              f"- **SR**: {agg['ok']}/{agg['attempted']} "
              f"({fmt(agg['sr_pct'], 1)}%)"
              + (f", {agg['pending']} pending (no estimate)"
                 if agg["pending"] else ""),
              f"- **AVG** (non-failed only): ATE {fmt(agg['avg_ate'], 4)} m, "
              f"RTE {fmt(agg['avg_rte'], 3)} cm, "
              f"compl {fmt(agg['avg_compl'], 1)}%",
              f"- **MED** (all attempted, failures = inf): "
              f"ATE {fmt(agg['med_ate'], 4)} m, RTE {fmt(agg['med_rte'], 3)} cm, "
              f"compl {fmt(agg['med_compl'], 1)}%",
              "",
              "Conventions: ATE = SE(3) Umeyama-aligned (no scale) RMSE; "
              "RTE = RPE delta=6 frames, translation RMSE; divergence "
              "thresholds ATE>10 m (1 m msd) / RTE>10 cm; AVG excludes "
              "failed sequences, MED counts failures as inf (VOCA).", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=r"F:\slam_bench")
    ap.add_argument("--packs", default=None,
                    help=r"pack output dir (default <root>\packs)")
    ap.add_argument("--datasets", default="euroc,tumvi,msd")
    ap.add_argument("--compress", action="store_true",
                    help="also build + evaluate the VOCA-compressed variant")
    ap.add_argument("--bitrate", default="500k")
    ap.add_argument("--only", default=None,
                    help="substring filter on sequence names")
    ap.add_argument("--replay-cmd", default=None,
                    help="shell template with {pack} and {est} placeholders")
    ap.add_argument("--results", default=None,
                    help=r"results markdown path (default <packs>\results.md)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        die(f"--root {root} does not exist")
    packs_root = Path(args.packs) if args.packs else root / "packs"
    results_md = Path(args.results) if args.results else packs_root / "results.md"
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            die(f"unknown dataset {d!r} in --datasets")

    rows = []
    seen = set()
    for dataset in datasets:
        for inp in discover_inputs(root, dataset):
            try:
                seqs = prep_dataset.discover_sequences(inp)
            except SystemExit:
                warn(f"skipping unusable input {inp}")
                continue
            for seq, seq_root in seqs:
                if args.only and args.only.lower() not in seq.lower():
                    continue
                if (dataset, seq) in seen:   # same seq reachable via both
                    continue                 # its archive and extracted dir
                seen.add((dataset, seq))
                info(f"== {dataset}/{seq} ==")
                pack = packs_root / dataset / seq
                if (pack / "meta.txt").exists():
                    info(f"    pack exists: {pack}")
                else:
                    input_root = inp if inp.is_dir() else inp.parent
                    prep_dataset.prep_one(dataset, seq_root, pack,
                                          input_root=input_root)

                eval_pack = pack
                if args.compress:
                    cpack = packs_root / dataset / f"{seq}_voca"
                    if (cpack / "meta.txt").exists():
                        info(f"    compressed pack exists: {cpack}")
                    else:
                        compress_voca.compress_pack(pack, cpack,
                                                    bitrate=args.bitrate)
                    eval_pack = cpack

                est = eval_pack / "est.tum"
                row = {"dataset": dataset, "seq": seq,
                       "ate": None, "rte": None, "compl": None}
                if run_replay(eval_pack, est, args.replay_cmd):
                    r = score_mod.score(est, eval_pack / "gt.tum", dataset)
                    print("    " + score_mod.human_line(r))
                    row["ate"] = r["ate_m"]
                    row["rte"] = r["rte_cm"]
                    row["compl"] = r["completion_pct"]
                    row["status"] = "diverged" if r["diverged"] else "ok"
                else:
                    row["status"] = "pending"
                rows.append(row)

    if not rows:
        die("no sequences found -- check --root layout / --only filter "
            "(and note: the current EuRoC zips on F: are HTML error pages)")

    agg = aggregate(rows)
    md = make_markdown(rows, agg, args.compress)
    results_md.parent.mkdir(parents=True, exist_ok=True)
    results_md.write_text(md, encoding="utf-8", newline="\n")
    print()
    print(md)
    info(f"results written to {results_md}")


if __name__ == "__main__":
    main()
