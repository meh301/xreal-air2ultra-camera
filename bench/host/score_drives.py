#!/usr/bin/env python3
"""Drive-scale ATE: SE(3)-Umeyama RMSE (no scale), reported in metres and
as % of GT path length — the indoor scorer's divergence caps don't apply
at kilometre scale. Associates on timestamps (0.03 s tolerance)."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(r"F:\slam_bench\results\drivepull")
EIG = Path(r"F:\slam_bench\results\lmfs_pull\drive_reloc_pad_eig")

def load(p):
    a = np.loadtxt(p)
    return a if a.ndim == 2 else a.reshape(1, -1)

def ate(est_p, gt_p):
    ca, cb = est_p.mean(0), gt_p.mean(0)
    H = (est_p - ca).T @ (gt_p - cb)
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    al = (R @ (est_p - ca).T).T + cb
    return float(np.sqrt(np.mean(np.sum((al - gt_p) ** 2, axis=1))))

print(f"{'run':42s} {'assoc':>6s} {'ATE m':>8s} {'%path':>6s}")
for seq in ["drive1_city", "drive2_city", "drive3_country"]:
    gt = load(ROOT / seq / "gt.tum")
    gt_t, gt_p = gt[:, 0], gt[:, 1:4]
    d = np.linalg.norm(np.diff(gt_p, axis=0), axis=1)
    path = float(d.sum())
    cands = []
    for arm in ["padeig", "padmeg"]:
        base = (EIG if arm == "padeig" else ROOT / "drive_reloc_pad_meg")
        for tr in ["map", "vio"]:
            p = base / f"{seq}__{arm}_{tr}.tum"
            if p.exists():
                cands.append((f"{seq} {arm} {tr}", p))
    for lc in [0, 1]:
        p = ROOT / f"{seq}_okvis_lc{lc}.final.tum"
        if p.exists():
            cands.append((f"{seq} okvis2 lc{lc}", p))
    for name, p in cands:
        try:
            est = load(p)
        except Exception:
            print(f"{name:42s}  load-fail")
            continue
        if est.shape[0] < 100:
            print(f"{name:42s}  short ({est.shape[0]})")
            continue
        t, ep = est[:, 0], est[:, 1:4]
        i = np.clip(np.searchsorted(gt_t, t), 0, len(gt_t) - 1)
        ok = np.abs(gt_t[i] - t) < 0.03
        if ok.sum() < 100:
            print(f"{name:42s}  no-assoc")
            continue
        a = ate(ep[ok], gt_p[i][ok])
        print(f"{name:42s} {int(ok.sum()):6d} {a:8.2f} {100*a/path:6.2f}")
    print(f"  ({seq}: GT path {path:.0f} m)")
