# Benchmark results site (self-hosted)

Zero-dependency Node server + static single-page explorer for the SLAM
benchmark program (bench/PROGRAM.md).

```
node server.js [port]        # default 8080, serves ./public
```

## Refreshing data

All content comes from `public/data/*.json`, generated on the bench host:

```
python bench/host/export_site_data.py \
    --results F:\slam_bench\results\matrix1 [more dirs ...] \
    --baselines F:\slam_bench\results\baselines \
    --logs F:\slam_bench\results\matrix1_logs \
    --out bench/site/public/data
```

- `results.json` — run-level rows {seq, group, arm, run, track, ate_cm, rte_cm}
- `baselines.json` — same shape for OKVIS2 / ORB-SLAM3 / OpenVINS runs
- `published.json` — literature reference numbers with regime annotations
  (curated by hand/agents; every entry carries its source URL)
- `ledger.json` — per-search closure ledger stats parsed from run logs
- `traj/<seq>_<arm>.json` — downsampled top-view trajectories + GT
- `meta.json` — generation stamp + protocol notes

## Features

Tabs per dataset group, arm filter chips, published-reference overlays,
click-a-bar drill-down (run scatter + zoomable trajectory viewer),
systems-vs-ours comparison, full sortable table, methodology. Charts are
plain SVG; the trajectory viewer supports wheel-zoom + drag-pan. Light and
dark theme follow the OS.
