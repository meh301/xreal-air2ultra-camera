# Evaluation Rules — SLAM Benchmark Program

The authoritative scoring rulebook for this repo's benchmark program.
Every number on the site, in `ITERATIONS.md`, or in a report must be
traceable to these rules; when a rule changes, change it HERE first and
note the date. Companion docs: `bench/PROGRAM.md` (what we benchmark and
why), `bench/ITERATIONS.md` (the verdict ledger), site Method tab
(public summary of this file).

*Last updated: 2026-07-17.*

---

## 1. What one run produces

One replay (`bench/replay/xr_replay_main.c`) over one sequence pack
emits BOTH trajectories, timestamp-aligned:

| track | file | meaning |
|---|---|---|
| VIO | `*_vio.tum` | raw Basalt odometry (with any in-optimizer priors the arm feeds it) |
| map | `*_map.tum` | session pose = live correction ∘ odometry (the AR-visible output) |

The map layer has no retroactive effect: both tracks are **causal**
(§2). Arms differ only in descriptors, retrieval model, matcher, and
env-gated map features — never in scoring.

## 2. Causal protocol

The pose scored at time *t* is the estimate the system had **at time
*t*** ("pose at first estimate"). No final-trajectory export, no
retro-smoothing, no post-hoc loop-closure rewriting. Consequences to
keep in mind:

- Correction snaps are *paid for* at the frame they occur.
- Published SLAM numbers are almost always NON-causal (final optimized
  trajectory) and therefore NOT comparable to ours — see §8.
- ORB-SLAM3's meter-scale causal ATE vs its centimeter published ATE is
  the canonical example of the gap.

## 3. ATE (Absolute Trajectory Error)

1. Associate estimate↔GT by timestamp (nearest, tolerance 50 ms).
2. Align with a **single SE(3) Umeyama** transform (6-DOF rigid, whole
   trajectory, no scale — scale is observable in stereo-inertial).
3. Per-sequence value = **RMSE** of translational residuals (cm).

Implementation: in-process numpy scorer in
`bench/host/export_site_data.py::score_one` (validated against evo to
<0.05 cm on 80 sampled runs).

**Divergence gates (ATE-only):** a run with ATE > 10 m (TUM-VI/EuRoC) or
> 1 m (MSD) is DIVERGED: excluded from aggregation, counted and shown
explicitly (`div (0/3)`). RTE never gates — an RTE gate would
structurally disqualify correction-based systems (one giant frame pair
per snap).

## 4. RTE (Relative Trajectory Error)

TUM-style fixed-delta relative pose error, **Δ = 6 frames** (VOCA/MSD
convention), translational part, RMSE per sequence, gap-guarded (pairs
spanning association gaps are dropped). Reported everywhere, gates
nothing. Snap artifacts are visible here by design.

## 5. Aggregation layers (state which layer you quote!)

| layer | statistic | rationale |
|---|---|---|
| residuals → sequence | RMSE | field standard (Sturm et al.) |
| runs → per-seq value | **median** over N runs | VIO is nondeterministic; ORB-SLAM3's own scripts do the same |
| sequences → group (ERROR metrics) | **median** (mean on hover/secondary) | robust to divergence-prone long seqs at fleet scale; non-standard vs papers' means — deliberate |
| sequences → group (RECALL metrics) | **mean** (median secondary) | recall is bounded and bimodal; a median lands on the easy mode (the 93.3-vs-75.0 incident, 2026-07-17) |

Rules of conduct:
- ALWAYS name the statistic when quoting a number ("median over
  sequences", "mean recall").
- Medians compare with medians, means with means — never across
  (the false "2× lead on TUM-long" incident).
- If a distribution is bimodal, report the modes separately; the split
  is usually the finding.
- Run counts: fleets = 3 runs minimum (5 preferred for baselines);
  fastbench iteration = 3. Long-group fleet noise floor: **±4 cm on
  group medians** — deltas inside it are not verdicts.

## 6. Relocalization benchmark

Cold probes against the run's own finished map (`--reloc N`, default
N=30, seeded RNG → reproducible probe set):

- A probe = a raw frame (or `--reloc-clip K` consecutive frames; the
  best-inlier frame lands) fed with **no VIO context except gravity**:
  the frame's own vio-track orientation quat (a kidnapped device still
  has its accelerometer; yaw is always solved, never trusted). The
  verifier is 4-DOF (yaw+translation) — identity-orientation probes are
  UNFAIR and produce false zeros (the EuRoC/MSD 0-recall incident).
- Error = landed session pose vs the run's own map-track pose at that
  frame (re-entry precision, GT-free, same frame as the map).
- Metrics per sequence: recall, recall@25 cm, recall@10 cm, median
  landing error over verified probes.
- Aggregation: **mean over sequences** (§5). `RELOC-SUMMARY` lines carry
  `clip=K`.

## 7. Robustness (kidnap) protocol

`--kidnap t0,dur`: camera frames replaced by BLACK for the window
(pockets are dark; dropping pairs deadlocks Basalt's IMU queue), IMU
continues, SHAKING held through the window + grace so the LOST latch
lands. Analysis: align map & VIO tracks to GT on the PRE-blackout
segment only, then report windowed median errors (pre / blackout /
post / tail) — whole-run causal ATE double-counts the blind coast.
Success story format: "post-weld X cm vs VIO Y cm".

## 8. Baselines & comparability

- OKVIS2 (LC on/off), ORB-SLAM3 (LC on/off), OpenVINS: **built and run
  by us, same machine, same causal protocol, same aggregation.** These
  are the only numbers we compare against directly.
- Published paper numbers appear ONLY as annotated reference lines
  (`published refs` class on the site) — regime-annotated, never in
  verdict sentences.
- OKVIS2 public code lacks loadMap → cross-system reloc uses the
  blackout protocol; ORB-SLAM3 can use Atlas save/load.
- Mono numbers (Sim(3)-aligned) are never comparable to stereo numbers.

## 9. Run-health & trust checklist (before believing ANY batch)

1. `xr_replay done:` line present, completion ≥ 99%.
2. **`vpr_ep=cuda`** in the done line (a CPU fallback = starved map:
   the curand/gpuenv incident invalidated a full A/B matrix).
3. Store counts sane: ~500 (corridors), ~1000 (magistrale), ~130
   (corridor4), ~220 (rooms). Collapsed stores = starved map thread =
   invalid closure/reloc behavior.
4. No `Exception during initialization` / `running on CPU instead` in
   logs (taint-sweep, delete, re-run — resume guards skip clean jobs).
5. LEDGER funnel lines available for reloc forensics (vprtop → searched
   → cand → bestm → n3 → nin).
6. Log-parsing caveat: stdout (printf: `RELOC k=`, done line) and
   stderr (LOGI) interleave NON-chronologically in log files (buffering)
   — pair probe LEDGERs to RELOC lines by ORDER (last-N), never by
   file position.

## 10. Arm registry (fleet v12+)

| arm | local desc | retrieval | matcher | map features |
|---|---|---|---|---|
| `bad` | BAD/TEBLID | — | NN+margin | baseline map |
| `vpr` | BAD/TEBLID | EigenPlaces-512 | NN | baseline |
| `megaloc` | BAD/TEBLID | MegaLoc-8448 | NN | baseline |
| `xdlg6` | dense XFeat (anchored) | MegaLoc | LighterGlue-L6 (candidate-kf budget) | baseline |
| `full` | dense XFeat | MegaLoc | LighterGlue-L6 | + COVKEEP + PGO + LMDESC + TIGHT (+TIGHTSUB from v13) |

Retired: sparse-xfeat arms (`xfeat/xvpr/xmegaloc`) — the sparse ONNX
export bakes its sampling grid at 480×640 and produces warped
descriptors at every other resolution.

## 11. Known pitfalls (each cost us once)

- Mean-vs-median cross-quote → false "2× lead" (ATE, 2026-07-16).
- Median headline on bimodal recall → "93.3%" vs honest 75.0 (2026-07-17).
- `-D` defines silently overridden by bare in-file `#define` — all
  bench-tunable constants must be `#ifndef`-guarded.
- gpuenv missing `curand` lib → silent CPU VPR → starved maps (whole
  batches invalidated). Hence `vpr_ep=` telemetry + checklist §9.
- Identity-orientation reloc probes vs the 4-DOF verifier → false zeros.
- Replay NEVER latches LOST organically (no shake source) — submap and
  recovery paths are only exercised via `--kidnap`.
- Sequence packs: EuRoC dense arms need the 736-crop packs (XFeat %32).
- Cloned containers inherit completion-marker files (DONE, *.done):
  sweep them before arming watchdogs — a stale marker fires premature
  completion events and can unblock chained launchers early.
- pkill discipline (three incidents): patterns must be self-escaped
  (`nam[e]`), a kill must NEVER share an ssh command with ANY text
  matching its pattern (including rm paths and relaunch text), kill by
  output-path-specific cmdline match rather than binary name when
  parallel workstreams share a binary, and killing a run does NOT kill
  its runner loop — kill the loop first or it respawns/advances.

## 12. Directory & tooling conventions

- Packs: `/mnt/processing/packs` (+`packs736`, `packs4s`) on the
  container; format frozen in `bench/README.md`.
- Results: `/mnt/processing/<batch>` → pulled to `F:\slam_bench\results\<batch>`.
- Scoring/export: `python bench/host/export_site_data.py --results ...
  --baselines ... --reloc <dirs, later overrides earlier> --traj --out
  bench/site/public/data` (score cache: `.score_cache.json`).
- Site: `node bench/site/server.js` (run detached from any agent session).
- Ledger of record: `bench/ITERATIONS.md` — every verdict with commit ids.
