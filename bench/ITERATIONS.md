# Iteration ledger — toward best-in-class tracking on an embedded budget

Living document. Every candidate improvement gets an entry: hypothesis, cost
against the mobile budget, test protocol, and — once run — the commit hash and
the measured verdict. Nothing ships on intuition; the benchmark decides.

## Protocol

- **Tune on the subset, validate on the fleet.** Standard fast loop:
  `/root/fastbench.sh <label> [defines] [env]` — hunt-list sequences we
  LOSE (corridor1/2/3, magistrale2, MH_05) + regression guards we WIN
  (corridor5, slides2, MH_01, room6) + room1 (unlockable) + MOO07/MOO15,
  arms bad+vpr+megaloc × 3 runs ≈ 40 min on GPU. Score with
  F:\slam_bench\score_sweep.py (adjust dir). Only subset winners earn a
  full fleet = 6 arms × 42 seqs × 5 runs (1,260 jobs, ~2–3 h GPU).
- **One change per fleet** so attribution stays clean (v4 → v5 → v6 …).
- **EuRoC is the canary**: the map layer must never hurt a healthy VIO.
  Any "improvement" that regresses EuRoC +map is trust-weighting done wrong.
- **Budget rule**: target is mobile/near-embedded. CPU-side changes must be
  O(existing work); per-frame NPU additions need a measured ms figure; heavy
  models only in per-event paths (closure candidates ≈ few Hz worst case).
- Metric: causal ATE, median-of-per-sequence-medians per dataset group
  (mean tracked as tail indicator). Scoring: `bench/host/export_site_data.py`
  (in-process, ~2 min full matrix). Sweep scoring: `F:\slam_bench\score_sweep.py`.

## Reference targets (same machine, same causal protocol)

| group | our VIO | our best +map (v6) | OKVIS2 | OKVIS2+LC | notes |
|---|---|---|---|---|---|
| EuRoC | 5.8 | 6.45 (bad) → 5.85 v7 | 5.19 | **3.82** | v7: map penalty = 0; gap is VIO-core |
| rooms | 5.6 | 5.56 → 5.51 v7 (xvpr) | 4.43 | **1.22** | their edge = segment reactivation |
| long | 44.9 | 31.1 (megaloc) | 89.6 | **31.8** | **PARITY** (median vs median!). Their profile is BIMODAL: corridor1–3 at 2.8–9.2 but corridor5/mag1/slides2 at 100–180; ours uniform ~20–45. Beating them = moving our working-closure seqs into their 3–9 cm class. (Earlier "we lead 2×" compared their MEAN 63 vs our MEDIAN — wrong, corrected.) |
| MSD | 5.3 | 5.30 | ~2 | ~2 | short seqs; closures rarely engage |

OKVIS2 design notes (what the gaps trace to): corrections enter through
optimization (never pose steps); closures are re-observations of old
landmarks; segment reactivation = tracking directly against the old map when
re-entering it. Their weakness: DBoW/BRISK recall on repetitive fisheye
corridors at large drift.

---

## Completed / verdicts

### ✅ Loop-closure confirmation bugs (the "loops=0" hunt)
- **Commits**: `95787bc` (pending survives sub-gate flicker; window 4→12 s),
  `905cc81` (strong single-frame confirm), `1e4073c` (confirm must be
  re-earned by a worth frame; single-frame confirm LOST-only).
- **Verdict**: rooms/all-arms zero-closure bug fixed; v3 over-correction
  (EuRoC 6.5→20 cm) rolled back by `1e4073c` → v4: EuRoC +map 8.3,
  long 45.2→30.8. **Architectural finding**: map trusts stored anchor over
  live VIO; when VIO > map quality, correct closures still hurt.

### ✅ MegaLoc query starvation → CUDA EP
- **Commit**: `0dec840` (`XR_ORT_CUDA` env, dlsym on dlopen handle).
- **Verdict**: corridor3 retrieval 44→610 queries, 0→1 closures. Replay now
  cadence-representative of Gen 5 NPU. (Device build untouched.)

### ✅ VPR periodic full-recall sweep — VALIDATED, KEEP
- **Commit**: `491b375`. Appearance embeddings alias on repetitive corridors
  (v4: BAD brute 30.8 beat EigenPlaces 38.3 / MegaLoc 44.2 on long).
- **Fleet v5 verdict**: megaloc long **44.2 → 33.5 (−10.6)**, xvpr −4.4,
  megaloc EuRoC −1.8; rooms/MSD flat. No regressions beyond noise.
- **Noise floor measured**: BAD arm (code-identical v4→v5) moved +4.3 on
  long → treat ±4 cm as the long-group significance bar for fleet deltas.

### ❌ Sub-gate correction servo (falsified)
- **Commits**: `491b375` (in), `ba5cc70` (revert).
- **Verdict**: MH_01 map 12→40 cm. With a biased map, any steady pull toward
  it compounds error; 2-frame agreement trivially satisfied at small dev.
  **Blocked on**: confidence weighting (below). Do not re-attempt before it.

### ✅ Tunable sweep infrastructure
- **Commit**: `5dcd1b1` (`#ifndef` guards + Makefile `EXTRA` hook).
- **Sweep 1 verdict** (216 runs, subset):
  - **SNAP_MIN 0.50 wins** — EuRoC ≈ VIO (protected), corridors keep gains,
    magistrale2 91→80. → fleet v6.
  - SNAP_MIN 0.15: EuRoC/mag2 catastrophic **but room1 13→7.7 cm** — early
    small corrections DO help rooms; a global gate can't unlock them
    (evidence for reactivation-lite).
  - cooldown 6 s, sweep-every-4: no clear win.

---

## In flight

### ✅ Fleet v6 — SNAP_MIN_M=0.50 — VALIDATED, NEW BASELINE (on site)
- **v5→v6**: EuRoC bad/vpr 7.6→6.45 (VIO 5.82 — map penalty now ~0.6 cm);
  rooms XFeat-family outlier CURED (xmegaloc 16.2→5.56 ≈ VIO, xvpr −5.3);
  long megaloc 33.5→**31.1 (best arm)**, xfeat −4.2, no losses past noise;
  MSD flat. All three success criteria met.
- **Cumulative v4→v6**: EuRoC +map 8.3→6.45 · rooms outliers gone ·
  long best ~31 (VIO 45) · MSD flat. SNAP_MIN 0.50 is the new default for
  bench builds (still 0.30 in device code — promote after iteration 2
  confirms it isn't masking what confidence-weighting should fix).

---

## Moon shot: map→VIO tight coupling

### ◐ Fleet v7 — pure tight coupling (`XR_TIGHT`) — SPLIT VERDICT
- **Commits**: `e19a061` (app) + basalt fork `55d6563`. Weak unary SE(3)
  priors (σ 7 cm / 2°, 0.7 s expiry) in Basalt's optimizer; posted on
  confirmed closures AND agreeing sub-gate frames; CORR fixed, VIO absorbs.
- **Fleet verdict**: EuRoC **WON** (bad+map 5.85 = VIO parity exactly);
  rooms **WON BIG** (xfeat 13.1→5.58, xvpr 12.7→**5.51 = first sub-VIO
  rooms result**); long **LOST** (megaloc 31→44.8 ≈ VIO — a 7 cm prior
  cannot absorb meter-scale corridor corrections and tight had replaced
  the snap entirely); MSD flat.
- **Lesson**: priors and snaps are complementary regimes, not rivals.

### ⏳ Fleet v8 — HYBRID (`525ef1f`)
- Confirmed closures: prior path only when dev ≤ 0.60 m and ang ≤ 20°;
  larger corrections keep classic snap+deform even under XR_TIGHT.
  CORR-advance follows the path taken. Sub-gate priors unchanged.
- Success = v7's EuRoC/rooms parity AND v6's long numbers (~31) together.

## Iteration 2 queue (ordered)

### ☐ Confidence-weighted deformation
- **Hypothesis**: deform the *weaker* side. Weight from map-anchor age /
  anchor-time VIO health vs live VIO health (tracked-feature count, IMU
  consistency). When VIO healthy and map old → heal the MAP toward VIO
  instead of stepping the pose. Directly targets the EuRoC 8.3→5.8 gap and
  unblocks the servo.
- **Cost**: zero (bookkeeping + existing graph_deform).
- **Test**: subset sweep first (weight-function variants), then fleet.

### ☐ Reactivation-lite (OKVIS2's rooms edge, map-layer-only)
- **Hypothesis**: when retrieval confirms we're inside mapped space, verify
  continuously (cheap PnP against the matched keyframe neighborhood each
  keyframe) and apply small confidence-weighted corrections — not gate-based.
  Sweep-1's snaplo room1 result (13→7.7) bounds the available win.
- **Cost**: one PnP vs ~5 covis keyframes per kf (~ms, CPU).

### ☐ Correction ramp (causal-ATE-friendly application)
- **Hypothesis**: apply confirmed corrections as a ramp over N frames instead
  of one pose step — approximates OKVIS2's optimization-spread correction.
  Whole-run causal ATE stops paying the step penalty; tail unchanged.
- **Cost**: zero.
- **Note**: display-side too (AR comfort: no visible snap).

### ☐ LighterGlue closure verifier (accelerated_features / verlab)
- **Hypothesis**: learned matching on closure *candidate pairs* raises the
  candidates→verified conversion (funnel currently dies at MNN association:
  "26 matches — unverified"). XFeat+LighterGlue ≈ SP+LG at ⅓ cost (MegaDepth
  0.444/0.610/0.746 vs 0.469/0.633/0.762).
- **Cost**: per-event only (few Hz worst case) — CPU/GPU fine in replay, NPU
  port (Gen 5) later. ONNX export via GlueFactory.
- **Test**: A/B verified-closure counts + inlier ratios on subset, then fleet.

### ☐ Retrieval shortlist tuning
- **Hypothesis**: VPR_SHORTLIST 12 too narrow under aliasing; sweep K ∈
  {12, 24, 48} × with/without full-recall sweep interplay.
- **Cost**: K dot-products of 512-D per query — negligible.

### ☐ Relocalization benchmark (new evaluation axis — user-proposed)
- **What**: after a full replay builds the session map, freeze it and feed
  N randomly sampled frames (seeded; single frames and short 1 s clips)
  from the raw data as stationary queries with no VIO context. Measure:
  reloc recall @ (10 cm, 1°) and @ (25 cm, 2°), median position error of
  successful relocs, and time/attempts-to-reloc. Cross-sequence variant:
  rooms1–6 share one physical room and the corridors overlap — map on
  seq A, probe frames from seq B = true VPR generalization.
- **Why**: isolates retrieval+verification quality from drift/closure
  dynamics — exactly the capability behind OKVIS2+LC's corridor
  bimodality, and the axis where MegaLoc-8448 should finally beat BAD.
  Also directly models the product scenario (headset wake / re-entry).
- **How**: xr_map already has the stationary-query path (q_only → full
  scan, RELOC_TOPK, store-suppressed). Needs: a probe API returning the
  PnP map-frame pose for a query image, a `--reloc N` replay mode, a
  scorer comparing against map-aligned GT, and a site tab (recall bars +
  error CDFs per arm).

### ☐ Per-keyframe descriptor→landmark direct index
- **Hypothesis**: BoW systems' direct-index trick; makes post-retrieval
  association O(matches) not O(pairs). Pure CPU savings, frees budget for
  LighterGlue.

## Iteration 3 queue (Basalt integration)

### ◐ XFeat keypoints seed Basalt detection (detector unification) — CHANNEL BUILT
- **Hypothesis**: one detection pass — XFeat dense (NPU, 3.6 ms A8W8/888)
  maxima seed Basalt's corner candidates; KLT still tracks (VIO precision
  preserved). Frees CPU, better-distributed keypoints.
- **Built** (`d4d63b9` app + basalt fork `44fcfa4`): OF-side seed store +
  merge in addPoints (dedup vs corners/tracks at half grid spacing, FAST
  stays as fill), VIT export, xr_slam_seed_keypoints wrapper. NOT yet
  producing: needs replay/app to run XFeat per frame on cam0 and post
  (XR_SEED env in replay; device uses the existing NPU path). Then subset
  A/B (VIO-track quality with/without seeds) → fleet v9.
- **Cost**: net CPU saving on device; GPU-per-frame in replay.

### ☐ Lifetime landmark descriptors
- **Hypothesis**: sample the (already computed) dense descriptor map at
  tracked landmark UVs every keyframe → landmarks carry multi-viewpoint
  descriptor statistics → wide-baseline association robustness up (compounds
  with LighterGlue). Plumbing half-exists (xr_slam reads Basalt landmark DB).

### ☐ Tight coupling (verified map observations → Basalt factors)
- **Hypothesis**: the actual OKVIS2 mechanism. Heavy surgery on Basalt's
  marginalization; hold until the above shows what gap remains.

## Parked / rejected

- ❌ Sub-gate servo without confidence weighting (see above).
- ❌ RTE as a divergence gate (`7b6b29a`) — structurally kills causal-LC
  systems; ATE-only gates, RTE reported.
- ☐ (someday) EVA/CVP fixed-function DFS for stereo — hardware access path
  documented in F:\slam_bench\CVP_EVA_BRIEF.md.

---

## Fleet history

| fleet | change under test | data | verdict |
|---|---|---|---|
| v2/matrix1+all | 6 arms baseline (broken closures) | matrix_all | superseded |
| v3 | closure fixes (over-eager confirm) | matrix_v3 | EuRoC regression — rolled back |
| v4 | `1e4073c` tightened confirm | matrix_v4 | current site baseline |
| v5 | + VPR full-recall sweep | matrix_v5 | ✅ keep (megaloc long −10.6) |
| v6 | + SNAP_MIN 0.50 | matrix_v6 | ✅ new baseline (site) |
