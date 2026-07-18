# Iteration ledger — toward best-in-class tracking on an embedded budget

> Scoring rulebook: **bench/EVALUATION.md** (metrics, aggregation, trust checklist). Quote numbers per its rules.

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

### 2026-07-18 — v2 rounds: both REJECTED, both informative → v3s fired
**radv v2 (radv2_ab)**: the dev>=0.10m + confirmed-site gates KILLED the
mechanism — mag2/slides2 fired ZERO rebuilds (their apparent deltas =
round variance: mag2 ctrl has swung 55→205→96 across three rounds!);
corr3 fired 0-1/run. v1's active ingredient was FREQUENCY (~12
rebuilds/run via the sub-gate stream) — TIGHT absorbs continuously so
per-closure corrections are tiny and a dev gate can never pass.
**radv v3** (committed): v1 sites+frequency, SPACE selection via the
fold discriminator (covis>=6 && vpr_alias_margin > LMMARG_ALIAS_MARGIN)
— mag1's aliased space has structurally low margins -> auto-disable.
radv3_ab fired (.15): n=10 on mag1/mag2/corr3 (the variance-heavy
verdict seqs), n=5 corr5/slides2.
**ZNCC floor (znf_ab): INVALID — units.** Basalt images are 16-bit;
floors 3/6 (8-bit thinking) never engaged; znf3-vs-znf6 deltas were
noise. Valid floor probe (1000, 16-bit) runs in the new round.
**ZNCC v3 = stage 9b transient gating (committed)**: XR_ZNCC=2 arms the
ZNCC branch only while a windowed frame-mean delta (>3.0 8-bit units
over 10 frames, 2s hold) detects a photometric transient — catches
V2_03 flicker AND MH_05's slow ramp, leaves dark/low-texture scenes on
stock normalization. Smoke: detector armed on V2_03 (dmean 3.66).
znt_ab fired (.58): ctrl/znt/znf1k on V2_03, MH_05, V1_03, slides1/2,
MOO02, n=5.

---

### 2026-07-18 — P3b held-out: NOT freeze-able as-is → radv v2 (quality-gated)
s12_heldout (.15, within-round, n=5, cm): the mag2/corr3 transformation
did NOT generalize — **mag1 99.2→134.4 (tail 438!)**, corr5 18.0→22.8,
slides2 33.9→39.0 harmed; corr2 18.0→16.9 small win; corr4 neutral;
EuRoC canary passes (MH_01/MH_05/V1_02 medians within noise; radv MH_01
tail 7.4 vs ctrl 68.4 outlier — actually TIGHTER). Mechanism read:
re-linearizing the window consistent with a MARGINAL closure amplifies
it — v1 triggered on any verified closure incl. the sub-gate site.
**radv v2** (committed): trigger only at the CONFIRMED-applied site,
gated covis>=6 && nin>=LMMARG_MIN && evidence-ratio>=60% && |corr|>=0.10m.
XR_RADV=2 selects v2 (1 = old). radv2_ab fired (.15: mag1/mag2/corr2/
corr3/corr5/slides2, ctrl vs radv2, n=5) — mag2/corr3 must keep their
wins, mag1/corr5/slides2 must go neutral.

### 2026-07-18 — ZNCC events A/B: V2_03 flicker TRANSFORMED, low-texture harm → floor v2
zn_em (.58, n=5, cm): **V2_03 17.2→6.8 (−60%, spread 6.7-7.1)** — the
designed-for flicker case delivers. MH_05 (exposure ramp) 18.3→19.2
(+61.5 outlier): ramps are gain-like, mean-norm already handles them.
Harm class confirmed low-texture: slides2 37.9→54.3, slides1 +2, MOO02
12.2→15.0, V1_03 5.7→6.7; MOO13 neutral, MOO01 noisy-neutral.
**ZNCC v2** (committed): XR_ZNCC_FLOOR sigma floor (intensity units) —
at the floor the normalization degrades toward plain mean-removal, so
weak-texture patches stop amplifying noise. znf_ab fired (.58: bracket
floor 3.0 vs 6.0 vs ctrl on slides1/2, MOO01/02, V1_03 + keep-case
V2_03, n=5).

---

### 2026-07-18 — DBoW2 APPLES-TO-APPLES VERDICT (user-requested)
Same probes, same verify/PnP pipeline (LighterGlue+PnP), only retrieval
swapped (XR_RELOC_CANDS = DBoW2/ORBvoc top-5 vs our EigenPlaces path);
single-frame rl19 protocol, RELOC-SUMMARY recall:
| seq | ours | DBoW2-fed | | seq | ours | DBoW2-fed |
|---|---|---|---|---|---|---|
| corr1 | 93.3 | 90.0 | | corr4 | 90.0 | 96.7 |
| corr2 | 86.7 | 80.0 | | corr5 | 96.7 | 100 |
| corr3 | 90.0 | 100 | | mag2 | 50.0 | 50.0 |
Mean 84.4 vs 87.8 — retrieval engine largely WASHES OUT after
verification (top-5 + geometric verify rescues weak top-1s; DBoW2's
retrieval-only 37-80% r1@3m understated it). med_err similar; corr5
r@10cm drops 90→77 with DBoW2 (spatially coarser candidates).
**mag2's 50% is NOT retrieval: 13/15 failed probes IDENTICAL across
engines** — place-not-in-map / matcher-level; next lever is matching
(burst rb19 already lifts mag2 to ~85%), not VPR. Practical note:
DBoW2 is CPU-cheap — viable fallback tier on-device; EigenPlaces keeps
the corr2-style wins and stays the pick.
Caveats: n=1 per arm per seq; DB cap 400 + ±3s exclusion matched to our
kf budget (dbow2_bench.cpp).

---

### 🏆 2026-07-18 (later) — P3b A/B: REVERSIBLE MARG DELIVERS IN BIG SPACES
s12_ab (.15, within-round, n=5, ATE cm, canonical scorer):
| seq | ctrl (fz19v2) | radv (+DELAYMARG+RADV) |
|---|---|---|
| magistrale2 | 204.7 (84-417) | **49.5 (36-78)** −76%, tail collapsed |
| corridor3 | 28.0 | **10.4** (best runs 7.6-7.9) |
| corridor1 | 32.4 | 30.2 (neutral — rigid-return autopsy holds) |
| rooms 1-6 | — | neutral, consistent ~+0.2 cm micro-cost |
60 readvances / 5 runs each (2 s debounce). The ABSORB-class mechanism
works exactly where the decomposition predicted: big aliased spaces
where folded-wrong info needed recovery. vs okvis2+lc reference bands:
mag2 49.5 would LEAD (their ~66); corr3 10.4 reaches their 2.8-11.4
band edge (from 28). corr1 remains the fortress (tracking-level).
NOT YET A FREEZE: held-out round fired (s12_heldout: mag1, corr2/4/5,
slides1/2, MH_01/MH_05/V1_02 canary) — must pass EuRoC-no-harm + the
rooms micro-cost must stay ≤noise.

### 2026-07-18 — ZNCC clean verdict (s910 zn refire, consistent lib, n=5)
zn (fz19v2+XR_ZNCC) vs fz19v2, ATE cm: **corr3 31.9→15.6 (−51%)**,
mag2 55.1→50.6, rooms uniformly −0.1..−1.1 (7/10 seqs win), corr1 +1.6,
**slides1 25.4→28.5 (the one harm — dark projector room)**. Events test
queued on .58 (zn_em: MH_05/V2_03/V1_03 + MOO01/02/13 + slides1/2 n=5,
after lib rebuild to stage-12). libcmp flag-off check: stage-11 lib corr1
7-43 cm bimodal = historical fz18 distribution ✓ no flag-off regression.

### 2026-07-18 — gate2_em wave scored (fz162 = fz19v2 on euroc/MSD, n=3)
MH_01 6.6 (false-LC jump CURED by covis gate), MH_03 5.4, MH_05 18.4,
V1_02 5.5, V2_02 6.4; MOO: 01 20.9, 03 15.6, 05 2.0, 07 1.5, 09 0.4,
13 33.7, 15 27.3. Site cell refresh pending the radv/zn freeze decision.

---

### 🏆 2026-07-18 — XRV stage 11 VALIDATED: reversible marginalization works inside basalt's sqrt form
- **P3a re-advance engine** (XR_DELAYMARG capture + XR_READVANCE rebuild):
  after a six-fix debugging arc, the stress smoke (readvance every 8th marg
  event, 12 full prior rebuilds installed, room1) scores **ATE 0.128 m vs
  baseline 0.129 m** — the rebuild is numerically transparent. The six
  fixes, in order (each smoke-verified, all in patch_stage10/11.py):
  1. IMU factors only between 15-dof frame_states (ImuBlock .at()s the
     states map; pose-demoted endpoints excluded).
  2. Prior-order resurrections need setLinTrue (computeDelta asserts the
     FEJ contract on every marg-order var).
  3. Stock keeps young sliding states OUT of the marg prior: aom/new-order
     restricted to prior-connectivity vars; live imu factors must NOT fold
     (double-count). Vel-bias-demoted window kfs get PROMOTED to 15-dof
     (pose at its own FEJ lin point + captured vel/bias), vel/bias columns
     re-marged per-column, object demoted back after.
  4. addLandmark copies neither obs nor the host→target index — resurrected
     landmarks re-add every obs via addObservation (≥2 in-aom obs guard:
     1-obs landmark QR is degenerate).
  5. Flow-lost landmarks are folded by stock at every marg event (the
     event's main vision info) — captured in full (lost_lms) and replayed
     via the lost_landmarks selection channel.
  6. **THE diverger** (30 km fly-off): the post-install rebase
     `marg_data.b -= H·delta` (stock sqrt_keypoint_vio.cpp ~1553,
     "recover delta-independent residual"). Without it the installed prior
     re-counts the entire accumulated delta as fresh residual — gradient
     diagnostics showed g_rebuilt ≈ b_incumbent exactly.
  Diagnostics that cracked it: dry-run mode (side effects w/o install →
  ATE parity proved prior content was the sole corruption), then per-var
  H-diag/b/gradient segment logging (H matched to 0.05%, b 500× off,
  g_n ≈ b_i ⇒ missing rebase term).
- **P3b closure trigger** (stage 12, XR_RADV): vit_tracker_xreal_readvance
  API chain + xr_map radv_trigger (2 s debounce) at both verified-closure
  sites. Smoke: 12 closure-triggered rebuilds, room1 map quick-ATE 0.070
  (typical ~0.13). A/B queued: s12_ab.sh on .15 (ctrl vs radv, rooms+
  corr1/3+mag2, n=5) — runs after the zn refire drains.
- **ZNCC verdict update**: +10 offset fix confirmed ALIVE and healthy
  (pure-VIO room1: base 0.129 / zncc 0.137); value case is the photometric
  events (MH_05 ramp, V2_03 flicker) — zn arm refired clean on .15
  (previous zn results were mixed-lib contaminated: the batch ran through
  the rebuild storm).
- **UNITS DISCIPLINE**: the site/per-seq tables are in **cm** ("map median
  ATE cm"). corr1 fz18 "36.9" = 36.9 cm. A units slip this session briefly
  manufactured a phantom 70× corridor regression + phantom .181 anomaly —
  both dissolved on canonical rescoring (corr1 fz18 med 0.37 m ✓ = 36.9 cm
  table row ✓ same batch).
- **Lockstep verdict (partial, .181 r234_lock)**: corr1 lock 5/5 runs at
  0.10 m vs control 10-run spread 0.10-0.40 med 0.13 — variance collapse
  confirmed on corr1; corr3 inconclusive (lock DNF'd at 5400 s timeouts;
  completion refire queued at 21600 s). XRV corridor A/Bs should run
  lockstep-on when wall-clock allows.
- **DBoW2 apples-to-apples queued** (.58 dbw_a2a.sh): dbow2_bench refired
  with per-probe CAND top-5 dumps → XR_RELOC_CANDS feeds them through the
  UNCHANGED verify/PnP pipeline on the same rl19 probes; control = the
  existing rl19 grid.

---

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

### ? Fleet v8 � hybrid verdict: EuRoC/rooms HELD (bad 6.47=VIO, xfeat rooms cured 5.57), long STILL LOST (megaloc 31->44).
Root cause: TIGHT_MAX_DEV (0.60) > SNAP_MIN (0.50) routed ALL confirmed closures (which arrive at 0.5-0.6m by construction) into the weak-prior path. The EuRoC/rooms wins come from SUB-GATE priors alone. Fix: confirmed closures always snap (TIGHT_MAX_DEV=0) - fastbench v9tight0 running (first 22-wide 15-min loop).

### ? 4Seasons driving group (NEW dataset � 'see the limit')
- Stereo GS 30fps 800x400 gray, 30cm baseline, IMU 2000Hz, RTK-fused GT
  (GNSSPoses.txt 7DOF keyframes; times.txt maps frame_id->unix ts).
  Downloading 3 training recordings (2 City-Loop-class 8.3/7.6GB + 1 mid
  5.8GB) to /mnt/processing/4seasons + calibration. TODO: prep converter
  (undistorted pinhole -> pack; 800x400 -> crop H to 384 for XFeat arms),
  euroc-style toml, new site group 'drive'. NOTE fastbench v9tight0 was
  INVALID (bare #define overrode -D; guards added) - v9 fleet relaunched
  clean and is the real test.

### ? Relocalization benchmark LIVE (f53dc62 + deadlock fixes)
First baselines (vpr arm, 25 cold single-frame probes): room1 recall
24% / r@25cm 8% / med 0.48m; corridor3 recall 4% / med 0.61m. Cold
single-shot reloc is our worst axis - funnel dies at association (the
LighterGlue target) and precision at single-kf PnP (multi-kf refine
target). TODO: clip-probes (temporal accumulation = realistic wake),
cross-sequence probes, per-arm comparison (MegaLoc should shine here),
site tab. Two concurrency bugs found by the bench itself: lost-wakeup +
MAP_LOCK self-deadlock (gdb) - both fixed, both product-relevant.

### ? Reloc site tab (user request) � pipeline extended
RELOC lines now emit exp=x,y,z (map-track pose at probe frame) and
got=x,y,z (landed pose) so the site can plot expected->landed vectors.
NEXT: exporter parses reloc logs -> data/reloc.json (per seq x arm:
probes + recall/r@25/r@10/med); site 'Reloc' tab = summary bars + spatial
plot on the orbit canvas (trajectory + expected->landed lines colored by
error, failures as X at expected). Cross-system blackout-reloc protocol
queued (OKVIS2 public code lacks loadMap - clip probes impossible there).

### ? Fleet v9 � confirmed-always-snap: rooms CURED (xfeat 13->5.55), EuRoC ok, long STILL lost (megaloc 31->43).
Eliminates the confirmed-closure routing as the cause. Root cause = SUB-GATE priors: on corridors they glue VIO to recently-mapped (equally drifted) kfs, preventing drift from ever crossing the gate where the big closure fires. v6 (no tight) = only config holding long 31. NEXT (v10): sub-gate prior ONLY when matched kf is a genuine revisit (age > ~30s) - old-kf agreement = loop info; recent-kf agreement = teaching VIO its own drift. Site stays on v6 until v10.

### ? v10/v11 � revisit-age gate (TIGHT_REVISIT_NS 30s) � SUBSET WINNER
fastbench v10revisit: EuRoC parity held (5.3-6.0), corridor1 30.1 = v6,
corridor5 23.8 & slides2-megaloc 29.9 = NEW BESTS, mag2 <=97.5. The gate
(priors only vs kfs >30s old) keeps tight-coupling wins AND long gains -
the unifier. Fleet v11 validating (same defines, XR_TIGHT).

### ? Blackout-splice generator (/root/blackout.py)
pack + mav0 modes: drop camera frames for [t0,t0+dur) while IMU
continues - the cross-system reloc protocol every baseline can run
(OKVIS2's forced-reloc path; ORB3 also has atlas save/load for true cold
probes). Next: generate blackout variants of room1/corridor3/MH_01,
run ours + okvis2 + orb3, add recovery metrics to reloc tab.

### ? 4Seasons converter (/root/conv4s.py) � CONVERTING
Standalone container converter replicating pack formats exactly (pinhole
=kb4-zero-dist, TS_cam_imu inverted for cam->IMU, GNSS scale applied to
GT, ADIS16465-inflated noises). 3 recordings converting in parallel ->
/mnt/processing/packs4s/{drive1_city,drive2_city,drive3_country};
xr_replay_drive (800x400) built; /root/out/drive.toml staged.

### ? Iteration-2 trio IMPLEMENTED (04614dd) � fastbench series running
- Reactivation-lite (XR_REACT): anchor-pinned single-kf verify @400ms
  inside mapped space; priors via tight channel. fastbench v12react.
- Correction ramp (XR_RAMP): display correction glides 500ms; map updates
  instantly. fastbench v13ramp (chained).
- Confidence-weighted corrections (XR_CONFW): healthy closures blend by
  inlier-ratio (40% floor, full @0.65). fastbench v14confw (chained).
- CAVEAT: v12 overlaps fleet v11 (34 jobs on 64 cores) - cadence-
  sensitive results preliminary; re-run winners on an idle box.

### ? 4Seasons first light: drive3_country replays 100% (24,629 frames,
1.6M imu) but VIO diverges to 100+km span (GT 1.7km) - extrinsic
convention suspect (TS_cam_imu direction). Variant B (no inversion)
smoke running; driving toml tuning next. Remaining queue: LighterGlue
verifier, lifetime descriptors + direct index, stage-3 coupling (parked).

### x Trio verdicts (fastbench, 3 runs/arm, vs v10revisit base)
- v12react: REJECT as-is. Wins MH_01 (4.9-5.4) + slides2/mag2-megaloc,
  but corridors collapse (corr2 bad 29->50, corr3 megaloc 40->59): the
  400ms anchor pin STARVES full retrieval exactly where distant loops
  pay the rent. Rework = pin as an ADDITIONAL cheap query, never a
  replacement (or drop anchor on movement > COVIS_R).
- v13ramp: was 100% NaN-poisoned (first-call blend read uninitialized
  R.q0 -> normalize(0) -> inf*0). FIXED (init q0/p0 + backdated t0).
  Salvage-scored (NaN rows stripped): ~ATE-neutral; wins slides2-vpr
  42->32, mag2-megaloc 97->83; loses corridor3 +4-11. Verdict: keep OFF
  by default - display-smoothness feature, not an accuracy feature.
- v14confw: REJECT as-is. slides2-vpr 42->27 headline, but MH_01-bad
  DOUBLES (5.3->10.6) + megaloc regressions (MH_05 23->32, corr5 24->33).
  Blending weakens exactly the strong closures it should trust.
- MOO15 "landing" under all three variants (28-73cm, 1-2/3 runs) vs v10
  all-div = near-gate coin flips, not signal.

### x RELOC BENCHMARK WAS UNFAIR — gravity fix (probe grav_q)
Full 13x3x30 grid exposed it: EuRoC + MSD probes = 0/30 on ALL arms
while mapping-phase PnP in the SAME runs verifies fine (nin 30-50).
Root cause: pnp2_ransac is 4-DOF (yaw+translation; roll/pitch TRUSTED
from the query's odom orientation) and probes passed IDENTITY - fatal
on any tilted frame / non-level body convention (EuRoC/MSD), partial on
level-ish handheld TUM-VI (rooms 20-63%, corridors ~3%, corridor4 43-53%).
Deployed system unaffected (live queries always carry VIO gravity).
Fix: xr_map_probe(img, grav_q, ...) + harness passes the frame's own
vio-track quat (gravity true, yaw ignored/solved - a kidnapped device
still has its accelerometer). v1 no-gravity results archived at
reloc_batch_v1nograv; full grid RERUNNING. r@10cm ceilings should also
lift (residual tilt was capping precision).

### x SAME-SESSION SUBMAPS built (the 5588-kf field freeze answer)
xr_kf.seg + CUR_SEG: LOST > SEG_OPEN_NS (10s, shake over) -> new segment
opens, mapping resumes in the continued odom frame (quarantined: pc-space
pooling/clustering/gating all seg-guarded - cross-frame pc distances are
meaningless). Cross-segment verified+CONFIRMED (always 2-frame: welds
rewrite stored poses irreversibly) closure -> rigid WELD: older segment
wins the frame (primary canonical), younger segment's kfs re-registered
via E, ids merged, CLOUD_DIRTY. Weld while LOST = recovery too. Caps +
confw skipped for cross-seg D (offset arbitrary by construction).
Replay NEVER latched LOST (harness never called freeze_storage - MOO15 &
magistrale1 smokes confirm) -> built --kidnap t0,dur: camera blackout
with SHAKING held + grace (on_pose's freeze(0) raced the latch away),
IMU continues, Basalt coasts blind -> genuine post-gap odom discontinuity.
MH_01 (60,15) + corridor1 (100,15) kidnap smokes running: expect LOST ->
submap seg=1 -> WELD -> map-track ATE recovers.

### x SUBMAP + WELD VALIDATED (kidnap smokes, b228d96/f299ca8)
--kidnap v2 = black frames not dropped pairs (dropped pairs deadlock
Basalt's bounded IMU queue; black is also the honest pocket-cam sim).
MH_01 (60,15): LOST -> seg=1 opens @+10s -> 153 kfs mapped in orphan
frame -> return to hall -> WELD (offset 1.50m 4deg, 2-frame confirmed)
-> post-weld map 36cm vs VIO 99cm (pre-blackout-aligned GT frame).
corridor1 (100,15): seg=1, 181 kfs, WELD offset 0.76m 1deg -> final
mocap window map 30.9cm vs VIO 94.3cm. Within-segment closure fired
correctly during the orphan phase (seg guards hold).
WRINKLE for later: pre-weld, within-segment display snaps move the
(arbitrary) segment registration - MH01 transiently 133->181cm vs GT.
Option: suppress display snaps while CUR_SEG unwelded.
FIELD IMPLICATION: the 5588-kf freeze becomes: 10s LOST -> mapping
resumes seamlessly; map heals when any old ground is re-seen.

### x Reloc grid v2 (gravity-fair) - the real portrait
recall / r@25 / med(when verified): MOO07 70%/70%/5cm (was 0),
MH_01 30-47%/17-33%/12-25cm (was 0), rooms 47-77% with r@25 ~= recall
(med 2-11cm - PRECISION IS FINE, the old 'single-kf PnP imprecise'
story was the gravity artifact), corridors still 7-13% (corridor4
50-57% - it re-walks its map). Retrieval arms ~equal on rooms; nothing
lifts corridors yet -> recall on repetitive/sparse-coverage areas is
THE reloc hunt (LighterGlue verifier + coverage-aware retrieval).
4Seasons drive: NOT units/extrinsics/images (all verified clean +
in-spec; VIO tracks ~30s then scale error compounds 3.2x @2min ->
explosion). Suspects: lever arm, cam-IMU time offset, BMI160 noise
config under engine vibration. PARKED pending core program.

### x FLEET v11 VALIDATED -> NEW SITE BASELINE (matrix_v11, exported)
Group medians map-track v11 (vs v6): euroc bad 5.79 (6.44) = VIO PARITY,
megaloc 5.97 (6.91); rooms flat ~5.6; LONG bad 30.46 (38.06), vpr 31.40
(38.80), megaloc 30.12 (31.12) - ALL THREE ARMS now at/below OKVIS2+LC
long 31.78; msd flat 5.30. The revisit-age gate (priors only vs kfs >30s
old) is the unifier: tight-coupling EuRoC wins + long-group gains, no
regressions. Site exported: results(v11) + baselines + gravity-fair
reloc.json + 520 traj files.

### x XFeat reloc arms + LighterGlue verifier (user callout: reloc had
### only tested BAD local descriptors - the 3 arms varied retrieval only)
reloc_xfeat grid (13 seqs x xfeat/xvpr/xmegaloc x 30 probes, EuRoC via
736-crop): XFeat lifts ROOM recall decisively (room3 90-97% vs BAD
73-77%, room2 83-90% vs 63-73%, corridor4 67-73% vs 50-57%) but
precision drops everywhere (med 0.11-0.34m vs BAD 0.02-0.11m) and MSD
regresses hard (MOO07 17-20% vs BAD 70%). Corridors 1/2/3/5 STILL dead
(0-13%).
LighterGlue INTEGRATED (xr_lighterglue.c: dlopen ORT, static-512 export,
caller-side kpt normalization, int8/127 dequant, pad+index-filter;
--lglue flag; reloc_pnp routes XFeat correspondence generation through
learned matching, NN fallback): corridor1 3.3->6.7% - real but marginal.
ROOT CAUSE isolated (probe LEDGERs): bestm 48-109 image matches collapse
to n3 0-36 landmark-backed pairs - kp->landmark association (8px join)
starves the 2D-3D stage; n3>=30 converts (~50% inliers, verifies),
n3<15 dies. Retrieval fine (vprtop 0.76+), matching fine, geometry
starved. Second mode: bestm~0 probes = viewpoint never mapped (one-way
corridor walks) - no matcher can fix that; needs map-side coverage.
NEXT UNLOCK: dense XFeat export for the ORT path + bilinear descriptor
sampling AT landmark uvs (exact 2D-3D association, like BAD's anchoring
and the device NPU dense tail) -> should fix corridor reloc recall, MSD
xfeat regression, rooms-xfeat 13cm fleet outlier, AND xfeat-arm loop
closures in one move.

### x DENSE-ANCHORED XFEAT: THE UNLOCK LANDED (f3e3131)
Two root-cause fixes shipped together:
1. SPARSE EXPORT WAS BROKEN OFF-TRACE-SIZE: onnx/xfeat.onnx bakes its
   InterpolateSparse2d normgrid divisor at the 480x640 trace ([479,639]
   constant) - at 512x512 y-scale ~25% off, at 640x480 axes TRANSPOSED.
   Every container xfeat result to date ran warped descriptors ("worked"
   only because query+store warp identically). Device unaffected (raster
   = trace; NPU is dense). Replaced by xfeat_dense_dyn.onnx (symbolic
   shapes, torch-validated at 3 resolutions, cos 0.996).
2. LANDMARK ANCHORING: dense C tail (float twin of the NPU tail) +
   xr_xfeat_sample -> descriptors AT landmark uvs, exact lm_of_kp.
Smoke (30 probes): room1 100% recall / r@25 100% / med 9.6cm (BAD 63%,
sparse-xfeat 57-73%); MOO07 100% / 9.4cm (was 70% BAD, 17% corrupt);
corridor3 16.7 -> 26.7% with LG6, med 3.4cm. Funnel healed: bestm=64 ->
n3=55 -> nin=49 (was 80 -> 13 -> 0).
LighterGlue L6 (FULL official 6-layer; our L3 was a half-depth cut, and
torch 2.13's dynamo exporter silently produced a degenerate L6 graph -
dynamo=False fixed it, validated 406/410) helps on CLEAN descriptors
(+10pts corridor3); on corrupted ones it was worse than L3 (deep
attention trusts descriptor-position consistency more).
Remaining corridor failures = bestm~0 probes: viewpoints never mapped
(one-way walks) -> map-side coverage is the next axis.
IN FLIGHT: dense_batch.sh (92 jobs) = full 13-seq reloc grid x
{xdense, xdenselg6} + hunt-list ATE x3 runs (does anchoring move the
corridor1-3 LOOP-CLOSURE hunt + the rooms-xfeat 13cm outlier?).

### x A/B FEATURE SET COMPLETE (user: "implement them all, flag-gated")
All remaining architecture items now built behind env flags (default
OFF), one binary A/B-tests everything:
- XR_COVKEEP  - viewpoint-diversity eviction (spatial cell 1.5m + yaw
  quadrant, per-segment; evict most-redundant, ties to LRU)
- XR_SEGQUIET - no display snaps from within-segment closures while a
  submap is unwelded (kidnap 133->181cm transient)
- XR_PGO      - Gauss-Seidel pose-graph relaxation replaces graph_deform
  (odom edges skip >10s gaps; closure prior w=4 on a virtual query node;
  12 alternating sweeps)
- XR_LMDESC   - lifetime landmark descriptor bank (8192 direct-mapped,
  freshest desc per landmark id, anchors win) + DIRECT 2D-3D reloc
  channel when retrieval fails while relocalizing (coverage-independent;
  covis proxy = distinct live owners among inliers)
- XR_REACT2   - reactivation anchor as ADDITIONAL candidate (fixes v12's
  pin-starves-retrieval corridor collapse)
- XR_MAPSEED  - stage-3-lite coupling: near a stored keyframe, reseed the
  VIO detector with the map's own landmark uvs (works with XR_SEED=1);
  full landmark-3D priors ride on this later
- --reloc-clip K (harness) - probes become K consecutive frames (waking
  device sees a stream); best-inlier frame lands; summary carries clip=K
All-flags room1 smoke: clean, 10/10 reloc @ clip=5, r@25 100%.
DENSE-STACK VERDICTS (dense_batch, arrived mid-implementation):
- Full reloc grid: MH_01+LG6 100% recall (BAD 30-47%), room4+LG6 93%
  (was 27%), corridors 27-37% w/ LG6 (was 3-13%), med 3-12cm. ANOMALY:
  corridor4 0/30 on both dense arms (was 43-73% everywhere else) - run
  mapped fine, probes never verify; OPEN.
- Hunt ATE (3 runs): corridor2 25.5 NEW BEST (VIO 50.1, v11 28.2),
  corridor1 27.8 new best; LG6 = closure-quality gate (MH_01 xdense 19.0
  -> lg6 6.6). corridor5-lg6 45 (vs xdense 25.1) + corridor3 mixed =
  needs fleet-scale runs. room1 map==vio 13.1 (no closure gain yet).
NEXT: A/B matrix over the new flags on the hunt list (one flag at a
time vs xdenselg6 base), then union of winners -> fleet.

### x A/B MATRIX SCORED + THE MAP-DENSITY CONFOUND
Store-count audit first: 20-wide all-MegaLoc batches STARVE the map
thread (dense extract + embed per pass under GPU contention) -> corridor
maps collapse from ~570 kfs (BAD arms, 12-wide) to 45-63 (dense arms,
20-wide). The dense-arm corridor reloc "collapse" and the RELOCSWEEP
non-result were DENSITY artifacts, not retrieval aliasing; the OOM-
tainted 323-store maps were closest to truth. Arms whose flags add
map-thread cost self-starved in the matrix (react2 60 / mapseed 123 /
tight 52 stores) - their ATE numbers discarded, clean 10-wide retest
chained. ARCH QUEUE: protect store throughput under load (skip embed on
store-only passes; congestion-aware search skip).
VALID ARMS vs dense+LG6 base (3 runs, hunt list):
- XR_COVKEEP: corridor1 7.1 (base 27.8!!), corridor3 27.7, MH_05 16.8,
  magistrale2 39.9 (base 86.9; v11 77-94; OKVIS2+LC 66.4 - FIRST TIME
  UNDER THEM) - viewpoint-diverse retention keeps closure anchors alive
  on long routes. Loss: corridor2 33.1 vs 25.5, MOO15 39 vs 31.
- XR_PGO: corridor1 10.5, corridor3 36.3, magistrale2 73.1; loss
  corridor2 50 vs 25.5. Composes with COVKEEP (different subsystems).
- XR_LMDESC (ATE): mixed; its RELOC result is the point:
### x LMDESC = THE CORRIDOR RELOC BREAKTHROUGH (ab_reloc, dense+LG6+bank)
corridor1 46.7% recall / r@10 43% / med 4.8cm, corridor2 53.3%,
corridor3 36.7%, corridor4 83.3% (r@10 77%), corridor5 40%, rooms
93-100%, MH_01 83%, MOO07 90% - the direct 2D-3D channel bypasses
retrieval exactly where it fails. clip15 similar except corridor1 0/30
= starved map (68 stores), not a clips failure.
IN FLIGHT: clean 5-wide corridor reloc rerun (xdense/xdenselg6/sweep) ->
union_batch chained (union=COVKEEP+PGO+LMDESC vs base + clean retests of
react2/mapseed/tight + union reloc grid) at 10-wide.

### x ROOT CAUSE OF THE STARVATION EPIDEMIC: my gpuenv rewrite dropped curand
The PID-spread gpuenv.sh rewrite omitted $SP/curand/lib from
LD_LIBRARY_PATH -> libonnxruntime_providers_cuda.so failed to load ->
EVERY run after that rewrite (the SCORED A/B matrix, reloc_clean v1,
reloc_sweep, union v1) ran MegaLoc on CPU at ~229ms/embed. That - not
LG alone - is the primary map-thread starvation source; LG-on-CPU
(30-80ms x up to 32 calls/search) compounded it. FIXES:
- gpuenv.sh: curand restored (verified "VPR: CUDA EP enabled" in live logs)
- xr_lighterglue.c: CUDA EP (same dlsym contract as vpr)
- reloc_pnp: LG budget = candidate keyframe only (was every pooled kf: 8x)
- XR_STOREGUARD flag: store-only passes skip the embed (device-relevant)
STATUS OF PRIOR VERDICTS: the A/B matrix numbers (COVKEEP corridor1 7.1,
magistrale2 39.9; PGO corridor1 10.5; LMDESC reloc corridors 37-83%) were
measured on CPU-VPR degraded maps - directionally promising, magnitudes
unconfirmed. Clean re-measurement running: clean corridors -> union_batch
with 8 arms (union, base2, covkeepc, pgoc, lmdescc, react2c, mapseedc,
tightc) x hunt list x3 at 10-wide, all-CUDA verified. Watchdog monitor
live; adversarial code-review workflow over the session's changes in
flight.

### x UNION MATRIX SCORED (10-wide, mostly healthy maps) + 25 REVIEW FIXES
Union (COVKEEP+PGO+LMDESC on dense+LG6) vs base, hunt medians:
corridor3 35.1 -> 12.6 (!!), corridor2 36.9 -> 20.7, magistrale2 91.9 ->
70.8, MOO15 43.0 -> 25.5, slides2 45.4 -> 32.4, corridor1 33.7 -> 28.6.
EuRoC REGRESSES (MH_01 11.2 -> 19.0); tight alone = 5.9 (parity) -> the
deploy config is UNION+TIGHT (compose: prior channel vs eviction/deform/
bank). Residual infra noise: sporadic VPR-CPU fallbacks at 10-wide (GPU
OOM transients; CPU-retry keeps runs alive; vpr_ep= telemetry added).
Adversarial review workflow (34 agents): 25 confirmed findings, ALL
FIXED (ce92fd4): XR_SEED extract race (compound extract_anchored API),
xfeat init fast-fail (silent BAD-under-xfeat-label!), LMB epoch clear,
lost+younger weld false recovery, probe react pinning, REACT2
standalone, SEGQUIET recovery exemption, universal duty cap w/ full pass
cost, dense shape validation, clip clamp, +15 more.
IN FLIGHT: fixed_matrix (base/union/tight/union+tight x hunt x3 + utf
reloc grid, FIXED binaries — the duty-cap fix changes cadence so all
arms re-measure) -> fleet decision.

### x FIXED-MATRIX VERDICT (review-fixed binaries, healthy maps ~500/1025)
ATE medians (hunt, 3 runs): utf (FULL = dense+LG6+COVKEEP+PGO+LMDESC+
TIGHT): MH_01 6.5 / MH_05 17.5 (EuRoC = tight parity), corridor2 20.1,
corridor3 14.0 (base 53.0), magistrale2 45.7 (base 90.6; OKVIS2+LC 66.4
— BEATEN on the hardest long seq), slides2 42.1, MOO15 35.8.残り
bimodality: corridor1 33.9 (unionf alone 11.6), corridor5 42.6 (unionf
16.5) — closure hit-or-miss at 3 runs; fleet medians will settle it.
utf RELOC grid (30 probes, healthy maps): corridor1 46.7%/r@10 43%,
corridor2 43%, corridor3 46.7%, corridor4 96.7%/r@10 93%, corridor5 60%,
slides2 76.7%, rooms 93-100% (med 4-9cm), magistrale2 10%. vs BAD arms'
3-13% corridors = 4-15x. The stack: dense anchoring fixed association,
LG6 fixed matching, LMDESC bank fixed retrieval-independence, healthy
maps fixed density.
FLEET v12 LAUNCHED: 600 jobs, 5 arms (bad/vpr/megaloc/xdlg6/full), all
40 seqs x3, 10-wide, sparse-xfeat arms retired. Watchdog live.

### x FLEET V12 VALIDATED — 'full' IS THE NEW BASELINE; LONG GROUP WON
600 jobs (40 seqs x 5 arms x 3 runs, review-fixed binaries, all-CUDA; 6
CPU-fallback runs detected via vpr_ep telemetry and re-run clean).
Group medians (map): euroc full 6.61 (VIO 6.47 = parity; best map arm),
rooms 5.21 (=VIO, closures gated), **LONG full 27.04 — OKVIS2+LC 31.78
BEATEN by 15% on the group median**, msd 6.20 (=VIO).
Long per-seq full vs OKVIS2+LC: corridor5 15.8/101.5 WIN, slides2
43.5/160.5 WIN, slides1 27.0/31.8 WIN, magistrale2 58.5/66.4 WIN,
magistrale1 195.9/179.8 close, corridor4 18.0/11.4 close, corridor2
21.8/9.2, corridor3 15.9/4.9 (was 49.7 on megaloc!), corridor1 32.2/2.8
(the remaining sore spot — their corridor1 LC is near-perfect).
Site exported with v12 + utf reloc grid. REMAINING GAPS (the next arc):
euroc 6.6 vs 3.82, rooms 5.2 vs 1.20, msd 6.2 vs 2.05 — all the same
mechanism (continuous map reuse below VIO drift level) = stage-3
landmark reprojection + reactivation, now unblocked by the lifetime
descriptor bank.

### x SUBGATE VERDICT: TIGHTSUB keeper (corridor bimodality), rooms need true stage-3
fullb vs fullsub vs fullsee (rooms+euroc+msd focus, 3 runs): TIGHTSUB
halves corridor2 same-batch (50.4 -> 26.3; sub-gate priors catch drift
early = less closure bimodality), EuRoC mixed-mild, corridor5/MOO15
neutral. ROOMS UNTOUCHED - root cause: TIGHT_REVISIT_NS=30s excludes
nearly all room-scale closures (everything is re-seen within seconds).
Sub-VIO rooms = landmark-level factors (stage-3) where young
re-observations are arbitrated per-point, not pose priors.
MAPSEED rejected AGAIN post-race-fix (V1_02 5.7->21.0, MOO15 div).
DECISION: XR_TIGHTSUB joins the 'full' arm for the next fleet.

### x 4SEASONS SUPERLONG UNLOCKED — two root causes, both required
Research-verified (libartipy SDK source): the distributed 'undistorted'
images are STEREO-RECTIFIED (cv2.fisheye.stereoRectify,
CALIB_ZERO_DISPARITY); TS_cam_imu is the RAW cam0 frame -> our packs
carried a 1.484deg cam-IMU rotation error (2.6% gravity leak). AND the
IMU noises were 40-200x below DM-VIO's proven 4Seasons values
(vibration-inflated: gyro_nd 0.0412, accel_nd 0.102).
Short-pack isolation (198s, GT 896m span): old calib 11,980m (12.3x
path) -> rect extrinsic 1,141m (1.277x) -> rect + DM-VIO noises 834m
(**1.014x — 1.4% scale error**). Bouguet split numerically self-verified
(caught a sign-eating yaml parse + a handedness flip before they hit the
bench). All 3 drive packs rebuilt (calib_orig.txt preserved); full-drive
first light running: 3 drives x {xdlg6, full+TIGHTSUB}, shutdown-hang
nanny active (xr_dr_drive hangs in a futex at exit — separate bug,
harness-side thread join, to fix later).
### x EXTRA DEFINES WERE NEVER WIRED — label correction, verdicts survive
Makefile.linux never referenced $(EXTRA): every -D passed through it
(the program's entire history) was INERT. True constants of every build:
SNAP_MIN_M=0.30 (not the labeled 0.50), TIGHT_MAX_DEV_M=0.60 (hybrid
routing, not the intended 0.0). AUDIT: all env-flag A/Bs (trio, union,
tight, TIGHTSUB, dense/LG6 stack, fleet v12, baselines-ours) are VALID —
arms shared identical constants; only labels lied. INVALIDATED: the
historic SNAP_MIN sweep (identical binaries; '0.50 wins' was noise) and
v9's '=0.0' define intent (inert — v9's gain was its code change).
Fixed 76dac2d: CFLAGS += $(EXTRA); placebo defines stripped from all
batch scripts so the newly-working flag doesn't change fleet-validated
semantics. OPEN: a REAL SNAP_MIN 0.30-vs-0.50 A/B. Big-map drive round 3
relaunched on a verified 2000-kf binary (BSS 128MB vs 29 = define in).
Caught by: round 2's log showing kf#199 under -DXR_MAP_MAX_KF=2000 — the
trust checklist (EVALUATION.md 9) gains: verify a define took effect by
an OBSERVABLE (BSS size, log constant echo) before believing its A/B.

### x AUGMENTATION A/B VERDICT (iteration container, 6 seqs x 5 arms x 30 probes)
Mean recall: base 45.0 / SEQVOTE 48.9 / DESPERATE 38.4 / ROTSTORE 31.7 /
all 38.3. SEQVOTE = KEEPER (wins 5/6: corr3 +10, corr4 +13, mag2 +6 —
temporal voting works). DESPERATE = large-map profile ONLY (corr4
67->30: wide+low-floor shortlist floods RANSAC with aliases; but mag2
+13 — the huge-space regime it was built for; stays in drive r4).
ROTSTORE = REJECT at 200-kf cap (store churn eats coverage; revisit
with big caps). Fleet v13 (TIGHTSUB validation) predates this verdict —
SEQVOTE folds into the freeze config pending fulleig; the deploy env
becomes FULL + XR_SEQVOTE=1.

### x RETRIEVAL VERDICT: EIGENPLACES WINS INSIDE FULL — deploy pick flipped
fulleig (EigenPlaces-512 in FULL+TIGHTSUB) vs MegaLoc counterparts:
RELOC (vs augbase, same config): corr1 46.7/36.7, corr2 50/40, corr3
43/43, corr4 100/66.7(!), corr5 60/56.7 — corridor mean 60.0 vs 48.7.
ATE (vs utf, TIGHTSUB-confounded): corr1 22.8/33.9, corr5 15.3/42.6,
slides2 30.2/42.1, rooms tie; ONLY magistrale2 regresses (51.1/45.7).
Mechanism hypothesis: with LG6+anchors carrying verification, retrieval
only needs DIVERSE nomination — the weaker 512-D embedding spreads the
shortlist; MegaLoc concentrates confidently on look-alikes. MegaLoc
keeps the LARGE-MAP profile (magistrale2-class, drives — pairs with
XR_DESPERATE there).
GEN5 IMPACT: retrieval 5ms / 45MB / 2KB-per-kf (from 25ms / 457MB /
33KB) — total resident model memory ~60MB.
FLEET v14 = THE FREEZE FLEET queued behind v13 on bench-1: arms bad /
xdlg6(meg) / fullm(meg+SEQVOTE deploy-twin) / full(EIG+SEQVOTE = the
deploy config). v14's full-vs-fullm is the definitive fleet-scale
retrieval pair; v14 'full' median table = the APK freeze numbers.

### x Retrieval-flip caveat (user challenge): verdict was INDOOR-ONLY
The EigenPlaces win was measured on corridors/rooms; published MegaLoc
advantages + our magistrale2 result both live in the large/outdoor
regime. OUTDOOR PAIR RUNNING: drive reloc r4-eig (iteration container)
vs r4-meg (bench-2 chain), identical 2000-kf/60m/DESPERATE/RELOCSWEEP
config. Deploy decision holds only if EigenPlaces doesn't collapse
outdoors; otherwise the pick becomes PROFILE-SPLIT (eig indoor / meg
large-map+outdoor — both models resident is fine within the 1GB budget).

### x FLEET V13 (TIGHTSUB at scale): NOT confirmed — big-space harm signal
v13-full (meg+FULL+TIGHTSUB) vs v12-full: euroc/rooms/msd ties; LONG
30.98 vs 27.04 — regression at the noise-floor edge, concentrated in
BIG spaces (corr5 45.0/15.8, mag2 78.7/58.5, mag1 252.8/195.9) while
corridor2 keeps the subgate win (20.0). Mechanism: in large aliased
spaces, wrong sub-gate matches occasionally 2-frame-confirm and post
bad priors — the exact failure the deviation gate existed to block.
DECISION RULE for the freeze: v14-full (eig+SEQVOTE+TIGHTSUB) long vs
v12's 27.0 decides — if ≥~31 again, TIGHTSUB drops from the deploy env
(SEQVOTE stays); refinement candidate: distance-guarded TIGHTSUB (post
sub-gate priors only for matches within corridor-scale range).
OUTDOOR PAIR (partial): eig drives 0/30 with ALL mitigations; funnel
dies at DESCRIPTOR MATCHING (bestm 22-31 vs indoor 60-120) — retrieval
moot; drive single-frame reloc is a different problem domain; indoor
deploy flip unaffected. Meg twin pending for the airtight pair.

### x STAGE-3 LANDMARK FACTORS: BUILT, SMOKE-VERIFIED, A/B QUEUED
Agent-built (worktree merge a912858 + nested basalt 6c34d3e): verified-
closure inlier landmarks -> fixed-3D reprojection factors in the sqrt
estimator (Huber 3sigma, <=32/frame, pose-block Jacobians numerically
validated 5e-10), session->odom via CORR^-1, posted at both tight sites,
NO revisit-age gate (per-point arbitration makes room-scale reuse safe —
the thing pose priors could not do). Container basalt patched+rebuilt
(patch_lmfact.py, symbol exported), room1 smoke: full proof chain — 146
posts, "32/32 factors applied", 99.9% completion, vpr_ep=cuda.
LMFACT A/B queued on iteration (rooms+euroc+msd x3, arms lfbase/lfonly/
lfts) — THE rooms-1.2cm attack. If lfonly closes rooms toward OKVIS2+LC,
stage-3 enters the freeze config.

### x WRONG-RESOLUTION POSTMORTEM: all outdoor "XFeat" results were BAD-descriptor runs
Every 4Seasons drive log (eig side AND the meg r3/r4 rounds) carried
~2000-2700 "XFeat dense: feats 64x48x100 != 64x50x100" errors: the dense
backbone floors H,W to /32, 800x400 -> 48 rows vs the compiled 50, shape
guard dropped EVERY extraction, keyframes+probes silently fell back to
BAD. The "outdoor single-frame reloc is matching-bound, retrieval moot"
verdict is hereby DOWNGRADED to BAD-descriptor evidence only. Fix
(fc73042): zero-pad inference input to next /32, all dense-grid math on
padded dims; clean resolutions bit-identical; drive3 smoke 0 errors.
Affected dirs carry INVALID_WRONGRES markers (drive_reloc, drive_reloc_big,
drive_reloc_r4 on bench-2; drive_reloc_r4eig on iteration). Redo running:
drive_reloc_pad_eig (.15) / drive_reloc_pad_meg (.58), r4 config, 3h caps.
Also: probe/sweep searches now emit the closure LEDGER (was vpr-gated —
the drive probes were invisible) with cov= appended.

### x STAGE-3 LMFACT A/B: REJECT for freeze (rooms unmoved, corridor harm)
lfbase/lfonly/lfts x (rooms 1-6, corr2/5, MH01/03/05, V102, MOO07/09/15) x3.
Aggregate map median 5.60 / 5.84 / 5.59 — flat. Rooms: no effect at all
(room1 13.13->13.08, room2-6 within noise) — the rooms gap vs OKVIS2+LC
is BA-quality-bound, not drift-bound; fixed-3D factors derived from our
own map reinforce the status quo when drift is cm-scale. Corridor harm:
corr2 20.4->24.5 (lfonly) / 28.4 (lfts), corr5 15.1->22.2 (lfonly).
Small MH gains (MH01 6.6->5.9) don't pay for it. XR_LMFACT stays available
as a flag; future angles: stronger weight / >32 factors / drift-scaled
activation. NOT in the freeze config.

### x SNAP_MIN REAL A/B: 0.50 KEEPER (now compiled default)
First honest sweep post-EXTRA-fix (binaries verified different): hunt
subset map median sn50 18.24 vs sn30 26.76. corridor1 30.4->10.1,
corridor5 31.9->20.1, magistrale2 55.1->45.4, slides2 -1; rooms flat;
only corridor2 +7.5. SNAP_MIN_M default raised 0.30 -> 0.50.

### x CLIP GRID: clip-15 wake-up burst >= clip-1 everywhere
corr1 50->53, corr2 53->57, corr3 30->57, corr4 87->100 (r@10 90),
corr5 60->63, mag2 10->23, room1 100->100 (r@10 70->87). Mean recall
55.7 -> 64.8. A ~0.5 s frame burst is what reloc needs; clip-15 becomes
the standard wake-up scenario alongside single-frame probes.

### x infra: bench-2 container moved 10.27.48.180 -> 10.27.48.181 (same fs,
chain PIDs survived); stage-3 of bench2_chain (big-map r3 + r4-meg, old
unpadded binary) killed by PID — superseded by the pad redo. Blackout
OKVIS outputs live in baseline_reloc/ (MH01_bo, corr1_bo, room1_bo).

### x FLEET V14 (freeze fleet, 480 runs): FREEZE CONFIRMED
Arms bad / xdlg6 / freezem(=FULL w/ MegaLoc) / freeze(=FULL w/ EIG+SEQVOTE
+TIGHTSUB), 40 seqs x3. All-seq map median: freeze 7.64 / freezem 8.03 /
bad 10.27 / xdlg6 10.71 — best program-wide aggregate to date.
Groups (med): euroc 6.62/6.72 (EIG~=meg), LONG 23.21 vs 41.39 (EIG ~2x
better — retrieval flip is decisively confirmed where it matters),
rooms 5.22 flat (gated), msd 6.15 flat. TIGHTSUB KEPT: freeze long 23.21
< v12 no-TIGHTSUB 27.04 (v13 harm signal was MegaLoc-context). Site arms
renamed freeze/freezem (v14 'full' config differs from v12/13 'full').
vs baselines: long BEATS OKVIS2+LC 31.8; euroc still behind OKVIS2+LC
3.82 (VIO-bound); rooms gap stands. Freeze block written to
GEN5_DEPLOYMENT.md §6. Remaining gates: blackout baselines, exit-hang.

### x pad-drive redo: first nonzero outdoor reloc
With real dense descriptors (post-fc73042): drive3 3/30 verified (was
0/30 with BAD), drive2 still 0, drive1 mapping. MegaLoc twin relaunched
on .58 after shipping xr_slam.c (stage-3 wrapper was .15-only — link
error caught the gap). Site: results.json 3774 rows / 13 arms, clip grid
in reloc.json, server on :8080 via .claude/launch.json (bench-site).

### x LMFACT SIGMA SWEEP: strength does not unlock rooms — stage-3-at-
closure-instants is SATURATED. lfs075/lfs125 aggregates 5.60/5.54 vs
lfbase 5.60 (rooms flat everywhere, corridor2 harm at every sigma:
21.5/26.0 vs 20.4). sigma=1.25 marginally best -> azmax carries it.
The rooms attack now rides on LMTRACK (continuous posting) + LOCALBA
(structure) in the aug matrix.

### x DRIVE ATE with REAL descriptors (pad runs, single-run, vs OKVIS2):
drive1 10.6km: OURS 144m/1.35% vs okvis2 353m/3.32%, +LC 344m/3.23%
  (2.4x BETTER on the longest drive; padeig d1 = redo on .181)
drive2 10.8km: ours eig 451m/4.18% meg 339m/3.14%; okvis2 lc0 DIVERGED
  (25,408 km ATE — total failure), lc1 pending
drive3 5.1km: okvis2 61.9m/1.22% beats ours (meg 101m/1.99%, eig 172m).
eig-vs-meg drive deltas are single-run VIO variance — do not over-read.
Net: we hold 1.4-4.2%% everywhere; OKVIS2 catastrophically fails drive2.

### x hunt15 queued (.15 idle after sweep): aug_long (azbase/azall/azmax
x corr1/3/4+mag2+slides2 — the big-space harm gate for LMTRACK/LOCALBA)
+ fb_drives (FARBEAR outdoor reloc, drive2/3 eig — the feature's target).

### x AUG-LONG HARM GATE: azall is a BIG-SPACE WINNER
azbase/azall/azmax x (corr1/3/4, mag2, slides2) x3, freeze base:
corr1 28.6->12.3, corr3 17.9->12.1, corr4+slides2 flat, mag2 50->66
(2-run cell, highest-variance seq). Aggregate 28.6 -> 18.7. LOCALBA+
LMTRACK+LMFACT actively HELPS corridors — the corridor-harm signal from
lfonly-era closure-instant factors INVERTS once structure is refined and
posting is continuous. azmax (adds sigma1.25+FARBEAR) is WORSE than
azall (corr3 35.5) — suspect FARBEAR indoors: 6m-gate rejects are not
genuinely far, bearing approx invalid -> biased yaw votes (azfb arm on
.58 isolates). FARBEAR outdoors: drive2 0->1/30, drive3 3->1/30 —
neutral, no recall breakthrough. Awaiting .58 matrix (rooms/euroc/msd)
before freeze-config promotion of azall.

### x BURSTPNP A/B (clip-15 wake-up, single runs): KEEPER
vs clip15 baseline: corr1 53.3->50.0 (r@10 40->43), corr3 56.7->63.3
(r@10 43->50), mag2 23.3->30.0 with med_err 0.22->0.10m. Mean recall
+3.3pts, mean r@10 +5.6pts, accuracy up across the board — the joint
solve fired on 14/18/9 of 30 probes. The wake-up story is now: 15-frame
burst + joint offset-aware 4-DOF consensus. Enters the freeze validation
round together with PGO4DOF.

### x DRIVE1 DEADLOCK ROOT-CAUSED AND FIXED (3594b26)
gdb on the wedged specimen: main blocked in xr_slam_push_imu (bounded
imu queue FULL), estimator blocked pushing out-states only main drains,
optical flow blocked behind it — a two-bounded-queue cycle with a
single-threaded harness. Drive imu = 2000 Hz -> ~100 blocking pushes
per inter-frame burst with zero polling. Fix: drain every 8 imu pushes.
Killed EVERY drive1_city attempt (3/3, kf~1600-1800). Redo relaunched
(.181, both retrieval arms, fixed harness).

### x XR_DEPTHFILL built + smoke-verified (5cd6f91 + 189264a)
Per-keypoint epipolar ZNCC on the rectified pair at kf store: drive3
smoke +2..+18 synthetic landmarks/kf (67 vs ~45 lm), auto-armed only for
distortion-free packs. A/B on drive2/3 reloc queued (.181) vs padeig
0/30, 3/30 baselines.

### x VIO SWEEP (EuRoC, the estimator-bound gap): vkf KEEPER-CANDIDATE
vbase 6.50 -> vkf 5.82 (denser keyframing: min_frames_after_kf 3,
kp_thresh 0.8), vobs 5.89 (obs_std 0.35; V1_02 4.48 best cell),
vhub 6.03, vfix flat, vwin CRASHES basalt (abs_order_map assertion at
max_states 5 — reject). Combos (vkfobs/vkfhub/vall) running on .15.
Path to OKVIS2's 5.19/3.82 is open.

### x OKVIS2 drive baselines FINAL: drive2 diverges in BOTH modes
(lc0 25,408 km / lc1 65,908 km ATE). Complete drive scoreboard: ours
1.35-4.2%% of path everywhere; OKVIS2 wins only drive3-country.

### x VIO COMBO: vkfobs KEEPER — EuRoC 6.50 -> 5.18, EDGES OKVIS2-no-LC (5.19)
vkfobs (min_frames_after_kf 3, kp_thresh 0.8, obs_std 0.35) = 5.18 med;
vall 5.24 (huber adds nothing), vkfhub 6.02. MH_05 18.6->14.7, V1_02
4.57. The estimator-bound gap is HALF closed by config alone. vkfobs
becomes the VIO tuning of the freeze candidate (euroc config; port the
same ratios to tumvi/msd configs next fleet).

### x DEPTHFILL outdoor A/B: marginal recall (drive3 3->4/30, drive2 0),
med err halves (0.94->0.50m) on 925 backfills. 3D availability was not
the last outdoor bottleneck either — remaining loss is verify-thresholds
vs genuine 60m-range aliasing. Next diagnostic: MASt3R oracle on failed
probes; also consider r@1m as the honest outdoor metric. DEPTHFILL kept
(free accuracy + denser maps), not a recall unlock.

### x SCOREBOARD CORRECTION (user-caught): MSD is NOT ours — OKVIS2+LC 2.29
Site MSD tab: okvis2+lc 2.29 (13 seq) < okvis2 5.85 < ours 6.13-6.15.
Every "MSD: we lead" claim was wrong — it compared against openvins/orb3
and stale memory, never the okvis2+lc row ON THE SITE. Honest scoreboard:
we lead LONG (23.2 vs 31.8) and drive robustness; okvis2+lc leads euroc
(3.82 vs 5.18 tuned), rooms (1.20 vs 5.2), msd (2.29 vs 6.15) — i.e.
every map==VIO-gated group. The azall matrix (running) is the pivotal
verdict for all three. RULE (now in EVALUATION.md): never claim a lead
without quoting the site's own group table for EVERY baseline.

### x OVERFITTING GUARDRAIL (user directive): tuning arms are selected on
SUBSETS (vkfobs on 5 euroc seqs; azall on hunt seqs). NOTHING enters the
freeze config on subset evidence alone: every keeper must be confirmed
on the FULL 40-seq fleet (held-out seqs included) before adoption, and
per-seq regressions > noise on held-out cells veto the keeper.

### x RELOC FUNNEL AUDIT (user-driven): the hand-off defect is the NN
prematch gate, NOT PnP. 450 probe frames/seq: 49-56%% die at bestm<24
while verification passes 90-92%% once a candidate exists; accepted
frames carry vprtop 0.86-0.92 (retrieval excellent). The weak matcher
gatekept the strong one. FIX: XR_TRUSTVPR (c9c1917) — vprtop>=0.75
forces the top-retrieval kf into candidates; LG+PnP arbitrate. A/B
chained on .15. mag2 remains the aliasing case (41%% die at PnP,
correctly; burst fusion is its lever).

### x EDGEGRAPH v1: REJECTED on measurement (24.1 -> 35.1, corr2 87cm,
mag2 diverged). Root cause is architectural: our LIVE pose couples to
the map only via CORR — whole-chain relaxation moved the map under a
live trajectory that never heard about it (OKVIS2 can relax everything
because its live state IS in the graph). Also sub-gate edges at 1 Hz are
self-drift-correlated (v9's lesson applies to edges) and 0.5m-soft DCS
kept bad edges alive. v2 (76210b4): applied-closure admission only,
phi=0.04, CORR re-derived from the relaxed tip. A/B chained after tv_ab.

### x STAGE-3 FLATNESS ROOT-CAUSED (activity audit of matrix logs):
azlt/azall posted ~236 factor batches/run, LOCALBA fired ~55x/run — the
mechanisms were LIVE and the ATE still did not move. Two causes: (a) in
low-drift regimes the factor 3D comes from a map the same VIO built
minutes earlier — the factors pull toward where the estimator already
is, BY CONSTRUCTION; (b) our basalt patch applies factors in optimize()
but never folds them into the marginalization prior — the information
EVAPORATES when the frame margs out (OKVIS2 persists reobservation info
through marginalization; that is the room-regime mechanism we lack).
Real fix = marg-persistent factors (deep basalt patch, frontier). Also:
LMIDX bank fallback fires 0.1-0.3x/run in mapping mode — kidnap-only
value, keep but do not grow. ACTIONABLE COROLLARY: rooms are estimator-
precision-bound like euroc -> port vkfobs to the tumvi config (chained
on .58: vtbase vs vtkfobs, rooms1-6 + corr1/3).

### x TRUSTVPR A/B (single-frame probes, means over 3 runs): KEEPER
corr1 51->53, corr2 37.8->61.1 (+23.3, r@25 35->56), corr3 55->53 (noise),
corr4 80->83, corr5 69->70, mag2 10->13. Mean over seqs 50.6 -> 55.7.
No harm, precision intact — the funnel fix pays exactly where repetitive
structure starves the NN prematch. Enters the freeze-confirmation set
with BURSTPNP (they compose: burst pools what trustvpr admits).

### x MSD-LOSERS DIAGNOSTIC: estimator-bound at the ROOT, not closure-bound
mbase/mlf/meg/mall all cluster 23-25 (vs okvis2+lc 2.3-4.9). Closures DO
engage (3-7 applied; MOO13 map 33 vs vio 48) but our VIO drifts 27-48cm
where OKVIS2 RAW VIO holds 5.7-15.1 — a 3-5x estimator gap BEFORE loop
closure even matters. Same disease as euroc -> vkfobs port to the msd
config chained on .58 (losers + MOO07/09 no-harm guard). The coupling
ladder is exonerated on this group: it cannot fix drift the estimator
creates 5x faster than the baseline's.

### x EDGEGRAPH PARKED after v3 (transactional accept still diverges):
the consistency metric measures INTERNAL agreement — an aliased edge
folding the chain into a self-consistent wrong shape passes it. v1-v3
arc: power is real (corr1 29->8 under v2 on a clean loop) but pose-only
edges cannot be made safe at our verify quality. Safe version = image-
level constraints inside the relaxation (revived landmark observations,
i.e. the marg-persistent-factor frontier). Flag stays, default off.

### x STATISTICAL WARNING (from eg3's off arm): corridor n=3 medians swing
2-5x between identical rounds (corr1 6.3 vs 29.1; corr3 14.7 vs 35.9).
Every corridor A/B verdict this session carries that caveat. RULE: the
held-out confirmation fleet runs corridors at n>=5; single-round corridor
deltas below ~2x are treated as noise.

### x VKFOBS PORTS: 3-for-3 KEEPER. tumvi: EVERY room improves (room6
1.59 — okvis2+lc at 1.20; rooms 5.2-5.6 -> 4.5-4.8; RTE down across the
board) — the FIRST mechanism to move rooms all session. msd: MOO01
23.3->19.1, MOO13 31->29.6, MOO07 1.48->1.18, no harm — helps but the
remaining ~2x to okvis2 raw VIO on MSD is beyond config tuning
(frontend/motion-regime difference; frontier). FLEET v15 launched:
fzbase (v14 cfg) vs fz15 (vkfobs all-configs + TRUSTVPR), full 40 seqs,
corridors n=5, held-out seqs included — the freeze decision.

### x KFCAP A/B (n=5): 400-KF CAP KEEPER for corridors — corr3 35.0->22.1
AND its 2 diverged c200 runs vanish; corr1 -5, mag2 -6, corr2 -3 (noise),
corr4 flat. The 200-cap eviction was discarding revisit anchors. RAM
~+40-80MB, fine per Gen5 budget. Folded INTO fleet v15's fz15 arm
(composition purity: fz15 = vkfobs-all-configs + TRUSTVPR + cap400);
both fleet halves restarted with fz15 rebuilt, fzbase runs preserved.

### x DRIVE1 "PROBE HANG" ROOT-CAUSED (2nd bug, distinct from imu-burst):
fd audit of the live specimen — all pack files at EOF, main stuck in
push_imu_sample, pipeline empty-idle. The harness's TAIL imu flush pushed
post-last-frame residue (~350 samples at 2kHz) into a queue that only
drains while frames follow. Indoor tails (200Hz) fit by luck; drive1's
crosses capacity deterministically = every historical drive1 "wedge" at
the end of stream. Fix bd76979: no tail flush (frameless imu is useless
to the estimator). drive1 eig+meg relaunched with both fixes.

### x DRIVE1 COMPLETE (first full runs ever, both fixes): eig map ATE
118m / 1.11%% of 10.6km — 3x BETTER than okvis2 lc0/lc1 (353/344m,
3.3%%); meg 211m/1.99%% (map) 139m/1.31%% (vio). Probes 0-1/30 — the
outdoor single-frame reloc conclusion is now complete across all drives.

### x FLEET15 tumvi half (n=5): fz15 (vkfobs+TRUSTVPR) CONFIRMS on
held-out — rooms 5.47->4.85 (EVERY room better, room6 1.51 vs okvis2+lc
1.20), long flat (24.0 vs 25.0, in-noise). COMPOSITION HONESTY: the
cap400 amendment never took effect (pkill self-match killed the amend
step on both containers) — fleet fz15 = vkfobs+TRUSTVPR at cap200.
Cap-invariant for rooms/euroc/msd (cap never binds); the untested
vkfobs x cap400 corridor interaction cell now runs on .15 (n=5).
euroc/msd half: launcher died twice (CRLF re-poisoning after re-scp; the
sed was killed by its own pkill) — fixed, resumed, 10 workers.

### x FLEET15 FULL VERDICT: vkfobs VETOED by held-out (the discipline works)
Full 40-seq, corridors n=5: rooms 5.47->4.85 (uniform, real) BUT
corridor1 6.66->24.95 (+18), magistrale2 47.8->68.8 (+21), mag1 +7.7,
MSD losers reversed (MOO01 +3.4), euroc subset promise 5.18 -> 6.39
(wash). TRUSTVPR is inert in mapping ATE (fires only while relocalizing)
so the scatter is pure vkfobs: denser keyframing is a big-space closure
lottery. THREE subset A/Bs had "confirmed" it — the held-out fleet
killed it. FREEZE BASE STAYS v14 config. Surviving keepers: TRUSTVPR +
BURSTPNP (reloc scenario), infra fixes, cap400 (interaction cell
pending). Follow-up: moderated point (kf_after 4, kp 0.75, obs 0.42)
on rooms + the harmed/helped corridors, fzbase fleet cells as comparator.

### x CAP400 INTERACTION CELL: does not rescue vetoed vkfobs (corr1 25.0,
mag2 75.9 under fz15+c400) — but its same-round base-config keeper
evidence stands, and with vkfobs vetoed THAT is the shipping
composition. XR_MAP_MAX_KF default 200 -> 400 (rooms/euroc/msd never
bind; corridors gain same-round; RAM +~40-80MB within Gen5 budget).
Cross-round corridor variance note: fzbase corr1 swung 6.66 <-> 28
between rounds at n=5 — corridor numbers are only comparable WITHIN a
round; the rulebook inherits this.

### FREEZE v15-FINAL = v14 flag set + XR_MAP_MAX_KF 400 + SNAP_MIN 0.50
+ TRUSTVPR & BURSTPNP (reloc/wake-up path) + the five pipeline fixes
(exit _exit, imu-burst polling, tail-flush, /32 padding, ledger telem).
vkfobs vetoed; LMFACT/LMTRACK/LOCALBA/FARBEAR/EDGEGRAPH remain flags,
off. Remaining frontiers: marg-persistent factors, MSD frontend gap.

### x VMOD (moderated tuning): half the rooms gain, FULL corridor harm
(room6 1.80 vs 1.51/2.07; corr1 24.4, mag2 70.3 — same as the vetoed
extreme; cross-round comparator caveat applies but the dose-response
shape is unambiguous). Conclusion: the keyframing tradeoff is REGIME-
BINARY, not dose-graded — no uniform setting gets the rooms prize.

### FRONTIER (designed, parked): ADAPTIVE KEYFRAMING — legitimate (not
overfitting) iff switched on an online-observable causal variable:
median triangulated landmark depth (rooms <5m, corridors/outdoor deeper)
— mechanism: fixed window + denser KFs = less per-KF baseline = costs
parallax only in DEEP scenes. Threshold from a parallax-angle target
(first principles, NOT tuned to benchmark splits); validated per-regime
held-out. Needs a basalt runtime-setter patch (kf policy fields are
static config today). Expected prize ~0.5-0.7cm on rooms — parked in
favour of the reloc moat per user strategy.

### STRATEGY (user): uniform behaviour everywhere -> the moat is
RELOCALIZATION. Remaining reloc levers, in order: (1) XR_INVIDX inverted
index (recall at scale, the last unbuilt lever), (2) time-to-relocalize
latency metric in the rulebook + harness (unmeasured, AR-critical,
differentiating), (3) multi-hypothesis reloc for aliased spaces (mag2),
(4) thumbnail retention + learned verify (Gen6 watch list).

### x STAGE-4 MARG-PERSISTENCE: THE BREAKTHROUGH (user-called hunch)
lmoff/lmev/lmmg, rooms n=3 + corr2/5 n=5 same-round: rooms NEARLY HALVE
(room1 13.1->6.4, room2 4.9->3.3, room3 5.6->3.3, room5 18.1->11.2),
corridors IMPROVE (corr2 22.4->18.1, corr5 21.8->18.7), aggregate
9.35 -> 5.02. The evaporating control (lmev~=lmoff) reproduces the old
flatness exactly — mechanism validated end-to-end: post -> keyframe-
anchored survival -> fold at kf-marg. Rooms gap vs okvis2+lc cut from
4.5x to <3x in one change. FLEET v16 (fzbase vs fz16 = LMFACT+LMTRACK+
LMMARG) launching for the held-out confirmation.

### x MICRO-FLEET: cap400+TRUSTVPR reloc step-change; kp256 promising
Reloc (cross-round caveat vs tv_ab2, but the jump is far beyond noise):
corridors 94-99%% single-frame recall (was 43-73%%), mag2 44-47%% (was
10-23%%) — the compiled cap400 default removed revisit-anchor eviction.
Missing cells = probe-phase TIMEOUTS at the bigger maps (latency metric
will quantify; probe duty may need relief). sh20 REJECT (slower, no
gain); tv065 insensitive (keep 0.75). ATE: kp256 21.6->18.1 subset —
promote to held-out cell; cad02 mild (-1.6).

### x FLEET16 MARG-PERSISTENCE: rooms CONFIRM held-out, big-space variance
Full 40-seq: rooms 5.32->3.87 (-1.46, uniform: room1 -5.2, room5 -6.5),
long 27.1->21.0 (-6.1), euroc/msd flat. 5/40 regress >2cm — BIMODAL on
aliased big spaces: mag1 fz16=[43.5,130,137] vs base=[64,73]; corr3
fz16=[7.7,8.1,21,32,42]. Root cause: a wrong-place closure now folds
PERMANENTLY into the marg prior (can't evaporate) — poisons aliased
spaces where the covis+2frame gate still passes wrong matches. Same
class as EDGEGRAPH, milder. FIX: confidence-gate the FOLD (nin>=high)
while keeping transient posts ungated. fz16.1 building.

### x INVIDX: REJECT (redundant with TRUSTVPR). recall 81.1 vs 81.7, +3ms
— TRUSTVPR already admits the top-VPR kf, no truncation gap left. But
LATENCY METRIC = GOLD: single-frame time-to-relocalize ~55ms, ttv ~55-61
(sub-frame; the AR-critical number nobody else measures). Keep the metric
+ rulebook it; drop the index.

### x LMMARG confidence gate (afe5be4): fold only nin>=14 closures;
LMTRACK re-posts stay transient (sigma sign, ABS for weight). Smoke:
118 fold-eligible + 192 transient posts cleanly separated. Focused A/B
fz16.1 running (.15 rooms/corridors/mag, .58 msd) vs fleet16's ungated
fz16 + fzbase cells — must KEEP rooms (room1 -5.2, room5 -6.5) while
killing mag1 bimodal [43/130/137].

### x LMMARG GATE PROGRESSION (the fold-safety problem):
v1 nin>=14: did NOT fix big-space poison (mag2 52->104) — aliased wrong
places have high inliers too. v2 ALIASING-MARGIN (VPR top vs nearest
distinct-place cosine, >0.06): the causal signal; small margin = two
places look alike = fold is a coin-flip. Room6 smoke: 123 folds still
fire in clean space. fz16.2 held-out A/B running (.15 tumvi n=5 on
aliased+winner seqs, .58 euroc+msd). This decides the freeze.

### x REMAINING LIST — built this round, A/B pending container free:
* XR_MULTIHYP (ee15fc9): 4-deep hypothesis ring so the true revisit
  confirms against ANY recent same-place alignment, not just the single
  PENDING_D slot an aliased frame overwrites. Flag-gated, default path
  bit-identical, compiles clean. Targets magistrale2 reloc recall.
* MSD frontend diagnostic (prepped): optical-flow sweep (levels 4, iters
  10, denser grid) on the losing MSD seqs — isolates whether the 3-5x
  raw-VIO drift vs OKVIS2 is KLT tracking on fast headset motion.
Still unbuilt: Cauchy kernel, GravCal probe weighting, right-eye stereo
probing, MASt3R oracle (Gen-6).

### x GATE v2 (aliasing-margin) VERDICT: rooms/corr1/5 wins KEPT (room5
10.9, room1 8.0, corr1 16.6 — all n=5), but aliased spaces STILL regress
(corr3 36 bimodal [5.9,34,36,44,48], mag2 88, mag1 108). THREE gate
designs (ungated / nin / vpr-margin) all fail the same way: closure-class
evidence cannot safely earn PERMANENT estimator info in aliased spaces
with our verification stack. STRUCTURAL conclusion, not parametric.
LMMARG STATUS: validated ROOM-SCALE feature (the biggest quality win of
the program: rooms 5.3->3.9-4.0 held-out, corr1/5 -8 to -16cm), unsafe
as a uniform default. This is the SECOND regime-split feature (after
vkfobs) — the adaptive/profile switch question is now worth two prizes,
OR the product ships LMMARG=on for the room-scale AR profile (the actual
Gen5 use case) — surfaced to the user as a product decision.

### x MULTIHYP A/B: KEEPER (reloc scenario) — uniformly positive on all 4
seqs: mag1 28.9->33.3 (+4.4), mag2 41.1->43.3, corr3 85.6->87.8, corr1
92.2->93.3 with r@10 80->88 and med_err down. Gains concentrate in the
aliased regime it targets; zero cost. Mapping-mode ATE safety needs its
own fleet check before default-on (more confirms = more applied closures).

### x MSD FRONTEND: REFUTED — all four optical-flow variants (levels 4,
iters 10, denser grid, combined) are noise around baseline; MOO01 stays
20-26 vs okvis2-raw 8.9. KLT tracking is NOT the MSD drift cause.
Remaining candidates: IMU noise model (sweep running via new
XR_IMU_NOISE_SCALE env — noises ride the calib feed, not config) and
calibration quality. ofgrid mildly helps easy seqs, hurts MOO13 — not a
keeper.

### x MSD IMU-NOISE: REFUTED (0.5x/2x/4x all noise around base on
MOO01/02/13). With the frontend also refuted, the MSD raw-VIO gap vs
OKVIS2 is NOT config-reachable — remaining suspects are CALIBRATION
quality / photometric conditioning in the MSD pack pipeline (pack-
generation level, not architecture). Diagnostic closed honestly.

## OPTION LIST STATE (end of the exhaustive arc)
Every flag-testable augmentation is built + measured. Keepers in the
deploy stack: v15-final freeze (v14 flags + cap400 + SNAP 0.50 + fixes)
+ reloc stack (TRUSTVPR, BURSTPNP, MULTIHYP pending mapping-safety
fleet) + latency metric. Validated-but-regime-split: LMMARG (rooms
breakthrough; product-profile decision pending), vkfobs (vetoed
uniform). Refuted: INVIDX, sh20, tv065, EDGEGRAPH (parked), MSD
frontend + imu-noise, sigma-strength, LMFACT-transient. Unbuilt tail:
Cauchy kernel, GravCal weighting, right-eye probing, MASt3R oracle.
Then: device day.

### x MULTIHYP SAFETY FLEET: rooms/corr2 clean, mag2 +11cm in mapping —
extra confirms admit closures aliased spaces punish (the arc's recurring
law). SCOPED to LOST frames only (the measured benefit is recovery
recall; healthy tracking keeps strict 2-frame). With that scope MULTIHYP
joins the freeze reloc stack. FINAL freeze reloc stack: TRUSTVPR +
BURSTPNP + MULTIHYP(lost-only) + cap400 + ~55ms latency.

### x STAGE-5 BUILT (item 2, staged): 5a fold-time ARBITRATION — the fold
decision moves from post-time (all 3 gates failed there) to marg-time:
median reprojection residual against the CURRENT state (seconds of
IMU+vision evidence since the closure) must stay <4px; rejected batches
consumed, never retried. + XR_LM_CAUCHY kernel option (okvis2 parity)
+ XR_LMMARG_AUTO scene-scale gate (7d0f129, item 1: median landmark
range EMA <8m earns folds). DECISIVE A/B on .15: s5off/s5arb/s5auto,
poison set + keepers, n=5, 120 runs. Fleet v17 finale suite STAGED on
all three containers (ATE + reloc scenario + drives) — fires on the s5
verdict. Full lmdb-injection (true revived observations) remains the
escalation if arbitration fails.

### x CAUCHY KERNEL: REJECT — rooms identical, corr2 worse (32 vs 22),
no gain anywhere. Huber stays (okvis2-parity curiosity closed).

### x STAGE-5 DECISIVE A/B (.15, n=5, 120 runs) — AUTO-GATE IS THE KEEPER
- s5arb (fold-time arbitration alone): FAIL. corr1 12.85->29.30, mag2
  diverges (all 5 runs unscoreable; runs complete 100% but trajectory
  degenerate). Fold-time residual-vs-current-state cannot catch
  aliased-space poison alone — 4th failed gate of that family.
- s5auto (arb + scene auto-gate <8m): UNIFORM-SAFE. Every seq >= control:
  rooms 10.4/16.6/1.84 vs 13.0/18.2/1.98, corr1 6.28 vs 12.85, corr3
  14.5 vs 15.8, mag2 49.6 vs 65.8, mag1/corr5 ~noise. Aggregate 15.54
  vs 16.96. FIRST composition where LMMARG is safe everywhere.
- Mechanism: corr1 arb 29.3 -> auto 6.3 proves the scene EMA reads
  corridors as big-space and BLOCKS folds there; safety comes from the
  gate, not the arbitration. Arbitration is taxing the rooms prize
  (12 fold-rejections in room1; rooms recovered only 10.4 vs historical
  ungated 6.4).
- FOLLOW-UP LAUNCHED: s5b disambiguation on .15 (160 runs, n=5,
  within-round): s5off / s5auto(4px) / g8(8px) / g999(gate-only, no
  arbitration). Prediction: g999 = full rooms prize + identical
  elsewhere (gate blocks all big-space folds). If confirmed, fz17
  composition adds XR_LMMARG_FOLD_PX=999 (gate does all the work).
- fleet17 DRIVES leg fired on .58 (composition-insensitive: scene gate
  blocks folds at km scale regardless of FOLD_PX). tumvi/euroc legs
  HELD until s5b verdict fixes the composition.

### x FLEET17 DRIVES LEG (.58) — fz17 composition VALIDATED at km scale
- drive1 1.19%/1.20% path (okvis2 3.2-3.3%, ~3x lead holds), drive2
  3.35% (okvis2 DIVERGES both modes), drive3 1.88% (best ours yet;
  okvis2 1.22 keeps drive3). map==vio everywhere -> scene gate blocked
  ALL folds at km scale, exactly as designed. Full stack (LMFACT/
  LMTRACK/LMMARG+AUTO/MULTIHYP) causes zero harm outdoors.
- Drive reloc sweep 0/30, 0/30, 4/30(13.3% drive3): NOT a regression —
  prior pad rounds were eig 0-10%, meg 0-3%. Outdoor single-frame reloc
  stays matching-bound (60m+ ranges kill LighterGlue inliers). Parked.
  fz17d drive3 13.3% is the best outdoor reloc number recorded.
- score_drives.py extended: optional scan dir for <seq>_<arm>_rN_<tr>.tum.
- RELOC17 leg (rl17/rb17, corr1/3/4+mag2, n=3, 24 runs) fired on .58.

### x RELOC17 LEG (.58, n=3) — wake-up reloc at finale composition
- Single-frame: corr1 88.9 / corr3 95.5 / corr4 95.6 / mag2 42.2 mean
  recall (in line with v15 freeze).
- BURST+clip15: corr1/3/4 ALL 100% (r@10cm 87-97%, med err 3-5cm),
  mag2 80.0% mean (was ~52-56% in prior burst rounds) — map-side
  landmark factors (fz17 stack) improve burst verify on the aliased
  hall. Best wake-up numbers recorded; blackout-OKVIS comparison
  pending on .181.
- DRIVES r2/r3 fired on .58 (headline 1.19%-vs-3.3% needs n=3).

### x STAGE-5B GATE DISAMBIGUATION (.15, n=5, 160 runs) — DEFENSE IN DEPTH
- Rooms DOSE-RESPONSE confirmed, monotone in arb threshold: room1
  13.0 -> 9.9(4px) -> 7.9(8px) -> 6.5(no-arb); room5 18.1 -> 16.9 ->
  14.2 -> 11.7. g999 = FULL historical prize (6.4/11 ungated).
- BUT the 8m gate LEAKS where median depth dips under it: corr3 g999
  29.5 vs off 17.9 (poison returns without arbitration). Arbitration
  catches the leakage: g8 best corridors of the round (corr1 5.3,
  corr3 10.8). Gate and arbitration are COMPLEMENTARY, not redundant —
  first-round "safety is all gate" read was too simple.
- Aggregates: g8 12.48 < g999 14.10 < s5off 16.79 < s5auto(4px) 17.96.
- Magistrale UNRESOLVED: deeply bimodal across rounds even for control
  (s5off lost 3/5 mag1 runs in s5 round, 0/5 here; mag1 med flips
  46<->90). g8 lost 3/5 mag2 runs here (n=2) — a within-round harm
  signal that must be settled before freeze.
- S5C FIRED (.15, 175 runs): gate-threshold x arb matrix (s5off/s5auto/
  g8/g8s6/g999s6) on discriminating seqs; xr_map.c gains
  XR_LMMARG_SCENE_M runtime override + scene_ema telemetry (3233768)
  to diagnose leakage directly.

### x STAGE-5C (.15, n=5, 175 runs) + SCENE-EMA TELEMETRY — MECHANISM CORRECTED
- TELEMETRY (decisive): corridors sit at scene_ema 1.9-2.4m ==
  rooms 2.1-2.8m (XFeat landmarks live on the NEAR WALLS; median range
  reads wall distance, not corridor length). mag1 5.7m med (78% <8m!),
  mag2 9.7m med (42% <8m). CONSEQUENCES: (a) the depth gate NEVER
  blocked corridor folds — every "gate saved the corridor" read (incl.
  s5-round corr1 arb-29-vs-auto-6) was corridor bimodal noise between
  BEHAVIORALLY IDENTICAL arms; (b) the gate only dose-reduces
  magistrale (6m: mag1 29%, mag2 22% fold-permission) — hence 6m arms
  kept 5/5 mag2 runs (46.6/49.1 vs off 72.2) where 8m arms lose runs.
- Replicated-across-rounds signals (the only trustworthy ones):
  ROOMS dose-response 3x (off 13/18 -> 4px 10-11/16 -> 8px 8/14 ->
  no-arb 6.5-7.5/10.6-11.7 = full prize); CORR3 needs arb (8px good
  9.1/10.8 in 3 of 4 draws, no-arb poison 29.5/32.4 both rounds);
  MAG2 factor arms >= off both rounds, 6m eliminates run-loss.
  corr1/corr5/mag1: noise (control itself flips 12<->30, loses runs).
- Ideal per-regime: rooms no-arb, corridors 8px, halls blocked. Depth
  can't split rooms|corridors -> MAP EXTENT can (rooms <7m diagonal
  forever, corridors >10m in seconds; huge margins, geometry not
  benchmark fitting). BUILT: XR_LMMARG_ADAPT (39fc341) — room-class
  persist batches ride sigma+1000, basalt stage-6 decodes and skips
  arbitration per batch (patch_stage6.py; stage5.py archived too).
- S5D FIRED (.15, 140 runs): s5off / g8s6 / ad(aptive) / g999s6.
  Prediction: ad = rooms ~g999s6 + corr3 ~g8 + mag2 ~6m arms.
- DRIVES n=3 (.58 done): medians drive1 1.44% (okvis2 3.32, every run
  wins), drive2 3.35 (okvis2 diverges), drive3 1.88 (okvis2 1.22 keeps
  it). Headline lead now variance-backed.

### x S5D VERDICT + CRITICAL BUG: PERSIST SIGN DESTROYED AT THE SETTER
- s5d (140 runs, n=5): ad best aggregate 16.04 (g999s6 16.63, off
  21.85, g8s6 22.84); ad best corr3 11.79 and best magistrale medians
  w/ most runs kept. BUT rooms ad 8.51/13.59 ~= g8s6, NOT g999s6
  6.93/10.87 — room-class fast-path underdelivered.
- ROOT CAUSE (via room-class rejection forensics): the ORIGINAL stage-3
  setter sanitization `slot->sigma_px = sigma_px > 0 ? sigma_px : 2.0f`
  rewrote every NEGATIVE (transient) sigma to +2.0. THE PERSIST SIGN
  NEVER SURVIVED STORAGE — since stage-4 day one, ALL batches (transient
  closures, LMTRACK re-posts) were fold-eligible; the app-side persist
  gates (nin, alias-margin, SCENE DEPTH GATE) never affected folding.
- INVALIDATED (were noise): every gate-vs-gate comparison — s5-round
  "auto uniform-safe", "6m keeps 5/5 mag2", all nin/alias gate verdicts
  of fleet16 era. The magistrale "dose-reduction" story: unfounded.
- STILL VALID (rode the working FOLD_PX axis): rooms dose-response 3x,
  corr3-needs-8px 2x, no-arb corridor poison 2x, Cauchy reject, and all
  s5off-vs-L-arm contrasts (LMFACT genuinely on/off). ad's +1000
  room-class channel also worked (positive, survives the rewrite).
- ACCIDENTAL DISCOVERY: buggy-g999s6's rooms edge (6.93/10.87) = the
  LMTRACK re-posts folding too. Track-factor folding HELPS rooms.
- FIXES (1569d1d): patch_stage7.py preserves the sign (magnitude-only
  sanitize); XR_LMTRACK_PERSIST promotes track-fold to a designed
  feature (scene-gated at call site, class-arbitrated in lmfact_post).
  Stage-7 basalt rebuilt on ALL THREE containers.
- S5E FIRED (.15, 140 runs): s5off / adf / adfl / g99fl — the FIRST
  round where persist gating actually functions. Scene gate's real
  magistrale effect measured for the first time.

### x BLACKOUT-OKVIS VERDICT: DNF — deadlocks at the sensor gap
All 6 runs (MH01/corr1/room1 x lc0/lc1) hang at the exact blackout
point (Progress: 47%/35%, 0% CPU, logs frozen 13h, no trajectory ever
written). The blackout protocol removes camera frames while IMU
continues (the physical model of a sleep/pocket event; blackout.py).
OKVIS2's reference synchronous harness never traverses the gap, so it
never reaches the re-anchor test — scored DNF at wake-up. OURS on the
same protocol: reloc17 = corridors 100% burst recall (3-5cm med err,
~55ms), mag2 80%. Claim wording must say "reference harness deadlocks
at the camera gap", not "cannot relocalize". Processes killed, .181
freed.

### x STAGE-5E (.15, n=5, 140 runs) — FIRST HONEST ROUND FLIPS THE ARB STORY
- g99fl (no-arb + 6m gate + LMTRACK_PERSIST) = BEST AGGREGATE OF THE
  PROGRAM: 11.83 (off 18.03, adf 17.12, adfl 26.24). Per-seq: room1
  6.55 / room5 11.09 (full prize), corr1 5.75 (best), corr3 11.83
  (best; off 22.1), mag2 58.9 with 5/5 runs kept (off 55.3 n=4),
  corr5 22.2 vs off 16.4 (one blemish), mag1 85.7 n=3 (unresolvable).
- LMTRACK_PERSIST hypothesis CONFIRMED: designed track-factor folding
  delivers the rooms prize (adfl 7.7/9.6 vs adf 11.9/14.7).
- The "corr3 no-arb poison" of s5b/s5c was a BUG ARTIFACT: folding
  EVERY transient un-arbitrated = poison; folding only the gated
  persist population un-arbitrated = best-in-round. On the fixed stack
  arbitration may be unnecessary — the candidate FINAL COMPOSITION is
  g99fl: LMFACT+LMTRACK+LMMARG+AUTO, SCENE_M=6, FOLD_PX=999,
  LMTRACK_PERSIST=1 (ADAPT not needed). Fold everything trusted,
  gate by scene depth only. Simple, uniform.
- DECISION RESTS ON: corr10 (.58 n=10: is corr5/corr1 real?) + mag10
  (.181 n=10: halls truly neutral?). Both running, same arms.

### x CORR10 POWER ROUND (.58, n=10, 120 runs) — CORRIDORS ARE ARM-NEUTRAL
corr1: off 26.6 / adf 30.3 / adfl 28.3 / g99fl 28.4. corr3: 15.2/15.2/
18.1/17.2. corr5: 22.8/23.7/19.3/23.1. At n=10 NO arm separates from
control in any corridor — every corridor "win"/"poison" of rounds
s5..s5e (incl. s5e's g99fl corr1 5.75) was bimodal-noise draw. The
factor stack is corridor-NEUTRAL: no harm, no help. Composition is
therefore decided by rooms (replicated low-noise win: g99fl/adfl
6.5-11) + halls (mag10 pending) + held-out (pending). Corridors clear
g99fl of the corr5-blemish suspicion (22.8 off vs 23.1 g99fl, noise).

### > FLEET18 EUROC+MSD LEG STARTED (.58) — base wave first
fzbase control (75 runs: 11 euroc736 + 14 MOO, n=3) fired with FROZEN
binaries (xr_f18_euroc736/xr_f18_msdmoo built once, REUSED by the fz18
final wave). CAVEAT: the two arms run in separate waves on the same
box+binary — acceptable for euroc/MSD (stable seqs; the within-round
law was written for bimodal corridors), ledgered for honesty. fz18
composition = s5e candidate (g99fl); veto path = edit F18 line before
firing final.

### x MAG10 POWER ROUND (.181, n=10, 80 runs) — HALLS CLEAR g99fl
mag1: g99fl 90.0 (8/10 runs kept!) vs off 112.8 (7/10), adfl 104.8
(6/10), adf 111.2 (6/10). mag2: g99fl 64.4 (8/10) ~= off 63.8 (7/10);
adfl COLLAPSES to 3/10 runs. The fixed scene gate makes the factor
stack hall-NEUTRAL-to-BETTER with improved run survival (16/20 vs
14/20); arbitration variants are the fragile ones.
=> COMPOSITION DECIDED pending held-out: fz18 = FB + LMFACT + LMTRACK
+ LMMARG + AUTO + SCENE_M=6 + FOLD_PX=999 + LMTRACK_PERSIST (no ADAPT,
no arbitration; the depth gate is the single safety). Evidence: rooms
full prize (s5e, low-noise, 3x dose-response lineage), corridors
neutral (corr10 n=10), halls neutral-to-better (mag10 n=10).
NOTE: reloc17 recall numbers predate stage-7 (mapping phase ran the
buggy fold-everything stack) — finale re-measures reloc at fz18.

### ✓ HELD-OUT CONFIRMS — fz18 FROZEN (the composition survives EVALUATION §12)
120/120 runs, zero divergences. Sequences tuning never saw: rooms
UNIFORMLY better (room2 3.42 vs 4.87, room3 3.71 vs 5.56, room4 4.17
vs 4.92, room6 1.75 vs 2.00), corr2 18.4 vs 25.1, corr4 neutral,
slides1 24.4 vs 25.8, slides2 33.0 vs 39.5. Aggregate g99fl 11.20 vs
off 11.84 (adfl 10.55 but hall-fragile per mag10 — g99fl stays the
pick: simpler, hall-robust).
FZ18 (FROZEN) = COVKEEP PGO LMDESC TIGHT TIGHTSUB SEQVOTE TRUSTVPR
+ LMFACT LMTRACK LMMARG LMMARG_AUTO SCENE_M=6 FOLD_PX=999
+ LMTRACK_PERSIST (+ reloc stack: cap400, SNAP 0.50, TRUSTVPR,
BURSTPNP, MULTIHYP-lost-only).
TUM-VI TABLE ALREADY COMPLETE from within-round pairs on the stage-7
stack: s5e (room1/5) + heldout (rooms2-4/6, corr2/4, slides) + corr10
n=10 (corr1/3/5) + mag10 n=10 (mag1/2). No tumvi finale rerun needed.
Remaining finale: .58 euroc+msd fz18 wave, .181 bigtum fz18 wave +
reloc re-measure (rl18/rb18). Drives banked at n=3.

### x FLEET18 BIGTUM FINALE (.181) + RELOC18 + CORRIDOR AUTOPSY
- Bigtum ATE (fzbase vs fz18 within-round): slides1 26.3->24.2, slides2
  42.6->36.1, mag2 ~tie, mag1 ours kept 4/5 runs vs control 1/5.
  Freeze story replicates on a third container.
- RELOC18 (wake-up at fz18, n=3): BURST corr1 98.9% / corr3 100% /
  corr4 100% (r@10cm 90-96%), mag2 85.6% (was 80 pre-stage-7);
  single-frame 94.4/86.7/96.7/47.8; med err 3-6cm. Best wake-up
  numbers of the program, now measured at the FROZEN stack.
- CORRIDOR-1 BIMODALITY AUTOPSY (40 within-round runs, corr10): closure
  telemetry IDENTICAL between modes (loops 2-3, same caps, similar
  0.5-0.8m corrections at return). Time-resolved error is FLAT in both
  GT segments: good runs 4-5cm everywhere, bad runs 29-31cm everywhere
  -> the bad mode is a SINGLE RIGID RETURN-REGISTRATION RESIDUAL
  (~60cm split by Umeyama), not drift. Corridor frontier = weld
  PRECISION at re-entry (okvis2's landmark-BA gets 2.8cm); our factor
  stack pulls toward the same PnP-derived 3D, hence arm-neutrality.
  NEXT-ARC candidates: post-weld joint refinement of the re-entry
  against stored landmarks (LOCALBA machinery exists), multi-closure
  fusion at return, subpixel PnP refinement.

### ✓✓ FINALE COMPLETE — fz18 across the entire program (2026-07-18)
EuRoC+MSD wave (.58, within-box waves, frozen binary): fz18 SAFE on all
25 seqs, helps where closures exist: MH01 7.03->4.65, MH02 6.88->6.08,
MH04 8.94->7.37, MOO01 25.9->20.5, rest neutral (MOO13 27.9->32.1
inside its historical spread). Aggregate 6.78->6.08.
PROGRAM SCOREBOARD AT CLOSE (vs okvis2+lc, site canonical):
- TUM-VI long 24.4 vs 31.8 (OURS; their corridors-1-4 fortress stands,
  our robustness wins corr5/mag/slides — they blow up 101/180/160)
- TUM-VI rooms 3.9 vs 1.20 (gap halved by factor stack)
- EuRoC 6.08 vs 3.82; MSD ~6 vs 2.29 (calibration-class, documented)
- 4Seasons drives n=3: 1.44%/3.35%/1.88% vs 3.32/DIVERGED/1.22
- Wake-up: burst 99-100% corridors r@10 90-96%, mag2 85.6%, ~55ms;
  OKVIS2 DNF (gap deadlock all 6 arms)
- fz18 uniform: ZERO regressions on 40 seqs; run-survival improves in
  halls. Site refreshed: 32164 rows, reloc18 tab, 580 trajectories.
NEXT ARC (documented, not started in anger): XR_LOOPBURST — corridor
return-registration via multi-frame joint solve (autopsy: single rigid
~60cm residual; single-frame PnP depth-axis weakness; machinery =
BURST accumulator + pnp2_ransac_burst over the confirm window).

### > XR_LOOPBURST BUILT (67483ba) + DUAL A/B FIRED; SITE CURATED
LOOPBURST: healthy-loop return-registration — accumulate each verified
re-entry frame's assigned (bearing, session-3D) pairs + VIO offsets
(anchor-relative, odom frame; CORR-step invariant), joint-solve
pnp2_ransac_burst over the walked baseline (>=3, target 6 verified
search frames, 2.5s window), apply as capped CORR polish (0.04-1.5m,
<15deg; larger fused delta = structural disagreement, distrust).
Targets the corr autopsy's ~60cm single-frame PnP depth-axis residual.
A/B: fz18 vs fz18+LOOPBURST, corridors 1-5 n=10 + room1/mag2 n=5 —
fired INDEPENDENTLY on .15 (lb_ab) and .58 (lb_ab2): corridor verdicts
count only if both rounds agree.
SITE: curated to two ours arms (fzbase/fz18; s5off/g99fl relabeled —
flag-identical envs), stage-7-era data only (880 rows from 32k), viewer
rewritten (headline tiles fixed, fz18 colors, reloc18 tab, 160 trajs),
drives+wake-up headline card added. Historical arms remain in result
dirs + this ledger only.

### x LOOPBURST v1: HARMFUL — mechanism identified, v2 built
v1 A/B (.15, n=10): corr2 19.3->38.0, corr3 17.2->32.1, corr5
18.3->44.6, room1 6.4->11.4, mag2 56->76; agg 18.3->32.1. Telemetry:
room1 took 17 polish applications PER RUN (5-7cm each) — the
accumulator cycled continuously on revisit stretches, micro-stepping
CORR against the estimator (the documented EuRoC snap failure), and
CORR-stepped inside the TIGHT envelope where the absorbing prior
double-corrects. The multi-frame SOLVE itself was healthy (85-95%
joint-inlier consensus); the APPLICATION POLICY was wrong.
v2 (480c836): one-shot ARMED only by an APPLIED closure, 20s
refractory, 0.10m polish floor, TIGHT-channel arbitration (inside the
envelope the fused D posts as a VIO prior; only beyond it steps CORR).
Dual A/B fired (.15 + .58, 240 runs); third replication on .181.

### x LOOPBURST v1 REPLICATED on .181 (sync error made it a v1 round —
.181 built pre-v2 source; ledgered honestly). Second independent n=10:
corr2 18.9->46.9, corr5 28.4->39.8, room1 7.2->10.5 (18 applications/
run again), corr4/mag2 neutral, corr1 "improvement" 32->26.6 = corridor
noise. v1 harm CONFIRMED cross-container. NEW defect visible in its
logs: 0.46-0.80m corrections APPLIED at 46-59% joint consensus (the
30% gate admits structurally-disagreeing solves — high-consensus
solves run 90-98%). v2 apply gate must also require >=60% consensus.
True v2 verdict pending from .15/.58.

### x LOOPBURST v2 VERDICT (dual n=10, .15+.58): PARKED — no gain, adds variance
v2 policy fixes WORKED (room1 harm gone: 6.7/7.2 and 12.6/12.9 ties;
one-shot + refractory + floor + TIGHT-channel all behaved). But
corridors disagree across rounds in BOTH directions (.15: corr1 17->29,
corr3 25->32 worse; .58: corr3 19->17 better, corr5 15->22 worse);
aggregates +1.2/+2.9. MECHANISM: the one-shot pass runs at the return
where accumulated search frames span too little baseline — every
frame's single-frame PnP shares the same depth-axis bias, the joint
solve reaches 90%+ consensus ON the bias and reproduces it; genuine
disagreement (0.5-0.8m) comes only with collapsed consensus (~50%) =
junk. Short-window fusion cannot observe what single frames cannot.
PARKED: XR_LOOPBURST ships flag-off; fz18 stays the freeze unchanged.
NEXT-ARC candidates for the corridor frontier (documented, unbuilt):
(a) landmark re-triangulation at re-entry (fresh stereo/multi-view
depth on the matched points before solving), (b) full lmdb-injection
(revived observations as optimizable states — the OKVIS2 mechanism),
(c) infra: corridor A/B sharpness is capped by run stochasticity
(fz18 corr1 17/29/32 across rounds at n=10) — deterministic replay
scheduling would multiply every future corridor A/B's power.

### x SITE REGRESSION FIXES (user-flagged) + DRIVES TAB + FULL RELOC GRID
User caught 5 regressions/gaps from the curation pass: (1) baseline
traj overlays dropped (my re-export omitted --baselines) — RESTORED
(195 baseline rows, okvis2/orb3/openvins overlays back); (2) reloc GT
outlines missing (parse_reloc needs <seq>__<arm>_map.tum siblings; my
staging had bare concatenated logs — also WRONG: 3 runs' session
frames mixed) — fixed via grid output naming; (3) only 4 reloc seqs
(old grid had 17) — FULL fz18 reloc grid fired on all 3 boxes (rl18+
rb18 x rooms1-6, slides1/2, corridors1-5, mag1/2, MH_01, MOO07);
(4) baseline relocs — OKVIS2 DNF note card added to Reloc tab
(deadlock evidence; OpenVINS no-reloc-by-design; ORB3 not benchmarked,
noted not claimed); (5) drives absent — now a REAL dataset group:
%%-of-path bars (drive1 1.44 vs 3.32; drive2 okvis2 bar absent =
diverged; aggregate ours 1.88 < okvis2+lc 2.24), okvis2 baseline rows
(.final.tum, generous to them), full trajectories (fz17d + okvis2 lc0/
lc1 + GT). export_site_data: drives GT root/fps 15/diverge 1000m/
path_m rows; app.js: drives group + %%path conversion + fz17d arm.

### x FULL fz18 RELOC GRID (17 seqs, 34 cells) — ON SITE with spatial plots
Mean recall: burst-15 95.1%, single-frame 87.5% (17 seqs). Per-seq
burst: rooms 100% all six, corridors 96.7-100%, MH_01 100%, MOO07
100%, slides 86.7/90.0, mag1 70.0/mag2 73.3 (r@25 53.3 both — the
aliased-hall wall as always). Reloc tab: DNF-baseline card + recall/
r@25/med-err grids + "where relocalization landed" spatial panel with
session-trajectory outlines (parse_reloc map.tum siblings) and
expected->landed vectors, all 17 seqs. Site deliverable COMPLETE.

### x PER-SEQUENCE ANALYSIS REPORT (analysis.html) — user-directed honesty pass
Median framing retired. Head-to-head: rooms 0W6L, euroc 2W8L1T, msd
3W8L3T(+1 DNF-win), long 4W4L1T, drives 2W1L. THREE-WAY DECOMPOSITION
(oursVIO/oursMAP/okvis-lc0/lc1 per seq) yields four measured classes:
- ABSORB (22 seqs, dominant): their closure ABSORPTION converts VIO->
  final at 1.7-31.9x (corr1 31.9x!); ours 1.0-1.2x on most (best 3.8x).
  Detection is NOT the gap (our recall 95%); absorption is. Root =
  reversible marginalization: OKVIS2 pose-graph edges cache common
  observations and revert to landmarks+observations at closure + async
  loop optimization (Leutenegger 2202.09199); Basalt sqrt prior is a
  one-way sink and our folds are irreversible by construction.
- TRACK (9 seqs): RTE outliers 1.3-2.5x (MOO01/02 1.9x, MOO13 2.5x,
  corridors, room1/5, V2_03) = KLT shear under blur/low texture; we WIN
  RTE on 24/39 elsewhere (down to 0.39x). KLT vs redetect-redescribe.
- DETECT (5 seqs): loops=0 on MH_02/MH_05/V1_03/V2_02/V2_03 while
  their covisibility matching gains up to 2.1x. Event-retrieval vs
  continuous covisibility detection.
- ROBUST (8 seqs): their LC self-harms (slides1 0.27x = 8.6->31.8!,
  V2_01 0.73x); our discipline never <0.91x in ~2000 runs. The moat.
RETIRED CLAIM: "MSD gap is calibration-level" — falsified by per-seq
split (5 MSD seqs at parity/win; losses split TRACK/ABSORB cleanly).
MH_01 92.7cm outlier run = early-map wrong-place closure (gate TODO).
ROADMAP RANKED: R1 reversible-marg/landmark reactivation (DM-VIO
delayed-marg or OKVIS2 edge<->obs cache, behind our gates) — 22 seqs;
R2 XFeat redetect fallback on KLT collapse — 9 seqs; R3 covisibility
closure path — 5 seqs; R4 deterministic replay infra; R5 early-map
gate; R6 far-matching for drive3. Report: site /analysis.html.

### x QUICK-LEVER VERDICTS + STAGE-8 BUILT + .58 CONTAMINATION FOUND
- SIGMA (post-stage-7 redo of the invalidated sweep, .15 n=5): REAL
  absorption lever. ALL SIX rooms improve at XR_LMF_SIGMA=1.0 (room1
  7.02->5.09, room6 1.72->1.52, agg 5.44->4.36); corr1 28.8->18.6-21.1.
  sg10 = fz19 candidate; in the s8 2x2 for confirmation.
- XR_SEED (dormant v9 detector unification, .58 n=5): mild TRACK-class
  gains (MOO13 35->31.6, MOO01 22->20, V2_02 6.6->5.7), neutral rest.
  Minor candidate; seeds still KLT-shear — not the blur answer.
- STAGE-8 XR_LMINJ ALIVE after a five-bug odyssey (patches 8..8e):
  (1) 1-obs landmark segfaults QR (needs >=2 rows) -> buffered creation;
  (2) newest-kf host not guaranteed in AbsOrderMap -> host like flow
  landmarks (current frame on kf measures); (3) take_kf is a member
  CONSUMED by its own branch -> kf_ids.count(now); (4) app re-match
  stream (1-2Hz) vs 150ms frame lifetime -> PER-FRAME COVISIBILITY
  MATCHER inside the estimator (project anchors, match flow kps 8px,
  sub-pixel agreement measured: 0.12-0.23px!); (5) lost-landmark test
  is flow connectivity -> injected ids exempt, retire with host kf.
  Smoke: 23 landmarks created, +23 obs streaming EVERY frame, clean
  full run. The OKVIS2 absorption mechanism now runs inside Basalt.
  2x2 A/B FIRED (.15, 220 runs): fz18/sg10/inj/injsg.
- .58 BASALT CONTAMINATION: header at stage-7 but xr_alive MISSING
  (stage-4 keyframe-anchored survival lost at some point) -> folds
  never fired on .58. CONTAMINATED: corr10 corridor-neutrality verdict,
  lbv2 round-2, fleet18 euroc/msd fz18 wave (understates factor stack;
  MH01 4.65 was achieved WITHOUT folds), cauchy (pre-7 anyway), seed
  round (within-round delta still valid). CLEAN: all .15 rounds, all
  .181 rounds (tree-synced; xr_alive=2 verified). FIX: full tree-sync
  .15->.58 (stage-8 level), rebuilt, euroc fz18 wave RE-FIRED. corr10
  re-run queued post-s8. All three containers now at identical
  stage-8 trees.

### x STAGE-8 2x2 VERDICT (.15, 220 runs): INJECTION WORKS, COMPOSES WITH SIGMA
Rooms (low-noise absorption class): inj improves ALL SIX over fz18
(room1 7.39->6.69, room2 3.40->2.97, room3 3.62->3.23, room4 4.19->
3.67, room6 1.75->1.69); rooms-class median fz18 3.91 -> inj 3.45 ->
sg10/injsg 3.36. CORR3 HALVED: 24.8 -> inj 12.5 (okvis2 4.9 — the
absorption gap finally moves on a corridor). Safety: mag2 59->50,
slides2 41->37 (both BETTER under inj). corr1 = bimodal noise (fz18
drew 6.35; sigma arms drew 26-32; lockstep infra will settle it).
11-seq aggregate misleads (corr1-dominated) — class medians are the
read. v1 injection captures a SLICE of the okvis2 mechanism (2-obs
landmarks, no old-obs revival, no loop re-opt) — modest but uniform
and safe. FZ19 CANDIDATE = fz18 + XR_LMF_SIGMA=1.0 + XR_LMINJ=1.
### x FIXED-.58 EUROC/MSD RE-RUN: the contamination was masking real gains
fz18(fixed) vs fzbase within-round: agg 6.78->5.17 (fold-less had
6.08). MOO02 23.7->12.6 (was 24.0!), MOO01 25.9->17.4, MOO14 8.5->4.8,
MOO06 5.3->3.5, MOO13 27.9->23.4, MOO15 27.7->19.5, V2_03 21.1->16.6,
MH_01 7.0->5.4, MH_04 8.9->6.8; V2_01 3.5->4.1 lone small regression.
Site refreshed with corrected cells. vs okvis2+lc: euroc 5.17 vs 3.82,
MSD gaps narrow substantially (MOO02 5.5x from 10.5x).
FZ19 VALIDATION FIRED: .58 euroc+msd fz19 wave; .15 TUM held-out
(corr4/5, mag1, slides1) fz18-vs-fz19.

### x FZ19 VALIDATION: TUM PASSES, EUROC SPLITS — decomposition fired
- TUM held-out (.15, n=5): fz19 > fz18 (corr5 27.2->21.4, mag1 102->89
  n2->n3, corr4/slides1 tie; agg 25.8->23.2). PASS.
- euroc/MSD (.58, n=3): agg 5.17->5.49; V2_03 16.6->22.6 (the
  TRACK-class seq — fast motion; hypothesis: 8px covis matcher attaches
  stale-3D observations under blur, or sigma-1.0 overweights factors
  vs a blurred frontend), V1_03 6.0->6.8, MOO15 19.5->26 (n=2 noisy),
  most others tie; MSD strong cells unchanged.
- NO FREEZE CALL until the euroc regression is attributed: injo (LMINJ
  only) + sgo (SIGMA only) waves fired on .58 (150 runs, same frozen
  binary). Uniform-composition principle holds: fz19 ships only if the
  regression is fixed or attributed to a gateable condition.

### ✓ ATTRIBUTION + FREEZE fz19 = fz18 + XR_LMINJ (sigma REJECTED uniform)
Decomposition (.58, 4 arms x 25 seqs, frozen binary): V2_03 fz18 16.6 /
injo 15.4 (BETTER) / sgo 24.3 (CULPRIT) / fz19(comp) 22.6. MOO13: injo
24.5 vs sgo 31.3. sigma-1.0's strong fixed-3D pull poisons the
TRACK-class (blurred frontend + overweighted stale 3D) — helps only
sharp-fisheye TUM. REJECTED as uniform lever (sigma stays 2.0).
INJECTION alone: TUM rooms all improve + corr3 halved + corr5/mag1/
mag2/slides2 better (2x2 + held-out, n=5) AND euroc/MSD neutral-to-
positive (V2_03 15.4, V2_01 3.57, MOO01 13.6; regressions all <=0.7cm
n=3 noise-level). UNIFORM-SAFE => FREEZE fz19 := fz18 + XR_LMINJ=1.
(Ledger note: "fz19" earlier meant fz18+sigma+inj — REDEFINED here to
the injection-only composition; the sigma composite is retired.)
corr10-v2 fired on fixed .58 (fz18 vs fz19, corridors n=10) — the
re-validation owed since the contamination find.

### ✓ CORR10-V2 (fixed .58, n=10): fz19 CONFIRMED — corr3 30.9->11.9
Independent-box n=10 replication of the injection corr3 win (2x2 had
24.8->12.5). First corridor decisively moved in the program: corr3 now
2.5x behind okvis2+lc (was 6.4x). corr1 fortress stands (32.5~33.6,
bimodal). corr2/4/5 neutral-to-better; agg 19.8->17.8. fz19 freeze
stands with corridor n=10 backing. SITE PUBLISHED: rooms strip now
okvis2+lc 1.20 < fz19 3.45 < okvis2(no LC) 3.89 — we beat their
VIO-only mode on rooms; TUM-long all three ours arms lead their 31.8.
Outstanding: lockstep determinism verdict (.181, slow by design).

### x USER TRAJECTORY ANALYSIS + EVENT FORENSICS => XR_DEFGUARD (fz20 cand)
USER (visual, per-seq): fz18 map track FOLLOWS factor-coupled VIO on
nearly every euroc/rooms/MSD seq — results there are VIO-driven; the
map layer's rare activations are high-RISK events (MH_01-fz19 ~80cm
jump; MH_05-fz18 total failure; MH_05-fz19 phantom 90-deg turn at end;
MOO15-fz19 fails). V2_03-fz18 = the ONE euroc case where reloc helps.
corridor1: closures only at the return (pathing = VIO elsewhere);
corridor4: NO system closes (ours or theirs). slides1: okvis2-VIO 8.6
beats everything incl their own +LC.
FORENSICS (event<->trajectory correlation, all flagged runs):
- MH_01-fz19 r2: ESCALATING deform cascade 1.28->1.43->1.80->2.94m,
  seeded by an early covis-3 closure (13/29 inl) whose inliers the
  UNGATED injection made persistent landmarks. r3 (no deforms) clean.
- MH_05-fz18 r1: deform cascade incl a 9/27-inlier (33%) closure.
- MOO15-fz19 r1: 7.97m deform ADMITTED on a ~10m scan (36/42 inl,
  aliased strong match) — magnitude absurd vs map extent.
- V2_03-fz18 (the good case): 94%/67%-ratio closures — the profile a
  quality floor must PRESERVE.
- MH_05 t+79.4s and V2_03 t+36.3s: DETERMINISTIC micro-jump showers
  (~0.12m x 50+, all arms/runs) = discrete VIO-core events (exposure/
  bias transient?) — named VIO-core investigation items.
FIXES (9b7231a): XR_DEFGUARD — deform needs inlier-ratio>=55%,
covis>=8, dev<=max(1.5, 0.6*map_extent), anti-escalation (bigger
deform within 45s needs ratio>=75%); rejected deforms DOWNGRADE to
TIGHT-prior soft absorption (never move the map on weak evidence).
+ INJECTION now requires the fold-grade gates (nin>=MIN, alias
margin) at both closure sites — a weak closure can no longer seed
persistent poison landmarks. fz20 = fz19 + DEFGUARD; A/B fired on the
EVENT SET (MH_01/05, V2_03, MOO13/15 + room1/corr3/mag2 sanity, n=5).

### x MICROCORR PARKED + THE CONTINUOUS-ABSORPTION TRUTH (user-driven)
MEASURED (offset-profile analysis, ledgered): okvis2+lc applies 4-15m
CUMULATIVE micro-adjustment per run over 1300-4100 frames (>1mm each,
top-5 frames = 2-5% of total — no steps); OUR map track is
BIT-IDENTICAL to factor-coupled VIO outside deforms (totalAdj 0.00m).
User's "VIO offset" read exact. MICROCORR v1 (output-servo toward
sub-gate D, <=2cm/event): UNIFORMLY HARMFUL (rooms 6.8->9.7, corr3
18->32) — chases single-frame PnP noise AND double-counts against
in-estimator factor absorption (LOOPBURST v1's lesson again). PARKED.
CORRECTED MODEL: our zero output-offset is BECAUSE factors correct
inside the odom; their continuous lc0-vs-lc1 divergence = in-estimator
+ in-MAP corrections (their keyframe graph stays loop-consistent
continuously; ours heals only at deforms). NEXT-ARC: continuous map
self-consistency — rate-limited background micro-PGO using sub-gate
verifications as edges to OLD keyframes (the valid edge class; NOT
EDGEGRAPH's self-drift-correlated edges). fz20-v2 (two-strikes
escalation) subset re-test fired (.58: corr3/MH_01/mag2/MOO15).

### x DEFGUARD ITERATIONS v2-v4 (event-set A/B, .58, n=5 each round)
- v2 (two-strikes): MH_01 4.71 (BETTER than unguarded 5.23 — cascade
  killed), mag2 42.9 5/5 kept, V2_03/MH_05/MOO13 preserved; corr3
  STILL 35 (blocked by covis>=8: its productive returns run covis 5-7).
- v3 (scene-conditional covis + alias-margin): corr3 STILL blocked
  (alias margin — corridor ends look alike) AND mag2 win LOST (relaxed
  corridor-class floor readmitted hall poison during scene_ok
  stretches). The covis/alias dimensions SEE-SAW corr3-vs-mag2 —
  static per-closure gates cannot separate them.
- v4 (minimal: ratio-55 + extent-cap + two-strikes only): every
  forensic cascade dies by one of the three; corr3's 58-80%-ratio
  non-escalating returns pass all. FINAL TIMEBOXED ROUND — if not
  uniform-safe, fz19 remains the freeze and DEFGUARD ships flag-off
  (tail-risk tool: the cascades are 1-2-in-10-run events that medians
  barely see; MH_01-fz19 scored 5.23 unguarded this round).

### ✓ DEFGUARD FINAL: flag-off tail-risk tool; fz19 REMAINS THE FREEZE
v4 (minimal): MH_05 16.6 (best yet), mag2/MOO13/MOO15/V2_03 fine,
corr3 26.4 (still pays vs 17.1), MH_01 n=3. Four rounds establish:
static output-layer deform gates cannot separate corridor CONVERGENCE
from hall CASCADES without collateral — the phenomena overlap in every
per-closure observable we have (ratio, covis, alias, magnitude,
escalation shape). DEFGUARD ships FLAG-OFF: valuable against the
1-2-in-10 cascade tail (MH_01-class), not as default. The structural
fix for both remains recoverable in-estimator absorption (R1 arc:
injection deepening + delayed-marg) + continuous map self-consistency.
FREEZE UNCHANGED: fz19 = fz18 + XR_LMINJ.

### x FZ19 RELOC GRID (15/17 seqs in; mag pending): INJECTION IMPROVES WAKE-UP
Burst-15 mean 98.2% (fz18-era grid 95.1): corridors 93-100 ALL, rooms
100 x6, MH_01/MOO07 100, slides 83/97. Single-frame 92.9 (was 87.5).
Verdict: injection's persistent landmarks IMPROVE recall (better map
metric quality at revisit). slides single-frame dipped (67 vs 87) —
burst covers it; noted. Lockstep retry (6h timeout) grinding on .181.

### ✓ FZ19 RELOC GRID COMPLETE (17 seqs): burst 95.7% mean, single 86.5%
mag1 26.7/66.7 (dipped vs fz18-era 40/70), mag2 50.0/86.7 (improved) —
hall noise; corridors/rooms/slides/MH/MOO as ledgered (93-100 burst).
Wake-up at the frozen composition CONFIRMED; site reloc tab = fz19.
Both open verdicts from the state report now closed except lockstep
(grinding, 6h budget). NEXT ARC ON DECK: continuous map
self-consistency (background micro-PGO, old-keyframe edges only).

### x DBOW2 RETRIEVAL BENCHMARK (user request) — the wake-up moat holds
Harness (bench/tools/dbow2_bench.cpp): ORB-SLAM3's DBoW2 + ORBvoc +
cv::ORB(1000) on OUR exact rl19 probe frames, db = stride-12 kfs capped
400, +-3s temporal exclusion (place recognition, not moment matching),
reference = fz19 trajectory. First run WITHOUT exclusion scored 100%
everywhere = temporal self-match confound (caught before quoting).
Corrected numbers — DBoW2 retrieval recall@1/@5 @3m vs OUR END-TO-END
VERIFIED single-frame (rl19) / burst (rb19):
  corridor1  57/70  vs  93/100   corridor2  80/87  vs  87/93
  corridor3  37/60  vs  90/100   corridor4  80/87  vs  90/100
  corridor5  47/63  vs  97/100   magistrale2 70/83 vs  50/87
VERDICT: DBoW2's retrieval CEILING (verification can only remove hits)
sits far below our end-to-end verified recall on corridors 1/3/5;
comparable corr2/4; mag2 retrieval 70% > our 50% single-frame verified
(caveat: unverified retrieval in an aliased hall — many hits would be
rejected or wrong-accepted; our burst 87% exceeds their ceiling).
CAVEATS: cv::ORB vs ORB3's octree extractor (slightly better spread);
retrieval-only vs end-to-end favors DBoW2 in this table. room1 cell
absent (probes parse; minor, not re-run).
