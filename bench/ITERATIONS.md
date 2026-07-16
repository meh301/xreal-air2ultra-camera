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
