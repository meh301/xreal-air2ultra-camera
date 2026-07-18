# Gen 5 Deployment Plan — FULL pipeline on Snapdragon 8 Elite Gen 5

Cost evaluation and device-day validation plan for shipping the FULL
configuration (dense-XFeat + MegaLoc + LighterGlue-L6 + COVKEEP + PGO +
LMDESC + TIGHT/TIGHTSUB) plus the depth pipeline to the Xperia 1 VIII
(SM8850, Hexagon V81 HTP, fp16-native HMX, larger VTCM,
`qnn_context_priority` preemption).

**Status: device IN HAND (2026-07-17). Deliberate strategy: finish the
pipeline improvement arc on the bench first; push to device once the
config freezes.** Companion docs: `bench/EVALUATION.md` (scoring rules),
`bench/ITERATIONS.md` (verdicts).

Confidence marks: **[M]** measured (SD888 HTP / bench container),
**[P]** projected (888→Gen5 median scaling ×3.55, measured across model
classes; V79 AI Hub anchors per the verified VPR research). Every [P]
becomes [M] during device-day validation (§5).

---

## 1. Architectural cost shape

The AR frame loop (camera → Basalt VIO → render, 30/60 fps) inherits
ONE new cost from FULL: TIGHT's pose priors — a few float terms inside
an optimization Basalt already runs. Everything else executes on the
map thread at keyframe/search cadence (~0.5-2 Hz) as duty-cycled NPU
bursts, or on event (closures/welds). The universal duty cap
(bench-validated) bounds map-thread cost at ~25% of its own cadence.

## 2. Mapping stack budget

| component | trigger | SD888 [M] | Gen 5 [P] |
|---|---|---|---|
| XFeat dense A8W8 | search/store pass | 3.6 ms | 1-1.5 ms |
| dense C tail + anchors | same pass | ~3 ms CPU | ~2 ms CPU |
| **EigenPlaces-512 embed (DEPLOY PICK, 2026-07-17)** | same pass | 16.6 ms [M] | **~5 ms** |
| MegaLoc embed fp16 (large-map profile only) | same pass | — | 18-35 ms |
| LighterGlue L6 (4.6 MB) | ≤4 calls/search (budgeted) | — | 1-3 ms/call |
| retrieval dots 200×8448 | per search | <1 ms | <1 ms |
| COVKEEP/PGO/weld/bank | event | µs-ms | µs-ms |
| LMDESC direct fallback | LOST-only, duty-capped | 20-40 ms CPU | ~10 ms |
| TIGHT/TIGHTSUB priors | per confirmed closure | ~0 | ~0 |

**One search+store pass ≈ 25-45 ms at 0.5-1 Hz → 2-4% NPU duty,
~50-100 mW average.** (Reference: the measured EVA tracking service runs
385 mW continuously.)

## 3. Depth budget — the largest NPU consumer

| config | SD888 [M] | Gen 5 [P] | note |
|---|---|---|---|
| LAS2 A16W8 @192 (shipped) | 32 ms HVX | 9-12 ms | d1 0.819, corr .987 |
| LAS2 A8W8 @192 | 13 ms HMX | ~4 ms | **unusable — A8 quant broke depth quality** |
| LAS2 fp16 @192 | n/a (V68 int-only) | **5-8 ms** | V81 fp16 HMX: HMX speed w/o quant damage — expected strict win, validate FIRST |
| LAS2 fp16 @256-288 | — | 12-20 ms | ~1.4× finer disparity |
| LAS2 fp16 @384 | — | 25-55 ms | near-native; 2× better far-field depth (err ∝ Z²/fB); needs V81 VTCM headroom |

Stereo cost scales ~4-8× per resolution doubling (cost volume =
pixels × disparity range; range grows with width). V68's 4 MB VTCM
forced tiling at 192; V81's larger scratchpad is the enabler for 288+.

**Two-tier depth architecture (recommended):**
1. Base: fp16 @192 at 10-30 Hz (occlusion continuity) — 6-15% duty.
2. Keyframe tier: @384 at 1-2 Hz aligned with map keyframes (meshing,
   stereo-verified landmarks via the dormant depth-calibration plumbing)
   — 3-8% duty.
3. Guided/joint-bilateral upsampling of tier-1 against full-res gray
   (1-2 ms GPU/CPU) — recovers most of the perceived 192→384 edge
   quality for free.

Total depth: **~10-20% duty, ~0.2-0.5 W.** EVA offload option: if the
Qualcomm EVA SDK entitlement lands, DFS does 720p60 <8 ms @385 mW [M]
on the dedicated block — the whole depth line leaves the HTP.

## 4. Totals & memory

| consumer | NPU duty | avg power |
|---|---|---|
| depth (two-tier) | 10-20% | 0.2-0.5 W |
| FULL mapping stack | 2-4% | 0.05-0.1 W |
| AR perception extras (hands 1.5 ms, det 0.9 ms, 6DoF 2-3 ms, seg 2 ms — feasibility study) | 5-10% | 0.1-0.2 W |
| **total** | **≤~35%** | **~0.4-0.8 W** |

**AR verdict: comfortable.** ≥65% NPU headroom for applications, an
untouched frame loop, and V81 priority preemption keeps latency-critical
models ahead of MegaLoc's 25 ms bursts. What FULL buys the AR layer:
78% mean cold-reloc at cm precision, submap self-healing (post-weld
15 cm vs VIO 72 cm after 15 s blindness), drift ≤ VIO everywhere,
TUM-long beyond OKVIS2+LC.

Memory (user budget: <1 GB is fine → no pressure):

| item | size |
|---|---|
| EigenPlaces resident (deploy) | **~45 MB** |
| MegaLoc fp16 (large-map profile, optional) | ~457 MB |
| XFeat + LG6 + LAS2 models | ~9 MB |
| session map @200 kf | 11 MB (float emb) |
| ORT/QNN arenas | ~50-100 MB |
| **total** | **~120-180 MB** (deploy) / +457 MB with the MegaLoc profile ✓ |

int8 stored embeddings (planned) are for map CAPACITY, not fit:
6 MB @200 kf → 29 MB @1000 kf (the eviction-horizon fix for
building-scale sessions; drives needed 2000).

## 5. Device-day validation order

1. **LAS2 fp16 @192 vs shipped A16W8** — expect strict win (speed AND
   quality); becomes the new depth base.
2. XFeat dense fp16/A8 on V81 + dense-tail timings; anchors on-device.
3. MegaLoc fp16 bring-up (322², W8 projection) — the big [P]→[M];
   validate the 18-35 ms window and DDR behavior.
4. LighterGlue L6 static-shape fp16 port (exists as ONNX; low risk).
5. VTCM ceiling: LAS2 @288/@384 feasibility.
6. `qnn_context_priority` replaces the `XR_NPU_GATE` mutex (V68-only
   workaround); per-run clock votes → burst/balanced (the 888
   overheating lesson — never `sustained_high` for periodic loads).
7. Whole-stack soak: depth two-tier + FULL map + VIO, thermals + power
   rails, then the on-device kidnap walk (the original field-failure
   scenario, now with submaps).

## 6. Pre-push gate (pipeline freeze criteria)

Ship to device when: (a) the augmentation A/B verdicts land (SEQVOTE /
DESPERATE / ROTSTORE), (b) blackout-reloc baselines published, (b2) fleet v14 'full' (EigenPlaces+SEQVOTE) confirms the retrieval flip at fleet scale, (c) the
stage-3 landmark-factor decision is made (build vs defer), and (d) the
harness exit-hang fix lands (also affects on-device shutdown paths).
Freeze = a fleet-validated env set baked into the APK defaults.

### 2026-07-17 freeze verdict (fleet v14, 480 runs)

Gates (a), (b2), (c) are CLOSED. The frozen config is
**dense XFeat + LighterGlue-L6 + descriptor union + tight coupling +
TIGHTSUB + EigenPlaces retrieval + SEQVOTE + SNAP_MIN 0.50**
(`XR_COVKEEP XR_PGO XR_LMDESC XR_TIGHT XR_TIGHTSUB XR_SEQVOTE`, EigenPlaces
model; SNAP_MIN 0.50 is now the compiled default).

All-40-seq map median: **freeze 7.64 cm** vs freezem (MegaLoc) 8.03,
xdlg6 10.71, bad 10.27. Per group (median over per-seq medians):

| group | bad | freeze | freezem | verdict |
|---|---|---|---|---|
| EuRoC (11) | 8.13 | **6.62** | 6.72 | behind OKVIS2+LC (3.82); VIO-bound |
| TUM-VI long (9) | 30.21 | **23.21** | 41.39 | BEATS OKVIS2+LC (31.8) by 8.6 cm; retrieval flip decisive (EIG ~2x MegaLoc) |
| rooms (6) | 5.22 | 5.22 | 5.24 | gated = VIO; OKVIS2+LC gap (~1.2 cm) stands, LMFACT rejected |
| MSD (14) | 6.22 | 6.15 | 6.13 | BEHIND OKVIS2+LC (2.29) and OKVIS2 (5.85) — map==VIO gated; the azall ladder is the live attack (CORRECTED 07-17: earlier "lead held" compared only OpenVINS/ORB3) |

TIGHTSUB STAYS: v14 freeze long 23.21 beats the v12 no-TIGHTSUB 27.04
(v13's harm signal was a MegaLoc-context artifact). Stage-3 LMFACT:
REJECTED for freeze (rooms unmoved, corridor harm — see ITERATIONS.md).
Remaining before push: (b) blackout baselines (running on bench-2) and
(d) the exit-hang fix.

### 2026-07-18 FREEZE v15-final (supersedes the 07-17 block)

APK default set: the v14 flag set (XR_COVKEEP XR_PGO XR_LMDESC XR_TIGHT
XR_TIGHTSUB XR_SEQVOTE, EigenPlaces retrieval) + XR_MAP_MAX_KF 400
(now the compiled default) + SNAP_MIN 0.50 + XR_TRUSTVPR and the
BURSTPNP wake-up path for relocalization + the five pipeline fixes.
vkfobs VIO tuning was VETOED by the held-out fleet (subset mirage:
corridor1 +18, magistrale2 +21). LMFACT / LMTRACK / LOCALBA / FARBEAR /
EDGEGRAPH ship as flags, OFF. Scoreboard at freeze: TUM-VI long and all
three 4Seasons drives lead OKVIS2+LC (drive1 1.11%% vs 3.3%%, okvis2
diverges drive2); EuRoC/rooms/MSD remain OKVIS2+LC's (3.82/1.20/2.29 vs
our 6.5/4.9-5.5/6.1) — the two documented frontiers (marg-persistent
factors, MSD frontend) are the path there. Wake-up reloc: burst-15 +
TrustVPR, corridors 50-73%% recall, rooms 100%%.

### 2026-07-18 FREEZE v16-final "fz18" (supersedes v15-final)

APK default set = v15-final reloc stack (cap400, SNAP_MIN 0.50,
TRUSTVPR, BURSTPNP, MULTIHYP lost-only, EigenPlaces, five pipeline
fixes) PLUS the map->VIO factor stack, now UNIFORM-ON: XR_LMFACT,
XR_LMTRACK, XR_LMMARG, XR_LMMARG_AUTO (scene gate, XR_LMMARG_SCENE_M=6),
XR_LMMARG_FOLD_PX=999 (no fold arbitration), XR_LMTRACK_PERSIST=1
(track factors fold too). No ADAPT, no room/corridor switching — the
depth gate is the single safety and it is REGIME-detection, not
closure-quality gating (four closure-quality gate families failed;
see ITERATIONS stage-5..7 arc). Requires the stage-7 basalt fix
(persist sign preserved at setXrLandmarkFactors — patch_stage7.py);
without it the flags silently degrade to fold-everything.

Held-out gate (EVALUATION 12): PASSED on 8 unseen seqs, zero
regressions, zero divergences (rooms all better, corr2 better, corr4
neutral, slides both better).

Scoreboard at freeze (within-round map medians, stage-7 stack):
- TUM-VI long: OURS 24.4 vs okvis2+lc 31.8 (orb3 82, openvins 55).
  We lead corr5/mag1/mag2/slides1/slides2 (their blowups: 101/180/160);
  okvis2+lc leads corridors 1-4 (2.8-11.4 vs our 17-28; corridor floor
  is arm-neutral tracking noise, the documented remaining frontier).
- TUM-VI rooms: OURS 3.9 vs okvis2+lc 1.20 (was 5.3 pre-LMMARG; gap
  halved by the factor stack; remainder is calibration-class).
- 4Seasons drives n=3: drive1 1.44%% vs 3.32%% (every run beats them),
  drive2 3.35%% vs DIVERGED, drive3 1.88%% vs 1.22%%.
- Wake-up: burst-15 corridors 100%% recall (med err 3-5 cm, ~55 ms);
  mag2 80%%. OKVIS2: DNF (reference harness deadlocks at the camera
  gap, all 6 arms). rl18/rb18 re-measure at fz18 in flight.

### 2026-07-18 FREEZE v17 "fz19" (supersedes v16-final)

fz19 = fz18 + **XR_LMINJ=1** (stage-8 closure-landmark injection:
verified closure inliers become real basalt lmdb landmarks with
per-frame covisibility observations inside the estimator — patches
8..8e; requires the stage-8 basalt build). Sigma stays 2.0 (the 1.0
variant poisons blur sequences — TRACK-class attribution in ledger).
Evidence: rooms all improve (2x2 n=5), corr3 24.8->12.5, corr5/mag1/
mag2/slides2 better (held-out n=5), euroc/MSD neutral-to-positive
(V2_03 15.4, MOO01 13.6, n=3). Uniform-safe: no regression beyond
noise anywhere; corridors n=10 re-validation in flight (corr10-v2).
NOTE: the .58 contamination episode (stage-4 alive-check missing)
invalidated the earlier corr10/euroc fleet cells — corrected numbers
are on the site; euroc aggregate is 5.17 (fzbase 6.78), MSD MOO02 12.6.
