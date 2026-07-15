# BENCHMARK PROGRAM v2 — Stereo-VIO + Session-Map + VPR, Causal-First (July 2026)

Protocol constants (locked, all with citable precedent): causal = pose of each frame at its first availability, per OKVIS2 §VII terminology ([arXiv:2202.09199](https://arxiv.org/abs/2202.09199)); ATE = SE(3)-Umeyama RMSE (evo), secondary 4-DoF posyaw + Sim(3) scale check per [Zhang & Scaramuzza IROS'18](https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf); RTE Δ=6 frames + RTE@{1m,10m,30s}; divergence = ∞ at >10 m ATE / >10 cm RTE (MSD/VOCA rule, [arXiv:2508.00088](https://arxiv.org/abs/2508.00088)); median of 5 runs, report all; pose coverage ≥99% or flagged; identical params across all sequences of a class; every table footer states alignment, causality, #runs, RTF, coverage. Both tracks (causal VIO-only + causal map-corrected) from our harness on every row; competitors' paper numbers always labeled NON-CAUSAL where applicable.

---

## 1. COMPARISON MATRIX

### 1a. We-run-them columns (same harness, same machine, causal, 5-run median) — MEASURED once run

| # | System | Mode | LC arm | Causal-extraction plan | Effort | Datasets to run |
|---|---|---|---|---|---|---|
| 0 | **Ours: patched Basalt + xr_map** | stereo-VI | VIO-only / +LC / +VPR (3 arms) | native (harness emits both tracks) | 0 | ALL |
| 1 | **Stock Basalt (mateosss/Monado)** | stereo-VI | none (upstream has no LC) | native VIT interface; already built | ~0 h | ALL — the non-negotiable ablation control |
| 2 | **OKVIS2** (ethz-mrl) | stereo-VI | `do_loop_closures: true/false` in euroc.yaml (verified), `do_final_ba: false`, `enforce_realtime: true` | `okvis_app_synchronous` reads ASL directly; callback-driven `okvis::Trajectory` gives real-time state; MSD precedent for "causal VIO-only mode" | 3–6 h | EuRoC, TUM-VI all, MSD subset, Hilti'22, NCD multi-cam, VECtor |
| 3 | **ORB-SLAM3 stereo-inertial** | stereo-VI | LC-off = neuter `LoopClosing` (`mbLoopDetected=false`, [issue #109](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/109)); note LC-off kills map-merge | start from [mateosss/ORB_SLAM3](https://gitlab.freedesktop.org/mateosss/ORB_SLAM3) (slam_tracker per-frame poses already implemented → glue to VIT); fallback: ~20-line `TrackStereo()` Tcw+timestamp logger. 24.04 build via [HackMD guide](https://hackmd.io/@dennis40816/rJqjMi6tJe) (OpenCV 4.6, drop Eigen pin, C++17) | 6–10 h | EuRoC, TUM-VI all, MSD subset, NCD multi-cam |
| 4 | **OpenVINS** | stereo-VI MSCKF | none exists (pure VIO) — clean drift-reference arm | causal by construction; ROS-free `ENABLE_ROS=OFF` build + custom feeder onto `VioManager::feed_measurement_*` (maps 1:1 to our harness), or ROS2 Jazzy serial ([PR #500](https://github.com/rpng/open_vins/pull/500)) | 2–8 h | EuRoC, TUM-VI rooms+corridors, MSD subset |
| 5 | *(stretch)* VINS-Fusion | stereo-VI | `loop_fusion` node = trivial ON/OFF | per-frame odometry topic is causal; needs 20.04 docker or Jazzy fork ([mzahana](https://github.com/mzahana/VINS-Fusion-ROS2-jazzy)) | 6–12 h | EuRoC, TUM-VI rooms only — run only if time remains; it loses anyway (causal EuRoC avg 0.138, OKVIS2-paper re-run) |

Watch item: [tum-vision/voca](https://github.com/tum-vision/voca) (ECCV 2026) — releases causal adapters for exactly systems 2–3; if code drops mid-program, adopt their adapters for cross-validation of ours. Do NOT cite their headline numbers as baselines (H.264 500kbps compressed input).

### 1b. Published-numbers columns (cite-only; label NON-CAUSAL unless noted)

| System | EuRoC SI avg (m) | TUM-VI rooms | Long TUM-VI | MSD (CAUSAL medians) | Source |
|---|---|---|---|---|---|
| ORB-SLAM3 | 0.035 | 0.009 avg | corr 0.01–0.21, mag 0.16–1.86, outd 4.6–54.8 | ATE ~8 cm / RTE ~2.0 cm, ~2% crashes | [arXiv:2007.11898](https://arxiv.org/abs/2007.11898) Tables II–III; [arXiv:2508.00088](https://arxiv.org/abs/2508.00088) (figure-read, approximate) |
| OKVIS2 | 0.031 non-causal / **0.071 causal** (4-DoF!) | 0.01 avg | mag 0.01–0.93, outd2/7/8 0.04–0.15 (loops closed) | **3.5 cm / 1.2 cm — the must-beat** | [arXiv:2202.09199](https://arxiv.org/abs/2202.09199); MSD paper |
| OKVIS2-X (VI mode) | 0.023 non-causal / 0.050 causal | — | — | — | [arXiv:2510.04612](https://arxiv.org/abs/2510.04612) |
| Basalt (paper/3rd-party) | 0.051 (ORB3 re-run, V203 fail) / own paper MH 0.06–0.13 | 0.02–0.13 | corr 0.21–0.42, mag 0.60–3.23, outd 4.1–255 | **4.0 cm / 1.0 cm (causal)** | [arXiv:1904.06504](https://arxiv.org/abs/1904.06504); ORB3 Table III; MSD |
| OpenMAVIS | 0.024 | — | — | — | [arXiv:2309.08142](https://arxiv.org/abs/2309.08142) |
| DM-VIO (mono-i — unfair, footnote only) | 0.037 mono / drift 0.47% | 0.02–0.13 | mag 0.73–2.35 | fails/∞ | [arXiv:2201.04114](https://arxiv.org/abs/2201.04114); MSD |
| VINS-Fusion | 0.138 (hostile re-run) | — | — | — | ORB3 Table II |
| OpenVINS | 0.183 causal (OKVIS2-X re-run) | — | — | discarded (failed easy subsets) | OKVIS2-X; MSD |
| Kimera(-2) | 0.119 / 0.100 causal | — | — | — | ORB3 re-run; OKVIS2-X |
| DROID-SLAM (stereo, no IMU, GPU) | 0.024 | — | — | — | NeurIPS'21 — non-causal global BA, flag |
| DPV-SLAM (mono, Sim3 — incomparable) | 0.024–0.026 | — | — | — | [arXiv:2408.01654](https://arxiv.org/abs/2408.01654) |
| AirSLAM / MASt3R-SLAM / VGGT-SLAM / MAC-VO | cite-only rows | — | — | — | see systems report; license/GPU caveats verbatim |
| Hilti'22/'23 leaderboards | — | — | OKVIS2 32.5/800 ('22 vision-best); MAVIS 452 pts / 0.051 m ('23) | — | [hilti-challenge.com](https://hilti-challenge.com) — the only public CAUSAL leaderboard |

Matrix cells to actually fill (run-ourselves): {Ours×3 arms, stock Basalt, OKVIS2×2 arms, ORB-SLAM3×2 arms, OpenVINS} × {EuRoC 11, TUM-VI room 6, TUM-VI corridor 5 + magistrale 6 + slides 3, MSD 8–12, NCD multi-cam 6–9, Hilti'22 4–8, VECtor 4} ≈ **9 system-arms × ~45 sequences × 5 runs ≈ 2,000 runs** — feasible as nightly batches on 64 cores (most systems are ≤4-thread; run ~16 concurrent, guard against CPU oversubscription skewing "real-time" arms: pin cores, run timing-sensitive arms serially).

---

## 2. DATASET ACQUISITION PLAN

Constraint handling: container has 29 GB → **stage per-sequence**: download/archive on host F: (475 GB budget), rsync one sequence in, run all system-arms on it, keep TUM trajectories + logs (MBs), delete images. Never hold >2 sequences on container.

| Pri | Dataset / sequences | Size (est.) | Effort | Loop-closure story it enables |
|---|---|---|---|---|
| P0 (have) | EuRoC 11, TUM-VI room1–6, MSD MOO01–16 | 0 new | 0 | regression anchor; room-scale; universal comparability |
| **P1 — this week** | **TUM-VI 512px EuRoC-format tarballs: corridor1–5, magistrale1–6, slides1–3** | ~1–5 GB/seq, ~35–60 GB total (est. — verify per-file on [download page](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)) | zero conversion — harness runs unchanged | THE headline plot: start/end-only mocap GT ⇒ ATE = end-to-end drift ⇒ loop-closed vs not is a step change; published Basalt/ORB3/OKVIS2 rows exist for every sequence (Class B protocol) |
| P2 | **MSD non-MOO: MIPB08 (37-min Beat Saber) + MIO subset (4–6 seqs) + 2 MGO** | ~10–60 GB (528 GB full set — cherry-pick per-sequence from [HF](https://huggingface.co/datasets/collabora/monado-slam-datasets)) | zero conversion | map-health over long sessions: keyframe churn, descriptor union, dense 1 cm GT full-duration; product-device realism; MSD causal medians = published yardstick |
| P3 | **TUM-VI outdoors1–8** (or 4 largest first) | ~5–15 GB/seq 512px (est.), ~60–100 GB | zero | km-scale one-shot loops; erratic LC gains expected (OKVIS2 closed 3/8) — honest-limits section |
| P4 | **Hilti 2022: 4 challenge seqs first (pick mixed difficulty), then rest** | 11–34 GB/seq → stage 1-at-a-time; ~60–150 GB on F: | **bag→ASL converter, ~1 day** (pure-python `rosbags`, no ROS; reusable for P5/P6); calib = kalibr-style provided | mm-grade TLS GT, multi-floor, dark/featureless → LOST/reloc stress; self-score offline vs both bin tables; external-referee submission portal still live |
| P5 | **Newer College 2021 multi-cam: quad-e/m/h, stairs, cloister, maths-e; then park** | ~5–20 GB/seq raw (est.) | reuse P4 converter (raw png/csv nearly trivial) | **dense full-trajectory cm GT** → per-frame ATE mid-loop, tunes *how much* correction the pose graph applies (TUM-VI can't) |
| P6 | **VECtor: units-dolly, units-scooter, corridors-walk, school-dolly** | small (~10–30 GB total, est.) | light (rosbag + toolbox) | multi-floor night loops; **BSD-3-Clear license** (only permissive long-range set — matters for anything shipped) |
| P7 (multi-session) | **Hilti 2023: 1 site's overlapping seqs (2–3 bags)** | 9–63 GB/seq | reuse converter | the ONLY set directly exercising cross-session reloc + multi-session LC (session-1-frame scoring ×200) — the xr_map+VPR product story |
| Stretch | BotanicGarden 7 public seqs → 4Seasons (VPR-recall only, registration-gated) → LaMAria | 50–150 GB | medium–high | outdoor foliage; season-shift VPR recall; city-scale leaderboard |
| Skip | KITTI (10 Hz soft-sync IMU), M2DGR (no gray stereo), UZH-FPV (no revisits), TartanAir (sim IMU), Hilti'21 (10 Hz stereo) | — | — | per datasets report |

F: running total through P7: ~300–450 GB — fits 475 GB if MSD stays cherry-picked and Hilti is staged/deleted after conversion (keep converted ASL, drop bags). All sizes marked est. must be verified against download pages before bulk pull.

---

## 3. OUR-STACK CONFIGURATIONS

**(a) BAD/TEBLID (shipped default)** — as on research @ c385ba1. Baseline arm for every sequence. No work.

**(b) XFeat descriptors (CPU ORT)** — dense XFeat already validated on HTP (A8W8 ≈ fp32 quality, memory: xfeat-npu-vpr); container runs the ONNX fp32 dense pass on CPU ORT. Same xr_map matching path, descriptor type swapped. Benchmark question: does learned-descriptor matching raise closure recall on low-texture (Hilti drywall, NCD cloister arches, VECtor night) without precision loss? Work: config plumb + ORT session in replay harness, ~0.5–1 day. NOTE: CPU ORT XFeat timing is NOT the on-device number — report quality only, cite 3.6 ms HTP separately (measured, our own).

**(c) XFeat + VPR retrieval (MUST BE BUILT)** — EigenPlaces-512 first (plumbing stand-in per memory), MegaLoc as the quality pick (clean-room caveat), ONNX CPU. Scope of the xr_map change (both touch the same candidate-selection seam — build as one retrieval module with two call sites):
1. **LOST-scan replacement**: current relocalization brute-scans keyframe descriptors; replace with VPR embedding top-K (K≈5) → PnP verify on K candidates only. Changes: per-keyframe embedding store (512-D f32 or int8), embed-on-keyframe-insert hook, top-K cosine ranker.
2. **Closure candidate pre-ranking**: gate pose-graph closure candidates by VPR similarity before descriptor matching + geometric verification — raises precision ceiling and cuts verify cost on long sequences (magistrale/Hilti where keyframe count grows 10–50×).
Benchmark-relevant deltas: closure recall (MR@100P per [GV-Bench](https://arxiv.org/abs/2407.11736)), reloc success @0.25m/2° + median time-to-relocalize (eval-vislam / OpenLORIS CS-R conventions), and end-drift ATE on Class B. Estimated build: **3–5 days** (retrieval module 2d, xr_map integration 1d, harness metrics/ledger plumbing 1–2d). Multi-session (Hilti'23) additionally needs map save/load across harness invocations — scope check: if xr_map lacks serialization, +2–3 days or defer P7.

**LC ledger (all arms, mandatory):** per run log every closure candidate → accepted/rejected, GT-truth label (<0.5 m/<10°), precision (target 1.0), retrieval-stage max-recall, mean pose-graph correction norm, reloc events + latency. This is what makes the iterate loop diagnosable.

---

## 4. ITERATE-AND-IMPROVE LOOP

**Diagnosis order when a dataset underperforms** (cheapest/most-upstream first — each step's symptom signature from Class-A/B diagnostics):
1. **Calibration/timeshift** — symptom: Sim(3) scale |1−s| > ~0.5%, roll/pitch residual after posyaw alignment ≠ 0, or RTE elevated uniformly. Fix: verify cam-IMU timeshift + extrinsics against dataset-provided kalibr files; Basalt's calibration ingestion per dataset. Never tune downstream until this is clean.
2. **VioConfig** (Basalt front-end: optical-flow patch counts, levels, keyframe criteria, huber, IMU noise densities) — symptom: causal RTE > ~1.2 cm or tracking dropouts on fast motion (slides, MSD beatsaber). Tune on ONE held-out sequence per dataset; freeze.
3. **xr_map keyframe gate** — symptom: closures missed because no keyframe near revisit (ledger shows zero candidates at GT-revisit times) OR keyframe explosion (memory/latency). Knobs: translation/rotation/covisibility thresholds.
4. **Matcher thresholds** (BAD/TEBLID hamming or XFeat cosine, ratio test, min inliers for PnP) — symptom: candidates found but rejected (ledger: high candidate count, low accept, GT says true).
5. **Closure verification + pose-graph weights** — symptom: accepted false closures (precision <1.0 — treat as P0 bug) or corrections too timid/violent (NCD dense GT shows mid-loop over/under-correction). Knobs: inlier floor, reprojection gate, 4-DoF edge covariances, VIO-protection window (memory: relocalization VIO-protection already built — verify it doesn't suppress legit corrections).
6. **VPR layer** (arm c only): K, similarity floor, embedding quantization — tuned via MR@100P curve, not end ATE.

**Regression protocol:** frozen **golden set** = {EuRoC MH01, MH05, V102, V203, TUM-VI room1, room3, corridor1, magistrale1, MSD MOO pick-2, slides1} × 3 arms × 3 runs, nightly on the 64-core box; per-sequence median ATE/RTE + closure precision tracked in a dashboard (extend perf-audit artifact pattern); **merge gate: no sequence regresses >5% or 5 mm (whichever larger) without an explicit waiver note**; any precision<1.0 closure = auto-block. Parameter changes land as named config deltas — one knob-set per commit so bisection works. Full matrix (all datasets × 5 runs) weekly, not nightly. All tuning on held-out sequences; golden-set numbers are never the tuning target (Goodhart guard). Per memory: results reporting includes rendered trajectory/closure gallery panels, not just tables.

---

## 5. PHASE PLAN

| Phase | Content | Effort (person-days) | Wall-clock |
|---|---|---|---|
| **A (wk 1)** | TUM-VI corridor/magistrale/slides download + our 2 arms + stock Basalt, causal, 5-run medians → first LC-matters plot vs published rows (§1b, cite-labeled). Stand up golden-set nightly. | 2–3 | 1 wk |
| **B (wk 1–2)** | OKVIS2 build + causal validation (both LC arms) on EuRoC+TUM-VI; protocol cross-check: our EuRoC causal Basalt ≈ 0.07-class sanity band. OpenVINS feeder. | 3–4 | 1–2 wk |
| **C (wk 2–3)** | ORB-SLAM3 causal (mateosss fork → VIT glue) + 5-run medians full matrix on P0–P3 data. MSD non-MOO pulls. First full comparison table v1 (measured). | 4–5 | 1–2 wk |
| **D (wk 3–4)** | bag→ASL converter; Hilti'22 (4 seqs) + NCD multi-cam + VECtor in-harness; Class C protocol incl. Hilti self-scoring both bin tables. XFeat-CPU arm (config b). | 4–6 | 1–2 wk |
| **E (wk 4–6)** | **Build VPR retrieval layer** (§3c) + closure-ledger metrics; A/B/c three-arm matrix everywhere; iterate loop on weak datasets per §4 order. | 6–8 | 2 wk |
| **F (wk 6–8)** | Hilti'23 multi-session (needs map serialization — scope gate), TUM-VI outdoors, Hilti leaderboard submission (external referee), stretch: VINS-Fusion row, BotanicGarden. Final report + gallery. | 5–7 | 2 wk |
| Total | | **~24–33 pd** | **~8 wk** |

Success criteria (grounded): causal RTE ≤1.0 cm everywhere (at 0.90 — measured, ours); room-scale causal ATE → 3.5–4 cm (OKVIS2/Basalt MSD-class); magistrale causal end-drift beating stock-Basalt 0.60–3.23 m by ≥2× with the VPR arm (projected); closure precision 1.0 across all runs; Hilti'22 causal self-score above OKVIS2's 32.5/800 vision reference (stretch, projected).

**Top 5 risks**
1. **Causal-extraction fidelity for competitors** — a subtly non-causal ORB-SLAM3/OKVIS2 dump flatters them (or us). Mitigation: validate each adapter by diffing causal-vs-final on one loop-heavy sequence — causal MUST show the pre-closure drift discontinuity; adopt VOCA adapters if released.
2. **ORB-SLAM3 instability** (shutdown segfaults, SI-init sensitivity, ~2% crashes per MSD) inflating effort and polluting medians. Mitigation: completion% as a first-class metric; crash = recorded run, not a redo; cap debugging at budget.
3. **VPR layer scope creep** (embedding store, multi-session serialization, MegaLoc licensing clean-room). Mitigation: EigenPlaces-512 plumbing first, MegaLoc swap behind the same interface; Hilti'23 gated on serialization scope check; license review before any publication naming MegaLoc.
4. **Class-B metric near-binarity** — start/end-GT ATE flips between "closed" and "not" causing noisy medians and false regression alarms. Mitigation: 5 runs always, report closure-success count alongside ATE, pair every Class-B conclusion with dense-GT NCD/VECtor evidence.
5. **Storage/transfer churn** (29 GB container, 528 GB MSD temptation, Hilti bags) stalling the run cadence. Mitigation: strict per-sequence staging with auto-delete, converted-ASL-only retention, sizes verified before bulk download; Hilti bags deleted post-conversion.

Everything in §1b is a published number with its source attached; everything from our harness so far is exactly one measured datapoint (TUM-VI room1 causal 13.05 cm VIO / 10.89 cm +LC, RTE 0.90 cm) — all other "ours" cells are pending measurement; all effort/wall-clock/GB figures are estimates.