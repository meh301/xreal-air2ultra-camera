# XRV — the estimator-core rebuild

Status: design committed 2026-07-18; keystone implementation begins on
the delayed-marginalization twin. Ledger of record: bench/ITERATIONS.md.

## Why a rebuild, and why this scope

Measured (site /analysis.html + ledger): our frontend out-tracks
OKVIS2's locally (RTE better on 24/39 seqs); the losses live in
(a) photometric robustness, (b) blur track survival — both frontend-
local, fixed by surgery (stage 9 ZNCC + successors) — and
(c) marginalization rigidity, (d) closure-absorption interface — both
STRUCTURAL. Nine patch stages of grafting onto the basalt fork bought
rooms 13→6.5 and corr3 31→12, but each stage fought undocumented
invariants (the stage-8 five-bug odyssey: QR row minimums, AbsOrderMap
membership, take_kf consumption, frame-lifetime races, lost-set
semantics). The conversion ceiling (ours 1.0-1.2× vs their 1.7-31.9×)
is set by (c)+(d), and they cannot be grafted away. The estimator core
gets rebuilt to make them native; everything that measurably wins stays.

## Module boundaries

KEPT (basalt, unchanged or lightly patched):
- Optical-flow frontend (patch KLT + stage-9 ZNCC; later: redetect-on-
  collapse, coherent-outlier frame gate)
- IMU preintegration types, camera models (kb4/ds), calibration I/O
- The VIT tracker `.so` interface — the replay harness, map layer,
  factor bridges, and every launcher work unchanged against XRV

REFERENCED:
- Demmel et al. sqrt marginalization (nullspace projection, QR prior
  updates) — the numerical form XRV keeps
- DM-VIO delayed marginalization — the reversibility mechanism
- OKVIS2 — pose-graph-edge ↔ observation duality, async loop opt

REBUILT (new code, `src/xrv/`):
1. `xrv_state.h` — window state (frames, keyframes, long-term poses)
   with explicit lifecycle contracts (every invariant the stage-8
   odyssey discovered, documented and asserted at the boundary).
2. `xrv_marg.{h,cpp}` — THE KEYSTONE: twin sqrt priors.
   - `live` prior: exactly today's QR-updated sqrt marg (fast path).
   - `delayed` prior: marginalization deferred by D keyframes (DM-VIO);
     re-advanced on demand with CURRENT linearization points.
   - On a verified closure: rebuild `live` from `delayed` + the
     closure's revived constraints, relinearized — folds stop being
     forever; wrong info is recoverable for D keyframes.
3. `xrv_landmarks.{h,cpp}` — landmark DB where injection and revival
   are native: every landmark carries its observation history (bounded)
   so marginalized observations can be revived as factors at closure
   (the OKVIS2 edge↔observation duality), with host-migration instead
   of host-death.
4. `xrv_absorb.{h,cpp}` — the closure interface: takes the map layer's
   verified closures (same bridge calls as today) and applies them as
   revived-observation factors + delayed-prior rebuilds, replacing the
   sigma-channel grafts of stages 3-8.

## Phasing (each phase A/B-able against fz19 on the full bench)

P1  Stage-9 frontend fixes land (ZNCC now; redetect + frame gate next).
    These carry into XRV unchanged.                       [in flight]
P2  xrv_marg twin prior behind a flag (XR_DELAYMARG): identical output
    when the delayed path is never consulted — a pure-refactor A/B
    (expect bit-parity modulo float order).
P3  Closure-triggered delayed re-advance + relinearization: the first
    behavioral change. Target: rooms (absorption showcase) and the
    corridor return-registration (corr1 fortress — revival at re-entry
    is exactly what the ~60cm rigid-residual autopsy calls for).
P4  Native revival replaces stage-3..8 grafts (LMFACT/LMMARG/LMINJ
    become one mechanism); the graft flags remain for A/B parity.
P5  Async loop optimization (OKVIS2-style) over the delayed window.

## Guardrails

- Every phase ships env-gated; fz19 stays the freeze until a phase
  beats it on the full within-round protocol (EVALUATION.md §12).
- The robustness discipline (verify gates, confirms, scene gating) is
  non-negotiable: XRV absorbs MORE only where evidence is recoverable.
- Determinism: XRV development waits on the lockstep verdict; if
  lockstep collapses corridor variance, all XRV corridor A/Bs run
  lockstep-on.
