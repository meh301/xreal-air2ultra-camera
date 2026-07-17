# Architecture sanity check vs OKVIS2 and the literature (2026-07-17)

Piece-by-piece review of the pipeline, cross-checked against OKVIS2
(arXiv:2202.09199), VINS-Mono (arXiv:1708.03852), ORB-SLAM3
(arXiv:2007.11898), XFeat (CVPR 2024), EigenPlaces (ICCV 2023), MegaLoc
(CVPRW 2025), and Kukelova et al. (ACCV 2010). Verdict per piece:
SOUND (literature-consistent), JUSTIFIED DIVERGENCE, or RISK (with the
concrete follow-up).

## 1. VIO backbone — Basalt sqrt keypoint VIO
**Ours:** odometry-only sliding window; the session map lives OUTSIDE the
estimator and couples back via priors/factors.
**OKVIS2:** keeps the long-term map IN the estimator — K=5 keyframe
window, least-covisible keyframes become pose-graph NODES via Schur
marginalization of common observations; loop closure revives those edges
back into landmarks + observations, and old landmarks are RE-OPTIMIZED
as live variables. Cauchy robustifier on observations. Realtime bounded
graph + asynchronous full-graph optimization (tens of ms to ~1 s).
**Verdict: JUSTIFIED DIVERGENCE.** In-estimator maps are why OKVIS2
absorbs sub-cm corrections continuously (their rooms ~1.2 cm); the cost
is an unbounded-ish graph and a second optimizer thread — wrong shape
for a phone-attached AR budget. Our TIGHTSUB→LMFACT→LMTRACK ladder is
the bounded approximation of the same mechanism.
**RISK:** EuRoC is VIO-bound (6.6 vs OKVIS2's 5.2 without LC): their
advantage there is the estimator itself, not loop closure. Basalt
noise/keyframe tuning is the one untested lever.

## 2. Frontend — dense XFeat + LighterGlue-L6 (+BAD/TEBLID fallback)
**Papers:** XFeat = 64-D dense descriptor map + reliability + keypoint
heatmap, early downsampling (the /32 grid constraint we hit on 800x400
is inherent to its pyramid). LightGlue-style matchers are pairwise —
correct to run only on the few verification candidates, never in
retrieval.
**Verdict: SOUND.** Our InterpolateSparse2d sampling convention is
torch-validated (5e-7); the candidate-only LG budget matches the papers'
intended split; NPU A8W8 deployment path is the mobile-correct choice.
BRISK+DBoW2 (OKVIS2) is a full generation older on both counts.

## 3. Retrieval — EigenPlaces-512 (deploy) / MegaLoc-8448 (large map)
**Papers:** EigenPlaces trains viewpoint robustness directly (multi-view
clusters); MegaLoc = DINOv2-B + SALAD at 8448-D. Model sizes alone
predict our measured outcome (eig ~= meg indoors at a fraction of the
cost; meg only pays off on big/aliased maps).
**OKVIS2 uses DBoW2:** weaker retrieval, but an inverted INDEX — full
recall with no shortlist. Our RELOCSWEEP finding (shortlist truncation
loses corridors) is exactly the gap an inverted index doesn't have —
the queued XR_INVIDX item is the principled answer, not just a speedup.
**Verdict: SOUND**, with INVIDX as the literature-backed missing piece.

## 4. Relocalization & verification
**Ours:** gravity-aided 2-point PnP (closed-form quadratic given
vertical) + covis pooling + one-to-one greedy matching + inlier-backed
distinct-keyframe support (covis>=3) + 2-frame temporal confirmation.
**Literature:** the solver is exactly the Kukelova/ACCV-2010 family
(known-vertical 2-point, quadratic in one unknown). Covis pooling =
ORB-SLAM covisibility. Our covis>=3 + 2-frame agreement mirrors
ORB-SLAM3's 3-covisible-keyframe consistency test; OKVIS2 uses plain
3D-2D RANSAC after DBoW2. VINS-Mono confirms the 4-DOF reasoning
(roll/pitch observable => drift is x,y,z,yaw only).
**Verdict: SOUND.** XR_FARBEAR (far matches vote yaw, near solve
translation) matches the established far/near split — distant features
carry rotation, nearby carry translation. Watch item: far votes join
`nin` for downstream strong-match gates — fb_drives + azfb arms are the
false-accept check.

## 5. Map->VIO coupling (stage 2 prior, stage 3 factors)
**Ours:** LMFACT = closure inlier landmarks as FIXED-3D reprojection
factors (Huber, sigma runtime-tunable, <=96/frame), LMTRACK re-posts
while the scene stays in view.
**VINS-Mono relocalization does exactly this** — loop-frame feature
correspondences enter the sliding window as factors with the old
keyframe pose held CONSTANT, and multiple loop frames constrain
simultaneously. Our LMTRACK is the "keep constraining while visible"
half of that.
**OKVIS2 goes further:** revived landmarks are VARIABLES (re-optimized),
so bad map 3D cannot pin the estimator.
**Verdict: SOUND mechanism, two deltas vs the SOTA:**
 (a) they constrain MULTIPLE window frames at once; we post to one frame
     per batch. Basalt-side buffering (8 batches) already supports
     posting the same landmarks against several recent frames —
     implement as XR_LMMULTI. **<- next lever if the matrix stays flat**
 (b) fixed 3D can teach map error back (the v9 lesson, per-point scale).
     XR_LOCALBA improving the stored 3D is the compensating piece — the
     azall composition is the right experiment.
Kernel note: OKVIS2 uses Cauchy, we use Huber — second-order lever only.

## 6. Pose graph / map deformation
**Ours:** Gauss-Seidel blend over the drifted tail (uniform weights,
closure weight 4.0), full-quaternion blending; submap WELD for
cross-segment merges.
**Literature:** VINS-Mono optimizes the pose graph in 4-DOF explicitly —
yaw + translation, roll/pitch LOCKED (they are observable and must not
be bent by closures). ORB-SLAM3's welding window ~= our weld. OKVIS2
distributes loop error by rotation averaging then position.
**Verdict: RISK (the weakest piece theoretically).** Full-quaternion
blending lets closure error leak into roll/pitch, which gravity says are
not drifting. Concrete fix queued: project every blended update onto
yaw-only about the original gravity-consistent attitude (XR_PGO4DOF).
Also: odom edges deserve length-proportional weights eventually.

## 7. XR_LOCALBA vs local BA in the SOTA
ORB-SLAM3: joint Schur BA in pixel space over the covisible set. Ours:
bounded resection-intersection alternation in BEARING space with 4-DOF
poses, plus write-back that unifies each keyframe's private landmark
copy. Alternation converges slower than joint BA but is solver-free,
microsecond-scale, and cannot blow up the map thread budget.
**Verdict: SOUND for refinement** (not reconstruction); the per-kf-copy
unification is nonstandard but required by our storage model, and is
precisely the fixed-3D-quality fix piece 5(b) needs.

## 8. Keyframe management
COVKEEP viewpoint-diversity eviction is ORB-SLAM's redundancy culling
with the sign flipped (evict most-redundant vs cull when 90% of points
are seen elsewhere). Same principle, bounded-memory formulation.
**Verdict: SOUND.**

## 9. Recovery / kidnap
OKVIS2 re-aligns the whole active window rigidly on a verified match and
fixes the rest asynchronously — structurally our snap + weld + PGO
pipeline. Our clip-15 wake-up result (a ~0.5 s burst recovers what
single frames cannot) is consistent with their multi-frame realignment.
**Verdict: SOUND.**

## Actions fed back into the hunt
1. **XR_PGO4DOF** — yaw-only projection of deformation updates (piece 6).
2. **XR_LMMULTI** — post closure factors to the last 3 frames, VINS-style
   (piece 5a; no basalt change needed).
3. Basalt VIO noise/keyframe sweep arms for the EuRoC gap (piece 1).
4. XR_INVIDX inverted index — promoted from perf item to recall item
   (piece 3).
5. (later) Cauchy kernel option in the basalt patch (piece 5).
