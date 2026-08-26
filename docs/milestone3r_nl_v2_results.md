# Milestone 3R-NL-v2 Validity Probe Results

## Scope and frozen provenance

This is a separate v2 probe.  V1 remains frozen and was not overwritten.  The
v2 configuration hash, recorded before v2 simulator data or learned-model
evaluation, is `9dc510bdec94e4715df96f6cc6f960f932d4dd37babea957e0f14325cf98e564`.

## V1 forensic result

The v1 audit covers 50,000 strict pairs.  Full generic reversal is not a
counterfactual swap: its maximum mismatch to the paired partner is 0.0900 for
damping/drive and 0.1288 for hysteresis, because it moves the two common final
synchronization frames to the beginning.  The unordered early-frame multiset,
current/final frames, action, timestamps, TCP and masks are all exactly
matched.  In contrast, reversing **only** the four early mode frames maps each
member to its partner with error 0.  This explains why generic reversal was
not a valid causal intervention in v1.

V2 therefore preregisters `T(h) = [reverse(h[0:4]), h[4:6]]`; timestamps are
common and retain their slot semantics.  Generic full reversal is only a
diagnostic.

## Bounded GPU probe data

The final probe used 6,720 candidate branch executions: 3,840 ID and 960 for
each of physical-parameter, temporal-delay, and held-mechanism OOD.  Including
smoke/diagnostic executions, v2 remained below 20,000 pre-approval executions.
The final generated data contains 1,600 ID strict pairs, 320 physical OOD,
224 temporal OOD, and 480 held-mechanism OOD pairs.  All accepted pairs have
zero final/current/action/timestamp/static matching error and zero pair-swap
error, with a 1.0 execution label-flip rate.

Candidate diversity on the ID execution log is non-degenerate for 83.33% of
worlds (the remaining 16.67% are separately flagged rotating damping/drive
worlds with no strict flips).  The mean branch-dependent candidate outcome
change rate and entropy are both 0.8333.  Every template has the same aggregate
success rate; candidate-template rate span is 0 and candidate-index leakage is
numerically zero.  On 30 non-degenerate test worlds, dynamic ranking has
Top-1=1.0, NDCG=1.0 and zero regret; static/action-only have tie-aware
Top-1=0.5 and regret=0.5.

## Independent OOD audit

ID damping is [0.1200, 0.1787] and physical OOD damping is
[0.2600, 0.3167]; ID frequency is [1.1500, 1.4412] and OOD frequency is
[1.8500, 2.1412].  No interval overlaps.  ID hysteresis delay is {0,1} and
temporal OOD delay is {2,3}; no overlap.  Held-mechanism fitting uses only
damping/drive while test uses only hysteretic mode memory.

## Fair models and causal diagnostics

Three fixed seeds selected GRU on validation.  Test pair order is GRU 1.000;
the strongest fair analytic baseline, MultiFrameLinearVelocity, is 0.600,
giving +0.400.  Static, action-only, unordered and LastTwo are 0.500.  GRU
has a +0.500 earlier-history advantage over final-two-only.

Counterfactual Swap Accuracy is 1.000 and Score-Swap Consistency is 1.000 for
GRU; all three static/action/unordered controls are 0.  The valid pair-swap
and independently permuted early history respectively swap/degrade decisions;
last-two-only also drops to 0.500.  Generic full reversal is 0.933 and is not
used as a gate.  Physical and temporal OOD each score 1.000; held mechanism
OOD scores 0.792.  GPU model peak allocated memory is 78.14 MiB.

## Probe decision: A — VALIDITY PROBE PASSED

All Phase-5 probe gates pass: strict matching and hidden-field audits,
candidate diversity, branch-dependent changes, independent OOD ranges,
counterfactual swap and control separation, history advantage, non-degenerate
ranking, learned-over-analytic advantage, and multi-family/multi-mechanism
support.  A separately frozen v2 full run is therefore authorized by the
probe.  It has **not** been started in this validity-probe milestone.

The complete collected regression suite passed: 166 tests in 55.02 s.  No
separately identifiable high-DPI/legacy-visual test node exists in this
checkout's pytest collection; it is therefore recorded as `not_present_in_collection`
and is not a state-only scientific gate.  No visual artifact was changed.

## Frozen full run

The probe-authorized full configuration has SHA-256
`12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47`.
It generated exactly 92,000 GPU-PhysX branch executions: 60,000 ID across five
valid family--mechanism cells, 8,000 physical-parameter OOD, 12,000 temporal
OOD, and 12,000 held-mechanism OOD. C5 and C10 are aligned prefixes of the
single C20 execution set and add zero simulator work.

The final ID data has 30,000 strict pairs. Every C20 world is non-degenerate,
with 10 success and 10 failure candidates per branch. Candidate template
success rates are all 0.5, so neither index nor template order predicts the
label. The original full audit incorrectly required exactly ten candidates;
the bookkeeping fix accepts the preregistered C20 (`>=10`) and did not alter
any simulator or model result.

Five-seed GRU confirmation yields pair order 1.000. The selected fair analytic
baseline is MultiFrameLinearVelocity; learned minus analytic is +0.400 with
grouped 95% CI [+0.3951, +0.4052] across 6,000 test pairs. Physical OOD is
1.000 versus 0.500 analytic; temporal OOD is 1.000 versus 0.8571; held
mechanism OOD is 0.7867 versus 0.6667. Counterfactual swap and score-swap
consistency are 1.000, and full-history advantage is +0.500.

For C5/C10/C20, dynamic Top-1/NDCG is 1.0/1.0 and tie-aware static/action
Top-1 is 0.5, giving a +0.5 action-ranking advantage. C20 p95 inference is
0.816 ms, and peak allocated GPU memory is 1,021.5 MiB.

## Final nonlinear-OOD decision: PASSED

All eleven frozen full gates pass: learned advantage and CI, positive advantage
on three OOD axes, three-family support, history and pair-swap causality,
non-degenerate ranking, dynamic control separation, C20 latency, hidden-field
audits, and regression checks. This completes the nonlinear-OOD portion of the
research gate. It does **not** authorize Vulkan, RGB-D, assets, or visual work.

The migrated runtime was repaired by redirecting its 24 stale Torch/CUDA/Triton
site-package links from the removed `/data/env/...` prefix to the matching
`/home/tzh/conda_envs/papers` prefix. The active interpreter is now
`/home/tzh/conda_envs/actmask/bin/python` with Torch 2.6.0+cu124, ManiSkill
3.0.1, SAPIEN 3.0.3 and the NVIDIA A16. The complete regression collection
contains 169 tests after the full-run checks.
