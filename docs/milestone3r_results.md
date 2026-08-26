# Milestone 3R Results

## Status

Phase 0 passes. `frozen_3q_reproduction.json` records SHA-256 provenance for
22 frozen 3Q data/source artifacts, confirms the fixed-phase rotating horizon
is 13 steps, and reruns a deterministic four-task 16-world subset. Its
three-seed grouped result reproduces static/unordered `0.5`, signed finite
difference and ordered temporal MLP `1.0`, and exact-reversed history `0.0`.
No frozen 3Q path was written by the reproduction command.

The corruption, correspondence, nonlinear-dynamics, learned-model, and ranking
conclusions remain pending.

## Phase 1: observable corruption layer

The complete frozen four-task data set has been materialized into 13 separately
seeded observable-only variants: mild/medium/severe Gaussian noise,
heteroscedastic observable-speed noise, sparse outliers, quantization,
timestamp jitter, irregular intervals, random and last-frame dropout, burst
occlusion, observation latency, and randomly selected partial coordinates.
The manifest contains 52 task-level audits. Each confirms labels and candidate
actions are unchanged; timestamp variants remain strictly monotonic; and frame
dropout has explicit visibility/confidence masks. No transformation receives
or reads a success label, hidden state, future trajectory, simulator ID, or
mechanism field.

## Phase 2: correspondence ambiguity

`moving_cube_intercept`, `signed_moving_window_placement`, and
`fixed_phase_rotating_capture_window` now have separately materialized,
observable two-object histories. Each has a same-scale distractor, crossing
trajectory, intermediate target occlusion, near-parallel early motion with late
divergence, and an identity-swap diagnostic. Fair files provide nearest-neighbor
estimated correspondence, soft correspondence, and no-explicit-identity set
forms; oracle identity is explicitly diagnostic-only and absent from fair keys.
All three task labels and candidate actions remain byte-identical to frozen 3Q.

## Phase 3: nonlinear GPU-PhysX task (implementation qualification)

The new, separately registered `ActMaskNonlinearSignedIntercept-v1` implements
eight execution mechanisms: constant acceleration, jerk, piecewise velocity,
delayed onset, damping, collision-like velocity change, curved motion, and a
post-observation regime switch. A CUDA 8-way smoke confirms finite observations
and execution `success` fields. Its eight-mechanism strict GPU probe now
passes: every common positive candidate trajectory succeeds and its same-action
negative-sign branch fails. Two pre-data corrections were used only to remove
late common-path re-intersections (collision-like motion and curved arc);
labels, thresholds, branch-specific actions, and frozen 3Q parameters were not
changed. Full strict-pair generation remains required before admission.

That generation now passes at the preregistered reduced scale: eight mechanisms
by 16 fixed worlds yields 128 strict signed groups and 256 GPU-PhysX examples,
with a balanced 0.5 execution-success rate. The accepted metadata and inputs
are under `nonlinear_full/`; every pair has matched current final frame and
candidate action and opposite execution labels.

## Phase 4--6: fair estimators, validation selection, and final seeds

All eight required observable-only analytic estimators are implemented. The
matrix is evaluated in batched CUDA form; `LastTwoFrameFiniteDifference` is
checked directly against the NumPy implementation. On the fixed validation
robustness suite, it is the strongest fair analytic method (mean pair order
`0.5344`).

`OracleCleanState` is separately reported as a diagnostic-only upper bound: it
receives the uncorrupted clean observable state for every corruption and is
marked `fair=false`. Fair selection and learned-versus-analytic comparison
explicitly exclude it.

The learned sweep compared OrderedTemporalMLP, GRU, temporal convolution, and
masked-history MLP under clean, mixed, and curriculum training. A fixed,
validation-only rule over eight distinct sensing failures selected the
clean-trained GRU: validation pair-order `0.8895` (96 task/seed/variant
observations). The final five deterministic seeds are `17,29,43,59,71`; no
per-corruption test selection was performed.

The completed three-seed validation-only condition audit does not change that
locked final recipe. Clean-only/mixed/curriculum/frame-dropout/jitter/pairwise
validation means are `0.9297/0.9207/0.9298/0.9323/0.9263/0.9342` (84
observations each). Curriculum is sequential clean-to-mild-to-medium and the
pairwise condition uses signed-group positive-over-negative loss. The GRU has
9,249 parameters; exact reversal remains a diagnostic rather than a
mislabelled training augmentation.

Across the task-specific action horizons, baseline parameter counts are
1,561 (action-only), 2,721 (static), 5,440 (temporal convolution), 7,537
(ordered/masked/unordered MLP), and 8,577--9,249 (GRU); the exact table is
`evaluation/model_parameter_counts.json`.

On the untouched robust test aggregate, the selected GRU has pair order
`0.9057` versus validation-selected LastTwo `0.5415`, a difference of
`+0.3642`. Grouped task-by-corruption bootstrap gives 95% CI
`[+0.2417,+0.4912]` over 32 groups. The learning advantage is particularly
large for last-frame dropout (`0.975` vs `0.000`), observation latency
(`0.975` vs `0.000`), random dropout (`0.8208` vs `0.5460`), severe Gaussian
noise (`0.9451` vs `0.7310`), and partial coordinates (`0.9298` vs `0.7403`).
The complete non-selected sweep, selection record, final five-seed scores, and
comparison are in `evaluation/learned_corruption_matrix.json`,
`evaluation/validation_selection.json`,
`evaluation/selected_gru_five_seed.json`, and
`evaluation/selected_vs_analytic.json`.

## Phase 7: frozen C5/C10/C20 ranking

Candidate sets are deterministic prefixes of the frozen 20 candidate IDs for
each signed-order held-out world; labels and candidate success semantics are
unchanged. On C20, clean GRU/LastTwo pair order is `1.0/1.0`. Under severe
Gaussian, random-dropout, last-frame-dropout, and latency respectively, GRU
pair order is `0.9725/0.9450/1.0000/1.0000`, while LastTwo is
`0.7310/0.5460/0.0000/0.0000`. The aggregate C5/C10/C20 Top-1 bootstrap deltas
are `+0.2243`, `+0.2379`, and `+0.2282`, each with CI excluding zero.

Because each frozen C20 group contains multiple successful candidates, Top-1
alone is intentionally easy for static controls. Pair order, AP, and regret
are therefore the discriminative metrics; static/action/unordered controls
remain at pair order `0.5`. Full per-task/per-corruption/per-C metrics are in
`evaluation/ranking_c5_c10_c20.json`.

## Phase 8: causal and correspondence controls

The five-seed selected GRU has clean C20 Top-1 `1.0` and exact-reversed Top-1
`0.0`, giving reversal causal drop `1.0`; corresponding pair order is
`1.0` versus `0.0`. Independently permuted history has pair order `0.512`,
mismatched history `0.518`, velocity-sign-removed `0.604`, and speed-magnitude
only `0.590`. Static MLP, action MLP, and unordered-history MLP each have
pair order `0.5`. Shuffled timestamps do not affect this fixed-interval data,
so timestamp availability is not claimed as the source of the learned gain.

The three-family ambiguity construction has no identity key in fair inputs.
At C20, nearest-neighbor, soft correspondence, and no-identity set encodings
all score Top-1 `1.0`; the oracle-to-nearest Top-1 gap is `0.0`. The diagnostic
oracle target track is occluded and has pair order `0.5`, while estimated
tracks have `1.0`, indicating that this synthetic distractor setup supplies
an alternative observable track cue rather than an oracle advantage. It is
reported as a diagnosis, not as evidence for model superiority.

## Phase 9: nonlinear OOD and current decision

Two held-mechanism tests, `jerk` and `regime_switch`, train without the held
mechanism and test only worlds 12--15. Both give GRU `1.0` and LastTwo `1.0`
pair order (delta `0.0`; 95% CI `[0,0]`, 8 groups). Thus the current nonlinear
execution pairs are valid but do **not** establish an OOD learned advantage:
their signed final-two-frame signal remains sufficient. Changing the task
success/dynamics definition after observing this result is explicitly outside
the authorized autonomy.

The clean-only selected GRU treats every non-clean corruption as unseen at
training time, including severe Gaussian noise, outliers, random dropout,
timestamp jitter, latency, partial coordinates, and the held-out
partial-coordinate family. However, explicit unseen distractor-count and
crossing-angle OOD variants have not been added, because doing so now would
require a new prespecified ambiguity protocol rather than a post-hoc tweak.

Consequently, Stage-3R decision **E (human decision required)** is the only
truthful current decision: the robust corruption result supports learned
temporal value, but the required two nonlinear OOD axes have no positive
learned-over-analytic support and two ambiguity OOD axes remain unprespecified.
This does not authorize a visual/RGB-D benchmark.

## Efficiency and tests

Selected GRU checkpoint size is 35,652 bytes, peak GPU allocation is
261,997,056 bytes, and C20 median/p95 batch inference is 0.371/0.417 ms
(about 130,411 candidates/s). The C20 verifier therefore exceeds 5 Hz by a
large margin. `final_regression.log` records 59 passing existing tests before
the run was intentionally interrupted during a pre-existing PIL rendering
test to avoid prolonged CPU use. All new 3R tests passed individually; the
complete suite has not yet obtained a final passing record in this session.

`regression_excluding_render.log` records `151 passed, 1 deselected` in 54.10
seconds. The remaining unrelated existing test renders six high-DPI Matplotlib
3D figures and was not completed under the requested CPU-use constraint.
`evaluation/completion_audit.json` is the machine-readable audit; it marks
only that renderer test and the non-linear OOD gate as outstanding.
