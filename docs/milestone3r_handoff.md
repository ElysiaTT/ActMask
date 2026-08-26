# Milestone 3R Handoff

## Current phase

Stage-3R decision gate and final regression. The robust corruption and causal
work is complete; no RGB-D/Vulkan work is authorized.

## Work completed

- Phase 0 freezes and reproduces 3Q: static/unordered `0.5`, signed and
  ordered `1.0`, reversal `0.0`, horizon 13, 22 provenance hashes.
- Thirteen deterministic observable corruption variants and their 52
  label/action preservation audits are complete.
- Three-family no-ID correspondence data and eight GPU-PhysX nonlinear
  mechanisms are complete; nonlinear data has 128 strict signed groups.
- Eight analytic estimators, four learned temporal methods, three training
  conditions, validation-only selection, and a final five-seed GRU run are
  complete.
- The three-seed validation audit additionally covers frame-dropout and
  timestamp-jitter augmentation, sequential severity curriculum, and signed
  group pairwise loss without altering the locked test recipe.
- Selected GRU robust pair-order `0.9057` versus selected LastTwo `0.5415`,
  difference `+0.3642`, grouped 95% CI `[+0.2417,+0.4912]`.
- Frozen C5/C10/C20 ranking, causal history interventions, correspondence
  diagnostic, held-mechanism nonlinear OOD, and C20 latency/memory measures
  are complete. C20 p95 is 0.417 ms and peak allocation is 262 MB.

## Blocker or decision

Current scientifically honest result is **Stage-3R E: human decision
required**, not A. The non-linear OOD mechanisms `jerk` and `regime_switch`
both retain perfect LastTwo performance, so they offer no learned-over-analytic
positive support (delta 0; CI `[0,0]`). Unseen distractor-count and
crossing-angle variants were not preregistered. The supplied rules prohibit
post-result changes to simulator success/dynamics or weakening OOD gates.

## Evidence

- `outputs/actmask/milestone3r_robust_signed_dynamics/frozen_3q_reproduction.json`
- `.../corruption_full/corruption_manifest.json`
- `.../nonlinear_full/nonlinear_summary.json`
- `.../evaluation/validation_selection.json`
- `.../evaluation/selected_vs_analytic.json`
- `.../evaluation/ranking_c5_c10_c20.json`
- `.../evaluation/causal_controls_five_seed.json`
- `.../evaluation/correspondence_five_seed.json`
- `.../evaluation/nonlinear_held_mechanism_ood.json`
- `.../evaluation/training_conditions_validation.json`
- `.../evaluation/completion_audit.json`

The minimal nonlinear reproduction is:

`CUDA_VISIBLE_DEVICES=0 /data/env/tzh/conda_envs/actmask/bin/python -c "from actmask.experiments.milestone3r_nonlinear_ood import run; print(run('outputs/actmask/milestone3r_robust_signed_dynamics/nonlinear_full','outputs/actmask/milestone3r_robust_signed_dynamics/evaluation'))"`

## Attempts already made

1. The nonlinear task used two pre-data physical trajectory corrections only
to eliminate common-path re-intersections. The strict execution pair audit
then passed.
2. A held-mechanism five-seed test was run without changing the task after its
result. It found ties, not a learned advantage; no third post-hoc task change
was attempted.

## Human decision required

1. Recommended: authorize a new **3R-NL preregistered extension** with fresh
   nonlinear pair mechanisms whose last two observable frames are matched but
   acceleration/jerk history changes execution outcome. Scientific consequence:
   directly tests the missing OOD gate. Engineering cost: moderate, new
   task/version/data/audits only. Risk: it may confirm the analytic rule is
   sufficient.
2. Authorize a new **3R-AMB preregistered extension** for three distractors and
   unseen crossing angles, keeping current labels/actions frozen. Scientific
   consequence: completes ambiguity OOD. Engineering cost: low-to-moderate.
   Risk: synthetic correspondence construction may remain non-discriminative.
3. Accept the current evidence as a robust state-observation result but stop
   before visual work. Scientific consequence: no claim of a nonlinear OOD
   learned advantage. Engineering cost: none. Risk: Stage-3R gate remains E,
   not visual authorization.

## Exact resume instruction

`Choose option 1, 2, or 3 above. For option 1 or 2, supply the new
preregistration and accept a separately versioned 3R extension; do not alter
the frozen 3Q/3R artifacts or current success definitions.`
