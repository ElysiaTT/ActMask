# Milestone 3P Handoff

## Current stage and phase

Stage A (CPU static-shortcut audit) is complete. Stage B0 GPU qualification
now passes; B1 local resource audit remains complete. The B2 GPU benchmark has
not started because no suitable local simulator or realistic labelled data is
available.

## Work completed

Implemented strict-observable static feature extraction, original/static-
balanced/matched distributions, eight exact counterfactual dynamic families,
static and temporal method catalogues, corruption tests, five-seed
confirmation, anti-shortcut ablations, figures, tables, and eleven focused
tests. The immutable CPU result root is
`outputs/actmask/milestone3p_cpu_shortcut_audit/`; B0/B1 evidence is in
`outputs/actmask/milestone3p_gpu_benchmark/gpu_qualification.json` and the
superseding runtime qualification record
`outputs/actmask/milestone3p_gpu_benchmark/gpu_qualification_runtime.json`.

## Decision or blocker

The original distribution is shortcut-heavy, but the matched benchmark is
valid: static performance collapses and observable temporal reasoning recovers
the task. The Stage-A gates authorized B0/B1, and B0 now passes on the RTX
4090 D. B2 remains blocked because there is no installed Isaac Sim runtime,
running Docker daemon, or local realistic observation/trajectory data with
labels. A human resource decision is required before benchmark design begins.

## Evidence

Original static Top-1 is 1.0000. Matched static Top-1 is 0.2708, while the
time-aligned observable baseline is 0.8542 (paired delta 0.5833; 95% CI
[0.4167, 0.7292]). Exact static-pair leakage Bayes accuracy is 0.5. The
completion extension adds held-out static permutation importance, all frozen
verifier corruption views, and extended ablations; final verification reports
134 passed tests. B0 passes with CPU--CUDA error `2.98e-8`, identical ranking,
repeatable CUDA output, and valid FP16 ranking on an RTX 4090 D. See
`docs/milestone3p_cpu_results.md`, `docs/milestone3p_gpu_integration_audit.md`,
`docs/milestone3p_gpu_results.md`, and the summary JSON files.

## Attempts already made

Prior 2E stopped the learned candidate-verifier line because static geometry
beat correct history. 3P retained that evidence, then eliminated the shortcut
with exact counterfactual matches and confirmed temporal causality under five
history corruptions. The completion extension found that the frozen old
verifier does not transfer as a temporal method on the new matched worlds and
that no small anti-shortcut intervention beats ordinary matched sampling.
B0 was rerun after a GPU was attached and passed CPU--CUDA/FP16/determinism
checks on a fixed mini-batch. B1 made only read-only local checks. No GPU
benchmark, simulator download, package install, container start, or external
project import has been attempted.

## Human decision required

Choose one of these concrete options:

1. **Provide an existing approved local simulator/runtime plus permitted
   assets and task family.** Scientific consequence: permits a faithful B2
   benchmark without changing the data-source claim. Engineering cost: medium.
   Risk: the available runtime may still lack faithful labels. **Recommended**
   if such a local stack already exists.
2. **Provide approved realistic recordings/trajectories and their success
   labels.** Scientific consequence: tests the shortcut claim against a
   recorded-data benchmark rather than a simulator. Engineering cost: medium.
   Risk: matching current geometry may be less exact than simulation.
3. **Explicitly authorize a named simulator/asset installation or download.**
   Scientific consequence: expands the resource basis of the project.
   Engineering cost: high. Risk: package/asset scale and reproducibility
   changes; this will not be inferred automatically.

## Exact resume instruction

After the human resource choice, issue this exact Codex amendment:
`Continue Milestone 3P Stage B. The approved GPU/runtime is <path>, the
approved task family/assets are <scope>, and installation/download is
<forbidden or explicitly allowed>.`

Then first re-run B0 GPU qualification. If it passes, implement a minimal
adapter only for the approved local runtime and assets, then run the bounded
GPU benchmark. Do not use an external source tree or download anything unless
the amendment explicitly authorizes it.
