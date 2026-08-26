# Draft skeleton (state-only benchmark paper)

## Abstract

Dynamic robot verifiers can appear successful while exploiting static, unordered, or short-history cues. We introduce execution-grounded counterfactual benchmarks that progressively remove these cues. In a frozen nonlinear state-only protocol with matched final two frames and action-diverse candidates, a lightweight GRU exceeds the selected fair analytic baseline by +0.400 pair-order accuracy (grouped 95% CI [0.3951, 0.4052]) and remains positive on three OOD axes. These are simulator state-observation results, not visual or real-robot results.

## Introduction

State the shortcut problem, the need for matched counterfactuals, and the paper's evaluation—not model-scaling—focus.

## Related work

Placeholder: robot action verification; counterfactual and causal evaluation; temporal state estimation; manipulation simulation.

## Problem setup and benchmark design

Define observable history, candidate actions, GPU-PhysX execution labels, grouped world splits, and the state-only fairness whitelist. Explain static matching, signed-order construction, and v2 early-frame pair-swap `T(h)=reverse(h[:4])+h[4:]` with common final frames.

## Shortcut taxonomy and methods

Describe static geometry, order-invariant motion, final-two-frame, and candidate-action degeneracy shortcuts. Then describe execution-grounded generation, signed-order pairs, nonlinear v2 pairs, and action-diverse candidates.

## Baselines and experiments

Report static, action-only, unordered-history, LastTwo, fair analytic, GRU, and diagnostic temporal controls. Organize the progression 3P → 3Q → 3R → v1 → v2; clearly distinguish intermediate failures from final evidence.

## Results

Insert `tables/main_v2_results.tex`, `tables/causal_controls.tex`, `tables/candidate_diversity.tex`, and the six figures. State all OOD, CI, latency, and candidate-diversity values with their frozen source paths.

## Limitations and future work

This is a state-only procedural GPU-PhysX benchmark. It does not establish RGB-D, visual tracking, real-robot transfer, policy improvement, or universal dynamics understanding. Future visual work requires the separate readiness protocol.

## Reproducibility

Reference the artifact manifest, source hashes, independent audit, red-team audit, and regeneration command.
