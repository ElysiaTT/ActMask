# Paper outline

Suggested title: **Do Dynamic Robot Verifiers Really Use Dynamics? Counterfactual Benchmarks for Static, Order-Invariant, and Nonlinear Temporal Shortcuts**.

Alternatives: *Counterfactual Stress Tests for Dynamic Robot Verifiers*; *When a Dynamic Verifier Need Not Use Dynamics*; *State-Only Counterfactual Benchmarks for Temporal Shortcut Detection*.

1. Abstract: state-only benchmark/evaluation contribution and bounded v2 result.
2. Introduction: why apparent dynamics performance may be shortcut performance.
3. Related work placeholders: action verification, counterfactual evaluation, state estimation, temporal models.
4. Problem setup: state history, candidate action, execution-grounded success, grouped splits.
5. Counterfactual benchmark design: matching, pairs, and audit criteria.
6. Shortcut taxonomy: static geometry; order-invariant motion; final-two-frame; candidate-action degeneracy.
7. Method: GPU-PhysX generation; signed-order pairs; nonlinear v2 pair-swap; action-diverse candidates.
8. Baselines: static/action/unordered; analytic estimators; lightweight temporal models.
9. Experiments: 3P, 3Q, 3R, 3R-NL-v1, and frozen v2 full.
10. Results: main, causal-control, diversity, latency, and audit tables.
11. Limitations: state-only procedural simulator scope.
12. Future visual/RGB-D extension: plan only.
13. Reproducibility checklist and artifact manifest.
