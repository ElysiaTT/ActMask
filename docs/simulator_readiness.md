# Simulator readiness — Milestone 2C

## Current status

`NO-GO FOR SIMULATOR INTEGRATION`

This is a CPU-only synthetic-study decision. It does not make claims about RGB-D, simulator trajectories, or scene flow.

## Passed gates

- No dependence on fixed cross-frame point IDs.
- 10/20-candidate utility decision value with paired statistical support.
- CPU runtime for shared 10-candidate evaluation.
- History/action reliance evidence.

## Failed gates

- The selected model is significantly worse than the strongest fair baseline on ID AP.
- No OOD condition has a positive paired AP confidence interval against that baseline.
- Action counterfactual matching does not meet the 0.80 threshold.

## Required remediation before reconsideration

1. Improve correspondence-robust mask AP against MultiHypothesisTrajectoryProximity while retaining permutation/resampling robustness.
2. Raise action-counterfactual matching to at least 0.80 on the held-out protocol.
3. Re-run the independent ID/OOD paired bootstrap and permutation tests; only a positive CI-supported advantage can change this decision.

See `outputs/actmask/milestone2c/summary.md` for the executed protocol and exact values.
