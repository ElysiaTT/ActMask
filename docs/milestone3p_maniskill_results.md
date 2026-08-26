# Milestone 3P ManiSkill GPU Pilot Results

## Result

**B6 decision: A — GO TO FULL GPU BENCHMARK.** This authorizes a larger
benchmark within the current simulator scope; it does not authorize new assets,
pretrained models, system changes, or a visual benchmark while Vulkan is
blocked.

This is a state-observation controlled-TCP pilot, not evidence of a real-robot
or RGB-D result. It validates the static-shortcut and temporal-counterfactual
construction before scale-up.

## Data integrity and labels

The GPU pilot has 1,944 examples: 648 per task family, 108 base worlds, and six
candidates per world. Every task has 108 static-balanced and 108 matched label-
flipping pairs. The strict model-input audit passes for all three tasks. For
every matched pair, current observation, candidate action, and static feature
vector have maximum difference `0.0` at tolerance `1e-6`; only preceding
observable history differs. Full evidence is
`maniskill_pilot/counterfactual_audit.json`.

Labels are GPU-PhysX outcomes after execution: interception distance, actual
release/containment/speed, or final rotating-target contact distance. Oracle
future dynamics are not model inputs.

## B5 protocol

All learned methods train only on grouped original-distribution training worlds
and evaluate held-out original, static-balanced, and matched worlds. Pilot
seeds are 17, 29, and 43. The key comparison is confirmed with seeds 17, 29,
43, 59, and 71.

Static methods: CurrentPositionProximity, EndpointDistance, static analytic,
logistic regression, depth-two decision tree, small static MLP, action-only,
and candidate-template-only. Dynamic methods: observable time-aligned motion
analytic, lightweight temporal MLP, and the existing `TemporalActMask` verifier
through a state-to-point adapter. All metrics are reported per task and regime
in `maniskill_pilot/baseline_report.json`; raw seed records are in
`baseline_records.json`.

## Selected aggregate comparison

Strongest static is logistic regression; selected fair dynamic is the temporal
MLP. The table is the three-seed mean across the three task families. CF-pair
accuracy is undefined outside matched groups.

| Regime | Method | Top-1 | Pairwise | AP | ROC-AUC | NDCG | Regret | CF-pair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | Logistic | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | — |
| Original | Temporal MLP | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | — |
| Static-balanced | Logistic | 0.5000 | 0.5335 | 0.5601 | 0.5000 | 0.7754 | 0.5000 | — |
| Static-balanced | Temporal MLP | 1.0000 | 1.0000 | 0.9914 | 0.9909 | 1.0000 | 0.0000 | — |
| Matched | Logistic | 0.5000 | 0.5370 | 0.5596 | 0.5000 | 0.7754 | 0.5000 | 0.5000 |
| Matched | Temporal MLP | 1.0000 | 1.0000 | 0.9907 | 0.9894 | 1.0000 | 0.0000 | 1.0000 |

The static method loses 0.500 Top-1 from original to matched. Temporal MLP
gains 0.500 on matched; paired bootstrap across 21 held-out base-world groups
has 95% CI `[0.500, 0.500]`. The five-seed confirmation reproduces static
0.500 and temporal 1.000 in every task family, with the same paired CI.

## Temporal corruption diagnostics

On matched test worlds temporal MLP Top-1 is 1.0000 with correct history. It
falls to 0.7937 under independent frame permutation, 0.5397 under label-blind
cross-scene history mismatch, 0.8889 under shuffled pre-decision motion, and
0.5000 when motion is zeroed. The TemporalActMask adapter similarly gives
1.0000 correct, 0.6270 permutation, 0.3413 mismatch, and 0.5000 zeroed.
Correct history therefore beats at least two corruption variants.

History reversal remains at 1.0000 for the learned methods. This is a concrete
limitation: the pilot proves reliance on pre-decision motion presence and
consistency, but these mechanisms do not force signed motion direction. The
simple time-aligned analytic is chance (0.5000), so it is not used for the B6
gate.

## B6 gate matrix

- Original static strength: pass (Top-1 1.0000).
- Matched static collapse >= 0.20: pass (drop 0.5000).
- Fair dynamic advantage >= 0.10, paired CI excluding zero: pass (0.5000;
  CI `[0.500, 0.500]`).
- Correct history beats two or more corruptions: pass.
- All three task families show the effect: pass (five-seed static 0.5000,
  temporal 1.0000 on matched test worlds).
- Hidden-state isolation and simulator-execution labels: pass.

The three-seed B5 sweep took 37.19 s on the RTX 4090 D. Peak PyTorch-tracked
allocation was 24,491,008 bytes; PhysX can own additional non-PyTorch runtime
memory. State-only vector smoke achieved 2,079.3 aggregate control-steps/s.
No unrelated GPU process was terminated.
