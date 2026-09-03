# TimeArrow-v3 frozen CPU result

Date: 2026-09-03

## Decision

**NO-GO for scaling the current construction.**

The learned ordered-history model beats the validation-selected analytic model,
but the result does not isolate temporal order. An action-and-target-only model
already solves most of the task, and an unordered-history model is essentially
tied with the ordered model. The apparent learned improvement therefore cannot
support a claim that ActMask learned or used temporal dynamics.

This negative result is useful: the CPU admission test caught the shortcut
before a costly simulator or vision experiment.

## Frozen run

- Protocol: `docs/timearrow_v3_cpu_preregister.md`
- Script: `scripts/cpu_timearrow_v3.py`
- Seeds: 7301, 7302, 7303, 7304, 7305
- Per seed: 1,200 train, 300 validation, 400 ID test, 400 parameter-OOD,
  and 400 held-mechanism worlds
- Seven same-state candidate executions are retained for every world
- Runtime: 96.46 seconds on CPU
- The validation-selected analytic model was `linear_system_id` for all five
  seeds

Values below are mean +/- sample standard deviation over five seeds.

| Split | Model | Pair-order | Top-1 | Normalized regret |
|---|---|---:|---:|---:|
| ID | selected analytic | 0.9239 +/- 0.0047 | 0.5960 +/- 0.0304 | 0.0959 +/- 0.0073 |
| ID | action + target | 0.9645 +/- 0.0035 | 0.7480 +/- 0.0295 | 0.0383 +/- 0.0050 |
| ID | current state | 0.9665 +/- 0.0018 | 0.7655 +/- 0.0119 | 0.0355 +/- 0.0019 |
| ID | unordered history | 0.9838 +/- 0.0005 | 0.8645 +/- 0.0165 | 0.0170 +/- 0.0019 |
| ID | ordered history | 0.9850 +/- 0.0013 | 0.8950 +/- 0.0206 | 0.0115 +/- 0.0011 |
| Parameter OOD | selected analytic | 0.8984 +/- 0.0060 | 0.6075 +/- 0.0143 | 0.1085 +/- 0.0083 |
| Parameter OOD | unordered history | 0.9438 +/- 0.0028 | 0.6170 +/- 0.0324 | 0.0791 +/- 0.0055 |
| Parameter OOD | ordered history | 0.9450 +/- 0.0031 | 0.6145 +/- 0.0341 | 0.0794 +/- 0.0068 |
| Held mechanism | selected analytic | 0.8593 +/- 0.0102 | 0.3260 +/- 0.0407 | 0.1460 +/- 0.0152 |
| Held mechanism | unordered history | 0.8388 +/- 0.0058 | 0.4540 +/- 0.0359 | 0.0565 +/- 0.0057 |
| Held mechanism | ordered history | 0.8401 +/- 0.0049 | 0.4555 +/- 0.0318 | 0.0589 +/- 0.0043 |

## Gate audit

| Frozen gate | Result | Evidence |
|---|---|---|
| Every world retains seven candidates | PASS | All five seeds and all splits report exactly seven |
| World splits are disjoint | PASS | All five seeds |
| History-free Top-1 is near 1/7 chance | **FAIL** | Action + target reaches 0.7480 ID Top-1 |
| Ordered exceeds analytic by at least 0.05 on ID | PASS | +0.0611 mean pair-order |
| Ordered exceeds analytic by at least 0.03 on parameter OOD | PASS | +0.0466 mean pair-order |
| Ordered exceeds unordered by at least 0.05 on ID | **FAIL** | +0.0011 mean pair-order |
| Ordered has lower ID regret than analytic | PASS | 0.0115 versus 0.0959 |
| Declared zero-training rules do not reach ordered ID | PASS | Best declared rule 0.8871 versus 0.9850 |

The ID ordered-minus-unordered difference was only 0.0023, 0.0007, 0.0012,
-0.0002, and 0.0018 across the five seeds. This is a stable failure, not a
single unlucky seed. On the held-mechanism split, the best declared rule reaches
0.9150 pair-order, above both the ordered model (0.8401) and selected analytic
model (0.8593).

## Cause

The target is sampled near the outcome of one randomly selected candidate.
Consequently, target displacement and candidate-action magnitude reveal much of
the ranking without any history. In addition, the fixed probe sequence lets
unordered position statistics identify response scale and mechanism well enough
to match the ordered trajectory model. Candidate-ID randomization removes the
old metadata shortcut but does not remove these task-construction shortcuts.

There are two reporting limitations to preserve explicitly:

1. The JSON check named `action_only_top1_near_chance` actually evaluates the
   preregistered action-and-target feature set (`goal_delta`, `action`), not a
   pure action-only model. This stricter history-free control fails by a large
   margin, so the naming issue does not weaken the NO-GO decision.
2. Candidate IDs are independently permuted and excluded from model features,
   but the v3 script did not emit a separate numeric candidate-ID-parity metric.
   This is an implementation omission in the preregistered integrity report and
   must be corrected before a future confirmatory run.

## Required next construction

A v4 generator should be admitted before any learned model is trained:

1. sample targets from fixed strata rather than near a candidate outcome, then
   balance accepted worlds so target/action features cannot identify the winner;
2. randomize probe order and expose ordered action-observation pairs, while
   matching current state, target, and unordered summaries across paired worlds;
3. use stateful mechanisms where probe order changes a latent state and therefore
   reverses at least one candidate preference;
4. run candidate-ID, action + target, current-state, and unordered-history audits
   on generated data first; reject the generator if any exceeds its frozen
   chance/headroom bound;
5. only after those data gates pass, fit the ordered model and evaluate untouched
   ID, parameter-OOD, and held-mechanism tests.

Until that construction passes, increasing model size, adding vision, or running
a larger simulator would measure shortcut capacity rather than the proposed
temporal-dynamics contribution.

## Reproduction

```powershell
python scripts\cpu_timearrow_v3.py `
  --output audit\results\timearrow_v3_cpu_full.json
```
