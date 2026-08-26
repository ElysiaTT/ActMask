# Milestone 3R-NL-v2 Plan

## Forensic basis and scope

V1 is frozen.  Its generic full-history reversal was not a physical
counterfactual: it moved the final common synchronization frames to the first
slots.  The v1 forensic audit instead establishes an exact observable map that
reverses only the four earlier branch-identifying frames while preserving the
two common final frames.  V2 uses this predeclared map as its causal operator;
generic reversal remains diagnostic only.

## Frozen protocol

The authoritative configuration is
`outputs/actmask/milestone3r_nl_v2/preregistered_config.json`.  It fixes six
history frames, matching tolerance `1e-6`, a common 20-step GPU-PhysX horizon,
three state-only families, two nonlinear mechanisms, three deterministic
seeds, and ten common candidate actions per base world.

For a pair, `T(h) = [reverse(h[0:4]), h[4:6]]`.  It must map each history to
the other member exactly, while action, current/final state, timestamps, TCP,
visibility/confidence, and static feature remain identical.  Counterfactual
Swap Accuracy measures whether ranks exchange after T; Score-Swap Consistency
tests whether transformed scores equal the paired partner scores.

## Candidate and OOD design

Each branch receives the identical ten action templates: five trajectories
toward each possible signed interception direction, with fixed variations in
start time, endpoint offset and pulse duration.  Simulator execution labels
each candidate separately.  Worlds must have non-identical candidate outcomes,
0.20--0.80 success fraction on at least 80% of accepted worlds, and at least
25% branch-dependent candidate outcome changes; no candidate-specific
threshold or action is allowed.

OOD datasets are independently generated before model fitting: disjoint
damping/frequency values, disjoint execution delays, and a held-mechanism
evaluation that fits only damping/drive then evaluates hysteretic mode memory.
The detailed disjoint ranges and counts are frozen in the configuration.

## Probe and decision rule

The bounded probe is at most 20,000 GPU-PhysX branch executions.  It runs all
specified fair analytic and small learned models.  It can authorize a full run
only after matching, hidden-field, diversity, OOD, swap, ranking and learned
advantage gates pass.  No rendering, packages, assets, models, or simulator
versions are in scope.

## Pre-generation hash

`9dc510bdec94e4715df96f6cc6f960f932d4dd37babea957e0f14325cf98e564`

This value was recorded before any v2 simulator data or learned-model
evaluation.  Any protocol change requires a new version.

## Full-run freeze

After the A probe decision, `full_run_config.json` freezes the separate full
run: 300 ID worlds in each of five valid family--mechanism cells, C20 source
actions with aligned C5/C10 prefixes, and independent physical (two-cell),
temporal (three-cell) and held-mechanism (three-cell) OOD datasets.  It uses
92,000 GPU-PhysX branch executions in total, below the 100,000 authorization.
Its hash is recorded before full data generation.

`12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47`
