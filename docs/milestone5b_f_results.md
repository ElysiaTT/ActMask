# Milestone 5B-F results

Frozen forensic configuration SHA-256:
`00f01975947f4a1f36781f1a9c1e9ec45893376f87fe456366d01196d78b7251`.
All required 5B-0/5B source files, checkpoint configurations, labels, and raw
scores were present and hash-inventoried without modifying them.

Both task mechanisms independently score 0.75 for ordered GRU, TCN, and
RelDynVerifier, so the plateau is not a one-task mixture. All three have the
same 120 failed test pair groups (Jaccard overlap 1.0); they succeed on branch
0 and achieve 0.5 Top-1 on branch 1. The oracle diagnostics score 1.0 on both
tasks and branches.

The fair-input audit found 640 exact equivalence classes whose anonymous fair
history and candidate action are byte-identical while their labels disagree.
They contain 1,280 examples. The resulting majority-vote Bayes proxy is
exactly 0.75. In the test split, 120 of 240 pair groups receive exact tied
scores; the frozen tie rule awards 0.5, giving exactly `1 - 120/(2*240) =
0.75`. There is no exact fair-input train/test overlap, and each task has 12
test base worlds and balanced 5/10 C10 labels.

Inference-only saved-checkpoint perturbations show the full method uses its
inputs: action zeroing changes mean score by 2.03 and rankings for every C10
group; relation zeroing changes mean score by 2.43 and rankings for 93.75% of
groups; temporal reversal changes mean score by 2.12. Token-order permutation
is invariant, as intended. Thus ablation plateau equivalence is not an
implementation-collapse explanation.

Every tested private association/identity/role/contact/branch diagnostic
closes the gap to 1.0, while mechanism label alone remains 0.5. The frozen
data aliases these private fields to the same latent branch selection, so this
audit identifies the absent observable association cue but cannot distinguish
which private alias should be exposed.

Final decision: `B. FAIR_OBSERVATION_UNIDENTIFIABLE`. Recommended next work is
`Milestone 5B-Data: Observable Identity / Contact Cue Redesign`; no scale-up is
authorized from this forensic milestone.
