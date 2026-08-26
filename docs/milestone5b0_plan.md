# Milestone 5B-0 plan

This completed probe is isolated at
`outputs/actmask/milestone5b0_relational_probe/`; no frozen prior milestone
output was changed. Its frozen configuration SHA-256 is
`390a71f61e8f6d9c874f29638cf3756e41275e69f9360727bdc2348e25e6fc57`.

It implements two no-render ManiSkill 3 GPU-PhysX label mechanisms:

- `TwoObjectIdentitySwapIntercept`: crossing identical-object trajectories;
  which identity/role is tracked selects the intercept family.
- `CounterMovingRelationalContact`: counter-moving local-contact relation;
  the tracked relational role selects the placement/contact family.

There are 64 base worlds per mechanism, paired branches, C10 actions, 2,560
GPU-PhysX candidate executions, and no rendered images. Exact-zero matching is
audited for current unordered state, centroid/velocity/acceleration,
covariance, pairwise distances, speed multiset, displacement, token multiset,
and candidate actions. Learned runs use seeds 17, 29, and 43 and must persist a
checkpoint, model and preprocessing configurations, data hash, command,
per-candidate raw scores, per-pair scores, and raw-score-derived metrics.
