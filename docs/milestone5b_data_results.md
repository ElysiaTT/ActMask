# Milestone 5B-Data results

Frozen configuration SHA-256:
`296a68818f531e8ff9e51e7f36b1080dd3e281c10029fb60d50fd7f1931a80f1`.

The repair adds persistent red/blue appearance markers, a visible red target
goal cue, and (for contact) a visible geometry-gap proxy. Marker multiset,
goal, all global moments, and candidate actions are exactly matched across the
paired branches; only the marker-to-trajectory association differs before the
terminal positional merge. The 128-world, 256-history, C10 no-render probe
uses 2,560 ManiSkill 3 GPU-PhysX candidate executions.

Moment/cue matching and C10 diversity both pass. Fair input has zero exact
label-conflicting equivalence classes and a Bayes proxy of 1.0. Cue-only,
current-only, and unordered cue/state controls each have Bayes proxy 0.5.
Action/current/global/covariance/speed/ICP shortcuts all score pair-order 0.5.

However, ordered GRU and temporal convolution both score 1.0, exactly matching
all oracle diagnostics. Thus the repair fixes the former unidentifiability but
makes marker-to-trajectory tracking too easy for standard temporal controls;
there is zero oracle-versus-standard headroom.

Final decision: `F. TASK_TOO_HARD_OR_NO_HEADROOM`. Milestone 5B-v2 is not
authorized. A later benchmark redesign must retain an observable cue while
making its temporal association nontrivial for standard temporal controls.
