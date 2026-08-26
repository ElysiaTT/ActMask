# R-Series final report

R-Series moved the project from synthetic evidence to public real-robot logs.
The DROID primary was honestly stopped after R1: the compact release has 46
language-valid trajectories, all with distinct full instructions, and cannot
support the promised repeated-task negative semantics.  Its invalid attempt
was kept rather than repaired in place.

The ASU TableTop backup supplied a compact real-UR5 benchmark with public RGB,
actions, states and object poses.  Strict logged continuation and future object
relation tasks passed all split, history-only, source-semantics and raw-score
audits.  Two formulations were truly evaluated and two independent splits
were retained.

The scientific answer is negative but useful: with simple shortcuts at chance,
standard current-state/proprioception temporal models nevertheless achieve
0.97 validation pair-order on the repeated-task split and 0.972 validation /
0.950 test after holding out goal-object language families.  The benchmark is
therefore saturated, and the method was correctly not trained.

Recommended track: retain the public adapter, strict task builder and baseline
audit as benchmark infrastructure; seek a new real-robot method dataset with
actual alternate-action outcomes or richer contact/object ambiguity before
claiming relational-verifier value.
