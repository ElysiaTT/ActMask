# ActMask

ActMask is a research prototype for action-conditioned future masking in
dynamic 3D scenes. The current checkpoint is a deliberately small,
CPU-only synthetic milestone: given points, per-point velocities, and a
candidate gripper motion, it predicts which points an action may affect.

The detailed milestone instructions, tensor contract, limitations, and
reproduction commands are in [README_actmask.md](README_actmask.md).
The current repository state and next-owner actions are in
[HANDOFF.md](HANDOFF.md).
