# Milestone 5B plan

5B trains a new, small, no-pretraining `RelDynVerifier` exclusively on the
frozen 5B-0 fair state probe. The frozen configuration is stored alongside the
new outputs before any training. It permits only anonymous object-token
histories, finite-difference velocities, candidate actions, and timestamps;
object/target/branch/mechanism IDs, labels, future state, candidate index, and
oracle fields are excluded from model inputs.

The method explicitly encodes both directed object-pair relations at every
timestep, conditions those relation tokens on a learned action embedding, uses
a GRU temporal memory, mean-pools the two directions, and scores the candidate.
Training uses BCE, within-C10 listwise positive mass, and permutation-score
consistency. Full training uses seeds 17/29/43; four minimal ablations use seed
17 under the same artifact contract.
