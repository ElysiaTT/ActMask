# Dataset card

## Summary

TimeArrow-Bench consists of procedural, state-only counterfactual records and execution labels generated in GPU PhysX/ManiSkill. It contains no human data, RGB-D observations, pretrained-model data, or real-robot trajectories.

## Data and schema

Fair model inputs contain observable history, timestamps, visibility/confidence, candidate actions, nominal timing, and TCP state. Labels record simulator execution success. Hidden physical parameters, branch, mechanism, family, simulator object ID, future trajectory/acceleration, and success itself are prohibited fair features. Exact NPZ/JSONL schemas are described in the benchmark card; paths and hashes are in the 3S artifact manifest.

## Intended use

Use for evaluating state-only dynamic action verifiers under static, order-invariant, short-history, and candidate-diversity controls. It is not intended for robot deployment, safety certification, visual perception evaluation, human-subject analysis, VLA training claims, or general-purpose physical reasoning claims.

## Limitations and risks

The task families are procedural GPU-PhysX worlds with a controlled TCP proxy. There is no real robot, RGB-D noise, tracking ambiguity, language, broad embodiment variation, or proof against all simulator-specific structure. Results should not be extrapolated beyond the listed families.
