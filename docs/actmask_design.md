# ActMask milestone-1 design

## Research question and hypothesis

ActMask asks a counterfactual question: **for this observed 3D state and this
specific candidate action, which points will become relevant, contacted, or
affected during the action?** The milestone-1 hypothesis is that current point
geometry alone is insufficient in a dynamic manipulation scene. Conditioning a
point predictor jointly on per-point motion, action direction, and action timing
should distinguish scenes that have the same current geometry but different
futures, and actions that traverse different future swept volumes.

This implementation tests only whether that signal can be represented and
learned in a deterministic toy setting. It does not test general robotic
competence, language grounding, visual perception, or sim-to-real transfer and
does not train a VLA model.

## Three masks that should not be conflated

| Mask | Question answered | Depends on the candidate action? |
| --- | --- | --- |
| Object mask | Which points belong to an object or semantic instance now? | No |
| Motion mask | Which points are moving, or have large motion magnitude? | No |
| Action-conditioned intervention mask | Which point trajectories enter this candidate gripper's swept volume during its execution? | Yes |

An object may be moving but never approached by the action, so an object or
motion mask can be positive while the intervention mask is negative. Conversely,
a stationary point can be intervention-positive when the action reaches it.
`MotionMagnitudeMask` deliberately ignores the action and
`ActionProximityMask` deliberately ignores future point motion; they expose
these two failure modes. `NoMask` assigns equal relevance everywhere.

## Synthetic paired data

`ToyDynamicDataset` generates all samples locally from a fixed seed. Every item
uses explicit `torch.float32` tensors and has this contract:

| Field | Shape/type | Meaning |
| --- | --- | --- |
| `points` | `[N, 3]` float | Current Cartesian point positions |
| `velocities` | `[N, 3]` float | Constant Cartesian point velocities |
| `action` | `[8]` float | Candidate gripper trajectory (schema below) |
| `mask` | `[N]` float | Binary future swept-volume label |
| `success` | scalar float | Whether the candidate intercepts the designated moving target |
| `pair_id` | scalar integer | Identifier shared by two counterfactual variants |
| `variant_id` | scalar integer | Variant 0 or 1 within a pair |
| `scenario` | string | Counterfactual family name |

Pairs share the same current points exactly, both variants have non-empty masks,
and their masks differ. Variant 0 is the successful candidate and variant 1 is
the unsuccessful candidate. Four scenario families rotate through the dataset:

1. `opposite_velocity`: the present geometry and candidate action are held
   fixed while the object's velocity direction is reversed.
2. `different_action_direction`: geometry and velocity are held fixed while the
   action endpoint/direction changes.
3. `different_timing`: the same current geometry, velocity, and spatial action
   path are paired with different action duration.
4. `interception_success`: the same moving target is paired with an action that
   intercepts it and a perpendicular action that misses it.

The dataset length is even so each pair is complete. `pair_id` must be used
instead of assuming loader order during paired evaluation. The fixed procedural
seed makes two datasets with the same constructor arguments element-wise
identical without modifying NumPy or PyTorch global random state.

## Action and ground-truth semantics

The eight action values have fixed units and ordering:

```text
[start_x, start_y, start_z,
 end_x,   end_y,   end_z,
 duration_seconds, gripper_radius]
```

XYZ and radius use the same synthetic Cartesian length unit, duration is in
seconds, and velocity is Cartesian length units per second.

For duration `T > 0`, the gripper center follows a constant-speed line segment
from `s` to `e`:

```text
g(t) = s + (t / T) (e - s),       0 <= t <= T
```

Each point follows the dataset's constant-velocity extrapolation:

```text
p_i(t) = p_i(0) + v_i t.
```

Both trajectories are evaluated at the configured number of uniformly spaced
times, including `t=0` and `t=T`. Point `i` is positive exactly when its minimum
sampled synchronous distance to the gripper center does not exceed the action's
radius:

```text
mask_i = 1[min_k ||p_i(t_k) - g(t_k)||_2 <= radius].
```

The comparison is synchronous: this is not merely distance to the spatial line
segment. Consequently, duration changes where a moving point is when the
gripper reaches a location. `success` describes interception of the designated
target for ranking evaluation; it is sample-level metadata and is not a second
per-point training label. The rule is a discretized geometric proxy for contact,
not a physics simulation of force, occlusion, collision, or deformation.

## Predictors

All predictors share the call signature
`forward(points, velocities, action) -> logits`, with batched input shapes
`[B,N,3]`, `[B,N,3]`, `[B,8]` and raw output shape `[B,N]`.

The learned `ActMaskModel` encodes per-point current position and velocity,
relative point offsets to the action start and end, and broadcast action timing
and radius. A small pointwise MLP fuses those features with an action embedding
and emits one logit per point. It contains no transformer, pretrained backbone,
CUDA kernel, or cross-point attention, which keeps a CPU smoke run small.

The comparison methods are:

- `NoMask`: constant zero logit (probability `0.5`) for every point;
- `MotionMagnitudeMask`: logit
  `(||v_i|| - speed_threshold) / temperature`, independent of the action;
- `ActionProximityMask`: logit `(radius - d_i) / temperature`, where `d_i` is
  the current point's Euclidean distance to the finite start/end line segment.
  It has no learned parameters and ignores point motion and action duration.

Raw logits are intentional. Probabilities are `sigmoid(logits)` and a point is
predicted positive when its probability is greater than or equal to the
configured threshold (default `0.5`).

## Optimization and reproducibility

Training minimizes the mean pointwise binary cross-entropy with logits,
`-[w_pos*y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))]`. The optional positive
weight is the training split's `negative_count / positive_count`, capped by the
configured maximum; validation and test labels do not influence it. Adam or
AdamW, learning rate, weight decay, epochs, batch size, and threshold come from
the YAML configuration. Python, NumPy, PyTorch, dataset, and data-loader seeds
are fixed; data loading defaults to zero workers.

Validation IoU selects the checkpoint, with validation loss breaking an IoU tie.
The checkpoint contains the small model state and configuration metadata and is
loaded with `map_location="cpu"`.
Metrics are computed from actual predictions, serialized to JSON, and never
presented as benchmark claims.

## Metric definitions

Let `TP`, `FP`, and `FN` be counts over thresholded point predictions and binary
targets. The implementation reports:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 TP / (2 TP + FP + FN)
IoU       = TP / (TP + FP + FN)
```

A zero denominator returns zero rather than NaN. Accuracy is not a headline
metric because sparse all-negative masks would make it misleading. Counts are
aggregated before the ratios are formed, so metrics are micro-averaged over
points rather than averaged per cloud.

Paired-scene consistency groups records by scenario and `pair_id`, XORs the two
thresholded predictions to obtain a predicted change mask, XORs their ground
truth masks, and computes the IoU of those two change masks. A pair whose two
change masks are both empty receives 1; the reported score is the arithmetic
mean over pairs. Successful-action ranking accuracy is computed on pairs with
different success labels using the mean predicted point probability as the
sample score. Higher score for the successful candidate receives 1, lower
receives 0, and an exact tie receives 0.5. These diagnostics complement IoU but
do not establish causal validity.

## Visualization

The visualization entrypoint forces Matplotlib's `Agg` backend before importing
`pyplot`, so it works without a display. A saved figure juxtaposes ground truth,
all three analytic baselines, and learned ActMask probabilities for the same
sample. Each 3D panel includes a subsampled velocity quiver and the candidate
action trajectory with start/end markers. A trained checkpoint is loaded on CPU
when present; the README directs users to train before interpreting learned
predictions. Figures are written beneath `outputs/actmask/`.

## Limitations

- The toy labels are produced by a known synchronous constant-velocity
  geometry rule. A parameter-free predictor that implements that exact rule
  can reconstruct the labels, while the three milestone baselines are
  intentionally weaker diagnostics. Learned-model scores therefore validate
  the software pipeline, not superiority over analytic geometry.
- Every action starts at a static distractor center so that the paired masks
  are non-empty, and target/distractor/background points occupy fixed index
  blocks. These construction choices create shortcuts and make the benchmark
  easier than a shuffled scene with optional all-negative actions.
- Points are clean synthetic XYZ samples with known constant velocities; there
  is no RGB, depth noise, occlusion, calibration error, tracking ambiguity, or
  learned scene flow.
- The gripper is a moving sphere on a straight line. Real kinematics, gripper
  orientation, robot links, collision constraints, acceleration, latency,
  compliance, and contact force are absent.
- Uniform temporal sampling can miss contact between samples and introduces a
  discretization boundary around the radius.
- Per-point independent prediction cannot reason about object identity,
  topology, global context, multi-object interactions, or deformation.
- The learned model sees the same procedural distribution used to define the
  labels. Generalization to new geometry, action families, sensors, simulators,
  and robots is unmeasured.
- Pair consistency and successful-action ranking are toy diagnostics; success
  does not encode task completion or downstream reward.
- The default report uses one deterministic split and a fixed probability
  threshold of `0.5`; it does not report threshold-free precision-recall
  curves, per-scenario uncertainty, or multi-seed variation.
- CPU smoke settings favor fast verification, not convergence, ablation quality,
  calibrated uncertainty, or statistical significance.

## Recommended next milestones

1. **Simulator adapter:** preserve the sample/action interface while replacing
   analytic trajectories with timestamped simulator states, collision/contact
   events, richer gripper poses, and held-out object/action splits.
2. **RGB-D ingestion:** add calibrated depth-to-point-cloud conversion, camera
   frame transforms, visibility/confidence features, and sensor-noise tests while
   keeping raw recordings outside source control.
3. **Scene flow:** estimate motion from recent point-cloud or RGB-D history,
   propagate uncertainty into the mask, and compare estimated flow with the
   current velocity-oracle upper bound.
4. **Model capacity:** only after the data/evaluation boundary is stable, test
   neighborhood or sparse 3D encoders, continuous-time collision distance, and
   probability calibration under controlled CPU/small-GPU experiments.
5. **Real robot:** add a read-only shadow-mode adapter first, synchronize camera
   and action clocks, transform everything into a documented frame, and validate
   safety envelopes offline. Any action execution must then pass independent
   workspace, collision, emergency-stop, and human-supervision review.

The key compatibility rule is to keep the public sample dictionary and action
schema stable. Simulator, perception, and robot sources should become adapters,
not reasons to entangle this package with neighboring repositories.
