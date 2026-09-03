# ActMask CPU-first direction review

Date: 2026-09-03

## Recommendation

Pivot the immediate experiment from generic **arrow-of-time classification** to
**counterfactual action-effect binding for history-aware robot verifiers**.

The proposed scientific question is:

> When a verifier ranks candidate actions from interaction history, does its
> decision actually depend on which past action caused which observed effect,
> or can it succeed from the current state, action/result marginals, candidate
> templates, or generic temporal smoothness?

This is narrower than proposing another memory architecture, but it is more
defensible and better aligned with the evidence already in this repository.
TimeArrow-v2 is exactly shortcut-solvable and TimeArrow-v3 remains almost
unchanged after removing history order. In contrast, the existing
History--Action Binding v2.1 symbolic construction has exact matched twins,
chance marginal controls, and a large binding-breaking drop. Its previous stop
was an execution error in the GPU/PhysX stage, not a rejection of the scientific
construction.

## Primary-source literature matrix

| Work | What it establishes | Remaining diagnostic gap | Consequence for ActMask |
|---|---|---|---|
| [HAVE, CoRL 2025](https://arxiv.org/abs/2509.00271) | Separates action generation from a verifier that attends from a candidate action to past actions and their observed flows; evaluates articulated objects, ambiguous doors, and uneven pickup. | The public paper reports current/history architecture ablations and history-length analysis, but no pair-preserving versus action-result-binding-breaking counterfactual control. | Use HAVE's query=proposal, key=past-action, value=past-effect semantics as motivation, but audit whether the binding is necessary rather than reproduce its architecture. |
| [Temporal Verification for Generative Robot Policies, 2026 project](https://hatchetproject.github.io/tev/) | Uses an observation-action temporal token and contrastive energy verifier; reports 6--18% success gains with a small add-on. | Public materials do not report shuffled or matched action-effect binding controls. Temporally mismatched negatives may be separable through generic trajectory smoothness. | A binding-controlled benchmark can distinguish temporal compatibility from genuine interaction identification. |
| [Action-Effect Memory (AEM), 2026](https://arxiv.org/abs/2606.12499) | Interleaves visual/action tokens and uses masked reconstruction to learn a compact action-effect memory; reports gains over single-frame and frame-stacking controls. | The paper motivates action-state causality but does not report a binding-breaking shuffle or matched marginal twin test. | Avoid competing on another memory encoder. Test the causal claim that the remembered action is paired with the correct effect. |
| [RMBench, 2026](https://arxiv.org/abs/2603.01229) | Formalizes memory-dependent manipulation and supplies nine non-Markovian tasks plus modular memory baselines. | Many tasks test recall of hidden task facts or stages, not identification of an unknown action-conditioned response law. | Position binding necessity as complementary to task memory complexity, not as another general memory benchmark. |
| [World Action Verifier (WAV), 2026](https://arxiv.org/abs/2604.01985) | Decomposes world-model verification into state plausibility and sparse inverse-dynamics reachability; uses relative error ranking for exploration. | Sparse inverse verification tests action imprint in a proposed transition but does not test whether a history-aware reward verifier binds multiple past interventions to their effects. | Add a strong sparse inverse/Fourier system-identification witness and report candidate ranking/regret, not only classification. |
| [LIBERO-CF, 2026](https://arxiv.org/abs/2602.17659) | Uses feasible alternative instructions in the same layouts to expose vision-over-language shortcuts; its conditional-minus-unconditional guidance improves counterfactual grounding. | Its intervention concerns language conditioning, not physical action-effect history. | Adopt the dual-branch diagnostic idea: compare full history-conditioned scores with a history-unconditioned branch under matched counterfactuals. |
| [Shortcut Learning in Generalist Robot Policies, CoRL 2025](https://proceedings.mlr.press/v305/xing25a.html) | Connects shortcut reliance to limited within-dataset diversity and fragmentation between robot datasets. | Dataset diversity alone does not prove that a particular causal/history variable is used. | Balance every candidate/template/mechanism cell and use paired interventions, instead of relying only on aggregate OOD success. |
| [TOMATO, 2024/2025](https://arxiv.org/abs/2410.23266) | Proposes multi-frame gain, frame-order sensitivity, and frame-information disparity to audit temporal reasoning. | Frame-order sensitivity can still reflect an arrow/endpoint cue and does not preserve action-effect pair identity. | Generalize the ladder: no-history gain, pair-order invariance, and action-effect binding sensitivity must be reported separately. |
| [VBenchComp, 2025](https://arxiv.org/abs/2505.14321) | Separates language-answerable, shuffled-frame-invariant semantic, and genuinely temporal video questions. | It diagnoses frame order, not interventional action-result correspondence. | Treat shuffled-history performance as a task-partitioning signal, not sufficient proof of dynamics reasoning. |
| [Counterfactual temporal sanity checks, 2025](https://proceedings.mlr.press/v296/rahman25a.html) | Tests temporal models on explicitly distorted data and expects degradation when temporal structure is truly used. | Arbitrary distortion can change multiple factors simultaneously. | Use two surgical distortions: move intact action-result pairs together, or break only their binding. |
| [Dynamics as Prompts, 2024/2025](https://arxiv.org/abs/2410.20357) | Uses interaction history for in-context system identification and reports sim-to-sim and sim-to-real parameter-estimation gains. | Parameter estimation may exploit task-specific trajectory statistics unless histories are counterfactually matched. | Include explicit system-identification baselines and parameter-OOD splits; do not call a learned ranker better if a small analytic fit saturates it. |
| [TuneNet, CoRL 2020](https://proceedings.mlr.press/v100/allevato20a.html) | Demonstrates one-shot residual system identification and OOD parameter estimation from limited observations. | It assumes a parameterized simulator comparison rather than testing verifier-history shortcuts. | Treat fast analytic/residual identification as a mandatory baseline, not an optional ablation. |
| [BISCUIT, UAI 2023](https://proceedings.mlr.press/v216/lippe23a.html) | Shows that agent interactions can identify causal variables under structured mechanism changes. | It targets latent causal representation recovery rather than candidate-action ranking. | Ground the benchmark in intervention identifiability: actions must vary exogenously and their corresponding effects must remain observable. |
| [CausalWorld, ICLR 2021](https://arxiv.org/abs/2010.04296) | Provides manipulation families with shared causal structure and controlled interventions over object/robot factors. | Broad transfer scores do not isolate action-effect binding. | Preserve mechanism-family and parameter OOD axes in later simulation, after CPU admission. |

## Direction ranking

### 1. Counterfactual action-effect binding audit -- highest priority

Why now:

- directly tests a causal dependency claimed or assumed by HAVE, TeV, and AEM;
- existing ActMask v2.1 code already contains matched twins, binding-breaking,
  pair-preserving, slot-permutation, and coordinate-rotation interventions;
- can be validated completely with NumPy/scikit-learn on CPU before any vision,
  robot, or simulator expense;
- produces a publishable benchmark/protocol contribution even if a transparent
  system-identification baseline is strongest.

Main risk: if a nearest-neighbor or fixed Fourier rule saturates every proposed
task, the result supports a benchmark but not a new learned verifier. The CPU
protocol must report this rather than hiding it.

### 2. Sparse forward-inverse disagreement -- second priority

This follows WAV and is scientifically plausible, but a credible experiment
requires both a forward world model and a low-dimensional inverse model plus
under-explored action data. It is less suitable for the first CPU-only step and
is closer to an already published 2026 method.

### 3. General non-Markovian memory policy -- third priority

RMBench, AEM, HAMLET, and recent memory-policy work make this area crowded.
ActMask should use such benchmarks later as external validity, not lead with a
new generic memory module.

### 4. Generic arrow-of-time task -- stop

The repository evidence shows endpoint/action and unordered-history routes can
solve the current constructions. Arrow classification can remain one audit
component but should not be the primary task or claim.

## CPU admission experiment

The next independent experiment should use exact matched twin worlds. Within a
twin, both branches must have:

- identical current public state and target;
- identical candidate actions and candidate-template counts;
- identical past-action multiset;
- identical past-result and past-utility multisets;
- different action-to-result binding and crossed candidate preferences.

Mandatory controls and interventions:

1. class prior, current-only, candidate-ID, action-only, result-only;
2. endpoint/arrow, repeat-last-failure, nearest and k-nearest history lookup;
3. linear, nonlinear/Fourier, and flexible nonparametric system identification;
4. learned marginal/unbound-history and bound-pair models with identical model
   capacity where applicable;
5. pair-preserving history permutation, binding-breaking result permutation,
   twin-history swap, candidate-slot permutation, and coordinate rotation;
6. novel candidate angles/magnitudes, parameter OOD, noise OOD, and a held
   response mechanism.

The primary metric is candidate-conditioned twin preference accuracy after
centering utilities within branch. Ordinary Top-1, pair-order, normalized
regret, calibration, and per-template results remain secondary. This prevents
shared easy candidate magnitudes from hiding failure to distinguish twins.

The dataset is admitted only if marginal controls stay at or below 0.60,
interaction shortcuts stay below 0.90, a preregistered binding-aware witness is
at least 0.90, breaking binding reduces it by at least 0.20, moving intact pairs
changes it by at most 0.02, and all invariance/integrity checks pass. Learned
headroom is a separate gate and must never be inferred merely from dataset
admission.

## Claim boundary

A successful CPU run would support only:

> The matched construction requires access to action-effect correspondence and
> is not solved by the declared marginal, slot, endpoint, or order controls.

It would not establish real-robot performance, visual tracking, general causal
reasoning, or superiority over HAVE/TeV/AEM. Those claims require later external
model evaluation and execution-grounded evidence. Passing CPU gates authorizes
planning that next stage; it does not authorize GPU use.

## 2026-09 design update after the v1/v2/v3 terminal runs

The three CPU runs change the recommendation from “build a better learned
verifier” to “build the missing audit and make existing verifiers pass it.”
This is also the cleaner gap relative to the newest literature:

- [HAVE](https://arxiv.org/abs/2509.00271) explicitly separates an
  unconditional proposal generator from a history-aware verifier. That makes
  candidate ranking the correct external interface for our benchmark; we do
  not need to reproduce its generator.
- [Temporal Verification](https://hatchetproject.github.io/tev/) reports 6–18%
  task-success gains by scoring temporal continuation with a very small
  verifier. The missing question for us is whether its temporal token uses
  action-to-effect correspondence or merely trajectory continuity.
- [Action-Effect Memory](https://arxiv.org/abs/2606.12499) interleaves actions
  and visual history and reports gains over single-frame and frame-stacking
  pretraining. A binding-preserving versus binding-breaking test directly
  checks the causal interpretation suggested by that representation.
- [World Action Verifier](https://arxiv.org/abs/2604.01985) argues that sparse
  inverse reachability can be easier than dense forward prediction and reports
  results on RoboMimic and ManiSkill. A sparse inverse model is therefore a
  mandatory strong baseline, not a learned-method variant to omit.
- [LIBERO-CF](https://arxiv.org/abs/2602.17659) demonstrates that plausible
  counterfactual instructions reveal vision-over-language shortcuts. Our audit
  applies the same counterfactual principle to physical action-effect history,
  without claiming the two tasks are interchangeable.
- [What Are We Actually Benchmarking in Robot Manipulation?](https://arxiv.org/abs/2606.04233)
  identifies shortcut solvability, weak statistical support, creeping
  overfitting, and data-source dependence as separate threats and reports that
  LIBERO and CALVIN fail several diagnostics. This directly supports keeping
  shortcut controls, hierarchical uncertainty, a frozen one-shot protocol, and
  an independent second simulator in the design.

### Concrete direction adjustment

The paper-level contribution should now be:

> A model-agnostic, counterfactual audit that separates action-effect binding
> use from candidate-selection utility under matched physical twins.

It should not be “ActMask is a new state-of-the-art verifier.” v2 and v3 are
negative baselines showing why causal-use evidence and utility improvement must
be reported separately.

The first GPU stage should generate low-dimensional execution truth, not train
a vision backbone. ManiSkill is the primary simulator because its public
trajectories carry restorable states and its GPU backend can branch many
candidates from one snapshot. RoboMimic/robosuite is the independent backup.
Only after state/effect metrics, interventions, replay determinism, split
isolation, and strong system-ID baselines pass should RGB/depth observations be
rendered and a visual history encoder be trained.

### Reviewer-resistant experimental ladder

1. **Data validity:** repeated restored-state rollout agrees within tolerance;
   candidate truth comes from execution; all rejected worlds are counted.
2. **Shortcut validity:** candidate ID, slot, current-only, action-only,
   effect-only, endpoint, and independently sorted/unbound controls remain weak.
3. **Binding validity:** complete-pair permutation is invariant; effect-only
   permutation and matched-twin history swap produce large degradation.
4. **Decision validity:** normalized regret and Top-1 beat nearest, kNN,
   local-linear, Poly2, RBF, sparse inverse, and any public current-state
   analytic baseline with seed-level and bootstrap support.
5. **Transport:** repeat on parameter, sparse-history, observation-noise, and
   held-mechanism splits with calibration reported separately.
6. **External validity:** freeze the code and repeat in robosuite; DROID or
   BridgeData may only provide a logged-real-data consistency appendix because
   they lack counterfactual snapshots.

This ladder answers the original concern about suspicious 0.5/near-1 tables:
those values are now an anti-saturation failure, not a result to optimize for.
