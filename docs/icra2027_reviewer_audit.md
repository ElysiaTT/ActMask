# ActMask / TimeArrow ICRA 2027 simulated reviewer audit

Audit date: 2026-08-09. This review targets the anonymous eight-page manuscript
`paper_icra2027/main.pdf` after the admission-intervention reframing, the
frozen public WAV MiniGrid external audit, and the matched real-robot
re-execution preregistration.

## Bottom line

Current recommendation: **borderline to weak accept**, with high confidence
that the work is technically auditable and moderate confidence that ICRA
reviewers will value a benchmark-admission intervention without a positive
real-robot counterfactual experiment. The reframing improves the paper's first
impression but deliberately does not change the evidence score.

The paper now has a coherent empirical triangle:

1. a positive, execution-grounded controlled benchmark with exact matched
   histories and action-diverse ranking;
2. a positive transition-necessity result on a pinned public verifier
   benchmark; and
3. two informative rejections showing that visual complexity or real logs do
   not automatically authorize a learned-method claim.

The strongest acceptance argument is not “our GRU is better.” It is that the
paper intervenes on the interpretation a benchmark can license and
demonstrates all three possible outcomes: admit a learned comparison, admit
transition evidence but not a new method, and reject a benchmark formulation.
The title, abstract, three-gate definition, and contribution list now present
that intervention before any GRU result.

## Simulated scores

| Dimension | Score | Rationale |
|---|---:|---|
| Technical correctness | 7/10 | Exact source hashes, matched interventions, grouped splits, clustered uncertainty, raw confusion counts, and independent verifiers are unusually strong. |
| Novelty | 6/10 | The novelty is the robot-verifier-specific admission ladder and refusal rule, not counterfactual evaluation in general. |
| Empirical evidence | 6/10 | Strong controlled and public MiniGrid results; visual and UR5 cases are negative by design; no positive matched real-robot re-execution. |
| Robotics relevance | 6/10 | Candidate verification, PhysX execution, RGB-D stress tests, and UR5 logs are relevant, but the clean positive result remains procedural. |
| Clarity | 8/10 | The admit-or-reject thesis and three gates now appear on page one; the manuscript remains dense and asks readers to track several benchmark decisions. |
| Reproducibility | 9/10 | Frozen sources, deterministic builders, generated tables, prediction hashes, and submission audits support independent checking. |

Plausible reviewer distribution:

- Reviewer A (evaluation/robot learning): **6, weak accept**.
- Reviewer B (real-robot manipulation): **5, borderline**, citing lack of matched
  real executions.
- Reviewer C (benchmark/reproducibility): **6–7, weak accept**.
- Expected meta-level outcome before rebuttal: approximately **5–6**, dependent
  on whether the area chair accepts benchmark diagnosis as a sufficient ICRA
  contribution.

## Major reviewer questions and current answers

### 0. Is the contribution only a checklist for failed benchmarks?

No. The unit of intervention is a benchmark's licensed interpretation. The
observable contract fixes legal evidence, execution-derived outcomes prevent
label substitution, the counterfactual ladder removes named alternatives, and
the strongest-fair-control gate determines whether a learned-method comparison
is admissible. The empirical cases exercise distinct terminal decisions rather
than presenting a collection of failures.

Residual risk: benchmark admission is still a protocol contribution. Reviewers
who require a new policy or verifier architecture may not value the refusal
rule as sufficient novelty.

### 1. Is this only a synthetic toy result?

Partly, but no longer only. The main TimeArrow positive is procedural and must
remain described that way. The pinned WAV MiniGrid audit adds an external
published benchmark: paired evidence beats the strongest single-state control
by 0.459–0.476 dynamic accuracy at every tested complexity, while the released
SparseIDM scores 0.940–0.953. The paper does not claim MiniGrid is a robot.

Residual risk: the external task is inverse action inference, not the paper's
binary candidate-success formulation. The manuscript explicitly presents it as
a transition-necessity stress test, not as a direct replication of TimeArrow.

### 2. Why is a GRU contribution interesting if it scores 1.000?

The GRU is intentionally not the method contribution. Its perfect score shows
that the final matched construction retains recoverable temporal signal after
shortcut removal. The relevant comparisons are the final-two score of 0.500,
the validation-selected analytic score of 0.600, the 0.400 clustered effect,
and the valid score-swap intervention.

Residual risk: some reviewers equate perfect scores with a toy benchmark.
Figure 1, five family–mechanism cells, three OOD axes, and the external WAV
audit reduce but do not eliminate this concern.

### 3. Does the external WAV result merely favor a larger model?

No. Current-only, next-only, and paired cells use the same two-slot CNN,
optimization, o6 training data, development split construction, and selection
metric; only the observable placed in each slot changes. A fixed transition
rule reaches 1.000 action accuracy, so the conclusion is that transition
evidence is necessary under the controls, not that learning is necessary.

### 4. Why is the transition-rule dynamic score below 1.000?

The public metric re-executes the decoded action with its lightweight physics
oracle and compares changed channels to logged next states. Even the true
action does not reproduce every logged changed channel, yielding an observed
ceiling of 0.947–0.969. The manuscript states this explicitly and does not
normalize other methods by the ceiling.

### 5. Is the real-robot evidence causal?

No, and the paper's refusal is the point. The positive is the actual next UR5
chunk; alternatives come from other logged contexts and were not re-executed
from the same state. The formulation is rejected before a proposed method is
trained. Calling it a positive real-robot verification result would be a
blocking scientific error.

### 6. Why reject the visual task if temporal networks beat chance?

Because a fair segmentation-free centroid-velocity estimator reaches 1.000
pair-order in all three families. Chance static controls establish temporal
necessity but not learned-method headroom. The separate centroid Top-1 values
also prevent pair-order saturation from being misreported as candidate-ranking
success.

## Remaining weaknesses that cannot be edited away

1. **No positive matched real-robot counterfactual execution.** Existing logs
   cannot repair this. A genuine fix requires hardware or a dataset that
   executes multiple candidate actions from matched states. A three-family,
   36-pair randomized physical cell is now preregistered with exact state/action,
   tracking, safety, outcome, and shortcut-ladder gates, but it is not evidence
   until executed. A separate audit of RRC 2020/2022, REASSEMBLE, Oopsie, and
   IMBench found real execution logs but no released exact same-state,
   same-candidate, different-history pair provenance.
2. **The principal positive identification test is procedural.** Its exact
   construction is a strength for diagnosis and a limitation for ecological
   validity. The abstract, contribution list, section title, and construction
   text now consistently state that it is an identification test rather than a
   natural-robot-data surrogate.
3. **External validation is discrete MiniGrid.** It improves provenance but not
   embodiment realism and does not cover WAV's AFS or robot experiments.
4. **The paper is a protocol contribution, not a new verifier.** Reviewers
   seeking state-of-the-art policy or manipulation performance may score it
   lower despite the explicit scope.
5. **Eight pages are fully used.** Further experiments require replacing, not
   simply appending, material. The current external audit is higher-value than
   an additional weak logged-data table.

## Resolved blocking issues

- Candidate-ranking aggregation is history-keyed; the earlier branch-pooled
  version is superseded and cannot enter generated tables.
- The state confidence interval resamples 300 independent base worlds rather
  than 6,000 correlated candidate pairs.
- Visual confidence intervals with a mismatched estimand are suppressed.
- The visual case has exact RGB-D matching and a generated, hash-checked
  complete-history montage.
- The public UR5 case is labeled logged-continuation identification and is
  explicitly rejected as counterfactual verification.
- The public WAV experiment was preregistered before trained single-state
  controls, pins upstream hashes, stores per-seed confusion counts and
  prediction hashes, and passes an independent verifier.
- The manuscript is double anonymous, uses the official class and frozen BST,
  embeds all fonts, contains no unresolved references or overfull boxes, and
  stays within eight total pages.
- Substantive OpenAI Codex use is disclosed in an anonymous Acknowledgment that
  names the affected sections and manner of use without people, funding, or
  identity-bearing details.
- The next real-robot experiment has a frozen machine-readable protocol and an
  independent verifier. It now repeats the full current/action/unordered/
  final-two/ordered/fair-analytic ladder and explicitly rejects a learned
  method if a legal ordered analytic rule saturates; no unexecuted number
  enters the paper.

## Submission gate

Scientific content is acceptable for a submission candidate when all of the
following remain true after the final clean build:

- claim package: 16 frozen sources, 36 passing invariants, 16 generated outputs;
- WAV external audit: 9 verifier checks and 15 decision gates pass;
- manuscript audit: all checks pass, at most 8 Letter pages, embedded non-Type-3
  fonts, no anonymity leak, no unresolved references, and explicit AI-use
  disclosure;
- two consecutive clean builds produce the same PDF SHA-256;
- the anonymous artifact archive contains only submission-safe files and its
  manifest reproduces every included hash.

No new result should be added after this gate unless it is preregistered,
independently auditable, and strong enough to replace existing page content.
