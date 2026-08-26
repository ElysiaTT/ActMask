# ICRA 2027 submission plan

Status date: 2026-08-09 (Asia/Shanghai)

## Target and hard constraints

- Venue: IEEE International Conference on Robotics and Automation (ICRA 2027).
- Official deadline: 2026-09-15, 11:59 PM Pacific time.
- Initial submission: double-anonymous, IEEE/ICRA two-column PDF.
- Hard length limit: eight pages total, including figures, tables,
  acknowledgments, and references. There is no conventional supplementary
  PDF; an optional video may be at most 180 seconds and 20 MB.
- Official sources:
  - https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/
  - https://ras.papercept.net/conferences/support/tex.php

## Frozen paper identity

Working title:

> **Earning a Dynamics Claim: An Execution-Grounded Admission Audit for Robot
> Action Verification**

The paper is an evaluation-intervention paper. Its contribution is an
auditable procedure that decides which interpretation a dynamic
candidate-action benchmark can license. It is not a new general-purpose
verifier, visual world model, safety system, or robot policy.

The central thesis is:

> A dynamic label and a high held-out score are insufficient evidence of
> temporal reasoning. A benchmark earns a dynamics or learned-method claim
> only by passing three gates: execution-grounded integrity under a declared
> observable contract, temporal necessity after progressive shortcut removal,
> and method headroom against the strongest fair non-proposed estimator.

## Main evidence and claim boundaries

### Main positive result: TimeArrow state-observation benchmark

- 92,000 GPU-PhysX candidate executions.
- Exact matched-pair construction with a valid early-history swap and common
  final two frames.
- Static/action/unordered/final-two controls and validation-selected fair
  analytic estimators.
- Frozen ID pair-order: GRU 1.000 versus MultiFrameLinearVelocity 0.600,
  difference +0.400 with a 300-base-world-clustered 95% interval
  [0.3767, 0.4217]. The earlier candidate-pair interval is superseded.
- Positive physical-parameter, temporal-delay, and held-mechanism OOD
  advantages.
- C20 action-diversity, split, leakage, swap, and latency audits.

Allowed interpretation: the benchmark construction removes the named
shortcuts and creates a controlled state-observation temporal distinction.

Forbidden interpretation: general physical reasoning, visual understanding,
real-robot transfer, safety, or end-to-end policy improvement.

### Visual stress-test case: benchmark admission can fail after RGB-D

Use only the frozen 4R-v3 point estimates together with the corrected 5A
reporting note. Suppress the invalid/non-comparable stored confidence
intervals.

- 3 procedural families, 128 base worlds per family, C10 candidates, two
  matched histories, and 7,680 final candidate executions.
- Current-frame and action-only pair-order remain at 0.500.
- Under combined held-camera plus dropout, ordered RGB-D GRU pair-order is
  0.665/0.537/0.569 and world-frame point-cloud GRU is
  0.860/0.727/0.767 across the three families.
- A fair segmentation-free centroid-velocity control reaches 1.000 pair-order
  in all three families. Therefore this visual construction is an audit
  counterexample, not a claimed hard benchmark or method win.
- Pair-order saturation does not imply perfect C10 action selection; report
  Top-1 separately and do not conflate the two estimands.

Allowed interpretation: the admission audit correctly rejects a visual task
whose matched-pair distinction is solvable by a simple fair geometric control.

Forbidden interpretation: a new visual verifier result, a positive visual
benchmark claim, or a confidence interval for the frozen seed-average point
estimate.

### Public WAV MiniGrid transition-necessity case

- The official upstream repository is pinned at commit `527159b06149beacfb3b2d77af7d938ca4efa32d`,
  with checkpoint, source, and data hashes checked before execution.
- Current-only, next-only, and paired cells use the same CNN architecture,
  optimization, development split, and five seeds.
- Paired evidence exceeds the strongest single-state control by
  0.459--0.476 official dynamic accuracy across o6/o8/o10/o12/o14.
- Released SparseIDM scores 0.940--0.953; a fixed transition rule has 1.000
  action accuracy and exposes the public oracle/log ceiling.

Allowed interpretation: the pinned MiniGrid inverse-dynamics split requires
the observed transition under the matched controls.

Forbidden interpretation: WAV Action Following Score, WAV robot results,
real-robot verification, or a new verifier architecture.

### Public real-robot log admission case

- The public ASU TableTop scope check uses 84 real-UR5 episodes, 11 official
  action--object tasks, and 1,151 C5 logged-continuation candidate sets.
- The held-out goal-object split keeps action/task/language/source controls at
  or below 0.534 validation pair-order, but a standard proprio-action GRU
  reaches 0.972 validation and 0.950 test.
- Alternate candidates are existing logged chunks, not actions re-executed
  from a matched decision state. The task is therefore rejected for both
  missing counterfactual execution grounding and standard-baseline saturation.

Allowed interpretation: the admission protocol identifies why a real-robot
log formulation cannot support a counterfactual action-verification claim.

Forbidden interpretation: real-robot success prediction, matched physical
counterfactuals, deployment, or real-robot transfer.

### Evidence kept outside the main result table
- VERIFIER-IID-LEARNABILITY-DIAG-V1: complete, audited negative diagnosis
  (`VILD_CAPACITY_OR_ARCHITECTURE_FAILURE`) on deterministic construction
  labels. It is a development stop condition, not physical validity evidence.
- RoboMIND/RM branches: several correctly terminated pilots, but including
  them would dilute the eight-page story and mix incompatible task semantics.

## Completed artifact work

1. The deterministic ICRA builder checks 16 source reports, 36 scientific
   invariants, the corrected raw-ranking archive, and 15 external-WAV gates.
2. The manuscript uses the official `ieeeconf` US-Letter format and fits all
   claims, acknowledgments, and references in eight pages.
3. The visual history montage, audit figure, tables, claim ledger, and public
   external audit are source-hash checked.
4. The 23-check manuscript verifier covers pages, fonts, references, log
   cleanliness, anonymity, claim boundaries, and the specific anonymous AI
   Acknowledgment required by ICRA 2027.
5. The anonymous archive is deterministic, hash-manifested, and passes its
   independent anonymity/path scan.
6. No optional video is currently needed for the paper's frozen claims.

## Experiment matrix and stopping rules

| Question | Evidence | Completion gate | Stop condition |
|---|---|---|---|
| Do ordinary dynamic labels permit shortcuts? | Historical ladder controls | Static/action or unordered controls expose at least one shortcut-heavy stage | Do not present historical stages as final benchmark evidence |
| Does final TimeArrow require earlier temporal evidence? | Final-two, valid pair-swap, score-swap, analytic controls | All frozen causal/audit checks pass and generated values match source hashes | Any mismatch blocks the claim |
| Are outcomes action-dependent? | C20 mixed outcomes, branch-dependent changes, Top-k ranking | Diversity and index/template audits pass | Degenerate candidates block ranking claims |
| Does the result persist off the ID cell? | Physical, delay, held-mechanism OOD | Report all frozen OOD axes, including the weaker held-mechanism result | No broader OOD or real-world wording |
| Is the visual version method-ready? | RGB-D, point-cloud, centroid controls under combined perturbation | Report fair-control headroom and separate pair-order from Top-1 | Centroid saturation means no method-training claim |
| Does a public verifier split require a transition? | Matched same-CNN single-state/paired cells, SparseIDM, fixed transition rule | Every preregistered o6--o14 gate passes | No inference beyond the pinned MiniGrid split |
| Can three physical families support matched re-execution? | Frozen 36-pair target-history swap, family-fixed candidate actions, execution-feedback checks, and six fixed ladder estimators | Every state/action/tracking/statistical/safety/shortcut and family-wise gate passes | Until execution, retain the UR5 log rejection; analytic saturation rejects learned headroom |
| Are statistics comparable? | Frozen group bootstrap and 5A correction | State CI binds to its displayed estimand; visual CI stays suppressed | Any estimator mismatch blocks CI reporting |

No new verifier-method training is authorized by this plan. The only open
scientific addition is the preregistered matched physical cell; it enters the
paper only if every frozen gate passes and replaces, rather than appends to,
the current UR5 paragraph and row.

The public-data feasibility audit in
`docs/icra2027_public_real_reexecution_feasibility.md` rejects approximate
pairing from RRC, REASSEMBLE, Oopsie, and IMBench because none currently proves
the same-state/same-candidate/different-history contract. Do not substitute a
nearby logged episode for physical re-execution.

## Eight-page allocation

| Component | Target pages |
|---|---:|
| Abstract + introduction | 0.9 |
| Related work | 0.5 |
| Problem and audit protocol | 1.5 |
| Controlled identification construction | 1.1 |
| State results and audits | 1.3 |
| Visual admission stress test | 1.0 |
| Limitations, conclusion | 0.5 |
| References | 1.2 |

## Schedule

- Aug 9: freeze the current eight-page anonymous submission candidate,
  deterministic archive, hashes, and matched-robot preregistration.
- Aug 10-20: decide whether suitable robot, actuated target, tracking, and a
  qualified operator are available; dry-run only before counted collection.
- Aug 21-27: if hardware is available, collect the frozen three-family 36-pair cell without
  inspecting intermediate outcomes; otherwise do not reopen the paper for a
  weaker substitute experiment.
- Aug 28-Sep 3: three independent paper reviews and one revision pass. Insert
  the physical cell only if its verifier passes every frozen gate and it can
  replace the UR5 log paragraph without exceeding eight pages.
- Sep 4-9: final PDF compliance, anonymization, and optional first video
  window.
- Sep 10-14: submission buffer; no new scientific branch.
- Sep 15: submit before the official deadline.

## Submission-ready definition

The project is ICRA-submission-ready only when all of the following are true:

- the paper is at most eight pages in the official format;
- every displayed number is regenerated from a frozen source with a recorded
  path and hash;
- the visual CI issue is explicitly suppressed rather than silently reused;
- all source, split, leakage, pair validity, and candidate-diversity checks
  referenced in the paper pass;
- a clean reproduction regenerates the tables and manuscript PDF;
- the PDF has no author identity, unresolved citation/reference, missing
  figure, Type-3 font, or formatting error;
- the final claim checklist contains no visual-method, real-robot, safety, or
  policy-improvement overclaim.
