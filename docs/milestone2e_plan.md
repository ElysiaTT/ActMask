# Milestone 2E: Full-Protocol Bottleneck Isolation

## Scope

Milestone 2E is CPU-only. It preserves all earlier outputs and writes only to
`outputs/actmask/milestone2e`. It does not install a simulator, use a GPU, or
change the frozen 2C validation selection.

## Preconditions

Before any model is trained, the scoring audit verifies score direction,
candidate ID/label alignment, availability of successful candidates, stable
tie handling, checkpoint provenance, validation-only calibration, and
synchronized observation/action transforms. Handcrafted candidate-set tests
are the blocking check.

## Factorial study

Correspondence variants are ground-truth identity (diagnostic only), full-3D
mutual NN, full-3D soft mutual assignment, and no explicit correspondence.
Backbones are the original vector model, the scalar invariant model, and an
action-frame-relative invariant model. All learned combinations are trained
from scratch under the same seeds, data, epochs, optimizer, and validation
selection rule. Ground-truth identity is not included in fair deployment
comparisons.

## Evaluation

The study uses C5/C10/C20/C50 candidate sets, grouped paired confidence
intervals, matching diagnostics by condition, rotation/axis/permutation
checks, action/history ablations, candidate-template diagnostics, ranking-head
ablation, OOD tests, and shared-scene C20 CPU timing. A coordinate-invariant
model must also retain absolute ranking quality.

## Decision policy

The final report chooses exactly one of GO, PIVOT TO ANALYTIC VERIFIER, STOP
POINT-CORRESPONDENCE LINE, or STOP CANDIDATE-VERIFIER LINE. Gates are fixed
before test evaluation and are never relaxed after seeing results.
