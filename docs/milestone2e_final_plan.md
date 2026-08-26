# Milestone 2E-Final Plan (v2)

## Scope and immutability

This is a versioned closure of Milestone 2E.  It writes only under
`outputs/actmask/milestone2e_final_v2/` and these final-v2 documents.  It
does not overwrite 2A, 2B, 2C, coordinate-audit, legacy 2E incremental, or
historical NDCG artifacts.  The legacy draft remains historical evidence only.

## Frozen protocol

The model choice remains the validation-selected fair arm:
`SoftCoordinateInvariantCorrespondence + ScalarInvariantFeatureBackbone`.
Existing checkpoints, candidate worlds, grouped splits, deterministic seeds,
calibration, and labels remain frozen.  Corrected outputs use only
`milestone2e-ranking-v2-standard-ndcg-utility-gain` and never pool that metric
with unversioned legacy NDCG.

## Execution phases

1. Audit the environment, existing artifacts, GPU state, and schemas.
2. Add a provenance-safe fair-baseline cache, with focused cache tests.
3. Create a versioned corrected evaluator that reuses frozen checkpoints and
   evaluates C5/C10/C20/C50 ranking invariance plus ID/OOD comparisons.
4. Run all-validation-promising temporal/shortcut diagnostics and the Phase
   6/7 completion extensions against versioned final artifacts.
5. Run the complete test suite, write the final report, and apply the fixed
   Stage A decision gates.

## GPU policy

Stage A is CPU-first because the project environment is a CPU-only PyTorch
build.  GPU acceleration is permitted only after an equivalence test with the
frozen CPU protocol.  No Stage A run will terminate or interfere with another
process to obtain GPU memory.  Stage B cannot start unless the final Stage A
decision is `GO TO GPU SIMULATOR INTEGRATION`.

### Execution record

The CPU A4 process was started first, then explicitly stopped at the user's
request before it produced a final artifact.  A separate qualification used
the same frozen C20 batch and all five selected checkpoints.  Utility logits
met `allclose(atol=1e-5, rtol=1e-4)` and stable C20 ranking order was identical
for every complete group.  Mask logits differed by up to `2.41e-5`, so this
does not qualify a general mask-evaluation migration.  A4 uses utility logits
only; it was consequently re-run as a clearly labelled GPU shadow/qualification
artifact under `temporal_gpu_qualification/`, never overwriting a CPU artifact.
The CUDA environment required a local `numpy.trapz` compatibility alias for
the identically defined `numpy.trapezoid` integral used by PR-AUC.  The original
CPU statistics code and all pre-existing artifacts were left unchanged.

## Prospective Stage-A decision policy

This policy is fixed before final-v2 ID/OOD, temporal, and completion results
are inspected. The Stage-A branch is `GO TO GPU SIMULATOR INTEGRATION` only
when all of the following hold for the validation-selected primary verifier:

1. At C20, hybrid minus the validation-only selected parameter-free fair
   ranking baseline has a paired 95% CI excluding zero in the favourable
   direction for Top-1 or normalized regret.
2. At least two of the five dynamic OOD axes (velocity, acceleration,
   curvature, observation delay, action delay) have a paired 95% CI excluding
   zero in the favourable direction.
3. Learned-only ranking drops by at least 0.05 Top-1 or rises by at least
   0.05 normalized regret under a causally relevant history/motion
   intervention, and it exceeds action-only, static-geometry-only, and
   candidate-template-only learned-only Top-1 by at least 0.05.
4. For SO(3), axis-permutation, and legal sign-flip views at every candidate
   count, both absolute Top-1 and normalized-regret mean deltas are at most
   0.02 in magnitude.
5. C20 has useful absolute ranking performance: the five-seed C20 hybrid
   mean Top-1 is at least 0.50 and mean normalized regret is at most 0.50.
   This conservative operationalization is recorded here because the supplied
   protocol requires usefulness but gives no numeric threshold.
6. All C20 seed p95 latencies are at most 200 ms; final-v2 leakage/split and
   cache audits pass; and the complete test suite passes.

If the fair analytic comparator is significantly stronger while the remaining
candidate-verification evidence is useful, choose `PIVOT TO ANALYTIC
VERIFIER`. If only the excluded oracle identity diagnostic is clearly useful,
choose `STOP POINT-CORRESPONDENCE LINE`. If temporal dependence, fair
advantage, dynamic-OOD support, invariance, absolute performance, latency, or
tests fail, choose `STOP LEARNED CANDIDATE-VERIFIER LINE`. Use `HUMAN
DECISION REQUIRED` only for a material unresolved ambiguity or external
dependency. No Stage-B action is authorized unless the first branch is
selected.
