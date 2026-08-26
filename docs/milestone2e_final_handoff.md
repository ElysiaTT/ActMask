# Milestone 2E-Final Handoff (v2)

## Closed decision

Stage A is closed with **STOP LEARNED CANDIDATE-VERIFIER LINE**. Do not begin
Stage B GPU simulator integration from this evidence set.

The selected learned verifier is stable under rigid transforms and has strong
absolute C20 ranking, but it is significantly beaten by the frozen fair
analytic comparator and fails both the two-axis dynamic-OOD gate and the
static-geometry shortcut gate.

## Authoritative new artifacts

- `outputs/actmask/milestone2e_final_v2/environment.json`
- `outputs/actmask/milestone2e_final_v2/evaluation_schema.json`
- `outputs/actmask/milestone2e_final_v2/frozen_input_manifest.json`
- `outputs/actmask/milestone2e_final_v2/fair_ranking_baseline_selection.json`
- `outputs/actmask/milestone2e_final_v2/invariance/primary_ranking_invariance.json`
- `outputs/actmask/milestone2e_final_v2/id_ood/primary_id_ood_ranking.json`
- `outputs/actmask/milestone2e_final_v2/completion/completion_extensions.json`
- `outputs/actmask/milestone2e_final_v2/gpu_equivalence/ranking_test_c20_first_batch.json`
- `outputs/actmask/milestone2e_final_v2/temporal_gpu_qualification/all_validation_promising.json`
- `outputs/actmask/milestone2e_final_v2/tests/full_pytest_after_gpu.log`

The temporal report is a GPU qualification/shadow artifact. CPU A4 was
explicitly stopped before it could write an artifact. The report must retain
that label in any downstream write-up; its utility-logit/ranking equivalence
qualification is documented in the paired GPU-equivalence artifact.

## Reuse constraints

- Never overwrite or reinterpret historical 2E incremental/unversioned-NDCG
  outputs as corrected-metric evidence.
- Do not promote the GPU temporal qualification to a general mask-equivalence
  claim: the strict all-output check found mask-logit differences up to
  `2.41e-5`; only A4's utility/ranking path was qualified.
- The `numpy.trapz` alias applied in the CUDA environment is a local API
  compatibility layer for the identically defined trapezoidal integral. It did
  not modify the CPU statistics source or legacy artifacts.
- Do not run Stage B without a newly approved protocol and evidence that
  repairs the fair-comparator, dynamic-OOD and static-shortcut failures.

## If this line is revisited

Treat a new study as a new experiment: preregister a design that prevents
static-geometry dominance, verifies at least two dynamic OOD improvements
against the frozen fair comparator, and repeats CPU--GPU qualification for any
additional output heads it evaluates. Do not select an arm after observing
these final-v2 test results.
