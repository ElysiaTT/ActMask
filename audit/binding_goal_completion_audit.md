# CPU-first binding goal completion audit

Date: 2026-09-03
Scope decision: **CPU/data/design package complete**
Execution decision: **GPU_START_NO_GO**

This audit maps the requested goal to concrete evidence. A completed planning
scope does not mean that GPU execution or a learned-method claim has passed.

| Requirement | Status | Evidence |
|---|---|---|
| Reusable, versioned dataset and prediction schemas | GO | `binding_bench/schema.py`; canonical SHA-256, candidate-ID alignment, matched-branch validation, and train/evaluation digest rejection. |
| External verifier boundary and independent evaluator | GO | `binding_bench/adapters.py`, `metrics.py`, and `evaluator.py`; evaluator truth is removed from method inputs. |
| Integrity, shortcut, and intervention checks | GO | `binding_bench/interventions.py`; complete-pair, binding-breaking, twin-swap, and slot controls with semantic invariants. Metrics report constant/tied shortcuts. |
| Per-seed/per-split statistics and uncertainty | GO | Five-seed rules, difficulty bins, seed-then-twin bootstrap, split leakage audit, and all-admission-split decision in `benchmark.py`. |
| Minimal CPU example and automated tests | GO | `audit/results/binding_bench_example/evaluation.json`; anti-saturated TPA 0.8813, regret 0.0351, binding-break regret +0.4834. Focused suite passes 8 tests. |
| Historical result interpretation | GO | v1 is explicitly a saturation/plumbing diagnostic; v2 and v3 remain `CPU_V2_NO_GO` and `CPU_V3_NO_GO`. No learned improvement is claimed. |
| Official-source data/simulator comparison | GO | `docs/binding_dataset_selection_2026-09.md` covers ManiSkill, RoboMimic, LIBERO, DROID, BridgeData V2, and Open X-Embodiment by counterfactual suitability. |
| Primary and backup bounded CPU probes | GO | Pinned ManiSkill and RoboMimic HDF5 files passed official LFS SHA-256 and structural inspection; temporary downloads were deleted. |
| Frozen GPU experiment and resource/cost boundary | GO | `docs/binding_gpu_preregister.json`, preregistration narrative, environment checklist, and fail-closed readiness checker. |
| Actual simulator replay and G1-G4 implementation | NO-GO | Current Windows host is unsupported for ManiSkill GPU simulation; G1-G4 modules are preregistered but not implemented or represented as completed. |
| GPU host, CUDA, and spending authorization | NO-GO | Local preflight fails Linux, runtime, CUDA, GPU/VRAM, named-host, and cost-authorization checks. |

## Terminal conclusion

The CPU benchmark contract, source selection, bounded format evidence, and GPU
experimental design are ready for handoff. The existing learned verifier has
not demonstrated robust gain. The next authorized action is not a large data
download or training run: it is a named Linux/NVIDIA G0 preflight with an
explicit cost ceiling, followed by implementation and bounded execution of G1
snapshot replay and G2 dataset smoke only.

Reproduce the audit with:

```powershell
python scripts/audit_binding_readiness.py `
  --output audit/results/binding_project_readiness.json `
  --markdown-output audit/binding_benchmark_readiness_report.md
```
