# Binding benchmark readiness audit

Generated: 2026-09-06T09:30:13.112093+00:00
Overall decision: **GPU_START_NO_GO**

| Layer | Decision |
|---|---|
| CPU G1/G2 transport smoke (archived evidence) | **GO** |
| CPU benchmark harness | **GO** |
| Current learned-method claim | **NO_GO** |
| Public data/source selection | **GO** |
| Frozen GPU protocol | **GO** |
| G1-G4 generator/method modules | **NO_GO** |
| GPU host and authorization | **NO_GO** |

## Evidence

- Targeted CPU tests: `..........                                                               [100%] 10 passed in 5.58s`
- Anti-saturation example: TPA `0.8812`, normalized regret `0.0351`, Top-1 `0.6438`.
- Public source choice: `maniskill` primary, `robomimic` backup; both bounded files passed official LFS SHA and structural inspection.
- Current learned result remains `CPU_V3_NO_GO`; the synthetic harness `METHOD_GO` is only a self-test.
- Local/authorization preflight failures: `platform, packages, maniskill_import, torch_import, cuda_available, nvidia_smi, vram, disk, cost_amount_set, gpu_hourly_price_set, cost_authorized, remote_host_identity_set, remote_host_approved`.

## Interpretation

The reusable CPU benchmark and the public-source decision are ready. Actual GPU
execution is not ready: the current host is unsupported, spending and the exact
remote host are unapproved. CPU G1/G2 is a limited physical transport fixture;
full GPU backend, G3 scientific admission and G4 methods remain pending.
Archived smoke reports are not evidence that the current checkout was rerun.

Next gate: Pull the frozen GitHub commit on Linux, reproduce CPU G1/G2, then pass remote G0 before separately implementing GPU backend and scientific G3/G4.
