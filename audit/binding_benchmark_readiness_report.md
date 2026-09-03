# Binding benchmark readiness audit

Generated: 2026-09-03T14:20:35.282558+00:00
Overall decision: **GPU_START_NO_GO**

| Layer | Decision |
|---|---|
| CPU benchmark harness | **GO** |
| Current learned-method claim | **NO_GO** |
| Public data/source selection | **GO** |
| Frozen GPU protocol | **GO** |
| G1-G4 generator/method modules | **NO_GO** |
| GPU host and authorization | **NO_GO** |

## Evidence

- Targeted CPU tests: `........                                                                 [100%] 8 passed in 8.00s`
- Anti-saturation example: TPA `0.8812`, normalized regret `0.0351`, Top-1 `0.6438`.
- Public source choice: `maniskill` primary, `robomimic` backup; both bounded files passed official LFS SHA and structural inspection.
- Current learned result remains `CPU_V3_NO_GO`; the synthetic harness `METHOD_GO` is only a self-test.
- Local/authorization preflight failures: `platform, packages, maniskill_import, torch_import, cuda_available, nvidia_smi, vram, disk, cost_amount_set, gpu_hourly_price_set, cost_authorized, remote_host_identity_set, remote_host_approved`.

## Interpretation

The reusable CPU benchmark and the public-source decision are ready. Actual GPU
execution is not ready: the current host is unsupported, spending and the exact
remote host are unapproved, and the preregistered G1-G4 simulator modules have
not yet been implemented. Planned commands are not completion evidence.

Next gate: Approve a named Linux/NVIDIA host and cost ceiling, pass remote G0, then implement and execute only the G1 snapshot smoke and G2 bounded dataset smoke.
