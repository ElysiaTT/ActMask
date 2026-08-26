# Reproducibility

Use exactly `/home/tzh/conda_envs/actmask/bin/python` (environment prefix `/home/tzh/conda_envs/actmask`). At freeze verification this provides Torch 2.6.0+cu124 with CUDA available on an NVIDIA A16, ManiSkill 3.0.1, and SAPIEN 3.0.3. The repaired links into `/home/tzh/conda_envs/papers` are recorded in the generated manifest and must not be reverted.

No environment variables are required to regenerate the paper package. From the repository root:

```bash
/home/tzh/conda_envs/actmask/bin/python scripts/reproduce_milestone3s_tables.py
```

This is CPU-light and does not run simulation; expected time is under one minute and output is `outputs/actmask/milestone3s_paper_package/`. It validates frozen artifact hashes/config bindings and writes audits, tables, figures, and `milestone3s_decision.json`. The artifact hashes are in `artifact_manifest.json`; expected v2 full config SHA-256 is `12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47`.

Run non-visual tests with:

```bash
/home/tzh/conda_envs/actmask/bin/python -m pytest -q
```

Legacy visual/Vulkan work is intentionally absent and non-blocking for this milestone. Do not install Vulkan, use sudo, generate RGB-D data, or rerun GPU simulation as part of this reproduction.

At the 3S freeze, the non-visual collection contained 171 tests. The compatible complete subset was `170 passed, 1 deselected, 16 warnings in 26.03s`; the exclusion was `test_tiny_two_seed_run_is_deterministic_and_writes_complete_schema`, a legacy Milestone-2 test that intentionally performs a CPU-only two-seed training/rendering integration run. It is not a 3S dependency and is recorded as a non-blocking exception because the active task constraint avoids long CPU execution. Run it separately only if CPU execution is explicitly authorized.
