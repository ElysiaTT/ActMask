# Reproducibility

Requirements: the migrated environment at `/home/tzh/conda_envs/actmask/bin/python`; Torch 2.6.0+cu124; CUDA-capable GPU available but not required for table regeneration. Run:

```bash
/home/tzh/conda_envs/actmask/bin/python scripts/reproduce_milestone3s_tables.py
```

Expected runtime is under one minute without GPU simulation. Expected output includes `artifact_manifest.json`, `independent_metric_audit.json`, `red_team_audit.json`, tables, figures, and decision JSON. The command writes only to the 3S package directory and verifies source bindings. For this release candidate, read paths relatively from repository root; do not duplicate the large `full_raw` artifacts.
