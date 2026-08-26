# TimeArrow-Bench release candidate

This is a documentation-only release candidate for the frozen **state-only** counterfactual dynamics benchmark. It does not publish data, weights, RGB-D assets, or a real-robot system. Large raw artifacts remain in the project output tree and are referenced by manifest.

Quick verification from the repository root:

```bash
/home/tzh/conda_envs/actmask/bin/python scripts/reproduce_milestone3s_tables.py
```

Expected result: a passing artifact manifest and independent metric audit under `outputs/actmask/milestone3s_paper_package/`; no simulator rollout is run. See the dataset card, benchmark card, and reproducibility guide before use.
