# Milestone 3S plan and freeze record

Milestone 3S packages existing state-only evidence; it does not run a new experiment. Frozen sources remain read-only under `outputs/actmask/milestone3p_*`, `milestone3q_signed_dynamics`, `milestone3r_robust_signed_dynamics`, `milestone3r_nl_v1`, and `milestone3r_nl_v2`.

The package generator is `scripts/reproduce_milestone3s_tables.py`. It verifies source existence and configuration bindings, hashes the relevant JSON artifacts, independently recomputes headline arithmetic without importing ActMask evaluators, runs red-team checks, then writes only to `outputs/actmask/milestone3s_paper_package/`.

Completion criteria are: manifest PASS, metric audit PASS, no blocking red-team FAIL, traceable tables/figures, paper documents, reproducibility instructions, visual-stage plan only, and a complete non-visual regression result. Visual/RGB-D/Vulkan execution is explicitly out of scope.
