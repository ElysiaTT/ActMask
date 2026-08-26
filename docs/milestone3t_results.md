# Milestone 3T results

The manuscript is deliberately a **state-only benchmark and evaluation** paper. Its final quantitative evidence remains frozen: 92,000 executions; GRU minus validation-selected MultiFrameLinearVelocity +0.400 pair-order accuracy; grouped 95% CI [0.3951, 0.4052]; three positive OOD axes; swap accuracy and score-swap consistency 1.0; C5/C10/C20 dynamic Top-1/NDCG 1.0; and C20 p95 0.816 ms.

The literature audit uses verifiable primary sources. The manuscript has complete LaTeX source, generated-table wrappers with provenance, figure references to the frozen 3S package, three simulated review reports, and a release candidate that references rather than duplicates large raw artifacts. Local TeX compilation is unavailable and was not installed.

Verification: `scripts/verify_milestone3t_manuscript.py` passed 45 static integrity checks. The compatible non-CPU-heavy regression command completed with 172 passed and one deselected legacy CPU-only Milestone-2 integration test in 26.98 seconds. The exception remains documented from 3S and is unrelated to manuscript logic.

Final manuscript decision: **B. MINIMAL VISUAL VALIDATION STRONGLY RECOMMENDED**. The frozen state-only benchmark is coherent and no blocking state-only validity issue was identified. However, all three reviewers identify state-only scope as the principal acceptance risk; an authorized minimal visual preflight would materially improve external-review confidence without changing the present claim.
