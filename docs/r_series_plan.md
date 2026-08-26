# R-Series plan and execution record

| Phase | Status | Evidence |
| --- | --- | --- |
| R0 survey | complete | `dataset_survey.json`, `r_series_dataset_survey.md` |
| R1 DROID adapter | complete pilot | `processed_subset/`; no fabricated language |
| DROID R2 | invalid/frozen | unique instructions and audited fallback; no R3 |
| R1 ASU backup adapter | complete | `processed_subset_asu_tabletop/`, 110/110 audit pass |
| R2 ASU strict tasks | complete | two versioned attempts under `asu_attempts/` |
| R3 baselines | complete | 66 raw-score-backed run artifacts and two CI audits |
| R4 method | intentionally not authorized | saturation gate failed on IID and held-out-object splits |
| R5 decision | complete | `final_decision.json`, `r5_completion_audit.json` |

The main outcome is a valid public real-robot benchmark/audit finding: ordinary
proprioceptive temporal modeling saturates this small logged ASU task, so it is
not suitable for an ICRA method claim for the proposed verifier.
