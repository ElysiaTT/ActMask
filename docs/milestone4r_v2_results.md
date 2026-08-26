# Milestone 4R-v2 results

The preregistered configuration SHA-256 is `f1678a155473f385a9cb3670c104af78a8d0f717726ae3e3989fc7162d31cc46`; frozen 4R remains `8b07ab21338da9bf9d39c8684b9bfb10e348ebe9a0bb4b4d27af17db1af47422`.

The no-render ManiSkill 3 GPU-PhysX preprobe completed 64 base worlds × C10 × two paired branches (1,280 executions). All 128 branch histories had exactly five successes; mixed-success and 2--8-success fractions were 1.0, entropy was 1.0 bit, pair disagreement and branch outcome change were 1.0, all ten candidate templates had a 0.5 rate, and candidate-index-only majority accuracy was 0.5. Decision-state and action equality checks passed.

The full visual regeneration completed 3 × 128 base worlds × C10 × two branches (7,680 executions), covering seven cameras and all eleven partial-observation conditions. Stored output is about 16 MiB, well below 20 GiB; total new candidate execution is 8,960, below 20,000. Exact pair action/current/final/unordered-history/mask/metadata matching, calibration, association, appearance variation, and hidden-field exclusion all pass.

The repaired placement family passes every strict v2 diversity/leakage gate: 256/256 mixed histories, all with five successes, 1.0 pair disagreement, zero template-rate span, and 0.5 candidate-index-only diagnostic accuracy.

The full benchmark nevertheless fails its preregistered all-family gate: `MovingCubeIntercept` has mixed-success fraction 0.5 and template-rate span 0.5; `FixedPhaseRotatingCaptureWindow` has mixed-success fraction 0.0546875. Thus the final decision is **E. BENCHMARK INVALID**. No oracle, fair baseline, or model is evaluated, and Milestone 4M is not authorized.
