# Milestone 4R-v3 results

The frozen v3 configuration SHA-256 is `c1011cf8fead7eba68881a151ee9de3df922a9efaab7e4c331bc973d4ebf159f`.

All-family no-render GPU-PhysX preprobes passed on their final preregistered attempt. MovingCube and placement each had 1.0 mixed-success and 1.0 entropy; FixedPhase had mixed-success 0.9765625, 2--8-success 0.9765625, entropy 0.9765625, template span 0.0234375, and candidate-index majority accuracy 0.51171875. All C10 actions were equal across pair branches and all state-replay checks passed.

The clean visual probe completed 3×128 base worlds×C10×two branches, or 7,680 candidate executions. Across the three candidate-grid preprobe attempts and full probe, total new candidate executions were 19,200 (under 20,000); final output was 19,892,676 bytes (18.97 MiB). All pair/current/final/unordered-history/mask/metadata checks, seven-camera calibration, all eleven observation conditions, association diagnostics, appearance variation, grouped splits, and hidden-field/segmentation exclusion passed.

Full diversity passed: MovingCube and placement have 1.0 mixed histories and entropy; FixedPhase has 0.9765625 for both mixed-history gates, 0.975985 entropy, 0.027344 template span, and 0.512891 candidate-index diagnostic accuracy.

Standard controls were then evaluated with three seeds, together with private oracle diagnostics. In the combined held-camera plus dropout regime, action-only/current/unordered RGB-D remain near chance (0.50/0.50/0.48 for MovingCube; 0.50/0.50/0.52 for placement; 0.50/0.50/0.5107 for FixedPhase). Ordered RGB-D GRUs reach 0.6647/0.5367/0.5693 and world-frame point-cloud GRUs 0.86/0.7267/0.7667. However, a fair segmentation-free centroid-velocity control saturates at 1.0 in all three families, matching the clean state oracle (1.0) and leaving model headroom 0.0.

Final decision: **B. ROBUST VISUAL BENCHMARK STILL TOO EASY**. The candidate benchmark is valid, but it is not scientifically appropriate to authorize Milestone 4M.

Derived measures are in `full_probe/derived_headroom_metrics.json`: the world-point-cloud GRU gives geometric-normalization advantages of about 0.19--0.20 over ordered RGB-D, while the centroid-velocity control is the best standard fair method in all families and makes state-minus-best-fair headroom exactly zero. Pair-order bootstrap confidence intervals, Top-1/Top-3, NDCG, regret, ECE, pair-swap, and score-swap values are retained per model and regime in `standard_baseline_report.json`.
