# ActMask Milestone 2B — Make Dynamics Necessary

**Final recommendation: GO.** This decision is the conjunction of the fixed acceptance gates below; no gate was changed after evaluation.

## What changed

`TemporalActMask` encodes four noisy observation frames, visible-history motion and confidence, constructs future point hypotheses and the nominal gripper trajectory, fuses time-aligned relative position/distance/velocity/contact features, and predicts point masks plus optional per-time contact logits. Its candidate utility head consumes temporal contact summaries, mask summaries, action embedding, visibility and confidence. The original concatenation model remains available as `ActMaskMLP`.

Training combines mask BCE, contact-time BCE, success BCE, pairwise action ranking, velocity/action counterfactual assignment, and irrelevant-background stability losses. Coefficients are recorded in every checkpoint and the run manifest.

| Loss | Coefficient |
| --- | --- |
| action_counterfactual | 0.25 |
| contact_time_bce | 0.20 |
| irrelevant_stability | 0.10 |
| mask_bce | 1.00 |
| pairwise_ranking | 0.30 |
| success_bce | 0.45 |
| velocity_counterfactual | 0.25 |

## Dataset and information boundary

Every group contains identical-geometry dynamics counterfactuals, five candidate actions per world, and one irrelevant-observation perturbation. Group assignment happens before expansion. The role balance is 10 hard-positive, 4 easy-positive, 18 hard-negative and 16 easy-negative points; labels are recomputed from exact execution trajectories.

Observable fields available identically to learned and fair methods: `points_history`, `visibility_history`, `timestamps`, `estimated_velocity`, `velocity_confidence`, `action_command`, `nominal_action_delay`, `observation_delay`.

Hidden fields used only for label construction and excluded-oracle evaluation: `future_times`, `exact_future_point_trajectories`, `exact_point_velocity`, `exact_point_acceleration`, `exact_gripper_execution_trajectory`, `exact_execution_delay`, `exact_execution_duration`, `exact_contact_matrix`, `exact_contact_state`.

Training-set hard-positive fraction: **0.4709**; hard-negative fraction: **0.4351** (both fixed minimum 0.40).

## Five-seed ID results

TemporalActMask ID AP is **0.9030 ± 0.0038**. The strongest validation-selected fair baseline is **EstimatedTimeAlignedTrajectoryProximity**, with ID AP **0.8781 ± 0.0000**. The hidden oracle reaches **1.0000 ± 0.0000** and is excluded from every fair selection and GO gate.

| Method | AP | IoU | F1 |
| --- | --- | --- | --- |
| ActMaskMLP | 0.8790 ± 0.0172 | 0.7019 ± 0.0158 | 0.8248 ± 0.0109 |
| TemporalActMask | 0.9030 ± 0.0038 | 0.7370 ± 0.0042 | 0.8486 ± 0.0028 |
| TemporalNoHistory | 0.3405 ± 0.0050 | 0.2278 ± 0.0023 | 0.3711 ± 0.0030 |
| TemporalShuffledHistory | 0.4872 ± 0.0013 | 0.4068 ± 0.0019 | 0.5784 ± 0.0019 |
| TemporalNoAction | 0.1477 ± 0.0037 | 0.1714 ± 0.0007 | 0.2926 ± 0.0010 |
| TemporalShuffledAction | 0.2580 ± 0.0015 | 0.2332 ± 0.0040 | 0.3782 ± 0.0052 |
| TemporalNoCounterfactual | 0.9048 ± 0.0033 | 0.7380 ± 0.0043 | 0.8492 ± 0.0029 |
| TemporalNoVisibility | 0.7871 ± 0.0040 | 0.6003 ± 0.0024 | 0.7503 ± 0.0019 |
| TemporalNoUtilityHead | 0.9037 ± 0.0022 | 0.7443 ± 0.0036 | 0.8534 ± 0.0023 |

## Fair baselines and excluded oracle

| Method | Status | ID AP |
| --- | --- | --- |
| CurrentPositionProximity | fair observable-only | 0.3418 ± 0.0000 |
| EstimatedVelocityFutureProximity | fair observable-only | 0.3787 ± 0.0000 |
| EstimatedTimeAlignedTrajectoryProximity | fair observable-only | 0.8781 ± 0.0000 |
| ConstantVelocityKalmanProximity | fair observable-only | 0.8466 ± 0.0000 |
| ExactHiddenTrajectoryOracle | excluded oracle | 1.0000 ± 0.0000 |

## Dynamics and hard cases

Velocity counterfactual matching: **0.8750 ± 0.0000**. Action counterfactual matching: **0.8261 ± 0.0000**. Irrelevant stability: **0.9755 ± 0.0006**.

Hard-positive recall: **0.7777 ± 0.0066**; hard-negative specificity: **0.9595 ± 0.0024**. Scenario-level results are in `aggregate/scenario_summary.csv`.

## Candidate-action ranking

| Scoring method | Top-1 success | Top-3 recall | Pairwise/AUC | Regret |
| --- | --- | --- | --- | --- |
| excluded_oracle_utility (excluded) | 0.9375 ± 0.0000 | 0.9375 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| fair_geometric_score | 0.8542 ± 0.0000 | 0.9375 ± 0.0000 | 0.9381 ± 0.0000 | 0.3771 ± 0.0000 |
| learned_utility_head | 0.8542 ± 0.0147 | 0.9375 ± 0.0000 | 0.9646 ± 0.0031 | 0.2813 ± 0.0641 |
| mask_mass | 0.8542 ± 0.0000 | 0.9375 ± 0.0000 | 0.9257 ± 0.0220 | 0.3654 ± 0.0065 |
| mean_mask_probability | 0.8542 ± 0.0000 | 0.9375 ± 0.0000 | 0.9257 ± 0.0220 | 0.3654 ± 0.0065 |
| time_aligned_predicted_contact_utility | 0.8542 ± 0.0000 | 0.9375 ± 0.0000 | 0.8593 ± 0.0058 | 0.3771 ± 0.0000 |
| top_k_mask_confidence | 0.8542 ± 0.0000 | 0.9375 ± 0.0000 | 0.9434 ± 0.0096 | 0.3771 ± 0.0000 |
| utility_fallback_no_head [TemporalNoUtilityHead] | 0.8542 ± 0.0000 | 0.9375 ± 0.0000 | 0.9159 ± 0.0054 | 0.3771 ± 0.0000 |

## Required OOD comparisons

The fair comparator is frozen from ID validation: **EstimatedTimeAlignedTrajectoryProximity**. Positive deltas favor TemporalActMask.

| OOD dataset | Condition | Temporal AP | Fair AP | Delta |
| --- | --- | --- | --- | --- |
| ood/unseen_acceleration | nonlinear | 0.8021 | 0.8000 | +0.0020 |
| ood/unseen_action_delay | delayed | 0.8723 | 0.8374 | +0.0348 |
| ood/unseen_curvature | nonlinear | 0.8354 | 0.8104 | +0.0250 |
| ood/unseen_hard_case_combination | nonlinear | 0.6167 | 0.5783 | +0.0385 |
| ood/unseen_noise_level | noisy | 0.5439 | 0.5276 | +0.0163 |
| ood/unseen_object_count | nonlinear | 0.8963 | 0.8804 | +0.0159 |
| ood/unseen_observation_delay | delayed | 0.8297 | 0.7975 | +0.0321 |
| ood/unseen_occlusion_rate | occluded | 0.7152 | 0.6504 | +0.0647 |

All nine held-out axes and ID-minus-OOD gaps are recorded in `aggregate/mask_summary.csv` and `aggregate/id_ood_gaps.csv`; clean/noisy/occluded/delayed/nonlinear rows are in `aggregate/condition_summary.csv`. Each held-out range is paired with its relevant stress scenario, so axis-level gaps are compound domain shifts and are not claimed as single-factor causal effects.

## Fixed gates

| Gate | Observed | Requirement | Result |
| --- | --- | --- | --- |
| action_removal_ap_drop | 0.7553 | >= 0.0500 | PASS |
| action_shuffle_ap_drop | 0.6450 | >= 0.0500 | PASS |
| history_removal_ap_drop | 0.5626 | >= 0.0500 | PASS |
| history_shuffle_ap_drop | 0.4158 | >= 0.0500 | PASS |
| velocity_counterfactual_matching | 0.8750 | >= 0.8000 | PASS |
| action_counterfactual_matching | 0.8261 | >= 0.8000 | PASS |
| irrelevant_perturbation_stability | 0.9755 | >= 0.8500 | PASS |
| top1_improvement_over_previous_full | 0.3542 | >= 0.1500 | PASS |
| temporal_beats_fair_baseline_on_required_ood_subset | 0.0647 | > 0.0000 | PASS |
| oracle_excluded | 1.0000 | == 1.0000 | PASS |
| all_cpu_tests_pass | 1.0000 | == 1.0000 | PASS |

## Reproducibility and tests

Seeds: 1401, 2403, 3407, 4411, 5413. Device: CPU. Total experiment runtime: **264.82s**; summed learned-training runtime: **168.53s**. Test result: **PASS — 75 passed in 38.26s**.

The run manifest stores configuration and split digests, environment, timestamps, checkpoint provenance, fixed thresholds, and completion status. Visual selections are deterministic and logged; a missing qualifying failure is rendered as a truthful placeholder rather than invented.

## Recommendation

All fixed gates pass. Milestone 2B is GO for completing the synthetic dynamics study, but this does not by itself authorize simulator integration.
