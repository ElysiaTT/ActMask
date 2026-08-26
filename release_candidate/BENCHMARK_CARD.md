# Benchmark card

## Task

Given an ordered observable state history and a candidate action, rank execution-derived success. Counterfactual pair members preserve specified observables while their simulator outcome changes. The final v2 protocol has matched final two frames and valid early-frame pair swapping.

## Regimes

The ladder covers ordinary shortcut-heavy worlds, static matching, signed-order matching, nonlinear final-two matching, and action-diverse C5/C10/C20 candidate sets. Full v2 evaluates ID plus physical-parameter, temporal-delay, and held-mechanism OOD. World modulo determines grouped train/validation/test splits.

## Metrics

Pair-order accuracy, Top-1 success, Top-3 success recall, corrected utility-gain NDCG, normalized regret, swap accuracy, score-swap consistency, and latency are defined in `paper/appendix/metric_definitions.tex`. Higher is better except normalized regret. Equal scores use frozen tie-aware evaluation.

## Frozen evidence

The full config SHA-256 is `12c692d032ba37bdb14761a1c63ff8ceccff72af5f037d936ebaed8748016a47`; 92,000 executions are recorded. Read the independent metric and red-team audits before interpreting scores.
