# Estimator-consistent reporting note for 4R-v3

The 4R-v3 raw predictions, aggregates and decision are frozen. The simulator
and centroid fairness finding are not implicated. The reporting issue is that
the published point estimate is a seed-average metric whereas the stored CI is
computed from an ensemble-score estimator; those CIs must not be cited as
uncertainty for the seed-average values.

Future reporting should use either seed-average point estimates with seed
standard deviation/min--max, or an ensemble-score point estimate with a
counterfactual-pair bootstrap CI. The required raw seed scores are absent from
the frozen artifacts, so this milestone does not recompute or replace numbers.
