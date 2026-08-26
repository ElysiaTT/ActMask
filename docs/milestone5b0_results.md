# Milestone 5B-0 results

The state probe passed all preregistered integrity, moment, candidate-diversity,
artifact, and estimator-consistency checks.

- Moment matching: 128/128 pairs pass with maximum error `0.0` for every
  frozen moment and branch-independent action check.
- C10: 256/256 histories have 5 successes and 5 failures; mixed and 2--8
  fractions are both 1.0, candidate-index majority is 0.50, and template span
  is 0.0.
- Pair-order means: action/current/global/centroid/covariance/speed/ICP and
  unordered controls are 0.50; ordered GRU and TCN are 0.75; both private
  identity/relational oracle diagnostics are 1.00.
- Oracle-minus-best-fair headroom is 0.25. Three-seed estimator summaries and
  bootstrap CIs are derived from saved raw scores; every reported CI contains
  its matching point estimate.
- Storage used: approximately 55 MB. GPU-PhysX candidate execution count:
  2,560, within the 5,000 execution and 2 GPU-hour bounds.

Final decision: `A. RELATIONAL_PROBE_PASSED_AUTHORIZE_5B`. This authorizes the
future Action-Conditioned Relational Dynamics Verifier milestone only. No final
5B verifier or pretrained model was trained in this probe.
