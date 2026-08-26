# Milestone 5B-Data-F results

Forensic configuration SHA-256:
`02ecc9f500b64a64e2dbd32bfbaa0e33b265113108f6f5211108e867ce758e69`.
All frozen 5B-Data data, labels, raw scores, checkpoints and configurations
were present and hash-inventoried.

GRU and TCN each achieve pair-order 1.0 on both task mechanisms and both
branches. But checkpoint-only ablations identify an implementation artifact:
the configured red target goal is absent from both saved baseline feature maps.
Further, appearance removal leaves pair-order at 1.0, while a consistent swap
of the two token slots drops both models to 0.0. The source generator emitted
canonical red-first detections, so stable token order directly exposes the
target role. This is not a valid fair target-cue baseline result.

Gap removal changes scores but preserves pair-order 1.0; hiding appearance in
final/middle frames and retaining only its first frame also preserve 1.0. Thus
the saturation is not caused by the continuous gap proxy. Tiny diagnostic GRU
retraining on frozen data remains 1.0 for a held-out gap range but is 0.0 for
leave-one-task, confirming that the IID score cannot establish compositional
generalization.

Final decision: `D. METRIC_OR_IMPLEMENTATION_ARTIFACT`. Recommended next work
is `Milestone 5B-Data-v2: Partial Cue / No Direct Proxy Redesign`: anonymous
independently permuted detections, explicit goal cue in every baseline input,
partial cue visibility across crossing, and valid held-out cue-role/contact
composition splits.
