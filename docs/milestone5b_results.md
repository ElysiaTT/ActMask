# Milestone 5B results

Frozen configuration SHA-256:
`02714137ef31f447ab39d865d04290b347dd557cbdaf0b74822f752297b78b0d`.
The source 5B-0 composite data SHA-256 matched the frozen value
`ba9ae19a2854ee796296c715c0f7b2326c59c8e2511f2312a370ed1c4ebad36a`.

`RelDynVerifier` uses object-position plus finite-difference velocity encoders,
two directed pairwise relational tokens, candidate-action conditioning inside
the relation encoder, a temporal GRU, symmetric mean pooling, and a candidate
score head. It used no pretrained parameters, rendered no data, and passed the
forbidden-field audit.

On the frozen 5B-0 test split (three seeds), RelDynVerifier pair-order is
0.75 ± 0.00. Top-1, Top-3, pair-swap and score-swap are likewise 0.75; NDCG
is 0.88525 and normalized regret 0.25. Frozen ordered-GRU and TCN baselines
are both 0.75; oracle diagnostics are 1.00. Hence method improvement is 0.00,
and it closes 0.00/0.25 oracle headroom.

The four seed-17 ablations (`no_pairwise_relations`,
`no_action_conditioning`, `no_temporal_memory`, and
`no_permutation_augmentation`) each also scored 0.75. This is consistent with
the intentionally fair-ambiguous half of 5B-0 rather than an oracle shortcut.

Final decision: `B. METHOD_NO_BETTER_THAN_STANDARD_TEMPORAL`. Milestone 5C is
not authorized. All artifacts and CI-consistency checks passed; the negative
decision is solely performance/ablation based.
