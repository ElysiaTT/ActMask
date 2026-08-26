# Milestone 5B-Data-v3 results

V3 generated 128 base worlds, 256 histories and 2,560 fresh GPU-PhysX C10
executions. Fair Bayes is 1.0 with zero exact conflicts; cue/current/token-slot
and gap proxies are 0.5. IID/OOD groups preserve complete pairs and C10 groups.
Nevertheless frozen v2 GRU and TCN standard controls evaluated on the new OOD
combinations each obtain 1.0, matching oracle and leaving zero headroom.
Decision: `E. STANDARD_TEMPORAL_STILL_SATURATES_OOD`.
