# Milestone 5B-Data-v2 results

This document records matching, representation, identifiability, baseline and
decision results for the independently versioned v2 partial-cue probe.

V2 passes representation hard gates: explicit goal cue is recorded in all
learned preprocessing maps, token color-slot maximum is 0.512, exact fair
collisions are zero, fair Bayes proxy is 1.0, and cue/current/gap/final/token
slot proxies are all 0.5. C10 and moment/cue/order matching pass over 2,560
GPU-PhysX executions. Yet GRU and TCN both reach 1.0, matching oracle and
leaving zero headroom. Final decision: `F. STANDARD_TEMPORAL_STILL_SATURATES`.
