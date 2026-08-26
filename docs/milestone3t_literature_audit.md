# Verified literature audit

The machine-readable audit is `outputs/actmask/milestone3t_manuscript_rc1/literature_sources.json`. It contains title, author record, year, venue/arXiv identifier, canonical URL, primary-source status, and the supported manuscript use for eleven references. The audit covers VLA/real-robot policies (RT-1), visual action-conditioned future prediction (Visual Foresight), world models (DreamerV3), shortcut learning, invariant/counterfactual evaluation, temporal visual representations, RLBench, CausalWorld, RoboCasa, and ManiSkill3.

The related-work section avoids a false equivalence: visual optical-flow/scene prediction, visual VLA policies, and action-conditioned world models seek models or controllers; TimeArrow-Bench is a state-only evaluation protocol that diagnoses whether a verifier can exploit matched shortcuts. RLBench, CausalWorld, RoboCasa, and ManiSkill3 provide useful benchmark/simulator context but are not claimed to contain this project's exact shortcut controls.

Unresolved items are explicit in the JSON. No source is used to support an unsupported priority, visual-success, or deployment claim.
