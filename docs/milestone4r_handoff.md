# Milestone 4R handoff

## Current stage

Phase 9 stop condition reached — decision E.

## Resume instruction

Set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/test_nvidia_icd.json` and
`XDG_RUNTIME_DIR=/tmp`. The final output is `robust_probe_v2/`, with audit,
runtime/storage report, headroom placeholder, and final decision written. Do
not run its standard baseline/oracle scripts: the preregistered candidate
diversity gate has failed and those results would be uninterpretable. Do not
use or report `robust_probe_v1/`, which is intentionally partial and
non-final.

Any future repair must be a new benchmark version and a newly frozen config,
not a modification of 4R. It must redesign `SignedMovingWindowPlacement` C10
candidate actions so each retained history has both success and failure
outcomes, then regenerate and re-audit before any model evaluation.
