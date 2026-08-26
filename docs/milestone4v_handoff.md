# Milestone 4V handoff

## Current stage

Stage I — separate visual-pilot evaluation. Renderer, deterministic RGB-D, and
camera OOD qualification pass; final 4V decision remains pending only the
point-cloud/oracle diagnostic coverage.

## Work completed

Recorded the qualified EGL ICD, procedural RGB-D/point-cloud smoke, frozen
visual protocol, separate reconstruction boundary, 128-world/three-family
visual pilot, deterministic corruption matrix, two-camera OOD renders, clean
and OOD fair controls, and explicit leakage audit.

## Blocker and evidence

There is no renderer blocker. The initial GLX ICD remains broken, but the
platform EGL ICD at `/etc/vulkan/icd.d/test_nvidia_icd.json` qualifies Vulkan
and SAPIEN. Every visual run must set that file via `VK_ICD_FILENAMES` and set
`XDG_RUNTIME_DIR=/tmp`.

## Attempts already made

1. Qualified EGL ICD, 128x128 RGB-D and point cloud with fixed-seed repeat rendering.
2. Generated the separate 7,680-candidate visual pilot under a frozen protocol.
3. Detected pair-member file order leakage, deterministically permuted rows and labels together, regenerated, and verified first-member success is 0.465--0.520.
4. Audited pair equality/history difference/split/segmentation/storage and ran static, unordered, ordered, corruption, and camera OOD controls.

## Human authorization required

No external authorization is currently required. Do not begin 4M: the
no-leakage visual controls are already saturated on primary ID in two families.

## Exact resume instruction

```bash
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/test_nvidia_icd.json
export XDG_RUNTIME_DIR=/tmp
/home/tzh/conda_envs/actmask/bin/python -m actmask.experiments.milestone4v_visual_audit
```
