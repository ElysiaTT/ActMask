# Milestone 4V results

## Stage A outcome

The initial blocker is superseded. The current node is an NVIDIA GeForce RTX 4090 D (24,564 MiB), driver 580.76.05, with CUDA-capable Torch 2.6.0+cu124, ManiSkill 3.0.1, and SAPIEN 3.0.3. The project filesystem has 39 GiB free, which accommodates the approved 20 GiB pilot ceiling.

The system's default NVIDIA GLX ICD still cannot provide `vkCreateInstance`, but the platform Vulkan guide's EGL ICD configuration does. `/etc/vulkan/icd.d/test_nvidia_icd.json` points to `/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`; its `vk_icdGetInstanceProcAddr(NULL, "vkCreateInstance")` result is non-null. `vulkaninfo --summary` enumerates the RTX 4090 D and NVIDIA driver. With `VK_ICD_FILENAMES=/etc/vulkan/icd.d/test_nvidia_icd.json` and `XDG_RUNTIME_DIR=/tmp`, the ActMask environment successfully reset ManiSkill `PushCube-v1` in `rgbd` mode and rendered a `torch.uint8` frame of shape `[1, 512, 512, 3]`.

## Decision

**Stage A qualified — proceed to the visual-protocol gate.** No benchmark data, visual pair, model, or visual evaluation has yet been generated. The benchmark must explicitly set the validated EGL ICD rather than relying on default ICD discovery, which will still log the broken GLX ICD warning.

## Frozen visual protocol

The renderer-qualified pilot is separately versioned from the frozen state-only
benchmark because exact reconstruction of the frozen 3R-NL-v2 worlds cannot be
proved from its saved state-only artifacts. `visual_protocol.json` fixes the
three procedural task families, 128 worlds × 10 candidates × six history
frames, grouped 60/20/20 base-world split, 128×128 primary camera, two OOD
cameras, all mild/medium/severe corruption levels, fair-input exclusions, and
the 20 GiB/eight-GPU-hour ceilings. Its SHA-256 is
`125f74eeb7ebae9a7074bf43614d982ef395c5ff1bb25b514d9d98c8642f7077`.

## Verification

The original blocked preflight checks are superseded and will be updated to assert the qualified EGL path. Frozen state-only evidence remains unchanged.

## Visual-pilot evidence (separate version)

The separately versioned pilot has 128 base worlds, two shared histories and
ten candidates per world for each of the three task families: 256 histories
and 2,560 GPU-PhysX candidate executions per family. The 7,680 candidate rows
occupy 11.7 MB including the two held-out-camera render sets. The file-level
audit finds exact current RGB/depth and action equality in every pair, distinct
earlier history, grouped split integrity, no segmentation in fair input, and
no candidate-file-order leakage after the deterministic row permutation.

On the no-leakage primary camera, static current-frame and action-only controls
have pair-order 0.500 for all three families. Unordered history reaches 1.000,
0.747, and 1.000; ordered RGB-D GRU reaches 1.000, 0.927, and 1.000 for cube,
moving-window, and rotating-window respectively. The standard GRU remains
near-saturated under most depth noise, quantization, dropout, occlusion, and
appearance conditions. Only burst frame dropout lowers it reliably to roughly
0.48--0.66. Clean-trained GRU is near chance under both preregistered OOD
camera positions (roughly 0.44--0.61 pair-order).

These results do **not** authorize Milestone 4M: unordered and ordered visual
controls are saturated on primary ID in two task families. The active next
audit is to complete the dedicated point-cloud/oracle diagnostic coverage
before assigning the single final 4V decision.

## Final pilot decision

**B. VISUAL BENCHMARK TOO EASY.** The point-cloud control completes the
remaining geometry diagnostic: a depth-backprojected, segmentation-free
point-cloud GRU reaches 1.000, 0.993, and 1.000 pair-order across the three
families. Since both standard RGB-D and geometric point-cloud temporal models
are saturated on primary ID, there is no defensible model-headroom rationale
for Milestone 4M. Burst-frame and camera-OOD drops are retained as robustness
diagnostics for a future *newly preregistered* benchmark, not as authorization
to add model capacity here.

The state-oracle diagnostic is recorded separately and is excluded from every
fair visual comparison. The same procedural task semantics' existing
state-history temporal oracle has matched-counterfactual Top-1 1.000 in all
three task families, confirming that the negative 4M decision is not caused
by state dynamics becoming intrinsically ambiguous.
