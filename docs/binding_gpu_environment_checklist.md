# Binding GPU remote-environment checklist

Current status: **not approved / do not start**

Run `scripts/check_binding_gpu_readiness.py` on the proposed Linux GPU host.
Every machine-generated check must pass, then manually attach the remaining
evidence below to the run manifest.

## Host and runtime

- [ ] Dedicated Linux x86_64 host; not Windows and not WSL.
- [ ] Exact hostname/provider/region/instance type recorded.
- [ ] `nvidia-smi` output archived with GPU model, VRAM and driver.
- [ ] At least 12 GiB VRAM for state pilot and 50 GiB free work-volume space.
- [ ] Python 3.11 virtual environment is isolated from the current Windows env.
- [ ] CUDA-enabled PyTorch imports and reports the intended GPU.
- [ ] ManiSkill 3.0.1 imports; `pip freeze` and package hashes are archived.
- [ ] Single-environment CPU PhysX and vector GPU PhysX smokes both complete.
- [ ] Vulkan is not required for state-only G0–G4. Before visual work, headless
  RGB/depth rendering and Vulkan device selection must be separately tested.
- [ ] No unrelated GPU process is killed or modified.

## Data and replay

- [ ] Pinned PickCube demo SHA-256 matches
  `0e6285c5e7a9293fbb1c10590a1be4fc115792cb816e00b37e9914369a0c3c94`.
- [ ] Download size is printed before download and stays inside the named data
  directory; no `all` dataset command is used.
- [ ] Dataset/code attribution and license files are copied into the artifact.
- [ ] 32 repeated restored-state/action checks meet the preregistered `1e-5`
  tolerance on the selected backend.
- [ ] Hidden mechanism code survives restore but is absent from public
  observations, prediction inputs and candidate IDs.
- [ ] Raw attempted and rejected worlds, not only retained examples, are saved.
- [ ] Train/validation/test twin, episode and scene/object/mechanism group
  overlap is exactly zero.

## Cost and stop controls

- [ ] Provider hourly GPU and storage/egress prices are recorded.
- [ ] User-approved maximum payable amount is written to
  `configs/binding_gpu_readiness.json`.
- [ ] `cost_authorized=true` and `remote_host_approved=true` are explicit;
  silence or available credit is not authorization.
- [ ] State pilot hard stops: 8 GPU-hours, 10 GiB output, 50 GiB minimum free
  disk, or the first failed data-integrity gate.
- [ ] Visual stage remains disabled even if the state pilot passes; it requires
  a new review and has separate 48 GPU-hour/120 GiB limits.
- [ ] Checkpoints and artifacts are written atomically and incomplete runs are
  marked incomplete rather than promoted to final results.

## Command

```bash
python scripts/check_binding_gpu_readiness.py \
  --config configs/binding_gpu_readiness.json \
  --workspace "$PWD" \
  --output audit/results/binding_gpu_readiness_remote.json
```

The command is read-only apart from its JSON report. `GPU_START_GO` means the
named host and cost boundary satisfy preflight; it does not bypass the staged
G1 simulator smoke or G2 dataset admission in the preregistration.
