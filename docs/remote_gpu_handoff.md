# Remote GPU handoff

Date: 2026-09-03
Repository focus: counterfactual action-effect binding audit
Current compute decision: **GPU_START_NO_GO**

2026-09-06 addendum: CPU G1/G2 source and Ubuntu CI are available; follow
`docs/binding_cpu_g1_g2.md` to reproduce them after pulling GitHub. The user
will choose and operate the Linux machine; this change does not perform SSH
migration. G3/G4 and GPU backend implementation remain pending.

## What should move through Git

Commit source code, configs, documentation, focused tests, and the curated JSON
evidence listed in `configs/binding_remote_manifest.json`. The migration audit
computes SHA-256 digests for this manifest so the remote checkout can be
compared with the reviewed local state.

Do not copy a Windows environment, cache, checkpoint, HDF5 dataset, video,
private key, or generated intervention suite. Those are ignored. Recreate the
Python environment on Linux and regenerate suites from the committed scripts.

The example's `evaluation.json` is curated evidence. Its `suite/` and
`predictions/` directories are intentionally not transported because their
NPZ payloads are generated and ignored; committing only their JSON manifests
would create broken artifacts.

## Local pre-transfer verification

From the repository root:

```powershell
python -m pip install -r requirements/binding-cpu.txt
python -m pytest tests/test_binding_bench.py -q
python scripts/check_remote_migration.py `
  --manifest configs/binding_remote_manifest.json `
  --output audit/results/remote_migration_audit.json `
  --markdown-output audit/remote_migration_report.md `
  --run-tests
```

`REMOTE_LAYOUT_GO` means the repository layout is transportable. A dirty Git
worktree still produces `GIT_TRANSFER_NO_GO`: review and commit the intended
files before relying on `git clone`. The checker never stages or commits files.

## Create a fail-closed host-specific config

After selecting the host and recording provider, GPU, hourly price, storage,
egress, and maximum payable amount:

```bash
mkdir -p configs/local
cp configs/binding_gpu_readiness.json \
  configs/local/binding_gpu_readiness.json
```

Edit only the ignored local copy. Fill `remote_host.hostname`, `provider`,
`region`, `instance_type`, `expected_gpu_model`, `gpu_hourly_price`, and
`approved_cost_amount`. Set `cost_authorized=true` and
`remote_host_approved=true` only after explicit review. A privately owned host
may record an hourly price of zero, but the field must still be explicit. Keep
the committed template false/null.

## Remote stages

1. Clone the reviewed commit into a named work volume with at least 50 GiB
   free. Never copy the local `.venv`.
2. Run the read-only system checks with
   `bash scripts/remote/g0_probe.sh`; the first run may fail because the
   isolated environment does not exist yet.
3. Choose an exact CUDA PyTorch wheel compatible with the archived driver. Set
   `BINDING_TORCH_SPEC` and `BINDING_TORCH_INDEX_URL`, then explicitly run
   `bash scripts/remote/bootstrap_binding_env.sh`.
4. Run `bash scripts/remote/g0_probe.sh` again. Continue only if it writes
   `GPU_START_GO`.
5. Implement and test G1 snapshot restore/replay. Do not start bulk generation.
6. Run a bounded G2 data smoke. Only admitted data can unlock G3 state
   generation and G4 baselines. Visual training remains a separate decision.

The authoritative thresholds, task families, sizes, stop rules, and resource
ceilings are in `docs/binding_gpu_preregister.json`. Commands in that document
for absent G1-G4 modules must not be represented as completed.

## Post-clone identity check

Run the migration checker again in the remote checkout without `--run-tests`,
then compare `file_sha256` with the reviewed local report. Differences must be
explained by a reviewed commit, never by ad-hoc file copying.
