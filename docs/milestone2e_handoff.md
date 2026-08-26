## Current phase

Phase 2 is the last fully implemented phase.  The Phase 0 scoring/indexing
audit artifact exists and reports `passed: true`; all four correspondence
paths and the three backbone paths are implemented.  Phase 3 was started but
only the diagnostic oracle/vector arm has persisted C5, C10, and C20 results.

## Blocker

Human-intervention Condition E applies: the requested frozen factorial
evaluation exceeds the four CPU-hour budget before the remaining required
arms, C50 evaluation, temporal diagnostics, OOD evaluation, invariance
evaluation, and latency reporting can be reached.  The task contract is also
explicitly CPU-only, so substituting a GPU is a protocol change rather than a
permitted continuation.

The current shell lacks the historical CPU Python test environment, so the
existing implementation cannot be re-executed here without installing the
declared dependencies.  Automatic installation/download is prohibited by the
task contract.

## Evidence

- Budget measurement command:
  `python3 - <<'PY' ... outputs/actmask/milestone2e/factorial_incremental ... PY`
  reported 15 persisted seed evaluations from 2026-07-15 11:32:56 through
  14:40:30: 187.57 minutes for one correspondence/backbone arm at C5, C10,
  and C20.  Linear projection is 8.34 CPU-hours for the minimum eight primary
  arms before C50 and all later diagnostics, or 12.50 CPU-hours for the twelve
  arms the existing runner enumerates.
- Persisted files:
  `outputs/actmask/milestone2e/factorial_incremental/GroundTruthIdentityCorrespondence__OriginalVectorBackbone__C{5,10,20}__seed{1601,2603,3607,4611,5613}.json`.
  No other factorial-arm result is present, and no final summary,
  OOD, temporal-corruption, invariance, or latency artifact is present.
- The frozen configuration at `configs/actmask/milestone2e_cpu.yaml` fixes
  CPU execution (`device: cpu`), five seeds, 500 test groups, C5/C10/C20/C50,
  and one training thread.
- Failing verification command:
  `python3 -m pytest -q tests/test_milestone2e.py`
  exits with `/usr/bin/python3: No module named pytest`.  The same interpreter
  also has no installed PyTorch package.  No install/download was attempted.
- `nvidia-smi -L` found no usable NVIDIA device; PyTorch CUDA availability
  therefore cannot be established in the absent project runtime.  Regardless,
  the frozen configuration prohibits CUDA use.
- `docs/milestone2e_results.md` claims a final branch, but its cited final
  JSON/CSV artifacts are absent.  Treat that document as an unverified
  historical draft; it has not been overwritten.

The smallest possible reduced run cannot preserve the frozen decision
protocol: the bundled `--smoke` path changes seeds, group counts, epochs, and
timing repetitions, and writes into the same 2E output directory.  Using it
would both create a non-comparable result and risk overwriting historical
artifacts.  No dataset reduction was therefore run.

## Attempts already made

1. Reused the persisted incremental factorial artifacts to calculate the
   conservative CPU projection.  It exceeds the explicit budget even before
   C50 and later phases.
2. Attempted the repository's blocking 2E test command with the available
   interpreter.  It stopped before collection because `pytest` is absent; no
   dependency installation or speculative code change was made.

## Work completed before stopping

- Read and applied the Milestone 2E CPU-only execution contract.
- Inspected the frozen 2E configuration, complete 2E implementation, tests,
  persisted scoring audit, training records, checkpoint inventory, and
  incremental factorial results.
- Verified that the scoring audit records higher-is-better ranking,
  candidate-ID-independent tie handling, transform-label preservation, stale
  checkpoint rejection, and candidate availability for C5/C10/C20/C50.
- Verified the persisted implementation exposes
  `GroundTruthIdentityCorrespondence`,
  `CoordinateInvariantMutualNN`,
  `SoftCoordinateInvariantCorrespondence`,
  `NoExplicitCorrespondence`, and all three required backbones.
- Preserved all existing 2A/2B/2C/2E artifacts and added this handoff only.

## Human decision required

1. **Accept the CPU-budget stop and keep the milestone incomplete (recommended).**
   Scientific consequence: no new final 2E scientific claim is made; the
   existing draft conclusion remains unverified.  Engineering cost: none.
   This is recommended because it follows the frozen CPU-only contract and
   its explicit four-CPU-hour cap.

2. **Authorize a revised CPU budget and restore the declared CPU test environment.**
   Scientific consequence: preserves the original CPU protocol if all final
   artifacts are regenerated and the historic draft is superseded with clear
   provenance.  Engineering cost: at least 8.34 projected CPU-hours for the
   required eight-arm C5/C10/C20 matrix, plus C50 and later diagnostics; the
   twelve-arm runner projects at least 12.50 CPU-hours before those additions.

3. **Approve a new GPU-enabled protocol as a separate milestone.**
   Scientific consequence: GPU timing, training behavior, and reproducibility
   become a new protocol; all fair comparisons must be rerun and cannot be
   combined with the existing CPU artifacts.  Engineering cost: provision a
   supported GPU and CUDA runtime, update the protocol/configuration, and
   regenerate outputs.  This is not recommended for the current task because
   it explicitly prohibits GPU use.

## Exact resume command

After choosing option 2 and restoring the declared CPU environment, resume
the compatible full protocol with:

`python3 -m actmask.experiments.milestone2e_full`

After choosing option 3, first provide an approved GPU protocol and a new
output location; do not run the command above against the frozen CPU outputs.
