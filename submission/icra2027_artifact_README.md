# Anonymous ICRA 2027 verification artifact

This compact artifact accompanies the anonymous paper “Earning a Dynamics
Claim: An Execution-Grounded Admission Audit for Robot Action Verification.”
It contains the frozen reports needed to regenerate all paper tables, the
public WAV MiniGrid audit with per-seed confusion counts and prediction hashes,
and the manuscript source/PDF.

No author identity, private path, account, or unpublished external credential
is included.

## Quick verification

From this `artifact/` directory, using Python 3.11 with NumPy and PyTorch:

```bash
python scripts/verify_icra2027_wav_external_audit.py
python scripts/build_icra2027_claim_package.py --output-dir /tmp/icra_claims
python -m pytest -q \
  tests/test_icra2027_claim_package.py \
  tests/test_icra2027_wav_external_audit.py
```

The first command checks nine provenance and recomputation conditions without
retraining. The table builder requires 16 exact JSON hashes, the frozen ranking
score archive, and 36 scientific invariants. Its output should match
`outputs/actmask/icra2027_submission/` byte for byte.

## Reproducing the external WAV run

Clone the public `world-action-verifier/wav_minigrid` repository, check out
commit `527159b06149beacfb3b2d77af7d938ca4efa32d`, and run:

```bash
python scripts/run_icra2027_wav_external_audit.py \
  --wav-root /path/to/wav_minigrid
```

The runner refuses modified upstream code, checkpoint, or data files. Default
training uses five seeds and may take about 25 minutes on a modern CUDA GPU.
The protocol was frozen before any trained single-state control in
`docs/icra2027_wav_external_audit_prereg.md`.

## Scope of this compact bundle

The included reports, confusion matrices, prediction hashes, and raw ranking
scores are sufficient to verify every number rendered in the paper tables.
Large simulator bundles, visual tensors, and public UR5 source logs are omitted
from this compact initial-submission artifact; their frozen derived reports and
exact provenance hashes are included. This bundle therefore verifies reported
claims and table derivation but does not retrain every state/visual/UR5 model.

The WAV result covers only its MiniGrid inverse-dynamics state-complexity split.
It does not cover WAV's Action Following Score, robot experiments, or
real-robot verification.
