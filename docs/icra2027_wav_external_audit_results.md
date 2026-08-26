# ICRA 2027 public WAV MiniGrid external audit

Status: **PASS** under the protocol frozen in
`docs/icra2027_wav_external_audit_prereg.md`.

## Scope

The audit covers only the public WAV MiniGrid inverse-dynamics
state-complexity split at upstream commit
`527159b06149beacfb3b2d77af7d938ca4efa32d`. It does not cover WAV's Action
Following Score, its robot experiments, or real-robot action verification.

The primary measure is WAV's official dynamic accuracy: execute the predicted
action with the released MiniGrid physics oracle and score the channels that
changed in the logged next state. Action accuracy is retained in the frozen
report and confusion counts.

## Frozen test results

Values below are dynamic accuracy. Learned controls and SparseIDM are
mean ± sample standard deviation over five seeds. “Single” is the stronger of
the separately evaluated current-only and next-only controls at each
complexity.

| Objects | Test N | Single | Paired CNN | SparseIDM | Transition rule | Pair margin |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1,000 | 0.410 ± 0.024 | 0.879 ± 0.012 | 0.947 ± 0.000 | 0.947 | 0.469 |
| 8 | 2,500 | 0.378 ± 0.052 | 0.853 ± 0.020 | 0.953 ± 0.000 | 0.958 | 0.475 |
| 10 | 2,500 | 0.329 ± 0.047 | 0.805 ± 0.044 | 0.940 ± 0.001 | 0.952 | 0.476 |
| 12 | 2,500 | 0.343 ± 0.042 | 0.811 ± 0.043 | 0.951 ± 0.001 | 0.962 | 0.468 |
| 14 | 2,500 | 0.341 ± 0.083 | 0.800 ± 0.041 | 0.947 ± 0.002 | 0.969 | 0.459 |

Macro dynamic accuracy is 0.356 ± 0.050 for current-only, 0.348 ± 0.034 for
next-only, 0.830 ± 0.029 for the paired CNN, 0.948 ± 0.0004 for the released
SparseIDM, and 0.958 for the fixed transition rule.

The transition rule reaches 1.000 action accuracy on every test split. Its
dynamic score is below one because the released physics oracle does not exactly
reproduce every changed channel in the logged transitions, even under the true
action. The rule score is therefore an observed metric ceiling rather than a
normalizer.

## Decision

All 15 preregistered gates pass:

- paired dynamic accuracy exceeds the stronger single-state control by at
  least 0.10 at every complexity (minimum observed margin 0.459);
- the released SparseIDM exceeds 0.90 at every complexity (minimum 0.940); and
- the transition rule exceeds the strongest single-state control at every
  complexity.

The result supports a narrowly scoped positive claim: this public
inverse-dynamics benchmark requires observed transition evidence under the
frozen controls. It does not imply that a learned dynamics model is necessary;
the fixed transition rule decodes all action labels.

## Artifacts

- Canonical report:
  `outputs/actmask/icra2027_wav_external_audit/audit.json`
- Canonical report SHA-256:
  `e9e1aaa2f37cc3b705b70f71f8e1439fdeb1d0f426e1d50b58ca331c6e6091f0`
- Runner: `scripts/run_icra2027_wav_external_audit.py`
- Independent verifier: `scripts/verify_icra2027_wav_external_audit.py`
- Verification report:
  `outputs/actmask/icra2027_wav_external_audit/verification.json`

The canonical report stores per-seed confusion matrices, exact dynamic-score
numerators and denominators, and prediction hashes for every test cell. The
independent verifier checks nine provenance, matrix, recomputation, and decision
conditions without retraining.
