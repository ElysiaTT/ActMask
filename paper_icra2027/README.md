# ICRA 2027 manuscript

This directory contains the double-anonymous ICRA 2027 manuscript source.
The original state-only draft remains in `../paper/` and is not overwritten.

Generate the frozen tables from the repository root:

```bash
/home/tzh/conda_envs/actmask/bin/python \
  scripts/build_icra2027_state_world_bootstrap.py
/home/tzh/conda_envs/actmask/bin/python \
  scripts/build_icra2027_claim_package.py \
  --output-dir paper_icra2027/generated
```

Generate the audited visual-history montage:

```bash
/home/tzh/conda_envs/actmask/bin/python \
  scripts/build_icra2027_state_audit_figure.py
/home/tzh/conda_envs/actmask/bin/python \
  scripts/build_icra2027_visual_history_figure.py
```

Verify the frozen public WAV MiniGrid external audit (this checks the report,
confusion counts, summaries, and preregistered gates without retraining):

```bash
/home/tzh/conda_envs/actmask/bin/python \
  scripts/verify_icra2027_wav_external_audit.py
```

To reproduce that experiment from scratch, clone the public
`world-action-verifier/wav_minigrid` repository at commit
`527159b06149beacfb3b2d77af7d938ca4efa32d`, then run:

```bash
/home/tzh/conda_envs/actmask/bin/python \
  scripts/run_icra2027_wav_external_audit.py \
  --wav-root /path/to/wav_minigrid
```

The runner rejects any upstream checkpoint, source, or data hash that differs
from the preregistration in `../docs/icra2027_wav_external_audit_prereg.md`.

Build all generated artifacts and compile with pdfLaTeX from the repository
root:

```bash
scripts/build_icra2027_pdf.sh
```

The script prefers the isolated local TeX tree when present and otherwise
uses `latexmk` from `PATH`. The audited pdfLaTeX output is written to both
`paper_icra2027/build_pdflatex/main.pdf` and `paper_icra2027/main.pdf`.

Run the submission audit from the repository root:

```bash
/home/tzh/conda_envs/actmask/bin/python \
  scripts/verify_icra2027_manuscript.py
```

The audit reports are written under
`outputs/actmask/icra2027_manuscript_audit/`.

Official template source:
`https://ras.papercept.net/conferences/support/files/ieeeconf.zip`.
The downloaded zip SHA-256 recorded on 2026-08-09 is
`11d1051d5fe3dafd1e25bc7a8b66265cbe65dc135e26440cc6661baeeeb90c76`.
The extracted `ieeeconf.cls` SHA-256 is
`4befef671c2a996889d325f5170d3387bf42aac9a37dcaa93724ad49816e4ec2`.

The initial submission is limited to eight total pages including references.
Do not report visual confidence intervals: the frozen correction shows that
they target a different estimator from the displayed three-seed mean.

ICRA 2027 requires substantive generative-AI use to be disclosed in the
Acknowledgment. The anonymous manuscript therefore contains a generic
Acknowledgment that names OpenAI Codex, the affected sections, and the manner
of use, while omitting people, funding, affiliations, and identity-bearing
details. `scripts/verify_icra2027_manuscript.py` checks both specificity and
anonymity of this section.

The paper tables are admitted only when sixteen frozen JSON hashes, the corrected
ranking raw-score hash, and all 36 scientific invariants pass. The preliminary
branch-pooled candidate-ranking values are superseded; use only the versioned
history-keyed report in `../outputs/actmask/milestone3r_nl_v2_rankfix/`.
The preliminary candidate-pair bootstrap interval is also superseded; the
paper uses the versioned 300-base-world cluster bootstrap under
`../outputs/actmask/milestone3r_nl_v2_statsfix/`.

The next evidence cell is frozen, but not yet executed, in
`../docs/icra2027_real_robot_matched_reexecution_prereg.md`. It requires at
least 36 randomized matched A/B physical re-executions across three families and repeats the fixed
action/current/unordered/final-two/ordered/fair-analytic ladder. It is
independently checked by
`../scripts/verify_icra2027_real_robot_matched_reexecution.py`.
Until every frozen gate passes, the manuscript must retain the current honest
public-UR5 logged-continuation rejection and must not claim matched real-robot
verification.

`../docs/icra2027_public_real_reexecution_feasibility.md` records why currently
identified public real-robot datasets cannot replace the physical collection:
none releases exact same-state, same-candidate, different-history pair
provenance under the paper's observable contract.
