# ActMask / TimeArrow ICRA 2027 submission checklist

Local status on 2026-08-09: **submission-ready**. External upload has not been
performed.

## Frozen deliverables

- Submission PDF: `paper_icra2027/main.pdf`
- PDF SHA-256:
  `d5daecd9db621017f88d1430647abecf11e8220eafcce316f8120a23cdee8dd0`
- Anonymous bundle: `dist/icra2027_submission_bundle.zip`
- Bundle SHA-256:
  `2dca5664afad4b429b6473db972e2fe0077584b70760e0974ce9577014a2053a`
- Bundle size: 2,527,357 bytes
- Internal simulated review: `docs/icra2027_reviewer_audit.md`

The paper is 8 US-Letter pages including references. It is double anonymous,
uses the official `ieeeconf.cls`, embeds all fonts, contains no Type-3 fonts,
has no unresolved references or overfull boxes, and includes an AI-use
disclosure in the anonymous Acknowledgment. The frozen title is “Earning a
Dynamics Claim: An Execution-Grounded Admission Audit for Robot Action
Verification.”

## Final verified evidence

- 16 frozen JSON sources and the corrected ranking raw-score archive
- 36 passing scientific invariants
- 16 deterministic generated claim/table outputs
- 9 passing external-WAV verifier checks
- 15/15 passing preregistered external-WAV decision gates
- 23 passing manuscript/compliance checks
- 17 passing targeted regression tests, including the matched real-robot
  protocol verifier
- two consecutive full builds with byte-identical PDFs
- two consecutive bundle builds with byte-identical ZIP archives
- 71 manifest-tracked bundle files with exact hashes and no detected
  identity/path leak (72 files including the manifest)

## Frozen but unexecuted next evidence

The matched real-robot re-execution cell is preregistered in
`docs/icra2027_real_robot_matched_reexecution_prereg.md`. It requires at least
36 randomized A/B pairs across three families, with a family-fixed committed
candidate trajectory, matched measured decision state, physical-tracking-derived
outcomes, and all statistical and safety gates passing. The verifier also repeats the fixed
action/current/unordered/final-two/ordered/fair-analytic ladder and rejects
learned-method headroom under analytic saturation. No compatible robot control stack or hardware was
found on this server, so **this is a ready-to-run protocol, not a reported
result**. The current paper correctly retains the public-UR5 log rejection.
The documented public-data search also found no released dataset with exact
same-state, same-candidate, different-history physical pair provenance; nearby
episodes must not be substituted post hoc.

## PaperPlaza actions that require an author

1. Create the ICRA 2027 PaperPlaza record before **2026-09-15 11:59 PM
   Pacific Time**; do not rely on the local Asia/Shanghai date boundary.
2. Enter the exact title and abstract from `paper_icra2027/main.tex`.
3. Add all authors, affiliations, conflicts, and required profile information in
   PaperPlaza while keeping the uploaded PDF anonymous.
4. Choose the closest official subject areas, led by robot learning and
   benchmark/evaluation topics.
5. Upload `paper_icra2027/main.pdf` only as the paper PDF and confirm PaperPlaza
   reports 8 pages and Letter size.
6. Do not upload a supplemental PDF: the ICRA 2027 call disallows it. An
   optional video must be at most 180 seconds and 20 MB; no video is currently
   prepared or needed for the paper's claims.
7. If an anonymous artifact link is desired, upload the prepared anonymous ZIP
   to an approved anonymous host/repository and verify anonymity again. The
   local build does not create accounts or upload externally.
8. Download the portal's submitted PDF, inspect every page once more, and record
   its hash and PaperPlaza paper ID.

Official references:

- ICRA 2027 call:
  `https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/`
- IEEE RAS ICRA page:
  `https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/`
- PaperCept template support:
  `https://ras.papercept.net/conferences/support/tex.php`

## Scope warning for submission and rebuttal

Do not describe the UR5 logged-continuation result as counterfactual real-robot
verification. Do not describe the visual pilot as unsolved learned-method
headroom. Do not generalize the public WAV result beyond its pinned MiniGrid
inverse-dynamics state-complexity split. The defensible contribution is the
three-gate benchmark-admission intervention and its controlled admit/reject
decisions. Do not add the preregistered robot cell to the paper unless every
frozen gate passes on genuine measured re-executions.
