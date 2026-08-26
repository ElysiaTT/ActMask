# TimeArrow-v2 audit

This directory is an isolated, reproducible audit workspace. The script reads
frozen TimeArrow-v2 artifacts and refreshes its own evidence
under `audit/results/`; it
does not train models or change `outputs/`, `paper_icra2027/generated`, or any
paper table.

Run the simulator-free audit and tests with the existing base Python:

```bash
/root/miniconda3/bin/python audit/timearrow_shortcut_audit.py
/root/miniconda3/bin/python -m unittest discover -s audit/tests -v
```

The main human-readable result is `audit/audit_report.md`. Environment and
bounded simulator re-execution evidence live separately under
`audit/environment/` and `audit/replay/`.
