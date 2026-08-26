from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_milestone3t_static_manuscript_verification_passes():
    subprocess.run([sys.executable,'scripts/verify_milestone3t_manuscript.py'],cwd=ROOT,check=True)
    report=json.loads((ROOT/'outputs/actmask/milestone3t_manuscript_rc1/manuscript_static_verification.json').read_text())
    assert report['passed']

def test_milestone3t_claim_map_and_review_scope_are_explicit():
    rc=ROOT/'outputs/actmask/milestone3t_manuscript_rc1'
    claims=json.loads((rc/'claim_traceability.json').read_text())['claims']
    assert len(claims)>=8 and all(item['artifact'].startswith('outputs/') for item in claims)
    for reviewer in ('reviewer_a.md','reviewer_b.md','reviewer_c.md'):
        text=(rc/reviewer).read_text().lower()
        assert 'potential fatal flaw' in text and 'score' in text
