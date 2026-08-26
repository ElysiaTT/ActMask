#!/usr/bin/env python3
"""Static integrity check for the Milestone 3T manuscript release candidate."""
from __future__ import annotations
import json, re, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RC=ROOT/'outputs/actmask/milestone3t_manuscript_rc1'
PAPER=ROOT/'paper'

REQUIRED=[
 'main.tex','abstract.tex','sections/introduction.tex','sections/related_work.tex','sections/problem_setup.tex','sections/benchmark_design.tex','sections/shortcut_taxonomy.tex','sections/baselines.tex','sections/experiments.tex','sections/results.tex','sections/limitations.tex','sections/conclusion.tex','references.bib','appendix/benchmark_details.tex','appendix/task_definitions.tex','appendix/metric_definitions.tex','appendix/statistical_protocol.tex','appendix/leakage_audits.tex','appendix/additional_results.tex','appendix/reproducibility.tex']
RELEASE=['README.md','LICENSE_PLACEHOLDER.md','DATASET_CARD.md','BENCHMARK_CARD.md','MODEL_CARD_GRU_BASELINE.md','SECURITY_AND_SAFETY.md','REPRODUCIBILITY.md','ARTIFACT_MANIFEST.md']

def key_at(value, dotted):
    for part in dotted.split('.'):
        wildcard=part.endswith('[*]')
        if wildcard: part=part[:-3]
        if isinstance(value,dict): value=value[part]
        else: raise KeyError(part)
        if wildcard:
            if not isinstance(value,list) or not value: raise KeyError(part+'[*]')
            value=value[0]
    return value

def main():
    checks=[]
    def add(name, ok, detail): checks.append({'name':name,'status':'PASS' if ok else 'FAIL','detail':detail})
    for relative in REQUIRED: add('paper:'+relative,(PAPER/relative).is_file(),str(PAPER/relative))
    for relative in RELEASE: add('release:'+relative,(ROOT/'release_candidate'/relative).is_file(),str(ROOT/'release_candidate'/relative))
    frozen=ROOT/'outputs/actmask/milestone3s_paper_package'
    decision=json.loads((frozen/'milestone3s_decision.json').read_text())
    add('frozen_3s_decision',decision['decision']=='A. PAPER PACKAGE READY',decision['decision'])
    claims=json.loads((RC/'claim_traceability.json').read_text())['claims']
    for claim in claims:
        path=ROOT/claim['artifact']; ok=path.is_file()
        if ok:
            try: key_at(json.loads(path.read_text()),claim['json_key'].split(';')[0]); detail='source and first JSON key resolve'
            except Exception as exc: ok=False; detail=repr(exc)
        else: detail='missing source'
        add('claim:'+claim['id'],ok,detail)
    tex='\n'.join(p.read_text() for p in PAPER.rglob('*.tex'))
    cited=set(re.findall(r'\\cite\{([^}]+)\}',tex)); cited={x.strip() for group in cited for x in group.split(',')}
    bib=(PAPER/'references.bib').read_text(); entries=set(re.findall(r'@\w+\{([^,]+),',bib))
    add('bibliography_keys',cited <= entries,{'cited':sorted(cited),'missing':sorted(cited-entries)})
    for table in (PAPER/'tables').glob('*.tex'):
        add('provenance:'+table.name,'Provenance:' in table.read_text(),str(table))
    abstract=(PAPER/'abstract.tex').read_text().lower(); limits=(PAPER/'sections/limitations.tex').read_text().lower()
    add('state_only_disclosure','state-only' in abstract and 'real-robot' in limits,'abstract and limitations qualify scope')
    forbidden=['we demonstrate vla superiority','we demonstrate real-robot transfer','we demonstrate rgb-d success','we establish universal physical reasoning']
    body='\n'.join((PAPER/x).read_text().lower() for x in ['abstract.tex','sections/introduction.tex','sections/results.tex','sections/conclusion.tex'])
    add('no_prohibited_claims',not any(x in body for x in forbidden),{'checked':forbidden})
    add('frozen_table_sources',all((frozen/'tables'/x).is_file() for x in ['main_v2_results.tex','causal_controls.tex','candidate_diversity.tex','milestone_progression.tex']),'3S generated table fragments')
    tool={x:shutil.which(x) for x in ('latexmk','pdflatex','xelatex','bibtex','biber','tectonic')}
    report={'schema':'milestone3t-static-verification-v1','checks':checks,'passed':all(c['status']=='PASS' for c in checks),'latex_tools':tool}
    (RC/'manuscript_static_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'latex_tools':tool}))
    if not report['passed']: raise SystemExit(1)
if __name__=='__main__': main()
