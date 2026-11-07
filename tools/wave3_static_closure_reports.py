"""Produce conservative static closure reports for Wave 3."""
from __future__ import annotations
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT.parent/'cantera-original'/'src'/'tpx'
PY=ROOT/'cantera_python'

def main():
    patterns=['1e3','1e-3','1e6','1e-6','kmol','kPa','bar','J','kJ','kg','density','critical']
    rows=[]
    for p in sorted(SRC.glob('*.*')):
        if p.suffix not in ('.cpp','.h'): continue
        txt=p.read_text(errors='ignore')
        for pat in patterns:
            for m in re.finditer(re.escape(pat),txt,re.I):
                line=txt.count('\n',0,m.start())+1
                rows.append({'source_file':str(p.relative_to(SRC)),'line':line,'expression':txt.splitlines()[line-1].strip(),'token':pat,'python_match_status':'MANUAL_REVIEW_REQUIRED'})
    (ROOT/'audit'/'solver_wave3_unit_scaling_audit.json').write_text(json.dumps({'source_scaling_records':rows,'SCALING_MISMATCHES':'UNRESOLVED_MANUAL_REVIEW','unexplained_scaling_differences':'NOT_CLOSED'},indent=2)+'\n')
    branches=['gas_vapor','liquid','two_phase','supercritical','saturation','quality_zero','quality_one','quality_intermediate','density_retry','density_bound','saturation_damping','phase_swap','critical','pair_inversion','reversed_pair','invalid_temperature','invalid_density','invalid_pressure','invalid_quality','failed_bracket','reference_state']
    (ROOT/'audit'/'solver_wave3_scientific_branch_coverage.json').write_text(json.dumps({'branches':[{ 'branch':b,'executed':False,'status':'REQUIRES_EXPLICIT_CASE_EVIDENCE'} for b in branches],'UNCOVERED_SCIENTIFIC_BRANCHES':len(branches)},indent=2)+'\n')
    print('scaling records',len(rows),'branches',len(branches))
if __name__=='__main__': main()
