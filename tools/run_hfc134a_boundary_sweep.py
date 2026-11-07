from __future__ import annotations
import json, math
from pathlib import Path
from cantera_python.tpx import HFC134a, PropertyPair
ROOT=Path(__file__).resolve().parents[1]
def main():
    rows=[]; failures=0
    o=HFC134a()
    for i in range(1000):
        f=(i+0.5)/1000
        T=o.Tmin()+(o.Tcrit()-o.Tmin())*f
        q=[0.0,0.001,0.01,0.5,0.99,0.999,1.0][i%7]
        try:
            o.Set(PropertyPair.TX,T,q)
            rows.append({'case':i,'T':T,'quality':q,'P':o.P(),'rho':o.Rho,'x':o.x(),'two_phase':o.TwoPhase(),'finite':all(math.isfinite(v) for v in (o.P(),o.Rho)),'status':'PASS'})
        except Exception as e:
            failures+=1; rows.append({'case':i,'T':T,'quality':q,'status':'FAIL','exception':type(e).__name__,'message':str(e)})
    (ROOT/'audit'/'solver_wave3_hfc134a_boundary.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (ROOT/'audit'/'solver_wave3_hfc134a_boundary_summary.json').write_text(json.dumps({'cases':1000,'failures':failures,'passes':1000-failures},indent=2)+'\n')
    print(json.dumps({'cases':1000,'failures':failures,'passes':1000-failures}))
if __name__=='__main__': main()
