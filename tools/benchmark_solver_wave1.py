"""Benchmark Wave 1; native numbers are explicitly supplementary release data."""
import json, platform, statistics, time
from pathlib import Path
import cantera_python as py

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT.parent/'cantera-original'/'data'
rows=json.loads((ROOT/'audit'/'wsl_oracle_data.json').read_text())['thermo_10000']
rows=rows[::max(1,len(rows)//1000)][:1000]
def run_python(kind):
    gases={m:py.Solution(DATA/('gri30.yaml' if m=='gri30' else 'airNASA9.yaml')) for m in ('gri30','airNASA9')}
    for m,g in gases.items(): g.TPX=(1000,101325,[1.0]+[0.0]*(g.n_species-1))
    vals=[]; t0=time.perf_counter()
    for r in rows:
        g=gases[r['mechanism']]; g.TPX=(r['T'],r['P'],r['X']); a,b={'HP':(r['enthalpy_mass'],r['P']),'UV':(r['int_energy_mass'],1/r['density']),'SP':(r['entropy_mass'],r['P']),'SV':(r['entropy_mass'],1/r['density'])}[kind]
        t=time.perf_counter(); getattr(g,{'HP':'setState_HP','UV':'setState_UV','SP':'setState_SP','SV':'setState_SV'}[kind])(a,b); vals.append((time.perf_counter()-t)*1e6)
    vals.sort(); return {'cases':len(vals),'total_s':sum(vals)/1e6,'mean_us':statistics.fmean(vals),'median_us':statistics.median(vals),'p95_us':vals[int(.95*len(vals))-1],'p99_us':vals[int(.99*len(vals))-1],'min_us':vals[0],'max_us':vals[-1]}
out={'pinned_native_status':'UNAVAILABLE','release_native_comparison':'NOT_RUN','environment':{'os':platform.platform(),'python':platform.python_version(),'processor':platform.processor()}}
out['python']={k:run_python(k) for k in ('HP','UV','SP','SV')}
(ROOT/'audit'/'solver_wave1_performance.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
