import json
from pathlib import Path
import cantera_python as cp
from cantera_python.state import _inverse_native

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'cantera-original' / 'data'
fixture = json.loads((ROOT / 'audit' / 'wsl_oracle_data.json').read_text())
rows = fixture['thermo_10000']
selected = rows[::max(1, len(rows)//40)][:40]
out = []
for i, row in enumerate(selected):
    gas = cp.Solution(DATA / ('gri30.yaml' if row['mechanism'] == 'gri30' else 'airNASA9.yaml'))
    gas.TPX = (row['T'], row['P'], row['X'])
    targets = [('HP', row['enthalpy_mass'], row['P']), ('UV', row['int_energy_mass'], 1/row['density']),
               ('SP', row['entropy_mass'], row['P']), ('SV', row['entropy_mass'], 1/row['density'])]
    for solver, target, second in targets:
        gas.TPX = (row['T'], row['P'], row['X'])
        trace = []
        _inverse_native(gas._thermo._state, target, second, quantity={'HP':'H','UV':'U','SP':'S_P','SV':'S_V'}[solver],
                        rtol=1e-9, max_iterations=500, trace=trace)
        out.append({'case': i, 'solver': solver, 'mechanism': row['mechanism'],
                    'initial': {'T': row['T'], 'P': row['P']}, 'iterations': trace})
(ROOT / 'audit' / 'solver_wave1_iteration_traces_python.jsonl').write_text(
    '\n'.join(json.dumps(x, separators=(',', ':')) for x in out) + '\n')
print(f'wrote {len(out)} traces')
