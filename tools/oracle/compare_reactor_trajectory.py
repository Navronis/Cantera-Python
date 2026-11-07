"""Compare the Python reactor trajectory against the pinned-oracle reference.

Reads provenance/oracle_reactor_trajectory.json (AUTHORITATIVE_ORACLE) and
integrates the same case with the pure-Python ReactorNet, then compares the
sampled time series for T, P, density, and key species mole fractions.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import cantera_python as ct  # noqa: E402
from cantera_python.reactor import IdealGasReactor, ReactorNet  # noqa: E402
from cantera_python.constants import OneAtm  # noqa: E402

PROV = ROOT / 'provenance'
BASE = PROV / 'oracle_reactor_trajectory.json'
OUT = PROV / 'pinned_native_differential_results.jsonl'
GRI30 = ROOT.parent / 'cantera-original' / 'data' / 'gri30.yaml'


def rel_err(a, b):
    scale = max(abs(a), abs(b))
    return 0.0 if a == b else abs(a - b) / scale


def main():
    base = json.loads(BASE.read_text())
    gas = ct.Solution(GRI30)
    gas.TPX = 1500.0, OneAtm, 'H2:2, O2:1, N2:4'
    r = IdealGasReactor(gas)
    net = ReactorNet(r)
    net.setTolerances(1e-9, 1e-15)

    rows = []
    worst = 0.0
    for ref in base['trajectory']:
        t = ref['t']
        if t > 0:
            net.advance(t)
        q = {
            'T': gas.temperature,
            'P': gas.pressure,
            'density': gas.density,
            'H2O_X': gas.X[gas.species_index('H2O')],
            'OH_X': gas.X[gas.species_index('OH')],
            'H2_X': gas.X[gas.species_index('H2')],
            'O2_X': gas.X[gas.species_index('O2')],
        }
        entry = {'t': t}
        e_max = 0.0
        for k, py in q.items():
            nat = ref[k]
            e = rel_err(py, nat)
            e_max = max(e_max, e)
            entry[k] = {'py': py, 'oracle': nat, 'rel_err': e}
        worst = max(worst, e_max)
        rows.append(entry)

    # Tolerance policy: transient stiff trajectories depend on integrator
    # step acceptance; state-dependent ignition timing shifts produce O(1e-2)
    # relative differences in trace species mid-ignition. T and major species
    # must track far tighter. Record, do not mask.
    summary = {
        'suite': 'reactor_trajectory_vs_pinned',
        'oracle_role': 'AUTHORITATIVE_ORACLE',
        'oracle_commit': base['oracle_commit'],
        'points': len(rows),
        'worst_rel_err_overall': worst,
        'worst_rel_err_T': max(e['T']['rel_err'] for e in rows),
        'worst_rel_err_H2O': max(e['H2O_X']['rel_err'] for e in rows),
        'trajectory_rows': rows,
    }
    with OUT.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(summary) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'trajectory_rows'}, indent=1))


if __name__ == '__main__':
    main()
