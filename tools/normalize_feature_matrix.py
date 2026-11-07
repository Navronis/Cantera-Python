"""Normalize the audited feature list to the required evidence schema."""
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / 'audit' / 'feature_matrix.csv').open(encoding='utf-8-sig') as stream:
        source = list(csv.DictReader(stream))
    fields = [
        'Domain', 'Feature', 'Native source location', 'Python source location',
        'Current status', 'Dependencies', 'Tests', 'Oracle cases',
        'Maximum absolute error', 'Maximum relative error', 'Known limitations',
        'Navronis priority', 'Next action',
    ]
    rows = []
    for old in source:
        status = old['Status']
        python_location = old['Python file/function']
        tests = old['Tests']
        cases = old['Oracle cases']
        maximum = old['Maximum error']
        limitation = old['Limitations']
        # Historical ORACLE_VALIDATED labels predate the required range and
        # sample-count gates. Preserve their evidence, but do not preserve the
        # stronger status until those gates are met.
        if status == 'ORACLE_VALIDATED':
            if old['Domain'] == 'Zero-dimensional reactors':
                status = 'FAILED_VALIDATION'
                limitation = ('Only short trajectories were compared and reported errors exceed '
                              'the current ignition/major-species acceptance targets; ' + limitation)
            elif old['Domain'] == 'Transport':
                status = 'PARTIAL'
                limitation = ('Only one oracle state was recorded; the required 1000-state '
                              'transport sweep is missing; ' + limitation)
            else:
                status = 'UNIT_VALIDATED'
                limitation = ('Exact-pinned evidence exists but does not cover the newly required '
                              'temperature/pressure/composition or per-rate ranges; ' + limitation)
        if old['Domain'] == 'Equilibrium' and old['Feature'].startswith('Chemical Equilibrium'):
            status = 'UNIT_VALIDATED'
            python_location = 'cantera_python.equilibrium.equilibrate_ideal_gas'
            tests = 'test_equilibrium.py; test_equilibrium_solver.py'
            cases = '2 released-reference states'
            maximum = '1.5e-8 K (HP temperature at 1 atm)'
            limitation = ('TP/HP ideal gas only; UV/SV/SP, multiphase VCS, charge '
                          'constraints, and nonideal activities are unported')
        priority = ('P0' if old['Domain'] in {'Thermodynamics', 'Kinetics', 'Equilibrium'}
                    else 'P1' if old['Domain'] in {'Transport', 'Numerical integration',
                                                   'Mechanism I/O', 'Public API compatibility'}
                    else 'P2')
        rows.append({
            'Domain': old['Domain'],
            'Feature': old['Feature'],
            'Native source location': old['Native Cantera source/class/function'],
            'Python source location': python_location,
            'Current status': status,
            'Dependencies': 'NumPy; mechanism data' if python_location not in {'None', ''} else 'unimplemented prerequisites',
            'Tests': tests,
            'Oracle cases': cases,
            'Maximum absolute error': maximum,
            'Maximum relative error': maximum,
            'Known limitations': limitation,
            'Navronis priority': priority,
            'Next action': ('expand exact-pinned boundary and production-mechanism validation'
                            if status in {'ORACLE_VALIDATED', 'UNIT_VALIDATED'}
                            else 'implement governing equations and add differential tests'),
        })
    with (ROOT / 'provenance' / 'feature_matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} feature rows')


if __name__ == '__main__':
    main()
