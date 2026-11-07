"""Supplementary TP/HP comparison against released Cantera 3.2.0."""
import json
from pathlib import Path
import sys

import cantera as native


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cantera_python as port  # noqa: E402
MECHANISM = ROOT.parent / 'cantera-original' / 'data' / 'gri30.yaml'
SPECIES = ('H2', 'O2', 'H2O', 'OH', 'H', 'O', 'N2', 'NO')


def state(gas):
    return {name: float(gas.X[gas.species_index(name)]) for name in SPECIES}


def main():
    cases = []
    for pressure_atm in (1.0, 30.0):
        reference = native.Solution(str(MECHANISM))
        candidate = port.Solution(MECHANISM)
        pressure = pressure_atm * native.one_atm
        for gas in (reference, candidate):
            gas.TPX = 300.0, pressure, 'H2:2, O2:1, N2:3.76'
        target_h = float(candidate.enthalpy_mass)
        reference.equilibrate('HP')
        candidate.equilibrate('HP')
        ref_x, got_x = state(reference), state(candidate)
        cases.append({
            'pressure_atm': pressure_atm,
            'reference_temperature_K': float(reference.T),
            'python_temperature_K': float(candidate.T),
            'temperature_absolute_error_K': abs(float(candidate.T - reference.T)),
            'enthalpy_residual_J_per_kg': float(candidate.enthalpy_mass - target_h),
            'maximum_selected_species_absolute_error': max(
                abs(got_x[name] - ref_x[name]) for name in SPECIES),
            'reference_mole_fractions': ref_x,
            'python_mole_fractions': got_x,
            'python_diagnostics': candidate.equilibrium_diagnostics,
        })
    report = {
        'role': 'SUPPLEMENTARY_RELEASED_REFERENCE',
        'reference_cantera_version': native.__version__,
        'pinned_commit': '726522be4e2a13454d8415b7ef799d621f665cf3',
        'note': 'Released 3.2.0 is not the exact pinned oracle.',
        'cases': cases,
    }
    output = ROOT / 'provenance' / 'equilibrium_cantera_3_2_comparison.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
