"""Re-evaluate the port against the serialized exact-pinned thermo fixture."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from collections import Counter

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import cantera_python as ct  # noqa: E402


FIXTURE = ROOT / 'audit' / 'wsl_oracle_data.json'
DATA = ROOT.parent / 'cantera-original' / 'data'
SUMMARY = ROOT / 'provenance' / 'thermo_10050_reconciliation.json'
STATISTICS = ROOT / 'provenance' / 'oracle_statistics.csv'
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCES = {
    'density': 1e-12, 'mean_MW': 1e-12,
    'cp_mole': 1e-6, 'cv_mole': 1e-6,
    'cp_mass': 1e-8, 'cv_mass': 1e-8,
    'enthalpy_mole': 1e-3, 'int_energy_mole': 1e-3,
    'entropy_mole': 1e-6, 'gibbs_mole': 1e-3,
    'enthalpy_mass': 1e-5, 'int_energy_mass': 1e-5,
    'entropy_mass': 1e-8, 'gibbs_mass': 1e-5,
    'chem_potentials': 1e-3, 'standard_gibbs_RT': 1e-12,
    'partial_molar_enthalpies': 1e-3,
    'partial_molar_entropies': 1e-6,
    'partial_molar_int_energies': 1e-3,
    'partial_molar_cp': 1e-6,
}


def main():
    fixture = json.loads(FIXTURE.read_text(encoding='utf-8'))
    phases = {
        'gri30': ct.Solution(DATA / 'gri30.yaml'),
        'airNASA9': ct.Solution(DATA / 'airNASA9.yaml', phase_name='airNASA9'),
    }
    scalars = ('density', 'mean_MW', 'cp_mole', 'cv_mole', 'cp_mass',
               'cv_mass', 'enthalpy_mole', 'int_energy_mole', 'entropy_mole',
               'gibbs_mole', 'enthalpy_mass', 'int_energy_mass',
               'entropy_mass', 'gibbs_mass')
    scalar_python = {'mean_MW': 'mean_molecular_weight'}
    vectors = ('chem_potentials', 'standard_gibbs_RT',
               'partial_molar_enthalpies', 'partial_molar_entropies',
               'partial_molar_int_energies', 'partial_molar_cp')
    errors = {name: {'abs': [], 'rel': []} for name in scalars + vectors}
    failures = 0
    worst = None
    normalization_max = 0.0
    for index, state in enumerate(fixture['thermo_10000']):
        gas = phases[state['mechanism']]
        gas.TPX = state['T'], state['P'], state['X']
        normalization_max = max(normalization_max, abs(sum(gas.X) - 1.0),
                                abs(sum(gas.Y) - 1.0))
        for name in scalars:
            actual = float(getattr(gas, scalar_python.get(name, name)))
            expected = float(state[name])
            absolute = abs(actual - expected)
            relative = absolute / max(abs(expected), ABSOLUTE_TOLERANCES[name])
            errors[name]['abs'].append(absolute)
            errors[name]['rel'].append(relative)
            passed = absolute <= ABSOLUTE_TOLERANCES[name] + RELATIVE_TOLERANCE * abs(expected)
            failures += not passed
            if worst is None or relative > worst['relative_error']:
                worst = {'state_index': index, 'tag': state['tag'],
                         'mechanism': state['mechanism'], 'property': name,
                         'temperature': state['T'], 'pressure': state['P'],
                         'absolute_error': absolute, 'relative_error': relative}
        vector_actual = {
            'chem_potentials': gas.chemical_potentials,
            'standard_gibbs_RT': np.asarray(gas.standard_chemical_potentials)
                                 / (ct.GasConstant * gas.T),
            'partial_molar_enthalpies': gas.partial_molar_enthalpies,
            'partial_molar_entropies': gas.partial_molar_entropies,
            'partial_molar_int_energies': gas.partial_molar_int_energies,
            'partial_molar_cp': gas.partial_molar_cp,
        }
        for name in vectors:
            actual = np.asarray(vector_actual[name], dtype=float)
            expected = np.asarray(state[name], dtype=float)
            absolute_values = np.abs(actual - expected)
            relative_values = absolute_values / np.maximum(
                np.abs(expected), ABSOLUTE_TOLERANCES[name])
            errors[name]['abs'].extend(absolute_values.tolist())
            errors[name]['rel'].extend(relative_values.tolist())
            failures += int(np.count_nonzero(
                absolute_values > ABSOLUTE_TOLERANCES[name]
                + RELATIVE_TOLERANCE * np.abs(expected)))
            local = int(np.argmax(relative_values))
            relative = float(relative_values[local])
            if worst is None or relative > worst['relative_error']:
                worst = {'state_index': index, 'tag': state['tag'],
                         'mechanism': state['mechanism'], 'property': name,
                         'species_index': local, 'temperature': state['T'],
                         'pressure': state['P'],
                         'absolute_error': float(absolute_values[local]),
                         'relative_error': relative}
    rows = []
    for name, values in errors.items():
        absolute = np.asarray(values['abs'])
        relative = np.asarray(values['rel'])
        rows.append({
            'feature': 'thermodynamics', 'property': name,
            'comparisons': len(relative), 'absolute_tolerance': ABSOLUTE_TOLERANCES[name],
            'relative_tolerance': RELATIVE_TOLERANCE,
            'max_absolute_error': float(absolute.max(initial=0.0)),
            'max_relative_error': float(relative.max(initial=0.0)),
            'p99_relative_error': float(np.percentile(relative, 99)),
            'median_relative_error': float(np.median(relative)),
            'rms_relative_error': float(math.sqrt(np.mean(relative ** 2))),
        })
    with STATISTICS.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    states = fixture['thermo_10000']
    tags = Counter(state['tag'] for state in states)
    report = {
        'fixture': str(FIXTURE.relative_to(ROOT)),
        'fixture_sha256': __import__('hashlib').sha256(FIXTURE.read_bytes()).hexdigest(),
        'oracle_commit': fixture['metadata']['cantera_commit'],
        'seed': fixture['metadata']['seed'],
        'states': len(states),
        'scalar_and_vector_components_compared': sum(row['comparisons'] for row in rows),
        'failed_components': int(failures),
        'temperature_range_K': [min(state['T'] for state in states),
                                max(state['T'] for state in states)],
        'pressure_range_Pa': [min(state['P'] for state in states),
                              max(state['P'] for state in states)],
        'random_state_count': tags['random_gri30'] + tags['random_air9'],
        'pure_species_state_count': sum(count for tag, count in tags.items()
                                        if tag.startswith('pure_')),
        'boundary_state_count': sum(count for tag, count in tags.items()
                                    if tag.startswith('boundary_')),
        'pressure_probe_count': sum(count for tag, count in tags.items()
                                    if tag.startswith('pressure_')),
        'normalization_max_residual': normalization_max,
        'worst_relative_case': worst,
        'statistics_file': str(STATISTICS.relative_to(ROOT)),
        'native_fixture_regenerated_this_run': False,
        'native_regeneration_blocker': 'WSL launch denied by host (E_ACCESSDENIED)',
    }
    SUMMARY.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
