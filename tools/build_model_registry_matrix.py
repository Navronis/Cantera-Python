"""Build a source-derived status matrix for thermo and reaction-rate factories."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'cantera-original'
DESTINATION = ROOT / 'provenance' / 'model_registry_matrix.jsonl'

RATE_PORTS = {
    'Arrhenius': 'ArrheniusRate',
    'two-temperature-plasma': 'TwoTempPlasmaRate',
    'constant-vibrational-relaxation': 'ConstantVibrationalRelaxationRate',
    'multi-state-resolved-vibrational-relaxation': 'MultiStateResolvedVibrationalRelaxationRate',
    'Starikovskiy-vibrational-relaxation': 'StarikovskiyVibrationalRelaxationRate',
    'Castela-vibrational-relaxation': 'CastelaVibrationalRelaxationRate',
    'electron-collision-plasma': 'ElectronCollisionPlasmaRate',
    'Blowers-Masel': 'BlowersMaselRate',
    'Lindemann': 'LindemannRate',
    'Troe': 'TroeRate',
    'SRI': 'SriRate',
    'Tsang': 'TsangRate',
    'pressure-dependent-Arrhenius': 'PlogRate',
    'linear-Burke': 'LinearBurkeRate',
    'Chebyshev': 'ChebyshevRate',
    'custom-rate-function': 'CustomFunc1Rate',
    'interface-Arrhenius': 'InterfaceArrheniusRate',
    'sticking-Arrhenius': 'StickingArrheniusRate',
    'interface-Blowers-Masel': 'InterfaceBlowersMaselRate',
    'sticking-Blowers-Masel': 'StickingBlowersMaselRate',
}
THERMO_PORTS = {
    'NASA7': 'NasaPoly2', 'Shomate': 'ShomatePoly2', 'NASA9': 'Nasa9PolyMultiTempRegion',
    'constant-cp': 'ConstCpPoly', 'piecewise-Gibbs': 'Mu0Poly',
}


def _records():
    rate_path = SOURCE / 'src' / 'kinetics' / 'ReactionRateFactory.cpp'
    thermo_path = SOURCE / 'src' / 'thermo' / 'SpeciesThermoFactory.cpp'
    rate_text, thermo_text = rate_path.read_text(encoding='utf-8'), thermo_path.read_text(encoding='utf-8')
    rate_names = re.findall(r'\breg\("([^"]+)"', rate_text)
    aliases = re.findall(r'\baddAlias\("([^"]+)",\s*"([^"]*)"', rate_text)
    thermo_names = re.findall(r'\bmodel\s*==\s*"([^"]+)"', thermo_text)
    rows = [{
        'kind': 'registry-metadata', 'schema_version': 1,
        'upstream_commit': '726522be4e2a13454d8415b7ef799d621f665cf3',
        'scope': 'Factory registrations are discovered from pinned source. “partial” means an equation object exists but factory/YAML construction or full integration does not. “unported” means no matching Python implementation has been recorded.',
    }]
    for name in rate_names:
        symbol = RATE_PORTS.get(name)
        rows.append({
            'kind': 'reaction-rate-model', 'factory_key': name,
            'source_path': 'src/kinetics/ReactionRateFactory.cpp',
            'python_symbol': symbol, 'status': 'partial' if symbol else 'unported',
            'scope': 'Scalar equation only; native factory setup, unit-stack parsing, and full Kinetics integration are unported.' if symbol else 'No Python equation or factory implementation recorded.',
        })
    for canonical, alias in aliases:
        rows.append({
            'kind': 'reaction-rate-alias', 'factory_key': alias,
            'canonical_key': canonical, 'source_path': 'src/kinetics/ReactionRateFactory.cpp',
            'status': 'partial' if canonical in RATE_PORTS else 'unported',
        })
    for name in thermo_names:
        symbol = THERMO_PORTS.get(name)
        rows.append({
            'kind': 'species-thermo-model', 'factory_key': name,
            'source_path': 'src/thermo/SpeciesThermoFactory.cpp',
            'python_symbol': symbol, 'status': 'partial' if symbol else 'unported',
            'scope': 'Polynomial/property equation only; YAML setup and MultiSpeciesThermo integration are unported.' if symbol else 'No Python equation or factory implementation recorded.',
        })
    return rows


def main():
    rows = _records()
    DESTINATION.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    print(f'wrote {len(rows)} rows to {DESTINATION}')


if __name__ == '__main__':
    main()
