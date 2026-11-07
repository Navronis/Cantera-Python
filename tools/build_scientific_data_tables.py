"""Regenerate the machine-readable catalog of scientific tables in this port."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cantera_python import constants
from cantera_python.elements import COMMON_ELEMENTS
from cantera_python.units import KNOWN_UNITS


def main():
    destination = ROOT / 'provenance' / 'scientific_data_tables.jsonl'
    physical_names = ('Pi', 'Sqrt2', 'Avogadro', 'Boltzmann', 'Planck', 'ElectronCharge',
                      'lightSpeed', 'OneAtm', 'OneBar', 'fineStructureConstant', 'ElectronMass',
                      'GasConstant', 'StefanBoltz', 'Faraday', 'permeability_0', 'epsilon_0',
                      'SmallNumber', 'BigNumber')
    rows = [
        {
            'kind': 'catalog-metadata',
            'schema_version': 1,
            'scope': 'Only data embedded in editable Python runtime modules is listed. Mechanism YAML, transport databases, and the complete upstream element/isotope table are not represented by this catalog.',
            'upstream_commit': '726522be4e2a13454d8415b7ef799d621f665cf3',
        },
        {
            'kind': 'physical-constants',
            'source_paths': ['include/cantera/base/ct_defs.h'],
            'units_note': 'Cantera base units; molar constants use kmol.',
            'values': {name: getattr(constants, name) for name in physical_names},
        },
        {
            'kind': 'element-atomic-weights',
            'source_paths': ['src/thermo/Elements.cpp'],
            'scope': 'COMMON_ELEMENTS only: complete periodic table and isotopes from Elements.cpp including electron.',
            'units': 'kg/kmol',
            'values': [{'symbol': element.symbol, 'name': element.name, 'atomic_weight': element.atomic_weight}
                       for element in (COMMON_ELEMENTS.get(symbol) for symbol in COMMON_ELEMENTS.symbols)],
        },
        {
            'kind': 'unit-definitions',
            'source_paths': ['src/base/Units.cpp'],
            'scope': 'KNOWN_UNITS in cantera_python.units; bounded subset of upstream knownUnits.',
            'base_dimensions': ['mass', 'length', 'time', 'temperature', 'current', 'quantity'],
            'values': [{'name': name, 'factor_to_cantera_base': unit.factor, 'dimensions': unit.dimensions}
                       for name, unit in sorted(KNOWN_UNITS.items())],
        },
        {
            'kind': 'thermo-polynomial-models',
            'source_paths': ['src/thermo/NasaPoly1.cpp', 'src/thermo/NasaPoly2.cpp', 'src/thermo/Nasa9Poly1.cpp', 'src/thermo/ShomatePoly.cpp', 'src/thermo/ConstCpPoly.cpp', 'src/thermo/Mu0Poly.cpp'],
            'scope': 'Equation implementations only; no mechanism-specific polynomial coefficient table has been converted.',
            'models': ['NasaPoly1', 'NasaPoly2', 'Nasa9Poly1', 'Nasa9PolyMultiTempRegion', 'ShomatePoly', 'ShomatePoly2', 'ConstCpPoly', 'Mu0Poly'],
        },
        {
            'kind': 'rate-law-models',
            'source_paths': ['src/kinetics/Arrhenius.cpp', 'src/kinetics/Falloff.cpp', 'src/kinetics/PlogRate.cpp', 'src/kinetics/ChebyshevRate.cpp', 'src/kinetics/VibrationalRelaxationRate.cpp', 'src/kinetics/InterfaceRate.cpp', 'src/kinetics/LinearBurkeRate.cpp', 'src/kinetics/ElectronCollisionPlasmaRate.cpp'],
            'scope': 'Equation implementations only; reaction mechanism parameter tables have not been converted.',
            'models': ['ArrheniusRate', 'TwoTempPlasmaRate', 'BlowersMaselRate', 'LindemannRate', 'TroeRate', 'SriRate', 'TsangRate', 'PlogRate', 'ChebyshevRate', 'LinearBurkeRate', 'CustomFunc1Rate', 'ConstantVibrationalRelaxationRate', 'CastelaVibrationalRelaxationRate', 'StarikovskiyVibrationalRelaxationRate', 'MultiStateResolvedVibrationalRelaxationRate', 'ElectronCollisionPlasmaRate', 'InterfaceArrheniusRate', 'StickingArrheniusRate', 'InterfaceBlowersMaselRate', 'StickingBlowersMaselRate'],
        },
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')
    print(f'wrote {len(rows)} rows to {destination}')


if __name__ == '__main__':
    main()
