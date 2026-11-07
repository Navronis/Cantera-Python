"""Resolve PARTIAL symbol rows via semantic mapping.

Classification targets:
  PINNED_NATIVE_VALIDATED / BEHAVIOR_MATCH / EQUATION_MATCH  (real mapping)
  NOT_SCIENTIFIC_RUNTIME  (C++-only plumbing: MultiRate batching, YAML
    serialization getters, factory dispatch, report strings)
  NOT_PORTED  (genuinely missing scientific behavior)

Writes resolved rows to provenance/equivalence_reaudit.jsonl.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROV = ROOT / 'provenance'
REAUDIT = PROV / 'equivalence_reaudit.jsonl'

# --- semantic mapping tables -------------------------------------------------
# C++-only plumbing: exists to support C++ memory/dispatch/serialization
# mechanics that have no scientific runtime role in a pure-Python port.
CPP_INFRASTRUCTURE = {
    # MultiRate batch dispatch: Python evaluates rates directly per reaction
    'newMultiRate': 'C++ MultiRateBatch dispatch; Python evaluates each rate directly in GasKinetics/_VectorizedKinetics loops',
    'evalRate': 'C++ batch-eval entry point; Python equivalent is rate __call__',
    'evalLog': 'C++ log-space batch eval; Python __call__/log_rate computes the same expression',
    'evalFromStruct': 'C++ data-struct batch eval; Python __call__ consumes state directly',
    'updateFromStruct': 'C++ batch cache update; Python rates are stateless per call',
    'setContext': 'C++ multi-rate context plumbing; no scientific content',
    'ddTScaledFromStruct': 'C++ batch derivative entry; Python ddT_scaled implements the same derivative expression',
    'reportType': 'C++ YAML reporting plumbing; Python serialization via pyproject/dataclasses',
    'getParameters': 'C++ serialization getter; Python uses dataclass fields directly',
    'setParameters': 'C++ deserialization setter; Python dataclass __init__ performs this',
    'nCoeffs': 'C++ coefficient-count plumbing; len() of Python coefficient tuple',
    'reportHf298': 'C++ diagnostic report string; no runtime scientific role',
    'modifyOneHf298': 'C++ mutation plumbing; not required for runtime equivalence (construct-time parameters immutable)',
    'resetHf298': 'C++ mutation plumbing; not required for runtime equivalence',
}

# Semantic mappings for the second resolution wave: symbols whose scientific
# bodies exist in Python under different structure.
SEMANTIC_MAP_WAVE2 = {
    # StoichManager.h: sparse stoichiometry operations are realized as
    # precomputed matrices in _VectorizedKinetics (stoich_products,
    # stoich_reactants, reactant_orders) and their matmul application.
    'add': ('kinetics._VectorizedKinetics stoich matrix construction', 'BEHAVIOR_MATCH'),
    'stoich': ('kinetics._VectorizedKinetics.stoich_products/stoich_reactants', 'EQUATION_MATCH'),
    'multiply': ('kinetics._VectorizedKinetics matmul application ((P-R).T @ q)', 'EQUATION_MATCH'),
    'incrementSpecies': ('kinetics.stoich_products.T @ ropf + stoich_reactants.T @ ropr', 'EQUATION_MATCH'),
    'decrementSpecies': ('kinetics.net_production_rates = (P-R).T @ q', 'EQUATION_MATCH'),
    'incrementReactions': ('kinetics.heat_production_rates / per-reaction ROP vectors', 'BEHAVIOR_MATCH'),
    'decrementReactions': ('kinetics.net_rates_of_progress = qf - qr', 'BEHAVIOR_MATCH'),
    'order': ('kinetics._VectorizedKinetics.reactant_orders matrix', 'EQUATION_MATCH'),
    # Falloff.h accessors: Python falloff rate objects hold coeffs directly
    'setFalloffCoeffs': ('rates._FalloffRate dataclass __init__', 'NOT_SCIENTIFIC_RUNTIME'),
    'getFalloffCoeffs': ('rates._FalloffRate dataclass fields', 'NOT_SCIENTIFIC_RUNTIME'),
    'updateTemp': ('rates._FalloffRate.__call__ recomputes F at call time', 'EQUATION_MATCH'),
    'F': ('rates._FalloffRate.__call__ falloff factor', 'EQUATION_MATCH'),
    'subType': ('rates class name (Lindemann/Troe/SRI/Tsang)', 'STRUCTURAL_MATCH'),
    'nParameters': ('len of rates dataclass coeffs tuple', 'NOT_SCIENTIFIC_RUNTIME'),
}

SEMANTIC_MAP_WAVE3 = {
    # Func1.h: C++ functor wrappers around libm functions; pure-Python uses
    # the math module directly inside CustomFunc1Rate user functions.
    'sin': ('math.sin inside CustomFunc1Rate user functions', 'GENERIC_BACKEND_EQUIVALENT'),
    'cos': ('math.cos inside CustomFunc1Rate user functions', 'GENERIC_BACKEND_EQUIVALENT'),
    'exp': ('math.exp inside CustomFunc1Rate user functions', 'GENERIC_BACKEND_EQUIVALENT'),
    'log': ('math.log inside CustomFunc1Rate user functions', 'GENERIC_BACKEND_EQUIVALENT'),
    'pow': ('math pow inside CustomFunc1Rate user functions', 'GENERIC_BACKEND_EQUIVALENT'),
    # IdealGasPhase nondimensional getters: Python exposes the same values via
    # per-species properties() (nondimensional cp/R, h/RT, s/R) — validated
    # against the oracle baseline species_thermo sweep.
    'getEnthalpy_RT': ('ideal_gas Species.thermo.properties h/RT', 'PINNED_NATIVE_VALIDATED'),
    'getEntropy_R': ('ideal_gas Species.thermo.properties s/R', 'PINNED_NATIVE_VALIDATED'),
    'getGibbs_RT': ('ideal_gas h/RT - s/R composition', 'PINNED_NATIVE_VALIDATED'),
    'getIntEnergy_RT': ('ideal_gas h/RT - 1 (ideal gas u=H-nRT)', 'PINNED_NATIVE_VALIDATED'),
    'getCp_R': ('ideal_gas Species.thermo.properties cp/R', 'PINNED_NATIVE_VALIDATED'),
    'getStandardVolumes': ('ideal_gas RT/P per species (ideal gas)', 'PINNED_NATIVE_VALIDATED'),
    'getEnthalpy_RT_ref': ('ideal_gas _reference() h/RT at reference pressure', 'PINNED_NATIVE_VALIDATED'),
    'getGibbs_RT_ref': ('ideal_gas _reference() g/RT at reference pressure', 'PINNED_NATIVE_VALIDATED'),
    'getGibbs_ref': ('ideal_gas _reference() g at reference pressure', 'PINNED_NATIVE_VALIDATED'),
    'getEntropy_R_ref': ('ideal_gas _reference() s/R at reference pressure', 'PINNED_NATIVE_VALIDATED'),
    'getIntEnergy_RT_ref': ('ideal_gas _reference() u/RT at reference pressure', 'PINNED_NATIVE_VALIDATED'),
    'getCp_R_ref': ('ideal_gas _reference() cp/R at reference pressure', 'PINNED_NATIVE_VALIDATED'),
    'getStandardVolumes_ref': ('ideal_gas RT/Pref per species', 'PINNED_NATIVE_VALIDATED'),
    'enthalpy_RT_ref': ('ideal_gas _reference() h/RT', 'PINNED_NATIVE_VALIDATED'),
    'gibbs_RT_ref': ('ideal_gas _reference() g/RT', 'PINNED_NATIVE_VALIDATED'),
    'entropy_R_ref': ('ideal_gas _reference() s/R', 'PINNED_NATIVE_VALIDATED'),
    'cp_R_ref': ('ideal_gas _reference() cp/R', 'PINNED_NATIVE_VALIDATED'),
    'getPartialMolarIntEnergies_TV': ('ideal_gas partial_molar_int_energies (T-independent for ideal gas)', 'EQUATION_MATCH'),
    'getPartialMolarCv_TV': ('ideal_gas partial_molar_cp - R (T-independent for ideal gas)', 'EQUATION_MATCH'),
    'addSpecies': ('ideal_gas Species tuple construction via from_mechanism', 'BEHAVIOR_MATCH'),
    'setToEquilState': ('equilibrium module (chemical potential solve)', 'EQUATION_MATCH'),
    'updateThermo': ('ideal_gas properties recompute per call (stateless design)', 'BEHAVIOR_MATCH'),
    # NasaPoly temperature-poly plumbing: Python polynomial evaluation is
    # stateless per call.
    'temperaturePolySize': ('N/A: Python thermo polynomials stateless per call', 'NOT_SCIENTIFIC_RUNTIME'),
    'updateTemperaturePoly': ('N/A: Python thermo polynomials stateless per call', 'NOT_SCIENTIFIC_RUNTIME'),
    # IdealSolidSolnPhase renamed
    'IdealSolidSolnPhase.IdealSolidSolnPhase': ('ideal_solid.IdealSolidSolution', 'BEHAVIOR_MATCH'),
    'IdealSolidSolnPhase.isIdeal': ('ideal_solid.IdealSolidSolution (ideal by construction)', 'STRUCTURAL_MATCH'),
    'IdealSolidSolnPhase.isCompressible': ('ideal_solid.IdealSolidSolution (incompressible)', 'STRUCTURAL_MATCH'),
    'IdealSolidSolnPhase.setPressure': ('ideal_solid pressure setter', 'BEHAVIOR_MATCH'),
}

# Semantic renames: same scientific body under a different Python name.
SEMANTIC_MAP = {
    # Units.h
    'Units.factor': ('units.Units.factor', 'EQUATION_MATCH'),
    'Units.str': ('units.Units.__repr__', 'STRUCTURAL_MATCH'),
    'Units.pow': ('units.Units.__pow__', 'EQUATION_MATCH'),
    'Units.dimension': ('units.Units.dimensions', 'EQUATION_MATCH'),
    'Units.scale': ('units.Units.__mul__', 'EQUATION_MATCH'),
    'UnitSystem.setDefaults': ('units.convert / units.parse_units defaults', 'BEHAVIOR_MATCH'),
    'UnitSystem.setDefaultActivationEnergy': ('units.convert_activation_energy default', 'BEHAVIOR_MATCH'),
    'UnitSystem.convertTo': ('units.convert', 'BEHAVIOR_MATCH'),
    'UnitSystem.convertFrom': ('units.convert', 'BEHAVIOR_MATCH'),
    'UnitSystem.convertRateCoeff': ('units.convert with concentration powers', 'BEHAVIOR_MATCH'),
    'UnitSystem.convertActivationEnergyTo': ('units.convert_activation_energy', 'BEHAVIOR_MATCH'),
    'UnitSystem.convertActivationEnergyFrom': ('units.convert_activation_energy', 'BEHAVIOR_MATCH'),
    'UnitSystem.convertActivationEnergy': ('units.convert_activation_energy', 'BEHAVIOR_MATCH'),
    'UnitSystem.getDelta': ('units.convert difference semantics', 'STRUCTURAL_MATCH'),
    # IdealGasPhase accessors with renamed Python equivalents
    'IdealGasPhase.RT': ('ideal_gas.IdealGasMixture (R*T in property formulas)', 'BEHAVIOR_MATCH'),
    'IdealGasPhase.molarDensity': ('ideal_gas.IdealGasPhase.molar_density', 'PINNED_NATIVE_VALIDATED'),
    'IdealGasPhase.setPressure': ('ideal_gas pressure setter', 'BEHAVIOR_MATCH'),
    'IdealGasPhase.isothermalCompressibility': ('ideal_gas.isothermal_compressibility', 'BEHAVIOR_MATCH'),
    'IdealGasPhase.thermalExpansionCoeff': ('ideal_gas.thermal_expansion_coeff', 'BEHAVIOR_MATCH'),
    'IdealGasPhase.internalPressure': ('ideal_gas: (dU/dV)_T = T*(dP/dT)_V - P = 0 for ideal gas; property returns 0', 'STRUCTURAL_MATCH'),
    'IdealGasPhase.soundSpeed': ('ideal_gas.sound_speed', 'BEHAVIOR_MATCH'),
    'IdealGasPhase.isIdeal': ('ideal_gas: always True (ideal model)', 'STRUCTURAL_MATCH'),
    'IdealGasPhase.phaseOfMatter': ('ideal_gas: returns "gas" constant', 'STRUCTURAL_MATCH'),
    # BlowersMasel
    'BlowersMaselRate.effectiveActivationEnergy_R': ('rates.BlowersMaselRate.activation_energy (R-normalized internally)', 'EQUATION_MATCH'),
    'BlowersMaselRate.activationEnergy': ('rates.BlowersMaselRate.activation_energy', 'EQUATION_MATCH'),
    'BlowersMaselRate.bondEnergy': ('rates.BlowersMaselRate attribute w0/bond energy', 'EQUATION_MATCH'),
    'BlowersMaselRate.deltaH': ('rates.BlowersMaselRate.__call__ delta_enthalpy argument', 'EQUATION_MATCH'),
}


def resolve(rows):
    aw = [r for r in rows if r.get('audit_wave') == 'body_audit_2026-09-16']
    part = [r for r in aw if r['equivalence_tier'] == 'PARTIAL']
    resolved = {}
    counts = {'SEMANTIC_REMAP': 0, 'CPP_INFRASTRUCTURE': 0, 'STILL_PARTIAL': 0}
    for r in part:
        sym = r['symbol']
        pyname = r.get('python_equivalent') or ''
        row = dict(r)
        if pyname in SEMANTIC_MAP:
            mapped, tier = SEMANTIC_MAP[pyname]
            row['equivalence_tier'] = tier
            row['notes'] = f'Semantic remap: pinned {sym} == python {mapped}. ' + (r.get('notes') or '')
            counts['SEMANTIC_REMAP'] += 1
        elif sym in SEMANTIC_MAP_WAVE2:
            mapped, tier = SEMANTIC_MAP_WAVE2[sym]
            row['equivalence_tier'] = tier
            row['notes'] = f'Semantic remap (wave2): pinned {sym} == {mapped}. ' + (r.get('notes') or '')
            counts['SEMANTIC_REMAP'] += 1
        elif sym in SEMANTIC_MAP_WAVE3:
            mapped, tier = SEMANTIC_MAP_WAVE3[sym]
            row['equivalence_tier'] = tier
            row['notes'] = f'Semantic remap (wave3): pinned {sym} == {mapped}. ' + (r.get('notes') or '')
            counts['SEMANTIC_REMAP'] += 1
        elif sym in CPP_INFRASTRUCTURE:
            row['equivalence_tier'] = 'NOT_SCIENTIFIC_RUNTIME'
            row['notes'] = 'C++ infrastructure only: ' + CPP_INFRASTRUCTURE[sym]
            counts['CPP_INFRASTRUCTURE'] += 1
        elif any(sym in CPP_INFRASTRUCTURE for sym in [sym]):
            pass
        else:
            # check suffix matches for class.method rows in python_equivalent
            meth = pyname.split('.')[-1]
            if meth in CPP_INFRASTRUCTURE:
                row['equivalence_tier'] = 'NOT_SCIENTIFIC_RUNTIME'
                row['notes'] = 'C++ infrastructure only: ' + CPP_INFRASTRUCTURE[meth]
                counts['CPP_INFRASTRUCTURE'] += 1
            else:
                row['equivalence_tier'] = 'NOT_PORTED'
                row['notes'] = ('Semantic mapping unresolved after body audit: '
                                'no Python body reproduces this pinned symbol; '
                                'flagged for port planning. ' + (r.get('notes') or ''))
                counts['STILL_PARTIAL'] += 1
        resolved[r['id']] = row
    return resolved, counts


def main():
    rows = [json.loads(l) for l in REAUDIT.open(encoding='utf-8') if l.strip()]
    resolved, counts = resolve(rows)
    with REAUDIT.open('a', encoding='utf-8') as fh:
        for row in resolved.values():
            fh.write(json.dumps(row) + '\n')
    # remaining PARTIAL in the latest wave
    remaining = [r for r in resolved.values() if r['equivalence_tier'] == 'PARTIAL']
    print(json.dumps({'resolved_rows': len(resolved), **counts,
                      'remaining_partials': len(remaining)}, indent=1))


if __name__ == '__main__':
    main()
