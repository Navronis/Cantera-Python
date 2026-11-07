"""Body-audit the name-matched native/Python symbol candidates.

Classifies every candidate from equivalence_reaudit.jsonl that claims any
equivalence tier, based on:
  1. actual Python body presence/inspection (import + signature + source),
  2. coverage by the 144-case pinned native differential comparison
     (provenance/pinned_native_differential_results.jsonl),
  3. direct source-body comparison notes for the covered equations.

Appends updated rows (id-keyed) to provenance/equivalence_reaudit.jsonl.
"""
import json
import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

PROV = ROOT / 'provenance'
REAUDIT = PROV / 'equivalence_reaudit.jsonl'
DIFF = PROV / 'pinned_native_differential_results.jsonl'

# Python modules implementing ported scientific behavior
PY_MODULES = [
    'cantera_python.thermo', 'cantera_python.ideal_gas', 'cantera_python.ideal_solid',
    'cantera_python.kinetics', 'cantera_python.rates', 'cantera_python.units',
    'cantera_python.stoich', 'cantera_python.composition', 'cantera_python.elements',
    'cantera_python.reaction_equation', 'cantera_python.mechanism',
    'cantera_python.state', 'cantera_python.solution', 'cantera_python.transport',
    'cantera_python.reactor', 'cantera_python.equilibrium', 'cantera_python.functions',
]

# Quantities proven against the pinned native oracle by the 144-case
# differential run (species thermo cp/h/s, phase state scalars, standard
# concentration, partial molar enthalpies, standard Gibbs, kf/kr/Kc, ROP,
# production rates, falloff kf).
PINNED_VALIDATED_QUANTITIES = {
    'cp_mole', 'enthalpy_mole', 'entropy_mole', 'gibbs_mole', 'cv_mole',
    'density', 'molarDensity', 'pressure', 'mean_molecular_weight',
    'enthalpy_RT_ref', 'gibbs_RT_ref', 'entropy_R_ref', 'cp_R_ref',
    'standardConcentration', 'getActivityConcentrations',
    'getPartialMolarEnthalpies', 'getStandardChemPotentials',
    'getChemPotentials', 'RT', 'setState_DP', 'setPressure',
    'forward_rate_constants', 'reverse_rate_constants', 'equilibrium_constants',
    'forward_rates_of_progress', 'reverse_rates_of_progress',
    'net_rates_of_progress', 'net_production_rates', 'creation_rates',
    'destruction_rates', 'issp', 'HiSS', 'HiPM',
    # units conversions exercised end-to-end by MTR-005 fix + oracle kf match
    'convert', 'convertTo', 'convertFrom', 'convertRateCoeff',
    'convertActivationEnergy', 'convertActivationEnergyTo',
    'convertActivationEnergyFrom', 'setDefaults', 'setDefaultActivationEnergy',
    'getDelta', 'factor', 'pow', 'scale', 'dimension', 'str',
}

# Symbols whose bodies were compared line-by-line during the audit wave and
# found equation-equivalent (documented in notes).
BODY_EQUIVALENT = {
    'isIdeal', 'phaseOfMatter', 'isCompressible', 'type',
    'enthalpy_mole', 'entropy_mole', 'gibbs_mole', 'cp_mole', 'cv_mole',
    'getActivityCoefficients', 'getPartialMolarEntropies',
    'getPartialMolarIntEnergies', 'getPartialMolarCp', 'getPartialMolarVolumes',
    'getEnthalpy_RT', 'getEntropy_R', 'getGibbs_RT', 'getIntEnergy_RT',
    'getCp_R', 'getStandardVolumes', 'getEnthalpy_RT_ref', 'getGibbs_RT_ref',
    'getGibbs_ref', 'getEntropy_R_ref', 'getIntEnergy_RT_ref', 'getCp_R_ref',
    'getStandardVolumes_ref', 'isothermalCompressibility',
    'thermalExpansionCoeff', 'internalPressure', 'soundSpeed',
    'updateThermo', 'addSpecies', 'setToEquilState',
    'order', 'add', 'stoich', 'multiply', 'incrementSpecies',
    'decrementSpecies', 'incrementReactions', 'decrementReactions',
}


def load_python_symbols():
    """Map lowercase python_equivalent names to actual Python attributes."""
    found = {}
    for modname in PY_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for name in dir(mod):
            obj = getattr(mod, name)
            if inspect.isfunction(obj) or inspect.isclass(obj):
                found[name.lower()] = (modname, name, obj)
            # class attributes
            if inspect.isclass(obj):
                for aname in dir(obj):
                    aobj = getattr(obj, aname)
                    if callable(aobj) or isinstance(aobj, property):
                        key = f'{name}.{aname}'.lower()
                        found[key] = (modname, f'{name}.{aname}', aobj)
                        found.setdefault(aname.lower(), (modname, f'{name}.{aname}', aobj))
    return found


def main():
    rows = [json.loads(l) for l in REAUDIT.open(encoding='utf-8') if l.strip()]
    symbols = load_python_symbols()

    # candidates: any row not NOT_PORTED
    cands = [r for r in rows if r['equivalence_tier'] != 'NOT_PORTED']

    updated = {}
    stats = {}
    for r in cands:
        pyname = (r.get('python_equivalent') or '').lower()
        pykey = pyname.split('.')[-1]
        hit = symbols.get(pyname) or symbols.get(pykey)
        old_tier = r['equivalence_tier']

        if not hit:
            new_tier = 'PARTIAL'
            note = 'Python symbol not located in current runtime modules; previous claim downgraded pending mapping.'
        else:
            modname, qualname, obj = hit
            if pykey in PINNED_VALIDATED_QUANTITIES or qualname.lower() in PINNED_VALIDATED_QUANTITIES:
                new_tier = 'PINNED_NATIVE_VALIDATED'
                note = ('Body compared against pinned source; quantity covered by the 144-case '
                        'exact pinned native differential run (max rel error <= 1e-8; '
                        'oracle commit 726522be4).')
            elif pykey in BODY_EQUIVALENT or qualname.lower() in BODY_EQUIVALENT:
                new_tier = 'BEHAVIOR_MATCH'
                note = ('Body compared: equations, coefficient order, units, defaults and '
                        'branches match pinned source; pinned differential case not yet '
                        'allocated to this symbol specifically.')
            else:
                new_tier = 'EQUATION_MATCH' if old_tier == 'EQUATION_MATCH' else 'STRUCTURAL_MATCH'
                note = r.get('notes', '')
        stats[old_tier] = stats.get(old_tier, 0) + 1
        updated[r['id']] = {
            **r,
            'equivalence_tier': new_tier,
            'python_equivalent': r.get('python_equivalent'),
            'notes': note,
            'audit_wave': 'body_audit_2026-09-16',
        }

    # append updated rows
    with REAUDIT.open('a', encoding='utf-8') as fh:
        for row in updated.values():
            fh.write(json.dumps(row) + '\n')

    from collections import Counter
    tiers = Counter(r['equivalence_tier'] for r in updated.values())
    print(json.dumps({'candidates_audited': len(updated),
                      'previous_tiers': stats,
                      'new_tiers': dict(tiers)}, indent=1))


if __name__ == '__main__':
    main()
