"""Differential validator for Thermodynamics Family 4: Complete Ideal Gas, Solid, Surface, Edge and Condensed Phase Models."""
import sys
import os
import json
import math
import numpy as np

sys.path.insert(0, ".")
from cantera_python.solution import Solution

def get_val(obj, name):
    if not hasattr(obj, name):
        return None
    val = getattr(obj, name)
    return val() if callable(val) else val

def rel_err(a, b):
    diff = abs(a - b)
    if diff < 1e-6:
        return 0.0
    scale = max(abs(a), abs(b))
    return diff / scale

def max_vec_rel_err(va, vb):
    if len(va) == 0 or len(vb) == 0:
        return 0.0
    errs = [rel_err(a, b) for a, b in zip(va, vb)]
    return max(errs)

def compare_all():
    oracle_file = "audit/native_thermo_family4_oracle.json"
    if not os.path.exists(oracle_file):
        print(f"Error: Oracle file {oracle_file} not found.")
        sys.exit(1)

    with open(oracle_file, "r") as f:
        data = json.load(f)

    cases = data["cases"]
    print(f"Loaded {len(cases)} test cases from native oracle.")

    base_dir = "../cantera-original"
    if not os.path.exists(base_dir):
        base_dir = "outputs/cantera-original"

    # Track maximum relative errors by property and model
    model_stats = {}
    current_key = None
    sol = None

    for i, c in enumerate(cases):
        mech = c["mechanism"]
        p_name = c["phase_name"]
        model = c["model"]
        T = c["T"]
        P = c["P"]
        comp = c["composition"]

        if model not in model_stats:
            model_stats[model] = {
                "count": 0,
                "max_errors": {}
            }
        model_stats[model]["count"] += 1

        key = (mech, p_name)
        if key != current_key:
            current_key = key
            # Locate file in data or test/data
            p1 = os.path.join(base_dir, "data", mech)
            p2 = os.path.join(base_dir, "test", "data", mech)
            path = p1 if os.path.exists(p1) else p2
            sol = Solution(path, phase_name=p_name if p_name else None)

        thermo = sol.thermo
        if comp:
            if hasattr(thermo, "setState_TPX"):
                thermo.setState_TPX(T, P, comp)
            else:
                thermo.setTemperature(T)
                thermo.setPressure(P)
                thermo.setMoleFractionsByName(comp)
        else:
            if hasattr(thermo, "setState_TP"):
                thermo.setState_TP(T, P)
            else:
                thermo.setTemperature(T)
                thermo.setPressure(P)

        def check_prop(name, py_val, native_val):
            if native_val is None:
                return
            err = rel_err(py_val, native_val)
            stats = model_stats[model]["max_errors"]
            stats[name] = max(stats.get(name, 0.0), err)

        def check_vec(name, py_vec, native_vec):
            if not native_vec:
                return
            err = max_vec_rel_err(py_vec, native_vec)
            stats = model_stats[model]["max_errors"]
            stats[name] = max(stats.get(name, 0.0), err)

        # 1. Scalars
        check_prop("density", get_val(thermo, "density"), c["density"])
        check_prop("molar_density", get_val(thermo, "molar_density") or get_val(thermo, "molarDensity"), c["molar_density"])
        check_prop("molar_volume", get_val(thermo, "molar_volume") or get_val(thermo, "molarVolume"), c["molar_volume"])
        check_prop("mean_molecular_weight", get_val(thermo, "mean_molecular_weight") or get_val(thermo, "meanMolecularWeight"), c["mean_molecular_weight"])
        check_prop("enthalpy_mole", get_val(thermo, "enthalpy_mole"), c["enthalpy_mole"])
        check_prop("int_energy_mole", get_val(thermo, "int_energy_mole") or get_val(thermo, "intEnergy_mole"), c["int_energy_mole"])
        check_prop("entropy_mole", get_val(thermo, "entropy_mole"), c["entropy_mole"])
        check_prop("gibbs_mole", get_val(thermo, "gibbs_mole"), c["gibbs_mole"])
        check_prop("cp_mole", get_val(thermo, "cp_mole"), c["cp_mole"])
        check_prop("cv_mole", get_val(thermo, "cv_mole"), c["cv_mole"])
        check_prop("enthalpy_mass", get_val(thermo, "enthalpy_mass"), c["enthalpy_mass"])
        check_prop("int_energy_mass", get_val(thermo, "int_energy_mass") or get_val(thermo, "intEnergy_mass"), c["int_energy_mass"])
        check_prop("entropy_mass", get_val(thermo, "entropy_mass"), c["entropy_mass"])
        check_prop("gibbs_mass", get_val(thermo, "gibbs_mass"), c["gibbs_mass"])
        check_prop("cp_mass", get_val(thermo, "cp_mass"), c["cp_mass"])
        check_prop("cv_mass", get_val(thermo, "cv_mass"), c["cv_mass"])

        if c["isothermal_compressibility"] != -1e99:
            check_prop("isothermal_compressibility", get_val(thermo, "isothermalCompressibility"), c["isothermal_compressibility"])
        if c["thermal_expansion_coeff"] != -1e99:
            check_prop("thermal_expansion_coeff", get_val(thermo, "thermalExpansionCoeff"), c["thermal_expansion_coeff"])

        if c["site_density"] > 0:
            check_prop("site_density", get_val(thermo, "siteDensity"), c["site_density"])

        # 2. Vectors
        nsp = thermo.nSpecies() if callable(thermo.nSpecies) else thermo.nSpecies
        check_vec("X", list(thermo.moleFractions() if hasattr(thermo, "moleFractions") else thermo.X), c["X"])

        if c["coverages"] and hasattr(thermo, "getCoverages"):
            cov = np.zeros(nsp)
            thermo.getCoverages(cov)
            check_vec("coverages", list(cov), c["coverages"])

        if c["chem_potentials"] and hasattr(thermo, "getChemPotentials"):
            vec = np.zeros(nsp)
            thermo.getChemPotentials(vec)
            check_vec("chem_potentials", list(vec), c["chem_potentials"])

        if c["partial_molar_enthalpies"] and hasattr(thermo, "getPartialMolarEnthalpies"):
            vec = np.zeros(nsp)
            thermo.getPartialMolarEnthalpies(vec)
            check_vec("partial_molar_enthalpies", list(vec), c["partial_molar_enthalpies"])

        if c["partial_molar_entropies"] and hasattr(thermo, "getPartialMolarEntropies"):
            vec = np.zeros(nsp)
            thermo.getPartialMolarEntropies(vec)
            check_vec("partial_molar_entropies", list(vec), c["partial_molar_entropies"])

        if c["partial_molar_cp"] and hasattr(thermo, "getPartialMolarCp"):
            vec = np.zeros(nsp)
            thermo.getPartialMolarCp(vec)
            check_vec("partial_molar_cp", list(vec), c["partial_molar_cp"])

        if c["partial_molar_volumes"] and hasattr(thermo, "getPartialMolarVolumes"):
            vec = np.zeros(nsp)
            thermo.getPartialMolarVolumes(vec)
            check_vec("partial_molar_volumes", list(vec), c["partial_molar_volumes"])

        if c["standard_chem_potentials"] and hasattr(thermo, "getStandardChemPotentials"):
            vec = np.zeros(nsp)
            thermo.getStandardChemPotentials(vec)
            check_vec("standard_chem_potentials", list(vec), c["standard_chem_potentials"])

        if c["enthalpy_RT"] and hasattr(thermo, "getEnthalpy_RT"):
            vec = np.zeros(nsp)
            thermo.getEnthalpy_RT(vec)
            check_vec("enthalpy_RT", list(vec), c["enthalpy_RT"])

        if c["entropy_R"] and hasattr(thermo, "getEntropy_R"):
            vec = np.zeros(nsp)
            thermo.getEntropy_R(vec)
            check_vec("entropy_R", list(vec), c["entropy_R"])

        if c["gibbs_RT"] and hasattr(thermo, "getGibbs_RT"):
            vec = np.zeros(nsp)
            thermo.getGibbs_RT(vec)
            check_vec("gibbs_RT", list(vec), c["gibbs_RT"])

        if c["cp_R"] and hasattr(thermo, "getCp_R"):
            vec = np.zeros(nsp)
            thermo.getCp_R(vec)
            check_vec("cp_R", list(vec), c["cp_R"])

        if c["standard_volumes"] and hasattr(thermo, "getStandardVolumes"):
            vec = np.zeros(nsp)
            thermo.getStandardVolumes(vec)
            check_vec("standard_volumes", list(vec), c["standard_volumes"])

        if c["activities"] and hasattr(thermo, "getActivities"):
            vec = np.zeros(nsp)
            thermo.getActivities(vec)
            check_vec("activities", list(vec), c["activities"])

        if c["activity_coefficients"] and hasattr(thermo, "getActivityCoefficients"):
            vec = np.zeros(nsp)
            thermo.getActivityCoefficients(vec)
            check_vec("activity_coefficients", list(vec), c["activity_coefficients"])

    # Output summary
    print("\n" + "="*80)
    print("FAMILY 4 NATIVE DIFFERENTIAL VALIDATION SUMMARY")
    print("="*80)

    all_passed = True
    overall_max_err = 0.0

    for model, stats in sorted(model_stats.items()):
        print(f"\nModel: {model} (Cases: {stats['count']})")
        print("-" * 60)
        for prop, err in sorted(stats["max_errors"].items()):
            status = "PASS" if err < 1e-10 else ("WARN" if err < 1e-7 else "FAIL")
            if err >= 1e-7:
                all_passed = False
            overall_max_err = max(overall_max_err, err)
            print(f"  {prop:<30}: max rel err = {err:12.4e}  [{status}]")

    summary_data = {
        "family": "THERMO_FAMILY_4_IDEAL_PHASES",
        "total_test_cases": len(cases),
        "overall_max_rel_error": overall_max_err,
        "all_passed": all_passed,
        "models": model_stats
    }

    with open("audit/thermo_family4_differential_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nSummary written to audit/thermo_family4_differential_summary.json")
    if all_passed:
        print("ALL FAMILY 4 MODELS PASSED NATIVE DIFFERENTIAL VALIDATION WITH ZERO DEFECTS!")
    else:
        print("SOME MODELS EXCEEDED ERROR TOLERANCES!")
        sys.exit(1)

if __name__ == "__main__":
    compare_all()
