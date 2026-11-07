"""Differential validator for Thermodynamics Family 3: ThermoPhase Base."""
import sys
import json
import math
import numpy as np

sys.path.insert(0, ".")
from cantera_python.solution import Solution

def compare_all():
    with open("audit/native_thermo_family3_oracle.json") as f:
        data = json.load(f)
    cases = data["cases"]

    max_rel_errors = {}
    tolerances = {
        "density": 1e-12,
        "molar_density": 1e-12,
        "molar_volume": 1e-12,
        "mean_molecular_weight": 1e-12,
        "enthalpy_mole": 1e-11,
        "int_energy_mole": 1e-11,
        "entropy_mole": 1e-11,
        "gibbs_mole": 1e-11,
        "cp_mole": 1e-11,
        "cv_mole": 1e-11,
        "enthalpy_mass": 1e-11,
        "int_energy_mass": 1e-11,
        "entropy_mass": 1e-11,
        "gibbs_mass": 1e-11,
        "cp_mass": 1e-11,
        "cv_mass": 1e-11,
        "isothermal_compressibility": 1e-11,
        "thermal_expansion_coeff": 1e-11,
        "sound_speed": 1e-11,
        "internal_pressure": 1e-10,
        "equivalence_ratio": 1e-11,
        "stoich_air_fuel_ratio": 1e-11,
        "T_rec_hp": 1e-6,
        "T_rec_uv": 1e-6,
        "T_rec_sp": 1e-6,
        "T_rec_sv": 1e-6,
        "chem_potentials": 1e-11,
        "partial_molar_enthalpies": 1e-11,
        "partial_molar_entropies": 1e-11,
        "partial_molar_int_energies": 1e-11,
        "partial_molar_cp": 1e-11,
        "partial_molar_volumes": 1e-11,
        "standard_chem_potentials": 1e-11,
        "enthalpy_RT": 1e-12,
        "entropy_R": 1e-12,
        "gibbs_RT": 1e-12,
        "int_energy_RT": 1e-12,
        "cp_R": 1e-12,
        "standard_volumes": 1e-12,
        "enthalpy_RT_ref": 1e-12,
        "entropy_R_ref": 1e-12,
        "gibbs_RT_ref": 1e-12,
        "int_energy_RT_ref": 1e-12,
        "cp_R_ref": 1e-12,
        "standard_volumes_ref": 1e-12,
        "activities": 1e-12,
        "activity_coefficients": 1e-12,
        "ln_activity_coefficients": 1e-12,
    }

    current_mech = None
    sol = None

    for i, c in enumerate(cases):
        mech = c["mechanism"]
        if mech != current_mech:
            current_mech = mech
            import os
            if os.path.exists(f"../cantera-original/data/{mech}"):
                mech_path = f"../cantera-original/data/{mech}"
            else:
                mech_path = f"outputs/cantera-original/data/{mech}"
            sol = Solution(mech_path)

        thermo = sol.thermo
        T = c["T"]
        P = c["P"]
        comp = c["composition"]

        thermo.setState_TPX(T, P, comp)

        # 1. Scalar properties
        py_scalars = {
            "density": thermo.density,
            "molar_density": thermo.molar_density if hasattr(thermo, 'molar_density') else thermo.molarDensity(),
            "molar_volume": thermo.molarVolume(),
            "mean_molecular_weight": thermo.meanMolecularWeight(),
            "enthalpy_mole": thermo.enthalpy_mole,
            "int_energy_mole": thermo.int_energy_mole,
            "entropy_mole": thermo.entropy_mole,
            "gibbs_mole": thermo.gibbs_mole,
            "cp_mole": thermo.cp_mole,
            "cv_mole": thermo.cv_mole,
            "enthalpy_mass": thermo.enthalpy_mass,
            "int_energy_mass": thermo.int_energy_mass,
            "entropy_mass": thermo.entropy_mass,
            "gibbs_mass": thermo.gibbs_mass,
            "cp_mass": thermo.cp_mass,
            "cv_mass": thermo.cv_mass,
            "isothermal_compressibility": thermo.isothermalCompressibility(),
            "thermal_expansion_coeff": thermo.thermalExpansionCoeff(),
            "sound_speed": thermo.soundSpeed(),
            "internal_pressure": thermo.internalPressure() if callable(thermo.internalPressure) else thermo.internalPressure,
            "equivalence_ratio": thermo.equivalenceRatio() if callable(thermo.equivalenceRatio) else thermo.equivalenceRatio,
        }

        if c["stoich_air_fuel_ratio"] > 0:
            if mech == "h2o2.yaml":
                py_scalars["stoich_air_fuel_ratio"] = thermo.stoichAirFuelRatio("H2:1", "O2:1", basis="mole")
            else:
                py_scalars["stoich_air_fuel_ratio"] = thermo.stoichAirFuelRatio("CH4:1", "O2:1, N2:3.76", basis="mole")

        # 2. Inversions
        H_mass = c["enthalpy_mass"]
        U_mass = c["int_energy_mass"]
        S_mass = c["entropy_mass"]
        V_mass = 1.0 / c["density"]

        # HP
        thermo.setState_TP(T + 50.0, P)
        thermo.setState_HP(H_mass, P, rtol=1e-9)
        py_scalars["T_rec_hp"] = thermo.temperature

        # UV
        thermo.setState_TP(T - 50.0, P)
        thermo.setState_UV(U_mass, V_mass, rtol=1e-9)
        py_scalars["T_rec_uv"] = thermo.temperature

        # SP
        thermo.setState_TP(T + 50.0, P)
        thermo.setState_SP(S_mass, P, rtol=1e-9)
        py_scalars["T_rec_sp"] = thermo.temperature

        # SV
        thermo.setState_TP(T - 50.0, P)
        thermo.setState_SV(S_mass, V_mass, rtol=1e-9)
        py_scalars["T_rec_sv"] = thermo.temperature

        # Reset to exact state
        thermo.setState_TPX(T, P, comp)

        # 3. Vector properties
        nsp = thermo.nSpecies() if callable(thermo.nSpecies) else thermo.nSpecies
        py_vectors = {
            "chem_potentials": np.zeros(nsp),
            "partial_molar_enthalpies": np.zeros(nsp),
            "partial_molar_entropies": np.zeros(nsp),
            "partial_molar_int_energies": np.zeros(nsp),
            "partial_molar_cp": np.zeros(nsp),
            "partial_molar_volumes": np.zeros(nsp),
            "standard_chem_potentials": np.zeros(nsp),
            "enthalpy_RT": np.zeros(nsp),
            "entropy_R": np.zeros(nsp),
            "gibbs_RT": np.zeros(nsp),
            "int_energy_RT": np.zeros(nsp),
            "cp_R": np.zeros(nsp),
            "standard_volumes": np.zeros(nsp),
            "enthalpy_RT_ref": np.zeros(nsp),
            "entropy_R_ref": np.zeros(nsp),
            "gibbs_RT_ref": np.zeros(nsp),
            "int_energy_RT_ref": np.zeros(nsp),
            "cp_R_ref": np.zeros(nsp),
            "standard_volumes_ref": np.zeros(nsp),
            "activities": np.zeros(nsp),
            "activity_coefficients": np.zeros(nsp),
            "ln_activity_coefficients": np.zeros(nsp),
        }

        thermo.getChemPotentials(py_vectors["chem_potentials"])
        thermo.getPartialMolarEnthalpies(py_vectors["partial_molar_enthalpies"])
        thermo.getPartialMolarEntropies(py_vectors["partial_molar_entropies"])
        thermo.getPartialMolarIntEnergies(py_vectors["partial_molar_int_energies"])
        thermo.getPartialMolarCp(py_vectors["partial_molar_cp"])
        thermo.getPartialMolarVolumes(py_vectors["partial_molar_volumes"])
        thermo.getStandardChemPotentials(py_vectors["standard_chem_potentials"])
        thermo.getEnthalpy_RT(py_vectors["enthalpy_RT"])
        thermo.getEntropy_R(py_vectors["entropy_R"])
        thermo.getGibbs_RT(py_vectors["gibbs_RT"])
        thermo.getIntEnergy_RT(py_vectors["int_energy_RT"])
        thermo.getCp_R(py_vectors["cp_R"])
        thermo.getStandardVolumes(py_vectors["standard_volumes"])
        thermo.getEnthalpy_RT_ref(py_vectors["enthalpy_RT_ref"])
        thermo.getEntropy_R_ref(py_vectors["entropy_R_ref"])
        thermo.getGibbs_RT_ref(py_vectors["gibbs_RT_ref"])
        thermo.getIntEnergy_RT_ref(py_vectors["int_energy_RT_ref"])
        thermo.getCp_R_ref(py_vectors["cp_R_ref"])
        thermo.getStandardVolumes_ref(py_vectors["standard_volumes_ref"])
        thermo.getActivities(py_vectors["activities"])
        thermo.getActivityCoefficients(py_vectors["activity_coefficients"])
        thermo.getLnActivityCoefficients(py_vectors["ln_activity_coefficients"])

        # Compare scalars
        for key, val_py in py_scalars.items():
            val_nat = c[key]
            denom = max(abs(val_nat), 1e-12)
            rel_err = abs(val_py - val_nat) / denom
            max_rel_errors[key] = max(max_rel_errors.get(key, 0.0), rel_err)

        # Compare vectors
        for key, vec_py in py_vectors.items():
            vec_nat = np.array(c[key], dtype=float)
            diff = np.abs(vec_py - vec_nat)
            denom = np.maximum(np.abs(vec_nat), 1e-12)
            rel_err = float(np.max(diff / denom))
            max_rel_errors[key] = max(max_rel_errors.get(key, 0.0), rel_err)

    # Check against tolerances
    failures = []
    for key, err in max_rel_errors.items():
        tol = tolerances.get(key, 1e-9)
        if err > tol:
            failures.append(f"{key}: err={err:.3e} > tol={tol:.3e}")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "total_cases_evaluated": len(cases),
        "failures": failures,
        "max_relative_errors": max_rel_errors,
        "thermo_family3_numerical_parity": len(failures) == 0
    }

    with open("audit/thermo_family3_differential_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Thermodynamics Family 3 Differential Validation Results:")
    print(f"Total Cases: {len(cases)}")
    for k, v in sorted(max_rel_errors.items()):
        status_str = "PASS" if v <= tolerances.get(k, 1e-9) else "FAIL"
        print(f"  {k:30s}: max_rel_err = {v:.2e} [{status_str}]")

    if failures:
        print("\nFAILURES:")
        for fail in failures:
            print("  ", fail)
        sys.exit(1)
    else:
        print("\nALL 90 DIFFERENTIAL VALIDATION CASES PASSED NUMERICAL PARITY!")

if __name__ == "__main__":
    compare_all()
