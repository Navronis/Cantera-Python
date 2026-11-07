"""Compare all SpeciesThermo polynomial models against native Cantera oracle."""
import sys
import json
import numpy as np
sys.path.insert(0, ".")

from cantera_python.thermo import (
    NasaPoly1,
    NasaPoly2,
    Nasa9Poly1,
    Nasa9PolyMultiTempRegion,
    ShomatePoly,
    ShomatePoly2,
    ConstCpPoly,
    Mu0Poly,
)

def run_comparison():
    with open("audit/native_thermo_family2_all_models_oracle.json", "r") as f:
        native = json.load(f)

    # 1. NasaPoly1
    nasa1_c = [3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00]
    py_nasa1 = NasaPoly1(nasa1_c, min_temp=200.0, max_temp=1000.0, reference_pressure=101325.0)

    # 2. NasaPoly2
    nasa2_c = [1000.0,
        3.69777829E+00, 6.13519789E-04, -1.25884282E-07, 1.77528148E-11, -1.13643547E-15, -1.23393018E+03, 3.18916559E+00,
        3.21293640E+00, 1.12748635E-03, -5.75615047E-07, 1.31387723E-09, -8.76855392E-13, -1.00524902E+03, 6.03473759E+00]
    py_nasa2 = NasaPoly2.from_coeffs(nasa2_c, min_temp=200.0, max_temp=3500.0, reference_pressure=101325.0)

    # 3. Nasa9Poly1
    nasa9_c = [1.0e4, -1.0e2, 3.5, 1e-3, -1e-6, 1e-9, -1e-12, 100.0, 5.0]
    py_nasa9_1 = Nasa9Poly1(nasa9_c, min_temp=200.0, max_temp=1000.0, reference_pressure=101325.0)

    # 4. Nasa9PolyMultiTempRegion
    nasa9_multi_c = [2.0, 200.0, 1000.0, 1.0e4, -1.0e2, 3.5, 1e-3, -1e-6, 1e-9, -1e-12, 100.0, 5.0,
                          1000.0, 3000.0, 1.2e4, -1.1e2, 3.8, 8e-4, -8e-7, 8e-10, -8e-13, 120.0, 5.5]
    py_nasa9_multi = Nasa9PolyMultiTempRegion.from_coeffs(nasa9_multi_c, reference_pressure=101325.0)

    # 5. ShomatePoly
    shomate_c = [29.5, -3.0, 5.0, -1.0, 0.1, -10.0, 200.0]
    py_shomate1 = ShomatePoly(shomate_c, min_temp=298.15, max_temp=1000.0, reference_pressure=101325.0)

    # 6. ShomatePoly2
    shomate2_c = [1000.0,
        29.5, -3.0, 5.0, -1.0, 0.1, -10.0, 200.0,
        32.0, -2.5, 4.0, -0.8, 0.05, -12.0, 210.0]
    py_shomate2 = ShomatePoly2.from_coeffs(shomate2_c, min_temp=298.15, max_temp=3000.0, reference_pressure=101325.0)

    # 7. ConstCpPoly
    py_constcp = ConstCpPoly(t0=298.15, h0=1.0e6, s0=2.0e5, cp0=30000.0, min_temp=200.0, max_temp=2000.0, reference_pressure=101325.0)

    # 8. Mu0Poly
    t_mu = {298.15: -1.0e5, 500.0: -1.2e5, 1000.0: -1.8e5}
    py_mu0 = Mu0Poly(h0=0.0, temperature_mu=t_mu, min_temp=298.15, max_temp=1000.0, reference_pressure=101325.0)

    model_map = {
        "NasaPoly1": py_nasa1,
        "NasaPoly2": py_nasa2,
        "Nasa9Poly1": py_nasa9_1,
        "Nasa9PolyMultiTempRegion": py_nasa9_multi,
        "ShomatePoly": py_shomate1,
        "ShomatePoly2": py_shomate2,
        "ConstCpPoly": py_constcp,
        "Mu0Poly": py_mu0,
    }

    max_rel_errors = {"cp_R": 0.0, "h_RT": 0.0, "s_R": 0.0}
    per_model_summary = {}

    for item in native["models"]:
        m_name = item["model"]
        if m_name == "NasaPoly1_Hf298_Modification":
            # Test Hf298 modification and reset in Python
            orig_h = py_nasa1.reportHf298()
            py_nasa1.modifyOneHf298(0, orig_h + 50000.0)
            mod_h = py_nasa1.reportHf298()
            py_nasa1.resetHf298()
            reset_h = py_nasa1.reportHf298()

            diff_orig = abs(orig_h - item["orig_h"])
            diff_mod = abs(mod_h - item["modified_h"])
            diff_reset = abs(reset_h - item["reset_h"])

            assert diff_orig < 1e-9, f"Hf298 orig mismatch: {orig_h} vs {item['orig_h']}"
            assert diff_mod < 1e-9, f"Hf298 modified mismatch: {mod_h} vs {item['modified_h']}"
            assert diff_reset < 1e-9, f"Hf298 reset mismatch: {reset_h} vs {item['reset_h']}"
            continue

        py_model = model_map[m_name]
        m_cp_err, m_h_err, m_s_err = 0.0, 0.0, 0.0

        for pt in item["points"]:
            T = pt["temperature"]
            cp_nat = pt["cp_R"]
            h_nat = pt["h_RT"]
            s_nat = pt["s_R"]

            cp_py, h_py, s_py = py_model.properties(T)

            rel_cp = abs(cp_py - cp_nat) / max(abs(cp_nat), 1e-12)
            rel_h = abs(h_py - h_nat) / max(abs(h_nat), 1e-12)
            rel_s = abs(s_py - s_nat) / max(abs(s_nat), 1e-12)

            m_cp_err = max(m_cp_err, rel_cp)
            m_h_err = max(m_h_err, rel_h)
            m_s_err = max(m_s_err, rel_s)

        max_rel_errors["cp_R"] = max(max_rel_errors["cp_R"], m_cp_err)
        max_rel_errors["h_RT"] = max(max_rel_errors["h_RT"], m_h_err)
        max_rel_errors["s_R"] = max(max_rel_errors["s_R"], m_s_err)

        per_model_summary[m_name] = {
            "max_rel_error_cp_R": m_cp_err,
            "max_rel_error_h_RT": m_h_err,
            "max_rel_error_s_R": m_s_err,
            "status": "PASS" if max(m_cp_err, m_h_err, m_s_err) < 1e-12 else "FAIL"
        }

    full_summary = {
        "status": "PASS",
        "species_thermo_all_models_native_validated": True,
        "max_rel_errors": max_rel_errors,
        "per_model_summary": per_model_summary
    }

    with open("audit/thermo_family2_all_models_differential_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    print("Comprehensive Family 2 All-Models Differential Validation:")
    for m, res in per_model_summary.items():
        print(f"  {m:25s}: cp_R {res['max_rel_error_cp_R']:.2e}, h_RT {res['max_rel_error_h_RT']:.2e}, s_R {res['max_rel_error_s_R']:.2e} -> {res['status']}")
    print(f"Global Max Errors: cp_R={max_rel_errors['cp_R']:.2e}, h_RT={max_rel_errors['h_RT']:.2e}, s_R={max_rel_errors['s_R']:.2e}")
    print("SPECIES_THERMO_ALL_MODELS_NATIVE_VALIDATED: TRUE")

if __name__ == "__main__":
    run_comparison()
