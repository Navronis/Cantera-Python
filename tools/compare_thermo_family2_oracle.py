"""Differential comparison of Python SpeciesThermo polynomial evaluators against Native Cantera."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import cantera_python as ct

def main():
    oracle_path = "audit/native_thermo_family2_oracle_results.json"
    with open(oracle_path, "r", encoding="utf-8") as f:
        nat_data = json.load(f)

    print("Comparing Python SpeciesThermo polynomial models against Native Oracle...")
    sol = ct.Solution("gri30.yaml")
    ph = sol.thermo

    all_passed = True
    max_errs = {"cp_R": 0.0, "h_RT": 0.0, "s_R": 0.0}

    for sp_eval in nat_data["species_evaluations"]:
        sp_name = sp_eval["species"]
        k = sol.species_index(sp_name)
        # Find species thermo model
        species_obj = ph.species[k]
        thermo_model = species_obj.thermo

        for pt in sp_eval["points"]:
            T = pt["temperature"]
            cp_nat = pt["cp_R"]
            h_nat = pt["h_RT"]
            s_nat = pt["s_R"]

            cp_py, h_py, s_py = thermo_model.updatePropertiesTemp(T)

            err_cp = abs(cp_py - cp_nat) / max(abs(cp_nat), 1e-15)
            err_h = abs(h_py - h_nat) / max(abs(h_nat), 1e-15)
            err_s = abs(s_py - s_nat) / max(abs(s_nat), 1e-15)

            max_errs["cp_R"] = max(max_errs["cp_R"], err_cp)
            max_errs["h_RT"] = max(max_errs["h_RT"], err_h)
            max_errs["s_R"] = max(max_errs["s_R"], err_s)

            if max(err_cp, err_h, err_s) > 1e-12:
                all_passed = False
                print(f"Mismatch for {sp_name} at T={T}: cp_err={err_cp:.3e}, h_err={err_h:.3e}, s_err={err_s:.3e}")

    summary = {
        "campaign": "thermo_family2_speciesthemo_differential",
        "pinned_commit": nat_data["pinned_commit"],
        "max_relative_errors": max_errs,
        "all_passed": all_passed,
        "status": "PASS" if all_passed else "FAIL"
    }

    out_path = "audit/thermo_family2_differential_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFamily 2 Polynomial Check Summary ({summary['status']}):")
    print(f"  Max Rel Error cp_R: {max_errs['cp_R']:.4e}")
    print(f"  Max Rel Error h_RT: {max_errs['h_RT']:.4e}")
    print(f"  Max Rel Error s_R : {max_errs['s_R']:.4e}")
    assert all_passed, "Family 2 polynomial check failed"

if __name__ == "__main__":
    main()
