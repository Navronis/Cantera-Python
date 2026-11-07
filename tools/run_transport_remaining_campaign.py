"""Differential Campaign Runner for Remaining Transport Models.

Compares:
- audit/native_transport_remaining_oracle_results.json
- audit/python_transport_remaining_port_results.json

Produces:
- audit/transport_remaining_differential_summary.json
"""
import json
import math
import sys
import numpy as np

def rel_diff(val_py, val_nat):
    denom = max(abs(val_nat), 1e-30)
    return abs(val_py - val_nat) / denom

def main():
    with open("audit/native_transport_remaining_oracle_results.json", "r", encoding="utf-8") as f:
        nat_data = json.load(f)
    with open("audit/python_transport_remaining_port_results.json", "r", encoding="utf-8") as f:
        py_data = json.load(f)

    nat_results = {r["case_id"]: r for r in nat_data["results"]}
    py_results = {r["case_id"]: r for r in py_data["results"]}

    summary = {
        "campaign": "transport_remaining_differential",
        "pinned_commit": nat_data.get("pinned_commit", "726522be4e2a13454d8415b7ef799d621f665cf3"),
        "total_cases": len(nat_results),
        "passed_cases": 0,
        "failed_cases": 0,
        "case_summaries": []
    }

    max_rel_errors = {
        "viscosity": 0.0,
        "thermal_conductivity": 0.0,
        "bulk_viscosity": 0.0,
        "electrical_conductivity": 0.0,
        "mobilities": 0.0,
        "mix_diff_coeffs": 0.0,
        "thermal_diff_coeffs": 0.0,
    }

    # Tolerances: analytical properties (viscosity, water, electrical) 1e-4;
    # polynomial fitting vs evaluation in ion collision integrals allows up to 2.0e-2 (native fit tolerance is 3.5%)
    tol_visc = 1e-4
    tol_cond = 1e-4
    tol_mobi = 1e-5
    tol_diff = 1e-5
    tol_elec = 1e-4

    for case_id, nat_case in nat_results.items():
        if case_id not in py_results:
            summary["failed_cases"] += 1
            summary["case_summaries"].append({
                "case_id": case_id,
                "status": "FAIL",
                "error": "Case missing from python results"
            })
            continue

        py_case = py_results[case_id]
        category = nat_case.get("category", "")
        passed = True
        failures = []

        if "error" in py_case:
            passed = False
            failures.append(f"Python error: {py_case['error']}")
        elif "error" in nat_case:
            passed = False
            failures.append(f"Native error: {nat_case['error']}")
        else:
            if category == "water_transport":
                # Compare viscosity
                mu_nat = float(nat_case["viscosity"])
                mu_py = float(py_case["viscosity"])
                err_mu = rel_diff(mu_py, mu_nat)
                max_rel_errors["viscosity"] = max(max_rel_errors["viscosity"], err_mu)
                if err_mu > tol_visc:
                    passed = False
                    failures.append(f"viscosity err={err_mu:.3e} (py={mu_py}, nat={mu_nat})")

                # Compare thermal conductivity
                lam_nat = float(nat_case["thermal_conductivity"])
                lam_py = float(py_case["thermal_conductivity"])
                err_lam = rel_diff(lam_py, lam_nat)
                max_rel_errors["thermal_conductivity"] = max(max_rel_errors["thermal_conductivity"], err_lam)
                if err_lam > tol_cond:
                    passed = False
                    failures.append(f"thermal_cond err={err_lam:.3e} (py={lam_py}, nat={lam_nat})")

                # Compare bulk viscosity (both 0.0)
                err_bulk = abs(float(py_case.get("bulk_viscosity", 0.0)) - float(nat_case.get("bulk_viscosity", 0.0)))
                max_rel_errors["bulk_viscosity"] = max(max_rel_errors["bulk_viscosity"], err_bulk)
                if err_bulk > 1e-12:
                    passed = False
                    failures.append(f"bulk_visc err={err_bulk:.3e}")

            elif category == "unity_lewis_transport":
                mu_nat = float(nat_case["viscosity"])
                mu_py = float(py_case["viscosity"])
                err_mu = rel_diff(mu_py, mu_nat)
                max_rel_errors["viscosity"] = max(max_rel_errors["viscosity"], err_mu)
                if err_mu > tol_visc:
                    passed = False
                    failures.append(f"viscosity err={err_mu:.3e}")

                lam_nat = float(nat_case["thermal_conductivity"])
                lam_py = float(py_case["thermal_conductivity"])
                err_lam = rel_diff(lam_py, lam_nat)
                max_rel_errors["thermal_conductivity"] = max(max_rel_errors["thermal_conductivity"], err_lam)
                if err_lam > tol_cond:
                    passed = False
                    failures.append(f"thermal_cond err={err_lam:.3e}")

                # Mix diff coeffs
                d_nat = np.array(nat_case["mix_diff_coeffs"], dtype=float)
                d_py = np.array(py_case["mix_diff_coeffs"], dtype=float)
                if len(d_nat) != len(d_py):
                    passed = False
                    failures.append(f"mix_diff_coeffs len mismatch {len(d_py)} != {len(d_nat)}")
                else:
                    max_d_err = np.max(np.abs(d_py - d_nat) / np.maximum(np.abs(d_nat), 1e-30))
                    max_rel_errors["mix_diff_coeffs"] = max(max_rel_errors["mix_diff_coeffs"], float(max_d_err))
                    if max_d_err > tol_diff:
                        passed = False
                        failures.append(f"mix_diff_coeffs max err={max_d_err:.3e}")

            elif category == "ion_gas_transport":
                mu_nat = float(nat_case["viscosity"])
                mu_py = float(py_case["viscosity"])
                err_mu = rel_diff(mu_py, mu_nat)
                max_rel_errors["viscosity"] = max(max_rel_errors["viscosity"], err_mu)
                if err_mu > tol_visc:
                    passed = False
                    failures.append(f"viscosity err={err_mu:.3e}")

                lam_nat = float(nat_case["thermal_conductivity"])
                lam_py = float(py_case["thermal_conductivity"])
                err_lam = rel_diff(lam_py, lam_nat)
                max_rel_errors["thermal_conductivity"] = max(max_rel_errors["thermal_conductivity"], err_lam)
                if err_lam > tol_cond:
                    passed = False
                    failures.append(f"thermal_cond err={err_lam:.3e}")

                sigma_nat = float(nat_case["electrical_conductivity"])
                sigma_py = float(py_case["electrical_conductivity"])
                abs_sigma = abs(sigma_py - sigma_nat)
                err_sigma = rel_diff(sigma_py, sigma_nat) if max(sigma_py, sigma_nat) > 1e-12 else 0.0
                max_rel_errors["electrical_conductivity"] = max(max_rel_errors["electrical_conductivity"], err_sigma)
                if abs_sigma > 1e-12 and err_sigma > tol_elec:
                    passed = False
                    failures.append(f"electrical_cond err={err_sigma:.3e}")

                # Mobilities
                mobi_nat = np.array(nat_case["mobilities"], dtype=float)
                mobi_py = np.array(py_case["mobilities"], dtype=float)
                max_mobi_err = np.max(np.abs(mobi_py - mobi_nat) / np.maximum(np.abs(mobi_nat), 1e-30))
                max_rel_errors["mobilities"] = max(max_rel_errors["mobilities"], float(max_mobi_err))
                if max_mobi_err > tol_mobi:
                    passed = False
                    failures.append(f"mobilities max err={max_mobi_err:.3e}")

                # Mix diff coeffs
                d_nat = np.array(nat_case["mix_diff_coeffs"], dtype=float)
                d_py = np.array(py_case["mix_diff_coeffs"], dtype=float)
                max_d_err = np.max(np.abs(d_py - d_nat) / np.maximum(np.abs(d_nat), 1e-30))
                max_rel_errors["mix_diff_coeffs"] = max(max_rel_errors["mix_diff_coeffs"], float(max_d_err))
                if max_d_err > tol_diff:
                    passed = False
                    failures.append(f"mix_diff_coeffs max err={max_d_err:.3e}")

            elif category == "transport_factory":
                nat_act = nat_case.get("actual_model", "")
                py_act = py_case.get("actual_model", "")
                # Normalize case (e.g. 'Water' vs 'water' or 'unity-Lewis-number')
                if nat_act.lower() != py_act.lower():
                    passed = False
                    failures.append(f"factory model mismatch: py={py_act} != nat={nat_act}")

        status = "PASS" if passed else "FAIL"
        if passed:
            summary["passed_cases"] += 1
        else:
            summary["failed_cases"] += 1

        summary["case_summaries"].append({
            "case_id": case_id,
            "category": category,
            "status": status,
            "failures": failures
        })

    summary["max_relative_errors"] = max_rel_errors
    summary["pass_rate"] = summary["passed_cases"] / summary["total_cases"] if summary["total_cases"] > 0 else 0.0

    with open("audit/transport_remaining_differential_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Differential Campaign Summary: {summary['passed_cases']}/{summary['total_cases']} PASSED ({summary['pass_rate']*100:.2f}%)")
    print(f"Max Relative Errors:")
    for k, v in max_rel_errors.items():
        print(f"  {k:25s}: {v:.3e}")
    print("=" * 60)

    if summary["failed_cases"] > 0:
        print("\nFailed Cases:")
        for cs in summary["case_summaries"]:
            if cs["status"] == "FAIL":
                print(f"  {cs['case_id']}: {cs['failures']}")

if __name__ == "__main__":
    main()
