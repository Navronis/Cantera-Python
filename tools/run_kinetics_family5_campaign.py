"""Family 5 Differential Validation Campaign against Pinned Native Oracle.

Loads audit/native_kinetics_family5_oracle_results.json and
audit/python_kinetics_family5_port_results.json, performs strict numerical
and structural comparison, and writes audit/kinetics_family5_differential_summary.json.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple


def compare_floats(val_py: float, val_nat: float, rtol: float = 1e-5, atol: float = 1e-8) -> Tuple[bool, float]:
    if math.isnan(val_py) and math.isnan(val_nat):
        return True, 0.0
    if math.isinf(val_py) and math.isinf(val_nat):
        return (val_py > 0) == (val_nat > 0), 0.0

    diff = abs(val_py - val_nat)
    denom = max(abs(val_py), abs(val_nat))
    rel_err = diff / denom if denom > atol else diff

    ok = (diff <= atol) or (rel_err <= rtol)
    return ok, rel_err


def run_campaign():
    native_path = Path("audit/native_kinetics_family5_oracle_results.json")
    python_path = Path("audit/python_kinetics_family5_port_results.json")
    summary_path = Path("audit/kinetics_family5_differential_summary.json")

    assert native_path.exists(), f"Missing native oracle results: {native_path}"
    assert python_path.exists(), f"Missing python port results: {python_path}"

    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)

    with open(python_path, "r", encoding="utf-8") as f:
        python_data = json.load(f)

    native_results = {r["case_id"]: r for r in native_data.get("results", [])}
    python_results = {r["case_id"]: r for r in python_data.get("results", [])}

    print(f"Comparing {len(python_results)} Python cases against {len(native_results)} Native cases...")

    case_reports = []
    total_passed = 0
    total_cases = len(native_results)
    max_global_rel_err = 0.0

    for case_id, nat in native_results.items():
        if case_id not in python_results:
            case_reports.append({
                "case_id": case_id,
                "status": "FAIL",
                "error": "Missing from Python results"
            })
            continue

        py = python_results[case_id]
        category = nat.get("category", "")
        passed = True
        discrepancies = []
        case_max_rel_err = 0.0

        if "error" in nat or "error" in py:
            if ("error" in nat) != ("error" in py):
                passed = False
                discrepancies.append(f"Error mismatch: nat={nat.get('error')}, py={py.get('error')}")
            case_reports.append({
                "case_id": case_id,
                "category": category,
                "passed": passed,
                "discrepancies": discrepancies,
                "max_rel_err": 0.0
            })
            if passed:
                total_passed += 1
            continue

        # 1. Compare forward_rate_constants
        if "forward_rate_constants" in nat and "forward_rate_constants" in py:
            nat_kf = nat["forward_rate_constants"]
            py_kf = py["forward_rate_constants"]
            if len(nat_kf) != len(py_kf):
                passed = False
                discrepancies.append(f"forward_rate_constants length mismatch: {len(py_kf)} vs {len(nat_kf)}")
            else:
                for idx, (vk_nat, vk_py) in enumerate(zip(nat_kf, py_kf)):
                    ok, r_err = compare_floats(vk_py, vk_nat, rtol=1e-5, atol=1e-8)
                    case_max_rel_err = max(case_max_rel_err, r_err)
                    if not ok:
                        passed = False
                        discrepancies.append(f"kf[{idx}] mismatch: py={vk_py:.10e}, nat={vk_nat:.10e}, rel_err={r_err:.2e}")

        # 2. Compare net_rates_of_progress
        if "net_rates_of_progress" in nat and "net_rates_of_progress" in py:
            nat_rop = nat["net_rates_of_progress"]
            py_rop = py["net_rates_of_progress"]
            if len(nat_rop) != len(py_rop):
                passed = False
                discrepancies.append(f"net_rates_of_progress length mismatch: {len(py_rop)} vs {len(nat_rop)}")
            else:
                for idx, (vr_nat, vr_py) in enumerate(zip(nat_rop, py_rop)):
                    ok, r_err = compare_floats(vr_py, vr_nat, rtol=1e-5, atol=1e-8)
                    case_max_rel_err = max(case_max_rel_err, r_err)
                    if not ok:
                        passed = False
                        discrepancies.append(f"rop[{idx}] mismatch: py={vr_py:.10e}, nat={vr_nat:.10e}, rel_err={r_err:.2e}")

        # 3. Compare twotemp_eval and twotemp_ddT_scaled
        if "twotemp_eval" in nat and "twotemp_eval" in py:
            ok, r_err = compare_floats(py["twotemp_eval"], nat["twotemp_eval"], rtol=1e-5, atol=1e-8)
            case_max_rel_err = max(case_max_rel_err, r_err)
            if not ok:
                passed = False
                discrepancies.append(f"twotemp_eval mismatch: py={py['twotemp_eval']:.10e}, nat={nat['twotemp_eval']:.10e}, rel_err={r_err:.2e}")

        if "twotemp_ddT_scaled" in nat and "twotemp_ddT_scaled" in py:
            ok, r_err = compare_floats(py["twotemp_ddT_scaled"], nat["twotemp_ddT_scaled"], rtol=1e-5, atol=1e-8)
            case_max_rel_err = max(case_max_rel_err, r_err)
            if not ok:
                passed = False
                discrepancies.append(f"twotemp_ddT_scaled mismatch: py={py['twotemp_ddT_scaled']:.10e}, nat={nat['twotemp_ddT_scaled']:.10e}, rel_err={r_err:.2e}")

        # 4. Compare custom delegator
        if "custom_rate_type" in nat and "custom_rate_type" in py:
            if nat["custom_rate_type"] != py["custom_rate_type"]:
                passed = False
                discrepancies.append(f"custom_rate_type mismatch: py={py['custom_rate_type']}, nat={nat['custom_rate_type']}")

        # 5. Compare multirate ops
        for key in ("rate_type", "multi_type", "caught_empty", "replace_ok", "caught_mismatch"):
            if key in nat and key in py:
                if nat[key] != py[key]:
                    passed = False
                    discrepancies.append(f"{key} mismatch: py={py[key]}, nat={nat[key]}")

        # 6. Compare kinetics derivatives
        for dkey in ("netProductionRates_ddT", "creationRates_ddT", "destructionRates_ddT", "netProductionRates_ddP"):
            if dkey in nat and dkey in py:
                nat_vec = nat[dkey]
                py_vec = py[dkey]
                if len(nat_vec) != len(py_vec):
                    passed = False
                    discrepancies.append(f"{dkey} length mismatch: {len(py_vec)} vs {len(nat_vec)}")
                else:
                    for idx, (vd_nat, vd_py) in enumerate(zip(nat_vec, py_vec)):
                        ok, r_err = compare_floats(vd_py, vd_nat, rtol=1e-5, atol=1e-8)
                        case_max_rel_err = max(case_max_rel_err, r_err)
                        if not ok:
                            passed = False
                            discrepancies.append(f"{dkey}[{idx}] mismatch: py={vd_py:.10e}, nat={vd_nat:.10e}, rel_err={r_err:.2e}")

        max_global_rel_err = max(max_global_rel_err, case_max_rel_err)
        if passed:
            total_passed += 1

        case_reports.append({
            "case_id": case_id,
            "category": category,
            "passed": passed,
            "discrepancies": discrepancies,
            "max_rel_err": case_max_rel_err
        })

    pass_rate = (total_passed / total_cases) * 100.0 if total_cases > 0 else 0.0
    summary = {
        "family": "FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "total_cases": total_cases,
        "total_passed": total_passed,
        "pass_rate_percent": pass_rate,
        "max_global_relative_error": max_global_rel_err,
        "all_passed": (total_passed == total_cases),
        "case_reports": case_reports
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DIFFERENTIAL CAMPAIGN RESULTS: FAMILY 5")
    print(f"Total Cases : {total_cases}")
    print(f"Passed      : {total_passed} ({pass_rate:.2f}%)")
    print(f"Max Rel Err : {max_global_rel_err:.2e}")
    print(f"All Passed  : {summary['all_passed']}")
    print("=" * 60)

    if not summary["all_passed"]:
        print("\nFailed cases:")
        for r in case_reports:
            if not r["passed"]:
                print(f"  - {r['case_id']}: {r['discrepancies']}")


if __name__ == "__main__":
    run_campaign()
