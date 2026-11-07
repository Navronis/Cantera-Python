"""Run differential comparison campaign between native Cantera OneDim oracle and Python port.

Generates audit/onedim_differential_summary.json.
"""

import json
import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main():
    native_path = repo_root / "audit" / "native_onedim_oracle_results.json"
    python_path = repo_root / "audit" / "python_onedim_port_results.json"
    summary_path = repo_root / "audit" / "onedim_differential_summary.json"

    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)["cases"]

    with open(python_path, "r", encoding="utf-8") as f:
        python_data = json.load(f)["cases"]

    assert len(native_data) == len(python_data), f"Count mismatch: native={len(native_data)}, py={len(python_data)}"

    total_cases = len(native_data)
    passed_cases = 0
    failed_cases = 0
    detailed_comparisons = []

    structure_mismatches = 0
    residual_mismatches = 0
    jacobian_mismatches = 0
    norm_mismatches = 0
    exception_mismatches = 0

    max_rel_norm_diff = 0.0
    max_abs_res_diff = 0.0
    max_abs_jac_diff = 0.0

    for i in range(total_cases):
        n_case = native_data[i]
        p_case = python_data[i]

        case_id = n_case["case_id"]
        assert case_id == p_case["case_id"], f"Case ID mismatch at {i}: {case_id} vs {p_case['case_id']}"

        n_exc = n_case["exception_thrown"]
        p_exc = p_case["exception_thrown"]

        case_passed = True
        notes = []

        if n_exc != p_exc:
            case_passed = False
            exception_mismatches += 1
            notes.append(f"Exception mismatch: native={n_exc}, py={p_exc}")
        elif n_exc:
            # Both threw expected exceptions
            notes.append(f"Expected exception triggered and matched in both.")
        else:
            # Check structure
            for field in ["n_domains", "points", "size", "bandwidth", "first_component_name"]:
                if n_case[field] != p_case[field]:
                    case_passed = False
                    structure_mismatches += 1
                    notes.append(f"{field} mismatch: native={n_case[field]}, py={p_case[field]}")

            # Check bounds
            for field in ["first_upper_bound", "first_lower_bound"]:
                if abs(n_case[field] - p_case[field]) > 1e-10:
                    case_passed = False
                    structure_mismatches += 1
                    notes.append(f"{field} mismatch: native={n_case[field]}, py={p_case[field]}")

            # Check weightedNorm
            n_wn = n_case["weighted_norm"]
            p_wn = p_case["weighted_norm"]
            rel_wn = abs(n_wn - p_wn) / (abs(n_wn) + 1e-12)
            if rel_wn > max_rel_norm_diff:
                max_rel_norm_diff = rel_wn
            if rel_wn > 1e-10:
                case_passed = False
                norm_mismatches += 1
                notes.append(f"weighted_norm mismatch: native={n_wn}, py={p_wn}, rel_diff={rel_wn}")

            # Check residual
            n_res = n_case["residual"]
            p_res = p_case["residual"]
            if len(n_res) != len(p_res):
                case_passed = False
                residual_mismatches += 1
                notes.append(f"residual length mismatch: native={len(n_res)}, py={len(p_res)}")
            else:
                res_diff = max((abs(a - b) for a, b in zip(n_res, p_res)), default=0.0)
                if res_diff > max_abs_res_diff:
                    max_abs_res_diff = res_diff
                if res_diff > 1e-10:
                    case_passed = False
                    residual_mismatches += 1
                    notes.append(f"residual max abs diff: {res_diff}")

            # Check Jacobian diagonal
            n_jac = n_case["jacobian_diag"]
            p_jac = p_case["jacobian_diag"]
            if len(n_jac) != len(p_jac):
                case_passed = False
                jacobian_mismatches += 1
                notes.append(f"jacobian_diag length mismatch: native={len(n_jac)}, py={len(p_jac)}")
            else:
                jac_diff = max((abs(a - b) for a, b in zip(n_jac, p_jac)), default=0.0)
                if jac_diff > max_abs_jac_diff:
                    max_abs_jac_diff = jac_diff
                if jac_diff > 1e-7:
                    case_passed = False
                    jacobian_mismatches += 1
                    notes.append(f"jacobian_diag max abs diff: {jac_diff}")

        if case_passed:
            passed_cases += 1
        else:
            failed_cases += 1

        detailed_comparisons.append({
            "case_id": case_id,
            "passed": case_passed,
            "exception_thrown": n_exc,
            "notes": "; ".join(notes) if notes else "Exact match",
        })

    summary = {
        "campaign": "OneDim Differential Parity Campaign",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate_percent": round(100.0 * passed_cases / total_cases, 2),
        "structure_mismatches": structure_mismatches,
        "norm_mismatches": norm_mismatches,
        "residual_mismatches": residual_mismatches,
        "jacobian_mismatches": jacobian_mismatches,
        "exception_mismatches": exception_mismatches,
        "max_rel_norm_diff": max_rel_norm_diff,
        "max_abs_residual_diff": max_abs_res_diff,
        "max_abs_jacobian_diff": max_abs_jac_diff,
        "parity_achieved": (failed_cases == 0),
        "cases": detailed_comparisons,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"OneDim Campaign Completed: {passed_cases}/{total_cases} passed ({summary['pass_rate_percent']}%)")
    print(f"Max rel norm diff: {max_rel_norm_diff:.2e}")
    print(f"Max abs residual diff: {max_abs_res_diff:.2e}")
    print(f"Max abs jacobian diff: {max_abs_jac_diff:.2e}")
    print(f"Parity achieved: {summary['parity_achieved']}")

    if failed_cases > 0:
        for c in detailed_comparisons:
            if not c["passed"]:
                print(f"  FAILED: {c['case_id']}: {c['notes']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
