"""Run differential comparison campaign between native Cantera Sim1D oracle and Python port.

Generates audit/sim1d_differential_summary.json.
"""

import json
import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main():
    native_path = repo_root / "audit" / "native_sim1d_oracle_results.json"
    python_path = repo_root / "audit" / "python_sim1d_port_results.json"
    summary_path = repo_root / "audit" / "sim1d_differential_summary.json"

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
    refine_result_mismatches = 0
    grid_mismatches = 0
    value_mismatches = 0
    jacobian_mismatches = 0
    adjoint_mismatches = 0
    exception_mismatches = 0

    max_abs_grid_diff = 0.0
    max_abs_val_diff = 0.0
    max_abs_jac_diff = 0.0
    max_abs_adj_diff = 0.0

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
            notes.append("Expected exception triggered and matched in both.")
        else:
            # Check structure
            for field in ["n_domains", "points", "size", "bandwidth", "refine_result", "queried_max_points", "queried_ts_regrid_max"]:
                if n_case[field] != p_case[field]:
                    case_passed = False
                    structure_mismatches += 1
                    notes.append(f"{field} mismatch: native={n_case[field]}, py={p_case[field]}")

            # Check queried_value
            if abs(n_case["queried_value"] - p_case["queried_value"]) > 1e-10:
                case_passed = False
                value_mismatches += 1
                notes.append(f"queried_value mismatch: native={n_case['queried_value']}, py={p_case['queried_value']}")

            # Check queried_criteria
            if len(n_case["queried_criteria"]) != len(p_case["queried_criteria"]):
                case_passed = False
                notes.append(f"queried_criteria length mismatch: {len(n_case['queried_criteria'])} vs {len(p_case['queried_criteria'])}")
            else:
                for c1, c2 in zip(n_case["queried_criteria"], p_case["queried_criteria"]):
                    if abs(c1 - c2) > 1e-10:
                        case_passed = False
                        notes.append(f"queried_criteria diff: {c1} vs {c2}")

            # Check domain_grids
            if len(n_case["domain_grids"]) != len(p_case["domain_grids"]):
                case_passed = False
                grid_mismatches += 1
                notes.append(f"domain_grids count mismatch: {len(n_case['domain_grids'])} vs {len(p_case['domain_grids'])}")
            else:
                for d_idx, (ng, pg) in enumerate(zip(n_case["domain_grids"], p_case["domain_grids"])):
                    if len(ng) != len(pg):
                        case_passed = False
                        grid_mismatches += 1
                        notes.append(f"domain {d_idx} grid points mismatch: native={len(ng)}, py={len(pg)}")
                    else:
                        g_diff = max((abs(a - b) for a, b in zip(ng, pg)), default=0.0)
                        if g_diff > max_abs_grid_diff:
                            max_abs_grid_diff = g_diff
                        if g_diff > 1e-10:
                            case_passed = False
                            grid_mismatches += 1
                            notes.append(f"domain {d_idx} grid max diff: {g_diff}")

            # Check domain_values
            if len(n_case["domain_values"]) != len(p_case["domain_values"]):
                case_passed = False
                value_mismatches += 1
                notes.append(f"domain_values count mismatch: {len(n_case['domain_values'])} vs {len(p_case['domain_values'])}")
            else:
                for d_idx, (nv, pv) in enumerate(zip(n_case["domain_values"], p_case["domain_values"])):
                    if len(nv) != len(pv):
                        case_passed = False
                        value_mismatches += 1
                        notes.append(f"domain {d_idx} values count mismatch: native={len(nv)}, py={len(pv)}")
                    else:
                        v_diff = max((abs(a - b) for a, b in zip(nv, pv)), default=0.0)
                        if v_diff > max_abs_val_diff:
                            max_abs_val_diff = v_diff
                        if v_diff > 1e-10:
                            case_passed = False
                            value_mismatches += 1
                            notes.append(f"domain {d_idx} values max diff: {v_diff}")

            # Check jacobian_diag
            n_jac = n_case["jacobian_diag"]
            p_jac = p_case["jacobian_diag"]
            if len(n_jac) != len(p_jac):
                case_passed = False
                jacobian_mismatches += 1
                notes.append(f"jacobian_diag length mismatch: {len(n_jac)} vs {len(p_jac)}")
            elif n_jac:
                j_diff = max((abs(a - b) for a, b in zip(n_jac, p_jac)), default=0.0)
                if j_diff > max_abs_jac_diff:
                    max_abs_jac_diff = j_diff
                if j_diff > 1e-7:
                    case_passed = False
                    jacobian_mismatches += 1
                    notes.append(f"jacobian_diag max diff: {j_diff}")

            # Check adjoint_lambda
            n_adj = n_case["adjoint_lambda"]
            p_adj = p_case["adjoint_lambda"]
            if len(n_adj) != len(p_adj):
                case_passed = False
                adjoint_mismatches += 1
                notes.append(f"adjoint_lambda length mismatch: {len(n_adj)} vs {len(p_adj)}")
            elif n_adj:
                adj_diff = max((abs(a - b) for a, b in zip(n_adj, p_adj)), default=0.0)
                if adj_diff > max_abs_adj_diff:
                    max_abs_adj_diff = adj_diff
                if adj_diff > 1e-7:
                    case_passed = False
                    adjoint_mismatches += 1
                    notes.append(f"adjoint_lambda max diff: {adj_diff}")

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
        "campaign": "Sim1D Differential Parity Campaign",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate_percent": round(100.0 * passed_cases / total_cases, 2),
        "structure_mismatches": structure_mismatches,
        "refine_result_mismatches": refine_result_mismatches,
        "grid_mismatches": grid_mismatches,
        "value_mismatches": value_mismatches,
        "jacobian_mismatches": jacobian_mismatches,
        "adjoint_mismatches": adjoint_mismatches,
        "exception_mismatches": exception_mismatches,
        "max_abs_grid_diff": max_abs_grid_diff,
        "max_abs_value_diff": max_abs_val_diff,
        "max_abs_jacobian_diff": max_abs_jac_diff,
        "max_abs_adjoint_diff": max_abs_adj_diff,
        "parity_achieved": (failed_cases == 0),
        "cases": detailed_comparisons,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Sim1D Campaign Completed: {passed_cases}/{total_cases} passed ({summary['pass_rate_percent']}%)")
    print(f"Max abs grid diff: {max_abs_grid_diff:.2e}")
    print(f"Max abs value diff: {max_abs_val_diff:.2e}")
    print(f"Max abs jacobian diff: {max_abs_jac_diff:.2e}")
    print(f"Max abs adjoint diff: {max_abs_adj_diff:.2e}")
    print(f"Parity achieved: {summary['parity_achieved']}")

    if failed_cases > 0:
        for c in detailed_comparisons:
            if not c["passed"]:
                print(f"  FAILED: {c['case_id']}: {c['notes']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
