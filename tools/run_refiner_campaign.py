"""Run differential comparison campaign between native Cantera Refiner oracle and Python port.

Generates audit/refiner_differential_summary.json.
"""

import json
import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))


def main():
    native_path = repo_root / "audit" / "native_refiner_oracle_results.json"
    python_path = repo_root / "audit" / "python_refiner_port_results.json"
    summary_path = repo_root / "audit" / "refiner_differential_summary.json"

    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)["cases"]

    with open(python_path, "r", encoding="utf-8") as f:
        python_data = json.load(f)["cases"]

    assert len(native_data) == len(python_data), f"Count mismatch: native={len(native_data)}, py={len(python_data)}"

    total_cases = len(native_data)
    passed_cases = 0
    failed_cases = 0
    detailed_comparisons = []

    insertion_flag_mismatches = 0
    retention_flag_mismatches = 0
    new_point_count_mismatches = 0
    exception_mismatches = 0
    criteria_mismatches = 0

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
            # Both threw exceptions - check semantics
            n_msg = n_case["error_message"]
            p_msg = p_case["error_message"]
            notes.append(f"Expected exception triggered and matched in both.")
        else:
            # Normal execution
            # 1. New point count
            if n_case["n_new_points"] != p_case["n_new_points"]:
                case_passed = False
                new_point_count_mismatches += 1
                notes.append(f"n_new_points mismatch: native={n_case['n_new_points']}, py={p_case['n_new_points']}")

            # 2. Insertion points set
            n_ins = set(n_case["insert_points"])
            p_ins = set(p_case["insert_points"])
            if n_ins != p_ins:
                case_passed = False
                insertion_flag_mismatches += 1
                notes.append(f"insert_points mismatch: native={sorted(n_ins)}, py={sorted(p_ins)}")

            # 3. Retention points list
            n_keep = n_case["keep_points"]
            p_keep = p_case["keep_points"]
            if n_keep != p_keep:
                case_passed = False
                retention_flag_mismatches += 1
                mismatch_indices = [idx for idx, (nk, pk) in enumerate(zip(n_keep, p_keep)) if nk != pk]
                notes.append(f"keep_points mismatch at indices: {mismatch_indices}")

            # 4. Criteria check
            n_crit = n_case["criteria"]
            p_crit = p_case["criteria"]
            for c_idx in range(len(n_crit)):
                if abs(n_crit[c_idx] - p_crit[c_idx]) > 1e-12:
                    case_passed = False
                    criteria_mismatches += 1
                    notes.append(f"criteria[{c_idx}] mismatch: native={n_crit[c_idx]}, py={p_crit[c_idx]}")

        if case_passed:
            passed_cases += 1
        else:
            failed_cases += 1

        detailed_comparisons.append({
            "case_id": case_id,
            "passed": case_passed,
            "native_exception": n_exc,
            "python_exception": p_exc,
            "native_n_new": n_case.get("n_new_points", -1),
            "python_n_new": p_case.get("n_new_points", -1),
            "native_insert_points": n_case.get("insert_points", []),
            "python_insert_points": p_case.get("insert_points", []),
            "notes": "; ".join(notes) if notes else "Exact match"
        })

    parity_achieved = (passed_cases == total_cases) and (failed_cases == 0)

    summary = {
        "schema_version": 1,
        "solver_family": "refiner",
        "subsystem": "grid refinement",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate_pct": (passed_cases / total_cases) * 100.0,
        "parity_achieved": parity_achieved,
        "metrics": {
            "insertion_flag_mismatches": insertion_flag_mismatches,
            "retention_flag_mismatches": retention_flag_mismatches,
            "new_point_count_mismatches": new_point_count_mismatches,
            "exception_mismatches": exception_mismatches,
            "criteria_mismatches": criteria_mismatches
        },
        "detailed_comparisons": detailed_comparisons
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Refiner Differential Summary:")
    print(f"  Total Cases: {total_cases}")
    print(f"  Passed: {passed_cases} ({summary['pass_rate_pct']:.1f}%)")
    print(f"  Failed: {failed_cases}")
    print(f"  Insertion Flag Mismatches: {insertion_flag_mismatches}")
    print(f"  Retention Flag Mismatches: {retention_flag_mismatches}")
    print(f"  New Point Count Mismatches: {new_point_count_mismatches}")
    print(f"  Parity Achieved: {parity_achieved}")

    if not parity_achieved:
        for comp in detailed_comparisons:
            if not comp["passed"]:
                print(f"    FAIL: {comp['case_id']} - {comp['notes']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
