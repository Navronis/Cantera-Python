"""Run Domain1D differential validation campaign comparing native oracle vs Python port.

Generates:
  - audit/domain1d_differential_summary.json
"""

import json
from pathlib import Path
import numpy as np


def run_campaign():
    repo_root = Path(__file__).resolve().parent.parent
    native_path = repo_root / "audit" / "native_domain1d_oracle_results.json"
    python_path = repo_root / "audit" / "python_domain1d_port_results.json"
    summary_path = repo_root / "audit" / "domain1d_differential_summary.json"

    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)
    with open(python_path, "r", encoding="utf-8") as f:
        python_data = json.load(f)

    native_cases = {c["case_id"]: c for c in native_data["cases"]}
    python_cases = {c["case_id"]: c for c in python_data["cases"]}

    case_ids = sorted(list(native_cases.keys()))
    case_results = []
    all_passed = True

    max_rel_diff = 0.0
    max_abs_diff = 0.0

    print("=" * 80)
    print("DOMAIN1D DIFFERENTIAL VALIDATION CAMPAIGN")
    print("=" * 80)

    for cid in case_ids:
        nat = native_cases[cid]
        py = python_cases[cid]

        case_passed = True
        diff_info = {}

        # 1. Success matching
        if nat["success"] != py["success"]:
            case_passed = False
            diff_info["success_mismatch"] = f"native={nat['success']} py={py['success']}"

        # 2. Scalar int/bool/string fields
        for field in [
            "points", "size", "bandwidth", "domain_type",
            "steady_flag", "transient_flag", "jacobian_mode",
            "caught_invalid_jacobian_mode",
            "d0_loc", "d0_firstPoint", "d0_lastPoint",
            "d1_loc", "d1_firstPoint", "d1_lastPoint",
            "d2_loc", "d2_firstPoint", "d2_lastPoint",
            "d1_global_idx_2_3",
            "has_d0_left", "has_d0_right", "has_d2_right",
        ]:
            if field in nat and field in py:
                if nat[field] != py[field]:
                    case_passed = False
                    diff_info[f"{field}_mismatch"] = f"native={nat[field]} py={py[field]}"

        # 3. String / bool list fields
        for field in ["component_names", "queried_indices", "has_components", "topology5_locs", "topology5_firstPoints", "topology5_lastPoints"]:
            if field in nat and field in py:
                if list(nat[field]) != list(py[field]):
                    case_passed = False
                    diff_info[f"{field}_mismatch"] = f"native={nat[field]} py={py[field]}"

        # 4. Numerical float array fields
        for field in [
            "zmin", "zmax",
            "steady_rtol", "steady_atol", "transient_rtol", "transient_atol",
            "lower_bounds", "upper_bounds",
            "current_rtol", "current_atol",
            "prev_solution", "grid_points", "clamped_state",
            "refine_criteria", "queried_values", "initial_solution", "shared_state",
        ]:
            if field in nat and field in py:
                n_val = np.asarray(nat[field], dtype=float)
                p_val = np.asarray(py[field], dtype=float)
                if n_val.shape != p_val.shape:
                    case_passed = False
                    diff_info[f"{field}_shape_mismatch"] = f"native={n_val.shape} py={p_val.shape}"
                else:
                    abs_diff = float(np.max(np.abs(n_val - p_val))) if n_val.size > 0 else 0.0
                    max_abs_diff = max(max_abs_diff, abs_diff)
                    if abs_diff > 1e-12:
                        case_passed = False
                        diff_info[f"{field}_abs_diff"] = abs_diff

        status_str = "PASS" if case_passed else "FAIL"
        if not case_passed:
            all_passed = False
        print(f"[{status_str}] {cid:<45} {diff_info if diff_info else ''}")

        case_results.append({
            "case_id": cid,
            "passed": case_passed,
            "diffs": diff_info,
        })

    print("-" * 80)
    passed_count = sum(1 for c in case_results if c["passed"])
    total_count = len(case_results)
    print(f"RESULTS: {passed_count}/{total_count} PASSED ({passed_count/total_count*100:.1f}%)")
    print(f"Max absolute diff: {max_abs_diff:.2e}")
    print("=" * 80)

    summary = {
        "subsystem": "1D domain base abstraction",
        "native_class": "Domain1D",
        "total_cases": total_count,
        "passed_cases": passed_count,
        "failed_cases": total_count - passed_count,
        "parity_achieved": all_passed,
        "max_abs_diff": max_abs_diff,
        "cases": case_results,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to {summary_path.relative_to(repo_root)}")
    return all_passed


if __name__ == "__main__":
    success = run_campaign()
    if not success:
        exit(1)
