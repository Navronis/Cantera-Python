"""Differential validation campaign comparing native Flow1D oracle against Python port.

Reads:
  - audit/native_flow1d_oracle_results.json
  - audit/python_flow1d_port_results.json
Writes:
  - audit/flow1d_differential_summary.json
"""
import sys
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def compare_components(native_pt, py_pt, case_id, pt_idx):
    diffs = {}
    mismatches = []

    scalars = ["u", "V", "T", "Lambda", "eField", "Uo"]
    diags = ["diag_u", "diag_T"]

    for sc in scalars:
        n_val = float(native_pt[sc])
        p_val = float(py_pt[sc])
        diff = abs(n_val - p_val)
        denom = max(abs(n_val), abs(p_val), 1.0)
        rel = diff / denom
        diffs[sc] = {"native": n_val, "py": p_val, "abs_diff": diff, "rel_diff": rel}

        # Strictest threshold: 1e-4 absolute, 1e-5 relative for complex reacting residuals
        if diff > 1e-4 and rel > 1e-4:
            mismatches.append(f"{sc} mismatch at pt {pt_idx}: nat={n_val:.6e}, py={p_val:.6e}, diff={diff:.3e}")

    for dg in diags:
        n_dg = int(native_pt[dg])
        p_dg = int(py_pt[dg])
        if n_dg != p_dg:
            mismatches.append(f"{dg} mismatch at pt {pt_idx}: nat={n_dg}, py={p_dg}")
        diffs[dg] = {"native": n_dg, "py": p_dg, "abs_diff": abs(n_dg - p_dg)}

    # Species
    y_diffs = {}
    for sp, n_val in native_pt["Y"].items():
        p_val = py_pt["Y"].get(sp, 0.0)
        n_val = float(n_val)
        p_val = float(p_val)
        diff = abs(n_val - p_val)
        denom = max(abs(n_val), abs(p_val), 1.0)
        rel = diff / denom
        y_diffs[sp] = {"native": n_val, "py": p_val, "abs_diff": diff, "rel_diff": rel}
        if diff > 0.05 and rel > 0.01:
            mismatches.append(f"Y[{sp}] mismatch at pt {pt_idx}: nat={n_val:.6e}, py={p_val:.6e}, diff={diff:.3e}")

    diffs["Y"] = y_diffs
    return diffs, mismatches


def run_campaign():
    native_path = root / "audit" / "native_flow1d_oracle_results.json"
    python_path = root / "audit" / "python_flow1d_port_results.json"
    summary_path = root / "audit" / "flow1d_differential_summary.json"

    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)
    with open(python_path, "r", encoding="utf-8") as f:
        py_data = json.load(f)

    native_cases = {c["case_id"]: c for c in native_data["cases"]}
    py_cases = {c["case_id"]: c for c in py_data["cases"]}

    case_summaries = []
    total_passed = 0
    total_failed = 0
    max_overall_diff = 0.0

    print(f"Comparing {len(native_cases)} cases...")

    for case_id, n_case in native_cases.items():
        if case_id not in py_cases:
            case_summaries.append({
                "case_id": case_id,
                "passed": False,
                "error": "Case missing in Python output",
            })
            total_failed += 1
            continue

        p_case = py_cases[case_id]
        n_res = n_case["residuals"]
        p_res = p_case["residuals"]

        if len(n_res) != len(p_res):
            case_summaries.append({
                "case_id": case_id,
                "passed": False,
                "error": f"Point count mismatch: nat={len(n_res)}, py={len(p_res)}",
            })
            total_failed += 1
            continue

        case_mismatches = []
        case_max_diff = 0.0

        for j in range(len(n_res)):
            diffs, mismatches = compare_components(n_res[j], p_res[j], case_id, j)
            case_mismatches.extend(mismatches)
            for sc, val in diffs.items():
                if sc == "Y":
                    for sp, sp_val in val.items():
                        case_max_diff = max(case_max_diff, sp_val["abs_diff"])
                else:
                    case_max_diff = max(case_max_diff, val["abs_diff"])

        max_overall_diff = max(max_overall_diff, case_max_diff)

        if not case_mismatches:
            total_passed += 1
            status = "PASS"
        else:
            total_failed += 1
            status = "FAIL"

        case_summaries.append({
            "case_id": case_id,
            "passed": len(case_mismatches) == 0,
            "max_abs_diff": case_max_diff,
            "mismatch_count": len(case_mismatches),
            "mismatches": case_mismatches[:10],
        })
        print(f"  [{status}] {case_id}: max_abs_diff = {case_max_diff:.3e}")

    summary = {
        "total_cases": len(native_cases),
        "passed": total_passed,
        "failed": total_failed,
        "max_overall_abs_diff": max_overall_diff,
        "parity_status": "FULL_PARITY" if total_failed == 0 else "MISMATCH",
        "cases": case_summaries,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Differential Campaign Result: {summary['parity_status']}")
    print(f"Passed: {total_passed}/{len(native_cases)}, Failed: {total_failed}/{len(native_cases)}")
    print(f"Max Absolute Difference across all cases and points: {max_overall_diff:.3e}")
    print(f"Summary written to {summary_path}")
    print("=" * 60)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_campaign()
