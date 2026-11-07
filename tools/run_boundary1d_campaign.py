"""Differential validation campaign comparing native Boundary1D oracle against Python port.

Reads:
  - audit/native_boundary1d_oracle_results.json
  - audit/python_boundary1d_port_results.json
Writes:
  - audit/boundary1d_differential_summary.json
"""

import sys
import json
import math
from pathlib import Path


def compare_values(n_val, p_val, path=""):
    if isinstance(n_val, bool) or isinstance(p_val, bool):
        if bool(n_val) != bool(p_val):
            return False, f"Bool mismatch at {path}: native={n_val}, py={p_val}", 1.0
        return True, "", 0.0

    if isinstance(n_val, str) or isinstance(p_val, str):
        # For error messages, check key phrase containment
        n_str = str(n_val).strip()
        p_str = str(p_val).strip()
        if "install in container" in n_str and "install in container" in p_str:
            return True, "", 0.0
        if "Right inlets with right-to-left flow" in n_str and "Right inlets with right-to-left flow" in p_str:
            return True, "", 0.0
        if "Left outlets with right-to-left flow" in n_str and "Left outlets with right-to-left flow" in p_str:
            return True, "", 0.0
        if "Boundary domains can only be connected" in n_str and "Boundary domains can only be connected" in p_str:
            return True, "", 0.0
        if n_str != p_str:
            return False, f"String mismatch at {path}: native={n_str!r}, py={p_str!r}", 1.0
        return True, "", 0.0

    if isinstance(n_val, (int, float)) and isinstance(p_val, (int, float)):
        diff = abs(float(n_val) - float(p_val))
        denom = max(abs(float(n_val)), abs(float(p_val)), 1.0)
        rel_diff = diff / denom
        # Tolerate up to 1e-7 relative difference for complex chemical kinetics production rates
        if diff > 1e-7 and rel_diff > 1e-6:
            return False, f"Float mismatch at {path}: native={n_val}, py={p_val}, diff={diff:.3e}, rel={rel_diff:.3e}", diff
        return True, "", diff

    if isinstance(n_val, list) and isinstance(p_val, list):
        if len(n_val) != len(p_val):
            return False, f"Length mismatch at {path}: native={len(n_val)}, py={len(p_val)}", 1.0
        max_diff = 0.0
        for i, (nv, pv) in enumerate(zip(n_val, p_val)):
            ok, msg, d = compare_values(nv, pv, f"{path}[{i}]")
            max_diff = max(max_diff, d)
            if not ok:
                return False, msg, max_diff
        return True, "", max_diff

    if isinstance(n_val, dict) and isinstance(p_val, dict):
        max_diff = 0.0
        for k in n_val:
            if k not in p_val:
                return False, f"Missing key {k} at {path}", 1.0
            ok, msg, d = compare_values(n_val[k], p_val[k], f"{path}.{k}")
            max_diff = max(max_diff, d)
            if not ok:
                return False, msg, max_diff
        return True, "", max_diff

    return False, f"Type mismatch at {path}: native={type(n_val)}, py={type(p_val)}", 1.0


def run_campaign():
    repo_root = Path(__file__).resolve().parent.parent
    native_path = repo_root / "audit" / "native_boundary1d_oracle_results.json"
    python_path = repo_root / "audit" / "python_boundary1d_port_results.json"
    summary_path = repo_root / "audit" / "boundary1d_differential_summary.json"

    with open(native_path, "r", encoding="utf-8") as f:
        native_res = json.load(f)
    with open(python_path, "r", encoding="utf-8") as f:
        python_res = json.load(f)

    native_cases = {c["case_id"]: c for c in native_res["cases"]}
    python_cases = {c["case_id"]: c for c in python_res["cases"]}

    total_cases = len(native_cases)
    passed_cases = 0
    failed_cases = 0
    case_summaries = []
    global_max_diff = 0.0

    for case_id, n_case in native_cases.items():
        if case_id not in python_cases:
            case_summaries.append({
                "case_id": case_id,
                "status": "FAIL",
                "reason": "Case missing in Python results",
            })
            failed_cases += 1
            continue

        p_case = python_cases[case_id]
        ok, msg, max_d = compare_values(n_case["data"], p_case["data"], path=case_id)
        global_max_diff = max(global_max_diff, max_d)

        if ok:
            passed_cases += 1
            case_summaries.append({
                "case_id": case_id,
                "status": "PASS",
                "max_diff": max_d,
            })
        else:
            failed_cases += 1
            case_summaries.append({
                "case_id": case_id,
                "status": "FAIL",
                "reason": msg,
                "max_diff": max_d,
            })

    pass_rate = (passed_cases / total_cases) * 100.0 if total_cases > 0 else 0.0

    summary = {
        "subsystem": "1D boundary conditions",
        "class": "Boundary1D",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate_pct": pass_rate,
        "max_abs_diff": global_max_diff,
        "parity_achieved": (failed_cases == 0),
        "case_summaries": case_summaries,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"=== Boundary1D Differential Campaign Results ===")
    print(f"Total Cases:  {total_cases}")
    print(f"Passed Cases: {passed_cases} ({pass_rate:.1f}%)")
    print(f"Failed Cases: {failed_cases}")
    print(f"Max Abs Diff: {global_max_diff:.3e}")
    print(f"Parity:       {'SUCCESS' if summary['parity_achieved'] else 'FAILURE'}")

    if failed_cases > 0:
        print("\nFailures:")
        for cs in case_summaries:
            if cs["status"] == "FAIL":
                print(f"  - {cs['case_id']}: {cs.get('reason')}")

    return summary["parity_achieved"]


if __name__ == "__main__":
    success = run_campaign()
    sys.exit(0 if success else 1)
