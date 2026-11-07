"""Differential Campaign Runner: Compare Python MultiNewton Port against Native Cantera Oracle.

Generates audit/multi_newton_differential_summary.json.
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import numpy as np


def compute_rel_error(val_py, val_nat, atol=1e-12):
    denom = max(abs(val_nat), atol)
    return abs(val_py - val_nat) / denom


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"

    oracle_file = audit_dir / "native_multi_newton_oracle_results.json"
    port_file = audit_dir / "python_multi_newton_port_results.json"
    summary_file = audit_dir / "multi_newton_differential_summary.json"

    with open(oracle_file, "r", encoding="utf-8") as f:
        native_data = {r["case_id"]: r for r in json.load(f)["results"]}

    with open(port_file, "r", encoding="utf-8") as f:
        python_data = {r["case_id"]: r for r in json.load(f)["results"]}

    cases_summary = []
    total_cases = len(native_data)
    passed_cases = 0

    max_x1_rel_err = 0.0
    max_x1_abs_err = 0.0
    max_step_rel_err = 0.0
    max_fbound_rel_err = 0.0
    max_s1_rel_err = 0.0

    TOL_REL = 1e-9
    TOL_ABS = 1e-9

    for case_id, nat in native_data.items():
        py = python_data.get(case_id)
        if not py:
            cases_summary.append({
                "case_id": case_id,
                "passed": False,
                "reason": "Missing in Python results"
            })
            continue

        mode = nat.get("mode", "solve")
        case_passed = True
        discrepancies = []

        if mode == "solve":
            nat_status = nat["status"]
            py_status = py["status"]
            if nat_status != py_status:
                case_passed = False
                discrepancies.append(f"Status mismatch: nat={nat_status}, py={py_status}")

            nat_iters = nat["last_iterations"]
            py_iters = py["last_iterations"]
            if nat_iters != py_iters:
                case_passed = False
                discrepancies.append(f"Iterations mismatch: nat={nat_iters}, py={py_iters}")

            nat_x1 = np.array(nat["x1"], dtype=float)
            py_x1 = np.array(py["x1"], dtype=float)
            if len(nat_x1) != len(py_x1):
                case_passed = False
                discrepancies.append(f"x1 size mismatch: nat={len(nat_x1)}, py={len(py_x1)}")
            else:
                abs_err = np.max(np.abs(py_x1 - nat_x1))
                rel_err = 0.0
                for a, b in zip(py_x1, nat_x1):
                    rel_err = max(rel_err, compute_rel_error(a, b))

                max_x1_abs_err = max(max_x1_abs_err, float(abs_err))
                max_x1_rel_err = max(max_x1_rel_err, float(rel_err))

                if abs_err > TOL_ABS and rel_err > TOL_REL:
                    case_passed = False
                    discrepancies.append(f"x1 numerical tolerance exceeded: max_abs={abs_err:.3e}, max_rel={rel_err:.3e}")

        elif mode == "step":
            nat_step = np.array(nat["step"], dtype=float)
            py_step = np.array(py["step"], dtype=float)
            abs_err = np.max(np.abs(py_step - nat_step))
            rel_err = 0.0
            for a, b in zip(py_step, nat_step):
                rel_err = max(rel_err, compute_rel_error(a, b))

            max_step_rel_err = max(max_step_rel_err, float(rel_err))
            if abs_err > TOL_ABS and rel_err > TOL_REL:
                case_passed = False
                discrepancies.append(f"step numerical tolerance exceeded: max_abs={abs_err:.3e}, max_rel={rel_err:.3e}")

        elif mode == "boundStep":
            nat_fb = nat["fbound"]
            py_fb = py["fbound"]
            rel_err = compute_rel_error(py_fb, nat_fb)
            max_fbound_rel_err = max(max_fbound_rel_err, rel_err)
            if rel_err > TOL_REL and abs(py_fb - nat_fb) > TOL_ABS:
                case_passed = False
                discrepancies.append(f"fbound tolerance exceeded: nat={nat_fb:.6e}, py={py_fb:.6e}, rel={rel_err:.3e}")

        elif mode == "dampStep":
            nat_status = nat["status"]
            py_status = py["status"]
            if nat_status != py_status:
                case_passed = False
                discrepancies.append(f"Status mismatch: nat={nat_status}, py={py_status}")

            nat_s1 = nat["s1"]
            py_s1 = py["s1"]
            rel_err = compute_rel_error(py_s1, nat_s1)
            max_s1_rel_err = max(max_s1_rel_err, rel_err)
            if rel_err > TOL_REL and abs(py_s1 - nat_s1) > TOL_ABS:
                case_passed = False
                discrepancies.append(f"s1 tolerance exceeded: nat={nat_s1:.6e}, py={py_s1:.6e}, rel={rel_err:.3e}")

            nat_x1 = np.array(nat["x1"], dtype=float)
            py_x1 = np.array(py["x1"], dtype=float)
            abs_err = np.max(np.abs(py_x1 - nat_x1))
            rel_err_x = 0.0
            for a, b in zip(py_x1, nat_x1):
                rel_err_x = max(rel_err_x, compute_rel_error(a, b))
            if abs_err > TOL_ABS and rel_err_x > TOL_REL:
                case_passed = False
                discrepancies.append(f"dampStep x1 tolerance exceeded: max_abs={abs_err:.3e}")

        elif mode == "multijac_ops":
            if nat["type"] != py["type"]:
                case_passed = False
                discrepancies.append(f"MultiJac type mismatch: nat={nat['type']}, py={py['type']}")
            if abs(nat["value_1_1"] - py["value_1_1"]) > 1e-12:
                case_passed = False
                discrepancies.append("MultiJac value(1,1) mismatch")
            if abs(nat["value_1_2"] - py["value_1_2"]) > 1e-12:
                case_passed = False
                discrepancies.append("MultiJac value(1,2) mismatch")
            if nat["info"] != py["info"]:
                case_passed = False
                discrepancies.append("MultiJac info mismatch")

            nat_sol = np.array(nat["sol"], dtype=float)
            py_sol = np.array(py["sol"], dtype=float)
            abs_err = np.max(np.abs(py_sol - nat_sol))
            rel_err = max(compute_rel_error(a, b) for a, b in zip(py_sol, nat_sol))
            if abs_err > TOL_ABS and rel_err > TOL_REL:
                case_passed = False
                discrepancies.append(f"MultiJac solve tolerance exceeded: max_abs={abs_err:.3e}")

        if case_passed:
            passed_cases += 1

        cases_summary.append({
            "case_id": case_id,
            "mode": mode,
            "passed": case_passed,
            "discrepancies": discrepancies
        })

    pass_rate = (passed_cases / total_cases) * 100.0 if total_cases > 0 else 0.0

    summary = {
        "schema_version": 1,
        "solver_family": "multi_newton",
        "subsystem": "1D Newton",
        "campaign_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_pct": pass_rate,
        "tolerances": {
            "rtol": TOL_REL,
            "atol": TOL_ABS
        },
        "max_errors": {
            "max_x1_relative_error": max_x1_rel_err,
            "max_x1_absolute_error": max_x1_abs_err,
            "max_step_relative_error": max_step_rel_err,
            "max_fbound_relative_error": max_fbound_rel_err,
            "max_s1_relative_error": max_s1_rel_err
        },
        "numerical_parity": (passed_cases == total_cases and total_cases > 0),
        "cases": cases_summary
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print("MULTINEWTON DIFFERENTIAL CAMPAIGN REPORT")
    print(f"Total Cases : {total_cases}")
    print(f"Passed Cases: {passed_cases} ({pass_rate:.1f}%)")
    print(f"Max x1 Rel Error    : {max_x1_rel_err:.3e}")
    print(f"Max x1 Abs Error    : {max_x1_abs_err:.3e}")
    print(f"Max step Rel Error  : {max_step_rel_err:.3e}")
    print(f"Max fbound Rel Error: {max_fbound_rel_err:.3e}")
    print(f"Max s1 Rel Error    : {max_s1_rel_err:.3e}")
    print(f"Numerical Parity    : {'PASS' if summary['numerical_parity'] else 'FAIL'}")
    print("=" * 70)

    if not summary["numerical_parity"]:
        for c in cases_summary:
            if not c["passed"]:
                print(f"FAILED {c['case_id']}: {c['discrepancies']}")


if __name__ == "__main__":
    main()
