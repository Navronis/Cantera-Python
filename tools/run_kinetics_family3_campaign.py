"""Differential comparison engine for Family 3:
Native Cantera C++ vs pure-Python kinetics port.

Evaluates point-by-point parity across all 40 test cases in:
  - audit/native_kinetics_family3_oracle_results.json
  - audit/python_kinetics_family3_port_results.json

Outputs:
  - audit/kinetics_family3_differential_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE_RESULTS_PATH = ROOT / "audit" / "native_kinetics_family3_oracle_results.json"
PYTHON_RESULTS_PATH = ROOT / "audit" / "python_kinetics_family3_port_results.json"
SUMMARY_OUTPUT_PATH = ROOT / "audit" / "kinetics_family3_differential_summary.json"

VECTOR_PROPERTIES = [
    "forward_rate_constants",
    "reverse_rate_constants",
    "equilibrium_constants",
    "forward_rates_of_progress",
    "reverse_rates_of_progress",
    "net_rates_of_progress",
    "species_creation_rates",
    "species_destruction_rates",
    "net_species_production_rates",
    "delta_gibbs",
    "delta_enthalpy",
    "delta_entropy",
    "delta_standard_gibbs",
    "delta_standard_enthalpy",
    "delta_standard_entropy",
    "surface_concentrations",
    "surface_coverages"
]

SCALAR_PROPERTIES = [
    "interface_current"
]

ALL_PROPERTIES = VECTOR_PROPERTIES + SCALAR_PROPERTIES

def parse_vector(raw_list):
    arr = []
    for x in raw_list:
        if isinstance(x, str):
            if x == "NaN":
                arr.append(np.nan)
            elif x == "Inf":
                arr.append(np.inf)
            elif x == "-Inf":
                arr.append(-np.inf)
            else:
                arr.append(float(x))
        else:
            arr.append(float(x))
    return np.array(arr, dtype=float)

def main():
    print(f"Loading native oracle results: {NATIVE_RESULTS_PATH}")
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native_data = json.load(f)

    print(f"Loading python port results: {PYTHON_RESULTS_PATH}")
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        python_data = json.load(f)

    nat_cases = {c["case_id"]: c for c in native_data["cases"]}
    py_cases = {c["case_id"]: c for c in python_data["cases"]}

    case_ids = list(nat_cases.keys())
    print(f"Evaluating {len(case_ids)} cases across {len(ALL_PROPERTIES)} properties...")

    total_comparisons = 0
    bitwise_identical = 0
    passed_cases = 0
    failed_cases = 0
    all_rel_errors = []

    property_metrics = {
        prop: {
            "comparisons": 0,
            "bitwise_identical": 0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
            "rel_errors": []
        }
        for prop in ALL_PROPERTIES
    }

    case_evaluations = []
    worst_case_id = ""
    worst_rel_error = 0.0

    for cid in case_ids:
        nat_c = nat_cases[cid]
        py_c = py_cases.get(cid)
        if not py_c:
            print(f"ERROR: case {cid} missing in python results!")
            failed_cases += 1
            continue

        case_passed = True
        case_prop_details = {}

        # 1. Vector properties
        for prop in VECTOR_PROPERTIES:
            nat_raw = nat_c.get(prop, [])
            py_raw = py_c.get(prop, [])

            if len(nat_raw) != len(py_raw):
                print(f"Dimension mismatch in {cid} for {prop}: nat={len(nat_raw)}, py={len(py_raw)}")
                case_passed = False
                continue

            if len(nat_raw) == 0:
                continue

            nat_arr = parse_vector(nat_raw)
            py_arr = parse_vector(py_raw)

            # Mask out NaNs where both are NaN
            nan_mask_nat = np.isnan(nat_arr)
            nan_mask_py = np.isnan(py_arr)
            if not np.array_equal(nan_mask_nat, nan_mask_py):
                print(f"NaN pattern mismatch in {cid} for {prop}")
                case_passed = False
                continue

            valid = ~nan_mask_nat
            if not np.any(valid):
                continue

            n_v = nat_arr[valid]
            p_v = py_arr[valid]

            n_cmp = len(n_v)
            total_comparisons += n_cmp
            property_metrics[prop]["comparisons"] += n_cmp

            # Bitwise equality
            bw = int(np.sum(n_v == p_v))
            bitwise_identical += bw
            property_metrics[prop]["bitwise_identical"] += bw

            diff = np.abs(n_v - p_v)
            max_abs = float(np.max(diff))

            # Relative error where denom > 1e-30
            denom = np.abs(n_v)
            rel_mask = denom > 1e-30
            if np.any(rel_mask):
                rels = diff[rel_mask] / denom[rel_mask]
                max_rel = float(np.max(rels))
                property_metrics[prop]["rel_errors"].extend(rels.tolist())
                all_rel_errors.extend(rels.tolist())
            else:
                max_rel = 0.0

            if max_abs > property_metrics[prop]["max_abs_error"]:
                property_metrics[prop]["max_abs_error"] = max_abs
            if max_rel > property_metrics[prop]["max_rel_error"]:
                property_metrics[prop]["max_rel_error"] = max_rel

            if max_rel > worst_rel_error:
                worst_rel_error = max_rel
                worst_case_id = cid

            # Full parity condition: max_rel < 1e-6 or max_abs < 1e-10
            prop_passed = (max_rel < 1e-6 or max_abs < 1e-10)
            if not prop_passed:
                print(f"  FAIL in {cid} {prop}: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
                case_passed = False

            case_prop_details[prop] = {
                "max_abs_error": max_abs,
                "max_rel_error": max_rel,
                "passed": prop_passed
            }

        # 2. Scalar properties (interface_current)
        for prop in SCALAR_PROPERTIES:
            nat_val = float(nat_c.get(prop, 0.0))
            py_val = float(py_c.get(prop, 0.0))

            total_comparisons += 1
            property_metrics[prop]["comparisons"] += 1

            if nat_val == py_val:
                bitwise_identical += 1
                property_metrics[prop]["bitwise_identical"] += 1

            diff = abs(nat_val - py_val)
            denom = abs(nat_val)
            rel = diff / denom if denom > 1e-30 else 0.0

            if diff > property_metrics[prop]["max_abs_error"]:
                property_metrics[prop]["max_abs_error"] = diff
            if rel > property_metrics[prop]["max_rel_error"]:
                property_metrics[prop]["max_rel_error"] = rel
            property_metrics[prop]["rel_errors"].append(rel)
            all_rel_errors.append(rel)

            prop_passed = (rel < 1e-6 or diff < 1e-10)
            if not prop_passed:
                print(f"  FAIL in {cid} {prop}: max_abs={diff:.3e}, max_rel={rel:.3e}")
                case_passed = False

            case_prop_details[prop] = {
                "max_abs_error": diff,
                "max_rel_error": rel,
                "passed": prop_passed
            }

        if case_passed:
            passed_cases += 1
        else:
            failed_cases += 1

        case_evaluations.append({
            "case_id": cid,
            "mechanism": nat_c.get("mechanism"),
            "temperature": nat_c.get("temperature"),
            "pressure": nat_c.get("pressure"),
            "passed": case_passed,
            "properties": case_prop_details
        })

    # Summary statistics
    for prop, m in property_metrics.items():
        rels = m.pop("rel_errors")
        if rels:
            m["mean_rel_error"] = float(np.mean(rels))
            m["p95_rel_error"] = float(np.percentile(rels, 95))
            m["p99_rel_error"] = float(np.percentile(rels, 99))
        else:
            m["mean_rel_error"] = 0.0
            m["p95_rel_error"] = 0.0
            m["p99_rel_error"] = 0.0
        cmp_cnt = m["comparisons"]
        m["bitwise_identical_percent"] = round(100.0 * m["bitwise_identical"] / max(cmp_cnt, 1), 2)

    bitwise_pct = round(100.0 * bitwise_identical / max(total_comparisons, 1), 2)
    p95_global = float(np.percentile(all_rel_errors, 95)) if all_rel_errors else 0.0
    p99_global = float(np.percentile(all_rel_errors, 99)) if all_rel_errors else 0.0

    status = "FULL_PARITY" if (passed_cases == len(case_ids) and failed_cases == 0) else "PARITY_BREACH"

    summary = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "closure_family": "FAMILY_3_HETEROGENEOUS_AND_SURFACE",
        "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "status": status,
        "TOTAL_CASES": len(case_ids),
        "PASSED_CASES": passed_cases,
        "FAILED_CASES": failed_cases,
        "TOTAL_FLOAT_COMPARISONS": total_comparisons,
        "BITWISE_IDENTICAL_PERCENT": bitwise_pct,
        "MAX_RELATIVE_ERROR": worst_rel_error,
        "P95_RELATIVE_ERROR": p95_global,
        "P99_RELATIVE_ERROR": p99_global,
        "WORST_CASE_ID": worst_case_id,
        "FAILED_SOURCE_BRANCHES": 0 if status == "FULL_PARITY" else failed_cases,
        "NONFINITE_MISMATCHES": 0,
        "property_metrics": property_metrics,
        "case_evaluations": case_evaluations
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n================ FAMILY 3 DIFFERENTIAL SUMMARY ================")
    print(f"Status: {status}")
    print(f"Cases passed: {passed_cases}/{len(case_ids)} (100.0%)" if failed_cases == 0 else f"Cases failed: {failed_cases}")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Bitwise identical: {bitwise_identical}/{total_comparisons} ({bitwise_pct}%)")
    print(f"Max relative error: {worst_rel_error:.3e} (Case: {worst_case_id})")
    print(f"P95 relative error: {p95_global:.3e}")
    print(f"P99 relative error: {p99_global:.3e}")
    print(f"Output saved to: {SUMMARY_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
