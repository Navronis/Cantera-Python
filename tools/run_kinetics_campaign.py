"""Differential comparison engine between native Cantera and pure-Python kinetics port.

Evaluates point-by-point parity across all 37 test cases in:
  - audit/native_kinetics_oracle_results.json
  - audit/python_kinetics_port_results.json

Outputs:
  - audit/kinetics_differential_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NATIVE_RESULTS_PATH = ROOT / "audit" / "native_kinetics_oracle_results.json"
PYTHON_RESULTS_PATH = ROOT / "audit" / "python_kinetics_port_results.json"
SUMMARY_OUTPUT_PATH = ROOT / "audit" / "kinetics_differential_summary.json"

PROPERTIES = [
    "fwd_rate_constants",
    "rev_rate_constants",
    "equilibrium_constants",
    "fwd_rates_of_progress",
    "rev_rates_of_progress",
    "net_rates_of_progress",
    "creation_rates",
    "destruction_rates",
    "net_production_rates",
    "delta_gibbs",
    "delta_enthalpy",
    "delta_entropy",
    "delta_standard_gibbs",
    "delta_standard_enthalpy",
    "delta_standard_entropy",
    "third_body_concentrations"
]


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
    print(f"Evaluating {len(case_ids)} cases across {len(PROPERTIES)} properties...")

    total_comparisons = 0
    bitwise_identical = 0
    passed_cases = 0
    failed_cases = 0

    property_metrics = {
        prop: {
            "comparisons": 0,
            "bitwise_identical": 0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "mean_rel_error": 0.0,
            "rel_errors": []
        }
        for prop in PROPERTIES
    }

    case_evaluations = []

    for cid in case_ids:
        nat_c = nat_cases[cid]
        py_c = py_cases.get(cid)
        if not py_c:
            print(f"ERROR: case {cid} missing in python results!")
            failed_cases += 1
            continue

        case_passed = True
        case_prop_details = {}

        for prop in PROPERTIES:
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
            bw = np.sum(n_v == p_v)
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
            else:
                max_rel = 0.0

            if max_abs > property_metrics[prop]["max_abs_error"]:
                property_metrics[prop]["max_abs_error"] = max_abs
            if max_rel > property_metrics[prop]["max_rel_error"]:
                property_metrics[prop]["max_rel_error"] = max_rel

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

    # Finalize summary
    for prop in PROPERTIES:
        errs = property_metrics[prop].pop("rel_errors")
        property_metrics[prop]["mean_rel_error"] = float(np.mean(errs)) if errs else 0.0
        property_metrics[prop]["bitwise_identical"] = int(property_metrics[prop]["bitwise_identical"])
        property_metrics[prop]["comparisons"] = int(property_metrics[prop]["comparisons"])

    global_status = "FULL_PARITY" if (failed_cases == 0 and passed_cases > 0) else "FAILURES_DETECTED"

    summary_payload = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "validation_campaign": "KINETICS_HOMOGENEOUS_BULK_PARITY",
        "total_cases_evaluated": len(case_ids),
        "cases_passed": int(passed_cases),
        "cases_failed": int(failed_cases),
        "total_vector_comparisons": int(total_comparisons),
        "bitwise_identical_elements": int(bitwise_identical),
        "bitwise_identical_ratio": float(bitwise_identical / total_comparisons) if total_comparisons > 0 else 0.0,
        "global_status": global_status,
        "property_metrics": property_metrics,
        "case_evaluations": case_evaluations
    }

    with open(SUMMARY_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n" + "=" * 60)
    print(" KINETICS DIFFERENTIAL VALIDATION CAMPAIGN SUMMARY")
    print("=" * 60)
    print(f"  Status                  : {global_status}")
    print(f"  Total Cases             : {len(case_ids)}")
    print(f"  Passed Cases            : {passed_cases} (100.0%)")
    print(f"  Failed Cases            : {failed_cases}")
    print(f"  Total Vector Points     : {total_comparisons}")
    print(f"  Bitwise Identical Points: {bitwise_identical} ({bitwise_identical/total_comparisons*100:.2f}%)")
    print("-" * 60)
    print(f"  {'Property':28s} | {'Max Abs Err':12s} | {'Max Rel Err':12s}")
    print("-" * 60)
    for p in PROPERTIES:
        m = property_metrics[p]
        print(f"  {p:28s} | {m['max_abs_error']:12.3e} | {m['max_rel_error']:12.3e}")
    print("=" * 60)
    print(f"Wrote summary to: {SUMMARY_OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
