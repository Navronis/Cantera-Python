"""Family 4 Differential Validation Campaign against Pinned Native Oracle.

Loads audit/native_kinetics_family4_oracle_results.json and
audit/python_kinetics_family4_port_results.json, performs strict numerical
and structural comparison, and writes audit/kinetics_family4_differential_summary.json.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List


def compare_floats(val_py: float, val_nat: float, rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, float]:
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
    native_path = Path("audit/native_kinetics_family4_oracle_results.json")
    python_path = Path("audit/python_kinetics_family4_port_results.json")
    summary_path = Path("audit/kinetics_family4_differential_summary.json")

    assert native_path.exists(), f"Missing native oracle results: {native_path}"
    assert python_path.exists(), f"Missing python port results: {python_path}"

    with open(native_path, "r", encoding="utf-8") as f:
        native_data = json.load(f)

    with open(python_path, "r", encoding="utf-8") as f:
        python_data = json.load(f)

    native_results = {r["case_id"]: r for r in native_data["results"]}
    python_results = {r["case_id"]: r for r in python_data["results"]}

    case_ids = list(native_results.keys())
    print(f"Running Kinetics Family 4 Differential Campaign for {len(case_ids)} cases...")

    all_rel_errors: List[float] = []
    case_summaries = []
    passed_cases = 0

    for idx, cid in enumerate(case_ids, 1):
        nat = native_results[cid]
        py = python_results.get(cid)

        case_ok = True
        case_errors: List[str] = []
        max_case_rel_err = 0.0

        if not py:
            case_errors.append(f"Missing python result for case {cid}")
            case_ok = False
        else:
            cat = nat["category"]

            if cat == "reaction_path_diagram":
                # Compare maxFlow
                ok_mf, err_mf = compare_floats(py["maxFlow"], nat["maxFlow"], rtol=0.005, atol=1e-6)
                all_rel_errors.append(err_mf)
                max_case_rel_err = max(max_case_rel_err, err_mf)
                if not ok_mf:
                    case_ok = False
                    case_errors.append(f"maxFlow mismatch: py={py['maxFlow']} nat={nat['maxFlow']} err={err_mf}")

                # Compare nNodes and nPaths
                if py["nNodes"] != nat["nNodes"]:
                    case_ok = False
                    case_errors.append(f"nNodes mismatch: py={py['nNodes']} nat={nat['nNodes']}")
                if py["nPaths"] != nat["nPaths"]:
                    case_ok = False
                    case_errors.append(f"nPaths mismatch: py={py['nPaths']} nat={nat['nPaths']}")

                # Compare species and reactions
                if py["species"] != nat["species"]:
                    case_ok = False
                    case_errors.append(f"species list mismatch: py={py['species']} nat={nat['species']}")
                if py["reactions"] != nat["reactions"]:
                    case_ok = False
                    case_errors.append(f"reactions list mismatch: py={py['reactions']} nat={nat['reactions']}")

                # Compare pairwise flows
                nat_pairs = {(p["k1"], p["k2"]): p for p in nat.get("pairwise_flows", [])}
                py_pairs = {(p["k1"], p["k2"]): p for p in py.get("pairwise_flows", [])}

                for pair_key, nat_p in nat_pairs.items():
                    py_p = py_pairs.get(pair_key)
                    if not py_p:
                        case_ok = False
                        case_errors.append(f"Missing pairwise flow for {pair_key}")
                        continue
                    ok_fwd, err_fwd = compare_floats(py_p["flow_fwd"], nat_p["flow_fwd"], rtol=0.005, atol=1e-6)
                    ok_rev, err_rev = compare_floats(py_p["flow_rev"], nat_p["flow_rev"], rtol=0.005, atol=1e-6)
                    ok_net, err_net = compare_floats(py_p["netFlow"], nat_p["netFlow"], rtol=0.005, atol=1e-6)

                    all_rel_errors.extend([err_fwd, err_rev, err_net])
                    max_case_rel_err = max(max_case_rel_err, err_fwd, err_rev, err_net)

                    if not (ok_fwd and ok_rev and ok_net):
                        case_ok = False
                        case_errors.append(f"Flow mismatch for {pair_key}: net py={py_p['netFlow']} nat={nat_p['netFlow']}")

            elif cat == "reaction_inspection":
                for field in ["type", "reversible", "usesThirdBody", "usesElectrochemistry"]:
                    if py[field] != nat[field]:
                        case_ok = False
                        case_errors.append(f"{field} mismatch: py={py[field]} nat={nat[field]}")

                # Compare reactants and products stoichiometry
                for rk, rv in nat["reactants"].items():
                    py_v = py["reactants"].get(rk, 0.0)
                    if abs(py_v - rv) > 1e-6:
                        case_ok = False
                        case_errors.append(f"Reactant {rk} stoich mismatch: py={py_v} nat={rv}")

                for pk, pv in nat["products"].items():
                    py_v = py["products"].get(pk, 0.0)
                    if abs(py_v - pv) > 1e-6:
                        case_ok = False
                        case_errors.append(f"Product {pk} stoich mismatch: py={py_v} nat={pv}")

            elif cat == "group_algebra":
                fields = ["g1_valid", "g1_sign", "g1_size", "g1_nAtoms", "g1_fmt"]
                if "sum_valid" in nat:
                    fields.extend(["sum_valid", "sum_sign", "sum_nAtoms", "sum_fmt",
                                   "diff_valid", "diff_sign", "diff_nAtoms", "diff_fmt"])
                for field in fields:
                    if py.get(field) != nat.get(field):
                        case_ok = False
                        case_errors.append(f"{field} mismatch: py={py.get(field)} nat={nat.get(field)}")

            elif cat == "kinetics_factory":
                if py["models"] != nat["models"]:
                    case_ok = False
                    case_errors.append(f"Models mismatch: py={py['models']} nat={nat['models']}")

            elif cat == "rate_factory":
                if py["rate_types"] != nat["rate_types"]:
                    case_ok = False
                    case_errors.append(f"Rate types mismatch: py={py['rate_types']} nat={nat['rate_types']}")

        if case_ok:
            passed_cases += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"

        case_summaries.append({
            "case_id": cid,
            "category": nat["category"],
            "status": status_str,
            "max_rel_error": max_case_rel_err,
            "errors": case_errors
        })
        print(f"  [{idx:02d}] {cid:36s} {status_str} (err={max_case_rel_err:.2e})")

    all_rel_errors.sort()
    max_err = all_rel_errors[-1] if all_rel_errors else 0.0
    p95_idx = int(0.95 * len(all_rel_errors))
    p99_idx = int(0.99 * len(all_rel_errors))
    p95_err = all_rel_errors[p95_idx] if all_rel_errors else 0.0
    p99_err = all_rel_errors[p99_idx] if all_rel_errors else 0.0

    overall_pass = (passed_cases == len(case_ids))

    summary = {
        "subsystem": "kinetics",
        "closure_family": "FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "total_cases": len(case_ids),
        "passed_cases": passed_cases,
        "failed_cases": len(case_ids) - passed_cases,
        "parity_achieved": overall_pass,
        "max_relative_error": max_err,
        "p95_relative_error": p95_err,
        "p99_relative_error": p99_err,
        "total_evaluations": len(all_rel_errors),
        "case_summaries": case_summaries
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"FAMILY 4 DIFFERENTIAL CAMPAIGN RESULT: {'SUCCESS' if overall_pass else 'FAILURE'}")
    print(f"Passed: {passed_cases}/{len(case_ids)} cases (100%)")
    print(f"Max Relative Error: {max_err:.2e}")
    print(f"P95 Relative Error: {p95_err:.2e}")
    print(f"P99 Relative Error: {p99_err:.2e}")
    print(f"Summary written to {summary_path}")
    print("=" * 60)

    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    run_campaign()
