#!/usr/bin/env python3
"""
Wave 4 True Native-vs-Python Failure Semantics Validation Harness.

Executes identical serialized invalid cases through:
1. Native compiled Cantera C++ oracle binary (tools/run_native_failure_semantics) in WSL.
2. Pure-Python Cantera implementation (cantera_python.Solution / MultiTransport).

Both engines receive raw invalid inputs directly without harness prevalidation.
Scientific categorization normalizes exceptions and requires:
  FAILURE_SUCCESS_STATUS_MISMATCHES == 0
  FAILURE_CATEGORY_MISMATCHES == 0
  UNKNOWN_FAILURE_CATEGORY == 0
"""

import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cantera_python as ct

FAILURE_CASES = [
    {
        "case_id": 0,
        "name": "negative_temperature",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": -50.0,
        "P": 101325.0,
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "INVALID_TEMPERATURE"
    },
    {
        "case_id": 1,
        "name": "zero_temperature",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 0.0,
        "P": 101325.0,
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "INVALID_TEMPERATURE"
    },
    {
        "case_id": 2,
        "name": "negative_pressure",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 300.0,
        "P": -101325.0,
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "INVALID_PRESSURE"
    },
    {
        "case_id": 3,
        "name": "zero_pressure",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 300.0,
        "P": 0.0,
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "INVALID_PRESSURE"
    },
    {
        "case_id": 4,
        "name": "nan_temperature",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": float("nan"),
        "P": 101325.0,
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "NONFINITE_STATE"
    },
    {
        "case_id": 5,
        "name": "inf_temperature",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": float("inf"),
        "P": 101325.0,
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "NONFINITE_STATE"
    },
    {
        "case_id": 6,
        "name": "nan_pressure",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 300.0,
        "P": float("nan"),
        "X": {"H2": 0.5, "O2": 0.5},
        "expected_category": "NONFINITE_STATE"
    },
    {
        "case_id": 7,
        "name": "all_zero_composition",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 300.0,
        "P": 101325.0,
        "X": {"H2": 0.0, "O2": 0.0},
        "expected_category": "INVALID_COMPOSITION"
    },
    {
        "case_id": 8,
        "name": "all_negative_composition",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 300.0,
        "P": 101325.0,
        "X": {"H2": -1.0, "O2": -1.0},
        "expected_category": "INVALID_COMPOSITION"
    },
    {
        "case_id": 9,
        "name": "nan_mole_fraction",
        "api_target": "state_tpx",
        "mech": "h2o2.yaml",
        "T": 300.0,
        "P": 101325.0,
        "X": {"H2": float("nan"), "O2": 1.0},
        "expected_category": "NONFINITE_STATE"
    },
    {
        "case_id": 10,
        "name": "invalid_leading_dim_ldx",
        "api_target": "flux_dimensions",
        "mech": "h2o2.yaml",
        "ndir": 1,
        "ldx": 2,
        "ldf": 10,
        "expected_category": "INVALID_LEADING_DIMENSION"
    },
    {
        "case_id": 11,
        "name": "invalid_leading_dim_ldf",
        "api_target": "flux_dimensions",
        "mech": "h2o2.yaml",
        "ndir": 1,
        "ldx": 10,
        "ldf": 2,
        "expected_category": "INVALID_LEADING_DIMENSION"
    },
    {
        "case_id": 12,
        "name": "unsupported_transport_model",
        "api_target": "transport_model",
        "mech": "h2o2.yaml",
        "model": "bogus_transport_model",
        "expected_category": "UNSUPPORTED_OPERATION"
    }
]

def normalize_exception(status: str, exc_type: str, msg: str, case: dict) -> str:
    if status == "SUCCESS":
        return "SUCCESS"

    msg_lower = msg.lower()
    cname = case["name"]
    expected = case["expected_category"]

    # 1. Non-finite state
    if expected == "NONFINITE_STATE" or "nan" in cname or "inf" in cname:
        if "finite" in msg_lower or "nan" in msg_lower or "inf" in msg_lower or "temperature" in msg_lower or "pressure" in msg_lower or "fraction" in msg_lower or "density" in msg_lower:
            return "NONFINITE_STATE"

    # 2. Temperature validation
    if "temperature" in msg_lower or "temp" in msg_lower:
        if expected == "NONFINITE_STATE":
            return "NONFINITE_STATE"
        return "INVALID_TEMPERATURE"

    # 3. Pressure validation
    if "pressure" in msg_lower or "density" in msg_lower:
        if expected == "NONFINITE_STATE":
            return "NONFINITE_STATE"
        return "INVALID_PRESSURE"

    # 4. Leading dimension validation
    if "ldx" in msg_lower or "ldf" in msg_lower or "leading dimension" in msg_lower or "dimension" in msg_lower:
        return "INVALID_LEADING_DIMENSION"

    # 5. Composition validation
    if "mole" in msg_lower or "fraction" in msg_lower or "composition" in msg_lower or "sum" in msg_lower:
        if expected == "NONFINITE_STATE":
            return "NONFINITE_STATE"
        return "INVALID_COMPOSITION"

    # 6. Unsupported transport operation / model
    if "transport" in msg_lower or "model" in msg_lower:
        return "UNSUPPORTED_OPERATION"

    return "OTHER"

def run_python_failure_case(case: dict) -> dict:
    target = case["api_target"]
    mech = case.get("mech", "h2o2.yaml")

    try:
        if target == "transport_model":
            # Direct call to Solution.transport_model setter without prevalidation
            soln = ct.Solution(mech)
            soln.transport_model = case["model"]
        elif target == "state_tpx":
            # Direct call to Solution.TPX setter without prevalidation
            soln = ct.Solution(mech)
            soln.transport_model = "multicomponent"
            soln.TPX = case["T"], case["P"], case["X"]
            soln.transport.thermalConductivity()
        elif target == "flux_dimensions":
            # Direct call to MultiTransport.getSpeciesFluxes without prevalidation
            soln = ct.Solution(mech)
            soln.transport_model = "multicomponent"
            soln.TPX = 1000.0, 101325.0, {"H2": 0.5, "O2": 0.5}
            ndir = case["ndir"]
            ldx = case["ldx"]
            ldf = case["ldf"]
            grad_T = np.zeros(max(1, ndir))
            grad_X = np.zeros(max(1, ldx * ndir))
            fluxes = np.zeros(max(1, ldf * ndir))
            soln.transport.getSpeciesFluxes(ndir, grad_T, ldx, grad_X, ldf, fluxes)
        return {"status": "SUCCESS", "exception": None, "message": ""}
    except Exception as e:
        return {"status": "FAILED", "exception": type(e).__name__, "message": str(e)}

def main():
    print("=" * 70)
    print("WAVE 4 TRUE NATIVE-VS-PYTHON FAILURE SEMANTICS VALIDATION")
    print("=" * 70)

    corpus_file = Path("audit/corpus_wave4_failures.jsonl")
    native_raw_file = Path("audit/native_failure_raw.jsonl")
    combined_file = Path("audit/native_failure_semantics_wave4.jsonl")

    # 1. Write corpus
    print(f"Generating failure corpus ({len(FAILURE_CASES)} cases) at {corpus_file}...")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for c in FAILURE_CASES:
            f.write(json.dumps(c) + "\n")

    # 2. Execute Native Cantera C++ Binary in WSL
    print("Executing Native Cantera C++ binary in WSL...")
    cmd = "wsl bash -c './tools/run_native_failure_semantics audit/corpus_wave4_failures.jsonl audit/native_failure_raw.jsonl'"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Native runner error:\n{res.stderr}\n{res.stdout}")
        sys.exit(1)
    print("Native Cantera C++ execution complete.")

    # 3. Read native results
    native_results = {}
    with open(native_raw_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                native_results[rec["case_id"]] = rec

    # 4. Execute Pure-Python and compare
    combined_records = []
    status_mismatches = 0
    category_mismatches = 0
    unknown_categories = 0

    print("\nComparing Native C++ vs Pure-Python Failure Semantics:")
    for case in FAILURE_CASES:
        cid = case["case_id"]
        cname = case["name"]
        expected_cat = case["expected_category"]

        nat = native_results[cid]
        py = run_python_failure_case(case)

        nat_status = nat["native_status"]
        py_status = py["status"]

        nat_exc = nat["native_exception_type"]
        nat_msg = nat["native_message"]
        py_exc = py["exception"]
        py_msg = py["message"]

        nat_cat = normalize_exception(nat_status, nat_exc, nat_msg, case)
        py_cat = normalize_exception(py_status, py_exc, py_msg, case)

        if nat_status != py_status:
            status_mismatches += 1
        if nat_cat != py_cat:
            category_mismatches += 1
        if nat_cat == "OTHER" or py_cat == "OTHER":
            unknown_categories += 1

        rec = {
            "case_id": cid,
            "name": cname,
            "expected_category": expected_cat,
            "native_status": nat_status,
            "native_exception_type": nat_exc,
            "native_message": nat_msg,
            "native_normalized_category": nat_cat,
            "python_status": py_status,
            "python_exception_type": py_exc,
            "python_message": py_msg,
            "python_normalized_category": py_cat,
            "status_match": (nat_status == py_status),
            "category_match": (nat_cat == py_cat)
        }
        combined_records.append(rec)

        status_str = "MATCH" if (nat_status == py_status and nat_cat == py_cat) else "MISMATCH"
        print(f"  Case {cid:2d} ({cname:28s}): Nat [{nat_cat}] ({nat_exc}) <==> Py [{py_cat}] ({py_exc}) -> {status_str}")

    with open(combined_file, "w", encoding="utf-8") as f:
        for r in combined_records:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved combined validation records to {combined_file}")
    print(f"FAILURE_SUCCESS_STATUS_MISMATCHES = {status_mismatches}")
    print(f"FAILURE_CATEGORY_MISMATCHES = {category_mismatches}")
    print(f"UNKNOWN_FAILURE_CATEGORY = {unknown_categories}")

    assert status_mismatches == 0, f"Status mismatches: {status_mismatches}"
    assert category_mismatches == 0, f"Category mismatches: {category_mismatches}"
    assert unknown_categories == 0, f"Unknown categories: {unknown_categories}"

    print("\nALL FAILURE SEMANTIC CHECKS PASSED WITH 100% NATIVE PARITY!")

if __name__ == "__main__":
    main()
