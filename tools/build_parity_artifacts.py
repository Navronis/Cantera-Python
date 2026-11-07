"""Regenerate conservative native-parity artifacts from reproduced evidence."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "provenance" / "pinned_native_differential_results.jsonl"
PARITY = ROOT / "audit" / "native_vs_python_parity.jsonl"
SUMMARY = ROOT / "audit" / "subsystem_parity_summary.json"
SCIENTIFIC = ROOT / "audit" / "scientific_status.json"
DASHBOARD = ROOT / "provenance" / "symbol_closure_dashboard.json"
ERROR = ROOT / "audit" / "error_distribution.json"


def main():
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    cases = [row for row in rows if "summary" not in row]
    parity_rows = []
    for row in cases:
        scalar = isinstance(row.get("python_value"), (int, float))
        parity_rows.append({
            "suite": row.get("suite"), "case": row.get("case"),
            "quantity": row.get("quantity"),
            "python_output": row.get("python_value"),
            "native_output": row.get("oracle_value"),
            "absolute_error": row.get("abs_error") if scalar else None,
            "relative_error": row.get("rel_error") if scalar else row.get("max_rel_error_group"),
            "comparison_kind": "scalar" if scalar else "aggregate_vector_maximum",
            "oracle_role": row.get("oracle_role"),
            "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "limitation": None if scalar else
                "underlying vector elements are absent; ULP and mixed errors unavailable",
        })
    PARITY.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                              for row in parity_rows), encoding="utf-8")

    scientific = json.loads(SCIENTIFIC.read_text(encoding="utf-8"))
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    error = json.loads(ERROR.read_text(encoding="utf-8"))
    summary = {
        "schema_version": 2,
        "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "complete": False,
        "whole_library_percentage": None,
        "percentage_policy": "No single percentage is scientifically defensible.",
        "runtime": {
            "native_cantera_imported": False,
            "native_cantera_shared_library_required": False,
            "parity_numerical_dependency": "scikit-sundae 1.1.3 / SUNDIALS 7.5.0",
            "native_python_sundials_loaded": False,
        },
        "focused_archived_fixture": {
            "records": len(parity_rows),
            "scalar_records": error["scalar_case_count"],
            "aggregate_vector_records": error["aggregate_vector_case_count"],
            "max_scalar_ulp": json.loads((ROOT / "audit" / "ulp_distribution.json").read_text())["max_ulp_distance"],
            "scope_limit": error["scope_limit"],
        },
        "solver_status_counts": scientific["counts"],
        "solver_complete": scientific["complete"],
        "canonical_symbol_closure": {
            key: dashboard[key] for key in (
                "TOTAL", "PINNED_NATIVE_VALIDATED",
                "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED",
                "NON_RUNTIME_CPP_ONLY_PROVEN", "PARTIAL", "NOT_PORTED",
                "fully_dispositioned", "remaining_without_final_disposition",
                "FULL_PORT_COMPLETE")
        },
        "open_solver_families": [
            {"subsystem": row["subsystem"], "status": row["status"]}
            for row in scientific["subsystems"]
        ],
        "authoritative_artifacts": [
            "audit/complete_solver_inventory.jsonl",
            "audit/native_environment_manifest.json",
            "audit/scientific_status.json",
            "audit/remaining_gaps.jsonl",
            "provenance/symbol_closure_dashboard.json",
        ],
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parity_records": len(parity_rows),
                      "solver_counts": Counter(x["status"] for x in scientific["subsystems"]),
                      "complete": False}))


if __name__ == "__main__":
    main()
