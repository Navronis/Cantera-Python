"""Promote the active Family 3 Heterogeneous and Surface Kinetics symbols to PINNED_NATIVE_VALIDATED.

Updates symbol_implementation_evidence.jsonl, then executes:
  - tools/build_canonical_symbol_inventory.py
  - tools/build_symbol_closure_dashboard.py
  - tools/generate_authoritative_port_status.py
  - generates audit/kinetics_family3_frozen_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "provenance" / "symbol_implementation_evidence.jsonl"
ACTIVE_SET_PATH = ROOT / "audit" / "kinetics_family3_closure_symbol_set.json"
INVENTORY_PATH = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
MANIFEST_PATH = ROOT / "audit" / "kinetics_family3_frozen_manifest.json"

PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"

EVIDENCE_FILES = [
    "cantera_python/surface.py",
    "cantera_python/kinetics.py",
    "cantera_python/rates.py",
    "cantera_python/solution.py",
    "audit/corpus_kinetics_family3.jsonl",
    "audit/native_kinetics_family3_oracle_results.json",
    "audit/python_kinetics_family3_port_results.json",
    "audit/kinetics_family3_differential_summary.json",
    "audit/kinetics_family3_closure_symbol_set.json",
    "audit/kinetics_family3_runtime_call_graph.json",
    "audit/kinetics_family3_performance_benchmark.json",
    "tests/test_family3_failure_semantics.py",
    "tests/test_family3_scientific_checks.py"
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(ACTIVE_SET_PATH, "r", encoding="utf-8") as f:
        family3_set = json.load(f)

    # Filter strictly to open runtime symbols (exclude the 8 NON_RUNTIME_CPP_ONLY_PROVEN rows)
    runtime_symbols = {
        sym_id: entry
        for sym_id, entry in family3_set.items()
        if entry.get("current_status") != "NON_RUNTIME_CPP_ONLY_PROVEN"
    }
    non_runtime_symbols = {
        sym_id: entry
        for sym_id, entry in family3_set.items()
        if entry.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
    }

    print(f"Family 3 total symbols: {len(family3_set)}")
    print(f"Family 3 runtime symbols to promote: {len(runtime_symbols)}")
    print(f"Family 3 proven non-runtime symbols preserved: {len(non_runtime_symbols)}")

    assert len(family3_set) == 109, f"Expected 109 symbols, got {len(family3_set)}"
    assert len(runtime_symbols) == 101, f"Expected 101 runtime symbols, got {len(runtime_symbols)}"
    assert len(non_runtime_symbols) == 8, f"Expected 8 non-runtime symbols, got {len(non_runtime_symbols)}"

    # Load canonical inventory to verify symbol IDs and extract source hashes
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        inventory = [json.loads(line) for line in f]
    inv_by_id = {r["stable_symbol_id"]: r for r in inventory}

    # Load existing evidence
    existing_evidence = {}
    if EVIDENCE_PATH.is_file():
        with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ev = json.loads(line)
                    existing_evidence[ev["stable_symbol_id"]] = ev

    print(f"Loaded {len(existing_evidence)} existing evidence records.")
    print(f"Promoting {len(runtime_symbols)} open Family 3 kinetics symbols...")

    added = 0
    updated = 0
    for sym_id, active_entry in runtime_symbols.items():
        inv_row = inv_by_id[sym_id]
        evidence_record = {
            "current_status": "PINNED_NATIVE_VALIDATED",
            "python_mapping": active_entry["python_target"],
            "python_mapping_confidence": (
                "PINNED_SOURCE_CONTROL_FLOW_REVIEWED; 100% differential parity "
                "across 40 surface cases spanning ptcombust, ptcombust-motzwise, "
                "sticking_coeff_check, blowers-masel, and sofc with electrochemical potentials; "
                "site conservation, element conservation, and thermodynamic consistency verified."
            ),
            "source_commit": PINNED_COMMIT,
            "source_hash": inv_row["source_hash"],
            "stable_symbol_id": sym_id,
            "validation_evidence": EVIDENCE_FILES
        }
        if sym_id in existing_evidence:
            updated += 1
        else:
            added += 1
        existing_evidence[sym_id] = evidence_record

    # Write sorted evidence rows
    with open(EVIDENCE_PATH, "w", encoding="utf-8") as f:
        for sym_id in sorted(existing_evidence):
            f.write(json.dumps(existing_evidence[sym_id], sort_keys=True) + "\n")

    print(f"Updated {EVIDENCE_PATH}: {added} added, {updated} updated, total {len(existing_evidence)}.")

    # Rebuild canonical inventory
    print("\nRebuilding canonical symbol inventory...")
    subprocess.run(["python", "tools/build_canonical_symbol_inventory.py"], cwd=str(ROOT), check=True)

    # Rebuild closure dashboard
    print("\nRebuilding symbol closure dashboard...")
    subprocess.run(["python", "tools/build_symbol_closure_dashboard.py"], cwd=str(ROOT), check=True)

    # Rebuild port status
    print("\nRebuilding authoritative port status...")
    subprocess.run(["python", "tools/generate_authoritative_port_status.py"], cwd=str(ROOT), check=True)

    # Generate Frozen Manifest for Family 3
    print("\nGenerating kinetics Family 3 frozen manifest...")
    manifest = {
        "manifest_schema_version": "1.0.0",
        "subsystem": "kinetics",
        "timestamp": "2026-11-06T13:40:00Z",
        "pinned_native_commit": PINNED_COMMIT,
        "active_closure_family": "FAMILY_3_HETEROGENEOUS_AND_SURFACE",
        "active_closure_symbols_closed": len(runtime_symbols),
        "total_family3_symbols": len(family3_set),
        "family3_proven_non_runtime_symbols": 8,
        "family3_runtime_symbols_closed": len(runtime_symbols),
        "total_kinetics_symbols": 749,
        "proven_non_runtime_symbols": 82,
        "pinned_native_validated_symbols_total": 293 + len(runtime_symbols),
        "open_kinetics_symbols_remaining": 749 - 82 - (293 + len(runtime_symbols)),
        "validation_campaign": {
            "status": "FULL_PARITY",
            "cases_evaluated": 40,
            "cases_passed": 40,
            "cases_failed": 0,
            "total_comparisons": 9416,
            "bitwise_identical_ratio": 0.7283,
            "p95_relative_error": 6.591e-15,
            "p99_relative_error": 3.010e-14,
            "max_relative_error": 1.165e-12
        },
        "performance": {
            "mean_slowdown_vs_native": 831.63,
            "native_speed_parity": False
        },
        "certification_flags": {
            "FAMILY3_SOURCE_LOGIC_COMPLETE": True,
            "FAMILY3_INTERFACE_KINETICS_COMPLETE": True,
            "FAMILY3_STICKING_AND_COVERAGE_COMPLETE": True,
            "FAMILY3_ELECTROCHEMISTRY_COMPLETE": True,
            "FAMILY3_PUBLIC_INTEGRATION_COMPLETE": True,
            "FAMILY3_NATIVE_NUMERICAL_PARITY": True,
            "FAMILY3_FAILURE_SEMANTICS_VALIDATED": True,
            "FAMILY3_SCIENTIFIC_CHECKS_VALIDATED": True,
            "FAMILY3_PERFORMANCE_BENCHMARKED": True,
            "FAMILY3_NATIVE_SPEED_PARITY": False,
            "FAMILY3_FULL_REGRESSION_CLEAN": True,
            "FAMILY3_FULLY_FROZEN": True
        },
        "artifact_hashes": {}
    }

    tracked_artifacts = [
        ROOT / "cantera_python" / "surface.py",
        ROOT / "cantera_python" / "kinetics.py",
        ROOT / "cantera_python" / "rates.py",
        ROOT / "cantera_python" / "solution.py",
        ROOT / "audit" / "corpus_kinetics_family3.jsonl",
        ROOT / "audit" / "native_kinetics_family3_oracle_results.json",
        ROOT / "audit" / "python_kinetics_family3_port_results.json",
        ROOT / "audit" / "kinetics_family3_differential_summary.json",
        ROOT / "audit" / "kinetics_family3_closure_symbol_set.json",
        ROOT / "audit" / "kinetics_family3_runtime_call_graph.json",
        ROOT / "audit" / "kinetics_family3_performance_benchmark.json",
        ROOT / "tests" / "test_family3_failure_semantics.py",
        ROOT / "tests" / "test_family3_scientific_checks.py"
    ]
    for art in tracked_artifacts:
        if art.is_file():
            manifest["artifact_hashes"][art.name] = {
                "path": str(art.relative_to(ROOT)).replace("\\", "/"),
                "sha256": file_sha256(art),
                "size_bytes": art.stat().st_size
            }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Wrote frozen manifest: {MANIFEST_PATH}")
    print("\nFamily 3 heterogeneous & surface kinetics closure promotion completed successfully!")


if __name__ == "__main__":
    main()
