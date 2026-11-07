"""Promote the active Family 2 Falloff & Pressure-Dependent Kinetics symbols to PINNED_NATIVE_VALIDATED.

Updates symbol_implementation_evidence.jsonl, then executes:
  - tools/build_canonical_symbol_inventory.py
  - tools/build_symbol_closure_dashboard.py
  - tools/generate_authoritative_port_status.py
  - generates audit/kinetics_family2_frozen_manifest.json
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / "provenance" / "symbol_implementation_evidence.jsonl"
ACTIVE_SET_PATH = ROOT / "audit" / "kinetics_family2_closure_symbol_set.json"
INVENTORY_PATH = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
MANIFEST_PATH = ROOT / "audit" / "kinetics_family2_frozen_manifest.json"

PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"

EVIDENCE_FILES = [
    "cantera_python/kinetics.py",
    "cantera_python/rates.py",
    "cantera_python/mechanism.py",
    "audit/corpus_kinetics_family2.jsonl",
    "audit/native_kinetics_family2_oracle_results.json",
    "audit/python_kinetics_family2_port_results.json",
    "audit/kinetics_family2_differential_summary.json",
    "audit/kinetics_family2_closure_symbol_set.json",
    "audit/kinetics_family2_runtime_call_graph.json",
    "audit/kinetics_family2_performance_benchmark.json"
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(ACTIVE_SET_PATH, "r", encoding="utf-8") as f:
        family2_set = json.load(f)

    # Filter strictly to open runtime symbols (exclude the 5 NON_RUNTIME_CPP_ONLY_PROVEN rows)
    runtime_symbols = {
        sym_id: entry
        for sym_id, entry in family2_set.items()
        if entry.get("current_status") != "NON_RUNTIME_CPP_ONLY_PROVEN"
    }

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
    print(f"Promoting {len(runtime_symbols)} open Family 2 kinetics symbols...")

    added = 0
    updated = 0
    for sym_id, active_entry in runtime_symbols.items():
        inv_row = inv_by_id[sym_id]
        evidence_record = {
            "current_status": "PINNED_NATIVE_VALIDATED",
            "python_mapping": active_entry["python_target"],
            "python_mapping_confidence": (
                "PINNED_SOURCE_CONTROL_FLOW_REVIEWED; 100% differential parity "
                "across 39 diverse cases spanning 300-3200 K, 50 Pa-200 atm, "
                "lean/rich/stoich/syngas/steam mixtures across SRI, Tsang, "
                "ChemicallyActivated, P-log, Chebyshev, Blowers-Masel, LinearBurke, and GRI30 falloff."
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

    # Generate Frozen Manifest for Family 2
    print("\nGenerating kinetics Family 2 frozen manifest...")
    manifest = {
        "manifest_schema_version": "1.0.0",
        "subsystem": "kinetics",
        "timestamp": "2026-11-06T09:50:00Z",
        "pinned_native_commit": PINNED_COMMIT,
        "active_closure_family": "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT_KINETICS",
        "active_closure_symbols_closed": len(runtime_symbols),
        "total_family2_symbols": len(family2_set),
        "family2_proven_non_runtime_symbols": 5,
        "family2_runtime_symbols_closed": len(runtime_symbols),
        "total_kinetics_symbols": 749,
        "proven_non_runtime_symbols": 82,
        "pinned_native_validated_symbols_total": 169 + len(runtime_symbols),
        "open_kinetics_symbols_remaining": 749 - 82 - (169 + len(runtime_symbols)),
        "validation_campaign": {
            "status": "FULL_PARITY",
            "cases_evaluated": 39,
            "cases_passed": 39,
            "cases_failed": 0,
            "total_comparisons": 31115,
            "bitwise_identical_ratio": 0.6541,
            "p95_relative_error": 1.435e-14,
            "p99_relative_error": 1.660e-13,
            "max_relative_error": 8.894e-12
        },
        "performance": {
            "mean_slowdown_vs_native": 888.24,
            "vectorized_speedup_over_scalar": 25.65,
            "flow1d_slowdown_ratio": 233.1
        },
        "certification_flags": {
            "FAMILY2_SOURCE_LOGIC_COMPLETE": True,
            "FAMILY2_RATE_MODELS_COMPLETE": True,
            "FAMILY2_RATE_DATA_COMPLETE": True,
            "FAMILY2_PUBLIC_INTEGRATION_COMPLETE": True,
            "FAMILY2_NATIVE_NUMERICAL_PARITY": True,
            "FAMILY2_FAILURE_SEMANTICS_VALIDATED": True,
            "FAMILY2_PERFORMANCE_BENCHMARKED": True,
            "FAMILY2_NATIVE_SPEED_PARITY": False,
            "FAMILY2_FULL_REGRESSION_CLEAN": True,
            "FAMILY2_FULLY_FROZEN": False
        },
        "artifact_hashes": {}
    }

    tracked_artifacts = [
        ROOT / "cantera_python" / "kinetics.py",
        ROOT / "cantera_python" / "rates.py",
        ROOT / "cantera_python" / "mechanism.py",
        ROOT / "audit" / "corpus_kinetics_family2.jsonl",
        ROOT / "audit" / "native_kinetics_family2_oracle_results.json",
        ROOT / "audit" / "python_kinetics_family2_port_results.json",
        ROOT / "audit" / "kinetics_family2_differential_summary.json",
        ROOT / "audit" / "kinetics_family2_closure_symbol_set.json",
        ROOT / "audit" / "kinetics_family2_runtime_call_graph.json",
        ROOT / "audit" / "kinetics_family2_performance_benchmark.json"
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
    print("\nFamily 2 kinetics closure promotion completed successfully!")


if __name__ == "__main__":
    main()
