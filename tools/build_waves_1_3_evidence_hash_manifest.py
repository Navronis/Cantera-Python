#!/usr/bin/env python3
"""Hash preserved Waves 1-3 inputs, outputs, probes, and test logs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"

FILES = [
    "audit/corpus_wave1.jsonl", "audit/raw_native_wave1.jsonl", "audit/raw_python_wave1.jsonl",
    "audit/corpus_wave1_failure_state.jsonl", "audit/raw_native_wave1_failure_state.jsonl",
    "audit/raw_python_wave1_failure_state.jsonl", "audit/wave1_failure_state_comparison.json",
    "tools/compare_native_vs_python.py", "audit/corpus_wave2.jsonl",
    "audit/raw_native_wave2.jsonl", "audit/raw_python_wave2.jsonl",
    "audit/raw_native_wave2_internal.jsonl", "audit/raw_python_wave2_internal.jsonl",
    "audit/wave2_return_code_distribution.json", "audit/wave2_internal_root_parity.jsonl",
    "audit/wave2_internal_root_parity_summary.md", "tools/compare_wave2_internals.py", "tools/build_wave2_return_code_distribution.py",
    "tools/run_native_cubic_internal_probe.cpp", "tools/run_native_cubic_internal_probe",
    "audit/corpus_wave3.jsonl", "audit/raw_native_wave3.jsonl", "audit/raw_python_wave3.jsonl",
    "audit/corpus_wave3_singular_jacobian.jsonl", "audit/raw_native_wave3_singular.jsonl",
    "audit/raw_python_wave3_singular.jsonl", "audit/wave3_fallback_activation_probe.json",
    "audit/wave3_singular_jacobian_comparison.json", "audit/wave3_cp_cv_step_sweep.jsonl",
    "tools/test_wave3_singular_differential.py", "tools/generate_wave3_fallback_activation_corpus.py",
    "audit/wave3_cp_cv_formula_audit.md", "audit/wave3_cp_cv_perturbation_results.json",
    "audit/wave3_metric_reconciliation.md", "audit/native_oracle_identity.json",
    "audit/corpus_wave1_failure_state.jsonl", "audit/raw_native_wave1_failure_state.jsonl",
    "audit/raw_python_wave1_failure_state.jsonl", "audit/wave1_failure_state_comparison.json",
    "tools/wave1_failure_state_probe.py", "tools/compare_wave1_failure_state.py",
    "audit/wave3_implemented_differently_audit.json", "audit/wave3_implemented_differently_disposition.md",
    "audit/waves_1_3_focused_unittest.log", "audit/waves_1_3_focused_unittest_results.json",
    "audit/waves_1_3_full_pytest.log", "audit/waves_1_3_test_results.json",
    "audit/waves_1_3_test_run_provenance.json", "audit/waves_1_3_native_rerun_provenance.json",
    "audit/native_oracle_rehash_verification.json", "tests/test_tpx_ieee_arithmetic.py",
    "cantera_python/state.py", "cantera_python/ideal_gas.py", "cantera_python/cubic_eos.py",
    "cantera_python/tpx.py",
    "audit/waves_1_3_repository_identity.json", "audit/waves_1_3_validation_gate.json",
    "audit/waves_1_3_current_head_gate.json",
    "tools/run_waves_1_3_pytest.py", "tools/run_waves_1_3_focused_unittest.py",
    "tools/build_waves_1_3_repository_identity.py", "tools/bind_waves_1_3_outputs.py",
    "tools/validate_waves_1_3_current_head.py", "tools/refresh_waves_1_3_current_head_manifest.py",
]


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def hash_file(rel):
    path = ROOT / rel
    return {"path": rel, "exists": path.is_file(), "sha256": sha_file(path) if path.is_file() else None, "size_bytes": path.stat().st_size if path.is_file() else None}


def main():
    identity = json.loads((AUDIT / "waves_1_3_repository_identity.json").read_text(encoding="utf-8"))
    native_identity = json.loads((AUDIT / "native_oracle_identity.json").read_text(encoding="utf-8"))
    native_provenance_path = AUDIT / "waves_1_3_native_rerun_provenance.json"
    native_provenance = json.loads(native_provenance_path.read_text(encoding="utf-8")) if native_provenance_path.is_file() else {}
    rehash_path = AUDIT / "native_oracle_rehash_verification.json"
    rehash = json.loads(rehash_path.read_text(encoding="utf-8")) if rehash_path.is_file() else {}
    native_rehash_verified = bool(
        rehash.get("verified") is True
        and rehash.get("pinned_native_commit") == native_identity.get("exact_commit")
        and native_provenance.get("native_rehash_verified") is True
    )
    rows = [hash_file(rel) for rel in FILES]
    report = {
        "schema": "waves-1-3-evidence-hash-manifest-v1",
        "waves_1_3_snapshot_id": identity["waves_1_3_snapshot_id"],
        "repository_identity_path": "audit/waves_1_3_repository_identity.json",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pinned_native_commit": native_identity.get("exact_commit"),
        "native_binary_identity": {
            "native_library_sha256_recorded_by_oracle_identity": native_identity.get("native_library_static_sha256"),
            "native_extension_sha256_recorded_by_oracle_identity": native_identity.get("native_extension_sha256"),
            "native_library_bytes_rehashed_for_recorded_rerun": native_rehash_verified,
            "native_oracle_environment_available_for_recorded_rerun": bool(native_provenance.get("native_rehash_verified")),
        },
        "native_rerun_provenance_sha256": sha_file(native_provenance_path) if native_provenance_path.is_file() else None,
        "native_rehash_verification_sha256": sha_file(rehash_path) if rehash_path.is_file() else None,
        "evidence_files": rows,
        "missing_evidence_paths": [row["path"] for row in rows if not row["exists"]],
        "note": "Native comparison rows and binary identity are accepted only when the separate run-provenance and rehash records verify their exact hashes and pinned commit; this manifest hashes those records and does not itself establish a clean committed snapshot.",
    }
    (AUDIT / "waves_1_3_evidence_hash_manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Hashed {sum(row['exists'] for row in rows)}/{len(rows)} evidence paths for snapshot {identity['waves_1_3_snapshot_id']}")


if __name__ == "__main__":
    main()
