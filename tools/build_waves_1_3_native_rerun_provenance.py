#!/usr/bin/env python3
"""Bind freshly produced Waves 1-3 raw records to repository/source hashes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
PIN = "726522be4e2a13454d8415b7ef799d621f665cf3"
OUT = AUDIT / "waves_1_3_native_rerun_provenance.json"


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def same_ids(*paths: Path) -> bool:
    sets = [{row.get("case_id") for row in rows(path)} for path in paths]
    return bool(sets and sets[0] and all(ids == sets[0] for ids in sets[1:]))


def main() -> int:
    identity = json.loads((AUDIT / "waves_1_3_repository_identity.json").read_text(encoding="utf-8"))
    oracle = json.loads((AUDIT / "native_oracle_identity.json").read_text(encoding="utf-8"))
    rehash_path = AUDIT / "native_oracle_rehash_verification.json"
    rehash = json.loads(rehash_path.read_text(encoding="utf-8")) if rehash_path.is_file() else {}

    run_groups = {
        "wave1_native": same_ids(AUDIT / "corpus_wave1.jsonl", AUDIT / "raw_native_wave1.jsonl"),
        "wave1_python": same_ids(AUDIT / "corpus_wave1.jsonl", AUDIT / "raw_python_wave1.jsonl"),
        "wave1_failure_native_python": same_ids(AUDIT / "corpus_wave1_failure_state.jsonl", AUDIT / "raw_native_wave1_failure_state.jsonl", AUDIT / "raw_python_wave1_failure_state.jsonl"),
        "wave2_public_native": same_ids(AUDIT / "corpus_wave2.jsonl", AUDIT / "raw_native_wave2.jsonl"),
        "wave2_public_python": same_ids(AUDIT / "corpus_wave2.jsonl", AUDIT / "raw_python_wave2.jsonl"),
        "wave2_internal_native": same_ids(AUDIT / "corpus_wave2.jsonl", AUDIT / "raw_native_wave2_internal.jsonl"),
        "wave2_internal_python": same_ids(AUDIT / "corpus_wave2.jsonl", AUDIT / "raw_python_wave2_internal.jsonl"),
        "wave3_native": same_ids(AUDIT / "corpus_wave3.jsonl", AUDIT / "raw_native_wave3.jsonl"),
        "wave3_python": same_ids(AUDIT / "corpus_wave3.jsonl", AUDIT / "raw_python_wave3.jsonl"),
        "wave3_singular_native": same_ids(AUDIT / "corpus_wave3_singular_jacobian.jsonl", AUDIT / "raw_native_wave3_singular.jsonl"),
        "wave3_singular_python": same_ids(AUDIT / "corpus_wave3_singular_jacobian.jsonl", AUDIT / "raw_python_wave3_singular.jsonl"),
    }
    fallback = json.loads((AUDIT / "wave3_fallback_activation_probe.json").read_text(encoding="utf-8"))
    run_groups["wave3_fallback"] = len(fallback.get("all_attempts", [])) == fallback.get("attempt_count")
    sweep = rows(AUDIT / "wave3_cp_cv_step_sweep.jsonl")
    run_groups["wave3_cp_cv_step_sweep"] = bool(sweep and all("nat_cv_fd" in row and "py_cv_fd" in row for row in sweep))

    raw_paths = [
        "audit/raw_native_wave1.jsonl", "audit/raw_python_wave1.jsonl",
        "audit/raw_native_wave1_failure_state.jsonl", "audit/raw_python_wave1_failure_state.jsonl",
        "audit/raw_native_wave2.jsonl", "audit/raw_python_wave2.jsonl",
        "audit/raw_native_wave2_internal.jsonl", "audit/raw_python_wave2_internal.jsonl",
        "audit/raw_native_wave3.jsonl", "audit/raw_python_wave3.jsonl",
        "audit/raw_native_wave3_singular.jsonl", "audit/raw_python_wave3_singular.jsonl",
        "audit/wave3_cp_cv_step_sweep.jsonl",
    ]
    source_paths = [
        "cantera_python/state.py", "cantera_python/thermo.py", "cantera_python/ideal_gas.py",
        "cantera_python/cubic_eos.py", "cantera_python/tpx.py", "cantera_python/tpx_exact.py",
        "tools/run_native_oracle_runner.py", "tools/run_python_port_runner.py",
        "tools/run_native_cubic_internal_probe.cpp", "tools/run_python_cubic_internal_probe.py",
        "tools/wave1_failure_state_probe.py", "tools/test_wave3_singular_differential.py",
        "tools/wave3_cp_cv_step_sweep.py", "tools/generate_wave3_fallback_activation_corpus.py",
    ]
    record = {
        "schema": "waves-1-3-native-rerun-provenance-v1",
        "waves_1_3_snapshot_id": identity.get("waves_1_3_snapshot_id"),
        "git_commit": identity.get("git_commit"),
        "branch": identity.get("branch"),
        "pinned_native_commit": PIN,
        "native_library_sha256": oracle.get("native_library_static_sha256"),
        "native_extension_sha256": oracle.get("native_extension_sha256"),
        "native_rehash_verification_sha256": sha(rehash_path),
        "native_rehash_verified": rehash.get("verified") is True,
        "completed_runs": [name for name, ok in run_groups.items() if ok],
        "run_group_checks": run_groups,
        "raw_result_sha256": {name: sha(ROOT / name) for name in raw_paths},
        "execution_source_sha256": {name: sha(ROOT / name) for name in source_paths},
        "native_probe_binary_sha256": sha(ROOT / "tools" / "run_native_cubic_internal_probe"),
        "run_started_at_utc": min(datetime.fromtimestamp((ROOT / name).stat().st_mtime, timezone.utc).isoformat()
                                   for name in raw_paths if (ROOT / name).is_file()),
        "provenance_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Raw execution records were rerun in the current checkout. This report records observed raw files and source hashes; the certification gate independently recomputes case and parity results.",
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}; completed run groups: {record['completed_runs']}")
    return 0 if all(run_groups.values()) and record["native_rehash_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
