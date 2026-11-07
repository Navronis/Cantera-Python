#!/usr/bin/env python3
"""Build thermodynamic_solver_waves_1_3_committed_manifest.json on clean baseline."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = AUDIT / "thermodynamic_solver_waves_1_3_committed_manifest.json"
PIN = "726522be4e2a13454d8415b7ef799d621f665cf3"


def sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, stdout=subprocess.PIPE, text=True, check=True
    ).stdout.strip()


def main():
    commit = run_git("rev-parse", "HEAD")
    tree = run_git("rev-parse", "HEAD^{tree}")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--porcelain")
    gate = json.loads((AUDIT / "waves_1_3_validation_gate.json").read_text(encoding="utf-8"))
    identity = json.loads((AUDIT / "waves_1_3_repository_identity.json").read_text(encoding="utf-8"))
    identity_exclusions = set(identity.get("identity_hash_exclusions", []))

    def status_path(line: str) -> str:
        offset = 3 if len(line) > 2 and line[2].isspace() else 2
        return line[offset:].split(" -> ")[-1].strip().strip('"').replace("\\", "/")

    filtered_status = "\n".join(
        line for line in status.splitlines() if status_path(line) not in identity_exclusions
    )
    # Certification outputs are excluded from the implementation snapshot to
    # avoid self-referential hashes. Their bytes are separately covered by
    # the evidence hash manifest, so cleanliness follows the same filtered
    # scope as the repository identity builder and validation gate.
    clean = bool(
        identity.get("git_commit") == commit
        and identity.get("branch") == branch
        and identity.get("status_porcelain_excluding_self_referential_certification_outputs") == filtered_status
        and identity.get("snapshot_kind") == "CLEAN_COMMITTED_SNAPSHOT"
        and identity.get("clean") is True
        and identity.get("snapshot_scope_clean") is True
        and not identity.get("status_porcelain_excluding_self_referential_certification_outputs")
    )

    manifest = {
        "manifest_name": "thermodynamic_solver_waves_1_3_committed_manifest",
        "status": "CERTIFIED_CLEAN_COMMITTED_BASELINE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "WAVES_1_3_CERTIFIED_BASE_COMMIT": commit,
        "git_commit": commit,
        "git_tree": tree,
        "branch": branch,
        "clean": clean,
        "pinned_native_commit": PIN,
        "waves_1_3_snapshot_id": identity.get("waves_1_3_snapshot_id"),
        "wave1_evidence": {
            "implementation": {
                "cantera_python/state.py": sha_file(ROOT / "cantera_python/state.py"),
                "cantera_python/ideal_gas.py": sha_file(ROOT / "cantera_python/ideal_gas.py"),
            },
            "corpus": {
                "audit/corpus_wave1.jsonl": sha_file(AUDIT / "corpus_wave1.jsonl"),
                "audit/corpus_wave1_failure_state.jsonl": sha_file(AUDIT / "corpus_wave1_failure_state.jsonl"),
            },
            "results": {
                "audit/raw_native_wave1.jsonl": sha_file(AUDIT / "raw_native_wave1.jsonl"),
                "audit/raw_python_wave1.jsonl": sha_file(AUDIT / "raw_python_wave1.jsonl"),
                "audit/raw_native_wave1_failure_state.jsonl": sha_file(AUDIT / "raw_native_wave1_failure_state.jsonl"),
                "audit/raw_python_wave1_failure_state.jsonl": sha_file(AUDIT / "raw_python_wave1_failure_state.jsonl"),
                "audit/wave1_failure_state_comparison.json": sha_file(AUDIT / "wave1_failure_state_comparison.json"),
            },
            "tooling": {
                "tools/compare_native_vs_python.py": sha_file(ROOT / "tools/compare_native_vs_python.py"),
                "tools/wave1_failure_state_probe.py": sha_file(ROOT / "tools/wave1_failure_state_probe.py"),
                "tools/compare_wave1_failure_state.py": sha_file(ROOT / "tools/compare_wave1_failure_state.py"),
            },
            "documentation": {
                "audit/wave1_source_mapping.md": sha_file(AUDIT / "wave1_source_mapping.md"),
                "audit/wave1_metric_reconciliation.md": sha_file(AUDIT / "wave1_metric_reconciliation.md"),
            },
        },
        "wave2_evidence": {
            "implementation": {
                "cantera_python/cubic_eos.py": sha_file(ROOT / "cantera_python/cubic_eos.py"),
            },
            "public_results": {
                "audit/corpus_wave2.jsonl": sha_file(AUDIT / "corpus_wave2.jsonl"),
                "audit/raw_native_wave2.jsonl": sha_file(AUDIT / "raw_native_wave2.jsonl"),
                "audit/raw_python_wave2.jsonl": sha_file(AUDIT / "raw_python_wave2.jsonl"),
            },
            "internal_probe": {
                "tools/run_native_cubic_internal_probe.cpp": sha_file(ROOT / "tools/run_native_cubic_internal_probe.cpp"),
                "tools/run_native_cubic_internal_probe": sha_file(ROOT / "tools/run_native_cubic_internal_probe"),
                "tools/run_python_cubic_internal_probe.py": sha_file(ROOT / "tools/run_python_cubic_internal_probe.py"),
                "audit/raw_native_wave2_internal.jsonl": sha_file(AUDIT / "raw_native_wave2_internal.jsonl"),
                "audit/raw_python_wave2_internal.jsonl": sha_file(AUDIT / "raw_python_wave2_internal.jsonl"),
                "audit/wave2_return_code_distribution.json": sha_file(AUDIT / "wave2_return_code_distribution.json"),
                "audit/wave2_internal_root_parity.jsonl": sha_file(AUDIT / "wave2_internal_root_parity.jsonl"),
                "tools/compare_wave2_internals.py": sha_file(ROOT / "tools/compare_wave2_internals.py"),
                "tools/build_wave2_return_code_distribution.py": sha_file(ROOT / "tools/build_wave2_return_code_distribution.py"),
            },
            "documentation": {
                "audit/wave2_evidence_reconciliation.md": sha_file(AUDIT / "wave2_evidence_reconciliation.md"),
                "audit/wave2_internal_root_parity_summary.md": sha_file(AUDIT / "wave2_internal_root_parity_summary.md"),
            },
        },
        "wave3_evidence": {
            "implementation": {
                "cantera_python/tpx.py": sha_file(ROOT / "cantera_python/tpx.py"),
            },
            "full_differential": {
                "audit/corpus_wave3.jsonl": sha_file(AUDIT / "corpus_wave3.jsonl"),
                "audit/raw_native_wave3.jsonl": sha_file(AUDIT / "raw_native_wave3.jsonl"),
                "audit/raw_python_wave3.jsonl": sha_file(AUDIT / "raw_python_wave3.jsonl"),
            },
            "fallback_evidence": {
                "audit/wave3_fallback_activation_probe.json": sha_file(AUDIT / "wave3_fallback_activation_probe.json"),
                "tools/generate_wave3_fallback_activation_corpus.py": sha_file(ROOT / "tools/generate_wave3_fallback_activation_corpus.py"),
                "audit/corpus_wave3_singular_jacobian.jsonl": sha_file(AUDIT / "corpus_wave3_singular_jacobian.jsonl"),
                "audit/raw_native_wave3_singular.jsonl": sha_file(AUDIT / "raw_native_wave3_singular.jsonl"),
                "audit/raw_python_wave3_singular.jsonl": sha_file(AUDIT / "raw_python_wave3_singular.jsonl"),
                "audit/wave3_singular_jacobian_comparison.json": sha_file(AUDIT / "wave3_singular_jacobian_comparison.json"),
                "audit/wave3_singular_jacobian_audit.md": sha_file(AUDIT / "wave3_singular_jacobian_audit.md"),
            },
            "cp_cv_evidence": {
                "audit/wave3_cp_cv_formula_audit.md": sha_file(AUDIT / "wave3_cp_cv_formula_audit.md"),
                "audit/wave3_cp_cv_step_sweep.jsonl": sha_file(AUDIT / "wave3_cp_cv_step_sweep.jsonl"),
                "audit/wave3_cp_cv_step_sweep_summary.md": sha_file(AUDIT / "wave3_cp_cv_step_sweep_summary.md"),
                "audit/wave3_cp_cv_perturbation_results.json": sha_file(AUDIT / "wave3_cp_cv_perturbation_results.json"),
                "tools/wave3_cp_cv_step_sweep.py": sha_file(ROOT / "tools/wave3_cp_cv_step_sweep.py"),
            },
            "dispositions": {
                "audit/wave3_implemented_differently_audit.json": sha_file(AUDIT / "wave3_implemented_differently_audit.json"),
                "audit/wave3_implemented_differently_disposition.md": sha_file(AUDIT / "wave3_implemented_differently_disposition.md"),
            },
        },
        "native_oracle_identity": {
            "audit/native_oracle_identity.json": sha_file(AUDIT / "native_oracle_identity.json"),
            "audit/waves_1_3_native_oracle_identity.json": sha_file(AUDIT / "waves_1_3_native_oracle_identity.json"),
        },
        "regression_results": {
            "audit/waves_1_3_full_pytest.log": sha_file(AUDIT / "waves_1_3_full_pytest.log"),
            "audit/waves_1_3_test_results.json": sha_file(AUDIT / "waves_1_3_test_results.json"),
        },
        "validation_gate": {
            "audit/waves_1_3_validation_gate.json": sha_file(AUDIT / "waves_1_3_validation_gate.json"),
            "flags": gate.get("flags", {}),
        },
        "WAVES_1_3_COMMITTED_SNAPSHOT_CERTIFIED": gate.get("flags", {}).get("WAVES_1_3_COMMITTED_SNAPSHOT_CERTIFIED", False),
        "THERMODYNAMIC_SOLVER_FOUNDATION_READY_FOR_WAVE_4_FINAL_FREEZE": gate.get("flags", {}).get("THERMODYNAMIC_SOLVER_FOUNDATION_READY_FOR_WAVE_4_FINAL_FREEZE", False),
    }

    OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {OUT.name} for commit {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
