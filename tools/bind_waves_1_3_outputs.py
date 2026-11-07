#!/usr/bin/env python3
"""Attach the current repository identity to derived Waves 1-3 reports.

Raw native records are deliberately excluded: they are historical observations
and must not be represented as rerun against the current Python snapshot.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
JSON_OUTPUTS = (
    "wave2_return_code_distribution.json",
    "wave3_singular_jacobian_comparison.json",
    "wave3_fallback_activation_probe.json",
    "wave3_cp_cv_perturbation_results.json",
    "waves_1_3_test_results.json",
    "waves_1_3_focused_unittest_results.json",
    "waves_1_3_current_head_gate.json",
    "wave3_implemented_differently_audit.json",
    "wave1_failure_state_comparison.json",
)
JSONL_OUTPUTS = ("wave2_internal_root_parity.jsonl",)
MARKDOWN_OUTPUTS = (
    "wave1_source_mapping.md",
    "wave2_internal_root_parity_summary.md",
    "wave2_evidence_reconciliation.md",
    "wave3_cp_cv_formula_audit.md",
    "wave3_metric_reconciliation.md",
    "wave3_cp_cv_step_sweep_summary.md",
    "wave3_implemented_differently_disposition.md",
    "wave3_singular_jacobian_audit.md",
    "solver_wave3_final_adversarial_audit.md",
)


def main() -> None:
    identity = json.loads((AUDIT / "waves_1_3_repository_identity.json").read_text(encoding="utf-8"))
    snapshot_id = identity["waves_1_3_snapshot_id"]
    provenance_path = AUDIT / "waves_1_3_native_rerun_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    required_raw = (
        "audit/raw_native_wave1.jsonl", "audit/raw_python_wave1.jsonl",
        "audit/raw_native_wave2.jsonl", "audit/raw_python_wave2.jsonl",
        "audit/raw_native_wave2_internal.jsonl", "audit/raw_python_wave2_internal.jsonl",
        "audit/raw_native_wave3.jsonl", "audit/raw_python_wave3.jsonl",
        "audit/raw_native_wave3_singular.jsonl", "audit/raw_python_wave3_singular.jsonl",
        "audit/wave3_cp_cv_step_sweep.jsonl",
    )
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    provenance_verified = bool(
        provenance.get("waves_1_3_snapshot_id") == snapshot_id
        and provenance.get("pinned_native_commit") == "726522be4e2a13454d8415b7ef799d621f665cf3"
        and all(provenance.get("raw_result_sha256", {}).get(name) == digest(ROOT / name) for name in required_raw)
    )
    scope = "LIVE_RERUN_HASH_BOUND" if provenance_verified else "SAVED_RAW_RECORDS_NOT_REPROVEN"
    binding_note = (
        "Report generated for this repository snapshot. Raw-run freshness is separately established only when the execution-provenance file verifies its raw hashes and pinned identity."
    )
    for name in JSON_OUTPUTS:
        path = AUDIT / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        data["waves_1_3_snapshot_id"] = snapshot_id
        data["repository_snapshot_kind"] = identity["snapshot_kind"]
        data["evidence_binding_scope"] = scope
        data["evidence_binding_note"] = binding_note
        if name == "waves_1_3_current_head_gate.json":
            data["status"] = "SUPERSEDED_HISTORICAL_SAVED_RECORD_GATE"
            data["superseded_by"] = "audit/waves_1_3_validation_gate.json"
            data["supersession_note"] = "Its historical TRUE output-evidence flags are not current-snapshot certification; use the evidence-derived validation gate for present status."
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name in MARKDOWN_OUTPUTS:
        path = AUDIT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        marker = f"Repository snapshot identity: `{snapshot_id}` (`{identity['snapshot_kind']}`)."
        if len(lines) > 1 and lines[1].startswith("Repository snapshot identity:"):
            lines[1] = marker
        else:
            lines.insert(1, marker)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name in JSONL_OUTPUTS:
        path = AUDIT / name
        if not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            row["waves_1_3_snapshot_id"] = snapshot_id
            row["evidence_execution_scope"] = scope
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    print(f"Bound derived Waves 1-3 reports to {snapshot_id}")


if __name__ == "__main__":
    main()
