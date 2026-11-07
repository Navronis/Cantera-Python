#!/usr/bin/env python3
"""Refresh the non-final current-worktree Waves 1-3 manifest from the gate."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = AUDIT / "thermodynamic_solver_waves_1_3_current_head_manifest.json"


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> None:
    identity = json.loads((AUDIT / "waves_1_3_repository_identity.json").read_text(encoding="utf-8"))
    gate = json.loads((AUDIT / "waves_1_3_validation_gate.json").read_text(encoding="utf-8"))
    impl = {rel: sha(ROOT / rel) for rel in (
        "cantera_python/state.py", "cantera_python/ideal_gas.py",
        "cantera_python/cubic_eos.py", "cantera_python/tpx.py",
    )}
    evidence = json.loads((AUDIT / "waves_1_3_evidence_hash_manifest.json").read_text(encoding="utf-8")) if (AUDIT / "waves_1_3_evidence_hash_manifest.json").is_file() else None
    result = {
        "manifest_name": "thermodynamic_solver_waves_1_3_current_head_manifest",
        "status": "RECERTIFICATION_OPEN",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "waves_1_3_snapshot_id": identity["waves_1_3_snapshot_id"],
        "repository_identity": {
            "commit": identity["git_commit"], "branch": identity["branch"],
            "snapshot_kind": identity["snapshot_kind"], "clean": identity["clean"],
            "working_tree_diff_sha256": identity["working_tree_diff_sha256"],
            "tracked_worktree_tree_sha256": identity["tracked_worktree_tree_sha256"],
        },
        "pinned_native_commit": gate["pinned_native_commit"],
        "native_identity_validation": gate["native_identity_validation"],
        "implementation_sha256": impl,
        "validation_gate_path": "audit/waves_1_3_validation_gate.json",
        "validation_gate_sha256": sha(AUDIT / "waves_1_3_validation_gate.json"),
        "validation_flags": gate["flags"],
        "evidence_hash_manifest_path": "audit/waves_1_3_evidence_hash_manifest.json" if evidence else None,
        "evidence_hash_manifest_sha256": sha(AUDIT / "waves_1_3_evidence_hash_manifest.json") if evidence else None,
        "claim_scope": "Current dirty working tree. Historical native records are not current-snapshot certification.",
        "committed_manifest_created": False,
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Refreshed {OUT}")


if __name__ == "__main__":
    main()
