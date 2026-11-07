#!/usr/bin/env python3
"""Record an auditable Git/worktree identity without modifying solver code."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "audit" / "waves_1_3_repository_identity.json"
PINNED_NATIVE = "726522be4e2a13454d8415b7ef799d621f665cf3"
# These files are outputs of the identity-bound certification step. Excluding
# them from the input diff avoids a self-referential hash; their bytes are
# separately recorded by the evidence manifest after the identity is written.
CERTIFICATION_OUTPUTS = {
    "audit/waves_1_3_repository_identity.json",
    "audit/waves_1_3_file_classification.json",
    "audit/waves_1_3_validation_gate.json",
    "audit/waves_1_3_evidence_hash_manifest.json",
    "audit/thermodynamic_solver_waves_1_3_committed_manifest.json",
    "audit/thermodynamic_solver_waves_1_3_current_head_manifest.json",
    "audit/wave2_return_code_distribution.json",
    "audit/wave3_singular_jacobian_comparison.json",
    "audit/wave3_fallback_activation_probe.json",
    "audit/wave3_cp_cv_perturbation_results.json",
    "audit/waves_1_3_test_results.json",
    "audit/waves_1_3_focused_unittest_results.json",
    "audit/waves_1_3_test_run_provenance.json",
    "audit/canonical_full_pytest.log",
    "audit/canonical_full_pytest_results.json",
    "audit/waves_1_3_native_rerun_provenance.json",
    "audit/native_oracle_rehash_verification.json",
    # Derived reports rewritten by bind_waves_1_3_outputs.py must not be part
    # of the identity input: binding the ID into these files would otherwise
    # change the next ID and invalidate the just-written native provenance.
    # Their final bytes are covered by the evidence hash manifest instead.
    "audit/native_parity_summary.json",
    "audit/wave1_failure_state_comparison.json",
    "audit/wave1_source_mapping.md",
    "audit/wave2_return_code_distribution.json",
    "audit/wave2_internal_root_parity.jsonl",
    "audit/wave2_internal_root_parity_summary.md",
    "audit/wave2_evidence_reconciliation.md",
    "audit/wave3_singular_jacobian_comparison.json",
    "audit/wave3_fallback_activation_probe.json",
    "audit/wave3_implemented_differently_audit.json",
    "audit/wave3_cp_cv_formula_audit.md",
    "audit/wave3_cp_cv_perturbation_results.json",
    "audit/wave3_cp_cv_step_sweep_summary.md",
    "audit/wave3_implemented_differently_disposition.md",
    "audit/wave3_singular_jacobian_audit.md",
    "audit/solver_wave3_final_adversarial_audit.md",
    "audit/wave3_metric_reconciliation.md",
    "audit/waves_1_3_current_head_gate.json",
    "audit/wave2_internal_root_parity.jsonl",
    "audit/wave3_implemented_differently_audit.json",
}


def run_git(*args: str, binary=False):
    result = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=True)
    return result.stdout if binary else result.stdout.decode("utf-8", "replace").strip()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def classify(path: str) -> str:
    p = Path(path)
    if p.suffix.lower() in {".md", ".rst", ".txt"}:
        return "documentation"
    if p.suffix.lower() in {".json", ".jsonl", ".csv"}:
        return "generated evidence"
    if path.startswith("cantera_python/"):
        return "scientific implementation"
    if path.startswith("tests/"):
        return "test"
    if path.startswith("audit/") or path.startswith("provenance/"):
        return "generated evidence"
    if path.startswith("tools/"):
        if p.suffix.lower() in {".py", ".cpp", ".ps1", ".sh"}:
            return "test" if p.name.startswith("test_") else "validation harness"
        return "temporary/debug file"
    return "documentation or configuration"


def extract_path(line: str) -> str:
    parts = line.strip().split(None, 1)
    path = parts[1] if len(parts) > 1 else line
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def main() -> None:
    commit = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    raw_status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    status_lines = raw_status.splitlines()
    paths = []
    for line in status_lines:
        path = extract_path(line)
        paths.append({"status": line[:2], "path": path, "classification": classify(path)})

    tracked_paths = run_git("ls-files", "-z").split("\0")
    tracked_hashes = []
    tree = hashlib.sha256()
    for rel in sorted(p for p in tracked_paths if p):
        file_path = ROOT / rel
        content_hash = sha_file(file_path) if file_path.exists() else "MISSING"
        tracked_hashes.append({"path": rel, "sha256": content_hash})
        if rel not in CERTIFICATION_OUTPUTS:
            tree.update(rel.encode("utf-8")); tree.update(b"\0"); tree.update(content_hash.encode("ascii")); tree.update(b"\n")

    untracked = run_git("ls-files", "--others", "--exclude-standard", "-z").split("\0")
    untracked_hashes = []
    diff_parts = []
    for line in status_lines:
        rel = extract_path(line)
        if rel in CERTIFICATION_OUTPUTS or line.startswith("??"):
            continue
        # Compare the working tree (including staged content) with HEAD, but
        # only for snapshot inputs. Generated identity-bound reports are
        # separately hashed after binding and are intentionally excluded.
        diff_parts.append(run_git("diff", "--binary", "HEAD", "--", rel, binary=True))
    for rel in sorted(p for p in untracked if p and p not in CERTIFICATION_OUTPUTS):
        path = ROOT / rel
        if path.is_file():
            h = sha_file(path)
            untracked_hashes.append({"path": rel, "sha256": h})
            diff_parts.append(rel.encode("utf-8") + b"\0" + h.encode("ascii") + b"\n")
    tracked_tree_hash = tree.hexdigest()
    snapshot_status_lines = [line for line in status_lines
                             if extract_path(line) not in CERTIFICATION_OUTPUTS]
    snapshot_status = "\n".join(snapshot_status_lines)
    clean = not snapshot_status_lines
    # Hash only the actual tracked diffs and untracked bytes in the filtered
    # snapshot. Hashing the unfiltered Git diff would re-introduce generated
    # reports and make the ID change when the gate binds the ID into them.
    diff_hash = sha(b"\n--FILTERED-WORKTREE-DIFF--\n".join(diff_parts))
    identity_core = {
        "git_commit": commit,
        "branch": branch,
        "status_porcelain_excluding_self_referential_certification_outputs": snapshot_status,
        "snapshot_scope_clean": not snapshot_status_lines,
        "working_tree_diff_sha256": diff_hash,
        "tracked_worktree_tree_sha256": tracked_tree_hash,
        "tracked_tree_excludes_certification_outputs": sorted(CERTIFICATION_OUTPUTS),
        "pinned_native_source_commit": PINNED_NATIVE,
    }
    snapshot_id = sha(json.dumps(identity_core, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    identity = {
        "schema": "waves-1-3-repository-identity-v1",
        "waves_1_3_snapshot_id": snapshot_id,
        "snapshot_kind": "CLEAN_COMMITTED_SNAPSHOT" if clean else "WORKING_TREE_SNAPSHOT_WITH_DIFF_HASH",
        **identity_core,
        "status_porcelain": raw_status,
        "clean": clean,
        "snapshot_status_excluding_certification_outputs": snapshot_status,
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "identity_hash_exclusions": sorted(CERTIFICATION_OUTPUTS),
        "tracked_file_count": len(tracked_hashes),
        "tracked_file_hashes": tracked_hashes,
        "untracked_file_hashes": untracked_hashes,
        "changed_path_classification": paths,
        "scope_note": "Snapshot identity records the source/evidence worktree as captured. Generated identity-bound gate and hash-manifest outputs are excluded to avoid self-reference and are separately hashed by the evidence manifest.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(f"snapshot_id={snapshot_id}")
    print(f"kind={identity['snapshot_kind']} branch={branch} commit={commit} clean={clean}")
    print(f"tracked_tree_sha256={tracked_tree_hash} working_diff_sha256={diff_hash}")
    print(f"status_entries={len(status_lines)}; evidence paths classified={len(paths)}")


if __name__ == "__main__":
    main()
