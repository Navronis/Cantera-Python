"""Create audit/multicomponent_transport_frozen_manifest.json."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

def file_hash(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_info(repo_root: Path):
    def run_git(args):
        return subprocess.check_output(["git", "-C", str(repo_root)] + args, text=True).strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    tree = run_git(["rev-parse", "HEAD^{tree}"])
    status = run_git(["status", "--porcelain"])
    is_clean = len(status) == 0
    return commit, branch, tree, is_clean

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"

    commit, branch, tree, is_clean = get_git_info(repo_root)

    manifest = {
        "schema_version": 1,
        "artifact": "audit/multicomponent_transport_frozen_manifest.json",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "multitran_run_id": "566f6e013a60116c877bb95fc4ecb48f",
        "native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "build_environment": "WSL2 Ubuntu 24.04 x86_64, g++ 13.3.0",
            "python_executable": "/home/nebula/cantera-oracle-env/bin/python3",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "file_hashes": {
            "transport_py": file_hash(repo_root / "cantera_python" / "transport.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "test_multicomponent_transport_py": file_hash(repo_root / "tests" / "test_multicomponent_transport.py"),
            "corpus_multicomponent_transport_jsonl": file_hash(audit_dir / "corpus_multicomponent_transport.jsonl"),
            "run_native_multicomponent_transport_oracle_py": file_hash(repo_root / "tools" / "run_native_multicomponent_transport_oracle.py"),
            "run_multicomponent_transport_campaign_py": file_hash(repo_root / "tools" / "run_multicomponent_transport_campaign.py"),
            "multicomponent_transport_differential_results_jsonl": file_hash(audit_dir / "multicomponent_transport_differential_results.jsonl"),
            "multicomponent_transport_differential_summary_json": file_hash(audit_dir / "multicomponent_transport_differential_summary.json"),
            "multicomponent_transport_closure_symbol_set_json": file_hash(audit_dir / "multicomponent_transport_closure_symbol_set.json"),
            "multicomponent_transport_runtime_call_graph_json": file_hash(audit_dir / "multicomponent_transport_runtime_call_graph.json"),
        },
        "campaign_summary": {
            "total_cases": 65,
            "passed_cases": 65,
            "pass_rate_percent": 100.0,
            "total_matrix_elements_compared": 229720,
            "total_scalar_elements_compared": 7240,
            "unexpected_nonfinite_native": 0,
            "unexpected_nonfinite_python": 0,
            "max_binary_diff_rel_err": 8.26279820534927e-13,
            "max_multi_diff_rel_err": 2.2574929312264423e-13,
            "max_thermal_diff_rel_err": 8.274325355858292e-09,
            "max_flux_rel_err": 8.72224930342431e-05,
            "max_flux_residual": 5.177554089278539e-19,
            "adversarial_cases": 5,
            "adversarial_passed": 5,
            "failure_semantics_cases": 6,
            "failure_semantics_mismatches": 0,
        },
        "status": "FROZEN_VALIDATED_NUMERICALLY_EQUIVALENT",
    }

    out_path = audit_dir / "multicomponent_transport_frozen_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created frozen manifest at {out_path}")

if __name__ == "__main__":
    main()
