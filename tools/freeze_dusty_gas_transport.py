"""Create audit/dusty_gas_transport_frozen_manifest.json."""

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
        "artifact": "audit/dusty_gas_transport_frozen_manifest.json",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "dusty_run_id": "a0f6357d731a18c658ed89201f122bf7",
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
            "test_dusty_gas_transport_py": file_hash(repo_root / "tests" / "test_dusty_gas_transport.py"),
            "corpus_dusty_gas_transport_jsonl": file_hash(audit_dir / "corpus_dusty_gas_transport.jsonl"),
            "dusty_gas_transport_mechanism_schema_json": file_hash(audit_dir / "dusty_gas_transport_mechanism_schema.json"),
            "run_native_dusty_gas_oracle_py": file_hash(repo_root / "tools" / "run_native_dusty_gas_oracle.py"),
            "run_dusty_gas_campaign_py": file_hash(repo_root / "tools" / "run_dusty_gas_campaign.py"),
            "raw_native_dusty_gas_transport_jsonl": file_hash(audit_dir / "raw_native_dusty_gas_transport.jsonl"),
            "raw_python_dusty_gas_transport_jsonl": file_hash(audit_dir / "raw_python_dusty_gas_transport.jsonl"),
            "dusty_gas_transport_differential_results_jsonl": file_hash(audit_dir / "dusty_gas_transport_differential_results.jsonl"),
            "dusty_gas_transport_differential_summary_json": file_hash(audit_dir / "dusty_gas_transport_differential_summary.json"),
            "dusty_gas_transport_adversarial_results_json": file_hash(audit_dir / "dusty_gas_transport_adversarial_results.json"),
            "dusty_gas_transport_failure_semantics_json": file_hash(audit_dir / "dusty_gas_transport_failure_semantics.json"),
            "dusty_gas_transport_closure_symbol_set_json": file_hash(audit_dir / "dusty_gas_transport_closure_symbol_set.json"),
            "dusty_gas_transport_runtime_call_graph_json": file_hash(audit_dir / "dusty_gas_transport_runtime_call_graph.json"),
        },
        "campaign_summary": {
            "total_cases": 66,
            "passed_cases": 66,
            "pass_rate_percent": 100.0,
            "total_matrix_elements_compared": 106833,
            "total_scalar_elements_compared": 2317,
            "unexpected_nonfinite_native": 0,
            "unexpected_nonfinite_python": 0,
            "max_multi_diff_rel_err": 2.581608754990329e-13,
            "max_flux_rel_err": 9.841407224266642e-11,
            "max_thermal_cond_rel_err": 8.711861411346242e-14,
            "adversarial_cases": 6,
            "adversarial_passed": 6,
            "failure_semantics_cases": 6,
            "failure_semantics_mismatches": 0,
        },
        "status": "VALIDATED_NUMERICALLY_EQUIVALENT",
    }

    out_path = audit_dir / "dusty_gas_transport_frozen_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created frozen manifest at {out_path}")

if __name__ == "__main__":
    main()
