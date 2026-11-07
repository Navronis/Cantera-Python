"""Lock starting snapshot for DustyGasTransport."""

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

    snapshot = {
        "schema_version": 1,
        "artifact": "audit/dusty_gas_transport_starting_snapshot.json",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "pinned_native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "build_environment": "WSL2 Ubuntu 24.04 x86_64, g++ 13.3.0",
            "python_executable": "/home/nebula/cantera-oracle-env/bin/python3",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "target_family": {
            "family": "dusty_gas_transport",
            "native_class": "DustyGasTransport",
            "native_file": "src/transport/DustyGasTransport.cpp",
            "runtime_symbols_total": 18,
            "status_before": "PARTIAL",
        },
        "file_hashes": {
            "transport_py": file_hash(repo_root / "cantera_python" / "transport.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "test_dusty_gas_transport_py": file_hash(repo_root / "tests" / "test_dusty_gas_transport.py"),
            "dusty_gas_closure_set_json": file_hash(audit_dir / "dusty_gas_transport_closure_symbol_set.json"),
            "dusty_gas_call_graph_json": file_hash(audit_dir / "dusty_gas_transport_runtime_call_graph.json"),
        },
        "frozen_dependencies": {
            "mixture_transport_differential_summary_json": file_hash(audit_dir / "mixture_transport_differential_summary.json"),
            "multicomponent_transport_frozen_manifest_json": file_hash(audit_dir / "multicomponent_transport_frozen_manifest.json"),
            "multicomponent_transport_differential_summary_json": file_hash(audit_dir / "multicomponent_transport_differential_summary.json"),
        },
        "global_starting_state": {
            "fully_dispositioned": 1015,
            "remaining_without_final_disposition": 4769,
        },
    }

    out_path = audit_dir / "dusty_gas_transport_starting_snapshot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Created starting snapshot at {out_path}")

if __name__ == "__main__":
    main()
