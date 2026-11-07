"""Lock starting snapshot for ChemEquil (single-phase equilibrium)."""

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
        "artifact": "audit/chem_equil_starting_snapshot.json",
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
            "family": "chem_equil",
            "native_classes": ["ChemEquil"],
            "native_file": "src/equil/ChemEquil.cpp",
            "runtime_symbols_total": 42,
            "runtime_symbols_open": 37,
            "status_before": "APPROXIMATE",
        },
        "file_hashes": {
            "equilibrium_py": file_hash(repo_root / "cantera_python" / "equilibrium.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "test_equilibrium_py": file_hash(repo_root / "tests" / "test_equilibrium.py"),
            "test_equilibrium_modes_py": file_hash(repo_root / "tests" / "test_equilibrium_modes.py"),
            "test_equilibrium_solver_py": file_hash(repo_root / "tests" / "test_equilibrium_solver.py"),
            "chem_equil_closure_set_json": file_hash(audit_dir / "chem_equil_closure_symbol_set.json"),
            "chem_equil_call_graph_json": file_hash(audit_dir / "chem_equil_runtime_call_graph.json"),
        },
        "frozen_dependencies": {
            "high_pressure_gas_transport_frozen_manifest_json": file_hash(audit_dir / "high_pressure_gas_transport_frozen_manifest.json"),
            "dusty_gas_transport_frozen_manifest_json": file_hash(audit_dir / "dusty_gas_transport_frozen_manifest.json"),
            "multicomponent_transport_frozen_manifest_json": file_hash(audit_dir / "multicomponent_transport_frozen_manifest.json"),
            "vcs_frozen_manifest_json": file_hash(audit_dir / "vcs_frozen_manifest.json"),
        },
        "global_starting_state": {
            "fully_dispositioned": 1071,
            "remaining_without_final_disposition": 4713,
        },
    }

    out_path = audit_dir / "chem_equil_starting_snapshot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Created starting snapshot at {out_path}")

if __name__ == "__main__":
    main()
