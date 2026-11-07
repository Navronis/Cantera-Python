"""Lock starting snapshot for CVodesIntegrator solver family closure."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

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

    import numpy as np
    import scipy

    closure_path = audit_dir / "cvodes_integrator_closure_symbol_set.json"
    call_graph_path = audit_dir / "cvodes_integrator_runtime_call_graph.json"

    snapshot = {
        "schema_version": 1,
        "artifact": "audit/cvodes_integrator_starting_snapshot.json",
        "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_identity": {
            "git_commit": commit,
            "git_branch": branch,
            "git_tree": tree,
            "is_worktree_clean": is_clean,
        },
        "python_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_sundae_installed": False,
            "canonical_runtime_backend": "PURE_PYTHON_BDF",
        },
        "native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "build_environment": "WSL2 Ubuntu 24.04 x86_64, g++ 13.3.0",
            "python_executable": "/home/nebula/cantera-oracle-env/bin/python3",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "file_hashes": {
            "numerics_py": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "reactor_py": file_hash(repo_root / "cantera_python" / "reactor.py"),
            "constants_py": file_hash(repo_root / "cantera_python" / "constants.py"),
            "test_numerics_py": file_hash(repo_root / "tests" / "test_numerics.py"),
            "test_reactor_network_py": file_hash(repo_root / "tests" / "test_reactor_network.py"),
            "test_reactor_conservation_py": file_hash(repo_root / "tests" / "test_reactor_conservation.py"),
            "closure_symbol_set_json": file_hash(closure_path),
            "runtime_call_graph_json": file_hash(call_graph_path),
        },
        "frozen_solver_dependencies": {
            "vcs_frozen_manifest_json": file_hash(audit_dir / "vcs_frozen_manifest.json"),
            "mixture_transport_frozen_manifest_json": file_hash(audit_dir / "mixture_transport_frozen_manifest.json"),
            "multicomponent_transport_frozen_manifest_json": file_hash(audit_dir / "multicomponent_transport_frozen_manifest.json"),
            "dusty_gas_transport_frozen_manifest_json": file_hash(audit_dir / "dusty_gas_transport_frozen_manifest.json"),
            "high_pressure_gas_transport_frozen_manifest_json": file_hash(audit_dir / "high_pressure_gas_transport_frozen_manifest.json"),
            "chem_equil_frozen_manifest_json": file_hash(audit_dir / "chem_equil_frozen_manifest.json"),
        },
    }

    out_path = audit_dir / "cvodes_integrator_starting_snapshot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Locked starting snapshot at {out_path}")

if __name__ == "__main__":
    main()
