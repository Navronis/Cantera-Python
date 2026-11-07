"""Create audit/multicomponent_transport_starting_snapshot.json."""

import hashlib
import json
import subprocess
from pathlib import Path

def file_hash(p: Path) -> str:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    git_head = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    git_branch = subprocess.check_output(["git", "-C", str(repo_root), "branch", "--show-current"], text=True).strip()
    git_tree = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD^{tree}"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], text=True).strip()

    native_root = repo_root.parent / "cantera-original"
    native_head = subprocess.check_output(["git", "-C", str(native_root), "rev-parse", "HEAD"], text=True).strip()

    snapshot = {
        "schema_version": 1,
        "solver_family": "multicomponent_transport",
        "timestamp_utc": "2026-11-04T21:16:00Z",
        "git": {
            "commit": git_head,
            "branch": git_branch,
            "tree": git_tree,
            "dirty": bool(git_status),
            "status_lines_count": len(git_status.splitlines()) if git_status else 0,
        },
        "native_oracle": {
            "pinned_source_commit": native_head,
            "expected_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "commit_match": (native_head == "726522be4e2a13454d8415b7ef799d621f665cf3"),
            "shared_library": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
            "python_env": "/home/nebula/cantera-oracle-env/bin/python3",
            "cantera_version": "4.0.0a2",
        },
        "key_source_hashes": {
            "cantera_python/transport.py": file_hash(repo_root / "cantera_python" / "transport.py"),
            "cantera_python/solution.py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "cantera_python/constants.py": file_hash(repo_root / "cantera_python" / "constants.py"),
        },
        "key_test_hashes": {
            "tests/test_multicomponent_transport.py": file_hash(repo_root / "tests" / "test_multicomponent_transport.py"),
            "tests/test_transport.py": file_hash(repo_root / "tests" / "test_transport.py"),
            "tests/test_mixture_transport.py": file_hash(repo_root / "tests" / "test_mixture_transport.py"),
            "tests/test_hp_transport.py": file_hash(repo_root / "tests" / "test_hp_transport.py"),
            "tests/test_dusty_gas_transport.py": file_hash(repo_root / "tests" / "test_dusty_gas_transport.py"),
        },
        "existing_multitransport_artifacts": {
            "fixtures/pinned_multitransport_trace_heavy.json": file_hash(repo_root / "tests" / "fixtures" / "pinned_multitransport_trace_heavy.json"),
            "audit/next_solver_selection.json": file_hash(audit_dir / "next_solver_selection.json"),
            "audit/wave4_matrix_harmonic_comparison.json": file_hash(audit_dir / "wave4_matrix_harmonic_comparison.json"),
            "audit/wave4_matrix_trace_heavy_comparison.json": file_hash(audit_dir / "wave4_matrix_trace_heavy_comparison.json"),
            "audit/wave4_matrix_uniform_comparison.json": file_hash(audit_dir / "wave4_matrix_uniform_comparison.json"),
        },
        "prerequisite_solver_families_frozen": {
            "vcs_multiphase_equilibrium": True,
            "mixture_transport": True,
            "thermodynamic_state_inversion": True,
        }
    }

    out_file = audit_dir / "multicomponent_transport_starting_snapshot.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Locked starting snapshot in {out_file}")

if __name__ == "__main__":
    main()
