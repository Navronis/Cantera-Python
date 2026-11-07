"""Select OneDim (1D residual assembly and multi-domain evaluation) as the next solver family."""

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
    status = run_git(["status", "--porcelain"])
    is_clean = len(status) == 0
    return commit, branch, is_clean


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    orig_root = repo_root.parent / "cantera-original"
    provenance_dir = repo_root / "provenance"

    commit, branch, is_clean = get_git_info(repo_root)

    inv_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    closure_symbols = {}
    with open(inv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sf = rec.get("source_file", "")
            cls = rec.get("class", "")
            sid = rec["stable_symbol_id"]
            if (
                ("include/cantera/oneD/OneDim.h" in sf)
                or ("src/oneD/OneDim.cpp" in sf)
                or (cls == "OneDim")
            ):
                closure_symbols[sid] = {
                    "canonical_index": rec["canonical_index"],
                    "stable_symbol_id": sid,
                    "native_file": rec["source_file"],
                    "native_class": rec.get("class") or "OneDim",
                    "native_method": rec["function_or_method"],
                    "python_target": f"cantera_python.oned.OneDim.{rec['function_or_method']}" if rec.get("class") == "OneDim" else f"cantera_python.oned.OneDim.{rec['function_or_method']}",
                    "current_status": rec.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": rec.get("scientific_runtime_relevance"),
                    "callers": rec.get("callers"),
                    "callees": rec.get("callees"),
                    "remaining_validation_requirement": "Differential parity against pinned native Cantera OneDim probe across multi-domain registration, locate, weightedNorm, eval, evalJacobian, time integration setup, steady mode, bounds, and component queries."
                }

    print(f"Found {len(closure_symbols)} symbols in OneDim closure set.")

    # 1. Save closure symbol set
    closure_file = audit_dir / "onedim_closure_symbol_set.json"
    with open(closure_file, "w", encoding="utf-8") as f:
        json.dump(closure_symbols, f, indent=2)

    # 2. Save starting snapshot
    oned_h = orig_root / "include" / "cantera" / "oneD" / "OneDim.h"
    oned_cpp = orig_root / "src" / "oneD" / "OneDim.cpp"
    oned_py = repo_root / "cantera_python" / "oned.py"
    test_onedim_py = repo_root / "tests" / "test_onedim.py"

    hasher = hashlib.sha256()
    with open(closure_file, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    closure_hash = hasher.hexdigest()

    snapshot = {
        "schema_version": 1,
        "artifact": "audit/onedim_starting_snapshot.json",
        "snapshot_timestamp_utc": "2026-11-05T14:05:00Z",
        "git_commit": commit,
        "git_branch": branch,
        "is_worktree_clean": is_clean,
        "solver_family": "onedim",
        "subsystem": "1D residual",
        "native_source": {
            "commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "header": "include/cantera/oneD/OneDim.h",
            "source": "src/oneD/OneDim.cpp",
            "onedim_h_sha256": file_hash(oned_h),
            "onedim_cpp_sha256": file_hash(oned_cpp),
        },
        "python_files": {
            "oned_py_sha256": file_hash(oned_py),
            "test_onedim_py_sha256": file_hash(test_onedim_py),
        },
        "closure_set_sha256": closure_hash,
        "inventory_counts": {
            "total_canonical_symbols": len(closure_symbols),
            "open_runtime_symbols": sum(1 for s in closure_symbols.values() if s["current_status"] in ("NOT_PORTED", "PARTIAL")),
            "non_runtime_proven_symbols": sum(1 for s in closure_symbols.values() if s["current_status"] == "NON_RUNTIME_CPP_ONLY_PROVEN"),
            "pinned_native_validated": sum(1 for s in closure_symbols.values() if s["current_status"] == "PINNED_NATIVE_VALIDATED"),
        }
    }

    snap_file = audit_dir / "onedim_starting_snapshot.json"
    with open(snap_file, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # 3. Call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "onedim",
        "subsystem": "1D residual",
        "entry_points": [
            "OneDim::eval",
            "OneDim::evalJacobian",
            "OneDim::weightedNorm",
            "OneDim::addDomain",
            "OneDim::resize",
            "OneDim::initTimeInteg",
            "OneDim::setSteadyMode"
        ],
        "dependencies": {
            "Domain1D": ["locate", "eval", "bandwidth", "hasAnalyticJacobian", "evalJacobianAnalytic", "initTimeInteg", "setSteadyMode"],
            "SystemJacobian": ["setValue", "reset", "incrementEvals", "setAge", "updateElapsed"],
            "MultiJac": ["initialize", "setBandwidth"],
            "SteadyStateSystem": ["eval", "evalJacobian", "weightedNorm", "initTimeInteg", "setSteadyMode", "resize"]
        }
    }
    cg_file = audit_dir / "onedim_runtime_call_graph.json"
    with open(cg_file, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    # 4. Update next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selected_solver_family": "onedim",
        "subsystem": "1D residual",
        "native_class": "OneDim",
        "native_header": "include/cantera/oneD/OneDim.h",
        "native_source": "src/oneD/OneDim.cpp",
        "python_target": "cantera_python.oned.OneDim",
        "closure_symbol_count": len(closure_symbols),
        "status": "IN_PROGRESS",
        "prerequisites_met": [
            "BandMatrix: VALIDATED_NUMERICALLY_EQUIVALENT",
            "MultiNewton: VALIDATED_NUMERICALLY_EQUIVALENT",
            "Refiner: VALIDATED_NUMERICALLY_EQUIVALENT"
        ],
        "objective": "Complete exact source logic and pinned differential validation for OneDim multi-domain container, residual evaluator, finite-difference Jacobian assembly, and weightedNorm."
    }
    with open(audit_dir / "next_solver_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    print(f"OneDim successfully selected. Snapshot saved to {snap_file}")


if __name__ == "__main__":
    main()
