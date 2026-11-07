"""Select Refiner (1D adaptive grid refinement) as next solver family.

Builds starting snapshot, closure set, and runtime call graph.
"""

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
    status = run_git(["status", "--porcelain"])
    is_clean = len(status) == 0
    return commit, branch, is_clean


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    orig_root = repo_root.parent / "cantera-original"
    provenance_dir = repo_root / "provenance"

    commit, branch, is_clean = get_git_info(repo_root)

    # 1. Extract Refiner canonical symbols from inventory
    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    refiner_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            src_file = row.get("source_file", "")
            if "refine.h" in src_file or "refine.cpp" in src_file:
                sid = row["stable_symbol_id"]
                method_name = sid.split(":")[-1]
                is_non_runtime = row.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
                refiner_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": "Refiner",
                    "native_method": method_name,
                    "python_target": f"cantera_python.oned.Refiner.{method_name}",
                    "current_status": row.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": (
                        "NONE_FINAL_DISPOSITION" if is_non_runtime else
                        "Differential parity against pinned native Cantera Refiner probe across grid analysis, "
                        "slope criteria, curvature/delta criteria, ratio criteria, pruning criteria, point insertion, "
                        "and grid refinement sequences."
                    )
                }

    closure_set_path = audit_dir / "refiner_closure_symbol_set.json"
    with open(closure_set_path, "w", encoding="utf-8") as f:
        json.dump(refiner_symbols, f, indent=2)
    print(f"Generated audit/refiner_closure_symbol_set.json ({len(refiner_symbols)} symbols).")

    open_runtime_count = sum(1 for s in refiner_symbols.values() if s.get("current_status") != "NON_RUNTIME_CPP_ONLY_PROVEN")
    non_runtime_count = len(refiner_symbols) - open_runtime_count
    not_ported_count = sum(1 for s in refiner_symbols.values() if s.get("current_status") == "NOT_PORTED")
    partial_count = sum(1 for s in refiner_symbols.values() if s.get("current_status") == "PARTIAL")
    implemented_unval_count = sum(1 for s in refiner_symbols.values() if s.get("current_status") == "IMPLEMENTED_UNVALIDATED")
    final_count = sum(1 for s in refiner_symbols.values() if s.get("current_status") in ("PINNED_NATIVE_VALIDATED", "NON_RUNTIME_CPP_ONLY_PROVEN"))

    print(f"REFINER_RUNTIME_SYMBOLS_TOTAL: {open_runtime_count}")
    print(f"REFINER_FINAL_BEFORE: {final_count}")
    print(f"REFINER_IMPLEMENTED_UNVALIDATED_BEFORE: {implemented_unval_count}")
    print(f"REFINER_PARTIAL_BEFORE: {partial_count}")
    print(f"REFINER_NOT_PORTED_BEFORE: {not_ported_count}")
    print(f"REFINER_NON_RUNTIME_ROWS: {non_runtime_count}")

    # 2. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selected_solver_family": "refiner",
        "subsystem": "grid refinement",
        "native_class": "Refiner",
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "With MultiNewton (1D damped Newton solver) and BandMatrix (banded linear algebra) fully validated "
            "and frozen, the next numerical component required for full 1D flame resolution (Sim1D / OneDim) is "
            "Refiner (grid refinement). Refiner implements adaptive mesh generation based on solution gradients (slope), "
            "curvatures (curve), grid spacing ratios (ratio), and point deletion (prune)."
        ),
        "dependency_status": {
            "MultiNewton": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "BandMatrix": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "CVodesIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ReactorNet": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "IdasIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "FlowReactor": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ChemEquil": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "MultiTransport": "VALIDATED_NUMERICALLY_EQUIVALENT"
        },
        "canonical_symbols_total": len(refiner_symbols),
        "open_runtime_symbols": open_runtime_count,
        "non_runtime_proven_symbols": non_runtime_count,
        "starting_snapshot_path": "audit/refiner_starting_snapshot.json",
        "closure_symbol_set_path": "audit/refiner_closure_symbol_set.json",
        "runtime_call_graph_path": "audit/refiner_runtime_call_graph.json"
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Updated {selection_path}")

    # 3. Build starting snapshot
    snapshot = {
        "schema_version": 1,
        "artifact": "audit/refiner_starting_snapshot.json",
        "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "is_worktree_clean": is_clean,
        "solver_family": "refiner",
        "subsystem": "grid refinement",
        "native_source": {
            "commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "header": "include/cantera/oneD/refine.h",
            "source": "src/oneD/refine.cpp",
            "refine_h_sha256": file_hash(orig_root / "include" / "cantera" / "oneD" / "refine.h"),
            "refine_cpp_sha256": file_hash(orig_root / "src" / "oneD" / "refine.cpp")
        },
        "python_files": {
            "oned_py_sha256": file_hash(repo_root / "cantera_python" / "oned.py"),
            "test_onedim_py_sha256": file_hash(repo_root / "tests" / "test_onedim.py")
        },
        "closure_set_sha256": file_hash(closure_set_path),
        "inventory_counts": {
            "total_canonical_symbols": len(refiner_symbols),
            "open_runtime_symbols": open_runtime_count,
            "non_runtime_proven_symbols": non_runtime_count,
            "not_ported": not_ported_count,
            "partial": partial_count,
            "implemented_unvalidated": implemented_unval_count,
            "final": final_count,
        }
    }

    snapshot_path = audit_dir / "refiner_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Generated {snapshot_path}")

    # 4. Build runtime call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "refiner",
        "subsystem": "grid refinement",
        "public_entry_points": {
            "Refiner.analyze(n, z, x)": {
                "callees": [
                    "Refiner.maxSlope()",
                    "Refiner.maxDelta()",
                    "Refiner.maxRatio()",
                    "Refiner.prune()",
                    "Refiner.newPointNeeded(j)",
                    "Refiner.keepPoint(j)"
                ],
                "scientific_purpose": "Analyze solution profiles and identify points for addition or deletion based on adaptation criteria"
            },
            "Refiner.nNewPoints()": {
                "callees": [],
                "scientific_purpose": "Return number of new points to be inserted into 1D mesh"
            },
            "Refiner.newGrid(n, z, nn, zn)": {
                "callees": ["Refiner.analyze(n, z, x)"],
                "scientific_purpose": "Generate refined coordinate vector zn by inserting and deleting grid nodes"
            }
        }
    }

    call_graph_path = audit_dir / "refiner_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Generated {call_graph_path}")


if __name__ == "__main__":
    main()
