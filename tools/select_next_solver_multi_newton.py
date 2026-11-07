"""Select MultiNewton (1D damped Newton solver) and MultiJac (banded block Jacobian) as next solver family.

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

    # 1. Extract MultiNewton and MultiJac canonical symbols from inventory
    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    solver_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            src_file = row.get("source_file", "")
            if "MultiNewton.h" in src_file or "MultiJac.h" in src_file:
                sid = row["stable_symbol_id"]
                method_name = sid.split(":")[-1]
                native_cls = "MultiNewton" if "MultiNewton" in src_file else "MultiJac"
                python_target = f"cantera_python.oned.{native_cls}.{method_name}"
                is_non_runtime = row.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
                solver_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": native_cls,
                    "native_method": method_name,
                    "python_target": python_target,
                    "current_status": row.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": (
                        "NONE_FINAL_DISPOSITION" if is_non_runtime else
                        "Differential parity against pinned native Cantera MultiNewton/MultiJac probe across "
                        "undamped Newton steps, bound step enforcement, damping iteration, convergence logic, "
                        "and pseudo-transient Jacobian updates."
                    )
                }

    closure_set_path = audit_dir / "multi_newton_closure_symbol_set.json"
    with open(closure_set_path, "w", encoding="utf-8") as f:
        json.dump(solver_symbols, f, indent=2)
    print(f"Generated audit/multi_newton_closure_symbol_set.json ({len(solver_symbols)} symbols).")

    open_runtime_count = sum(1 for s in solver_symbols.values() if s.get("current_status") != "NON_RUNTIME_CPP_ONLY_PROVEN")
    non_runtime_count = len(solver_symbols) - open_runtime_count
    not_ported_count = sum(1 for s in solver_symbols.values() if s.get("current_status") == "NOT_PORTED")
    partial_count = sum(1 for s in solver_symbols.values() if s.get("current_status") == "PARTIAL")
    implemented_unval_count = sum(1 for s in solver_symbols.values() if s.get("current_status") == "IMPLEMENTED_UNVALIDATED")
    final_count = sum(1 for s in solver_symbols.values() if s.get("current_status") in ("PINNED_NATIVE_VALIDATED", "NON_RUNTIME_CPP_ONLY_PROVEN"))

    print(f"MULTI_NEWTON_RUNTIME_SYMBOLS_TOTAL: {open_runtime_count}")
    print(f"MULTI_NEWTON_FINAL_BEFORE: {final_count}")
    print(f"MULTI_NEWTON_IMPLEMENTED_UNVALIDATED_BEFORE: {implemented_unval_count}")
    print(f"MULTI_NEWTON_PARTIAL_BEFORE: {partial_count}")
    print(f"MULTI_NEWTON_NOT_PORTED_BEFORE: {not_ported_count}")
    print(f"MULTI_NEWTON_NON_RUNTIME_ROWS: {non_runtime_count}")

    # 2. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selected_solver_family": "multi_newton",
        "subsystem": "1D Newton",
        "native_class": "MultiNewton",
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "Following the numerical freeze of BandMatrix (banded linear algebra) with 100% differential parity "
            "and sub-millisecond execution matching native LAPACK, the next essential solver family is MultiNewton "
            "(1D damped Newton solver) and MultiJac (banded block Jacobian for multi-domain 1D problems). MultiNewton "
            "solves the block-banded nonlinear steady systems arising in 1D flame and flow configurations (OneDim, Sim1D, "
            "Flow1D) using Kee et al. damping, variable component bound enforcement, and adaptive Jacobian aging."
        ),
        "dependency_status": {
            "BandMatrix": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "CVodesIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ReactorNet": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "IdasIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "FlowReactor": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ChemEquil": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "MultiTransport": "VALIDATED_NUMERICALLY_EQUIVALENT"
        },
        "canonical_symbols_total": len(solver_symbols),
        "open_runtime_symbols": open_runtime_count,
        "non_runtime_proven_symbols": non_runtime_count,
        "starting_snapshot_path": "audit/multi_newton_starting_snapshot.json",
        "closure_symbol_set_path": "audit/multi_newton_closure_symbol_set.json",
        "runtime_call_graph_path": "audit/multi_newton_runtime_call_graph.json"
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Updated {selection_path}")

    # 3. Build starting snapshot
    snapshot = {
        "schema_version": 1,
        "artifact": "audit/multi_newton_starting_snapshot.json",
        "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "is_worktree_clean": is_clean,
        "solver_family": "multi_newton",
        "subsystem": "1D Newton",
        "native_source": {
            "commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "headers": [
                "include/cantera/oneD/MultiNewton.h",
                "include/cantera/oneD/MultiJac.h",
                "include/cantera/numerics/SystemJacobian.h"
            ],
            "sources": [
                "src/oneD/MultiNewton.cpp",
                "src/oneD/MultiJac.cpp"
            ],
            "MultiNewton_h_sha256": file_hash(orig_root / "include" / "cantera" / "oneD" / "MultiNewton.h"),
            "MultiNewton_cpp_sha256": file_hash(orig_root / "src" / "oneD" / "MultiNewton.cpp"),
            "MultiJac_h_sha256": file_hash(orig_root / "include" / "cantera" / "oneD" / "MultiJac.h"),
            "MultiJac_cpp_sha256": file_hash(orig_root / "src" / "oneD" / "MultiJac.cpp"),
            "SystemJacobian_h_sha256": file_hash(orig_root / "include" / "cantera" / "numerics" / "SystemJacobian.h")
        },
        "python_files": {
            "oned_py_sha256": file_hash(repo_root / "cantera_python" / "oned.py"),
            "test_onedim_py_sha256": file_hash(repo_root / "tests" / "test_onedim.py")
        },
        "closure_set_sha256": file_hash(closure_set_path),
        "inventory_counts": {
            "total_canonical_symbols": len(solver_symbols),
            "open_runtime_symbols": open_runtime_count,
            "non_runtime_proven_symbols": non_runtime_count,
            "not_ported": not_ported_count,
            "partial": partial_count,
            "implemented_unvalidated": implemented_unval_count,
            "final": final_count,
        }
    }

    snapshot_path = audit_dir / "multi_newton_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Generated {snapshot_path}")

    # 4. Build runtime call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "multi_newton",
        "subsystem": "1D Newton",
        "public_entry_points": {
            "MultiNewton.step(x, step, r, loglevel)": {
                "callees": [
                    "SteadyStateSystem.eval(x, step)",
                    "SteadyStateSystem.linearSolver()",
                    "SystemJacobian.solve(step, step)",
                    "SystemJacobian.info()",
                    "SteadyStateSystem.componentName(row)"
                ],
                "scientific_purpose": "Compute undamped Newton correction step delta_x = -J^{-1} F(x)"
            },
            "MultiNewton.boundStep(x0, step0, r, loglevel)": {
                "callees": [
                    "SteadyStateSystem.upperBound(i)",
                    "SteadyStateSystem.lowerBound(i)",
                    "SteadyStateSystem.componentName(i)"
                ],
                "scientific_purpose": "Compute multiplier fbound in [0, 1] ensuring x0 + fbound*step0 stays within [lower, upper] bounds"
            },
            "MultiNewton.dampStep(x0, step0, x1, step1, s1, r, loglevel, writetitle)": {
                "callees": [
                    "SteadyStateSystem.weightedNorm(step0)",
                    "boundStep(x0, step0, r, loglevel-1)",
                    "step(x1, step1, r, loglevel-1)",
                    "SteadyStateSystem.weightedNorm(step1)",
                    "SteadyStateSystem.ssnorm(x1, step1)"
                ],
                "scientific_purpose": "Find damping factor alpha_k such that norm(delta x_{k+1}) < norm(delta x_k) or converges"
            },
            "MultiNewton.solve(x0, x1, r, loglevel)": {
                "callees": [
                    "SystemJacobian.age()",
                    "SteadyStateSystem.evalJacobian(x)",
                    "SteadyStateSystem.factorizeJacobian()",
                    "step(x, stp, r, loglevel-1)",
                    "SystemJacobian.incrementAge()",
                    "dampStep(x, stp, x1, stp1, s1, r, loglevel-1, write_header)"
                ],
                "scientific_purpose": "Top-level bounded damped Newton iteration solving F(x) = 0 for 1D multi-domain networks"
            },
            "MultiJac.reset()": {
                "callees": ["BandMatrix.bfill(0.0)"],
                "scientific_purpose": "Zero out Jacobian elements and reset age counter"
            },
            "MultiJac.initialize(nVars)": {
                "callees": ["BandMatrix.resize(nVars, kl, ku)"],
                "scientific_purpose": "Dimension Jacobian for nVars variables and allocate steady-state diagonal cache"
            },
            "MultiJac.setBandwidth(bw)": {
                "callees": ["BandMatrix.resize(nVars, bw, bw)"],
                "scientific_purpose": "Configure banded storage bandwidth kl=bw, ku=bw"
            },
            "MultiJac.setValue(row, col, value)": {
                "callees": ["BandMatrix.setValue(row, col, value)"],
                "scientific_purpose": "Set individual element J(row, col) and cache steady diagonal"
            },
            "MultiJac.updateTransient(rdt, mask)": {
                "callees": ["BandMatrix.setValue(n, n, v)", "MultiJac.factorize()"],
                "scientific_purpose": "Update diagonal terms M = J - mask * rdt and factorize system matrix"
            },
            "MultiJac.solve(b, x)": {
                "callees": ["BandMatrix.solve(b, x)"],
                "scientific_purpose": "Solve linear system M * x = b using banded LU factorization"
            }
        }
    }

    call_graph_path = audit_dir / "multi_newton_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Generated {call_graph_path}")


if __name__ == "__main__":
    main()
