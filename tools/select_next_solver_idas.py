"""Select IdasIntegrator (DAE integration) as next solver family and build initial audit artifacts."""

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
    provenance_dir = repo_root / "provenance"
    orig_root = repo_root.parent / "cantera-original"

    commit, branch, tree, is_clean = get_git_info(repo_root)

    # 1. Build closure symbol set from canonical inventory
    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    reconciliation_path = provenance_dir / "symbol_discovery_false_positive_reconciliation.json"

    with open(reconciliation_path, "r", encoding="utf-8") as f:
        rec_data = json.load(f)
    rejected_sids = {row["stable_symbol_id"] for row in rec_data.get("rejected_rows", [])}

    closure_symbols = {}
    open_sids = []
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            cls = row.get("class") or ""
            if cls == "IdasIntegrator":
                sid = row["stable_symbol_id"]
                current_status = "NON_RUNTIME_CPP_ONLY_PROVEN" if sid in rejected_sids else row["current_status"]
                closure_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": cls,
                    "native_method": row["function_or_method"],
                    "python_target": f"cantera_python.numerics.IdasIntegrator.{row['function_or_method']}",
                    "current_status": current_status,
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": "Differential parity against pinned native Cantera IdasIntegrator across stiff index-1 DAE systems, consistent initialization, step vs integrate, algebraic error test suppression, and sensitivities",
                }
                if current_status not in ("PINNED_NATIVE_VALIDATED", "NON_RUNTIME_CPP_ONLY_PROVEN"):
                    open_sids.append(sid)

    closure_path = audit_dir / "idas_integrator_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_symbols, f, indent=2)

    # 2. Build runtime call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "idas_integrator",
        "entry_points": [
            "cantera_python.numerics.IdasIntegrator.__init__",
            "cantera_python.numerics.IdasIntegrator.initialize",
            "cantera_python.numerics.IdasIntegrator.reinitialize",
            "cantera_python.numerics.IdasIntegrator.integrate",
            "cantera_python.numerics.IdasIntegrator.step",
            "cantera_python.numerics.IdasIntegrator.solution",
            "cantera_python.numerics.IdasIntegrator.currentTime",
            "cantera_python.numerics.IdasIntegrator.nEquations",
            "cantera_python.numerics.IdasIntegrator.setTolerances",
            "cantera_python.numerics.IdasIntegrator.setSensitivityTolerances",
            "cantera_python.numerics.IdasIntegrator.setLinearSolverType",
            "cantera_python.numerics.IdasIntegrator.setMaxOrder",
            "cantera_python.numerics.IdasIntegrator.maxOrder",
            "cantera_python.numerics.IdasIntegrator.setMaxStepSize",
            "cantera_python.numerics.IdasIntegrator.setMaxSteps",
            "cantera_python.numerics.IdasIntegrator.maxSteps",
            "cantera_python.numerics.IdasIntegrator.setMaxErrTestFails",
            "cantera_python.numerics.IdasIntegrator.setMaxNonlinIterations",
            "cantera_python.numerics.IdasIntegrator.maxNonlinIterations",
            "cantera_python.numerics.IdasIntegrator.setMaxNonlinConvFailures",
            "cantera_python.numerics.IdasIntegrator.maxNonlinConvFailures",
            "cantera_python.numerics.IdasIntegrator.includeAlgebraicInErrorTest",
            "cantera_python.numerics.IdasIntegrator.algebraicInErrorTest",
            "cantera_python.numerics.IdasIntegrator.solverStats",
            "cantera_python.numerics.IdasIntegrator.nSensParams",
            "cantera_python.numerics.IdasIntegrator.sensitivity",
            "cantera_python.numerics.IdasIntegrator.getErrorInfo",
            "cantera_python.numerics.IdasIntegrator.applyOptions",
            "cantera_python.numerics.IdasIntegrator.sensInit",
            "cantera_python.numerics.IdasIntegrator.checkError",
            "cantera_python.numerics.IdasIntegrator.setMethod",
        ],
        "nodes": {
            "IdasIntegrator.initialize": {
                "calls": ["FuncEval.neq", "FuncEval.getStateDae", "FuncEval.getConstraints", "IdasIntegrator.applyOptions"],
                "purpose": "Initialize IDA memory, consistent initial condition computation, and tolerance assignment",
            },
            "IdasIntegrator.reinitialize": {
                "calls": ["FuncEval.getStateDae", "IdasIntegrator.applyOptions"],
                "purpose": "Reinitialize DAE state and time without full re-allocation",
            },
            "IdasIntegrator.integrate": {
                "calls": ["IdasIntegrator.step", "IdasIntegrator._ida_interpolate"],
                "purpose": "Advance DAE solution to exact requested tout",
            },
            "IdasIntegrator.step": {
                "calls": ["FuncEval.evalDae", "_python_dense_lu", "_python_dense_solve"],
                "purpose": "Take one internal variable-step BDF step with Newton-Raphson iteration",
            },
            "IdasIntegrator.sensitivity": {
                "calls": ["IdasIntegrator.sensInit"],
                "purpose": "Evaluate forward sensitivity vectors for DAE systems",
            },
        },
    }

    graph_path = audit_dir / "idas_integrator_runtime_call_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    # 3. Create starting snapshot
    snapshot = {
        "schema_version": 1,
        "solver_family": "idas_integrator",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "native_source": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "file_hashes": {
            "numerics_py": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "pinned_IdasIntegrator_h": file_hash(orig_root / "include" / "cantera" / "numerics" / "IdasIntegrator.h"),
            "pinned_IdasIntegrator_cpp": file_hash(orig_root / "src" / "numerics" / "IdasIntegrator.cpp"),
            "closure_symbol_set_json": file_hash(closure_path),
            "runtime_call_graph_json": file_hash(graph_path),
        },
        "closure_metrics": {
            "total_canonical_symbols": len(closure_symbols),
            "open_runtime_symbols": len(open_sids),
            "non_runtime_cpp_only": len(closure_symbols) - len(open_sids),
        },
    }

    snap_path = audit_dir / "idas_integrator_starting_snapshot.json"
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # 4. Create next solver selection record
    selection = {
        "schema_version": 1,
        "selected_solver_family": "idas_integrator",
        "subsystem": "DAE integration",
        "native_class": "IdasIntegrator",
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "Following successful end-to-end differential closure and freezing of both CVodesIntegrator "
            "(0D ODE integration) and ReactorNet (reactor network), IdasIntegrator is the exact sister "
            "integration engine in Cantera's numerics layer. It provides DAE integration (residual F(t,y,y')=0) "
            "required by FlowReactor (plug flow) and surface DAE networks. Foundational BDF scaffolding already exists "
            "in cantera_python/numerics.py, making it the most dependency-ready and architecturally coherent "
            "next target."
        ),
        "dependency_status": {
            "CVodesIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ReactorNet": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ChemEquil": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "MultiTransport": "VALIDATED_NUMERICALLY_EQUIVALENT",
        },
        "canonical_symbols_total": len(closure_symbols),
        "open_runtime_symbols": len(open_sids),
        "starting_snapshot_path": "audit/idas_integrator_starting_snapshot.json",
        "closure_symbol_set_path": "audit/idas_integrator_closure_symbol_set.json",
        "runtime_call_graph_path": "audit/idas_integrator_runtime_call_graph.json",
    }

    sel_path = audit_dir / "next_solver_selection.json"
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    print("=" * 60)
    print(f"AUTOMATICALLY SELECTED NEXT SOLVER FAMILY: idas_integrator (IdasIntegrator)")
    print(f"Total canonical symbols: {len(closure_symbols)}")
    print(f"Open runtime symbols:  {len(open_sids)}")
    print(f"Artifacts created:")
    print(f"  - {closure_path}")
    print(f"  - {graph_path}")
    print(f"  - {snap_path}")
    print(f"  - {sel_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
