"""Select CVodesIntegrator (0D ODE integration) as next solver family and build artifacts."""

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
            src = row.get("source_file") or ""
            if "CVodesIntegrator" in cls or "CVodesIntegrator" in src:
                sid = row["stable_symbol_id"]
                current_status = "NON_RUNTIME_CPP_ONLY_PROVEN" if sid in rejected_sids else row["current_status"]
                closure_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": cls,
                    "native_method": row["function_or_method"],
                    "python_target": f"cantera_python.numerics.CVodesIntegrator.{row['function_or_method']}",
                    "current_status": current_status,
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": "Differential parity against pinned native SUNDIALS CVODE across variable-step BDF stiff decay, oscillator, and reactor kinetics ODE benchmarks",
                }
                if current_status not in ("PINNED_NATIVE_VALIDATED", "NON_RUNTIME_CPP_ONLY_PROVEN"):
                    open_sids.append(sid)

    closure_path = audit_dir / "cvodes_integrator_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_symbols, f, indent=2)

    # 2. Build runtime call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "cvodes_integrator",
        "entry_points": [
            "cantera_python.numerics.CVodesIntegrator.initialize",
            "cantera_python.numerics.CVodesIntegrator.reinitialize",
            "cantera_python.numerics.CVodesIntegrator.integrate",
            "cantera_python.numerics.CVodesIntegrator.step",
            "cantera_python.numerics.CVodesIntegrator.setTolerances",
            "cantera_python.numerics.CVodesIntegrator.solution",
            "cantera_python.numerics.CVodesIntegrator.derivative",
        ],
        "nodes": {
            "cantera_python.numerics.CVodesIntegrator.initialize": {
                "callees": [
                    "cantera_python.numerics.Integrator.initialize",
                    "cantera_python.numerics.CVodesIntegrator._cv_initialize",
                ],
                "description": "Initialize CVODE workspace, allocate Nordsieck history array zn, set tolerances and order",
            },
            "cantera_python.numerics.CVodesIntegrator.step": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator._cv_step",
                    "cantera_python.numerics.CVodesIntegrator._cv_predict",
                    "cantera_python.numerics.CVodesIntegrator._cv_correct",
                    "cantera_python.numerics.CVodesIntegrator._cv_order_step_control",
                ],
                "description": "Perform one internal variable-order variable-step BDF step",
            },
            "cantera_python.numerics.CVodesIntegrator.integrate": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator.step",
                    "cantera_python.numerics.CVodesIntegrator.solution",
                ],
                "description": "Advance ODE solution from current time to requested tout",
            },
            "cantera_python.numerics.CVodesIntegrator._cv_correct": {
                "callees": [
                    "cantera_python.numerics.DenseMatrix.solve",
                    "cantera_python.numerics.FuncEval.eval",
                ],
                "description": "Modified Newton iteration with numerical/analytic Jacobian and pivoted LU factor/solve",
            },
        },
    }

    call_graph_path = audit_dir / "cvodes_integrator_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    # 3. Create next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "native_oracle_pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "selected_solver_family": "cvodes_integrator",
        "native_class": "CVodesIntegrator",
        "native_file": "src/numerics/CVodesIntegrator.cpp",
        "subsystem": "0D ODE integration",
        "current_family_status": "PARTIAL",
        "target_family_status": "VALIDATED_NUMERICALLY_EQUIVALENT",
        "rationale": (
            "All transport solvers (mixture, multicomponent, dusty_gas, high_pressure) and equilibrium solvers "
            "(VCS multiphase, single-phase ChemEquil) are fully validated against pinned native Cantera (726522be4e2a13454d8415b7ef799d621f665cf3) "
            "and frozen. CVodesIntegrator (0D ODE integration) is the foundational ODE solver driving Cantera's reactor network and kinetics. "
            "Its pure-Python variable-step BDF multi-step kernel (orders 1-5, Nordsieck history, Newton corrector, error test retry) "
            "is already in place with 33 open symbols in the canonical inventory."
        ),
        "prerequisite_dependencies_satisfied": [
            "thermodynamic_state_inversion",
            "ideal_gas_thermo",
            "pure_fluid_properties",
            "cubic_eos",
            "vcs_multiphase_equilibrium",
            "single_phase_equilibrium",
            "mixture_transport",
            "multicomponent_transport",
            "dusty_gas_transport",
            "high_pressure_gas_transport",
        ],
        "open_symbol_burn_down_target": len(open_sids),
        "total_closure_symbols": len(closure_symbols),
        "target_closure_subgraph": open_sids,
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    # 4. Create cvodes_integrator_starting_snapshot.json
    snapshot = {
        "schema_version": 1,
        "artifact": "audit/cvodes_integrator_starting_snapshot.json",
        "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "file_hashes": {
            "numerics_py": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "test_numerics_py": file_hash(repo_root / "tests" / "test_numerics.py"),
            "test_reactor_network_py": file_hash(repo_root / "tests" / "test_reactor_network.py"),
            "cvodes_closure_set_hash": file_hash(closure_path),
            "cvodes_call_graph_hash": file_hash(call_graph_path),
            "frozen_dependencies": {
                "vcs_frozen_manifest_json": file_hash(audit_dir / "vcs_frozen_manifest.json"),
                "mixture_transport_frozen_manifest_json": file_hash(audit_dir / "mixture_transport_frozen_manifest.json"),
                "multicomponent_transport_frozen_manifest_json": file_hash(audit_dir / "multicomponent_transport_frozen_manifest.json"),
                "dusty_gas_transport_frozen_manifest_json": file_hash(audit_dir / "dusty_gas_transport_frozen_manifest.json"),
                "high_pressure_gas_transport_frozen_manifest_json": file_hash(audit_dir / "high_pressure_gas_transport_frozen_manifest.json"),
                "chem_equil_frozen_manifest_json": file_hash(audit_dir / "chem_equil_frozen_manifest.json"),
            },
        },
        "initial_symbol_counts": {
            "total": len(closure_symbols),
            "open": len(open_sids),
            "non_runtime_cpp_only": len(closure_symbols) - len(open_sids),
        },
    }

    snapshot_path = audit_dir / "cvodes_integrator_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"CVodesIntegrator selection and snapshot artifacts created:")
    print(f"  Selection: {selection_path}")
    print(f"  Snapshot:  {snapshot_path}")
    print(f"  Closure:   {closure_path} ({len(closure_symbols)} symbols, {len(open_sids)} open)")
    print(f"  Graph:     {call_graph_path}")

if __name__ == "__main__":
    main()
