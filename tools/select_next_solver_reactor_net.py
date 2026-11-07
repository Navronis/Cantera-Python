"""Select ReactorNet (reactor network) as next solver family and build initial audit artifacts."""

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
            if cls == "ReactorNet":
                sid = row["stable_symbol_id"]
                current_status = "NON_RUNTIME_CPP_ONLY_PROVEN" if sid in rejected_sids else row["current_status"]
                closure_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": cls,
                    "native_method": row["function_or_method"],
                    "python_target": f"cantera_python.reactor.ReactorNet.{row['function_or_method']}",
                    "current_status": current_status,
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": "Differential parity against pinned native Cantera ReactorNet across homogeneous reactors, network coupling, flow devices, walls, advance limits, and sensitivities",
                }
                if current_status not in ("PINNED_NATIVE_VALIDATED", "NON_RUNTIME_CPP_ONLY_PROVEN"):
                    open_sids.append(sid)

    closure_path = audit_dir / "reactor_network_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_symbols, f, indent=2)

    # 2. Build runtime call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "reactor_network",
        "entry_points": [
            "cantera_python.reactor.ReactorNet.__init__",
            "cantera_python.reactor.ReactorNet.initialize",
            "cantera_python.reactor.ReactorNet.reinitialize",
            "cantera_python.reactor.ReactorNet.advance",
            "cantera_python.reactor.ReactorNet.step",
            "cantera_python.reactor.ReactorNet.time",
            "cantera_python.reactor.ReactorNet.distance",
            "cantera_python.reactor.ReactorNet.setInitialTime",
            "cantera_python.reactor.ReactorNet.getInitialTime",
            "cantera_python.reactor.ReactorNet.setRelativeTolerance",
            "cantera_python.reactor.ReactorNet.setAbsoluteTolerance",
            "cantera_python.reactor.ReactorNet.clearAbsoluteTolerance",
            "cantera_python.reactor.ReactorNet.setTolerances",
            "cantera_python.reactor.ReactorNet.setSensitivityTolerances",
            "cantera_python.reactor.ReactorNet.setMaxTimeStep",
            "cantera_python.reactor.ReactorNet.setMaxSteps",
            "cantera_python.reactor.ReactorNet.maxSteps",
            "cantera_python.reactor.ReactorNet.setMaxErrTestFails",
            "cantera_python.reactor.ReactorNet.setLinearSolverType",
            "cantera_python.reactor.ReactorNet.linearSolverType",
            "cantera_python.reactor.ReactorNet.setPreconditioner",
            "cantera_python.reactor.ReactorNet.checkPreconditionerSupported",
            "cantera_python.reactor.ReactorNet.neq",
            "cantera_python.reactor.ReactorNet.eval",
            "cantera_python.reactor.ReactorNet.evalDae",
            "cantera_python.reactor.ReactorNet.evalSteady",
            "cantera_python.reactor.ReactorNet.getState",
            "cantera_python.reactor.ReactorNet.getStateDae",
            "cantera_python.reactor.ReactorNet.updateState",
            "cantera_python.reactor.ReactorNet.getDerivative",
            "cantera_python.reactor.ReactorNet.getConstraints",
            "cantera_python.reactor.ReactorNet.nparams",
            "cantera_python.reactor.ReactorNet.registerSensitivityParameter",
            "cantera_python.reactor.ReactorNet.sensitivityParameterName",
            "cantera_python.reactor.ReactorNet.sensitivity",
            "cantera_python.reactor.ReactorNet.solveSteady",
            "cantera_python.reactor.ReactorNet.steadyJacobian",
            "cantera_python.reactor.ReactorNet.jacobian",
            "cantera_python.reactor.ReactorNet.finiteDifferenceJacobian",
            "cantera_python.reactor.ReactorNet.solverStats",
            "cantera_python.reactor.ReactorNet.setAdvanceLimits",
            "cantera_python.reactor.ReactorNet.hasAdvanceLimits",
            "cantera_python.reactor.ReactorNet.getAdvanceLimits",
        ],
        "nodes": {
            "cantera_python.reactor.ReactorNet.initialize": {
                "callees": [
                    "cantera_python.reactor.ReactorNet.updateTolerances",
                    "cantera_python.numerics.CVodesIntegrator.initialize",
                    "cantera_python.reactor.ReactorBase.updateConnected",
                ],
                "description": "Initialize reactor network, connect reactors, allocate state vectors, and initialize CVodesIntegrator",
            },
            "cantera_python.reactor.ReactorNet.reinitialize": {
                "callees": [
                    "cantera_python.reactor.ReactorNet.updateTolerances",
                    "cantera_python.numerics.CVodesIntegrator.reinitialize",
                ],
                "description": "Reinitialize CVodesIntegrator without resetting network configuration",
            },
            "cantera_python.reactor.ReactorNet.advance": {
                "callees": [
                    "cantera_python.reactor.ReactorNet.initialize",
                    "cantera_python.numerics.CVodesIntegrator.integrate",
                    "cantera_python.reactor.ReactorNet.updateState",
                ],
                "description": "Advance network state to specified target time using CVodesIntegrator with optional advance limit enforcement",
            },
            "cantera_python.reactor.ReactorNet.step": {
                "callees": [
                    "cantera_python.reactor.ReactorNet.initialize",
                    "cantera_python.numerics.CVodesIntegrator.step",
                    "cantera_python.reactor.ReactorNet.updateState",
                ],
                "description": "Take one internal variable-order BDF step toward time + 1.0",
            },
            "cantera_python.reactor.ReactorNet.eval": {
                "callees": [
                    "cantera_python.reactor.ReactorNet.updateState",
                    "cantera_python.reactor.ReactorBase.evalRhs",
                    "cantera_python.reactor.FlowDevice.eval",
                    "cantera_python.reactor.Wall.eval",
                ],
                "description": "FuncEval ODE right-hand side evaluation for all connected reactors and devices",
            },
            "cantera_python.reactor.ReactorNet.updateState": {
                "callees": [
                    "cantera_python.reactor.ReactorBase.updateState",
                ],
                "description": "Scatter solution vector y to individual reactor states (mass, volume, species, energy)",
            },
            "cantera_python.reactor.ReactorNet.sensitivity": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator.sensitivity",
                ],
                "description": "Extract forward sensitivity d(y_k)/d(p_p) from CVodesIntegrator",
            },
        },
    }

    call_graph_path = audit_dir / "reactor_network_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    # 3. Create next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "native_oracle_pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "selected_solver_family": "reactor_network",
        "native_class": "ReactorNet",
        "native_file": "src/zeroD/ReactorNet.cpp",
        "subsystem": "reactor network",
        "current_family_status": "PARTIAL",
        "target_family_status": "VALIDATED_NUMERICALLY_EQUIVALENT",
        "rationale": (
            "All foundational solvers - transport (mixture, multicomponent, dusty_gas, high_pressure), "
            "equilibrium (VCS multiphase, ChemEquil single-phase), and the foundational 0D ODE integrator "
            "(CVodesIntegrator) - are 100% validated against pinned native Cantera (726522be4e2a13454d8415b7ef799d621f665cf3) "
            "and frozen. ReactorNet is the primary 0D chemical kinetics and reactor orchestration solver in Cantera. "
            "It couples reactors, walls, and flow devices via FuncEval and delegates ODE integration directly to CVodesIntegrator."
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
            "cvodes_integrator",
        ],
        "open_symbol_burn_down_target": len(open_sids),
        "total_closure_symbols": len(closure_symbols),
        "target_closure_subgraph": open_sids,
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    # 4. Create reactor_network_starting_snapshot.json
    snapshot = {
        "schema_version": 1,
        "artifact": "audit/reactor_network_starting_snapshot.json",
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
            "reactor_py": file_hash(repo_root / "cantera_python" / "reactor.py"),
            "numerics_py": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "test_reactor_network_py": file_hash(repo_root / "tests" / "test_reactor_network.py"),
            "test_reactor_conservation_py": file_hash(repo_root / "tests" / "test_reactor_conservation.py"),
            "reactor_net_closure_set_hash": file_hash(closure_path),
            "reactor_net_call_graph_hash": file_hash(call_graph_path),
            "pinned_ReactorNet_h": file_hash(orig_root / "include" / "cantera" / "zeroD" / "ReactorNet.h"),
            "pinned_ReactorNet_cpp": file_hash(orig_root / "src" / "zeroD" / "ReactorNet.cpp"),
            "frozen_dependencies": {
                "vcs_frozen_manifest_json": file_hash(audit_dir / "vcs_frozen_manifest.json"),
                "mixture_transport_frozen_manifest_json": file_hash(audit_dir / "mixture_transport_frozen_manifest.json"),
                "multicomponent_transport_frozen_manifest_json": file_hash(audit_dir / "multicomponent_transport_frozen_manifest.json"),
                "dusty_gas_transport_frozen_manifest_json": file_hash(audit_dir / "dusty_gas_transport_frozen_manifest.json"),
                "high_pressure_gas_transport_frozen_manifest_json": file_hash(audit_dir / "high_pressure_gas_transport_frozen_manifest.json"),
                "chem_equil_frozen_manifest_json": file_hash(audit_dir / "chem_equil_frozen_manifest.json"),
                "cvodes_integrator_frozen_manifest_json": file_hash(audit_dir / "cvodes_integrator_frozen_manifest.json"),
            },
        },
        "initial_symbol_counts": {
            "total": len(closure_symbols),
            "open": len(open_sids),
            "non_runtime_cpp_only": len(closure_symbols) - len(open_sids),
        },
    }

    snapshot_path = audit_dir / "reactor_network_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"ReactorNet selection and starting snapshot artifacts created:")
    print(f"  Selection: {selection_path}")
    print(f"  Snapshot:  {snapshot_path}")
    print(f"  Closure:   {closure_path} ({len(closure_symbols)} symbols, {len(open_sids)} open)")
    print(f"  Graph:     {call_graph_path}")

if __name__ == "__main__":
    main()
