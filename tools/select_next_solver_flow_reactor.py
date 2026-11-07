"""Select FlowReactor as the next dependency-ready solver family and build its starting snapshot and closure set."""

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

    # 2. Extract FlowReactor canonical symbols from inventory
    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    flow_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if "FlowReactor.h" in row.get("source_file", "") and row.get("symbol_type") in ("method_or_function", "FUNCTION_OR_METHOD"):
                sid = row["stable_symbol_id"]
                method_name = sid.split(":")[-1]
                is_non_runtime = row.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
                flow_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": "FlowReactor",
                    "native_method": method_name,
                    "python_target": f"cantera_python.reactor.FlowReactor.{method_name}",
                    "current_status": row.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": "NONE_FINAL_DISPOSITION" if is_non_runtime else "Differential parity against pinned native Cantera FlowReactor across plug flow PFR systems with axial distance DAE integration"
                }

    with open(audit_dir / "flow_reactor_closure_symbol_set.json", "w", encoding="utf-8") as f:
        json.dump(flow_symbols, f, indent=2)
    print(f"Generated audit/flow_reactor_closure_symbol_set.json ({len(flow_symbols)} symbols).")

    open_runtime_count = sum(1 for s in flow_symbols.values() if s.get("current_status") != "NON_RUNTIME_CPP_ONLY_PROVEN")
    non_runtime_count = len(flow_symbols) - open_runtime_count

    # 1. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selected_solver_family": "flow_reactor",
        "subsystem": "flow reactor DAE",
        "native_class": "FlowReactor",
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "Following successful end-to-end differential closure and freezing of IdasIntegrator (DAE integration), "
            "FlowReactor is the exact and only domain in Cantera that utilizes IdasIntegrator for plug flow reactor "
            "differential-algebraic equations (Kee Chemically Reacting Flow equations: mass continuity, momentum, energy, "
            "and species conservation). All dependencies (CVodesIntegrator, ReactorNet, IdasIntegrator, ChemEquil, "
            "MultiTransport) are fully frozen and validated."
        ),
        "dependency_status": {
            "IdasIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "CVodesIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ReactorNet": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ChemEquil": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "MultiTransport": "VALIDATED_NUMERICALLY_EQUIVALENT"
        },
        "canonical_symbols_total": len(flow_symbols),
        "open_runtime_symbols": open_runtime_count,
        "non_runtime_proven_symbols": non_runtime_count,
        "starting_snapshot_path": "audit/flow_reactor_starting_snapshot.json",
        "closure_symbol_set_path": "audit/flow_reactor_closure_symbol_set.json",
        "runtime_call_graph_path": "audit/flow_reactor_runtime_call_graph.json"
    }

    with open(audit_dir / "next_solver_selection.json", "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print("Updated audit/next_solver_selection.json for FlowReactor.")

    # 3. Create audit/flow_reactor_runtime_call_graph.json
    call_graph = {
        "schema_version": 1,
        "solver_family": "flow_reactor",
        "native_class": "FlowReactor",
        "nodes": [
            "FlowReactor.__init__",
            "FlowReactor.type",
            "FlowReactor.isOde",
            "FlowReactor.timeIsIndependent",
            "FlowReactor.area",
            "FlowReactor.setArea",
            "FlowReactor.speed",
            "FlowReactor.distance",
            "FlowReactor.setMassFlowRate",
            "FlowReactor.massFlowRate",
            "FlowReactor.getStateDae",
            "FlowReactor.updateState",
            "FlowReactor.getConstraints",
            "FlowReactor.evalDae",
            "FlowReactor.componentIndex",
            "FlowReactor.componentName",
            "FlowReactor.initialize",
            "IdasIntegrator.initialize",
            "IdasIntegrator.integrate",
            "IdasIntegrator.step",
            "IdasIntegrator.solution"
        ],
        "edges": [
            {"caller": "ReactorNet.initialize", "callee": "FlowReactor.getStateDae"},
            {"caller": "ReactorNet.initialize", "callee": "FlowReactor.getConstraints"},
            {"caller": "ReactorNet.initialize", "callee": "IdasIntegrator.initialize"},
            {"caller": "ReactorNet.advance", "callee": "IdasIntegrator.integrate"},
            {"caller": "ReactorNet.step", "callee": "IdasIntegrator.step"},
            {"caller": "IdasIntegrator._dae_residual", "callee": "FlowReactor.evalDae"},
            {"caller": "FlowReactor.evalDae", "callee": "FlowReactor.updateState"}
        ]
    }
    with open(audit_dir / "flow_reactor_runtime_call_graph.json", "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print("Generated audit/flow_reactor_runtime_call_graph.json.")

    # 4. Create audit/flow_reactor_starting_snapshot.json
    snapshot = {
        "schema_version": 1,
        "artifact": "audit/flow_reactor_starting_snapshot.json",
        "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "is_worktree_clean": is_clean,
        "active_solver_family": "flow_reactor",
        "subsystem": "flow reactor DAE",
        "file_hashes": {
            "reactor_py": file_hash(repo_root / "cantera_python" / "reactor.py"),
            "numerics_py": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "idas_integrator_frozen_manifest_json": file_hash(audit_dir / "idas_integrator_frozen_manifest.json"),
            "reactor_network_frozen_manifest_json": file_hash(audit_dir / "reactor_network_frozen_manifest.json"),
            "pinned_FlowReactor_h": file_hash(orig_root / "include" / "cantera" / "zeroD" / "FlowReactor.h"),
            "pinned_FlowReactor_cpp": file_hash(orig_root / "src" / "zeroD" / "FlowReactor.cpp"),
        },
        "canonical_symbols_total": len(flow_symbols),
        "open_runtime_symbols": open_runtime_count,
        "non_runtime_proven_symbols": non_runtime_count,
        "target": "End-to-end plug flow PFR DAE simulation coupled to IdasIntegrator"
    }
    with open(audit_dir / "flow_reactor_starting_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print("Generated audit/flow_reactor_starting_snapshot.json.")


if __name__ == "__main__":
    main()
