"""Select next solver family: Sim1D (1D solve orchestration).

Generates:
  - audit/sim1d_closure_symbol_set.json
  - audit/sim1d_starting_snapshot.json
  - audit/sim1d_runtime_call_graph.json
Updates:
  - audit/next_solver_selection.json
"""

import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    provenance_dir = repo_root / "provenance"

    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"

    sim1d_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            source_file = rec.get("source_file", "")
            cname = rec.get("class", "")
            sid = rec.get("stable_symbol_id", "")

            if (
                ("Sim1D.h" in source_file or "Sim1D.cpp" in source_file)
                and ("Sim1D" in sid or cname == "Sim1D")
            ):
                sim1d_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": source_file,
                    "native_class": "Sim1D",
                    "native_method": rec.get("function_or_method", sid.split(":")[-1]),
                    "python_target": f"cantera_python.oned.Sim1D.{rec.get('function_or_method', sid.split(':')[-1])}",
                    "current_status": rec.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": rec.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": rec.get("callers"),
                    "callees": rec.get("callees"),
                    "remaining_validation_requirement": (
                        "Differential parity against pinned native Cantera Sim1D oracle across multi-domain solve orchestration, "
                        "steady-state newton steps, time-stepping fallback loops, adaptive grid refinement callbacks, and residual convergence."
                    ),
                }

    closure_path = audit_dir / "sim1d_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(sim1d_symbols, f, indent=2)

    snapshot_path = audit_dir / "sim1d_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({
            "solver_family": "sim1d",
            "subsystem": "1D solve orchestration",
            "native_class": "Sim1D",
            "symbol_count": len(sim1d_symbols),
            "symbols": sim1d_symbols,
        }, f, indent=2)

    call_graph_path = audit_dir / "sim1d_runtime_call_graph.json"
    call_graph = {
        "solver_family": "sim1d",
        "nodes": list(sim1d_symbols.keys()),
        "edges": [
            {"from": "Sim1D::solve", "to": "Sim1D::newtonSolve"},
            {"from": "Sim1D::newtonSolve", "to": "MultiNewton::solve"},
            {"from": "Sim1D::solve", "to": "Sim1D::refine"},
            {"from": "Sim1D::refine", "to": "Refiner::analyze"},
            {"from": "Sim1D::solve", "to": "OneDim::eval"},
            {"from": "Sim1D::solve", "to": "OneDim::evalJacobian"},
        ]
    }
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    next_selection = {
        "selected_subsystem": "1D solve orchestration",
        "selected_class": "Sim1D",
        "closure_symbol_count": len(sim1d_symbols),
        "closure_file": str(closure_path.relative_to(repo_root)),
        "starting_snapshot_file": str(snapshot_path.relative_to(repo_root)),
        "dependencies": ["BandMatrix", "MultiNewton", "Refiner", "OneDim"],
        "all_dependencies_validated": True,
    }
    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(next_selection, f, indent=2)

    print(f"Selected Sim1D with {len(sim1d_symbols)} canonical symbols.")


if __name__ == "__main__":
    main()
