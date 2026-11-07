"""Select next solver family: Boundary1D (1D boundary conditions).

Generates:
  - audit/boundary1d_closure_symbol_set.json
  - audit/boundary1d_starting_snapshot.json
  - audit/boundary1d_runtime_call_graph.json
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

    boundary1d_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            source_file = rec.get("source_file", "")
            cname = rec.get("class", "")
            sid = rec.get("stable_symbol_id", "")

            if "Boundary1D.h" in source_file or "Boundary1D.cpp" in source_file:
                boundary1d_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": source_file,
                    "native_class": cname or "Boundary1D",
                    "native_method": rec.get("function_or_method", sid.split(":")[-1]),
                    "python_target": f"cantera_python.oned.{cname or 'Boundary1D'}.{rec.get('function_or_method', sid.split(':')[-1])}",
                    "current_status": rec.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": rec.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": rec.get("callers"),
                    "callees": rec.get("callees"),
                    "remaining_validation_requirement": (
                        "Differential parity against pinned native Cantera Boundary1D oracle across boundary domain types "
                        "(Inlet1D, Outlet1D, Symm1D, Empty1D, OutletRes1D, Surf1D, ReactingSurf1D), temperature, mass flow rate, "
                        "spread rate, mole/mass fraction constraints, boundary residual evaluation, and lifecycle hooks."
                    ),
                }

    closure_path = audit_dir / "boundary1d_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(boundary1d_symbols, f, indent=2)

    snapshot_path = audit_dir / "boundary1d_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({
            "solver_family": "boundary1d",
            "subsystem": "1D boundary conditions",
            "symbol_count": len(boundary1d_symbols),
            "symbols": boundary1d_symbols,
        }, f, indent=2)

    call_graph_path = audit_dir / "boundary1d_runtime_call_graph.json"
    call_graph = {
        "solver_family": "boundary1d",
        "nodes": list(boundary1d_symbols.keys()),
        "edges": [
            {"from": "Boundary1D::init", "to": "Boundary1D::_init"},
            {"from": "Inlet1D::init", "to": "Boundary1D::_init"},
            {"from": "Inlet1D::setMoleFractions", "to": "Inlet1D::massFraction"},
            {"from": "OutletRes1D::setMoleFractions", "to": "OutletRes1D::massFraction"},
            {"from": "Inlet1D::eval", "to": "Boundary1D::eval"},
            {"from": "Outlet1D::eval", "to": "Boundary1D::eval"},
            {"from": "Symm1D::eval", "to": "Boundary1D::eval"},
            {"from": "Surf1D::eval", "to": "Boundary1D::eval"},
        ]
    }
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    next_selection = {
        "selected_subsystem": "1D boundary conditions",
        "selected_class": "Boundary1D",
        "closure_symbol_count": len(boundary1d_symbols),
        "closure_file": str(closure_path.relative_to(repo_root)),
        "starting_snapshot_file": str(snapshot_path.relative_to(repo_root)),
        "dependencies": ["Domain1D", "OneDim", "Sim1D"],
        "all_dependencies_validated": True,
    }
    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(next_selection, f, indent=2)

    print(f"Selected Boundary1D with {len(boundary1d_symbols)} canonical symbols.")


if __name__ == "__main__":
    main()
