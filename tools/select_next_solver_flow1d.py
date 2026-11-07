"""Select next solver family: Flow1D (1D reacting flow domains).

Generates:
  - audit/flow1d_closure_symbol_set.json
  - audit/flow1d_starting_snapshot.json
  - audit/flow1d_runtime_call_graph.json
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

    flow1d_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            source_file = rec.get("source_file", "")
            cname = rec.get("class", "")
            sid = rec.get("stable_symbol_id", "")

            if "Flow1D" in source_file or "Flow1D" in sid or "StFlow" in sid:
                method_name = rec.get("function_or_method", sid.split(":")[-1])
                target_cls = cname or "Flow1D"
                flow1d_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": source_file,
                    "native_class": target_cls,
                    "native_method": method_name,
                    "python_target": f"cantera_python.oned.{target_cls}.{method_name}",
                    "current_status": rec.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": rec.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": rec.get("callers"),
                    "callees": rec.get("callees"),
                    "remaining_validation_requirement": (
                        "Differential parity against pinned native Cantera Flow1D oracle across flow types "
                        "(unstrained-flow, free-flow, axisymmetric-flow), continuity, axial/radial momentum, "
                        "energy equation, species conservation, multicomponent/mixture-averaged diffusion, "
                        "Soret thermal diffusion, radiation models, and adaptive grid interpolation."
                    ),
                }

    closure_path = audit_dir / "flow1d_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(flow1d_symbols, f, indent=2)

    snapshot_path = audit_dir / "flow1d_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({
            "solver_family": "flow1d",
            "subsystem": "1D reacting flow domains",
            "symbol_count": len(flow1d_symbols),
            "symbols": flow1d_symbols,
        }, f, indent=2)

    call_graph_path = audit_dir / "flow1d_runtime_call_graph.json"
    call_graph = {
        "solver_family": "flow1d",
        "nodes": list(flow1d_symbols.keys()),
        "edges": [
            {"from": "Flow1D::eval", "to": "Flow1D::updateTransport"},
            {"from": "Flow1D::eval", "to": "Flow1D::updateDiffFluxes"},
            {"from": "Flow1D::eval", "to": "Flow1D::evalContinuity"},
            {"from": "Flow1D::eval", "to": "Flow1D::evalMomentum"},
            {"from": "Flow1D::eval", "to": "Flow1D::evalEnergy"},
            {"from": "Flow1D::eval", "to": "Flow1D::evalSpecies"},
            {"from": "Flow1D::setTransportModel", "to": "Flow1D::_setTransport"},
            {"from": "Flow1D::setupGrid", "to": "Flow1D::_on_grid_changed"},
        ]
    }
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    next_selection = {
        "selected_subsystem": "1D reacting flow domains",
        "selected_class": "Flow1D",
        "closure_symbol_count": len(flow1d_symbols),
        "closure_file": str(closure_path.relative_to(repo_root)),
        "starting_snapshot_file": str(snapshot_path.relative_to(repo_root)),
        "dependencies": ["Domain1D", "Boundary1D", "OneDim", "Sim1D", "Refiner", "MultiNewton"],
        "all_dependencies_validated": True,
    }
    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(next_selection, f, indent=2)

    print(f"Selected Flow1D with {len(flow1d_symbols)} canonical symbols.")


if __name__ == "__main__":
    main()
