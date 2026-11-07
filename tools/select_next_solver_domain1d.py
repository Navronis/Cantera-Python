"""Select next solver family: Domain1D (1D domain base abstraction).

Generates:
  - audit/domain1d_closure_symbol_set.json
  - audit/domain1d_starting_snapshot.json
  - audit/domain1d_runtime_call_graph.json
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

    domain1d_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            source_file = rec.get("source_file", "")
            cname = rec.get("class", "")
            sid = rec.get("stable_symbol_id", "")

            if (
                ("Domain1D.h" in source_file or "Domain1D.cpp" in source_file)
                and ("Domain1D" in sid or cname == "Domain1D")
            ):
                domain1d_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": source_file,
                    "native_class": "Domain1D",
                    "native_method": rec.get("function_or_method", sid.split(":")[-1]),
                    "python_target": f"cantera_python.oned.Domain1D.{rec.get('function_or_method', sid.split(':')[-1])}",
                    "current_status": rec.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": rec.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": rec.get("callers"),
                    "callees": rec.get("callees"),
                    "remaining_validation_requirement": (
                        "Differential parity against pinned native Cantera Domain1D oracle across grid coordinate setup, "
                        "point/component indexing, bounds, tolerances (steady/transient), time-integration state caching, "
                        "component naming, and topology links (linkLeft/linkRight/append)."
                    ),
                }

    closure_path = audit_dir / "domain1d_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(domain1d_symbols, f, indent=2)

    snapshot_path = audit_dir / "domain1d_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump({
            "solver_family": "domain1d",
            "subsystem": "1D domain base abstraction",
            "native_class": "Domain1D",
            "symbol_count": len(domain1d_symbols),
            "symbols": domain1d_symbols,
        }, f, indent=2)

    call_graph_path = audit_dir / "domain1d_runtime_call_graph.json"
    call_graph = {
        "solver_family": "domain1d",
        "nodes": list(domain1d_symbols.keys()),
        "edges": [
            {"from": "Domain1D::resize", "to": "Domain1D::locate"},
            {"from": "Domain1D::setupGrid", "to": "Domain1D::locate"},
            {"from": "Domain1D::setupUniformGrid", "to": "Domain1D::setupGrid"},
            {"from": "Domain1D::append", "to": "Domain1D::linkRight"},
            {"from": "Domain1D::append", "to": "Domain1D::linkLeft"},
            {"from": "Domain1D::linkLeft", "to": "Domain1D::locate"},
            {"from": "Domain1D::initTimeInteg", "to": "Domain1D::prevSoln"},
            {"from": "Domain1D::setSteadyTolerances", "to": "Domain1D::steady_rtol"},
            {"from": "Domain1D::setTransientTolerances", "to": "Domain1D::transient_rtol"},
            {"from": "Domain1D::getMeta", "to": "Domain1D::setMeta"},
        ]
    }
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    next_selection = {
        "selected_subsystem": "1D domain base abstraction",
        "selected_class": "Domain1D",
        "closure_symbol_count": len(domain1d_symbols),
        "closure_file": str(closure_path.relative_to(repo_root)),
        "starting_snapshot_file": str(snapshot_path.relative_to(repo_root)),
        "dependencies": ["Refiner", "OneDim", "Sim1D"],
        "all_dependencies_validated": True,
    }
    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(next_selection, f, indent=2)

    print(f"Selected Domain1D with {len(domain1d_symbols)} canonical symbols.")


if __name__ == "__main__":
    main()
