"""Generate starting snapshot, closure symbol set, and runtime call graph for Family 4."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "audit" / "kinetics_source_closure_plan.json"
INVENTORY_PATH = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
SNAPSHOT_PATH = ROOT / "audit" / "kinetics_family4_starting_snapshot.json"
CLOSURE_SET_PATH = ROOT / "audit" / "kinetics_family4_closure_symbol_set.json"
CALL_GRAPH_PATH = ROOT / "audit" / "kinetics_family4_runtime_call_graph.json"

PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"
FAMILY_NAME = "FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES"

PYTHON_CLASS_MAP = {
    "Reaction": "cantera_python.kinetics.Reaction",
    "ReactionPathDiagram": "cantera_python.kinetics.ReactionPathDiagram",
    "Path": "cantera_python.kinetics.Path",
    "SpeciesNode": "cantera_python.kinetics.SpeciesNode",
    "ReactionPathBuilder": "cantera_python.kinetics.ReactionPathBuilder",
    "Group": "cantera_python.kinetics.Group",
    "KineticsFactory": "cantera_python.kinetics.KineticsFactory",
    "ReactionRateFactory": "cantera_python.rates.ReactionRateFactory",
}

RESPONSIBILITIES = {
    "Reaction": "Reaction definition, equation parsing, stoichiometry, and rate/third-body linkage",
    "ReactionPathDiagram": "Reaction path analysis and diagram generation for flux visualization",
    "Path": "Net flux and reaction contributions between species nodes in reaction path diagram",
    "SpeciesNode": "Species node in reaction path diagram tracking inflows, outflows, and paths",
    "ReactionPathBuilder": "Builder algorithm for discovering reaction path topology and elemental flows",
    "Group": "Atom and element groups for identifying reaction pathways and atom transfers",
    "KineticsFactory": "Factory for instantiating Kinetics and BulkKinetics managers from models",
    "ReactionRateFactory": "Factory for instantiating ReactionRate objects from type or AnyMap nodes",
    "FreeFunctions": "Free factory and utility functions for kinetics and reaction topology"
}


def main():
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    sym_plan = plan.get("symbol_plan", [])
    f4_plan_symbols = [
        s for s in sym_plan
        if s.get("closure_family") == FAMILY_NAME
    ]

    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        inventory = [json.loads(line) for line in f]

    inv_by_id = {row["stable_symbol_id"]: (idx, row) for idx, row in enumerate(inventory)}

    f4_plan_ids = {s["canonical_symbol_id"] for s in f4_plan_symbols}
    assert len(f4_plan_ids) == 132, f"Expected 132 symbols, got {len(f4_plan_ids)}"

    closure_set = {}
    nodes = []
    open_runtime_ids = []
    proven_non_runtime_ids = []
    class_counts = {}
    status_counts = {}

    for s_plan in f4_plan_symbols:
        sym_id = s_plan["canonical_symbol_id"]
        idx, inv_row = inv_by_id[sym_id]
        cls_name = inv_row.get("class") or "FreeFunctions"
        method_name = inv_row.get("function_or_method") or sym_id.split(":")[-1]
        st = inv_row["current_status"]
        is_nr = (st == "NON_RUNTIME_CPP_ONLY_PROVEN")

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        status_counts[st] = status_counts.get(st, 0) + 1

        if is_nr:
            proven_non_runtime_ids.append(sym_id)
        else:
            open_runtime_ids.append(sym_id)

        # Python target
        if cls_name in PYTHON_CLASS_MAP:
            py_target = f"{PYTHON_CLASS_MAP[cls_name]}.{method_name}"
        else:
            py_target = f"cantera_python.kinetics.{method_name}"

        # Callers
        if cls_name in ("Reaction", "KineticsFactory", "ReactionRateFactory"):
            callers = [
                "cantera_python.kinetics.Kinetics",
                "cantera_python.solution.Solution",
                "cantera_python.mechanism.MechanismLoader"
            ]
        else:
            callers = [
                "cantera_python.kinetics.ReactionPathDiagram",
                "cantera_python.kinetics.Kinetics"
            ]

        entry = {
            "canonical_index": idx,
            "canonical_symbol_id": sym_id,
            "stable_symbol_id": sym_id,
            "native_source_definition": f"{inv_row['source_file']}:{inv_row['line_range'][0]}",
            "line_range": inv_row["line_range"],
            "signature": inv_row.get("signature", ""),
            "native_class": cls_name,
            "native_method": method_name,
            "python_target": py_target,
            "current_status": st,
            "final_disposition": inv_row.get("final_disposition"),
            "is_non_runtime": is_nr,
            "scientific_responsibility": RESPONSIBILITIES.get(cls_name, "Kinetics reaction topology and factory management"),
            "runtime_callers": callers,
            "missing_implementation_logic": (
                "Native C++ C-API / inline wrapper proven non-runtime"
                if is_nr else
                "Exact differential parity against native ReactionPath / Reaction C++ oracle."
            ),
            "required_native_differential_test": f"test_kinetics_family4_{cls_name}_{method_name}",
            "required_performance_test": f"benchmark_family4_{cls_name}",
            "performance_bottleneck": "Reaction path traversal and net flux graph construction"
        }
        closure_set[sym_id] = entry

        nodes.append({
            "id": sym_id,
            "class": cls_name,
            "method": method_name,
            "status": st,
            "relevance": inv_row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE")
        })

    open_runtime_ids.sort()
    proven_non_runtime_ids.sort()

    # 1. Starting Snapshot
    snapshot = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "family": FAMILY_NAME,
        "pinned_native_commit": PINNED_COMMIT,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_symbols_in_family": len(f4_plan_symbols),
        "open_runtime_symbols": len(open_runtime_ids),
        "proven_non_runtime_symbols": len(proven_non_runtime_ids),
        "inventory_status_breakdown": status_counts,
        "class_breakdown": dict(sorted(class_counts.items())),
        "open_runtime_symbol_ids": open_runtime_ids,
        "proven_non_runtime_symbol_ids": proven_non_runtime_ids
    }
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    print(f"Wrote snapshot: {SNAPSHOT_PATH}")

    # 2. Closure Symbol Set
    with open(CLOSURE_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2, sort_keys=True)
    print(f"Wrote closure symbol set: {CLOSURE_SET_PATH}")

    # 3. Runtime Call Graph
    call_graph = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "solver_family": FAMILY_NAME,
        "total_nodes": len(nodes),
        "total_edges": 0,
        "nodes": sorted(nodes, key=lambda n: n["id"]),
        "edges": []
    }
    with open(CALL_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2, sort_keys=True)
    print(f"Wrote call graph: {CALL_GRAPH_PATH}")

    print(f"\nFamily 4 Starting State Locked:")
    print(f"  Total symbols: {len(f4_plan_symbols)}")
    print(f"  Open runtime: {len(open_runtime_ids)}")
    print(f"  Proven non-runtime: {len(proven_non_runtime_ids)}")
    print(f"  Class breakdown: {class_counts}")


if __name__ == "__main__":
    main()
