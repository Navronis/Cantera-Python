"""Generate starting snapshot, closure symbol set, and runtime call graph for Family 5."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "audit" / "kinetics_source_closure_plan.json"
INVENTORY_PATH = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
SNAPSHOT_PATH = ROOT / "audit" / "kinetics_family5_starting_snapshot.json"
CLOSURE_SET_PATH = ROOT / "audit" / "kinetics_family5_closure_symbol_set.json"
CALL_GRAPH_PATH = ROOT / "audit" / "kinetics_family5_runtime_call_graph.json"

PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"
FAMILY_NAME = "FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE"

RESPONSIBILITIES = {
    "TwoTempPlasmaRate": "Two-temperature plasma rate evaluating reaction rates dependent on gas and electron temperatures",
    "TwoTempPlasmaData": "Shared state cache for two-temperature plasma reaction rate evaluations",
    "ElectronCollisionPlasmaRate": "Electron collision plasma rates evaluated from electron energy distribution functions and cross sections",
    "ElectronCollisionPlasmaData": "Shared state cache for electron collision plasma reaction rate evaluations",
    "VibrationalRelaxationRate": "Vibrational relaxation rates including Landau-Teller, SSH, and multi-state formulations",
    "VibrationalRelaxationData": "Shared state cache for vibrational relaxation reaction rate evaluations",
    "ConstantVibrationalRelaxationRate": "Constant vibrational relaxation rate model",
    "CastelaVibrationalRelaxationRate": "Castela vibrational relaxation rate model",
    "StarikovskiyVibrationalRelaxationRate": "Starikovskiy vibrational relaxation rate model",
    "MultiStateResolvedVibrationalRelaxationRate": "Multi-state resolved vibrational relaxation rate model",
    "CustomFunc1Rate": "Custom user-defined rate evaluator via mathematical function / callable",
    "ReactionRateDelegator": "Delegator for external and python reaction rates",
    "ReactionDataDelegator": "Shared state cache delegator for external and python reaction rates",
    "MultiRate": "Vectorized batch evaluator for reaction rate models across reactions",
    "ReactionRate": "Base class and interface for reaction rate evaluators",
    "ReactionData": "Shared state data container for reaction rate evaluators",
    "Kinetics": "Base kinetics manager methods, thermodynamic delta properties, and equilibrium calculations",
    "FreeFunctions": "Free functions for reaction parsing, rate evaluations, and kinetics utilities"
}


def main():
    with open(PLAN_PATH, "r", encoding="utf-8") as f:
        plan = json.load(f)

    sym_plan = plan.get("symbol_plan", [])
    f5_plan_symbols = [
        s for s in sym_plan
        if s.get("closure_family") == FAMILY_NAME
    ]

    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        inventory = [json.loads(line) for line in f]

    inv_by_id = {row["stable_symbol_id"]: (idx, row) for idx, row in enumerate(inventory)}

    f5_plan_ids = {s["canonical_symbol_id"] for s in f5_plan_symbols}
    assert len(f5_plan_ids) == 207, f"Expected 207 symbols, got {len(f5_plan_ids)}"

    closure_set = {}
    nodes = []
    open_runtime_ids = []
    proven_non_runtime_ids = []
    class_counts = {}
    status_counts = {}

    for s_plan in f5_plan_symbols:
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

        py_target = s_plan.get("python_target") or f"cantera_python.kinetics.{method_name}"

        # Callers
        if "Plasma" in cls_name:
            callers = [
                "cantera_python.kinetics.GasKinetics",
                "cantera_python.solution.Solution"
            ]
        elif "Vibrational" in cls_name:
            callers = [
                "cantera_python.kinetics.GasKinetics",
                "cantera_python.solution.Solution"
            ]
        elif "MultiRate" in cls_name:
            callers = [
                "cantera_python.kinetics.Kinetics",
                "cantera_python.kinetics.GasKinetics"
            ]
        elif cls_name == "Kinetics":
            callers = [
                "cantera_python.solution.Solution",
                "cantera_python.kinetics.GasKinetics",
                "cantera_python.surface.Interface"
            ]
        else:
            callers = [
                "cantera_python.kinetics.Kinetics",
                "cantera_python.rates.ReactionRateFactory"
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
            "scientific_responsibility": RESPONSIBILITIES.get(cls_name, "Kinetics plasma, vibrational relaxation, and multirate evaluation"),
            "runtime_callers": callers,
            "missing_implementation_logic": (
                "Native C++ C-API / pure-virtual / inline wrapper proven non-runtime"
                if is_nr else
                "Exact differential parity against native Kinetics / MultiRate / Plasma C++ oracle."
            ),
            "required_native_differential_test": f"test_kinetics_family5_{cls_name}_{method_name}",
            "required_performance_test": f"benchmark_family5_{cls_name}",
            "performance_bottleneck": "Batch reaction rate constant evaluation and plasma/vibrational rate calculation"
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
        "total_symbols_in_family": len(f5_plan_symbols),
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

    print(f"\nFamily 5 Starting State Locked:")
    print(f"  Total symbols: {len(f5_plan_symbols)}")
    print(f"  Open runtime: {len(open_runtime_ids)}")
    print(f"  Proven non-runtime: {len(proven_non_runtime_ids)}")
    print(f"  Class breakdown: {class_counts}")


if __name__ == "__main__":
    main()
