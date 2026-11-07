#!/usr/bin/env python3
"""Build Family 2 Phase 1 Artifacts:
- audit/kinetics_family2_closure_symbol_set.json
- audit/kinetics_family2_runtime_call_graph.json
- audit/kinetics_family2_starting_snapshot.json
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    with open(ROOT / "audit" / "kinetics_source_closure_plan.json", "r", encoding="utf-8") as f:
        plan = json.load(f)

    with open(ROOT / "audit" / "kinetics_complete_runtime_call_graph.json", "r", encoding="utf-8") as f:
        full_call_graph = json.load(f)

    inv_by_id = {}
    with open(ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            inv_by_id[row["stable_symbol_id"]] = row

    fam2_symbols = [s for s in plan["symbol_plan"] if s.get("closure_family") == "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT"]
    print(f"Total Family 2 symbols found in closure plan: {len(fam2_symbols)}")

    # 1. Build kinetics_family2_closure_symbol_set.json
    closure_set = {}
    non_runtime_symbols = []
    open_runtime_symbols = []

    for s in fam2_symbols:
        cid = s["canonical_symbol_id"]
        inv_row = inv_by_id.get(cid, {})

        is_non_runtime = (s.get("final_disposition") == "NON_RUNTIME_CPP_ONLY_PROVEN" or
                          inv_row.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN" or
                          inv_row.get("final_disposition") == "NON_RUNTIME_CPP_ONLY_PROVEN")

        # Determine scientific responsibility
        cls_name = s.get("class") or ""
        method_name = s.get("method") or ""
        if "Falloff" in cls_name or "Troe" in cls_name or "Sri" in cls_name or "Tsang" in cls_name or "Lindemann" in cls_name:
            resp = "Falloff rate coefficient calculation with reduced pressure and collider broadening"
        elif "Plog" in cls_name:
            resp = "Pressure-dependent logarithmic interpolation of Arrhenius rate expressions"
        elif "Chebyshev" in cls_name:
            resp = "Two-dimensional Chebyshev polynomial rate evaluation in temperature and pressure"
        elif "BlowersMasel" in cls_name:
            resp = "Reaction enthalpy-dependent activation energy Blowers-Masel rate calculation"
        elif "LinearBurke" in cls_name:
            resp = "Linear Burke mixture-averaged collider and third-body rate calculation"
        else:
            resp = "Pressure-dependent reaction rate and falloff helper function"

        entry = {
            "stable_symbol_id": cid,
            "canonical_symbol_id": cid,
            "canonical_index": inv_row.get("canonical_index"),
            "native_source_definition": s.get("native_source_definition"),
            "native_class": cls_name,
            "native_method": method_name,
            "signature": s.get("signature") or inv_row.get("signature"),
            "line_range": inv_row.get("line_range"),
            "python_target": s.get("python_target"),
            "current_status": inv_row.get("current_status", s.get("current_implementation_status")),
            "final_disposition": "NON_RUNTIME_CPP_ONLY_PROVEN" if is_non_runtime else inv_row.get("final_disposition"),
            "scientific_responsibility": resp,
            "runtime_callers": ["cantera_python.kinetics.BulkKinetics", "cantera_python.kinetics.GasKinetics", "cantera_python.solution.Solution"],
            "required_native_differential_test": f"test_kinetics_family2_{cls_name}_{method_name}",
            "required_performance_test": f"benchmark_family2_{cls_name}",
            "missing_implementation_logic": s.get("missing_scientific_logic", "Source-exact numerical parity against pinned native Cantera C++ oracle"),
            "performance_bottleneck": s.get("performance_bottleneck", "Vectorized evaluation across temperature and pressure arrays"),
            "is_non_runtime": is_non_runtime
        }
        closure_set[cid] = entry
        if is_non_runtime:
            non_runtime_symbols.append(cid)
        else:
            open_runtime_symbols.append(cid)

    with open(ROOT / "audit" / "kinetics_family2_closure_symbol_set.json", "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2, sort_keys=True)
    print(f"Wrote audit/kinetics_family2_closure_symbol_set.json ({len(closure_set)} symbols: {len(open_runtime_symbols)} open runtime, {len(non_runtime_symbols)} non-runtime)")

    # 2. Build kinetics_family2_runtime_call_graph.json
    f2_cids = set(closure_set.keys())
    sub_nodes = [node for node in full_call_graph.get("nodes", []) if node.get("id") in f2_cids]
    sub_edges = [edge for edge in full_call_graph.get("edges", []) if edge.get("source") in f2_cids or edge.get("target") in f2_cids]

    call_graph = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "solver_family": "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT",
        "total_nodes": len(sub_nodes),
        "total_edges": len(sub_edges),
        "nodes": sub_nodes,
        "edges": sub_edges
    }
    with open(ROOT / "audit" / "kinetics_family2_runtime_call_graph.json", "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2, sort_keys=True)
    print(f"Wrote audit/kinetics_family2_runtime_call_graph.json ({len(sub_nodes)} nodes, {len(sub_edges)} edges)")

    # 3. Build kinetics_family2_starting_snapshot.json
    snapshot = {
        "schema_version": "1.0.0",
        "timestamp": "2026-11-06T09:25:00Z",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "subsystem": "kinetics",
        "family": "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT",
        "total_symbols_in_family": len(fam2_symbols),
        "open_runtime_symbols": len(open_runtime_symbols),
        "proven_non_runtime_symbols": len(non_runtime_symbols),
        "class_breakdown": {
            str(cls) if cls is not None else "FreeFunctions": len([s for s in fam2_symbols if s.get("class") == cls])
            for cls in sorted(set(s.get("class") for s in fam2_symbols), key=lambda x: str(x))
        },
        "inventory_status_breakdown": {
            str(st): len([s for s in fam2_symbols if inv_by_id.get(s["canonical_symbol_id"], {}).get("current_status") == st])
            for st in sorted(set(inv_by_id.get(s["canonical_symbol_id"], {}).get("current_status") for s in fam2_symbols), key=lambda x: str(x))
        },
        "open_runtime_symbol_ids": sorted(open_runtime_symbols),
        "proven_non_runtime_symbol_ids": sorted(non_runtime_symbols)
    }
    with open(ROOT / "audit" / "kinetics_family2_starting_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    print(f"Wrote audit/kinetics_family2_starting_snapshot.json")

if __name__ == "__main__":
    main()
