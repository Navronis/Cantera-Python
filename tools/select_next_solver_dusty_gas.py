"""Update next solver selection and build closure symbol set for DustyGasTransport."""

from datetime import datetime, timezone
import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    provenance_dir = repo_root / "provenance"

    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    dusty_symbols = []
    with open(inventory_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "DustyGasTransport" in rec["source_file"]:
                    dusty_symbols.append(rec)

    # 1. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "native_oracle_pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "selected_solver_family": "dusty_gas_transport",
        "native_class": "DustyGasTransport",
        "native_file": "src/transport/DustyGasTransport.cpp",
        "subsystem": "porous transport",
        "current_family_status": "PARTIAL",
        "target_family_status": "VALIDATED_NUMERICALLY_EQUIVALENT",
        "rationale": (
            "MultiTransport (multicomponent transport) is now 100% closed, validated against pinned native Cantera "
            "(726522be4e2a13454d8415b7ef799d621f665cf3) across all 65 test cases (229,720 matrix elements), "
            "and frozen. DustyGasTransport (porous transport) is the direct dependency descendant of MultiTransport "
            "(embedding MultiTransport as its underlying gas transport provider for binary diffusion and viscosity). "
            "It has 17 PARTIAL open symbols ready for full end-to-end native validation."
        ),
        "prerequisite_dependencies_satisfied": [
            "thermodynamic_state_inversion",
            "ideal_gas_thermo",
            "pure_fluid_properties",
            "vcs_multiphase_equilibrium",
            "mixture_transport",
            "multicomponent_transport"
        ],
        "open_symbol_burn_down_target": 17,
        "target_closure_subgraph": [r["stable_symbol_id"] for r in dusty_symbols if r["current_status"] == "PARTIAL"],
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Updated {selection_path}")

    # 2. Build audit/dusty_gas_transport_closure_symbol_set.json
    closure_set = {}
    for r in dusty_symbols:
        closure_set[r["stable_symbol_id"]] = {
            "stable_symbol_id": r["stable_symbol_id"],
            "native_file": r["source_file"],
            "native_class": r.get("class", "DustyGasTransport"),
            "native_method": r["function_or_method"],
            "python_target": r.get("python_mapping") or f"cantera_python.transport.DustyGasTransport.{r['function_or_method']}",
            "current_status": r["current_status"],
            "scientific_runtime_relevance": r.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
            "callers": r.get("callers", []),
            "callees": r.get("callees", []),
            "remaining_validation_requirement": "Differential parity against pinned native Cantera DustyGasTransport",
        }

    closure_path = audit_dir / "dusty_gas_transport_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)
    print(f"Created {closure_path} with {len(closure_set)} symbols")

    # 3. Build audit/dusty_gas_transport_runtime_call_graph.json
    call_graph = {
        "schema_version": 1,
        "entry_point": "DustyGasTransport::getMultiDiffCoeffs and getMolarFluxes",
        "nodes": closure_set,
        "edges": [
            {"from": "include/cantera/transport/DustyGasTransport.h:64:getMultiDiffCoeffs", "to": "include/cantera/transport/DustyGasTransport.h:175:updateMultiDiffCoeffs"},
            {"from": "include/cantera/transport/DustyGasTransport.h:175:updateMultiDiffCoeffs", "to": "include/cantera/transport/DustyGasTransport.h:145:updateTransport_T"},
            {"from": "include/cantera/transport/DustyGasTransport.h:175:updateMultiDiffCoeffs", "to": "include/cantera/transport/DustyGasTransport.h:153:updateTransport_C"},
            {"from": "include/cantera/transport/DustyGasTransport.h:175:updateMultiDiffCoeffs", "to": "include/cantera/transport/DustyGasTransport.h:198:eval_H_matrix"},
            {"from": "include/cantera/transport/DustyGasTransport.h:198:eval_H_matrix", "to": "include/cantera/transport/DustyGasTransport.h:168:updateBinaryDiffCoeffs"},
            {"from": "include/cantera/transport/DustyGasTransport.h:198:eval_H_matrix", "to": "include/cantera/transport/DustyGasTransport.h:185:updateKnudsenDiffCoeffs"},
            {"from": "src/transport/DustyGasTransport.cpp:105:cbar", "to": "include/cantera/transport/DustyGasTransport.h:175:updateMultiDiffCoeffs"},
            {"from": "src/transport/DustyGasTransport.cpp:106:gradc", "to": "include/cantera/transport/DustyGasTransport.h:175:updateMultiDiffCoeffs"},
        ],
    }
    call_graph_path = audit_dir / "dusty_gas_transport_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Created {call_graph_path}")

if __name__ == "__main__":
    main()
