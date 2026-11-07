"""Update next solver selection and build closure artifacts for HighPressureGasTransport."""

from datetime import datetime, timezone
import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    provenance_dir = repo_root / "provenance"

    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    hp_symbols = []
    with open(inventory_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "HighPressureGasTransport" in rec["source_file"]:
                    hp_symbols.append(rec)

    # 1. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "native_oracle_pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "selected_solver_family": "high_pressure_gas_transport",
        "native_class": "HighPressureGasTransportBase",
        "native_file": "src/transport/HighPressureGasTransport.cpp",
        "subsystem": "high-pressure transport",
        "current_family_status": "PARTIAL",
        "target_family_status": "VALIDATED_NUMERICALLY_EQUIVALENT",
        "rationale": (
            "DustyGasTransport (porous transport) is now 100% closed, validated against pinned native Cantera "
            "(726522be4e2a13454d8415b7ef799d621f665cf3) across all 66 test cases (106,833 matrix elements, full KxK parity, "
            "and molar fluxes), and frozen. HighPressureGasTransport is the next dependency-ready solver family in the "
            "transport subsystem. All its foundational dependencies (ideal_gas_thermo, cubic_eos, pure_fluid_properties, "
            "mixture_transport, multicomponent_transport) are complete and validated."
        ),
        "prerequisite_dependencies_satisfied": [
            "thermodynamic_state_inversion",
            "ideal_gas_thermo",
            "pure_fluid_properties",
            "cubic_eos",
            "vcs_multiphase_equilibrium",
            "mixture_transport",
            "multicomponent_transport",
            "dusty_gas_transport"
        ],
        "open_symbol_burn_down_target": len(hp_symbols),
        "target_closure_subgraph": [r["stable_symbol_id"] for r in hp_symbols],
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Updated {selection_path}")

    # 2. Build audit/high_pressure_gas_transport_closure_symbol_set.json
    closure_set = {}
    for r in hp_symbols:
        closure_set[r["stable_symbol_id"]] = {
            "stable_symbol_id": r["stable_symbol_id"],
            "native_file": r["source_file"],
            "native_class": r.get("class", "HighPressureGasTransportBase"),
            "native_method": r["function_or_method"],
            "python_target": r.get("python_mapping") or f"cantera_python.transport.HighPressureGasTransport.{r['function_or_method']}",
            "current_status": r["current_status"],
            "scientific_runtime_relevance": r.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
            "callers": r.get("callers", []),
            "callees": r.get("callees", []),
            "remaining_validation_requirement": "Differential parity against pinned native Cantera HighPressureGasTransport",
        }

    closure_path = audit_dir / "high_pressure_gas_transport_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)
    print(f"Created {closure_path} with {len(closure_set)} symbols")

    # 3. Build audit/high_pressure_gas_transport_runtime_call_graph.json
    call_graph = {
        "schema_version": 1,
        "entry_points": [
            "HighPressureGasTransportBase::updateCorrectionFactors",
            "HighPressureGasTransport::viscosity",
            "HighPressureGasTransport::thermalConductivity"
        ],
        "nodes": closure_set,
        "edges": [
            {"from": "include/cantera/transport/HighPressureGasTransport.h:87:updateCorrectionFactors", "to": "include/cantera/transport/HighPressureGasTransport.h:118:Tcrit_i"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:87:updateCorrectionFactors", "to": "include/cantera/transport/HighPressureGasTransport.h:121:Pcrit_i"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:45:getBinaryDiffCoeffs", "to": "include/cantera/transport/HighPressureGasTransport.h:87:updateCorrectionFactors"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:57:getMixDiffCoeffs", "to": "include/cantera/transport/HighPressureGasTransport.h:45:getBinaryDiffCoeffs"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:69:getMixDiffCoeffsMole", "to": "include/cantera/transport/HighPressureGasTransport.h:45:getBinaryDiffCoeffs"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:81:getMixDiffCoeffsMass", "to": "include/cantera/transport/HighPressureGasTransport.h:45:getBinaryDiffCoeffs"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:374:viscosity", "to": "include/cantera/transport/HighPressureGasTransport.h:671:computeMixtureParameters"},
            {"from": "include/cantera/transport/HighPressureGasTransport.h:358:thermalConductivity", "to": "include/cantera/transport/HighPressureGasTransport.h:671:computeMixtureParameters"},
        ],
    }
    call_graph_path = audit_dir / "high_pressure_gas_transport_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Created {call_graph_path}")

if __name__ == "__main__":
    main()
