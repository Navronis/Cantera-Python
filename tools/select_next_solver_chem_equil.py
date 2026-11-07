"""Update next solver selection and build closure artifacts for ChemEquil (single-phase equilibrium)."""

from datetime import datetime, timezone
import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    provenance_dir = repo_root / "provenance"

    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    chemequil_symbols = []
    with open(inventory_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                idx = rec["canonical_index"]
                if (358 <= idx <= 367) or (4318 <= idx <= 4349):
                    chemequil_symbols.append(rec)

    # 1. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "native_oracle_pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "selected_solver_family": "chem_equil",
        "native_class": "ChemEquil",
        "native_file": "src/equil/ChemEquil.cpp",
        "subsystem": "single-phase equilibrium",
        "current_family_status": "APPROXIMATE",
        "target_family_status": "VALIDATED_NUMERICALLY_EQUIVALENT",
        "rationale": (
            "All transport solvers (mixture_transport, multicomponent_transport, dusty_gas_transport, "
            "and high_pressure_gas_transport) plus nonideal VCS multiphase equilibrium are 100% closed, "
            "validated against pinned native Cantera (726522be4e2a13454d8415b7ef799d621f665cf3), and frozen. "
            "ChemEquil (single-phase element potential chemical equilibrium) is the next dependency-ready solver "
            "family. All its foundational dependencies (ideal_gas_thermo, thermodynamic_state_inversion, "
            "numerics_householder_qr) are complete and validated."
        ),
        "prerequisite_dependencies_satisfied": [
            "thermodynamic_state_inversion",
            "ideal_gas_thermo",
            "pure_fluid_properties",
            "cubic_eos",
            "vcs_multiphase_equilibrium",
            "mixture_transport",
            "multicomponent_transport",
            "dusty_gas_transport",
            "high_pressure_gas_transport"
        ],
        "open_symbol_burn_down_target": sum(1 for r in chemequil_symbols if r.get("final_disposition") is None),
        "target_closure_subgraph": [r["stable_symbol_id"] for r in chemequil_symbols],
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Updated {selection_path}")

    # 2. Build audit/chem_equil_closure_symbol_set.json
    closure_set = {}
    for r in chemequil_symbols:
        fn = r["function_or_method"]
        cls = r.get("class")
        if cls:
            py_target = f"cantera_python.equilibrium.{cls}.{fn}"
        elif fn == "_equilflag":
            py_target = "cantera_python.equilibrium._equilflag"
        else:
            py_target = f"cantera_python.equilibrium.ChemEquil.{fn}"

        closure_set[r["stable_symbol_id"]] = {
            "canonical_index": r["canonical_index"],
            "stable_symbol_id": r["stable_symbol_id"],
            "native_file": r["source_file"],
            "native_class": r.get("class") or "ChemEquil",
            "native_method": fn,
            "python_target": r.get("python_mapping") or py_target,
            "current_status": r["current_status"],
            "scientific_runtime_relevance": r.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
            "callers": r.get("callers", []),
            "callees": r.get("callees", []),
            "remaining_validation_requirement": "Differential parity against pinned native Cantera ChemEquil across TP, HP, SP, UV, SV property pairs",
        }

    closure_path = audit_dir / "chem_equil_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)
    print(f"Created {closure_path} with {len(closure_set)} symbols")

    # 3. Build audit/chem_equil_runtime_call_graph.json
    call_graph = {
        "schema_version": 1,
        "entry_points": [
            "ChemEquil::equilibrate",
            "ChemEquil::initialize",
            "ChemEquil::estimateEP_Brinkley"
        ],
        "nodes": closure_set,
        "edges": [
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:159:initialize"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:244:update"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:174:setInitialMoles"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:213:estimateEP_Brinkley"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:170:setToEquilState"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:228:dampStep"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:241:adjustEloc"},
            {"from": "include/cantera/equil/ChemEquil.h:109:equilibrate", "to": "include/cantera/equil/ChemEquil.h:19:_equilflag"},
            {"from": "include/cantera/equil/ChemEquil.h:159:initialize", "to": "include/cantera/equil/ChemEquil.h:151:nAtoms"},
        ],
    }
    call_graph_path = audit_dir / "chem_equil_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Created {call_graph_path}")

if __name__ == "__main__":
    main()
