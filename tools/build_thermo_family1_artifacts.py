"""Build Family 1 artifacts for Thermodynamics subsystem closure:
Family 1: Phase Core State & Multi-Component Foundation (Phase.h, Phase.cpp - 119 symbols).
"""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
inventory_path = root / "provenance" / "canonical_5784_symbol_inventory.jsonl"
audit_dir = root / "audit"

def main():
    family1_symbols = []
    with open(inventory_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            src = rec.get("source_file", "")
            if src in ("include/cantera/thermo/Phase.h", "src/thermo/Phase.cpp"):
                family1_symbols.append(rec)

    print(f"Found {len(family1_symbols)} symbols in Thermo Family 1 (Phase.h & Phase.cpp)")

    # 1. Family selection artifact
    family_selection = {
        "selected_family": "THERMO_FAMILY_1_PHASE_CORE_STATE_FOUNDATION",
        "subsystem": "thermo",
        "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "symbol_count": len(family1_symbols),
        "source_files": [
            "include/cantera/thermo/Phase.h",
            "src/thermo/Phase.cpp"
        ],
        "rationale": (
            "Phase.h and Phase.cpp define the authoritative base multi-component state container (Phase) "
            "for all thermodynamic subsystems in Cantera. Every ThermoPhase, IdealGasPhase, SurfPhase, "
            "Cubic/Real gas phase, and electrolyte solution inherits state variables (T, P, rho, Y, X, C, V), "
            "element definitions, species definitions, molecular weights, and coordinate representations "
            "directly from Phase. Closing Phase first establishes the immutable multi-component base required "
            "by all downstream thermo models."
        ),
        "dependency_ordering": [
            "THERMO_FAMILY_1_PHASE_CORE_STATE_FOUNDATION (Phase.h, Phase.cpp - 119 symbols)",
            "THERMO_FAMILY_2_SPECIES_THERMO_POLYNOMIAL_MODELS (NasaPoly1/2, Shomate, MultiSpeciesThermo - ~70 symbols)",
            "THERMO_FAMILY_3_THERMOPHASE_BASE_ARCH_RELATIONS (ThermoPhase.h, ThermoPhase.cpp - 188 symbols)",
            "THERMO_FAMILY_4_IDEAL_GAS_AND_SOLUTION_PHASES (IdealGasPhase, IdealSolidSolnPhase, SurfPhase - ~160 symbols)",
            "THERMO_FAMILY_5_REAL_GAS_AND_CUBIC_EOS (PengRobinson, RedlichKwong, MixtureFugacityTP - ~180 symbols)",
            "THERMO_FAMILY_6_ELECTROLYTES_AND_LIQUIDS (WaterSSTP, DebyeHuckel, HMWSoln, PDSS - ~350 symbols)",
            "THERMO_FAMILY_7_PLASMA_AND_NON_EQUILIBRIUM (PlasmaPhase, EEDF - ~120 symbols)"
        ]
    }
    with open(audit_dir / "thermo_next_family_selection.json", "w", encoding="utf-8") as f:
        json.dump(family_selection, f, indent=2)

    # 2. Active closure symbol set artifact
    closure_set = {
        "family": "THERMO_FAMILY_1_PHASE_CORE_STATE_FOUNDATION",
        "subsystem": "thermo",
        "symbol_count": len(family1_symbols),
        "symbols": [
            {
                "canonical_index": s["canonical_index"],
                "stable_symbol_id": s["stable_symbol_id"],
                "source_file": s["source_file"],
                "class": s.get("class", "Phase"),
                "function_or_method": s["function_or_method"],
                "signature": s.get("signature", ""),
                "current_status": s["current_status"],
                "scientific_runtime_relevance": s.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                "python_target": f"cantera_python.phases.Phase.{s['function_or_method']}"
            }
            for s in family1_symbols
        ]
    }
    with open(audit_dir / "thermo_active_closure_symbol_set.json", "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)

    # 3. Active starting snapshot
    from collections import Counter
    snapshot = {
        "family": "THERMO_FAMILY_1_PHASE_CORE_STATE_FOUNDATION",
        "subsystem": "thermo",
        "total_symbols": len(family1_symbols),
        "status_distribution": dict(Counter(s["current_status"] for s in family1_symbols)),
        "global_inventory_metrics": {
            "total_canonical": 5784,
            "pinned_native_validated": 1627,
            "non_runtime_cpp_only_proven": 759,
            "fully_dispositioned": 2386,
            "unresolved": 3398
        }
    }
    with open(audit_dir / "thermo_active_starting_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    # 4. Runtime call graph
    call_graph = {
        "family": "THERMO_FAMILY_1_PHASE_CORE_STATE_FOUNDATION",
        "root_class": "Phase",
        "primary_callers": [
            "ThermoPhase (derived thermo class, inherits state)",
            "Kinetics (reads species concentrations, temperature, mole fractions)",
            "Transport (reads composition, molecular weights, density, temperature)",
            "Reactor (reads density, mass fractions, volume)",
            "OneDim / Domain1D (reads composition, density, molecular weights)"
        ],
        "core_method_clusters": {
            "identification_and_elements": [
                "name", "setName", "type", "elementName", "elementIndex", "elementNames",
                "atomicWeight", "atomicWeights", "atomicNumber", "nElements", "checkElementIndex"
            ],
            "species_and_molecular_weights": [
                "speciesName", "speciesIndex", "speciesNames", "nSpecies", "checkSpeciesIndex",
                "molecularWeight", "molecularWeights", "meanMolecularWeight"
            ],
            "state_coordinates": [
                "temperature", "setTemperature", "density", "setDensity", "molarDensity",
                "molarVolume", "pressure", "setPressure"
            ],
            "composition_representations": [
                "getMoleFractions", "setMoleFractions", "setMoleFractionsByName",
                "getMassFractions", "setMassFractions", "setMassFractionsByName",
                "getConcentrations", "setConcentrations", "concentration", "moleFraction", "massFraction"
            ],
            "partial_molar_quantities_and_stoichiometry": [
                "nAtoms", "getAtoms", "elementalMassFraction", "elementalMoleFraction"
            ]
        }
    }
    with open(audit_dir / "thermo_active_runtime_call_graph.json", "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    print("Generated Family 1 artifacts in audit/ successfully.")

if __name__ == "__main__":
    main()
