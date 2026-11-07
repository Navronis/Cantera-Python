"""Build Thermodynamics Family 2 (SpeciesThermo) closure planning artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
COVERAGE = ROOT / "audit" / "thermo_complete_family_coverage.json"

SELECTION_FILE = ROOT / "audit" / "thermo_family2_selection.json"
SYMBOL_SET_FILE = ROOT / "audit" / "thermo_family2_closure_symbol_set.json"
CALL_GRAPH_FILE = ROOT / "audit" / "thermo_family2_call_graph.json"
SNAPSHOT_FILE = ROOT / "audit" / "thermo_family2_starting_snapshot.json"

def main():
    with open(COVERAGE, "r", encoding="utf-8") as f:
        coverage = json.load(f)

    fam2_info = coverage["families"]["THERMO_FAMILY_2_SPECIES_THERMO"]
    fam2_ids = set(fam2_info["canonical_ids"])

    fam2_symbols = []
    with open(INVENTORY, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("stable_symbol_id") in fam2_ids:
                fam2_symbols.append(rec)

    print(f"Found {len(fam2_symbols)} canonical symbols for Family 2.")

    # 1. Selection
    selection = {
        "family": "THERMO_FAMILY_2_SPECIES_THERMO",
        "subsystem": "thermodynamics",
        "description": "Species standard-state thermodynamic property parameterizations (NASA-7, NASA-9, Shomate, ConstCp, Mu0) and multi-species thermo managers.",
        "canonical_symbol_count": len(fam2_symbols),
        "source_files": fam2_info["source_files"],
        "primary_classes": [
            "SpeciesThermoInterpType",
            "NasaPoly1",
            "NasaPoly2",
            "Nasa9Poly1",
            "Nasa9PolyMultiTempRegion",
            "ShomatePoly",
            "ShomatePoly2",
            "ConstCpPoly",
            "Mu0Poly",
            "MultiSpeciesThermo",
            "Species",
            "Elements"
        ],
        "scientific_governing_equations": [
            "NASA-7: Cp0/R = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4",
            "NASA-7: H0/RT = a0 + a1/2*T + a2/3*T^2 + a3/4*T^3 + a4/5*T^4 + a5/T",
            "NASA-7: S0/R  = a0*ln(T) + a1*T + a2/2*T^2 + a3/3*T^3 + a4/4*T^4 + a6",
            "NASA-9: Cp0/R = a0/T^2 + a1/T + a2 + a3*T + a4*T^2 + a5*T^3 + a6*T^4",
            "Shomate: Cp0   = A + B*t + C*t^2 + D*t^3 + E/t^2 (t = T/1000)",
            "ConstCp: Cp0/R = c0, H0/RT = c0*(T-T0)/T + h0/RT, S0/R = c0*ln(T/T0) + s0/R"
        ],
        "validation_strategy": "Corpus evaluation against pinned native Cantera (726522be4e2a13454d8415b7ef799d621f665cf3) across temperature sweeps."
    }
    with open(SELECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    # 2. Closure Symbol Set
    closure_set = {
        "family": "THERMO_FAMILY_2_SPECIES_THERMO",
        "subsystem": "thermodynamics",
        "symbol_count": len(fam2_symbols),
        "symbols": [
            {
                "canonical_index": s.get("canonical_index"),
                "stable_symbol_id": s.get("stable_symbol_id"),
                "source_file": s.get("source_file"),
                "class": s.get("class"),
                "function_or_method": s.get("function_or_method"),
                "signature": s.get("signature"),
                "current_status": s.get("current_status"),
                "scientific_runtime_relevance": s.get("scientific_runtime_relevance"),
                "python_target": f"cantera_python.species_thermo.{s.get('class', 'free')}.{s.get('function_or_method')}"
            }
            for s in fam2_symbols
        ]
    }
    with open(SYMBOL_SET_FILE, "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)

    # 3. Call Graph
    call_graph = {
        "family": "THERMO_FAMILY_2_SPECIES_THERMO",
        "root_class": "SpeciesThermoInterpType",
        "primary_callers": [
            "ThermoPhase::update_T",
            "IdealGasPhase::updateThermo",
            "MultiSpeciesThermo::update_one",
            "Kinetics::updateROP"
        ],
        "core_method_clusters": {
            "evaluation": ["updatePropertiesTemp", "updateProperties", "reportParameters"],
            "bounds": ["minTemp", "maxTemp", "refPressure"],
            "manager": ["MultiSpeciesThermo::install", "MultiSpeciesThermo::update_temp", "MultiSpeciesThermo::update_one"]
        }
    }
    with open(CALL_GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    # 4. Starting Snapshot
    status_counts = {}
    for s in fam2_symbols:
        st = s.get("current_status")
        status_counts[st] = status_counts.get(st, 0) + 1

    snapshot = {
        "family": "THERMO_FAMILY_2_SPECIES_THERMO",
        "subsystem": "thermodynamics",
        "total_symbols": len(fam2_symbols),
        "status_distribution": status_counts,
        "global_inventory_metrics": {
            "canonical_total": 5784,
            "pinned_native_validated": 1737,
            "proven_non_runtime": 759,
            "fully_dispositioned": 2496,
            "unresolved": 3288
        }
    }
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print("Family 2 closure artifacts generated successfully!")

if __name__ == "__main__":
    main()
