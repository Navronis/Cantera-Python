"""Build audit artifacts for Thermodynamics Family 6: Electrolytes and Condensed Phases."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
PROVENANCE = ROOT / "provenance"

def main():
    coverage_path = AUDIT / "thermo_complete_family_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    f6_info = coverage["families"]["THERMO_FAMILY_6_ELECTROLYTES_CONDENSED"]
    f6_ids = set(f6_info["canonical_ids"])

    inv_path = PROVENANCE / "canonical_5784_symbol_inventory.jsonl"
    inventory = [json.loads(line) for line in inv_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    f6_symbols = [r for r in inventory if r["stable_symbol_id"] in f6_ids]
    assert len(f6_symbols) == 555, f"Expected 555 symbols, got {len(f6_symbols)}"

    status_dist = Counter(r["current_status"] for r in f6_symbols)
    print("Family 6 Status distribution:", status_dist)

    # 1. Starting Snapshot
    dashboard_path = PROVENANCE / "symbol_closure_dashboard.json"
    dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))

    starting_snapshot = {
        "family": "THERMO_FAMILY_6_ELECTROLYTES_CONDENSED",
        "subsystem": "thermodynamics",
        "total_symbols": len(f6_symbols),
        "status_distribution": dict(status_dist),
        "global_inventory_metrics": {
            "canonical_total": dashboard["TOTAL"],
            "pinned_native_validated": dashboard["PINNED_NATIVE_VALIDATED"],
            "proven_non_runtime": dashboard["NON_RUNTIME_CPP_ONLY_PROVEN"],
            "fully_dispositioned": dashboard["fully_dispositioned"],
            "unresolved": dashboard["remaining_without_final_disposition"]
        }
    }
    (AUDIT / "thermo_family6_starting_snapshot.json").write_text(
        json.dumps(starting_snapshot, indent=2) + "\n", encoding="utf-8"
    )

    # 2. Closure Symbol Set
    closure_symbols = []
    for r in f6_symbols:
        cls_name = r.get("class") or ""
        fn_name = r.get("function_or_method", "")
        # Map to python module
        if "PDSS" in cls_name:
            py_mod = "pdss"
        elif cls_name in ("DebyeHuckel", "HMWSoln", "IdealMolalSoln", "MolalityVPSSTP", "WaterSSTP"):
            py_mod = "electrolytes"
        elif cls_name in ("MargulesVPSSTP", "RedlichKisterVPSSTP", "GibbsExcessVPSSTP", "BinarySolutionTabulatedThermo", "IdealSolnGasVPSS", "VPStandardStateTP"):
            py_mod = "phases"
        else:
            py_mod = "electrolytes"

        py_target = r.get("python_mapping") or f"cantera_python.{py_mod}.{cls_name}.{fn_name}" if cls_name else f"cantera_python.{py_mod}.{fn_name}"
        closure_symbols.append({
            "stable_symbol_id": r["stable_symbol_id"],
            "canonical_index": r["canonical_index"],
            "source_file": r["source_file"],
            "line_range": r["line_range"],
            "signature": r["signature"],
            "function_or_method": fn_name,
            "class": cls_name,
            "current_status": r["current_status"],
            "scientific_runtime_relevance": r.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
            "python_target": py_target
        })

    closure_set = {
        "family": "THERMO_FAMILY_6_ELECTROLYTES_CONDENSED",
        "subsystem": "thermodynamics",
        "symbol_count": len(closure_symbols),
        "symbols": closure_symbols
    }
    (AUDIT / "thermo_family6_closure_symbol_set.json").write_text(
        json.dumps(closure_set, indent=2) + "\n", encoding="utf-8"
    )

    # 3. Runtime Call Graph
    call_graph = {
        "family": "THERMO_FAMILY_6_ELECTROLYTES_CONDENSED",
        "root_classes": [
            "VPStandardStateTP",
            "PDSS",
            "PDSS_ConstVol",
            "PDSS_SSVol",
            "PDSS_HKFT",
            "PDSS_Water",
            "PDSSFactory",
            "WaterSSTP",
            "MolalityVPSSTP",
            "DebyeHuckel",
            "HMWSoln",
            "IdealMolalSoln",
            "IdealSolnGasVPSS",
            "GibbsExcessVPSSTP",
            "MargulesVPSSTP",
            "RedlichKisterVPSSTP",
            "BinarySolutionTabulatedThermo"
        ],
        "inherits_from": "ThermoPhase",
        "primary_callers": [
            "Kinetics (ElectrolyteKinetics, InterfaceKinetics)",
            "Transport (AqueousTransport, LiquidTransport)",
            "Reactor (ConstVolumeReactor, FlowReactor)",
            "Equilibrium / VCS (Electrolyte multiphase equilibrium, precipitation, dissolution)",
            "Electrochemistry (Battery and fuel cell electrolyte potential gradients)"
        ],
        "core_model_clusters": {
            "VPStandardStateTP": [
                "createInstallPDSS",
                "standardConcentration",
                "getActivityConcentrations",
                "getStandardChemPotentials",
                "getEnthalpy_RT",
                "getEntropy_R",
                "getGibbs_RT",
                "getCp_R",
                "getIntEnergy_RT",
                "getPartialMolarVolumes",
                "initThermo",
                "initLengths",
                "setState_TP",
                "calcDensity"
            ],
            "PDSS": [
                "enthalpy_mole",
                "intEnergy_mole",
                "entropy_mole",
                "gibbs_mole",
                "cp_mole",
                "cv_mole",
                "molarVolume",
                "density",
                "pressure",
                "setPressure",
                "setTemperature",
                "setState_TP",
                "setState_TR",
                "critTemperature",
                "critPressure",
                "critDensity",
                "initThermo"
            ],
            "PDSS_HKFT": [
                "calc_HKFT_soln",
                "lookup_HKFT_params",
                "enthalpy_mole",
                "entropy_mole",
                "cp_mole",
                "molarVolume",
                "gibbs_mole"
            ],
            "PDSS_Water": [
                "enthalpy_mole",
                "entropy_mole",
                "cp_mole",
                "molarVolume",
                "gibbs_mole",
                "density",
                "pressure",
                "dthermalExpansionCoeffdT"
            ],
            "MolalityVPSSTP": [
                "setMolalities",
                "setMolalitiesByName",
                "getMolalities",
                "calcDensity",
                "standardConcentration",
                "getStandardChemPotentials",
                "getActivityCoefficients"
            ],
            "DebyeHuckel": [
                "A_Debye_coeffs",
                "B_Debye_coeffs",
                "s_Beta_ij",
                "getActivities",
                "getActivityCoefficients",
                "getMolalityActivityCoefficients",
                "getChemPotentials",
                "getPartialMolarEnthalpies",
                "getPartialMolarEntropies",
                "getPartialMolarCp",
                "getPartialMolarVolumes"
            ],
            "HMWSoln": [
                "Beta0_ij",
                "Beta1_ij",
                "Beta2_ij",
                "Cphi_ij",
                "Theta_ij",
                "Psi_ijk",
                "s_B_ij",
                "s_C_ij",
                "calc_Pitzer_coeffs",
                "getActivities",
                "getActivityCoefficients",
                "getMolalityActivityCoefficients",
                "getChemPotentials",
                "getPartialMolarEnthalpies",
                "getPartialMolarEntropies",
                "getPartialMolarCp",
                "getPartialMolarVolumes"
            ],
            "MargulesVPSSTP": [
                "getActivities",
                "getActivityCoefficients",
                "getChemPotentials",
                "getPartialMolarEnthalpies",
                "getPartialMolarEntropies",
                "getPartialMolarVolumes"
            ],
            "RedlichKisterVPSSTP": [
                "getActivities",
                "getActivityCoefficients",
                "getChemPotentials",
                "getPartialMolarEnthalpies",
                "getPartialMolarEntropies",
                "getPartialMolarVolumes"
            ]
        }
    }
    (AUDIT / "thermo_family6_runtime_call_graph.json").write_text(
        json.dumps(call_graph, indent=2) + "\n", encoding="utf-8"
    )

    print("Family 6 audit artifacts generated successfully:")
    print("  - audit/thermo_family6_starting_snapshot.json")
    print("  - audit/thermo_family6_closure_symbol_set.json")
    print("  - audit/thermo_family6_runtime_call_graph.json")

if __name__ == "__main__":
    main()
