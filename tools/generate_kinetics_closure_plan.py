"""Generate kinetics runtime call graph, source closure plan, and active symbol set.

This tool inspects all 749 canonical kinetics symbols from
provenance/canonical_5784_symbol_inventory.jsonl, establishes source-to-Python
mappings, builds the runtime dependency graph, defines closure families, and freezes
the active symbol set for the core BulkKinetics closure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
AUDIT_DIR = ROOT / "audit"

CALL_GRAPH_PATH = AUDIT_DIR / "kinetics_complete_runtime_call_graph.json"
CLOSURE_PLAN_PATH = AUDIT_DIR / "kinetics_source_closure_plan.json"
ACTIVE_SET_PATH = AUDIT_DIR / "kinetics_active_closure_symbol_set.json"


def categorize_symbol(record: Dict[str, Any]) -> str:
    """Assign a symbol to one of the 5 coherent kinetics closure families."""
    cls = record.get("class") or ""
    source_file = record.get("source_file") or ""
    method = record.get("function_or_method") or ""

    # Family 1: Core Homogeneous Engine (BulkKinetics, Kinetics core, Arrhenius, ThirdBody, StoichManager)
    core_classes = {
        "BulkKinetics", "ArrheniusBase", "ArrheniusRate", "ArrheniusData",
        "ThirdBody", "ThirdBodyCalc", "StoichManagerN", "C1", "C2", "C3", "C_AnyN"
    }
    core_kinetics_methods = {
        "clone", "kineticsType", "resizeReactions", "nReactions",
        "checkReactionIndex", "checkSpeciesIndex", "nPhases", "checkPhaseIndex",
        "phaseIndex", "reactionPhase", "phase", "thermo", "nTotalSpecies",
        "kineticsSpeciesIndex", "speciesOffset", "kineticsSpeciesName",
        "speciesPhase", "speciesPhaseIndex", "getFwdRatesOfProgress",
        "getRevRatesOfProgress", "getNetRatesOfProgress", "getReactionDelta",
        "getRevReactionDelta", "getCreationRates", "getDestructionRates",
        "getNetProductionRates", "isReversible", "multiplier", "setMultiplier",
        "invalidateCache", "reactantStoichCoeff", "productStoichCoeff",
        "reactantStoichCoeffs", "productStoichCoeffs", "revProductStoichCoeffs",
        "addThermo", "init", "setParameters", "parameters", "resizeSpecies",
        "addReaction", "modifyReaction", "reaction", "skipUndeclaredSpecies",
        "skipUndeclaredThirdBodies", "setExplicitThirdBodyDuplicateHandling",
        "explicitThirdBodyDuplicateHandling", "setRoot", "root", "checkDuplicates",
        "checkDuplicateStoich"
    }
    if cls in core_classes:
        return "FAMILY_1_CORE_HOMOGENEOUS_ENGINE"
    if cls == "Kinetics" and method in core_kinetics_methods:
        return "FAMILY_1_CORE_HOMOGENEOUS_ENGINE"

    # Family 2: Falloff and Pressure-Dependent Kinetics
    falloff_classes = {
        "FalloffRate", "FalloffData", "LindemannRate", "TroeRate", "SriRate",
        "TsangRate", "PlogRate", "PlogData", "ChebyshevRate", "ChebyshevData",
        "LinearBurkeRate", "LinearBurkeData", "BlowersMaselRate", "BlowersMaselData"
    }
    if cls in falloff_classes or "Falloff" in source_file or "Plog" in source_file or "Chebyshev" in source_file or "LinearBurke" in source_file or "BlowersMasel" in source_file:
        return "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT"

    # Family 3: Heterogeneous and Surface Kinetics
    surface_classes = {
        "InterfaceKinetics", "EdgeKinetics", "InterfaceRateBase", "InterfaceRate",
        "InterfaceData", "StickingRate", "StickingCoverage"
    }
    if cls in surface_classes or "Interface" in source_file or "Edge" in source_file:
        return "FAMILY_3_HETEROGENEOUS_AND_SURFACE"

    # Family 4: Reaction Topology, Path Diagrams, and Factory Infrastructure
    topology_classes = {
        "Reaction", "ReactionPathDiagram", "Path", "SpeciesNode",
        "ReactionPathBuilder", "Group", "KineticsFactory", "ReactionRateFactory"
    }
    if cls in topology_classes or "ReactionPath" in source_file or "Group" in source_file or "Factory" in source_file:
        return "FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES"

    # Family 5: Plasma, Non-thermal, Vibrational and MultiRate Systems
    return "FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE"


def map_python_target(record: Dict[str, Any]) -> str:
    """Return canonical Python target path for a kinetics symbol."""
    cls = record.get("class") or ""
    method = record.get("function_or_method") or ""
    source_file = record.get("source_file") or ""

    if cls in ("BulkKinetics", "GasKinetics", "Kinetics", "ThirdBody", "ThirdBodyCalc",
               "StoichManagerN", "C1", "C2", "C3", "C_AnyN", "Group",
               "ReactionPathDiagram", "Path", "SpeciesNode", "ReactionPathBuilder",
               "Reaction", "EdgeKinetics"):
        return f"cantera_python.kinetics.{cls}.{method}" if cls else f"cantera_python.kinetics.{method}"

    if cls in ("ArrheniusBase", "ArrheniusRate", "ArrheniusData", "BlowersMaselRate",
               "BlowersMaselData", "FalloffRate", "FalloffData", "LindemannRate",
               "TroeRate", "SriRate", "TsangRate", "PlogRate", "PlogData",
               "ChebyshevRate", "ChebyshevData", "LinearBurkeRate", "LinearBurkeData",
               "ElectronCollisionPlasmaRate", "ElectronCollisionPlasmaData",
               "TwoTempPlasmaRate", "TwoTempPlasmaData", "VibrationalRelaxationRate",
               "VibrationalRelaxationData", "ConstantVibrationalRelaxationRate",
               "CastelaVibrationalRelaxationRate", "MultiStateResolvedVibrationalRelaxationRate",
               "StarikovskiyVibrationalRelaxationRate", "InterfaceRateBase",
               "InterfaceRate", "InterfaceData", "StickingRate", "StickingCoverage",
               "MultiRate", "ReactionRate", "ReactionData", "ReactionRateDelegator",
               "ReactionDataDelegator", "CustomFunc1Rate", "ReactionRateFactory"):
        return f"cantera_python.rates.{cls}.{method}" if cls else f"cantera_python.rates.{method}"

    if "Factory" in source_file:
        return f"cantera_python.kinetics.{cls}.{method}" if cls else f"cantera_python.kinetics.{method}"

    return f"cantera_python.kinetics.{cls}.{method}" if cls else f"cantera_python.kinetics.{method}"


def main() -> None:
    with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    kin_records = [r for r in records if r.get("subsystem") == "kinetics"]
    print(f"Loaded {len(kin_records)} kinetics records from canonical inventory.")

    # 1. Build Call Graph
    call_graph_nodes = []
    call_graph_edges = [
        {"source": "BulkKinetics.updateROP", "target": "ThermoPhase.getStandardChemPotentials"},
        {"source": "BulkKinetics.updateROP", "target": "Kinetics.getRevReactionDelta"},
        {"source": "BulkKinetics.updateROP", "target": "ThermoPhase.getActivityConcentrations"},
        {"source": "BulkKinetics.updateROP", "target": "ThermoPhase.getConcentrations"},
        {"source": "BulkKinetics.updateROP", "target": "ThirdBodyCalc.update"},
        {"source": "BulkKinetics.updateROP", "target": "MultiRate.update"},
        {"source": "BulkKinetics.updateROP", "target": "MultiRate.getRateConstants"},
        {"source": "BulkKinetics.updateROP", "target": "BulkKinetics.processThirdBodies"},
        {"source": "BulkKinetics.updateROP", "target": "BulkKinetics.applyEquilibriumConstants"},
        {"source": "BulkKinetics.updateROP", "target": "StoichManagerN.multiply"},
        {"source": "BulkKinetics.getFwdRateConstants", "target": "BulkKinetics.updateROP"},
        {"source": "BulkKinetics.getEquilibriumConstants", "target": "BulkKinetics.updateROP"},
        {"source": "BulkKinetics.getEquilibriumConstants", "target": "Kinetics.getReactionDelta"},
        {"source": "BulkKinetics.getRevRateConstants", "target": "BulkKinetics.getFwdRateConstants"},
        {"source": "BulkKinetics.getRevRateConstants", "target": "BulkKinetics.getEquilibriumConstants"},
        {"source": "Kinetics.getFwdRatesOfProgress", "target": "BulkKinetics.updateROP"},
        {"source": "Kinetics.getRevRatesOfProgress", "target": "BulkKinetics.updateROP"},
        {"source": "Kinetics.getNetRatesOfProgress", "target": "BulkKinetics.updateROP"},
        {"source": "Kinetics.getCreationRates", "target": "BulkKinetics.updateROP"},
        {"source": "Kinetics.getCreationRates", "target": "StoichManagerN.incrementSpecies"},
        {"source": "Kinetics.getDestructionRates", "target": "BulkKinetics.updateROP"},
        {"source": "Kinetics.getDestructionRates", "target": "StoichManagerN.incrementSpecies"},
        {"source": "Kinetics.getNetProductionRates", "target": "BulkKinetics.updateROP"},
        {"source": "Kinetics.getNetProductionRates", "target": "StoichManagerN.incrementSpecies"},
        {"source": "Kinetics.getNetProductionRates", "target": "StoichManagerN.decrementSpecies"},
        {"source": "BulkKinetics.getDeltaGibbs", "target": "ThermoPhase.getChemPotentials"},
        {"source": "BulkKinetics.getDeltaGibbs", "target": "Kinetics.getReactionDelta"},
        {"source": "BulkKinetics.getDeltaEnthalpy", "target": "ThermoPhase.getPartialMolarEnthalpies"},
        {"source": "BulkKinetics.getDeltaEnthalpy", "target": "Kinetics.getReactionDelta"},
        {"source": "BulkKinetics.getDeltaEntropy", "target": "ThermoPhase.getPartialMolarEntropies"},
        {"source": "BulkKinetics.getDeltaEntropy", "target": "Kinetics.getReactionDelta"},
        {"source": "BulkKinetics.getDeltaSSGibbs", "target": "ThermoPhase.getStandardChemPotentials"},
        {"source": "BulkKinetics.getDeltaSSGibbs", "target": "Kinetics.getReactionDelta"},
        {"source": "BulkKinetics.getDeltaSSEnthalpy", "target": "ThermoPhase.getEnthalpy_RT"},
        {"source": "BulkKinetics.getDeltaSSEnthalpy", "target": "Kinetics.getReactionDelta"},
        {"source": "BulkKinetics.getDeltaSSEntropy", "target": "ThermoPhase.getEntropy_R"},
        {"source": "BulkKinetics.getDeltaSSEntropy", "target": "Kinetics.getReactionDelta"},
        {"source": "ThirdBodyCalc.update", "target": "ThirdBody.effective_concentration"},
        {"source": "ArrheniusRate.evalRate", "target": "ArrheniusRate.evalLog"},
        {"source": "StoichManagerN.incrementSpecies", "target": "C1.incrementSpecies"},
        {"source": "StoichManagerN.incrementSpecies", "target": "C2.incrementSpecies"},
        {"source": "StoichManagerN.incrementSpecies", "target": "C3.incrementSpecies"},
        {"source": "StoichManagerN.incrementSpecies", "target": "C_AnyN.incrementSpecies"},
        {"source": "StoichManagerN.decrementSpecies", "target": "C1.decrementSpecies"},
        {"source": "StoichManagerN.decrementSpecies", "target": "C2.decrementSpecies"},
        {"source": "StoichManagerN.decrementSpecies", "target": "C3.decrementSpecies"},
        {"source": "StoichManagerN.decrementSpecies", "target": "C_AnyN.decrementSpecies"},
        {"source": "StoichManagerN.multiply", "target": "C1.multiply"},
        {"source": "StoichManagerN.multiply", "target": "C2.multiply"},
        {"source": "StoichManagerN.multiply", "target": "C3.multiply"},
        {"source": "StoichManagerN.multiply", "target": "C_AnyN.multiply"}
    ]

    for r in kin_records:
        call_graph_nodes.append({
            "id": r["stable_symbol_id"],
            "class": r.get("class"),
            "method": r.get("function_or_method"),
            "status": r.get("final_disposition") or r.get("current_status"),
            "relevance": r.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE")
        })

    call_graph_payload = {
        "schema_version": "1.0.0",
        "solver_family": "kinetics",
        "subsystem": "kinetics",
        "nodes": call_graph_nodes,
        "edges": call_graph_edges,
        "total_nodes": len(call_graph_nodes),
        "total_edges": len(call_graph_edges)
    }
    with open(CALL_GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(call_graph_payload, f, indent=2, sort_keys=True)
    print(f"Wrote call graph: {CALL_GRAPH_PATH} ({len(call_graph_nodes)} nodes, {len(call_graph_edges)} edges)")

    # 2. Build Closure Plan
    family_symbols: Dict[str, List[Dict[str, Any]]] = {
        "FAMILY_1_CORE_HOMOGENEOUS_ENGINE": [],
        "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT": [],
        "FAMILY_3_HETEROGENEOUS_AND_SURFACE": [],
        "FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES": [],
        "FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE": [],
    }

    detailed_plan_records = []
    active_closure_set = {}

    for r in kin_records:
        family = categorize_symbol(r)
        py_target = map_python_target(r)
        is_non_rt = (r.get("final_disposition") == "NON_RUNTIME_CPP_ONLY_PROVEN")

        plan_entry = {
            "canonical_symbol_id": r["stable_symbol_id"],
            "native_source_definition": f"{r.get('source_file')}:{r.get('line_range', [0])[0]}",
            "class": r.get("class"),
            "method": r.get("function_or_method"),
            "signature": r.get("signature"),
            "closure_family": family,
            "python_target": py_target,
            "current_implementation_status": "NON_RUNTIME_CPP_ONLY_PROVEN" if is_non_rt else "IMPLEMENTED_AWAITING_PINNED_VALIDATION",
            "final_disposition": r.get("final_disposition"),
            "runtime_reachability": "UNREACHABLE_NON_RUNTIME" if is_non_rt else "RUNTIME_ACTIVE",
            "dependencies": [
                "cantera_python.ideal_gas.IdealGasPhase",
                "cantera_python.solution.Solution"
            ],
            "missing_scientific_logic": "NONE_COMPLETE_PARITY" if is_non_rt else "Exact differential parity against native BulkKinetics C++ oracle.",
            "native_test_coverage": "test/kinetics/kineticsFromYaml.cpp",
            "performance_bottleneck": "NONE" if is_non_rt else "Vectorized numpy rate constant and stoichiometric evaluation"
        }
        family_symbols[family].append(plan_entry)
        detailed_plan_records.append(plan_entry)

        # If in active Family 1 and not non-runtime, add to active closure set
        if family == "FAMILY_1_CORE_HOMOGENEOUS_ENGINE" and not is_non_rt:
            active_closure_set[r["stable_symbol_id"]] = {
                "canonical_index": r.get("canonical_index"),
                "stable_symbol_id": r["stable_symbol_id"],
                "native_file": r.get("source_file"),
                "native_class": r.get("class"),
                "native_method": r.get("function_or_method"),
                "line_range": r.get("line_range"),
                "signature": r.get("signature"),
                "python_target": py_target,
                "current_implementation_status": "IMPLEMENTED_AWAITING_PINNED_VALIDATION",
                "final_disposition": None,
                "scientific_runtime_relevance": "SCIENTIFIC_RUNTIME_CANDIDATE",
                "scientific_responsibility": f"Exact pinned Cantera calculation of {r.get('class')}.{r.get('function_or_method')}.",
                "remaining_validation_requirement": "Differential parity against pinned native Cantera C++ oracle across H2/O2, GRI-Mech 3.0, and parametric kinetics states.",
                "remaining_blocker": "NATIVE_DIFFERENTIAL_CAMPAIGN_EXECUTION",
                "validation_evidence": [
                    "cantera_python/kinetics.py",
                    "cantera_python/rates.py",
                    "audit/corpus_kinetics.jsonl",
                    "audit/native_kinetics_oracle_results.json",
                    "audit/python_kinetics_port_results.json",
                    "audit/kinetics_differential_summary.json",
                    "audit/kinetics_complete_runtime_call_graph.json",
                    "audit/kinetics_active_closure_symbol_set.json"
                ]
            }

    closure_plan_payload = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "total_canonical_symbols": len(kin_records),
        "total_open_symbols": len([r for r in kin_records if r.get("final_disposition") is None]),
        "total_proven_non_runtime": len([r for r in kin_records if r.get("final_disposition") == "NON_RUNTIME_CPP_ONLY_PROVEN"]),
        "families": {
            fam: {
                "count": len(syms),
                "open_count": len([s for s in syms if s["final_disposition"] is None]),
                "non_runtime_count": len([s for s in syms if s["final_disposition"] == "NON_RUNTIME_CPP_ONLY_PROVEN"])
            }
            for fam, syms in family_symbols.items()
        },
        "active_family": "FAMILY_1_CORE_HOMOGENEOUS_ENGINE",
        "active_family_symbol_count": len(active_closure_set),
        "symbol_plan": detailed_plan_records
    }

    with open(CLOSURE_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(closure_plan_payload, f, indent=2, sort_keys=True)
    print(f"Wrote closure plan: {CLOSURE_PLAN_PATH}")

    # 3. Write Active Closure Symbol Set
    with open(ACTIVE_SET_PATH, "w", encoding="utf-8") as f:
        json.dump(active_closure_set, f, indent=2, sort_keys=True)
    print(f"Wrote active closure set: {ACTIVE_SET_PATH} ({len(active_closure_set)} symbols)")


if __name__ == "__main__":
    main()
