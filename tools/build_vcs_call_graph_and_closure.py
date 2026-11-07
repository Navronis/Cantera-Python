"""Build VCS call graph and closure symbol set from canonical inventory and pinned source."""

import json
from collections import Counter
from pathlib import Path

def build_call_graph_and_closure():
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    inventory_path = repo_root / "provenance" / "canonical_5784_symbol_inventory.jsonl"
    
    with open(inventory_path, "r", encoding="utf-8") as f:
        all_symbols = [json.loads(line) for line in f]

    symbol_map = {s["stable_symbol_id"]: s for s in all_symbols}

    # Identify all VCS symbols and MultiPhase / public equilibrium symbols that reach VCS
    vcs_symbols = [
        s for s in all_symbols
        if "vcs" in s["stable_symbol_id"].lower()
        or "vcs" in s.get("source_file", "").lower()
        or (s.get("class") and "vcs" in s["class"].lower())
    ]
    
    # Public equilibrium symbols reaching VCS
    entry_symbols = [
        s for s in all_symbols
        if s.get("source_file") in [
            "include/cantera/equil/MultiPhase.h",
            "src/equil/MultiPhase.cpp",
            "src/equil/vcs_MultiPhaseEquil.cpp",
        ] and any(kw in s.get("function_or_method", "").lower() for kw in ["equilibrate", "solve", "vcs"])
    ]

    # Combine unique symbols in the VCS runtime subgraph
    combined_symbols = {}
    for s in vcs_symbols + entry_symbols:
        combined_symbols[s["stable_symbol_id"]] = s

    # Call graph edges: map callers and callees based on VCS call structure
    # Pinned C++ VCS state machine call structure:
    # MultiPhase::equilibrate -> vcs_MultiPhaseEquil::equilibrate -> vcs_MultiPhaseEquil::equilibrate_TP -> VCS_SOLVE::solve_TP
    # VCS_SOLVE::solve_TP calls:
    #   - vcs_evalSS_TP
    #   - vcs_prep
    #   - vcs_inest
    #   - vcs_tmoles
    #   - vcs_setFlagsVolPhases
    #   - vcs_dfe
    #   - solve_tp_component_calc:
    #       - vcs_basopt
    #       - vcs_updateVP
    #       - vcs_deltag
    #       - vcs_evaluate_speciesType
    #       - vcs_elabcheck
    #       - vcs_elcorr
    #   - solve_tp_inner:
    #       - vcs_popPhaseID
    #       - vcs_RxnStepSizes
    #       - vcs_apply_minor_trial
    #       - vcs_apply_major_trial
    #       - vcs_apply_zeroed_trial
    #       - vcs_glob_step_damp
    #       - vcs_updateVP
    #       - vcs_total_gibbs
    #       - vcs_switch_pos
    #       - delta_species
    #       - vcs_zero_species
    #       - vcs_delete_species
    #       - vcs_delete_multiphase
    #   - solve_tp_equilib_check:
    #       - solve_tp_elem_abund_check
    #   - solve_tp_elem_abund_check:
    #       - vcs_elabcheck
    #       - vcs_elcorr
    #   - vcs_recheck_deleted:
    #       - vcs_reinsert_deleted
    #   - vcs_add_all_deleted
    #   - vcs_VolPhase methods for thermo/activity calculations

    call_graph = {}
    closure_set = {}

    vcs_runtime_symbols_total = 0
    vcs_already_final = 0
    vcs_partial = 0
    vcs_not_ported = 0

    for sid, s in sorted(combined_symbols.items()):
        status = s.get("current_status", "NOT_PORTED")
        is_non_runtime = (status == "NON_RUNTIME_CPP_ONLY_PROVEN")
        
        # Determine python target
        src_file = s.get("source_file", "")
        src_cls = s.get("class") or ""
        src_fn = s.get("function_or_method") or ""
        
        py_target = None
        if "vcs_VolPhase" in src_cls:
            py_target = f"cantera_python.vcs.vcs_VolPhase.{src_fn}"
        elif "VCS_SOLVE" in src_cls or "vcs_solve" in src_file.lower():
            py_target = f"cantera_python.vcs.VCS_SOLVE.{src_fn}"
        elif "MultiPhase" in src_cls or "MultiPhase" in src_file:
            py_target = f"cantera_python.vcs.MultiPhase.{src_fn}"
        else:
            py_target = f"cantera_python.vcs.{src_fn}"

        # Assign callers and dependencies
        deps = []
        callers = []
        
        if "solve_TP" in src_fn or "vcs_solve_TP" in src_fn:
            callers = ["cantera_python.vcs.MultiPhase.equilibrate", "cantera_python.vcs.vcs_equilibrate"]
            deps = [
                "cantera_python.vcs.VCS_SOLVE.vcs_evalSS_TP",
                "cantera_python.vcs.VCS_SOLVE.vcs_prep",
                "cantera_python.vcs.VCS_SOLVE.solve_tp_component_calc",
                "cantera_python.vcs.VCS_SOLVE.solve_tp_inner",
                "cantera_python.vcs.VCS_SOLVE.solve_tp_equilib_check",
                "cantera_python.vcs.VCS_SOLVE.solve_tp_elem_abund_check",
                "cantera_python.vcs.VCS_SOLVE.vcs_recheck_deleted",
                "cantera_python.vcs.VCS_SOLVE.vcs_add_all_deleted",
            ]
        elif "solve_tp_inner" in src_fn:
            callers = ["cantera_python.vcs.VCS_SOLVE.solve_TP"]
            deps = [
                "cantera_python.vcs.VCS_SOLVE.vcs_RxnStepSizes",
                "cantera_python.vcs.VCS_SOLVE.vcs_apply_minor_trial",
                "cantera_python.vcs.VCS_SOLVE.vcs_apply_major_trial",
                "cantera_python.vcs.VCS_SOLVE.vcs_apply_zeroed_trial",
                "cantera_python.vcs.VCS_SOLVE.vcs_glob_step_damp",
                "cantera_python.vcs.VCS_SOLVE.vcs_updateVP",
            ]
        elif "solve_tp_component_calc" in src_fn:
            callers = ["cantera_python.vcs.VCS_SOLVE.solve_TP", "cantera_python.vcs.VCS_SOLVE.solve_tp_inner"]
            deps = [
                "cantera_python.vcs.VCS_SOLVE.vcs_basopt",
                "cantera_python.vcs.VCS_SOLVE.vcs_updateVP",
                "cantera_python.vcs.VCS_SOLVE.vcs_deltag",
                "cantera_python.vcs.VCS_SOLVE.vcs_evaluate_speciesType",
                "cantera_python.vcs.VCS_SOLVE.vcs_elabcheck",
                "cantera_python.vcs.VCS_SOLVE.vcs_elcorr",
            ]

        entry = {
            "stable_symbol_id": sid,
            "source_file": src_file,
            "class": src_cls,
            "method": src_fn,
            "source_lines": s.get("line_range", []),
            "python_target": py_target,
            "dependency_symbols": deps,
            "caller_symbols": callers,
            "current_status": status,
        }
        call_graph[sid] = entry

        if not is_non_runtime:
            vcs_runtime_symbols_total += 1
            closure_set[sid] = entry
            if status in ["EQUIVALENT_PROVEN", "NATIVE_VALIDATED"]:
                vcs_already_final += 1
            elif status in ["PARTIAL", "IMPLEMENTED_UNVALIDATED", "SOURCE_VALIDATED"]:
                vcs_partial += 1
            elif status == "NOT_PORTED":
                vcs_not_ported += 1

    # Write output artifacts
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open(audit_dir / "vcs_complete_runtime_call_graph.json", "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    with open(audit_dir / "vcs_closure_symbol_set.json", "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)

    print("=== VCS CLOSURE SYMBOL INVENTORY BASELINE ===")
    print(f"VCS_RUNTIME_SYMBOLS_TOTAL = {vcs_runtime_symbols_total}")
    print(f"VCS_ALREADY_FINAL = {vcs_already_final}")
    print(f"VCS_PARTIAL = {vcs_partial}")
    print(f"VCS_NOT_PORTED = {vcs_not_ported}")

if __name__ == "__main__":
    build_call_graph_and_closure()
