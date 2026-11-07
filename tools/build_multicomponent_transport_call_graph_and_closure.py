"""Build MultiTransport call graph and closure symbol set from canonical inventory."""

import json
from collections import Counter
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    inventory_path = repo_root / "provenance" / "canonical_5784_symbol_inventory.jsonl"
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    with open(inventory_path, "r", encoding="utf-8") as f:
        all_symbols = [json.loads(line) for line in f]

    multi_symbols = [
        s for s in all_symbols
        if s.get("class") == "MultiTransport" or "MultiTransport" in s.get("source_file", "")
    ]

    print(f"Total MultiTransport candidate symbols: {len(multi_symbols)}")

    call_graph = {}
    closure_set = {}

    multitran_runtime_symbols_total = 0
    multitran_equivalent_before = 0
    multitran_native_validated_before = 0
    multitran_implemented_unvalidated_before = 0
    multitran_partial_before = 0
    multitran_not_ported_before = 0

    for s in sorted(multi_symbols, key=lambda x: x["stable_symbol_id"]):
        sid = s["stable_symbol_id"]
        status = s.get("current_status", "NOT_PORTED")
        is_non_runtime = (status == "NON_RUNTIME_CPP_ONLY_PROVEN")
        src_file = s.get("source_file", "")
        src_cls = s.get("class") or ""
        src_fn = s.get("function_or_method") or ""

        # Determine python target
        py_target = None
        if src_cls == "MultiTransport":
            py_target = f"cantera_python.transport.MultiTransport.{src_fn}"
        else:
            py_target = f"cantera_python.transport.{src_fn}"

        # Assign callers and dependencies based on MultiTransport runtime graph
        deps = []
        callers = []

        if src_fn in ("getMultiDiffCoeffs", "multicomponent_diffusion_coefficients", "multi_diff_coeffs"):
            callers = [
                "cantera_python.solution.Solution.multi_diff_coeffs",
                "cantera_python.transport.MultiTransport.getSpeciesFluxes",
            ]
            deps = [
                "cantera_python.transport.MultiTransport.update_T",
                "cantera_python.transport.MultiTransport.update_C",
                "cantera_python.transport.MultiTransport.eval_L0000",
                "cantera_python.transport.MultiTransport.solveLMatrixEquation",
            ]
        elif src_fn in ("getThermalDiffCoeffs", "thermal_diff_coeffs"):
            callers = [
                "cantera_python.solution.Solution.thermal_diff_coeffs",
                "cantera_python.transport.MultiTransport.getSpeciesFluxes",
            ]
            deps = [
                "cantera_python.transport.MultiTransport.update_T",
                "cantera_python.transport.MultiTransport.update_C",
                "cantera_python.transport.MultiTransport.updateThermal_T",
                "cantera_python.transport.MultiTransport.solveLMatrixEquation",
            ]
        elif src_fn in ("thermalConductivity", "thermal_conductivity"):
            callers = ["cantera_python.solution.Solution.thermal_conductivity"]
            deps = [
                "cantera_python.transport.MultiTransport.update_T",
                "cantera_python.transport.MultiTransport.update_C",
                "cantera_python.transport.MultiTransport.updateThermal_T",
                "cantera_python.transport.MultiTransport.solveLMatrixEquation",
            ]
        elif src_fn == "solveLMatrixEquation":
            callers = [
                "cantera_python.transport.MultiTransport.getMultiDiffCoeffs",
                "cantera_python.transport.MultiTransport.getThermalDiffCoeffs",
                "cantera_python.transport.MultiTransport.thermalConductivity",
            ]
            deps = [
                "cantera_python.transport.MultiTransport.eval_L0000",
                "cantera_python.transport.MultiTransport.eval_L0010",
                "cantera_python.transport.MultiTransport.eval_L1000",
                "cantera_python.transport.MultiTransport.eval_L1010",
                "cantera_python.transport.MultiTransport.eval_L1001",
                "cantera_python.transport.MultiTransport.eval_L0100",
                "cantera_python.transport.MultiTransport.eval_L0001",
                "cantera_python.transport.MultiTransport.eval_L0110",
                "cantera_python.transport.MultiTransport.eval_L0101",
            ]
        elif src_fn.startswith("eval_L"):
            callers = ["cantera_python.transport.MultiTransport.solveLMatrixEquation"]
            deps = [
                "cantera_python.transport.MultiTransport.updateThermal_T",
                "cantera_python.transport.MultiTransport.hasInternalModes",
            ]
        elif src_fn in ("getSpeciesFluxes", "getMassFluxes", "getMolarFluxes"):
            callers = ["cantera_python.solution.Solution.get_species_fluxes"]
            deps = [
                "cantera_python.transport.MultiTransport.getMultiDiffCoeffs",
                "cantera_python.transport.MultiTransport.getThermalDiffCoeffs",
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
            "tests": ["tests/test_multicomponent_transport.py"],
            "current_evidence": s.get("evidence"),
            "remaining_validation_requirement": "Pinned native oracle differential validation across full matrices, thermal diffusion, and fluxes."
        }
        call_graph[sid] = entry

        if not is_non_runtime:
            multitran_runtime_symbols_total += 1
            closure_set[sid] = entry
            if status in ("EQUIVALENT_PROVEN", "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED"):
                multitran_equivalent_before += 1
            elif status in ("NATIVE_VALIDATED", "PINNED_NATIVE_VALIDATED"):
                multitran_native_validated_before += 1
            elif status == "IMPLEMENTED_UNVALIDATED":
                multitran_implemented_unvalidated_before += 1
            elif status == "PARTIAL":
                multitran_partial_before += 1
            elif status == "NOT_PORTED":
                multitran_not_ported_before += 1

    with open(audit_dir / "multicomponent_transport_runtime_call_graph.json", "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    with open(audit_dir / "multicomponent_transport_closure_symbol_set.json", "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)

    print("=== MULTICOMPONENT TRANSPORT CLOSURE SYMBOL INVENTORY BASELINE ===")
    print(f"MULTITRAN_RUNTIME_SYMBOLS_TOTAL = {multitran_runtime_symbols_total}")
    print(f"MULTITRAN_EQUIVALENT_BEFORE = {multitran_equivalent_before}")
    print(f"MULTITRAN_NATIVE_VALIDATED_BEFORE = {multitran_native_validated_before}")
    print(f"MULTITRAN_IMPLEMENTED_UNVALIDATED_BEFORE = {multitran_implemented_unvalidated_before}")
    print(f"MULTITRAN_PARTIAL_BEFORE = {multitran_partial_before}")
    print(f"MULTITRAN_NOT_PORTED_BEFORE = {multitran_not_ported_before}")

if __name__ == "__main__":
    main()
