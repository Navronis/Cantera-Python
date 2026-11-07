"""Build Mixture Transport call graph and closure symbol set."""

import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    inventory_path = repo_root / "provenance" / "canonical_5784_symbol_inventory.jsonl"
    
    with open(inventory_path, "r", encoding="utf-8") as f:
        all_symbols = [json.loads(line) for line in f]
        
    symbol_map = {s["stable_symbol_id"]: s for s in all_symbols}
    
    # Identify relevant symbols for Mixture Transport:
    # 1. All symbols in MixTransport.h and MixTransport.cpp
    # 2. All symbols in GasTransport.h and GasTransport.cpp
    # 3. All symbols in Transport.h and Transport.cpp
    # 4. Relevant collision integral symbols in MMCollisionInt.h / MMCollisionInt.cpp
    # 5. TransportData.h symbols needed for gas transport
    
    trans_files = {
        'include/cantera/transport/MixTransport.h',
        'src/transport/MixTransport.cpp',
        'include/cantera/transport/GasTransport.h',
        'src/transport/GasTransport.cpp',
        'include/cantera/transport/Transport.h',
        'src/transport/Transport.cpp',
        'src/transport/MMCollisionInt.h',
        'src/transport/MMCollisionInt.cpp',
        'include/cantera/transport/TransportData.h',
    }
    
    selected_symbols = {}
    for s in all_symbols:
        f = s.get("source_file", "")
        cls = s.get("class") or ""
        fn = s.get("function_or_method") or ""
        sid = s["stable_symbol_id"]
        
        # Include if in trans_files or explicitly MixTransport/GasTransport/Transport/MMCollisionInt
        if f in trans_files or cls in ("MixTransport", "GasTransport", "Transport", "MMCollisionInt"):
            selected_symbols[sid] = s

    print(f"Total candidate symbols selected: {len(selected_symbols)}")

    call_graph = {}
    closure_set = {}
    
    mixtran_runtime_symbols_total = 0
    mixtran_already_final = 0
    mixtran_partial = 0
    mixtran_not_ported = 0
    
    for sid, s in sorted(selected_symbols.items()):
        status = s.get("current_status", "NOT_PORTED")
        is_non_runtime = (status == "NON_RUNTIME_CPP_ONLY_PROVEN")
        src_file = s.get("source_file", "")
        src_cls = s.get("class") or ""
        src_fn = s.get("function_or_method") or ""
        
        # Determine python target
        py_target = None
        if src_cls == "MixTransport":
            py_target = f"cantera_python.transport.MixTransport.{src_fn}"
        elif src_cls == "GasTransport":
            py_target = f"cantera_python.transport.GasTransport.{src_fn}"
        elif src_cls == "Transport":
            py_target = f"cantera_python.transport.Transport.{src_fn}"
        elif src_cls == "MMCollisionInt":
            py_target = f"cantera_python.transport.MMCollisionInt.{src_fn}"
        elif src_cls == "GasTransportData":
            py_target = f"cantera_python.transport.GasTransportData.{src_fn}"
        elif "TransportData" in src_cls:
            py_target = f"cantera_python.transport.TransportData.{src_fn}"
        else:
            py_target = f"cantera_python.transport.{src_fn}"
            
        # Assign callers and dependencies based on MixTransport runtime graph
        deps = []
        callers = []
        
        if src_fn == "viscosity":
            callers = ["cantera_python.solution.Solution.viscosity"]
            deps = [
                "cantera_python.transport.GasTransport.update_T",
                "cantera_python.transport.MixTransport.update_C",
                "cantera_python.transport.GasTransport.updateViscosity_T",
            ]
        elif src_fn == "updateViscosity_T":
            callers = ["cantera_python.transport.GasTransport.viscosity", "cantera_python.transport.MixTransport.getThermalDiffCoeffs"]
            deps = ["cantera_python.transport.GasTransport.updateSpeciesViscosities"]
        elif src_fn == "updateSpeciesViscosities":
            callers = ["cantera_python.transport.GasTransport.updateViscosity_T", "cantera_python.transport.GasTransport.getSpeciesViscosities"]
            deps = ["cantera_python.transport.GasTransport.update_T"]
        elif src_fn == "thermalConductivity":
            callers = ["cantera_python.solution.Solution.thermal_conductivity"]
            deps = [
                "cantera_python.transport.MixTransport.update_T",
                "cantera_python.transport.MixTransport.update_C",
                "cantera_python.transport.MixTransport.updateCond_T",
            ]
        elif src_fn == "updateCond_T":
            callers = ["cantera_python.transport.MixTransport.thermalConductivity"]
            deps = ["cantera_python.transport.MixTransport.update_T"]
        elif src_fn in ("getMixDiffCoeffs", "mix_diff_coeffs"):
            callers = [
                "cantera_python.solution.Solution.mix_diff_coeffs",
                "cantera_python.transport.MixTransport.getMobilities",
                "cantera_python.transport.MixTransport.getThermalDiffCoeffs",
                "cantera_python.transport.MixTransport.getSpeciesFluxes",
            ]
            deps = [
                "cantera_python.transport.GasTransport.update_T",
                "cantera_python.transport.MixTransport.update_C",
                "cantera_python.transport.GasTransport.updateDiff_T",
            ]
        elif src_fn in ("getBinaryDiffCoeffs", "binary_diffusion_coefficients"):
            callers = [
                "cantera_python.solution.Solution.binary_diff_coeffs",
                "cantera_python.transport.MixTransport.getMixDiffCoeffs",
            ]
            deps = [
                "cantera_python.transport.GasTransport.update_T",
                "cantera_python.transport.GasTransport.updateDiff_T",
            ]
        elif src_fn == "updateDiff_T":
            callers = [
                "cantera_python.transport.GasTransport.getBinaryDiffCoeffs",
                "cantera_python.transport.GasTransport.getMixDiffCoeffs",
                "cantera_python.transport.MixTransport.getThermalDiffCoeffs",
            ]
            deps = ["cantera_python.transport.GasTransport.update_T"]
        elif src_fn == "init":
            callers = ["cantera_python.transport.TransportFactory.newTransport"]
            deps = [
                "cantera_python.transport.GasTransport.setupCollisionParameters",
                "cantera_python.transport.GasTransport.setupCollisionIntegral",
            ]
        elif src_fn == "setupCollisionIntegral":
            callers = ["cantera_python.transport.GasTransport.init"]
            deps = [
                "cantera_python.transport.GasTransport.fitCollisionIntegrals",
                "cantera_python.transport.GasTransport.fitProperties",
            ]
        elif src_fn == "fitProperties":
            callers = ["cantera_python.transport.GasTransport.setupCollisionIntegral"]
            deps = ["cantera_python.transport.GasTransport.fitDiffCoeffs"]
            
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
            mixtran_runtime_symbols_total += 1
            closure_set[sid] = entry
            if status in ["EQUIVALENT_PROVEN", "NATIVE_VALIDATED", "PINNED_NATIVE_VALIDATED", "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED"]:
                mixtran_already_final += 1
            elif status in ["PARTIAL", "IMPLEMENTED_UNVALIDATED", "SOURCE_VALIDATED"]:
                mixtran_partial += 1
            elif status == "NOT_PORTED":
                mixtran_not_ported += 1

    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    with open(audit_dir / "mixture_transport_runtime_call_graph.json", "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
        
    with open(audit_dir / "mixture_transport_closure_symbol_set.json", "w", encoding="utf-8") as f:
        json.dump(closure_set, f, indent=2)
        
    print("=== MIXTURE TRANSPORT CLOSURE SYMBOL INVENTORY BASELINE ===")
    print(f"MIXTRAN_RUNTIME_SYMBOLS_TOTAL = {mixtran_runtime_symbols_total}")
    print(f"MIXTRAN_ALREADY_FINAL = {mixtran_already_final}")
    print(f"MIXTRAN_PARTIAL = {mixtran_partial}")
    print(f"MIXTRAN_NOT_PORTED = {mixtran_not_ported}")

if __name__ == "__main__":
    main()
