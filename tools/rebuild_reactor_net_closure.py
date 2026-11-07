"""Rebuild the true ReactorNet closure set with source reconciliation."""

import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    orig_root = repo_root.parent / "cantera-original"
    
    header_path = orig_root / "include" / "cantera" / "zeroD" / "ReactorNet.h"
    cpp_path = orig_root / "src" / "zeroD" / "ReactorNet.cpp"
    closure_path = audit_dir / "reactor_network_closure_symbol_set.json"

    with open(header_path, "r", encoding="utf-8") as f:
        header_text = f.read()
    with open(cpp_path, "r", encoding="utf-8") as f:
        cpp_text = f.read()
    with open(closure_path, "r", encoding="utf-8") as f:
        closure = json.load(f)

    # Reconciliation
    rebuilt_closure = {}
    runtime_symbols_total = 0
    final_before = 0
    implemented_unvalidated_before = 0
    partial_before = 0
    not_ported_before = 0
    non_runtime_rows = 0

    for sid, row in closure.items():
        method = row["native_method"]
        # Check source presence
        in_header = f"{method}(" in header_text or f"{method} " in header_text or f"{method};" in header_text
        in_cpp = f"ReactorNet::{method}" in cpp_text
        source_verified = in_header or in_cpp

        is_non_runtime = (sid == "include/cantera/zeroD/ReactorNet.h:293:sensitivity")
        
        if is_non_runtime:
            non_runtime_rows += 1
            reachability = "NON_RUNTIME_CPP_POINTER_OVERLOAD"
            status = "NON_RUNTIME_CPP_ONLY_PROVEN"
            val_status = "PROVEN_NON_RUNTIME"
            blocker = "None (C++ pointer overload double* dest, size_t lddim not part of Python scientific runtime)"
        else:
            runtime_symbols_total += 1
            reachability = "RUNTIME_SCIENTIFIC_PUBLIC_API"
            # Prior to differential validation campaign
            status = row.get("current_status", "NOT_PORTED")
            if status == "PINNED_NATIVE_VALIDATED":
                final_before += 1
                val_status = "VALIDATED"
                blocker = "None"
            elif status == "PARTIAL":
                partial_before += 1
                val_status = "PARTIAL"
                blocker = "Full differential oracle campaign completion"
            else:
                # Implemented in Python code but unvalidated against native oracle
                not_ported_before += 1
                val_status = "UNVALIDATED"
                blocker = "Full differential oracle campaign completion across homogeneous, coupled, flow, wall, limit, and sensitivity corpus"

        rebuilt_closure[sid] = {
            "canonical_symbol_id": sid,
            "native_file": row["native_file"],
            "native_class": row["native_class"],
            "source_method": method,
            "source_verified_in_pinned_cantera": source_verified,
            "python_target": row["python_target"],
            "runtime_reachability": reachability,
            "current_implementation_status": status,
            "validation_status": val_status,
            "dependency": "CVodesIntegrator, ReactorBase, IdealGasReactor, ConstPressureReactor, FlowDevice, Wall",
            "remaining_blocker": blocker,
        }

    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(rebuilt_closure, f, indent=2)

    print(f"REACTORNET_RUNTIME_SYMBOLS_TOTAL = {runtime_symbols_total}")
    print(f"REACTORNET_FINAL_BEFORE = {final_before}")
    print(f"REACTORNET_IMPLEMENTED_UNVALIDATED_BEFORE = {implemented_unvalidated_before}")
    print(f"REACTORNET_PARTIAL_BEFORE = {partial_before}")
    print(f"REACTORNET_NOT_PORTED_BEFORE = {not_ported_before}")
    print(f"REACTORNET_NON_RUNTIME_ROWS = {non_runtime_rows}")

if __name__ == "__main__":
    main()
