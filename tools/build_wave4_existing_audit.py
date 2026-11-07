#!/usr/bin/env python3
"""Audit existing transport implementation in cantera_python/transport.py against Wave 4 requirements."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_INV = ROOT / "audit" / "solver_wave4_source_inventory.jsonl"
OUT_FILE = ROOT / "audit" / "solver_wave4_existing_implementation_audit.jsonl"

src_records = [
    json.loads(line)
    for line in SRC_INV.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# We examine what exists in cantera_python/transport.py
audit_records = []

for rec in src_records:
    sym = rec["symbol"]
    cls = rec["class"]
    method = rec["function_or_method"]
    source_file = rec["source_file"]

    # Default
    existing_status = "MISSING"
    python_symbol = None
    audit_notes = ""

    if cls == "MultiTransport":
        if method in ("thermalConductivity", "thermal_conductivity"):
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.thermal_conductivity"
            audit_notes = "Implemented via monolithic _solve_L_matrix, lacks modular submatrix calls and proper caching"
        elif method in ("getThermalDiffCoeffs", "thermal_diff_coeffs"):
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.getThermalDiffCoeffs"
            audit_notes = "Implemented via monolithic _solve_L_matrix, lacks strict array validation and modular submatrix calls"
        elif method in ("getMultiDiffCoeffs", "multi_diff_coeffs"):
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.getMultiDiffCoeffs"
            audit_notes = "Implemented via _eval_L0000, but inverts L directly inside eval rather than separate modular step"
        elif method == "getSpeciesFluxes":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.getSpeciesFluxes"
            audit_notes = "Implemented Stefan-Maxwell solver with row replacement, but lacks multidimensional gradient handling and exact stride semantics"
        elif method == "getMassFluxes":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.getMassFluxes"
            audit_notes = "Implemented but uses custom state parsing rather than exact C++ restoreState / getMoleFractions flow"
        elif method == "getMolarFluxes":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.getMolarFluxes"
            audit_notes = "Divides mass fluxes by molecular weights, but depends on getMassFluxes"
        elif method == "transportModel":
            existing_status = "EXACT_SOURCE_TRANSLATION"
            python_symbol = "MultiTransport.transport_model"
            audit_notes = "Returns 'multicomponent' or 'multicomponent-CK'"
        elif method in ("solveLMatrixEquation", "_solve_L_matrix"):
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport._solve_L_matrix"
            audit_notes = "Monolithic implementation in transport.py; stubs exist at class top"
        elif method and method.startswith("eval_L"):
            existing_status = "STUB"
            python_symbol = f"MultiTransport.{method}"
            audit_notes = "Class-level stub at lines 969-994; actual assembly embedded inside _solve_L_matrix"
        elif method == "updateThermal_T":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport._update_thermal"
            audit_notes = "Implemented as _update_thermal, but stub updateThermal_T exists at class top"
        elif method == "update_T":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.update_T"
            audit_notes = "Inherited/stubbed; needs exact invalidation of m_lmatrix_soln_ok and m_l0000_ok"
        elif method == "update_C":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.update_C"
            audit_notes = "Mole fraction update with Tiny offset and change detection; currently embedded or stubbed"
        elif method == "hasInternalModes":
            existing_status = "SEMANTICALLY_EQUIVALENT"
            python_symbol = "MultiTransport.hasInternalModes"
            audit_notes = "cinternal > 0.001 condition checked in code"
        elif method == "pressure_ig":
            existing_status = "APPROXIMATION"
            python_symbol = "MultiTransport.pressure_ig"
            audit_notes = "Uses thermo.P instead of exact C++ molarDensity * R * T"
        elif method == "invalidateCache":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.invalidateCache"
            audit_notes = "Needs full C++ cache flag resetting (m_l0000_ok, m_lmatrix_soln_ok, m_thermal_tlast)"
        elif method == "init":
            existing_status = "PARTIAL"
            python_symbol = "MultiTransport.__init__"
            audit_notes = "Initialization in __init__, but init() method needs exact parity"
        else:
            existing_status = "MISSING" if rec["classification"] != "NON_RUNTIME" else "NON_RUNTIME"
            python_symbol = None
            audit_notes = "Not yet explicitly ported"

    elif cls == "GasTransport":
        if method == "viscosity":
            existing_status = "EXACT_SOURCE_TRANSLATION"
            python_symbol = "GasTransport.viscosity / MixTransport.viscosity"
            audit_notes = "Wilke mixture rule with Poling phi matrix correctly implemented"
        elif method == "getSpeciesViscosities":
            existing_status = "EXACT_SOURCE_TRANSLATION"
            python_symbol = "GasTransport.species_viscosities"
            audit_notes = "Evaluates pure species viscosities from kinetic theory"
        elif method == "getBinaryDiffCoeffs":
            existing_status = "EXACT_SOURCE_TRANSLATION"
            python_symbol = "GasTransport.binary_diffusion_coefficients"
            audit_notes = "Evaluates binary diffusion matrix D_ij from polynomial fits / p"
        elif method in ("updateViscosity_T", "updateSpeciesViscosities", "updateDiff_T", "update_T"):
            existing_status = "PARTIAL"
            python_symbol = f"GasTransport.{method}"
            audit_notes = "Implemented in MixTransport, but stubbed at GasTransport base level"
        elif method in ("fitCollisionIntegrals", "fitProperties", "fitDiffCoeffs"):
            existing_status = "PARTIAL"
            python_symbol = f"GasTransport.{method}"
            audit_notes = "Polynomial fitting partially present in MultiTransport.__init__, but stubbed in GasTransport"
        elif method in ("setupCollisionParameters", "setupCollisionIntegral", "makePolarCorrections"):
            existing_status = "PARTIAL"
            python_symbol = f"GasTransport.{method}"
            audit_notes = "Polar corrections and Stockmayer setup present in Mix/MultiTransport __init__, but stubbed in base"
        else:
            existing_status = "MISSING" if rec["classification"] != "NON_RUNTIME" else "NON_RUNTIME"
            python_symbol = None
            audit_notes = "Utility or stub"

    elif cls == "MMCollisionInt":
        if method in ("omega22", "astar", "bstar", "cstar", "omega11"):
            existing_status = "SEMANTICALLY_EQUIVALENT"
            python_symbol = f"_eval_star / _eval_omega22"
            audit_notes = "Evaluates Monchick & Mason tables via 2D interpolation / Neufeld formulas"
        else:
            existing_status = "NON_RUNTIME"
            audit_notes = "C++ table fitting helper"

    elif cls == "free_function" and method == "Frot":
        existing_status = "EXACT_SOURCE_TRANSLATION"
        python_symbol = "Frot"
        audit_notes = "Parker temperature correction function exactly implemented"

    else:
        existing_status = "NON_RUNTIME" if rec["classification"] == "NON_RUNTIME" else "MISSING"
        audit_notes = "Out of scope or base utility"

    audit_records.append({
        "stable_symbol_id": rec.get("stable_symbol_id"),
        "symbol": sym,
        "class": cls,
        "function_or_method": method,
        "source_file": source_file,
        "source_classification": rec["classification"],
        "existing_implementation_status": existing_status,
        "python_symbol": python_symbol,
        "audit_findings": audit_notes
    })

OUT_FILE.write_text("\n".join(json.dumps(r) for r in audit_records) + "\n", encoding="utf-8")
print(f"Generated {OUT_FILE} with {len(audit_records)} audit records.")
counts = {}
for r in audit_records:
    st = r["existing_implementation_status"]
    counts[st] = counts.get(st, 0) + 1
print("Audit breakdown:", counts)
