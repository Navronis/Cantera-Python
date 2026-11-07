#!/usr/bin/env python3
"""Build a source-checked map of pinned TPX property-pair dispatch."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CPP = (ROOT.parent / "cantera-original/src/tpx/Sub.cpp").read_text(encoding="utf-8")
HEADER = (ROOT.parent / "cantera-original/include/cantera/tpx/Sub.h").read_text(encoding="utf-8")
PYTHON = (ROOT / "cantera_python/tpx.py").read_text(encoding="utf-8")

# Native enum code, reverse enum code, independent variables, solver route,
# and source-defined phase/bounds behavior. Numeric closure is tracked apart
# from dispatch reachability and is intentionally not inferred here.
PAIRS = [
    ("TV", 12, "VT", -12, "T, v", "set_T; set_v", "direct single-phase state", "T in [Tmin,Tmax]; v>0", "native CanteraError; Python TPXError"),
    ("HP", 34, "PH", -34, "h, P", "Lever(Pgiven,P,h); otherwise set_xy(H,P)", "Lever rule at saturation pressure; otherwise inverse solve", "pressure and EOS domain", "source solver/Lever exception; Python TPXError"),
    ("SP", 54, "PS", -54, "s, P", "Lever(Pgiven,P,s); otherwise set_xy(S,P)", "Lever rule at saturation pressure; otherwise inverse solve", "pressure and EOS domain", "source solver/Lever exception; Python TPXError"),
    ("PV", 42, "VP", -42, "P, v", "Lever(Pgiven,P,v); otherwise set_xy(P,V)", "Lever rule at saturation pressure; otherwise inverse solve", "pressure, positive volume, EOS domain", "source solver/Lever exception; Python TPXError"),
    ("TP", 14, "PT", -14, "T, P", "set_T; saturation check; select TX endpoint; set_xy(T,P)", "reject saturated mixture without quality; choose vapor below Ps and liquid above Ps", "T in [Tmin,Tmax]; pressure/EOS domain", "saturated-mixture or solver exception"),
    ("UV", 62, "VU", -62, "u, v", "set_xy(U,V)", "general inverse route; two-phase behavior is solver-defined", "positive volume and EOS domain", "source solver exception"),
    ("ST", 51, "TS", -51, "s, T", "Lever(Tgiven,T,s); otherwise set_xy(S,T)", "Lever rule at fixed saturation temperature; otherwise inverse solve", "T and EOS domain", "source solver/Lever exception"),
    ("SV", 52, "VS", -52, "s, v", "set_xy(S,V)", "general inverse route; two-phase behavior is solver-defined", "positive volume and EOS domain", "source solver exception"),
    ("UP", 64, "PU", -64, "u, P", "Lever(Pgiven,P,u); otherwise set_xy(U,P)", "Lever rule at saturation pressure; otherwise inverse solve", "pressure and EOS domain", "source solver/Lever exception"),
    ("VH", 23, "HV", -23, "v, h", "set_xy(V,H)", "general inverse route; two-phase behavior is solver-defined", "positive volume and EOS domain", "source solver exception"),
    ("TH", 13, "HT", -13, "T, h", "set_xy(T,H)", "general inverse route; two-phase behavior is solver-defined", "T and EOS domain", "source solver exception"),
    ("SH", 53, "HS", -53, "s, h", "set_xy(S,H)", "general inverse route; two-phase behavior is solver-defined", "EOS domain", "source solver exception"),
    ("PX", 47, "XP", -47, "P, quality", "set_T(Tmin); validate quality/pressure; Tsat; update_sat; lever density", "explicit two-phase state; quality endpoints are saturated liquid/vapor", "quality in [0,1]; Ps(Tmin)<=P<=Pcrit", "native validates after setting Tmin; Python mirrors that mutation order"),
    ("TX", 17, "XT", -17, "T, quality", "validate quality/T; set_T; update_sat; lever density", "explicit two-phase state; quality endpoints are saturated liquid/vapor", "quality in [0,1]; Tmin<=T<=Tcrit", "native validates before state mutation; Python mirrors order"),
]

enum_values = {name: int(value) for name, value in re.findall(r"\b([A-Z]{2})\s*=\s*(-?\d+)", HEADER)}
rows = []
for name, code, reverse, reverse_code, variables, route, phases, bounds, failure in PAIRS:
    assert enum_values.get(name) == code, (name, enum_values.get(name), code)
    assert enum_values.get(reverse) == reverse_code, (reverse, enum_values.get(reverse), reverse_code)
    assert f"case PropertyPair::{name}:" in CPP, name
    assert f"PropertyPair.{name}" in PYTHON, name
    assert f"PropertyPair.{reverse}: PropertyPair.{name}" in PYTHON, reverse
    rows.append({
        "native_enum": name,
        "native_code": code,
        "reverse_enum": reverse,
        "reverse_code": reverse_code,
        "independent_variables": variables,
        "python_dispatch": f"cantera_python/tpx.py:Substance.Set ({name})",
        "source_solver_route": route,
        "phase_and_saturation": phases,
        "source_bounds": bounds,
        "failure_semantics": failure,
        "source_mapping": "include/cantera/tpx/Sub.h; src/tpx/Sub.cpp:352-469",
        "numerical_validation": "OPEN; dispatch reachability does not prove inverse-solver parity",
        "implementation_status": "SOURCE_MAPPED; NUMERICAL_CLOSURE_OPEN",
    })

out = ROOT / "audit/solver_wave3_property_pair_matrix.jsonl"
out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
print(f"Wrote {len(rows)} source-checked pair mappings to {out.relative_to(ROOT)}")
