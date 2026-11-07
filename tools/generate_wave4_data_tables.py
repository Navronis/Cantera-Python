#!/usr/bin/env python3
"""Generate provenance/solver_wave4_data_tables.jsonl."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "provenance" / "solver_wave4_data_tables.jsonl"

tables = [
    {
        "table_id": "TBL_MM_DELTA_GRID",
        "name": "Monchick & Mason Reduced Dipole Moment Delta* Grid",
        "source_reference": "MMCollisionInt.cpp:18-19",
        "dimensions": [8],
        "values": [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5],
        "description": "Grid of reduced dipole moment parameters delta* for tabulated Stockmayer collision integrals."
    },
    {
        "table_id": "TBL_MM_TSTAR22_GRID",
        "name": "Monchick & Mason T* Grid for Omega(2,2)*",
        "source_reference": "MMCollisionInt.cpp:34-39",
        "dimensions": [37],
        "values": [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0
        ],
        "description": "Reduced temperature grid T* = k_B * T / epsilon for viscosity collision integral Omega(2,2)*."
    },
    {
        "table_id": "TBL_MM_TSTAR_RATIOS_GRID",
        "name": "Monchick & Mason T* Grid for Collision Ratios A*, B*, C*",
        "source_reference": "MMCollisionInt.cpp:82-87",
        "dimensions": [39],
        "values": [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
            1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0,
            18.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 75.0, 100.0, 500.0
        ],
        "description": "Extended reduced temperature grid for ratios A* = Omega(2,2)*/Omega(1,1)*, B*, and C*."
    },
    {
        "table_id": "TBL_MM_OMEGA22",
        "name": "Monchick & Mason Omega(2,2)* Tabulated Values",
        "source_reference": "MMCollisionInt.cpp:41-79",
        "dimensions": [37, 8],
        "entry_count": 296,
        "description": "2D table of Stockmayer viscosity collision integrals Omega(2,2)*(T*, delta*)."
    },
    {
        "table_id": "TBL_MM_ASTAR",
        "name": "Monchick & Mason Collision Integral Ratio A* Tabulated Values",
        "source_reference": "MMCollisionInt.cpp:89-129",
        "dimensions": [39, 8],
        "entry_count": 312,
        "description": "2D table of collision integral ratio A* = Omega(2,2)* / Omega(1,1)*."
    },
    {
        "table_id": "TBL_MM_BSTAR",
        "name": "Monchick & Mason Collision Integral Ratio B* Tabulated Values",
        "source_reference": "MMCollisionInt.cpp:131-171",
        "dimensions": [39, 8],
        "entry_count": 312,
        "description": "2D table of collision integral ratio B* = [5 * Omega(1,2)* - 4 * Omega(1,3)*] / Omega(1,1)*."
    },
    {
        "table_id": "TBL_MM_CSTAR",
        "name": "Monchick & Mason Collision Integral Ratio C* Tabulated Values",
        "source_reference": "MMCollisionInt.cpp:173-213",
        "dimensions": [39, 8],
        "entry_count": 312,
        "description": "2D table of collision integral ratio C* = Omega(1,2)* / Omega(1,1)*."
    },
    {
        "table_id": "TBL_PARKER_PARAMETERS",
        "name": "Parker Rotational Relaxation Collision Constant Factors",
        "source_reference": "MultiTransport.cpp:29-31",
        "constants": {
            "c1": 2.784163996767669,
            "c2": 4.4674011002723395,
            "c3": 5.568327993535338
        },
        "description": "Universal analytical constants for Parker temperature correction polynomial: c1 = 0.5 * pi^1.5, c2 = 0.25 * pi^2 + 2, c3 = pi^1.5."
    },
    {
        "table_id": "TBL_TRANSPORT_DATA_SCHEMA",
        "name": "Cantera Species Gas Transport Data Parameter Definitions",
        "source_reference": "TransportData.h:35-72",
        "fields": {
            "geometry": "Molecular shape: 'atom' (monatomic, crot=0), 'linear' (crot=1.0), or 'nonlinear' (crot=1.5)",
            "diameter": "Lennard-Jones collision diameter sigma [Angstrom = 1e-10 m]",
            "well_depth": "Lennard-Jones potential well depth epsilon / k_B [Kelvin]",
            "dipole": "Permanent electric dipole moment mu [Debye = 1e-21 / c C*m]",
            "polarizability": "Mean electric polarizability alpha [Angstrom^3 = 1e-30 m^3]",
            "rotational_relaxation": "Rotational collision relaxation number Z_rot at 298 K [dimensionless]"
        },
        "description": "Physical parameters loaded from mechanism YAML definition for kinetic theory calculations."
    }
]

OUT_FILE.write_text("\n".join(json.dumps(tbl) for tbl in tables) + "\n", encoding="utf-8")
print(f"Generated {OUT_FILE} with {len(tables)} data table definitions.")
