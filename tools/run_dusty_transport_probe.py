"""Emit deterministic DustyGas transport observables for native/Python comparison.

Run from the canonical repository with either the pinned Cantera environment or
the standalone cantera_python package. The JSON output intentionally records
the complete coefficient matrix for independent comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MECHANISM = ROOT.parent / "cantera-original" / "data" / "h2o2.yaml"


def _load_api(backend: str):
    if backend == "python":
        sys.path.insert(0, str(ROOT))
        import cantera_python as ct
    else:
        import cantera as ct
    return ct


def _case(case_id: str, temperature: float, pressure: float,
          composition: str, porosity: float, tortuosity: float,
          pore_radius: float, particle_diameter: float,
          permeability: float) -> dict:
    return {
        "case_id": case_id,
        "T": temperature,
        "P": pressure,
        "X": composition,
        "porosity": porosity,
        "tortuosity": tortuosity,
        "mean_pore_radius": pore_radius,
        "mean_particle_diameter": particle_diameter,
        "permeability": permeability,
    }


CASES = (
    _case("ambient_hydrogen_rich", 300.0, 101325.0,
          "H2:2.0, O2:1.0, N2:3.76", 0.2, 0.3, 1e-4, 5e-4, 0.0),
    _case("reference_fixture", 500.0, 101325.0,
          "O2:2.0, H2:1.0, H2O:1.0", 0.2, 0.3, 1e-4, 5e-4, 0.0),
    _case("hot_pressurized_dilute", 1400.0, 8.0 * 101325.0,
          "H2:0.15, H:0.01, O2:0.12, H2O:0.20, N2:0.52", 0.37, 0.61,
          4e-5, 7e-4, 2e-12),
    _case("trace_species_low_porosity", 850.0, 0.4 * 101325.0,
          "H2:0.001, HO2:0.002, H2O2:0.003, AR:0.994", 0.08, 0.92,
          2e-5, 3e-4, 1e-10),
)


def run(backend: str) -> dict:
    ct = _load_api(backend)
    records = []
    for case in CASES:
        gas = ct.DustyGas(str(MECHANISM))
        gas.TPX = case["T"], case["P"], case["X"]
        gas.porosity = case["porosity"]
        gas.tortuosity = case["tortuosity"]
        gas.mean_pore_radius = case["mean_pore_radius"]
        gas.mean_particle_diameter = case["mean_particle_diameter"]
        gas.permeability = case["permeability"]
        diffusivities = np.asarray(gas.multi_diff_coeffs, dtype=float)
        if diffusivities.shape != (gas.n_species, gas.n_species):
            raise RuntimeError(f"unexpected diffusion matrix shape: {diffusivities.shape}")
        if not np.all(np.isfinite(diffusivities)):
            raise RuntimeError(f"non-finite diffusion coefficient in {case['case_id']}")
        conductivity = float(gas.thermal_conductivity)
        if not math.isfinite(conductivity):
            raise RuntimeError(f"non-finite thermal conductivity in {case['case_id']}")
        t1, rho1, y1 = float(gas.T), float(gas.density), np.asarray(gas.Y).copy()
        gas.TPX = (
            case["T"] * 1.02,
            case["P"] * 0.98,
            "H2:0.41, H:0.01, O:0.004, O2:0.15, OH:0.03, "
            "H2O:0.22, HO2:0.005, H2O2:0.015, AR:0.035, N2:0.12",
        )
        molar_fluxes = np.asarray(gas.molar_fluxes(
            t1, float(gas.T), rho1, float(gas.density), y1,
            np.asarray(gas.Y), 1e-3), dtype=float)
        if molar_fluxes.shape != (gas.n_species,) or not np.all(np.isfinite(molar_fluxes)):
            raise RuntimeError(f"invalid molar flux vector in {case['case_id']}")
        records.append({
            **case,
            "species_names": list(gas.species_names),
            "multi_diff_coeffs": diffusivities.tolist(),
            "thermal_conductivity": conductivity,
            "molar_fluxes": molar_fluxes.tolist(),
        })
    return {
        "schema_version": 1,
        "backend": backend,
        "package_version": getattr(ct, "__version__", "unknown"),
        "mechanism_sha256": __import__("hashlib").sha256(MECHANISM.read_bytes()).hexdigest(),
        "cases": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("native", "python"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.backend} DustyGas probe: {args.output} ({len(payload['cases'])} cases)")


if __name__ == "__main__":
    main()
