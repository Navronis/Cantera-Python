"""Supplementary DustyGas comparison against installed released Cantera.

This utility is validation-only. ``cantera_python`` never imports native
Cantera at runtime, and this result is not a substitute for the pinned oracle.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cantera as native
import cantera_python as python


MECHANISM = ROOT.parent / "cantera-original" / "data" / "h2o2.yaml"
OUTPUT = ROOT / "provenance" / "dusty_gas_released_comparison.json"


def configure(gas):
    pressure = getattr(native, "one_atm", python.OneAtm)
    gas.TPX = 500.0, pressure, "O2:2.0, H2:1.0, H2O:1.0"
    gas.porosity = 0.2
    gas.tortuosity = 0.3
    gas.mean_pore_radius = 1e-4
    gas.mean_particle_diameter = 5e-4


def state(gas):
    return float(gas.T), float(gas.density), np.asarray(gas.Y)


def main():
    reference = native.DustyGas(str(MECHANISM))
    candidate = python.DustyGas(MECHANISM)
    configure(reference)
    configure(candidate)

    d_reference = np.asarray(reference.multi_diff_coeffs)
    d_candidate = np.asarray(candidate.multi_diff_coeffs)
    t1r, rho1r, y1r = state(reference)
    t1p, rho1p, y1p = state(candidate)
    reference.TPX = 500.0, native.one_atm, "O2:2.0, H2:1.001, H2O:0.999"
    candidate.TPX = 500.0, python.OneAtm, "O2:2.0, H2:1.001, H2O:0.999"
    t2r, rho2r, y2r = state(reference)
    t2p, rho2p, y2p = state(candidate)
    f_reference = reference.molar_fluxes(
        t1r, t2r, rho1r, rho2r, y1r, y2r, 1e-4)
    f_candidate = candidate.molar_fluxes(
        t1p, t2p, rho1p, rho2p, y1p, y2p, 1e-4)

    significant_d = np.abs(d_reference) > 1e-18
    significant_f = np.abs(f_reference) > 1e-12
    payload = {
        "schema_version": 1,
        "reference": f"released Cantera {native.__version__}",
        "pinned_source_commit": python.UPSTREAM_COMMIT,
        "authoritative_for_pinned_commit": False,
        "runtime_dependency": False,
        "mechanism": str(MECHANISM),
        "state": {"temperature_K": 500.0, "pressure_Pa": python.OneAtm},
        "multi_diff_coeffs": {
            "shape": list(d_reference.shape),
            "max_absolute_error": float(np.max(np.abs(d_candidate - d_reference))),
            "max_relative_error_significant": float(np.max(
                np.abs((d_candidate[significant_d] - d_reference[significant_d])
                       / d_reference[significant_d]))),
        },
        "molar_fluxes": {
            "max_absolute_error": float(np.max(np.abs(f_candidate - f_reference))),
            "max_relative_error_significant": float(np.max(
                np.abs((f_candidate[significant_f] - f_reference[significant_f])
                       / f_reference[significant_f]))),
            "reference": f_reference.tolist(),
            "candidate": f_candidate.tolist(),
        },
        "interpretation": (
            "The dusty-gas matrix and Darcy/Knudsen equations are ported from "
            "the pinned source. Remaining differences include the subordinate "
            "gas transport implementation and released-versus-pinned source drift."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
