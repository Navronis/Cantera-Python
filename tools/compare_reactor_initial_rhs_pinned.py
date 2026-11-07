"""Compare the initial ideal-gas reactor state and RHS to the pinned oracle.

Run from the pinned-oracle Python environment with this repository on
PYTHONPATH. The native network advances only 1e-12 seconds so CVODE exposes
its first derivative through ReactorNet.get_derivative(1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cantera as native
import cantera_python as python_port
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIN = "726522be4e2a13454d8415b7ef799d621f665cf3"
MECHANISM = ROOT.parent / "cantera-original" / "data" / "gri30.yaml"


def verify_oracle() -> dict:
    identity = json.loads((ROOT / "audit/native_oracle_identity.json").read_text())
    extension = Path(native._cantera.__file__).resolve()
    digest = hashlib.sha256(extension.read_bytes()).hexdigest()
    if identity.get("exact_commit") != PIN:
        raise RuntimeError("Native oracle identity does not match the pinned commit")
    if identity.get("native_extension_sha256") != digest:
        raise RuntimeError("Loaded native extension hash differs from the oracle record")
    return {
        "pinned_commit": PIN,
        "native_version": native.__version__,
        "native_extension_sha256": digest,
        "python_runtime_path": str(Path(python_port.__file__).resolve()),
    }


def make_phase(module):
    gas = module.Solution(str(MECHANISM))
    gas.TPX = 1500.0, 101325.0, "H2:2, O2:1, N2:4"
    reactor = module.IdealGasReactor(gas)
    network = module.ReactorNet([reactor])
    network.rtol = 1e-8
    network.atol = 1e-12
    return gas, reactor, network


def compare() -> dict:
    oracle = verify_oracle()

    native_gas, native_reactor, native_net = make_phase(native)
    native_net.advance(1e-12)
    native_state = np.asarray(native_reactor.get_state(), dtype=float)
    native_rhs = np.asarray(native_net.get_derivative(1), dtype=float)

    py_gas, py_reactor, py_net = make_phase(python_port)
    py_net.initialize()
    py_state = np.asarray(py_net._get_global_state(), dtype=float)
    phase_kinetics = {}
    for name in (
        "net_production_rates",
        "forward_rates_of_progress",
        "reverse_rates_of_progress",
    ):
        reference = np.asarray(getattr(native_gas, name), dtype=float)
        candidate = np.asarray(getattr(py_gas, name), dtype=float)
        if reference.shape != candidate.shape:
            raise RuntimeError(f"Kinetics vector shape mismatch for {name}")
        errors = np.abs(reference - candidate)
        scales = np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), 1e-300)
        largest = []
        for index in np.argsort(errors)[::-1][:15]:
            largest.append({
                "index": int(index),
                "native": float(reference[index]),
                "python": float(candidate[index]),
                "absolute_error": float(errors[index]),
                "relative_error": float(errors[index] / scales[index]),
            })
        phase_kinetics[name] = {
            "size": int(reference.size),
            "max_absolute_error": float(np.max(errors)),
            "max_relative_error": float(np.max(errors / scales)),
            "rms_relative_error": float(np.sqrt(np.mean((errors / scales) ** 2))),
            "largest_absolute_error_entries": largest,
        }
    py_rhs = np.asarray(py_net._global_rhs(0.0, py_state.copy()), dtype=float)

    if native_state.shape != py_state.shape or native_rhs.shape != py_rhs.shape:
        raise RuntimeError(
            "Native/Python state layout differs: "
            f"states {native_state.shape}/{py_state.shape}, "
            f"RHS {native_rhs.shape}/{py_rhs.shape}")

    abs_error = np.abs(native_rhs - py_rhs)
    scale = np.maximum(np.maximum(np.abs(native_rhs), np.abs(py_rhs)), 1e-300)
    rel_error = abs_error / scale
    candidates = []
    species_names = list(py_gas.species_names)
    labels = ["mass", "volume", "temperature"] + species_names
    for index in np.argsort(rel_error)[::-1][:20]:
        candidates.append({
            "index": int(index),
            "component": labels[index] if index < len(labels) else f"state_{index}",
            "native_rhs": float(native_rhs[index]),
            "python_rhs": float(py_rhs[index]),
            "absolute_error": float(abs_error[index]),
            "relative_error": float(rel_error[index]),
        })

    state_abs = np.abs(native_state - py_state)
    state_scale = np.maximum(np.maximum(np.abs(native_state), np.abs(py_state)), 1e-300)
    return {
        "oracle": oracle,
        "case": {
            "mechanism": "gri30.yaml",
            "initial_T_K": 1500.0,
            "initial_P_Pa": 101325.0,
            "initial_composition": "H2:2, O2:1, N2:4",
            "native_derivative_query_advance_s": 1e-12,
            "rtol": 1e-8,
            "atol": 1e-12,
        },
        "state": {
            "size": int(native_state.size),
            "max_absolute_error": float(np.max(state_abs)),
            "max_relative_error": float(np.max(state_abs / state_scale)),
        },
        "phase_kinetics": phase_kinetics,
        "rhs": {
            "size": int(native_rhs.size),
            "max_absolute_error": float(np.max(abs_error)),
            "max_relative_error": float(np.max(rel_error)),
            "rms_relative_error": float(np.sqrt(np.mean(rel_error ** 2))),
            "largest_relative_error_components": candidates,
        },
    }


if __name__ == "__main__":
    result = compare()
    output = ROOT / "audit/reactor_initial_rhs_pinned_differential.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(result, indent=2))
