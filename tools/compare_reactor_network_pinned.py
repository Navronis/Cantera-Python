"""Compare selected Python ReactorNet trajectories to the pinned Cantera build.

Run with the configured pinned-oracle Python from WSL, with this checkout on
PYTHONPATH. The report captures the oracle extension hash and the exact source
commit recorded by the native-oracle identity artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cantera as native
import cantera_python as python_port
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
ORIGINAL = ROOT.parent / "cantera-original"
PIN = "726522be4e2a13454d8415b7ef799d621f665cf3"


def _oracle_identity() -> dict:
    identity = json.loads((AUDIT / "native_oracle_identity.json").read_text())
    if identity.get("exact_commit") != PIN:
        raise RuntimeError("Recorded native oracle does not identify the pinned commit")
    extension = Path(native._cantera.__file__).resolve()
    digest = hashlib.sha256(extension.read_bytes()).hexdigest()
    if digest != identity.get("native_extension_sha256"):
        raise RuntimeError("Loaded native extension does not match the recorded oracle hash")
    return {
        "pinned_source_commit": PIN,
        "native_cantera_version": native.__version__,
        "native_extension_path": str(extension),
        "native_extension_sha256": digest,
        "python_port_path": str(Path(python_port.__file__).resolve()),
        "python_version": sys.version,
        "worktree_head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip(),
        "python_source_hashes": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "cantera_python/kinetics.py",
                "cantera_python/numerics.py",
                "cantera_python/reactor.py",
                "tests/test_solution_kinetics_reactor.py",
                "tools/compare_reactor_network_pinned.py",
            )
        },
    }


def _capture(reactors, simulation, target_time: float) -> dict:
    simulation.advance(target_time)

    def phase_of(reactor):
        return getattr(reactor, "thermo", getattr(reactor, "phase", None))

    solver_stats = getattr(simulation, "solver_stats", None)
    if solver_stats is None:
        solver_stats = getattr(simulation, "_last_solver_stats", None)
    if callable(solver_stats):
        solver_stats = solver_stats()
    if solver_stats is not None:
        solver_stats = {
            str(key): value.item() if hasattr(value, "item") else value
            for key, value in dict(solver_stats).items()
            if isinstance(value, (str, int, float, bool)) or hasattr(value, "item")
        }
    effective_atol = getattr(simulation, "_atol_array", None)
    if effective_atol is not None:
        unique_atols, atol_counts = np.unique(effective_atol, return_counts=True)
        effective_atol = {
            "component_values_and_counts": {
                f"{float(value):.17g}": int(count)
                for value, count in zip(unique_atols, atol_counts)
            }
        }
    python_integrator = getattr(simulation, "_native_python_integrator", None)
    if python_integrator is not None:
        solver_stats["order_histogram"] = dict(
            python_integrator._cv_order_histogram)
        solver_stats["current_order"] = int(python_integrator._cv_q)
        solver_stats["qwait"] = int(python_integrator._cv_qwait)

    reactor_states = [
        {
            "temperature_K": float(reactor.T),
            "pressure_Pa": float(phase_of(reactor).P),
            "mass_kg": float(reactor.mass),
            "volume_m3": float(reactor.volume),
            "internal_energy_J": float(
                reactor.mass * phase_of(reactor).int_energy_mass),
            "mass_fractions": [float(value) for value in phase_of(reactor).Y],
        }
        for reactor in reactors
    ]
    return {
        "time": float(simulation.time),
        "rtol": float(simulation.rtol),
        "atol": float(simulation.atol),
        "effective_atol": effective_atol,
        "solver_stats": solver_stats,
        "reactors": reactor_states,
        "total_mass_kg": sum(state["mass_kg"] for state in reactor_states),
        "total_internal_energy_J": sum(
            state["internal_energy_J"] for state in reactor_states),
    }


def _run(backend, case: str, solver_backend: str | None = None) -> dict:
    is_native = backend is native
    mechanism = str(ORIGINAL / "data" / "gri30.yaml")
    reactors = []
    for pressure, temperature in (
        ((2.0 if case == "valve" else 1.0) * 101325.0,
         300.0 if case == "valve" else 1000.0),
        (101325.0, 300.0),
    ):
        gas = backend.Solution(mechanism)
        gas.TPX = temperature, pressure, "N2:1"
        reactors.append(backend.IdealGasReactor(gas, volume=1.0))

    if case == "valve":
        device = backend.Valve(reactors[0], reactors[1], **({} if is_native else {"Kv": 1e-5}))
        if is_native:
            device.valve_coeff = 1e-5
    else:
        if is_native:
            device = backend.Wall(reactors[0], reactors[1], A=1.0, U=500.0)
        else:
            device = backend.Wall(reactors[0], reactors[1], area=1.0, U=500.0)

    if backend is python_port:
        simulation = backend.ReactorNet(
            reactors, solver_backend=solver_backend or "native_python")
    else:
        simulation = backend.ReactorNet(reactors)
    simulation.rtol = 1e-9
    return _capture(reactors, simulation, 0.005 if case == "valve" else 0.01)


def _differences(reference: dict, candidate: dict) -> dict:
    output = {}
    for i, (ref, got) in enumerate(zip(reference["reactors"], candidate["reactors"])):
        for field in ("temperature_K", "pressure_Pa", "mass_kg", "volume_m3"):
            output[f"reactor_{i}.{field}"] = {
                "native": ref[field],
                "python": got[field],
                "absolute_error": abs(ref[field] - got[field]),
                "relative_error": abs(ref[field] - got[field]) / max(
                    abs(ref[field]), abs(got[field]), 1e-300),
            }
        errors = [abs(a - b) for a, b in zip(ref["mass_fractions"], got["mass_fractions"])]
        output[f"reactor_{i}.mass_fractions"] = {
            "max_absolute_error": max(errors, default=0.0),
            "species_index_at_max": errors.index(max(errors)) if errors else None,
        }
    for field in ("total_mass_kg", "total_internal_energy_J"):
        output[field] = {
            "native": reference[field],
            "python": candidate[field],
            "absolute_error": abs(reference[field] - candidate[field]),
            "relative_error": abs(reference[field] - candidate[field]) / max(
                abs(reference[field]), abs(candidate[field]), 1e-300),
        }
    return output


def main() -> None:
    record = {
        "oracle": _oracle_identity(),
        "integrator_scope": (
            "Python port uses its in-package Python CV_BDF implementation; "
            "native reference uses the pinned Cantera ReactorNet backend."
        ),
        "kinetics_source_mapping": {
            "pinned_file": "include/cantera/kinetics/StoichManager.h",
            "behavior": (
                "C1/C2/C3 paths use direct concentration multiplication; "
                "C_AnyN zeros a reaction product if any used concentration "
                "is non-positive."
            ),
        },
        "cases": {},
    }
    for name in ("valve", "wall"):
        ref = _run(native, name)
        got = _run(python_port, name, "native_python")
        case_record = {
            "native": ref,
            "python": got,
            "differences": _differences(ref, got),
        }
        try:
            sundials = _run(python_port, name, "parity")
        except ImportError as err:
            case_record["python_sundials_backend_status"] = {
                "available": False,
                "reason": str(err),
            }
        else:
            case_record["python_sundials_backend_status"] = {"available": True}
            case_record["python_sundials_backend"] = sundials
            case_record["sundials_backend_differences"] = _differences(ref, sundials)
            case_record["python_vs_sundials_backend_differences"] = _differences(
                sundials, got)
        record["cases"][name] = case_record
    out = AUDIT / "reactor_network_python_pinned_differential.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
