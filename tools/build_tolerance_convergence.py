"""Generate a reproducible solver tolerance sweep against an analytic ODE case."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cantera_python as ct
from cantera_python.validation import compare_scalar


OUTPUT = ROOT / "audit" / "tolerance_convergence.json"


def mechanism():
    return {
        "phases": [{"name": "gas", "thermo": "ideal-gas", "kinetics": "gas",
                    "transport": "none", "species": ["A", "B"],
                    "reactions": "all"}],
        "species": [
            {"name": name, "composition": {"H": 1},
             "thermo": {"model": "constant-cp", "h0": 0.0,
                        "s0": 0.0, "cp0": 30000.0}}
            for name in ("A", "B")],
        "reactions": [{"equation": "A => B",
                       "rate-constant": {"A": 10.0, "b": 0.0, "Ea": 0.0}}],
    }


def run(backend, rtol, atol, target=1e-2):
    gas = ct.Solution(mechanism())
    gas.TPX = 1000.0, ct.OneAtm, "A:1"
    net = ct.ReactorNet(ct.IdealGasConstPressureReactor(gas),
                        solver_backend=backend)
    net.setTolerances(rtol, atol)
    net.advance(target)
    expected = 1.0 - math.exp(-10.0 * target)
    actual = gas.Y[gas.species_index("B")]
    metric = compare_scalar(actual, expected, atol=atol, rtol=rtol,
                            relative_floor=1e-300, near_zero_floor=atol)
    return {"backend": backend, "rtol": rtol, "atol": atol,
            "target_time": target, "quantity": "Y_B",
            "analytic_value": expected, **metric.as_dict(),
            "solver_stats": getattr(net, "_last_solver_stats", None)}


def main():
    records = []
    for backend in ("parity", "legacy"):
        for rtol in (1e-6, 1e-8, 1e-10, 1e-12):
            records.append(run(backend, rtol, rtol * 1e-6))
    artifact = {
        "schema_version": 1,
        "problem": "isothermal first-order A=>B constant-pressure reactor",
        "reference": "analytic Y_B(t)=1-exp(-10*t)",
        "native_oracle": None,
        "scope_limit": (
            "This isolates integration convergence and is not a substitute for "
            "the missing exact-pinned native reactor trajectory sweep."),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "records": len(records),
                      "max_abs": max(r["absolute_error"] for r in records)}))


if __name__ == "__main__":
    main()
