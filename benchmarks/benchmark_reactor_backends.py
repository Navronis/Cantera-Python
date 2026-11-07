"""Small reproducible 0D backend benchmark with an analytic correctness gate."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import cantera_python as ct


OUTPUT = ROOT / "audit" / "performance_comparison.json"


MECHANISM = {
    "phases": [{"name": "gas", "thermo": "ideal-gas", "kinetics": "gas",
                "transport": "none", "species": ["A", "B"], "reactions": "all"}],
    "species": [{"name": name, "composition": {"H": 1},
                 "thermo": {"model": "constant-cp", "h0": 0.0,
                            "s0": 0.0, "cp0": 30000.0}}
                for name in ("A", "B")],
    "reactions": [{"equation": "A => B",
                   "rate-constant": {"A": 10.0, "b": 0.0, "Ea": 0.0}}],
}


def one(backend):
    gas = ct.Solution(MECHANISM)
    gas.TPX = 1000.0, ct.OneAtm, "A:1"
    net = ct.ReactorNet(ct.IdealGasConstPressureReactor(gas), solver_backend=backend)
    start = time.perf_counter()
    net.advance(1e-2)
    elapsed = time.perf_counter() - start
    actual = gas.Y[gas.species_index("B")]
    error = abs(actual - (1.0 - math.exp(-0.1)))
    return elapsed, error, getattr(net, "_last_solver_stats", None)


def main():
    records = []
    for backend in ("parity", "legacy"):
        one(backend)  # warm-up imports and dynamic libraries
        samples = [one(backend) for _ in range(7)]
        times = [sample[0] for sample in samples]
        records.append({
            "backend": backend, "repetitions": len(samples),
            "median_wall_seconds": statistics.median(times),
            "min_wall_seconds": min(times), "max_wall_seconds": max(times),
            "max_analytic_absolute_error": max(s[1] for s in samples),
            "last_solver_stats": samples[-1][2],
        })
    artifact = {
        "schema_version": 1,
        "case": "two-species constant-pressure first-order reactor to t=0.01 s",
        "timed_region": "integration only; mechanism construction excluded",
        "warmup": "one untimed solve per backend",
        "native_cantera": None,
        "native_python": None,
        "scope_limit": "Smoke benchmark only; broad required benchmark matrix remains open.",
        "records": records,
    }
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact["records"]))


if __name__ == "__main__":
    main()
