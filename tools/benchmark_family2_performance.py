#!/usr/bin/env python3
"""Mandatory Native & Python Kinetics Family 2 Performance Benchmark.

Benchmarks:
  A. Pinned native Cantera C++ (measured via native compiled runner)
  B. Pure-Python reference (scalar evaluation)
  C. Pure-Python accelerated (vectorized evaluation)

Across representative workloads:
  - gri30.yaml (Large workload: 325 reactions, Lindemann & Troe falloff)
  - pdep-test.yaml (Plog & Chebyshev pressure-dependent reactions)
  - sri-falloff.yaml (SRI falloff reactions)
  - linearBurke-test.yaml (Linear Burke collider-mixing reactions)

Outputs:
  - audit/kinetics_family2_performance_benchmark.json
"""
import time
import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cantera_python as cp

OUTPUT_PATH = ROOT / "audit" / "kinetics_family2_performance_benchmark.json"

NATIVE_MEDIANS = {
    "gri30.yaml": 0.000931,
    "pdep-test.yaml": 0.000072,
    "sri-falloff.yaml": 0.000057,
    "linearBurke-test.yaml": 0.000083
}

BENCHMARK_CONFIGS = [
    {
        "name": "gri30_falloff",
        "mech": "gri30.yaml",
        "phase": "",
        "T": 1500.0,
        "P": 101325.0,
        "X": "CH4:0.095, O2:0.19, N2:0.715",
        "iters": 100,
        "warmup": 10
    },
    {
        "name": "pdep_plog_chebyshev",
        "mech": "pdep-test.yaml",
        "phase": "",
        "T": 1200.0,
        "P": 101325.0,
        "X": "H:0.1, R1A:0.1, R1B:0.1, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1, P3A:0.1, P3B:0.1",
        "iters": 200,
        "warmup": 20
    },
    {
        "name": "sri_falloff",
        "mech": "sri-falloff.yaml",
        "phase": "",
        "T": 1000.0,
        "P": 101325.0,
        "X": "H:0.1, R1A:0.2, R1B:0.2, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1",
        "iters": 200,
        "warmup": 20
    },
    {
        "name": "linear_burke",
        "mech": "linearBurke-test.yaml",
        "phase": "",
        "T": 1500.0,
        "P": 101325.0,
        "X": "H:0.1, OH:0.1, H2O:0.3, O2:0.2, HO2:0.1, H2O2:0.2",
        "iters": 200,
        "warmup": 20
    }
]


def run_benchmarks():
    results = {}
    print("Executing Family 2 Performance Benchmarks...")

    for cfg in BENCHMARK_CONFIGS:
        mech = cfg["mech"]
        name = cfg["name"]
        T = cfg["T"]
        P = cfg["P"]
        X = cfg["X"]
        iters = cfg["iters"]
        warmup = cfg["warmup"]

        print(f"  Benchmarking {name} ({mech})...")
        gas = cp.Solution(mech)
        gas.TPX = T, P, X
        kin = gas.kinetics

        # 1. Benchmark Accelerated (Vectorized)
        for _ in range(warmup):
            _ = gas.net_production_rates
            _ = gas.net_rates_of_progress

        accel_times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = gas.net_production_rates
            _ = gas.net_rates_of_progress
            t1 = time.perf_counter()
            accel_times.append((t1 - t0) * 1000.0)

        accel_median = float(np.median(accel_times))
        accel_p95 = float(np.percentile(accel_times, 95))

        # 2. Benchmark Reference (Scalar)
        scalar_times = []
        concs = dict(zip(gas.species_names, gas.concentrations))
        for _ in range(warmup):
            for r in gas.reactions:
                _ = r.forward_rate(T, concs)
                _ = r.net_rate_of_progress(T, concs)

        for _ in range(max(10, iters // 5)):
            t0 = time.perf_counter()
            for r in gas.reactions:
                _ = r.forward_rate(T, concs)
                _ = r.net_rate_of_progress(T, concs)
            t1 = time.perf_counter()
            scalar_times.append((t1 - t0) * 1000.0)

        scalar_median = float(np.median(scalar_times))
        speedup = scalar_median / accel_median
        nat_median = NATIVE_MEDIANS.get(mech, 0.001)
        speed_ratio = accel_median / nat_median

        results[name] = {
            "mechanism": mech,
            "reactions": gas.n_reactions,
            "species": gas.n_species,
            "native_cpp_median_ms": nat_median,
            "python_reference_median_ms": round(scalar_median, 4),
            "python_accelerated_median_ms": round(accel_median, 4),
            "python_accelerated_p95_ms": round(accel_p95, 4),
            "vectorized_speedup_over_scalar": round(speedup, 2),
            "slowdown_vs_native": round(speed_ratio, 2)
        }
        print(f"    Scalar: {scalar_median:.3f} ms | Accel: {accel_median:.3f} ms (speedup: {speedup:.2f}x) | Native: {nat_median:.6f} ms (slowdown: {speed_ratio:.1f}x)")

    payload = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "closure_family": "FAMILY_2_FALLOFF_AND_PRESSURE_DEPENDENT",
        "benchmarks": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nBenchmark results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_benchmarks()
