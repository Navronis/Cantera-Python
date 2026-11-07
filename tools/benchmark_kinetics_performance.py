"""Benchmark kinetics evaluation performance: Native C++ vs Reference Python vs Accelerated Python.

Measures wall-clock time and speed ratio across multiple mechanisms (h2o2.yaml, gri30.yaml)
for forward rate constants, rates of progress, and net production rates.
Writes audit/kinetics_performance_benchmark.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
import numpy as np

import cantera_python as cp

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "audit" / "kinetics_performance_benchmark.json"

ITERATIONS = 1000

BENCHMARK_CONFIGS = [
    {
        "mechanism": "h2o2.yaml",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "H2:2, O2:1, N2:3.76"
    },
    {
        "mechanism": "gri30.yaml",
        "temperature": 1800.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.095, O2:0.21, N2:0.79"
    }
]


def benchmark_native_wsl(mech, T, P, X, iters):
    script = f"""
import cantera as ct, time
g = ct.Solution('{mech}')
g.TPX = {T}, {P}, '{X}'
# warmup
for _ in range(50):
    _ = g.forward_rate_constants
    _ = g.net_production_rates

t0 = time.perf_counter()
for _ in range({iters}):
    kf = g.forward_rate_constants
    rop = g.net_rates_of_progress
    wdot = g.net_production_rates
t1 = time.perf_counter()
total_ms = (t1 - t0) * 1000.0
print(f"{{total_ms / {iters}:.6f}}")
"""
    cmd = ["wsl", "/home/nebula/cantera-oracle-env/bin/python3", "-c", script]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def benchmark_python_accelerated(mech, T, P, X, iters):
    gas = cp.Solution(mech)
    gas.TPX = T, P, X
    # warmup
    for _ in range(50):
        _ = gas.forward_rate_constants
        _ = gas.net_production_rates

    t0 = time.perf_counter()
    for _ in range(iters):
        kf = gas.forward_rate_constants
        rop = gas.net_rates_of_progress
        wdot = gas.net_production_rates
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    return total_ms / iters


def benchmark_python_scalar(mech, T, P, X, iters):
    gas = cp.Solution(mech)
    gas.TPX = T, P, X
    kin = gas.kinetics
    # Disable fast path to test scalar reference path
    fast_backup = getattr(kin, "_fast", None)
    kin._fast = None
    try:
        # warmup
        for _ in range(20):
            _ = gas.forward_rate_constants
            _ = gas.net_production_rates

        sub_iters = min(iters, 100)
        t0 = time.perf_counter()
        for _ in range(sub_iters):
            kf = gas.forward_rate_constants
            rop = gas.net_rates_of_progress
            wdot = gas.net_production_rates
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000.0
        return total_ms / sub_iters
    finally:
        kin._fast = fast_backup


def main():
    print(f"Starting Kinetics Performance Benchmark ({ITERATIONS} iterations per config)...")

    benchmark_results = []
    ratios = []

    for cfg in BENCHMARK_CONFIGS:
        mech = cfg["mechanism"]
        T = cfg["temperature"]
        P = cfg["pressure"]
        X = cfg["mole_fractions"]

        print(f"\nBenchmarking {mech} (T={T}K, P={P}Pa)...")

        native_ms = benchmark_native_wsl(mech, T, P, X, ITERATIONS)
        print(f"  Native C++ (WSL)      : {native_ms:.4f} ms / eval")

        py_fast_ms = benchmark_python_accelerated(mech, T, P, X, ITERATIONS)
        print(f"  Accelerated Python    : {py_fast_ms:.4f} ms / eval")

        py_scalar_ms = benchmark_python_scalar(mech, T, P, X, ITERATIONS)
        print(f"  Reference Scalar Py   : {py_scalar_ms:.4f} ms / eval")

        ratio_to_native = py_fast_ms / native_ms
        speedup_over_scalar = py_scalar_ms / py_fast_ms
        ratios.append(ratio_to_native)

        print(f"  Slowdown vs Native    : {ratio_to_native:.2f}x")
        print(f"  Speedup vs Scalar     : {speedup_over_scalar:.2f}x")

        benchmark_results.append({
            "mechanism": mech,
            "temperature": T,
            "pressure": P,
            "mole_fractions": X,
            "iterations": ITERATIONS,
            "native_cpp_ms_per_eval": native_ms,
            "python_accelerated_ms_per_eval": py_fast_ms,
            "python_scalar_ms_per_eval": py_scalar_ms,
            "slowdown_vs_native": ratio_to_native,
            "speedup_over_scalar": speedup_over_scalar
        })

    mean_slowdown = float(np.mean(ratios))
    print(f"\nMean Slowdown vs Native C++: {mean_slowdown:.2f}x")

    payload = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mean_slowdown_vs_native": mean_slowdown,
        "configurations": benchmark_results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote benchmark report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
