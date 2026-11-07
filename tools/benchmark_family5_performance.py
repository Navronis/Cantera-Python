"""Benchmark execution performance of Python Kinetics Family 5 vs native Cantera.

Measures throughput of:
1. TwoTempPlasmaRate evaluation
2. Oxygen plasma forward rate constants (TwoTemp + ElectronCollision)
3. Vibrational relaxation forward rate constants
4. Kinetics getNetProductionRates_ddT
Compares against native C++ timings in audit/native_kinetics_family5_benchmark.json.
Writes audit/kinetics_family5_performance_benchmark.json.
"""

import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cantera_python as ct
from cantera_python.rates import TwoTempPlasmaRate, TwoTempPlasmaData


def run_benchmark():
    native_bench_path = ROOT / "audit" / "native_kinetics_family5_benchmark.json"
    out_path = ROOT / "audit" / "kinetics_family5_performance_benchmark.json"

    native_bench = {}
    if native_bench_path.exists():
        with open(native_bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for b in data.get("benchmarks", []):
                native_bench[b["name"]] = b["avg_ms"]

    results = []

    # Benchmark 1: TwoTempPlasmaRate evaluation (100,000 iterations)
    ttRate = TwoTempPlasmaRate()
    node = {
        "rate-constant": {
            "A": 1.0e13,
            "b": 0.5,
            "Ea-gas": 1.0e7,
            "Ea-electron": 2.0e7,
            "b-gas": 0.1,
        }
    }
    ttRate.setParameters(node)
    data = TwoTempPlasmaData()
    data.update(T=1200.0, Te=25000.0)
    ttRate.evalFromStruct(data)  # warm up

    iters1 = 100000
    t0 = time.perf_counter()
    dummy1 = 0.0
    for _ in range(iters1):
        dummy1 += ttRate.evalFromStruct(data)
    t1 = time.perf_counter()
    total_ms1 = (t1 - t0) * 1000.0
    avg_ms1 = total_ms1 / iters1
    nat_avg1 = native_bench.get("twotemp_plasma_eval", 0.0)
    slowdown1 = avg_ms1 / nat_avg1 if nat_avg1 > 0 else 0.0

    results.append({
        "name": "twotemp_plasma_eval",
        "iterations": iters1,
        "python_avg_ms": avg_ms1,
        "native_avg_ms": nat_avg1,
        "slowdown": slowdown1,
    })
    print(f"Python twotemp_plasma_eval: {avg_ms1:.6f} ms/iter (native: {nat_avg1:.6f} ms/iter, slowdown: {slowdown1:.1f}x)")

    # Benchmark 2: Oxygen plasma rates (TwoTemp + ElectronCollision) (10,000 iterations)
    sol2 = ct.Solution("oxygen-plasma.yaml")
    sol2.thermo.electron_temperature = 7736.3312484
    sol2.thermo.electron_energy_levels = [0.0, 0.1, 1.0, 10.0]
    sol2.thermo.electron_energy_distribution = [
        1.0382504607964884, 1.032585600335941, 0.600752131409468, 1.8014445999315386e-24
    ]
    sol2.TP = 1200.0, 101325.0
    kin2 = sol2.kinetics
    kin2.forward_rate_constants()  # warm up

    iters2 = 10000
    t0 = time.perf_counter()
    dummy2 = 0.0
    for _ in range(iters2):
        kf = kin2.forward_rate_constants()
        dummy2 += kf[0] + kf[1]
    t1 = time.perf_counter()
    total_ms2 = (t1 - t0) * 1000.0
    avg_ms2 = total_ms2 / iters2
    nat_avg2 = native_bench.get("oxygen_plasma_rates", 0.0)
    slowdown2 = avg_ms2 / nat_avg2 if nat_avg2 > 0 else 0.0

    results.append({
        "name": "oxygen_plasma_rates",
        "iterations": iters2,
        "python_avg_ms": avg_ms2,
        "native_avg_ms": nat_avg2,
        "slowdown": slowdown2,
    })
    print(f"Python oxygen_plasma_rates: {avg_ms2:.6f} ms/iter (native: {nat_avg2:.6f} ms/iter, slowdown: {slowdown2:.1f}x)")

    # Benchmark 3: Vibrational relaxation rates (10,000 iterations)
    sol3 = ct.Solution("vibrational-relaxation.yaml")
    sol3.TP = 1500.0, 101325.0
    kin3 = sol3.kinetics
    kin3.forward_rate_constants()  # warm up

    iters3 = 10000
    t0 = time.perf_counter()
    dummy3 = 0.0
    for _ in range(iters3):
        kf = kin3.forward_rate_constants()
        dummy3 += kf[0]
    t1 = time.perf_counter()
    total_ms3 = (t1 - t0) * 1000.0
    avg_ms3 = total_ms3 / iters3
    nat_avg3 = native_bench.get("vibrational_relaxation_rates", 0.0)
    slowdown3 = avg_ms3 / nat_avg3 if nat_avg3 > 0 else 0.0

    results.append({
        "name": "vibrational_relaxation_rates",
        "iterations": iters3,
        "python_avg_ms": avg_ms3,
        "native_avg_ms": nat_avg3,
        "slowdown": slowdown3,
    })
    print(f"Python vibrational_relaxation_rates: {avg_ms3:.6f} ms/iter (native: {nat_avg3:.6f} ms/iter, slowdown: {slowdown3:.1f}x)")

    # Benchmark 4: Kinetics getNetProductionRates_ddT (1,000 iterations)
    sol4 = ct.Solution("h2o2.yaml")
    sol4.TPX = 1200.0, 506625.0, "H2:0.5, O2:0.5"
    kin4 = sol4.kinetics
    kin4.getNetProductionRates_ddT()  # warm up

    iters4 = 1000
    t0 = time.perf_counter()
    dummy4 = 0.0
    for _ in range(iters4):
        dw = kin4.getNetProductionRates_ddT()
        dummy4 += dw[0]
    t1 = time.perf_counter()
    total_ms4 = (t1 - t0) * 1000.0
    avg_ms4 = total_ms4 / iters4
    nat_avg4 = native_bench.get("kinetics_derivatives_ddT", 0.0)
    slowdown4 = avg_ms4 / nat_avg4 if nat_avg4 > 0 else 0.0

    results.append({
        "name": "kinetics_derivatives_ddT",
        "iterations": iters4,
        "python_avg_ms": avg_ms4,
        "native_avg_ms": nat_avg4,
        "slowdown": slowdown4,
    })
    print(f"Python kinetics_derivatives_ddT: {avg_ms4:.6f} ms/iter (native: {nat_avg4:.6f} ms/iter, slowdown: {slowdown4:.1f}x)")

    payload = {
        "family": "FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "benchmarks": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Benchmark results written to {out_path}")


if __name__ == "__main__":
    run_benchmark()
