"""Benchmark execution performance of Python Kinetics Family 4 vs native Cantera.

Measures ReactionPathDiagram.build() throughput across mechanisms and compares
against native C++ timings in audit/native_kinetics_family4_benchmark.json.
Writes audit/kinetics_family4_performance_benchmark.json.
"""

import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cantera_python as ct
from cantera_python.kinetics import newReactionPathDiagram


def run_benchmark():
    native_bench_path = Path("audit/native_kinetics_family4_benchmark.json")
    out_path = Path("audit/kinetics_family4_performance_benchmark.json")

    native_bench = {}
    if native_bench_path.exists():
        with open(native_bench_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for b in data.get("benchmarks", []):
                native_bench[b["name"]] = b["avg_ms"]

    results = []

    # Benchmark 1: gri30 C
    sol1 = ct.Solution("gri30.yaml")
    sol1.TPX = 1500.0, 101325.0, "CH4:0.1, O2:0.2, N2:0.7"
    diag1 = newReactionPathDiagram(sol1.kinetics, "C")
    diag1.build() # warm up

    iters1 = 50
    t0 = time.perf_counter()
    for _ in range(iters1):
        diag1.build()
    t1 = time.perf_counter()
    total_ms1 = (t1 - t0) * 1000.0
    avg_ms1 = total_ms1 / iters1

    nat_avg1 = native_bench.get("gri30_C_build", 0.0)
    slowdown1 = avg_ms1 / nat_avg1 if nat_avg1 > 0 else 0.0

    results.append({
        "name": "gri30_C_build",
        "iterations": iters1,
        "python_avg_ms": avg_ms1,
        "native_avg_ms": nat_avg1,
        "slowdown": slowdown1
    })
    print(f"Python gri30 C: {avg_ms1:.4f} ms/iter (native: {nat_avg1:.4f} ms/iter, slowdown: {slowdown1:.1f}x)")

    # Benchmark 2: h2o2 H
    sol2 = ct.Solution("h2o2.yaml")
    sol2.TPX = 1000.0, 101325.0, "H2:2.0, O2:1.0, AR:7.0"
    diag2 = newReactionPathDiagram(sol2.kinetics, "H")
    diag2.build() # warm up

    iters2 = 500
    t0 = time.perf_counter()
    for _ in range(iters2):
        diag2.build()
    t1 = time.perf_counter()
    total_ms2 = (t1 - t0) * 1000.0
    avg_ms2 = total_ms2 / iters2

    nat_avg2 = native_bench.get("h2o2_H_build", 0.0)
    slowdown2 = avg_ms2 / nat_avg2 if nat_avg2 > 0 else 0.0

    results.append({
        "name": "h2o2_H_build",
        "iterations": iters2,
        "python_avg_ms": avg_ms2,
        "native_avg_ms": nat_avg2,
        "slowdown": slowdown2
    })
    print(f"Python h2o2 H: {avg_ms2:.4f} ms/iter (native: {nat_avg2:.4f} ms/iter, slowdown: {slowdown2:.1f}x)")

    # Benchmark 3: ptcombust C
    sol3 = ct.Solution("ptcombust.yaml", "gas")
    sol3.TPX = 1000.0, 101325.0, "CH4:0.1, O2:0.2, AR:0.7"
    diag3 = newReactionPathDiagram(sol3.kinetics, "C")
    diag3.build() # warm up

    iters3 = 100
    t0 = time.perf_counter()
    for _ in range(iters3):
        diag3.build()
    t1 = time.perf_counter()
    total_ms3 = (t1 - t0) * 1000.0
    avg_ms3 = total_ms3 / iters3

    nat_avg3 = native_bench.get("ptcombust_C_build", 0.0)
    slowdown3 = avg_ms3 / nat_avg3 if nat_avg3 > 0 else 0.0

    results.append({
        "name": "ptcombust_C_build",
        "iterations": iters3,
        "python_avg_ms": avg_ms3,
        "native_avg_ms": nat_avg3,
        "slowdown": slowdown3
    })
    print(f"Python ptcombust C: {avg_ms3:.4f} ms/iter (native: {nat_avg3:.4f} ms/iter, slowdown: {slowdown3:.1f}x)")

    avg_slowdown = sum(r["slowdown"] for r in results) / len(results) if results else 0.0

    report = {
        "subsystem": "kinetics",
        "closure_family": "FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "mean_slowdown_factor": avg_slowdown,
        "benchmarks": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nMean Slowdown Factor: {avg_slowdown:.1f}x")
    print(f"Benchmark report written to {out_path}")


if __name__ == "__main__":
    run_benchmark()
