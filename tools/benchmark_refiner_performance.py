"""Benchmark Refiner execution performance: Native Cantera vs Python reconstruction.

Generates audit/refiner_performance_benchmark.json.
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python.oned import Refiner


class BenchmarkDomain:
    def __init__(self, nv: int, np: int):
        self.n_components = nv
        self.n_points = np
        self._names = [f"component_{i}" for i in range(nv)]

    def nPoints(self) -> int:
        return self.n_points

    def nComponents(self) -> int:
        return self.n_components

    def componentName(self, n: int) -> str:
        return self._names[n]

    def component_name(self, n: int) -> str:
        return self._names[n]

    def index(self, n: int, j: int) -> int:
        return j * self.n_components + n

    def isFree(self) -> bool:
        return False

    def is_free(self) -> bool:
        return False

    def twoPointControlEnabled(self) -> bool:
        return False

    def two_point_control_enabled(self) -> bool:
        return False


def run_native_benchmark():
    compile_cmd = (
        "cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python && "
        "g++ -O3 -std=c++20 -I/home/nebula/cantera-pinned-oracle/include "
        "-L/home/nebula/cantera-pinned-oracle/build/lib tools/benchmark_native_refiner.cpp "
        "-o tools/benchmark_native_refiner -lcantera_shared "
        "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib && ./tools/benchmark_native_refiner"
    )
    subprocess.run(["wsl", "bash", "-c", compile_cmd], check=True)

    with open(repo_root / "audit" / "refiner_native_benchmark_results.json", "r") as f:
        return json.load(f)["benchmarks"]


def main():
    print("Running Native Refiner benchmark in WSL...")
    native_benchmarks = run_native_benchmark()

    sizes = [25, 100, 500, 2000]
    nv = 4
    repetitions = 200

    print("Running Python Refiner benchmark...")
    python_benchmarks = []

    for s_idx, n in enumerate(sizes):
        dom = BenchmarkDomain(nv, n)
        ref = Refiner(dom)
        ref.setCriteria(10.0, 0.5, 0.5, 0.02)
        ref.setMaxPoints(100000)

        z = np.linspace(0.0, 1.0, n, dtype=float)
        x = np.zeros(n * nv, dtype=float)
        for j in range(n):
            zj = z[j]
            T = 300.0 + 1700.0 * 0.5 * (1.0 + math.tanh(20.0 * (zj - 0.4)))
            u = 0.5 + 2.0 * 0.5 * (1.0 + math.tanh(20.0 * (zj - 0.4)))
            ych4 = 0.05 * 0.5 * (1.0 - math.tanh(20.0 * (zj - 0.4)))
            yo2 = 0.20 * 0.5 * (1.0 - math.tanh(20.0 * (zj - 0.4)))
            x[dom.index(0, j)] = T
            x[dom.index(1, j)] = u
            x[dom.index(2, j)] = ych4
            x[dom.index(3, j)] = yo2

        # Warmup
        for _ in range(10):
            ref.analyze(n, z, x)

        timings_ms = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            ref.analyze(n, z, x)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

        timings_ms.sort()
        median_ms = timings_ms[len(timings_ms) // 2]
        mean_ms = sum(timings_ms) / len(timings_ms)
        min_ms = timings_ms[0]
        max_ms = timings_ms[-1]

        python_benchmarks.append({
            "grid_size": n,
            "n_components": nv,
            "repetitions": repetitions,
            "median_ms": median_ms,
            "mean_ms": mean_ms,
            "min_ms": min_ms,
            "max_ms": max_ms,
        })

    comparisons = []
    speed_ratios = []

    for nb, pb in zip(native_benchmarks, python_benchmarks):
        ratio = pb["median_ms"] / max(nb["median_ms"], 1e-9)
        speed_ratios.append(ratio)
        comparisons.append({
            "grid_size": nb["grid_size"],
            "n_components": nb["n_components"],
            "native_median_ms": nb["median_ms"],
            "python_median_ms": pb["median_ms"],
            "slowdown_factor": round(ratio, 2)
        })

    overall_slowdown = float(np.median(speed_ratios))

    summary = {
        "schema_version": 1,
        "solver_family": "refiner",
        "subsystem": "grid refinement",
        "overall_slowdown_factor": round(overall_slowdown, 2),
        "native_equivalent_speed_parity": overall_slowdown <= 2.0,
        "comparisons": comparisons
    }

    out_file = repo_root / "audit" / "refiner_performance_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Refiner Performance Benchmark Summary:")
    print(f"  Overall Slowdown Factor: {overall_slowdown:.2f}x")
    for comp in comparisons:
        print(f"  N={comp['grid_size']}: Native={comp['native_median_ms']:.4f} ms, Python={comp['python_median_ms']:.4f} ms -> {comp['slowdown_factor']}x")


if __name__ == "__main__":
    main()
