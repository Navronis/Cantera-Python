"""Benchmark Sim1D execution performance: Native Cantera vs Python reconstruction.

Generates audit/sim1d_performance_benchmark.json.
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

from cantera_python.oned import Sim1D, Domain1D


class PyBenchSimDomain(Domain1D):
    def __init__(self, name: str, dtype: str, nv: int, np_pts: int, bw: int, grid: list[float]):
        super().__init__(nv=nv, points=np_pts, name=name)
        self.m_id = name
        self.name = name
        self.m_domain_type = dtype
        self._names = [f"{name}_var_{i}" for i in range(self.m_nv)]
        self.setBandwidth(bw)
        self.m_min = [-100.0] * self.m_nv
        self.m_max = [10000.0] * self.m_nv
        self.m_rtol_ss = [1e-4] * self.m_nv
        self.m_atol_ss = [1e-9] * self.m_nv
        if grid:
            self.setupGrid(grid)

    def nComponents(self) -> int:
        return self.m_nv

    def nPoints(self) -> int:
        return self.m_points

    def component_name(self, k: int) -> str:
        return self._names[int(k)]

    def componentName(self, k: int) -> str:
        return self._names[int(k)]

    def index(self, k: int, pt: int) -> int:
        return int(pt) * self.m_nv + int(k)

    def size(self) -> int:
        return self.m_points * self.m_nv

    def initialValue(self, n: int, j: int) -> float:
        return 1.0


def run_native_benchmark():
    compile_cmd = (
        "cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python && "
        "g++ -O3 -std=c++20 -I/home/nebula/cantera-pinned-oracle/include "
        "-L/home/nebula/cantera-pinned-oracle/build/lib tools/benchmark_native_sim1d.cpp "
        "-o tools/benchmark_native_sim1d -lcantera_shared "
        "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib && ./tools/benchmark_native_sim1d"
    )
    subprocess.run(["wsl", "bash", "-c", compile_cmd], check=True)

    with open(repo_root / "audit" / "sim1d_native_benchmark_results.json", "r") as f:
        return json.load(f)["benchmarks"]


def main():
    print("Running Native Sim1D benchmark in WSL...")
    native_benchmarks = run_native_benchmark()

    sizes = [25, 100, 500, 2000]
    nv = 4
    repetitions = 50

    comparisons = []

    print("\nRunning Python Sim1D benchmark...")
    for idx, total_points in enumerate(sizes):
        bulk_pts = (total_points - 2) if total_points > 2 else 1

        z_inlet = [0.0]
        z_bulk = [float(j) / max(bulk_pts - 1, 1) for j in range(bulk_pts)]
        z_outlet = [1.0]

        timings_ms = []

        for _ in range(repetitions):
            inlet = PyBenchSimDomain("inlet", "inlet", nv, 1, 1, z_inlet)
            flow = PyBenchSimDomain("flow", "flow", nv, bulk_pts, 3, z_bulk)
            outlet = PyBenchSimDomain("outlet", "outlet", nv, 1, 1, z_outlet)

            sim = Sim1D([inlet, flow, outlet])
            sim.setMaxGridPoints(-1, 100000)
            sim.setRefineCriteria(1, 10.0, 0.4, 0.4, 0.0)

            for j in range(bulk_pts):
                zj = z_bulk[j]
                val = 300.0 + 1700.0 / (1.0 + math.exp(-25.0 * (zj - 0.4)))
                for c in range(nv):
                    sim._setValue(1, c, j, val * (1.0 + 0.1 * c))

            t0 = time.perf_counter()
            ref_res = sim.refine(0)
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

        timings_ms.sort()
        py_median_ms = timings_ms[len(timings_ms) // 2]

        nat_bench = native_benchmarks[idx]
        nat_median_ms = nat_bench["median_ms"]
        slowdown = py_median_ms / nat_median_ms if nat_median_ms > 0 else 0.0

        print(f"Points={total_points} (size={total_points * nv}): Native={nat_median_ms:.5f} ms, Python={py_median_ms:.5f} ms -> {slowdown:.2f}x")

        comparisons.append({
            "grid_size": total_points,
            "n_components": nv,
            "total_size": total_points * nv,
            "native_median_ms": nat_median_ms,
            "python_median_ms": py_median_ms,
            "slowdown_factor": round(slowdown, 2),
        })

    overall_slowdown = round(float(np.median([c["slowdown_factor"] for c in comparisons])), 2)

    result = {
        "schema_version": 1,
        "solver_family": "sim1d",
        "subsystem": "1D solve orchestration",
        "overall_slowdown_factor": overall_slowdown,
        "native_equivalent_speed_parity": False,
        "comparisons": comparisons,
    }

    out_file = repo_root / "audit" / "sim1d_performance_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved benchmark results to {out_file}")
    print(f"Overall slowdown factor: {overall_slowdown}x")

    # Update native_performance_parity_backlog.json
    backlog_path = repo_root / "audit" / "native_performance_parity_backlog.json"
    if backlog_path.exists():
        with open(backlog_path, "r", encoding="utf-8") as f:
            backlog = json.load(f)

        backlog["backlog_items"]["sim1d"] = {
            "subsystem": "1D solve orchestration",
            "native_class": "Sim1D",
            "workload": "3_domains_refine_analysis_interpolation_grid_sizes_25_to_2000",
            "overall_slowdown_factor": overall_slowdown,
            "numerical_parity": True,
            "native_speed_parity": False,
            "diagnosis": "Grid refinement orchestration across multi-domain container involves domain loop, Refiner analyze step, midpoint insertions, and linear solution interpolation in pure Python.",
            "optimization_plan": "Vectorize multi-component linear interpolation and new grid array allocations using NumPy contiguous blocks.",
        }

        with open(backlog_path, "w", encoding="utf-8") as f:
            json.dump(backlog, f, indent=2)
        print(f"Updated backlog at {backlog_path}")


if __name__ == "__main__":
    main()
