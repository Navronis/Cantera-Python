"""Benchmark Domain1D execution performance: Native Cantera vs Python reconstruction.

Generates audit/domain1d_performance_benchmark.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python.oned import Domain1D


class BenchDomain1D(Domain1D):
    def eval(self, j, x, r, mask, rdt=0.0):
        pass

    def initialValue(self, n, j):
        return 1.0


def run_native_benchmark():
    cmd = (
        "cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python && "
        "./tools/benchmark_native_domain1d"
    )
    res = subprocess.run(["wsl", "bash", "-c", cmd], capture_output=True, text=True, check=True)
    lines = res.stdout.strip().split("\n")
    data = []
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) == 4:
            data.append({
                "points": int(parts[0]),
                "reps": int(parts[1]),
                "median_us": float(parts[2]),
                "median_ms": float(parts[2]) / 1000.0,
                "mean_us": float(parts[3]),
                "mean_ms": float(parts[3]) / 1000.0,
            })
    return data


def main():
    print("Running Native Domain1D benchmark in WSL...")
    native_benchmarks = run_native_benchmark()

    sizes = [25, 100, 500, 2000]
    nv = 5
    repetitions = 1000

    comparisons = []

    print("\nRunning Python Domain1D benchmark...")
    for idx, np_pts in enumerate(sizes):
        timings_us = []

        # Warmup
        for _ in range(50):
            d = BenchDomain1D(nv, np_pts)
            d.setupUniformGrid(np_pts, 0.1, 0.0)
            d.setSteadyTolerances(1e-4, 1e-9, -1)
            d.setTransientTolerances(1e-4, 1e-11, -1)
            d.setBounds(0, 0.0, 100.0)
            d.locate()

        for _ in range(repetitions):
            t0 = time.perf_counter()
            d = BenchDomain1D(nv, np_pts)
            d.setupUniformGrid(np_pts, 0.1, 0.0)
            d.setSteadyTolerances(1e-4, 1e-9, -1)
            d.setTransientTolerances(1e-4, 1e-11, -1)
            d.setBounds(0, 0.0, 100.0)
            d.locate()
            t1 = time.perf_counter()
            timings_us.append((t1 - t0) * 1e6)

        timings_us.sort()
        py_median_us = timings_us[len(timings_us) // 2]
        py_median_ms = py_median_us / 1000.0

        nat_bench = native_benchmarks[idx]
        nat_median_us = nat_bench["median_us"]
        nat_median_ms = nat_bench["median_ms"]
        slowdown = py_median_us / nat_median_us if nat_median_us > 0 else 0.0

        print(f"Points={np_pts:4d}: Native={nat_median_us:7.3f} us, Python={py_median_us:7.3f} us -> {slowdown:6.2f}x")

        comparisons.append({
            "grid_points": np_pts,
            "components": nv,
            "repetitions": repetitions,
            "native_median_us": nat_median_us,
            "native_median_ms": nat_median_ms,
            "python_median_us": py_median_us,
            "python_median_ms": py_median_ms,
            "slowdown_factor": slowdown,
        })

    overall_slowdown = sum(c["slowdown_factor"] for c in comparisons) / len(comparisons)
    print(f"\nMean Slowdown Across Domain Sizes: {overall_slowdown:.2f}x")

    bench_output = {
        "subsystem": "1D domain base abstraction",
        "native_class": "Domain1D",
        "benchmark_workload": "Domain1D construction, uniform grid setup, tolerances configuration, bounds, and locate across grid sizes",
        "grid_sizes": sizes,
        "nv": nv,
        "repetitions": repetitions,
        "overall_slowdown_factor": overall_slowdown,
        "detailed_comparisons": comparisons,
    }

    out_file = repo_root / "audit" / "domain1d_performance_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(bench_output, f, indent=2)

    print(f"Benchmark summary saved to {out_file.relative_to(repo_root)}")

    # Update native_performance_parity_backlog.json
    backlog_path = repo_root / "audit" / "native_performance_parity_backlog.json"
    if backlog_path.exists():
        with open(backlog_path, "r", encoding="utf-8") as f:
            backlog = json.load(f)

        found = False
        for item in backlog.get("backlog", []):
            if item.get("subsystem") == "1D domain base abstraction" or item.get("native_class") == "Domain1D":
                item["measured_slowdown_factor"] = overall_slowdown
                item["benchmark_file"] = str(out_file.relative_to(repo_root))
                found = True
                break

        if not found:
            backlog.setdefault("backlog", []).append({
                "subsystem": "1D domain base abstraction",
                "native_class": "Domain1D",
                "measured_slowdown_factor": overall_slowdown,
                "target_slowdown_factor": 5.0,
                "benchmark_file": str(out_file.relative_to(repo_root)),
            })

        with open(backlog_path, "w", encoding="utf-8") as f:
            json.dump(backlog, f, indent=2)
        print(f"Updated {backlog_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
