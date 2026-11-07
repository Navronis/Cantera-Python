"""Benchmark Flow1D performance: Native Cantera C++ vs Python reconstruction.

Measures residual evaluation timing across canonical 1D flow domain configurations:
- FreeFlow (10 points, energy enabled)
- AxisymmetricFlow (10 points, radiation enabled)
- UnstrainedFlow (10 points, energy enabled)
- FreeFlow (25 points, energy enabled)

Outputs results to audit/flow1d_performance_benchmark.json and updates
audit/native_performance_parity_backlog.json.
"""

import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import cantera_python as ct
from cantera_python.oned import (
    c_offset_U,
    c_offset_V,
    c_offset_T,
    c_offset_L,
    c_offset_E,
    c_offset_Uo,
    c_offset_Y,
)


def run_native_benchmark():
    compile_and_run_cmd = (
        "export CANTERA_DATA=/home/nebula/cantera-pinned-oracle/data && "
        "cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python && "
        "g++ -O3 -std=c++20 -I/home/nebula/cantera-pinned-oracle/include "
        "-L/home/nebula/cantera-pinned-oracle/build/lib tools/benchmark_native_flow1d.cpp "
        "-o tools/benchmark_native_flow1d -lcantera_shared "
        "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib && ./tools/benchmark_native_flow1d"
    )
    res = subprocess.run(["wsl", "bash", "-c", compile_and_run_cmd], capture_output=True, text=True)
    if res.returncode != 0:
        print("WSL compile/run failed:", res.stderr)
        raise RuntimeError("Native benchmark failed to compile or run")
    print(res.stdout)
    with open(repo_root / "audit" / "flow1d_native_benchmark_results.json", "r") as f:
        return {item["name"]: item for item in json.load(f)["benchmarks"]}


def run_python_benchmark(name, domain_type, energy_enabled, radiation_enabled, soret_enabled, points, reps):
    sol = ct.Solution("h2o2.yaml", transport="Mix")
    inlet = ct.Inlet1D(sol, name="inlet")
    flow = ct.Flow1D(sol, id="flame", points=points)
    outlet = ct.Outlet1D(sol, name="outlet")

    grid = np.linspace(0.0, 0.02, points)
    flow.setupGrid(grid)
    flow.setPressure(101325.0)

    if domain_type == "free-flow":
        flow.setFreeFlow()
    elif domain_type == "axisymmetric-flow":
        flow.setAxisymmetricFlow()
    elif domain_type == "unstrained-flow":
        flow.setUnstrainedFlow()

    if energy_enabled:
        flow.solveEnergyEqn()
    else:
        flow.fixTemperature()

    flow.enableRadiation(radiation_enabled)
    if radiation_enabled:
        flow.setBoundaryEmissivities(0.1, 0.1)
    flow.enableSoret(soret_enabled)

    sim = ct.Sim1D((inlet, flow, outlet))

    x_global = np.zeros(sim.size(), dtype=float)
    r_global = np.zeros(sim.size(), dtype=float)
    mask_global = np.zeros(sim.size(), dtype=int)

    flow_loc = flow.loc()
    nc = flow.nComponents()
    nsp = sol.n_species

    for j in range(points):
        frac = float(j) / float(points - 1)
        T = 300.0 + (2200.0 - 300.0) * frac
        u = 0.2 + (2.0 - 0.2) * frac
        V = 0.0
        L = 0.0
        Uo = 0.0

        base = flow_loc + j * nc
        x_global[base + c_offset_U] = u
        x_global[base + c_offset_V] = V
        x_global[base + c_offset_T] = T
        x_global[base + c_offset_L] = L
        x_global[base + c_offset_E] = 0.0
        x_global[base + c_offset_Uo] = Uo

        for k in range(nsp):
            sp = sol.species_name(k)
            Yk = 0.01
            if sp == "H2":
                Yk = 0.028 * (1.0 - 0.9 * frac)
            elif sp == "O2":
                Yk = 0.226 * (1.0 - 0.9 * frac)
            elif sp == "H2O":
                Yk = 0.15 * frac
            elif sp == "N2":
                Yk = 0.73
            elif sp == "AR":
                Yk = 0.70 * (1.0 - 0.5 * frac)
            x_global[base + c_offset_Y + k] = Yk

    # Warmup
    for _ in range(10):
        flow.eval(None, x_global, r_global, mask_global, 0.0)

    durations = []
    for _ in range(reps):
        t0 = time.perf_counter()
        flow.eval(None, x_global, r_global, mask_global, 0.0)
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1e6)  # microseconds

    durations.sort()
    median_us = durations[len(durations) // 2]
    mean_us = float(np.mean(durations))
    min_us = durations[0]
    max_us = durations[-1]

    print(f"  [{name}] Python Median: {median_us:.2f} us, Mean: {mean_us:.2f} us")
    return {
        "name": name,
        "domain_type": domain_type,
        "points": points,
        "reps": reps,
        "median_us": median_us,
        "mean_us": mean_us,
        "min_us": min_us,
        "max_us": max_us,
    }


def main():
    print("Running Native Flow1D benchmarks in WSL...")
    native_benchmarks = run_native_benchmark()

    print("\nRunning Python Flow1D benchmarks...")
    py_benchmarks = {}

    bench_configs = [
        ("FreeFlow_10pts", "free-flow", True, False, False, 10, 200),
        ("AxisymmetricFlow_10pts_Rad", "axisymmetric-flow", True, True, False, 10, 200),
        ("UnstrainedFlow_10pts", "unstrained-flow", True, False, False, 10, 200),
        ("FreeFlow_25pts", "free-flow", True, False, False, 25, 100),
    ]

    for cfg in bench_configs:
        name = cfg[0]
        res = run_python_benchmark(*cfg)
        py_benchmarks[name] = res

    comparisons = []
    slowdowns = []

    print("\n" + "=" * 80)
    print(f"{'Benchmark':<30} | {'Native (us)':<12} | {'Python (us)':<12} | {'Slowdown':<10}")
    print("-" * 80)

    for name, py_b in py_benchmarks.items():
        nat_b = native_benchmarks.get(name)
        if nat_b:
            nat_median = nat_b["median_us"]
            py_median = py_b["median_us"]
            slowdown = py_median / nat_median if nat_median > 0 else float("inf")
            slowdowns.append(slowdown)
            print(f"{name:<30} | {nat_median:<12.2f} | {py_median:<12.2f} | {slowdown:<10.1f}x")
            comparisons.append({
                "name": name,
                "domain_type": py_b["domain_type"],
                "points": py_b["points"],
                "native_median_us": nat_median,
                "python_median_us": py_median,
                "slowdown_factor": slowdown,
                "native_reps": nat_b["reps"],
                "python_reps": py_b["reps"],
            })

    mean_slowdown = float(np.mean(slowdowns)) if slowdowns else 0.0
    print("-" * 80)
    print(f"Mean Slowdown Factor: {mean_slowdown:.1f}x")
    print("=" * 80)

    report = {
        "subsystem": "1D flow equations (Flow1D / StFlow)",
        "native_classes": ["Flow1D", "StFlow", "FreeFlow", "AxisymmetricFlow", "UnstrainedFlow"],
        "benchmark_workload": "Flow1D residual evaluation across 4 configurations (10-25 grid points, H2/O2 kinetics, mixture-averaged transport, radiation)",
        "mean_slowdown_factor": mean_slowdown,
        "comparisons": comparisons,
    }

    out_file = repo_root / "audit" / "flow1d_performance_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote benchmark summary to {out_file}")

    # Update native_performance_parity_backlog.json
    backlog_file = repo_root / "audit" / "native_performance_parity_backlog.json"
    if backlog_file.exists():
        with open(backlog_file, "r") as f:
            backlog_data = json.load(f)

        backlog_data["backlog_items"]["flow1d"] = {
            "subsystem": "1D flow equations",
            "native_classes": ["Flow1D", "StFlow", "FreeFlow", "AxisymmetricFlow", "UnstrainedFlow"],
            "workload": "residual_eval_across_4_configurations_10_to_25_pts_h2o2_kinetics_mix_transport",
            "mean_slowdown_factor": mean_slowdown,
            "numerical_parity": True,
            "native_speed_parity": False,
            "diagnosis": "Flow1D residual calculation in Python is dominated by thermodynamic and transport property evaluations (mixture-averaged diffusion coefficients, species enthalpies, net production rates) evaluated at each grid point, plus pure Python finite difference difference quotient assembly.",
            "optimization_plan": "Vectorize thermodynamic state updates and species property batches across grid slices; utilize C/Cython or Numba kernels for inner spatial finite-difference operators.",
        }

        # Check if flow1d is already in backlog list
        existing = [item for item in backlog_data.get("backlog", []) if item.get("native_class") in ("Flow1D", "StFlow")]
        if not existing:
            backlog_data.setdefault("backlog", []).append({
                "subsystem": "1D flow equations",
                "native_class": "Flow1D",
                "measured_slowdown_factor": mean_slowdown,
                "target_slowdown_factor": 5.0,
                "benchmark_file": "audit/flow1d_performance_benchmark.json",
            })
        else:
            for item in existing:
                item["measured_slowdown_factor"] = mean_slowdown

        with open(backlog_file, "w") as f:
            json.dump(backlog_data, f, indent=2)
        print(f"Updated {backlog_file}")


if __name__ == "__main__":
    main()
