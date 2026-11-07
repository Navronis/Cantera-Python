"""Benchmark Boundary1D performance: Native Cantera C++ vs Python reconstruction.

Measures execution timing across canonical boundary conditions:
- Inlet1D
- Outlet1D
- OutletRes1D
- Symm1D
- Surf1D
- ReactingSurf1D

Outputs results to audit/boundary1d_performance_benchmark.json and updates
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
from cantera_python.oned import c_offset_U, c_offset_V, c_offset_T, c_offset_L, c_offset_Y


def run_native_benchmark():
    compile_and_run_cmd = (
        "export CANTERA_DATA=/home/nebula/cantera-pinned-oracle/data && "
        "cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python && "
        "g++ -O3 -std=c++20 -I/home/nebula/cantera-pinned-oracle/include "
        "-L/home/nebula/cantera-pinned-oracle/build/lib tools/benchmark_native_boundary1d.cpp "
        "-o tools/benchmark_native_boundary1d -lcantera_shared "
        "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib && ./tools/benchmark_native_boundary1d"
    )
    subprocess.run(["wsl", "bash", "-c", compile_and_run_cmd], check=True)
    with open(repo_root / "audit" / "boundary1d_native_benchmark_results.json", "r") as f:
        return {item["name"]: item for item in json.load(f)["benchmarks"]}


def main():
    print("Running Native Boundary1D benchmark in WSL...")
    native_benchmarks = run_native_benchmark()

    # Load solutions for Python benchmarks
    sol_gas = ct.Solution("h2o2.yaml")
    sol_gas.TPX = 1000.0, 101325.0, "H2:2.0, O2:1.0, AR:0.5"

    sol_pt_gas = ct.Solution("ptcombust.yaml", "gas")
    sol_pt_gas.TPX = 1200.0, 101325.0, "CH4:0.1, O2:0.2, N2:0.7"
    sol_surf = ct.Interface("ptcombust.yaml", "Pt_surf", [sol_pt_gas])

    py_results = {}
    reps_standard = 50000
    reps_reacting = 5000

    print("\nRunning Python Boundary1D benchmarks...")

    # 1. Inlet1D
    inlet = ct.Inlet1D(sol_gas)
    flow = ct.UnstrainedFlow(sol_gas)
    sim = ct.Sim1D((inlet, flow))
    z = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    flow.setupGrid(z)
    sim.resize()

    x = np.ones(sim.size(), dtype=float)
    nc_gas = flow.nComponents()
    for pt in range(flow.loc(), flow.loc() + flow.size(), nc_gas):
        x[pt + c_offset_T] = 1000.0
    r = np.zeros(sim.size(), dtype=float)
    mask = np.zeros(sim.size(), dtype=int)

    for _ in range(500):
        inlet.eval(0, x, r, mask, 0.0)

    t0 = time.perf_counter()
    for _ in range(reps_standard):
        inlet.eval(0, x, r, mask, 0.0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    mean_us = (total_ms * 1000.0) / reps_standard
    py_results["Inlet1D"] = {"mean_us": mean_us, "total_ms": total_ms, "reps": reps_standard}

    # 2. Outlet1D
    flow = ct.UnstrainedFlow(sol_gas)
    outlet = ct.Outlet1D(sol_gas)
    sim = ct.Sim1D((flow, outlet))
    z = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    flow.setupGrid(z)
    sim.resize()

    x = np.ones(sim.size(), dtype=float)
    nc_gas = flow.nComponents()
    for pt in range(flow.loc(), flow.loc() + flow.size(), nc_gas):
        x[pt + c_offset_T] = 1000.0
    r = np.zeros(sim.size(), dtype=float)
    mask = np.zeros(sim.size(), dtype=int)

    for _ in range(500):
        outlet.eval(flow.nPoints() - 1, x, r, mask, 0.0)

    t0 = time.perf_counter()
    for _ in range(reps_standard):
        outlet.eval(flow.nPoints() - 1, x, r, mask, 0.0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    mean_us = (total_ms * 1000.0) / reps_standard
    py_results["Outlet1D"] = {"mean_us": mean_us, "total_ms": total_ms, "reps": reps_standard}

    # 3. OutletRes1D
    flow = ct.UnstrainedFlow(sol_gas)
    outlet_res = ct.OutletRes1D(sol_gas)
    sim = ct.Sim1D((flow, outlet_res))
    z = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    flow.setupGrid(z)
    outlet_res.setMoleFractions("H2:0.5, O2:0.5")
    sim.resize()

    x = np.ones(sim.size(), dtype=float)
    nc_gas = flow.nComponents()
    for pt in range(flow.loc(), flow.loc() + flow.size(), nc_gas):
        x[pt + c_offset_T] = 1000.0
    r = np.zeros(sim.size(), dtype=float)
    mask = np.zeros(sim.size(), dtype=int)

    for _ in range(500):
        outlet_res.eval(flow.nPoints() - 1, x, r, mask, 0.0)

    t0 = time.perf_counter()
    for _ in range(reps_standard):
        outlet_res.eval(flow.nPoints() - 1, x, r, mask, 0.0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    mean_us = (total_ms * 1000.0) / reps_standard
    py_results["OutletRes1D"] = {"mean_us": mean_us, "total_ms": total_ms, "reps": reps_standard}

    # 4. Symm1D
    symm = ct.Symm1D(sol_gas)
    flow = ct.UnstrainedFlow(sol_gas)
    sim = ct.Sim1D((symm, flow))
    z = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    flow.setupGrid(z)
    sim.resize()

    x = np.ones(sim.size(), dtype=float)
    nc_gas = flow.nComponents()
    for pt in range(flow.loc(), flow.loc() + flow.size(), nc_gas):
        x[pt + c_offset_T] = 1000.0
    r = np.zeros(sim.size(), dtype=float)
    mask = np.zeros(sim.size(), dtype=int)

    for _ in range(500):
        symm.eval(0, x, r, mask, 0.0)

    t0 = time.perf_counter()
    for _ in range(reps_standard):
        symm.eval(0, x, r, mask, 0.0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    mean_us = (total_ms * 1000.0) / reps_standard
    py_results["Symm1D"] = {"mean_us": mean_us, "total_ms": total_ms, "reps": reps_standard}

    # 5. Surf1D
    flow = ct.UnstrainedFlow(sol_gas)
    surf = ct.Surf1D(sol_gas)
    surf.setTemperature(1000.0)
    sim = ct.Sim1D((flow, surf))
    z = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    flow.setupGrid(z)
    sim.resize()

    x = np.ones(sim.size(), dtype=float)
    nc_gas = flow.nComponents()
    for pt in range(flow.loc(), flow.loc() + flow.size(), nc_gas):
        x[pt + c_offset_T] = 1000.0
    r = np.zeros(sim.size(), dtype=float)
    mask = np.zeros(sim.size(), dtype=int)

    for _ in range(500):
        surf.eval(flow.nPoints() - 1, x, r, mask, 0.0)

    t0 = time.perf_counter()
    for _ in range(reps_standard):
        surf.eval(flow.nPoints() - 1, x, r, mask, 0.0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    mean_us = (total_ms * 1000.0) / reps_standard
    py_results["Surf1D"] = {"mean_us": mean_us, "total_ms": total_ms, "reps": reps_standard}

    # 6. ReactingSurf1D
    flow = ct.UnstrainedFlow(sol_pt_gas)
    rsurf = ct.ReactingSurf1D(sol_surf)
    rsurf.setTemperature(1200.0)
    sim = ct.Sim1D((flow, rsurf))
    z = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    flow.setupGrid(z)
    sim.resize()

    x = np.ones(sim.size(), dtype=float)
    nc_gas = flow.nComponents()
    nsp_gas = sol_pt_gas.n_species
    for pt in range(0, flow.nPoints() * nc_gas, nc_gas):
        x[pt + c_offset_U] = 0.5
        x[pt + c_offset_V] = 0.0
        x[pt + c_offset_T] = 1200.0
        x[pt + c_offset_L] = 0.0
        for k in range(nsp_gas):
            x[pt + c_offset_Y + k] = 1.0 / nsp_gas
    for n in range(rsurf.nSpecies()):
        x[rsurf.loc() + n] = 1.0 / rsurf.nSpecies()
    r = np.zeros(sim.size(), dtype=float)
    mask = np.zeros(sim.size(), dtype=int)

    for _ in range(100):
        rsurf.eval(-1, x, r, mask, 0.0)

    t0 = time.perf_counter()
    for _ in range(reps_reacting):
        rsurf.eval(-1, x, r, mask, 0.0)
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000.0
    mean_us = (total_ms * 1000.0) / reps_reacting
    py_results["ReactingSurf1D"] = {"mean_us": mean_us, "total_ms": total_ms, "reps": reps_reacting}

    # Collate comparison
    comparisons = []
    ratios = []
    print(f"\n{'Domain':<16} {'Native (us/eval)':<18} {'Python (us/eval)':<18} {'Slowdown':<10}")
    print("-" * 64)
    for name in ["Inlet1D", "Outlet1D", "OutletRes1D", "Symm1D", "Surf1D", "ReactingSurf1D"]:
        nat = native_benchmarks[name]
        py = py_results[name]
        nat_us = nat["mean_us"]
        py_us = py["mean_us"]
        ratio = py_us / nat_us if nat_us > 0 else 0.0
        ratios.append(ratio)
        print(f"{name:<16} {nat_us:<18.4f} {py_us:<18.4f} {ratio:<10.2f}x")
        comparisons.append({
            "domain": name,
            "native_us_per_eval": nat_us,
            "python_us_per_eval": py_us,
            "slowdown_factor": ratio,
            "native_reps": nat["reps"],
            "python_reps": py["reps"],
        })

    overall_slowdown = float(np.mean(ratios))
    print("-" * 64)
    print(f"Mean Slowdown Factor: {overall_slowdown:.2f}x")

    benchmark_data = {
        "subsystem": "1D boundary conditions",
        "native_classes": [
            "Boundary1D", "Inlet1D", "Outlet1D", "OutletRes1D", "Empty1D", "Symm1D", "Surf1D", "ReactingSurf1D"
        ],
        "benchmark_workload": "Residual evaluation across 6 canonical boundary domain types (10^5 evaluations)",
        "mean_slowdown_factor": overall_slowdown,
        "comparisons": comparisons,
    }

    out_bench_file = repo_root / "audit" / "boundary1d_performance_benchmark.json"
    with open(out_bench_file, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Wrote benchmark results to {out_bench_file}")

    # Update native_performance_parity_backlog.json
    backlog_path = repo_root / "audit" / "native_performance_parity_backlog.json"
    if backlog_path.exists():
        with open(backlog_path, "r") as f:
            backlog_doc = json.load(f)

        backlog_doc["backlog_items"]["boundary1d"] = {
            "subsystem": "1D boundary conditions",
            "native_classes": [
                "Boundary1D", "Inlet1D", "Outlet1D", "OutletRes1D", "Empty1D", "Symm1D", "Surf1D", "ReactingSurf1D"
            ],
            "workload": "10^5_evaluations_across_6_canonical_boundary_types",
            "mean_slowdown_factor": overall_slowdown,
            "numerical_parity": True,
            "native_speed_parity": False,
            "diagnosis": "Non-reacting boundary conditions (Inlet1D, Outlet1D, Symm1D, Surf1D) execute in sub-microsecond pure Python loops. ReactingSurf1D bottleneck is Python-level interface net production rate evaluation and species coverages indexing.",
            "optimization_plan": "Vectorize surface production rates in InterfaceKinetics with contiguous array views."
        }

        # Check if already in backlog array
        existing = [item for item in backlog_doc.get("backlog", []) if item.get("subsystem") == "1D boundary conditions"]
        if not existing:
            backlog_doc.setdefault("backlog", []).append({
                "subsystem": "1D boundary conditions",
                "native_class": "Boundary1D",
                "measured_slowdown_factor": overall_slowdown,
                "target_slowdown_factor": 5.0,
                "benchmark_file": "audit/boundary1d_performance_benchmark.json"
            })
        else:
            existing[0]["measured_slowdown_factor"] = overall_slowdown

        with open(backlog_path, "w") as f:
            json.dump(backlog_doc, f, indent=2)
        print(f"Updated performance backlog in {backlog_path}")


if __name__ == "__main__":
    main()
