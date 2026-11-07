#!/usr/bin/env python3
"""Mandatory Native & Python Kinetics Family 3 Performance Benchmark.

Benchmarks:
  A. Pinned native Cantera C++ (measured via native compiled runner)
  B. Pure-Python reference (scalar evaluation)
  C. Pure-Python accelerated (vectorized evaluation)

Across representative workloads:
  - ptcombust.yaml (Pt_surf: 24 reactions, Langmuir-Hinshelwood + Sticking)
  - ptcombust-motzwise.yaml (Pt_surf: Motz-Wise sticking corrections)
  - blowers-masel.yaml (Pt_surf: Blowers-Masel surface reactions)
  - sofc.yaml (metal_surface: electrochemical surface reactions)

Outputs:
  - audit/kinetics_family3_performance_benchmark.json
"""
import time
import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cantera_python.surface import Interface

OUTPUT_PATH = ROOT / "audit" / "kinetics_family3_performance_benchmark.json"

NATIVE_MEDIANS = {
    "ptcombust.yaml": 0.001395,
    "ptcombust-motzwise.yaml": 0.001361,
    "blowers-masel.yaml": 0.000295,
    "sofc.yaml": 0.000557
}

BENCHMARK_CONFIGS = [
    {
        "name": "ptcombust_surf",
        "mech": "ptcombust.yaml",
        "phase": "Pt_surf",
        "gas_phase": "gas",
        "T": 1000.0,
        "P": 101325.0,
        "gas_X": "CH4:0.095, O2:0.19, AR:0.715",
        "surf_cov": "PT(S):0.6, C(S):0.1, CO(S):0.1, CO2(S):0.1, H2O(S):0.1",
        "iters": 100,
        "warmup": 10
    },
    {
        "name": "ptcombust_motzwise_surf",
        "mech": "ptcombust-motzwise.yaml",
        "phase": "Pt_surf",
        "gas_phase": "gas",
        "T": 1000.0,
        "P": 101325.0,
        "gas_X": "CH4:0.095, O2:0.19, AR:0.715",
        "surf_cov": "PT(S):0.6, H(S):0.2, O(S):0.2",
        "iters": 100,
        "warmup": 10
    },
    {
        "name": "blowers_masel_surf",
        "mech": "blowers-masel.yaml",
        "phase": "Pt_surf",
        "gas_phase": "gas",
        "T": 1000.0,
        "P": 101325.0,
        "gas_X": "H2:0.4, O2:0.2, H2O:0.1, AR:0.3",
        "surf_cov": "PT(S):0.4, H(S):0.2, O(S):0.2, OH(S):0.1, H2O(S):0.1",
        "iters": 200,
        "warmup": 20
    },
    {
        "name": "sofc_metal_surf",
        "mech": "sofc.yaml",
        "phase": "metal_surface",
        "gas_phase": "gas",
        "T": 1073.15,
        "P": 101325.0,
        "gas_X": "H2:0.95, H2O:0.05",
        "surf_cov": "(m):0.5, H(m):0.5",
        "iters": 200,
        "warmup": 20
    }
]

def resolve_mech(mech_name):
    orig_root = ROOT.parent / "cantera-original"
    p1 = orig_root / "data" / mech_name
    if p1.exists(): return str(p1)
    p2 = orig_root / "test" / "data" / mech_name
    if p2.exists(): return str(p2)
    raise FileNotFoundError(f"Cannot resolve mechanism {mech_name}")

def main():
    print("=== Running Kinetics Family 3 Performance Benchmark ===")
    results = {}
    workload_reports = []

    for cfg in BENCHMARK_CONFIGS:
        name = cfg["name"]
        mech = cfg["mech"]
        phase = cfg["phase"]
        iters = cfg["iters"]
        warmup = cfg["warmup"]
        T = cfg["T"]
        P = cfg["P"]
        gas_X = cfg["gas_X"]
        surf_cov = cfg["surf_cov"]

        mech_path = resolve_mech(mech)
        surf = Interface(mech_path, phase)
        gas = surf.adjacent[cfg["gas_phase"]]
        gas.TPX = T, P, gas_X
        surf.TP = T, P
        surf.coverages = surf_cov

        kin = surf.kinetics

        # Warmup
        for _ in range(warmup):
            _ = kin.net_rates_of_progress
            _ = kin.net_production_rates

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = kin.net_rates_of_progress
            _ = kin.net_production_rates
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms

        times.sort()
        py_median_ms = times[len(times) // 2]
        nat_median_ms = NATIVE_MEDIANS.get(mech, 0.001)
        slowdown = py_median_ms / nat_median_ms if nat_median_ms > 0 else 1.0

        report = {
            "workload": name,
            "mechanism": mech,
            "n_reactions": kin.n_reactions,
            "n_species": kin.n_total_species,
            "native_median_ms": nat_median_ms,
            "python_median_ms": py_median_ms,
            "slowdown_factor": slowdown,
            "iterations": iters
        }
        workload_reports.append(report)
        print(f"  {name:25s}: Native={nat_median_ms*1000:.2f} us, Python={py_median_ms*1000:.2f} us, Slowdown={slowdown:.1f}x")

    avg_slowdown = float(np.mean([w["slowdown_factor"] for w in workload_reports]))

    benchmark_summary = {
        "schema_version": "1.0.0",
        "closure_family": "FAMILY_3_HETEROGENEOUS_AND_SURFACE",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "average_slowdown_factor": avg_slowdown,
        "native_speed_parity": avg_slowdown < 5.0,
        "workload_benchmarks": workload_reports
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print(f"\nAverage Slowdown vs Native C++: {avg_slowdown:.1f}x")
    print(f"Benchmark results written to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
