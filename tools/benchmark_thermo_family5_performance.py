"""Microbenchmarking Python implementation against native Cantera Family 5."""
import sys
import time
import json
from pathlib import Path

# Ensure outputs/cantera-python is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cantera_python as ct
from cantera_python.electrolytes import WaterProps

native_bench_file = Path("audit/native_thermo_family5_benchmark.json")
with open(native_bench_file, "r", encoding="utf-8") as f:
    native_data = json.load(f)

N = native_data.get("iterations", 5000)
print(f"Running Family 5 Python performance benchmark ({N} iterations)...")

# 1. Peng-Robinson
pr = ct.Solution("co2_PR_example.yaml", "CO2-PR")
start_pr = time.perf_counter()
dummy_pr = 0.0
for i in range(N):
    T = 300.0 + (i % 200) * 1.0
    P = 1.0e5 + (i % 50) * 1.0e5
    pr.TP = (T, P)
    dummy_pr += pr.enthalpy_mole + pr.cp_mole + pr.sound_speed
time_pr = time.perf_counter() - start_pr

# 2. Redlich-Kwong
rk = ct.Solution("co2_RK_example.yaml", "carbondioxide")
start_rk = time.perf_counter()
dummy_rk = 0.0
for i in range(N):
    T = 300.0 + (i % 200) * 1.0
    P = 1.0e5 + (i % 50) * 1.0e5
    rk.TP = (T, P)
    dummy_rk += rk.enthalpy_mole + rk.cp_mole + rk.sound_speed
time_rk = time.perf_counter() - start_rk

# 3. PureFluidPhase
water = ct.Solution("liquidvapor.yaml", "water")
start_water = time.perf_counter()
dummy_water = 0.0
for i in range(N):
    T = 300.0 + (i % 150) * 1.0
    P = 1.0e5 + (i % 50) * 1.0e5
    water.TP = (T, P)
    dummy_water += water.density + water.enthalpy_mass + water.entropy_mass
time_water = time.perf_counter() - start_water

# 4. WaterProps IAPWS
wp = WaterProps()
start_wp = time.perf_counter()
dummy_wp = 0.0
for i in range(N):
    T = 300.0 + (i % 150) * 1.0
    P = 1.0e5 + (i % 50) * 1.0e5
    dummy_wp += wp.density_IAPWS(T, P)
time_wp = time.perf_counter() - start_wp

total_python = time_pr + time_rk + time_water + time_wp
native_total = native_data["total_time_sec"]

summary = {
    "iterations": N,
    "native_results": native_data["workloads"],
    "native_total_sec": native_total,
    "python_results": {
        "peng_robinson": {"time_sec": time_pr, "us_per_call": time_pr / N * 1e6, "dummy": dummy_pr},
        "redlich_kwong": {"time_sec": time_rk, "us_per_call": time_rk / N * 1e6, "dummy": dummy_rk},
        "pure_fluid": {"time_sec": time_water, "us_per_call": time_water / N * 1e6, "dummy": dummy_water},
        "water_props": {"time_sec": time_wp, "us_per_call": time_wp / N * 1e6, "dummy": dummy_wp},
    },
    "python_total_sec": total_python,
    "ratio_python_to_native": {
        "peng_robinson": time_pr / native_data["workloads"]["peng_robinson"]["time_sec"],
        "redlich_kwong": time_rk / native_data["workloads"]["redlich_kwong"]["time_sec"],
        "pure_fluid": time_water / native_data["workloads"]["pure_fluid"]["time_sec"],
        "water_props": time_wp / native_data["workloads"]["water_props"]["time_sec"],
        "overall": total_python / native_total,
    }
}

print("\n" + "="*70)
print("BENCHMARK SUMMARY (5000 iterations):")
print(f"  Peng-Robinson:    Native {native_data['workloads']['peng_robinson']['time_sec']:.4f}s vs Python {time_pr:.4f}s ({summary['ratio_python_to_native']['peng_robinson']:.2f}x)")
print(f"  Redlich-Kwong:    Native {native_data['workloads']['redlich_kwong']['time_sec']:.4f}s vs Python {time_rk:.4f}s ({summary['ratio_python_to_native']['redlich_kwong']:.2f}x)")
print(f"  PureFluidPhase:   Native {native_data['workloads']['pure_fluid']['time_sec']:.4f}s vs Python {time_water:.4f}s ({summary['ratio_python_to_native']['pure_fluid']:.2f}x)")
print(f"  WaterProps IAPWS: Native {native_data['workloads']['water_props']['time_sec']:.4f}s vs Python {time_wp:.4f}s ({summary['ratio_python_to_native']['water_props']:.2f}x)")
print(f"  TOTAL TIME:       Native {native_total:.4f}s vs Python {total_python:.4f}s ({summary['ratio_python_to_native']['overall']:.2f}x)")
print("="*70)

summary_file = Path("audit/thermo_family5_benchmark_summary.json")
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote benchmark summary to {summary_file}")
