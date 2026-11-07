"""Benchmark Pure Python Phase Core vs Pinned Native Cantera."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import json
import numpy as np
import cantera_python as ct

def main():
    iterations = 5000
    print(f"Benchmarking Pure Python Phase Core ({iterations} iterations)...")

    sol = ct.Solution("gri30.yaml")
    ph = sol.thermo

    # 1. setState_TD benchmark
    t0 = time.perf_counter()
    for i in range(iterations):
        T = 300.0 + (i % 1000)
        rho = 0.1 + (i % 50) * 0.01
        ph.setState_TD(T, rho)
    t1 = time.perf_counter()
    time_state_sec = t1 - t0
    us_state = time_state_sec / iterations * 1e6
    print(f"setState_TD: {time_state_sec:.4f} s ({us_state:.4f} us/call)")

    # 2. setMoleFractions benchmark
    x = np.zeros(sol.n_species)
    x[0] = 0.1; x[1] = 0.2; x[2] = 0.7
    t0 = time.perf_counter()
    for i in range(iterations):
        x[0] = 0.1 + (i % 10) * 0.01
        ph.setMoleFractions(x)
    t1 = time.perf_counter()
    time_mf_sec = t1 - t0
    us_mf = time_mf_sec / iterations * 1e6
    print(f"setMoleFractions: {time_mf_sec:.4f} s ({us_mf:.4f} us/call)")

    # 3. saveState / restoreState benchmark
    state = np.zeros(ph.stateSize())
    t0 = time.perf_counter()
    for i in range(iterations):
        ph.saveState(state)
        ph.restoreState(state)
    t1 = time.perf_counter()
    time_sr_sec = t1 - t0
    us_sr = time_sr_sec / iterations * 1e6
    print(f"saveState/restoreState: {time_sr_sec:.4f} s ({us_sr:.4f} us/call)")

    nat_data = {}
    try:
        with open("audit/native_thermo_family1_benchmark.json", "r") as f:
            nat_data = json.load(f)
    except Exception:
        pass

    results = {
        "iterations": iterations,
        "setState_TD_us_per_call": us_state,
        "setMoleFractions_us_per_call": us_mf,
        "saveRestoreState_us_per_call": us_sr,
        "native_setState_TD_us_per_call": nat_data.get("setState_TD_us_per_call"),
        "native_setMoleFractions_us_per_call": nat_data.get("setMoleFractions_us_per_call"),
        "native_saveRestoreState_us_per_call": nat_data.get("saveRestoreState_us_per_call"),
        "ratios": {
            "setState_TD_python_vs_native": us_state / nat_data["setState_TD_us_per_call"] if "setState_TD_us_per_call" in nat_data else None,
            "setMoleFractions_python_vs_native": us_mf / nat_data["setMoleFractions_us_per_call"] if "setMoleFractions_us_per_call" in nat_data else None,
            "saveRestoreState_python_vs_native": us_sr / nat_data["saveRestoreState_us_per_call"] if "saveRestoreState_us_per_call" in nat_data else None,
        }
    }

    out_path = "audit/thermo_family1_benchmark_summary.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote benchmark results to {out_path}")

if __name__ == "__main__":
    main()
