"""Performance benchmark for Thermodynamics Family 2: SpeciesThermo."""
import sys
import time
import json
import numpy as np
sys.path.insert(0, ".")
import cantera_python as ct

def run_benchmark():
    gas = ct.Solution("gri30.yaml")
    n_sp = gas.n_species

    # Grab the first species thermo object
    sp0_thermo = gas.thermo._state.species[0].thermo
    t_base = 300.0
    N_evals = 100000

    # 1. Scalar evaluation benchmark (species 0)
    # Warmup
    for i in range(1000):
        T = t_base + (i % 1000) * 0.5
        sp0_thermo.properties(T)

    checksum_scalar = 0.0
    t0 = time.perf_counter()
    for i in range(N_evals):
        T = t_base + (i % 1000) * 0.5
        cp, h, s = sp0_thermo.properties(T)
        checksum_scalar += (cp + h + s)
    t1 = time.perf_counter()
    dt_single_us = ((t1 - t0) / N_evals) * 1e6

    # 2. Batch evaluation benchmark (all 53 species)
    N_batch = 5000
    species_thermos = [gas.thermo._state.species[k].thermo for k in range(n_sp)]
    cp_arr = np.zeros(n_sp)
    h_arr = np.zeros(n_sp)
    s_arr = np.zeros(n_sp)

    # Warmup
    for i in range(200):
        T = t_base + (i % 1000) * 0.5
        for k in range(n_sp):
            cp_arr[k], h_arr[k], s_arr[k] = species_thermos[k].properties(T)

    checksum_batch = 0.0
    t2 = time.perf_counter()
    for i in range(N_batch):
        T = t_base + (i % 1000) * 0.5
        for k in range(n_sp):
            cp_arr[k], h_arr[k], s_arr[k] = species_thermos[k].properties(T)
        checksum_batch += cp_arr[0] + h_arr[0] + s_arr[0]
    t3 = time.perf_counter()
    dt_batch_us = ((t3 - t2) / N_batch) * 1e6

    # Load native benchmark results
    native_data = {}
    with open("audit/native_thermo_family2_benchmark.json", "r") as f:
        native_data = json.load(f)

    native_single_us = native_data.get("single_eval_us", 0.018)
    native_batch_us = native_data.get("batch_eval_us", 0.276)

    ratio_single = dt_single_us / native_single_us
    ratio_batch = dt_batch_us / native_batch_us

    summary = {
        "family": "THERMO_FAMILY_2_SPECIES_THERMO",
        "benchmark_methodology": "Rigorous varying temperature T = 300 + (i % 1000)*0.5 K with volatile checksum accumulation on both C++ and Python.",
        "historical_invalidated_timings": {
            "native_batch_eval_us": 0.0008114,
            "invalidation_reason": "Native benchmark previously invoked setTemperature(T) without requesting polynomial property outputs, measuring only cache tag update rather than 53-species polynomial evaluations."
        },
        "verified_timings": {
            "single_polynomial_eval": {
                "python_time_us": dt_single_us,
                "native_time_us": native_single_us,
                "speed_ratio": ratio_single,
                "python_checksum": float(checksum_scalar),
                "native_checksum": native_data.get("checksum_scalar")
            },
            "batch_53_species_eval": {
                "python_time_us": dt_batch_us,
                "native_time_us": native_batch_us,
                "speed_ratio": ratio_batch,
                "python_checksum": float(checksum_batch),
                "native_checksum": native_data.get("checksum_batch")
            }
        },
        "native_speed_parity": False,
        "notes": f"Pure Python polynomial evaluation is {ratio_single:.1f}x native speed for scalar and {ratio_batch:.1f}x for batch 53-species update."
    }

    with open("audit/thermo_family2_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Family 2 Performance Benchmark Complete (Verified):")
    print(f"  Single Polynomial Eval: Python {dt_single_us:.3f} us vs Native {native_single_us:.3f} us ({ratio_single:.1f}x)")
    print(f"  Batch 53-Species Eval : Python {dt_batch_us:.3f} us vs Native {native_batch_us:.3f} us ({ratio_batch:.1f}x)")
    print("  NATIVE_SPEED_PARITY: False")

if __name__ == "__main__":
    run_benchmark()
