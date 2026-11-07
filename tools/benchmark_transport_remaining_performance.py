"""Benchmark Pure Python Remaining Transport Models (Water, UnityLewis, IonGas)."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import time
import json
import numpy as np
import cantera_python as ct
from cantera_python.transport import WaterTransport

def main():
    iterations = 10000
    print(f"Benchmarking Pure Python Transport Remaining Models ({iterations} iterations each)...")

    # 1. WaterTransport
    sol_water = ct.Solution("water.yaml")
    sol_water.TP = 300.0, 101325.0
    tran_water = WaterTransport(sol_water.thermo)
    sol_water.thermo._density = 997.0476
    sol_water.thermo._isothermal_compressibility = 4.5246e-10
    sol_water.thermo._thermal_expansion_coeff = 0.000257

    t0 = time.perf_counter()
    sum_mu = 0.0
    sum_lam = 0.0
    for _ in range(iterations):
        sum_mu += float(tran_water.viscosity)
        sum_lam += float(tran_water.thermalConductivity())
    t1 = time.perf_counter()
    time_water_sec = t1 - t0
    print(f"WaterTransport: {time_water_sec:.4f} s ({time_water_sec / iterations * 1e6:.2f} us/call)")

    # 2. UnityLewisTransport
    sol_ul = ct.Solution("gri30.yaml", "gri30")
    sol_ul.TPX = 1200.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715"
    sol_ul.transport_model = "unity-Lewis-number"
    tran_ul = sol_ul.transport
    nsp_ul = sol_ul.n_species
    d_ul = np.zeros(nsp_ul)

    t0 = time.perf_counter()
    for _ in range(iterations):
        sum_lam += float(tran_ul.thermal_conductivity)
        tran_ul.getMixDiffCoeffs(d_ul)
    t1 = time.perf_counter()
    time_ul_sec = t1 - t0
    print(f"UnityLewisTransport: {time_ul_sec:.4f} s ({time_ul_sec / iterations * 1e6:.2f} us/call)")

    # 3. IonGasTransport (use 1000 iterations for Python to keep runtime snappy)
    iterations_ion = 1000
    sol_ion = ct.Solution("ch4_ion.yaml", "gas")
    sol_ion.TPX = 1500.0, 101325.0, "CO:0.05, CO2:0.05, H2O:0.15, N2:0.74, E:0.005, H3O+:0.005"
    sol_ion.transport_model = "ionized-gas"
    tran_ion = sol_ion.transport
    nsp_ion = sol_ion.n_species
    mobi_ion = np.zeros(nsp_ion)
    d_ion = np.zeros(nsp_ion)

    t0 = time.perf_counter()
    for _ in range(iterations_ion):
        sum_mu += float(tran_ion.viscosity)
        tran_ion.getMobilities(mobi_ion)
        tran_ion.getMixDiffCoeffs(d_ion)
    t1 = time.perf_counter()
    time_ion_sec = t1 - t0
    print(f"IonGasTransport: {time_ion_sec:.4f} s ({time_ion_sec / iterations_ion * 1e6:.2f} us/call)")

    # Load native benchmark if available to compute speedup/slowdown ratios
    nat_data = {}
    try:
        with open("audit/native_transport_remaining_benchmark.json", "r") as f:
            nat_data = json.load(f)
    except Exception:
        pass

    results = {
        "water_transport": {
            "iterations": iterations,
            "seconds": time_water_sec,
            "us_per_call": time_water_sec / iterations * 1e6,
            "native_us_per_call": nat_data.get("water_transport_us_per_call"),
            "ratio_py_to_native": (time_water_sec / iterations * 1e6) / nat_data["water_transport_us_per_call"] if "water_transport_us_per_call" in nat_data else None
        },
        "unity_lewis_transport": {
            "iterations": iterations,
            "seconds": time_ul_sec,
            "us_per_call": time_ul_sec / iterations * 1e6,
            "native_us_per_call": nat_data.get("unity_lewis_us_per_call"),
            "ratio_py_to_native": (time_ul_sec / iterations * 1e6) / nat_data["unity_lewis_us_per_call"] if "unity_lewis_us_per_call" in nat_data else None
        },
        "ion_gas_transport": {
            "iterations": iterations_ion,
            "seconds": time_ion_sec,
            "us_per_call": time_ion_sec / iterations_ion * 1e6,
            "native_us_per_call": nat_data.get("ion_gas_us_per_call"),
            "ratio_py_to_native": (time_ion_sec / iterations_ion * 1e6) / nat_data["ion_gas_us_per_call"] if "ion_gas_us_per_call" in nat_data else None
        }
    }

    with open("audit/transport_remaining_performance_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
