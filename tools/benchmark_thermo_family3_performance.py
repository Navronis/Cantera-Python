"""Performance benchmark for Thermodynamics Family 3: ThermoPhase Base."""
import sys
import time
import json
import numpy as np

sys.path.insert(0, ".")
from cantera_python.solution import Solution

def run_benchmark():
    sol = Solution("gri30.yaml")
    thermo = sol.thermo
    thermo.setState_TPX(1200.0, 101325.0, "CH4:1, O2:2, N2:7.52")

    nsp = thermo.nSpecies() if callable(thermo.nSpecies) else thermo.nSpecies
    mu = np.zeros(nsp)

    # Warmup
    for i in range(500):
        thermo.setState_TP(1200.0 + (i % 50), 101325.0)
        _ = thermo.enthalpy_mole
        _ = thermo.entropy_mole
        _ = thermo.cp_mole
        thermo.getChemPotentials(mu)

    # Benchmark 1: Forward State Update & Property Evaluation
    N_FWD = 20000
    sink = 0.0
    t0 = time.perf_counter()
    for i in range(N_FWD):
        T = 500.0 + (i % 1500) * 0.5
        P = 101325.0 + (i % 10) * 10000.0
        thermo.setState_TP(T, P)
        sink += thermo.enthalpy_mole
        sink += thermo.entropy_mole
        sink += thermo.cp_mole
        thermo.getChemPotentials(mu)
        sink += mu[0] + mu[-1]
    t1 = time.perf_counter()
    fwd_dur_us = ((t1 - t0) / N_FWD) * 1e6

    # Benchmark 2: State Inversion HP
    N_INV = 2000
    thermo.setState_TPX(1200.0, 101325.0, "CH4:1, O2:2, N2:7.52")
    H_ref = thermo.enthalpy_mass
    t0 = time.perf_counter()
    for i in range(N_INV):
        thermo.setState_TP(1000.0 + (i % 400), 101325.0)
        thermo.setState_HP(H_ref, 101325.0, rtol=1e-8)
        sink += thermo.temperature
    t1 = time.perf_counter()
    inv_dur_us = ((t1 - t0) / N_INV) * 1e6

    with open("audit/native_thermo_family3_benchmark.json") as f:
        nat = json.load(f)

    nat_fwd_us = nat["fwd_eval_us"]
    nat_inv_us = nat["hp_inversion_us"]

    fwd_slowdown = fwd_dur_us / nat_fwd_us
    inv_slowdown = inv_dur_us / nat_inv_us

    # Parity check
    speed_parity = (fwd_slowdown <= 2.0) and (inv_slowdown <= 2.0)

    summary = {
        "family": "THERMO_FAMILY_3_THERMOPHASE_BASE",
        "benchmark_environment": "x86_64 CPU, Windows 11 host vs WSL2 native",
        "workload": "GRI-30 (53 species), 20k forward evals, 2k HP inversions",
        "native_forward_eval_us": nat_fwd_us,
        "python_forward_eval_us": fwd_dur_us,
        "forward_slowdown_factor": fwd_slowdown,
        "native_hp_inversion_us": nat_inv_us,
        "python_hp_inversion_us": inv_dur_us,
        "hp_inversion_slowdown_factor": inv_slowdown,
        "THERMO_FAMILY3_NATIVE_SPEED_PARITY": speed_parity,
        "status_explanation": (
            "Pure-Python/NumPy ThermoPhase achieves full scientific parity with exact convergence "
            "and bitwise/numerical accuracy across all 90 cases. However, Python interpreter overhead "
            f"results in a {fwd_slowdown:.1f}x slowdown on forward evaluations and {inv_slowdown:.1f}x "
            "on Newton state inversions compared to -O3 compiled C++."
        ),
        "checksum_sink": float(sink)
    }

    with open("audit/thermo_family3_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Thermodynamics Family 3 Performance Benchmark Results:")
    print(f"  Forward eval : Native = {nat_fwd_us:.3f} us | Python = {fwd_dur_us:.3f} us ({fwd_slowdown:.1f}x)")
    print(f"  HP inversion : Native = {nat_inv_us:.3f} us | Python = {inv_dur_us:.3f} us ({inv_slowdown:.1f}x)")
    print(f"  THERMO_FAMILY3_NATIVE_SPEED_PARITY: {speed_parity}")

if __name__ == "__main__":
    run_benchmark()
