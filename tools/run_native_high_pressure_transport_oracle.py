"""Native Cantera HighPressure Gas Transport Oracle Runner.

Executes within the WSL Ubuntu pinned Cantera oracle environment against
the pinned Cantera commit 726522be4e2a13454d8415b7ef799d621f665cf3.
"""

import json
from pathlib import Path
import sys
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: run_native_high_pressure_transport_oracle.py <corpus_jsonl> <output_jsonl> <run_id>")
        sys.exit(1)

    corpus_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    run_id = sys.argv[3]

    import cantera as ct
    assert ct.__version__ is not None

    results = []
    cached_phases = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            mech_path = case["mechanism_path"]
            phase_name = case.get("phase_name")
            model = case["model"]

            cache_key = (mech_path, phase_name, model)
            if cache_key not in cached_phases:
                if phase_name:
                    gas = ct.Solution(mech_path, phase_name)
                else:
                    gas = ct.Solution(mech_path)
                gas.transport_model = model
                cached_phases[cache_key] = gas
            else:
                gas = cached_phases[cache_key]

            # Set thermodynamic state
            gas.TPX = float(case["T"]), float(case["P"]), case["X"]

            # Evaluate transport properties
            visc = float(gas.viscosity)
            cond = float(gas.thermal_conductivity)
            bin_diff = np.asarray(gas.binary_diff_coeffs, dtype=float)
            mix_diff = np.asarray(gas.mix_diff_coeffs, dtype=float)
            mix_diff_mole = np.asarray(gas.mix_diff_coeffs_mole, dtype=float)
            mix_diff_mass = np.asarray(gas.mix_diff_coeffs_mass, dtype=float)

            rec = {
                "run_id": run_id,
                "case_id": case_id,
                "mechanism": Path(mech_path).name,
                "model": model,
                "n_species": gas.n_species,
                "viscosity": visc,
                "thermal_conductivity": cond,
                "binary_diff_coeffs": bin_diff.tolist(),
                "mix_diff_coeffs": mix_diff.tolist(),
                "mix_diff_coeffs_mole": mix_diff_mole.tolist(),
                "mix_diff_coeffs_mass": mix_diff_mass.tolist(),
            }
            results.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Native oracle evaluated {len(results)} cases successfully -> {output_path}")

if __name__ == "__main__":
    main()
