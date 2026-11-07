"""Run native Cantera oracle in WSL on Mixture Transport corpus."""

import json
import sys
from pathlib import Path
import cantera as ct
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_native_mixture_transport_oracle.py <corpus_jsonl> <output_jsonl>")
        sys.exit(1)

    corpus_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    with open(corpus_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    cached_phases = {}
    results = []

    for case in cases:
        mech = case["mechanism"]
        if mech not in cached_phases:
            # Native data path
            data_file = f"/home/nebula/cantera-pinned-oracle/data/{mech}"
            phase = ct.Solution(data_file)
            phase.transport_model = "mixture-averaged"
            cached_phases[mech] = phase
        else:
            phase = cached_phases[mech]

        phase.TPX = case["T"], case["P"], case["X"]

        visc = float(phase.viscosity)
        cond = float(phase.thermal_conductivity)
        sp_visc = [float(v) for v in phase.species_viscosities]
        bdiff = phase.binary_diff_coeffs.tolist()
        mdiff = [float(v) for v in phase.mix_diff_coeffs]
        mdiff_mole = [float(v) for v in phase.mix_diff_coeffs_mole]
        mdiff_mass = [float(v) for v in phase.mix_diff_coeffs_mass]

        results.append({
            "case_id": case["case_id"],
            "mechanism": case["mechanism"],
            "T": case["T"],
            "P": case["P"],
            "viscosity": visc,
            "thermal_conductivity": cond,
            "species_viscosities": sp_visc,
            "binary_diff_coeffs": bdiff,
            "mix_diff_coeffs": mdiff,
            "mix_diff_coeffs_mole": mdiff_mole,
            "mix_diff_coeffs_mass": mdiff_mass,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Native oracle evaluated {len(results)} cases successfully.")

if __name__ == "__main__":
    main()
