"""Native Cantera ChemEquil differential oracle runner for WSL."""

import json
import os
import sys
from pathlib import Path
import cantera as ct

def main():
    if len(sys.argv) < 4:
        print("Usage: run_native_chem_equil_oracle.py <corpus_path> <output_path> <run_id>")
        sys.exit(1)

    corpus_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    run_id = sys.argv[3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    cached_phases = {}

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            mech = case["mechanism"]
            mode = case["mode"]
            solver = case.get("solver", "element_potential")

            try:
                if mech not in cached_phases:
                    cached_phases[mech] = ct.Solution(mech)
                gas = cached_phases[mech]

                t_init = float(case["T"])
                p_init = float(case["P"])
                gas.TPX = t_init, p_init, case["X"]

                gas.equilibrate(mode, solver=solver)

                names = list(gas.species_names)
                x_vals = gas.X
                mu_vals = gas.chemical_potentials

                results.append({
                    "case_id": case_id,
                    "run_id": run_id,
                    "status": "PASS",
                    "mode": mode,
                    "T_final": float(gas.T),
                    "P_final": float(gas.P),
                    "density_final": float(gas.density),
                    "enthalpy_final": float(gas.enthalpy_mass),
                    "entropy_final": float(gas.entropy_mass),
                    "int_energy_final": float(gas.int_energy_mass),
                    "mean_molecular_weight": float(gas.mean_molecular_weight),
                    "mole_fractions": {name: float(x_vals[i]) for i, name in enumerate(names)},
                    "chemical_potentials": {name: float(mu_vals[i]) for i, name in enumerate(names)},
                })
            except Exception as e:
                results.append({
                    "case_id": case_id,
                    "run_id": run_id,
                    "status": "ERROR",
                    "mode": mode,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Native oracle evaluated {len(results)} cases with run_id {run_id}")

if __name__ == "__main__":
    main()
