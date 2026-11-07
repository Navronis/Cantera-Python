"""Native Cantera DustyGas Transport Oracle Runner.

Executes within the WSL Ubuntu pinned Cantera oracle environment against
the pinned Cantera commit 726522be4e2a13454d8415b7ef799d621f665cf3.
"""

import json
from pathlib import Path
import sys
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: run_native_dusty_gas_oracle.py <corpus_jsonl> <output_jsonl> <run_id>")
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
            mech_name = case["mechanism"]

            if mech_name not in cached_phases:
                cached_phases[mech_name] = ct.DustyGas(mech_name)
            gas = cached_phases[mech_name]

            # Set base thermodynamic state
            gas.TPX = case["T"], case["P"], case["X"]

            # Set porous medium parameters
            gas.porosity = case["porosity"]
            gas.tortuosity = case["tortuosity"]
            gas.mean_pore_radius = case["pore_radius"]
            gas.mean_particle_diameter = case["particle_diameter"]
            if case.get("permeability", -1.0) >= 0:
                gas.permeability = case["permeability"]
            else:
                gas.permeability = -1.0

            # 1. Porous multicomponent diffusion matrix (K x K)
            multi_diff = gas.multi_diff_coeffs
            assert multi_diff.shape == (gas.n_species, gas.n_species)

            # 2. Thermal conductivity
            cond = float(gas.thermal_conductivity)

            # 3. Molar fluxes across two states
            delta = float(case["delta"])
            T1 = float(case["state1"]["T"])
            T2 = float(case["state2"]["T"])
            P1 = float(case["state1"]["P"])
            P2 = float(case["state2"]["P"])
            X1 = case["state1"]["X"]
            X2 = case["state2"]["X"]

            gas.TPX = T1, P1, X1
            rho1 = float(gas.density)
            Y1 = gas.Y.copy()

            gas.TPX = T2, P2, X2
            rho2 = float(gas.density)
            Y2 = gas.Y.copy()

            # Restore original state and porous parameters
            gas.TPX = case["T"], case["P"], case["X"]
            gas.porosity = case["porosity"]
            gas.tortuosity = case["tortuosity"]
            gas.mean_pore_radius = case["pore_radius"]
            gas.mean_particle_diameter = case["particle_diameter"]
            if case.get("permeability", -1.0) >= 0:
                gas.permeability = case["permeability"]
            else:
                gas.permeability = -1.0

            fluxes = gas.molar_fluxes(T1, T2, rho1, rho2, Y1, Y2, delta)
            assert fluxes.shape == (gas.n_species,)

            rec = {
                "run_id": run_id,
                "case_id": case_id,
                "mechanism": mech_name,
                "n_species": gas.n_species,
                "multi_diff_coeffs": multi_diff.tolist(),
                "thermal_conductivity": cond,
                "molar_fluxes": fluxes.tolist(),
            }
            results.append(rec)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Successfully evaluated {len(results)} DustyGas test cases with native Cantera.")

if __name__ == "__main__":
    main()
