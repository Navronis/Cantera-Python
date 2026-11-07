"""
Pure Python runner for Wave 4 MultiTransport differential validation.
Evaluates 10,000 cases from audit/corpus_wave4.jsonl and writes results to audit/python_wave4_results.jsonl.
"""

import sys
sys.path.insert(0, ".")
import json
import numpy as np
import cantera_python as ct

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / 'cantera-original' / 'data'
GRI30_PATH = DATA_DIR / 'gri30.yaml'
H2O2_PATH = DATA_DIR / 'h2o2.yaml'

def run_python_wave4(input_path="audit/corpus_wave4.jsonl", output_path="audit/python_wave4_results.jsonl"):
    print("Loading mechanisms for Python evaluation...")
    soln_gri = ct.Solution(GRI30_PATH)
    soln_gri.transport_model = "multicomponent"
    
    soln_h2o2 = ct.Solution(H2O2_PATH)
    soln_h2o2.transport_model = "multicomponent"

    count = 0
    print("Processing Wave 4 cases...")
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            if not line.strip():
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            mech = case["mech"]
            T = float(case["T"])
            P = float(case["P"])
            X = case["X"]
            grad_T_val = float(case["grad_T"])
            grad_X_map = case["grad_X"]

            soln = soln_h2o2 if mech == "h2o2.yaml" else soln_gri
            nsp = soln.n_species

            # Set state
            soln.TPX = T, P, X
            tr = soln.transport

            # 1. Viscosity
            visc = float(tr.viscosity)

            # 2. Thermal conductivity
            cond = float(tr.thermalConductivity())

            # 3. Species viscosities
            sp_visc = list(tr.species_viscosities())

            # 4. Thermal diffusion coefficients
            dt = list(tr.thermal_diff_coeffs)

            # 5. Multicomponent diffusion coefficients
            dmulti = tr.multi_diff_coeffs
            dmulti_diag = [float(dmulti[k, k]) for k in range(nsp)]
            dmulti_row0 = [float(dmulti[0, k]) for k in range(nsp)]

            # 6. Diffusive species mass fluxes
            grad_T = np.array([grad_T_val])
            grad_X = np.zeros(nsp)
            for k in range(nsp):
                sp_name = soln.species_names[k]
                if sp_name in grad_X_map:
                    grad_X[k] = grad_X_map[sp_name]
            
            fluxes = np.zeros(nsp)
            tr.getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes)
            flux_sum = float(np.sum(fluxes))

            record = {
                "case_id": case_id,
                "viscosity": visc,
                "thermal_conductivity": cond,
                "flux_sum": flux_sum,
                "sp_visc": sp_visc,
                "dt": dt,
                "fluxes": fluxes.tolist(),
                "dmulti_diag": dmulti_diag,
                "dmulti_row0": dmulti_row0
            }
            outfile.write(json.dumps(record) + "\n")

            count += 1
            if count % 2000 == 0:
                print(f"Evaluated {count} / 10000 Python cases...")

    print(f"Successfully evaluated all {count} Python cases!")

if __name__ == "__main__":
    run_python_wave4()
