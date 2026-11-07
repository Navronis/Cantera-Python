"""
Parallel pure Python runner for Wave 4 MultiTransport differential validation.
Uses multiprocessing with 14 workers to evaluate all 10,000 cases in parallel.
"""

import sys
import os
import json
import time
from pathlib import Path
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import cantera_python as ct

DATA_DIR = repo_root.parent / 'cantera-original' / 'data'
GRI30_PATH = DATA_DIR / 'gri30.yaml'
H2O2_PATH = DATA_DIR / 'h2o2.yaml'

def process_chunk(cases):
    # Initialize mechanism once per worker process
    soln_gri = ct.Solution(GRI30_PATH)
    soln_gri.transport_model = "multicomponent"
    
    soln_h2o2 = ct.Solution(H2O2_PATH)
    soln_h2o2.transport_model = "multicomponent"

    results = []
    for case in cases:
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
        sp_visc = [float(x) for x in tr.species_viscosities()]

        # 4. Thermal diffusion coefficients
        dt = [float(x) for x in tr.thermal_diff_coeffs]

        # 5. Binary diffusion coefficients (full N x N)
        tr.updateThermal_T()
        bdiff = tr.binary_diff_coeffs.tolist()

        # 6. Multicomponent diffusion coefficients (full N x N)
        dmulti = tr.multi_diff_coeffs
        dmulti_list = dmulti.tolist()
        dmulti_diag = [float(dmulti[k, k]) for k in range(nsp)]
        dmulti_row0 = [float(dmulti[0, k]) for k in range(nsp)]
        bdiff_row0 = [float(bdiff[0][k]) for k in range(nsp)]

        # 7. Diffusive species mass fluxes
        grad_T = np.array([grad_T_val])
        grad_X = np.zeros(nsp)
        for k in range(nsp):
            sp_name = soln.species_names[k]
            if sp_name in grad_X_map:
                grad_X[k] = grad_X_map[sp_name]
        
        fluxes = np.zeros(nsp)
        tr.getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes)
        flux_sum = float(np.sum(fluxes))
        molar_fluxes = [float(fluxes[k] / soln.molecular_weights[k]) for k in range(nsp)]

        record = {
            "case_id": case_id,
            "viscosity": visc,
            "thermal_conductivity": cond,
            "flux_sum": flux_sum,
            "sp_visc": sp_visc,
            "dt": dt,
            "fluxes": fluxes.tolist(),
            "molar_fluxes": molar_fluxes,
            "bdiff": bdiff,
            "dmulti": dmulti_list,
            "dmulti_diag": dmulti_diag,
            "dmulti_row0": dmulti_row0,
            "bdiff_row0": bdiff_row0,
        }
        results.append(record)

    return results

def main(input_path="audit/corpus_wave4.jsonl", output_path="audit/python_wave4_results.jsonl", num_workers=12):
    print(f"Reading {input_path}...")
    all_cases = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                all_cases.append(json.loads(line))

    n_total = len(all_cases)
    chunk_size = 200
    chunks = [all_cases[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
    print(f"Loaded {n_total} cases ({len(chunks)} chunks of {chunk_size}). Running across {num_workers} worker processes...")

    import concurrent.futures
    t0 = time.time()
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            all_results.extend(res)
            completed += len(res)
            print(f"Completed {completed} / {n_total} cases (elapsed: {time.time() - t0:.1f}s)...")

    # Sort results by case_id
    all_results.sort(key=lambda r: r["case_id"])
    print(f"Writing {len(all_results)} ordered results to {output_path}...")
    with open(output_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    t_total = time.time() - t0
    print(f"Parallel Python evaluation finished in {t_total:.1f}s ({n_total / t_total:.1f} cases/s)!")

if __name__ == "__main__":
    main()
