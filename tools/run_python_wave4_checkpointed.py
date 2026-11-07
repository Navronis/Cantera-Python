import json
import time
import os
import sys
from pathlib import Path
import numpy as np
import concurrent.futures

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Worker globals for pre-warmed singletons
_soln_gri = None
_soln_h2o2 = None

def init_worker():
    global _soln_gri, _soln_h2o2
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    import cantera_python as ct
    _soln_gri = ct.Solution("gri30.yaml")
    _soln_gri.transport_model = "multicomponent"
    _soln_h2o2 = ct.Solution("h2o2.yaml")
    _soln_h2o2.transport_model = "multicomponent"

def evaluate_chunk(cases):
    global _soln_gri, _soln_h2o2
    results = []
    for case in cases:
        case_id = case["case_id"]
        mech = case["mech"]
        T = float(case["T"])
        P = float(case["P"])
        X = case["X"]
        grad_T_val = float(case["grad_T"])
        grad_X_map = case["grad_X"]

        soln = _soln_h2o2 if mech == "h2o2.yaml" else _soln_gri
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
    print(f"Reading corpus from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        all_cases = [json.loads(line) for line in f if line.strip()]

    n_total = len(all_cases)
    chunk_size = 100
    chunks = [all_cases[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
    print(f"Loaded {n_total} cases into {len(chunks)} chunks of {chunk_size}.")
    print(f"Launching ProcessPoolExecutor with {num_workers} pre-warmed workers...")

    t0 = time.time()
    results_by_id = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {executor.submit(evaluate_chunk, chunk): i for i, chunk in enumerate(chunks)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            chunk_results = future.result()
            for r in chunk_results:
                results_by_id[r["case_id"]] = r
            completed += len(chunk_results)
            if completed % 1000 == 0 or completed == n_total:
                elapsed = time.time() - t0
                print(f"Progress: {completed} / {n_total} cases evaluated ({completed / elapsed:.1f} cases/s, elapsed: {elapsed:.1f}s)...")

    # Sort strictly by case_id
    print(f"Sorting and writing {len(results_by_id)} records to {output_path}...")
    sorted_ids = sorted(results_by_id.keys())
    with open(output_path, "w", encoding="utf-8") as f:
        for cid in sorted_ids:
            f.write(json.dumps(results_by_id[cid]) + "\n")

    t_total = time.time() - t0
    print(f"Python full-matrix evaluation completed in {t_total:.2f}s ({n_total / t_total:.1f} cases/s)!")

if __name__ == "__main__":
    main()
