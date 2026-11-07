import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import json
import time
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
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    import cantera_python as ct
    _soln_gri = ct.Solution("gri30.yaml")
    _soln_gri.transport_model = "multicomponent"
    _soln_h2o2 = ct.Solution("h2o2.yaml")
    _soln_h2o2.transport_model = "multicomponent"

def evaluate_chunk_to_file(chunk_idx: int, cases: list, chunk_path: str, wave4_run_id: str):
    global _soln_gri, _soln_h2o2
    p = Path(chunk_path)
    if p.exists() and p.stat().st_size > 1000:
        # Check if complete and matches wave4_run_id
        with open(p, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        if len(lines) == len(cases) and f'"wave4_run_id": "{wave4_run_id}"' in lines[0]:
            return (chunk_idx, len(cases), "CACHED")

    tmp_path = chunk_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as out:
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
                "wave4_run_id": wave4_run_id,
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
            out.write(json.dumps(record) + "\n")

    if os.path.exists(chunk_path):
        os.remove(chunk_path)
    os.rename(tmp_path, chunk_path)
    return (chunk_idx, len(cases), "COMPUTED")

def main(input_path="audit/corpus_wave4.jsonl", output_path="audit/python_wave4_results.jsonl", chunk_dir_path=None, num_workers=12, wave4_run_id="wave4_run_verified"):
    if chunk_dir_path is None:
        chunk_dir_path = "audit/py_adv_chunks" if "adversarial" in input_path else "audit/py_chunks"
    chunk_dir = Path(chunk_dir_path)
    chunk_dir.mkdir(exist_ok=True, parents=True)

    print(f"Reading corpus from {input_path} (wave4_run_id: {wave4_run_id})...", flush=True)
    with open(input_path, "r", encoding="utf-8") as f:
        all_cases = [json.loads(line) for line in f if line.strip()]

    n_total = len(all_cases)
    chunk_size = 100
    chunks = [all_cases[i:i + chunk_size] for i in range(0, n_total, chunk_size)]
    print(f"Loaded {n_total} cases into {len(chunks)} chunks of {chunk_size}.", flush=True)
    print(f"Launching ProcessPoolExecutor with {num_workers} workers writing directly to {chunk_dir}...", flush=True)

    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {}
        for idx, chunk in enumerate(chunks):
            c_path = str(chunk_dir / f"chunk_{idx:04d}.jsonl")
            f = executor.submit(evaluate_chunk_to_file, idx, chunk, c_path, wave4_run_id)
            futures[f] = idx

        completed_cases = 0
        for future in concurrent.futures.as_completed(futures):
            chunk_idx, n_cases, status = future.result()
            completed_cases += n_cases
            elapsed = time.time() - t0
            print(f"  Chunk {chunk_idx:04d} ({status}): {completed_cases} / {n_total} cases complete ({completed_cases/elapsed:.1f} cases/s, {elapsed:.1f}s elapsed)", flush=True)

    print(f"\nAll chunks complete! Assembling into {output_path}...", flush=True)
    total_written = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for idx in range(len(chunks)):
            c_path = chunk_dir / f"chunk_{idx:04d}.jsonl"
            with open(c_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line)
                        total_written += 1

    t_total = time.time() - t0
    sz_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Successfully assembled {total_written} cases into {output_path} ({sz_mb:.2f} MB) in {t_total:.1f}s!", flush=True)

if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "audit/corpus_wave4.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "audit/python_wave4_results.jsonl"
    cdir = sys.argv[3] if len(sys.argv) > 3 else None
    run_id = sys.argv[4] if len(sys.argv) > 4 else "wave4_run_verified"
    main(input_path=inp, output_path=out, chunk_dir_path=cdir, wave4_run_id=run_id)
