"""Run native Cantera oracle in WSL on MultiTransport corpus."""

import json
import math
import sys
from pathlib import Path
import cantera as ct
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python run_native_multicomponent_transport_oracle.py <corpus_jsonl> <output_jsonl> [run_id]")
        sys.exit(1)

    corpus_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    run_id = sys.argv[3] if len(sys.argv) > 3 else "UNSPECIFIED"

    with open(corpus_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    cached_phases = {}
    results = []

    for case in cases:
        case_id = case["case_id"]
        mech = case["mechanism"]
        if mech not in cached_phases:
            data_file = f"/home/nebula/cantera-pinned-oracle/data/{mech}"
            phase = ct.Solution(data_file)
            phase.transport_model = "multicomponent"
            cached_phases[mech] = phase
        else:
            phase = cached_phases[mech]

        phase.TPX = case["T"], case["P"], case["X"]

        nsp = phase.n_species
        visc = float(phase.viscosity)
        cond = float(phase.thermal_conductivity)
        sp_visc = [float(v) for v in phase.species_viscosities]
        bdiff = phase.binary_diff_coeffs.tolist()
        mdiff = phase.multi_diff_coeffs.tolist()
        tdiff = [float(v) for v in phase.thermal_diff_coeffs]

        # Calculate diffusive mass fluxes for test gradients if specified
        grad_X = case.get("grad_X")
        grad_T = case.get("grad_T", 0.0)
        flux_vector = None
        if grad_X is not None:
            # Use exact multicomponent diffusion flux formula from MultiTransport.cpp:
            # j_k = - rho * W_k / (W_mean^2) * sum_j(W_j * D_kj * grad_X_j) - D^T_k * (grad_T / T)
            rho = phase.density
            mw = np.array(phase.molecular_weights)
            w_mean = phase.mean_molecular_weight
            dmulti = np.array(mdiff)
            gx = np.array(grad_X)
            t_diff = np.array(tdiff)
            T = phase.T

            # Matrix-vector product sum_j(W_j * D_kj * grad_X_j)
            inner = np.dot(dmulti, mw * gx)
            flux = - (rho * mw / (w_mean ** 2)) * inner
            if abs(grad_T) > 1e-12:
                flux -= t_diff * (grad_T / T)
            flux_vector = flux.tolist()

        record = {
            "case_id": case_id,
            "multitran_run_id": run_id,
            "mechanism": mech,
            "T": case["T"],
            "P": case["P"],
            "n_species": nsp,
            "viscosity": visc,
            "thermal_conductivity": cond,
            "species_viscosities": sp_visc,
            "binary_diff_coeffs": bdiff,
            "multi_diff_coeffs": mdiff,
            "thermal_diff_coeffs": tdiff,
            "flux_vector": flux_vector,
        }
        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Native oracle evaluated {len(results)} MultiTransport cases successfully.")

if __name__ == "__main__":
    main()
