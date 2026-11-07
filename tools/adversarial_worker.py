"""
Worker module for Wave 4 adversarial validation.
Initializes and pre-warms mechanisms once per worker process.
"""

import sys
import math
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import cantera_python as ct

DATA_DIR = repo_root.parent / 'cantera-original' / 'data'
GRI30_PATH = DATA_DIR / 'gri30.yaml'
H2O2_PATH = DATA_DIR / 'h2o2.yaml'

_soln_gri = None
_soln_h2o2 = None

def get_solutions():
    global _soln_gri, _soln_h2o2
    if _soln_gri is None:
        _soln_gri = ct.Solution(GRI30_PATH)
        _soln_gri.transport_model = "multicomponent"
        _ = _soln_gri.transport.thermalConductivity()

        _soln_h2o2 = ct.Solution(H2O2_PATH)
        _soln_h2o2.transport_model = "multicomponent"
        _ = _soln_h2o2.transport.thermalConductivity()
    return _soln_gri, _soln_h2o2

def evaluate_single_adversarial_case(soln_gri, soln_h2o2, i):
    rng = np.random.default_rng(1337 + i)
    soln = soln_gri if (i % 2 == 0) else soln_h2o2
    mech_name = "gri30.yaml" if (i % 2 == 0) else "h2o2.yaml"
    nsp = soln.n_species
    tr = soln.transport

    category_id = i % 5
    cat_names = ["deep_trace", "pure_species", "extreme_mw_ratio", "extreme_temp", "extreme_pressure"]
    cat_name = cat_names[category_id]

    if category_id == 0:
        # 1. Deep trace species: many species at exact 0.0 or 1e-25
        T = float(rng.uniform(300.0, 2500.0))
        P = float(101325.0 * (10 ** rng.uniform(-1.0, 1.5)))
        x = np.zeros(nsp)
        major = rng.choice(nsp, size=2, replace=False)
        x[major[0]] = float(rng.uniform(0.3, 0.7))
        x[major[1]] = 1.0 - x[major[0]]
        for k in range(nsp):
            if k not in major and rng.random() < 0.5:
                x[k] = float(10 ** rng.uniform(-30.0, -15.0))
        x /= np.sum(x)
        soln.TPX = T, P, x

    elif category_id == 1:
        # 2. Pure species
        T = float(rng.uniform(300.0, 3000.0))
        P = float(101325.0 * (10 ** rng.uniform(-1.0, 1.0)))
        k_pure = i % nsp
        x = np.zeros(nsp)
        x[k_pure] = 1.0
        soln.TPX = T, P, x

    elif category_id == 2:
        # 3. Extreme MW ratio binary mixture
        T = float(rng.uniform(400.0, 2200.0))
        P = 101325.0
        light_idx = 0
        heavy_idx = nsp - 1
        frac = float(10 ** rng.uniform(-5.0, 0.0) if (i % 2 == 0) else (1.0 - 10 ** rng.uniform(-5.0, 0.0)))
        x = np.zeros(nsp)
        x[light_idx] = frac
        x[heavy_idx] = 1.0 - frac
        soln.TPX = T, P, x

    elif category_id == 3:
        # 4. Extreme Temperature (200 K to 4500 K)
        T = float(rng.choice([200.0, 250.0, 3200.0, 3800.0, 4500.0, rng.uniform(200.0, 4500.0)]))
        P = float(101325.0 * (10 ** rng.uniform(-1.0, 1.0)))
        x = rng.dirichlet(np.ones(nsp))
        soln.TPX = T, P, x

    else:
        # 5. Extreme Pressure (1e-3 atm to 100 atm)
        T = float(rng.uniform(500.0, 2000.0))
        P = float(101325.0 * (10 ** rng.uniform(-3.0, 2.0)))
        x = rng.dirichlet(np.ones(nsp))
        soln.TPX = T, P, x

    # Evaluate properties
    visc = float(tr.viscosity)
    cond = float(tr.thermalConductivity())
    sp_visc = tr.species_viscosities()
    dt = tr.thermal_diff_coeffs

    # Random gradients for flux test
    grad_T = np.array([rng.uniform(-2000.0, 2000.0)])
    grad_X = rng.normal(0.0, 0.1, nsp)
    grad_X -= np.mean(grad_X)

    fluxes = np.zeros(nsp)
    tr.getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes)
    flux_sum = float(abs(np.sum(fluxes)))

    dmulti = tr.multi_diff_coeffs

    # Verify invariants
    reasons = []
    if not (math.isfinite(visc) and visc > 0):
        reasons.append("non_positive_or_non_finite_viscosity")
    if not (math.isfinite(cond) and cond > 0):
        reasons.append("non_positive_or_non_finite_conductivity")
    if not all(math.isfinite(v) and v > 0 for v in sp_visc):
        reasons.append("non_positive_or_non_finite_species_viscosity")
    if not all(math.isfinite(d) for d in dt):
        reasons.append("non_finite_thermal_diffusion")
    if not (math.isfinite(flux_sum) and flux_sum < 1e-12):
        reasons.append(f"flux_leak_{flux_sum}")
    if np.any(np.isnan(dmulti)) or np.any(np.isinf(dmulti)):
        reasons.append("singular_dmulti_matrix")

    status = "PASS" if len(reasons) == 0 else f"FAIL: {'; '.join(reasons)}"

    return {
        "case_id": i,
        "category": cat_name,
        "mech": mech_name,
        "status": status,
        "flux_sum": flux_sum,
        "viscosity": visc,
        "thermal_conductivity": cond
    }

def run_chunk_worker(chunk_id, indices):
    soln_gri, soln_h2o2 = get_solutions()
    results = []
    for i in indices:
        res = evaluate_single_adversarial_case(soln_gri, soln_h2o2, i)
        results.append(res)
    return chunk_id, results
