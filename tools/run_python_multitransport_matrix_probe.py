"""Write the corresponding full Python MultiTransport matrix probe."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_multicomponent_transport import GRI30_PATH
import cantera_python as ct
from cantera_python.constants import Avogadro, Boltzmann, Pi
from cantera_python.transport import _eval_omega22

X = [
    0.269205, 0.000107082, 1.36377e-09, 4.35475e-10, 4.34036e-06,
    0.192249, 3.59356e-13, 2.78061e-12, 4.7406e-18, 4.12955e-17,
    2.58549e-14, 8.96502e-16, 6.09056e-11, 7.56752e-09, 0.192253,
    0.0385036, 1.49596e-08, 2.22378e-08, 1.43096e-13, 1.45312e-15,
    1.96948e-12, 8.41937e-19, 3.18852e-13, 7.93625e-18, 3.20653e-15,
    1.15149e-19, 1.61189e-18, 1.4719e-15, 5.24728e-13, 6.90582e-17,
    6.37248e-12, 5.93728e-11, 2.71219e-09, 2.66645e-06, 6.57142e-11,
    9.52453e-08, 1.26006e-14, 3.49802e-12, 1.19232e-11, 7.17782e-13,
    1.85347e-07, 8.25325e-14, 5.00914e-20, 1.54407e-16, 3.07176e-11,
    4.93198e-08, 4.84792e-12, 0.307675, 0.0, 6.21649e-29,
    8.42393e-28, 6.77865e-18, 2.19225e-16,
]

profile = sys.argv[2] if len(sys.argv) > 2 else "trace-heavy"
temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 1200.0
pressure = float(sys.argv[4]) if len(sys.argv) > 4 else 1e5
if profile not in {"trace-heavy", "uniform", "harmonic"}:
    raise ValueError(f"Unknown composition profile: {profile}")
if profile == "uniform":
    X = None
elif profile == "harmonic":
    X = 1.0 / np.arange(1, len(X) + 1, dtype=float)
    X /= X.sum()

gas = ct.Solution(GRI30_PATH)
gas.transport_model = "multicomponent"
if X is None:
    X = np.full(gas.n_species, 1.0 / gas.n_species)
gas.TPX = temperature, pressure, X
transport = gas.transport
transport.solveLMatrixEquation()
matrix = transport._Lmatrix
rhs = transport._L_soln_b
solution = transport._L_soln_a

out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "audit" / "wave4_internal_matrix_python.txt"
with out_path.open("w", encoding="utf-8", newline="\n") as out:
    out.write(f"{gas.n_species}\n")
    for row in matrix:
        out.write(" ".join(format(float(v), ".17g") for v in row) + "\n")
    out.write(" ".join(format(float(v), ".17g") for v in rhs) + "\n")
    out.write(" ".join(format(float(v), ".17g") for v in solution) + "\n")

diagnostics = {
    "molefracs": transport._molefracs,
    "m_bdiff": transport._m_bdiff,
    "astar": transport._astar,
    "bstar": transport._bstar,
    "cstar": transport._cstar,
    "cinternal": transport._cinternal,
    "rotrelax": transport._rotrelax,
    "viscosity": transport._visc_cached,
    "sigma": transport._diam * 1e-10,
    "epsilon": transport._well * Boltzmann,
    "delta": transport._delta_matrix,
}
raw_viscosity = []
omega22_now = []
tstar_now = []
for k in range(gas.n_species):
    sigma = transport._diam[k] * 1e-10
    tstar = gas.T / transport._well[k]
    tstar_now.append(tstar)
    omega22 = _eval_omega22(tstar, transport._delta_matrix[k, k])
    omega22_now.append(omega22)
    raw_viscosity.append(
        (5.0 / 16.0)
        * np.sqrt(Pi * gas.molecular_weights[k] / Avogadro
                  * Boltzmann * gas.T)
        / (omega22 * Pi * sigma * sigma))
diagnostics["viscosity_raw"] = raw_viscosity
diagnostics["tstar"] = tstar_now
diagnostics["omega22"] = omega22_now
with Path(str(out_path) + ".diag").open("w", encoding="utf-8", newline="\n") as out:
    for name, values in diagnostics.items():
        out.write(name + " " + " ".join(
            format(float(v), ".17g") for v in np.asarray(values).ravel()) + "\n")
