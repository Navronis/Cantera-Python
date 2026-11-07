"""Create a pinned-native public-property fixture from the current C++ probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cantera_python import Solution
from cantera_python.constants import GasConstant
NATIVE = AUDIT / "wave4_native_trace_heavy.txt"
DIAG = Path(str(NATIVE) + ".diag")

values = np.fromstring(NATIVE.read_text(encoding="utf-8"), sep=" ")
nsp = int(values[0])
size = 3 * nsp
matrix_end = 1 + size * size
rhs = values[matrix_end:matrix_end + size]
solution = values[matrix_end + size:matrix_end + 2 * size]
diagnostics = {}
for line in DIAG.read_text(encoding="utf-8").splitlines():
    key, raw = line.split(maxsplit=1)
    diagnostics[key] = np.fromstring(raw, sep=" ")

gas = Solution(ROOT.parent / "cantera-original" / "data" / "gri30.yaml")
if gas.n_species != nsp:
    raise RuntimeError("Native probe and Python mechanism species counts differ")
x = diagnostics["molefracs"]
mw = np.asarray(gas.molecular_weights, dtype=float)
conductivity = -4.0 * sum(rhs[nsp:3*nsp] * solution[nsp:3*nsp])
thermal_diffusion = (1.6 / GasConstant) * mw * x * solution[:nsp]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

identity = json.loads((ROOT / "audit" / "native_oracle_identity.json").read_text())
fixture = {
    "provenance": {
        "pinned_commit": identity["exact_commit"],
        "native_extension_sha256": identity["native_extension_sha256"],
        "native_shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        "native_runner_sha256": sha(ROOT / "tools" / "run_native_multitransport_matrix_probe"),
        "native_runner_source_sha256": sha(ROOT / "tools" / "run_native_multitransport_matrix_probe.cpp"),
        "native_probe_output_sha256": sha(NATIVE),
        "native_probe_diag_sha256": sha(DIAG),
        "python_probe_output_sha256": sha(AUDIT / "wave4_python_trace_heavy.txt"),
        "native_matrix_reference": "audit/current_tree_parity/wave4_native_trace_heavy.txt",
    },
    "state": {
        "temperature_K": 1200.0,
        "pressure_Pa": 100000.0,
        "mole_fractions": x.tolist(),
        "species_names": list(gas.species_names),
    },
    "source_formula": {
        "file": "src/transport/MultiTransport.cpp",
        "conductivity": "-4 * sum(b[k+nsp] * a[k+nsp], k=0..2*nsp-1)",
        "thermal_diffusion": "(1.6 / GasConstant) * molecular_weight[k] * X[k] * a[k]",
    },
    "native_expected": {
        "thermal_conductivity_W_m_K": float(conductivity),
        "thermal_diffusion_coefficients": thermal_diffusion.tolist(),
        "matrix_condition_2": float(np.linalg.cond(values[1:matrix_end].reshape(size, size))),
        "native_residual_inf": float(np.linalg.norm(
            values[1:matrix_end].reshape(size, size) @ solution - rhs, ord=np.inf)),
    },
}
out = ROOT / "tests" / "fixtures" / "pinned_multitransport_trace_heavy.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
print(out.relative_to(ROOT))
print("thermal conductivity:", fixture["native_expected"]["thermal_conductivity_W_m_K"])
print("thermal diffusion coefficient count:", len(thermal_diffusion))
