"""Generate a pinned-native reference for MultiPhase container accessors.

Run with the isolated pinned Cantera environment. This is an oracle utility;
the runtime package and its tests consume only the generated JSON fixture.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cantera as ct

ROOT = Path(__file__).resolve().parents[2]
gas = ct.Solution("gri30.yaml")
gas.TPX = 1200.0, ct.one_atm, "CH4:1.0, O2:2.0, N2:7.52"
graphite = ct.Solution("graphite.yaml")
graphite.TP = gas.TP
mixture = ct.Mixture([(gas, 1.0), (graphite, 0.2)])

elements = []
for phase_index in range(mixture.n_phases):
    phase = mixture.phase(phase_index)
    for index in range(phase.n_elements):
        name = phase.element_name(index)
        if name not in elements:
            elements.append(name)

extension = Path(ct._cantera.__file__)
commit = subprocess.run(
    ["git", "-C", "/home/nebula/cantera-pinned-oracle", "rev-parse", "HEAD"],
    check=True, capture_output=True, text=True,
).stdout.strip()
payload = {
    "pinned_commit": commit,
    "native_version": ct.__version__,
    "native_extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
    "state": {
        "temperature_K": mixture.T,
        "pressure_Pa": mixture.P,
        "phase_names": mixture.phase_names,
        "phase_moles": [mixture.phase_moles(i) for i in range(mixture.n_phases)],
    },
    "element_names": elements,
    "species_names": mixture.species_names,
    "species_moles": mixture.species_moles.tolist(),
    "element_moles": {name: mixture.element_moles(name) for name in elements},
    "phase_charge": [mixture.phase_charge(i) for i in range(mixture.n_phases)],
    "charge": mixture.charge,
}
target = ROOT / "tests" / "fixtures" / "pinned_multiphase_container.json"
target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps({
    "fixture": str(target.relative_to(ROOT)),
    "pinned_commit": commit,
    "native_version": ct.__version__,
    "species_count": len(payload["species_names"]),
    "elements": elements,
    "extension_sha256": payload["native_extension_sha256"],
}, indent=2))
