"""Generate a pinned-oracle reactor trajectory reference.

Runs the exact pinned native build on the GRI-30 H2/O2 ignition case and
writes a sampled time series to provenance/oracle_reactor_trajectory.json.
"""
import json
import sys

import cantera as ct

out = {"oracle_version": ct.__version__, "oracle_commit": ct.__git_commit__}

gas = ct.Solution('gri30.yaml')
gas.TPX = 1500.0, 101325.0, 'H2:2, O2:1, N2:4'
gas.set_multiplier(1.0)  # ensure kinetics enabled

# Sample the ignition trajectory: fixed small time points.
times = [0.0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 8e-6, 1e-5, 2e-5]
rows = []
prev_t = 0.0
reactor = ct.IdealGasReactor(gas)
net = ct.ReactorNet([reactor])
net.rtol = 1e-9
net.atol = 1e-15
for t in times:
    net.advance(t)
    rows.append({
        "t": t,
        "T": gas.T,
        "P": gas.P,
        "density": gas.density,
        "H2O_X": gas.X[gas.species_index('H2O')],
        "OH_X": gas.X[gas.species_index('OH')],
        "H2_X": gas.X[gas.species_index('H2')],
        "O2_X": gas.X[gas.species_index('O2')],
    })
out["trajectory"] = rows
out["reactor"] = {"type": "IdealGasReactor", "mechanism": "gri30.yaml",
                  "state0": "1500 K, 1 atm, H2:2 O2:1 N2:4"}
out["tolerances"] = {"rtol": 1e-9, "atol": 1e-15}

json.dump(out, sys.stdout, indent=1)
