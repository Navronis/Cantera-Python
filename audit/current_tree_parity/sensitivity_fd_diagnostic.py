"""One-off diagnostic for the full-suite reactor sensitivity discrepancy."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_reactor_conservation import ReactingReactorTests
from cantera_python.reactor import IdealGasConstPressureReactor, ReactorNet

case = ReactingReactorTests()
gas = case._two_species_solution()
reactor = IdealGasConstPressureReactor(gas)
network = ReactorNet(reactor)
reactor.add_sensitivity_reaction(0)
initial = copy.deepcopy(network)
network.advance(1e-2)
augmented = network._sensitivity_derivatives[:, 0]

def trajectory(multiplier):
    run = copy.deepcopy(initial)
    run._reactors[0].kinetics.set_multiplier(multiplier, 0)
    run.advance(1e-2, _compute_sensitivities=False)
    return np.asarray(run._get_global_state())

baseline = trajectory(1.0)
results = {"augmented": augmented.tolist(), "steps": {}}
for step in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    forward = (trajectory(1.0 + step) - baseline) / step
    backward = (baseline - trajectory(1.0 - step)) / step
    centered = (trajectory(1.0 + step) - trajectory(1.0 - step)) / (2.0 * step)
    results["steps"][str(step)] = {
        "forward": forward.tolist(),
        "backward": backward.tolist(),
        "centered": centered.tolist(),
        "centered_max_rel": float(np.max(np.abs(centered - augmented)
                                         / np.maximum(np.abs(augmented), 1e-30))),
    }
print(json.dumps(results, indent=2))
