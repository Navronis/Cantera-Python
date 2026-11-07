"""Unit test for reactor network simultaneous forward sensitivities."""

import math
import numpy as np
import pytest

import cantera_python as ct
from cantera_python.reactor import IdealGasReactor, ReactorNet


def test_reactor_forward_sensitivities():
    gas = ct.Solution(ct.DATA_DIR / "h2o2.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "H2:2.0, O2:1.0, AR:4.0"

    r = IdealGasReactor(gas)
    sim = ReactorNet([r])

    # Add sensitivity parameters: reaction 0 and reaction 1
    sim.add_sensitivity_reaction(0)
    sim.add_sensitivity_reaction(1)

    assert sim.n_sensitivity_params == 2
    assert "reaction 0" in sim.sensitivity_parameter_name(0)
    assert "reaction 1" in sim.sensitivity_parameter_name(1)

    # Advance the simulation
    target_time = 1e-6
    sim.advance(target_time)

    sens = sim.sensitivities()
    assert sens.shape == (sim.n_vars, 2)
    assert np.all(np.isfinite(sens))

    # Sensitivity of temperature with respect to reaction 0 should be finite
    s_T = sim.sensitivity("temperature", 0)
    assert math.isfinite(s_T)
