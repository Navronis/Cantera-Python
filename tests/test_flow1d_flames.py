"""Validation of Flow1D / StFlow through real 1D flame configurations.

Tests:
1. Freely propagating premixed flame (`FreeFlame`).
2. Burner-stabilized flame (`BurnerFlame`).
3. Counterflow diffusion flame (`CounterflowDiffusionFlame`).
4. Fixed-temperature continuity flame mode.
5. Radiation-enabled flame configuration.
6. Axisymmetric flame with two-point control.
"""
import numpy as np
import pytest
from pytest import approx

import cantera_python as ct
from cantera_python.oned import FreeFlow, AxisymmetricFlow


class TestRealFlameSimulations:
    def test_free_flame_configuration(self):
        gas = ct.Solution("h2o2.yaml")
        gas.TPX = 300.0, ct.OneAtm, "H2:0.8, O2:0.5, AR:2.0"
        flame = ct.FreeFlame(gas, width=0.05)
        flame.inlet.T = 300.0
        flame.inlet.X = "H2:0.8, O2:0.5, AR:2.0"
        flame.set_initial_guess()

        assert flame.flame.domain_type == "free-flow"
        assert flame.flame.isFree() is True
        assert flame.flame.isStrained() is False
        assert len(flame.grid) > 3
        assert flame.inlet.mdot > 0.0

        # Verify initial velocity satisfies positive mass flow
        assert np.all(flame.velocity > 0.0)

        # Verify fixed temperature location is within bounds
        tfixed = 900.0
        flame.fixed_temperature = tfixed
        assert flame.fixed_temperature == approx(tfixed)
        zfixed = flame.fixed_temperature_location
        assert flame.grid[0] <= zfixed <= flame.grid[-1]

    def test_burner_flame_configuration(self):
        gas = ct.Solution("h2o2.yaml")
        gas.TPX = 300.0, ct.OneAtm, "H2:1.0, O2:1.0"
        burner = ct.BurnerFlame(gas, width=0.02)
        burner.burner.mdot = 0.2
        burner.burner.T = 300.0
        burner.burner.X = "H2:1.0, O2:1.0"
        burner.set_initial_guess()

        assert burner.flame.domain_type == "unstrained-flow"
        assert burner.flame.isFree() is False
        assert burner.flame.isStrained() is False
        assert burner.burner.mdot == approx(0.2)

    def test_counterflow_strained_configuration(self):
        gas = ct.Solution("h2o2.yaml")
        gas.TPX = 300.0, ct.OneAtm, "H2:1.0, AR:1.0"
        flame = ct.CounterflowDiffusionFlame(gas, width=0.03)
        flame.fuel_inlet.mdot = 0.5
        flame.fuel_inlet.X = "H2:1.0"
        flame.oxidizer_inlet.mdot = 0.5
        flame.oxidizer_inlet.X = "O2:1.0"
        flame.set_initial_guess()

        assert flame.flame.domain_type == "axisymmetric-flow"
        assert flame.flame.isFree() is False
        assert flame.flame.isStrained() is True

        # Test two-point control on strained flame
        flame.flame.enableTwoPointControl(True)
        assert flame.flame.twoPointControlEnabled() is True
        flame.flame.setLeftControlPoint(0.005, 500.0)
        flame.flame.setRightControlPoint(0.025, 1200.0)
        assert flame.flame.leftControlPointCoordinate() == approx(0.005)
        assert flame.flame.leftControlPointTemperature() == approx(500.0)
        assert flame.flame.rightControlPointCoordinate() == approx(0.025)
        assert flame.flame.rightControlPointTemperature() == approx(1200.0)

    def test_fixed_temperature_continuity_solve(self):
        gas = ct.Solution("h2o2.yaml")
        gas.TPX = 400.0, 2.0 * ct.OneAtm, "H2:0.5, O2:0.5, AR:3.0"
        flame = ct.FreeFlame(gas, width=0.04)
        flame.inlet.T = 400.0
        flame.inlet.X = "H2:0.5, O2:0.5, AR:3.0"
        flame.set_initial_guess()
        flame.energy_enabled = False
        flame.solve(loglevel=0, refine_grid=False)

        assert not flame.energy_enabled
        rhou = flame.inlet.mdot
        for rhou_j in flame.density * flame.velocity:
            assert rhou_j == approx(rhou, rel=1e-3)

    def test_radiation_coupled_flame(self):
        gas = ct.Solution("h2o2.yaml")
        gas.TPX = 300.0, ct.OneAtm, "H2:1.0, O2:0.5, AR:2.0"
        flame = ct.FreeFlame(gas, width=0.05)
        flame.set_initial_guess()
        flame.radiation_enabled = True
        flame.flame.boundary_emissivities = 0.25, 0.75

        assert flame.radiation_enabled is True
        assert flame.flame.m_epsilon_left == approx(0.25)
        assert flame.flame.m_epsilon_right == approx(0.75)
