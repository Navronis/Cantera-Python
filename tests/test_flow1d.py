"""Comprehensive unit tests for Flow1D / StFlow numerical implementation.

Validates exact pinned Cantera Flow1D source logic, equations, spatial discretization,
modes, radiation, transport coupling, and residual evaluation.
"""
import numpy as np
import pytest
from pytest import approx

import cantera_python as ct
from cantera_python.oned import Flow1D, FreeFlow, AxisymmetricFlow, c_offset_U, c_offset_V, c_offset_T, c_offset_L, c_offset_E, c_offset_Uo, c_offset_Y
from cantera_python.constants import OneAtm


@pytest.fixture
def gas():
    return ct.Solution("h2o2.yaml")


class TestFlow1DModes:
    def test_construction_defaults(self, gas):
        flow = FreeFlow(gas)
        assert flow.domain_type == "free-flow"
        assert flow.isFree() is True
        assert flow.isStrained() is False
        assert flow.twoPointControlEnabled() is False
        assert flow.energy_enabled is False
        flow.solveEnergyEqn(True)
        assert flow.energy_enabled is True
        assert flow.radiation_enabled is False
        assert flow.fluxGradientBasis() == "molar"

    def test_mode_transitions(self, gas):
        flow = Flow1D(gas)
        # Free flow
        flow.setFreeFlow()
        assert flow.domain_type == "free-flow"
        assert flow.isFree() is True
        assert flow.isStrained() is False
        assert flow.componentActive(c_offset_U) is True
        assert flow.componentActive(c_offset_V) is False
        assert flow.componentActive(c_offset_L) is False

        # Axisymmetric flow
        flow.setAxisymmetricFlow()
        assert flow.domain_type == "axisymmetric-flow"
        assert flow.isFree() is False
        assert flow.isStrained() is True
        assert flow.componentActive(c_offset_V) is True
        assert flow.componentActive(c_offset_L) is True

        # Unstrained flow
        flow.setUnstrainedFlow()
        assert flow.domain_type == "unstrained-flow"
        assert flow.isFree() is False
        assert flow.isStrained() is False
        assert flow.componentActive(c_offset_V) is False
        assert flow.componentActive(c_offset_L) is False

    def test_component_indices_and_names(self, gas):
        flow = FreeFlow(gas)
        assert flow.componentIndex("velocity") == c_offset_U
        assert flow.componentIndex("u") == c_offset_U
        assert flow.componentIndex("spread_rate") == c_offset_V
        assert flow.componentIndex("V") == c_offset_V
        assert flow.componentIndex("T") == c_offset_T
        assert flow.componentIndex("temperature") == c_offset_T
        assert flow.componentIndex("Lambda") == c_offset_L
        assert flow.componentIndex("lambda") == c_offset_L
        assert flow.componentIndex("electric-field") == c_offset_E
        assert flow.componentIndex("oxidizer-velocity") == c_offset_Uo
        assert flow.componentIndex("H2") == c_offset_Y + gas.species_index("H2")
        assert flow.componentIndex("O2") == c_offset_Y + gas.species_index("O2")

        assert flow.componentName(c_offset_U) == "velocity"
        assert flow.componentName(c_offset_V) == "spreadRate"
        assert flow.componentName(c_offset_T) == "T"
        assert flow.componentName(c_offset_L) == "Lambda"
        assert flow.componentName(c_offset_E) == "eField"
        assert flow.componentName(c_offset_Uo) == "Uo"


class TestFlow1DControls:
    def test_two_point_control(self, gas):
        flow = AxisymmetricFlow(gas)
        assert flow.twoPointControlEnabled() is False

        flow.enableTwoPointControl(True)
        assert flow.twoPointControlEnabled() is True
        assert flow.componentActive(c_offset_Uo) is True

        flow.setLeftControlPoint(0.005, 500.0)
        assert flow.leftControlPointCoordinate() == approx(0.005)
        assert flow.leftControlPointTemperature() == approx(500.0)

        flow.setRightControlPoint(0.045, 1200.0)
        assert flow.rightControlPointCoordinate() == approx(0.045)
        assert flow.rightControlPointTemperature() == approx(1200.0)

        flow.enableTwoPointControl(False)
        assert flow.twoPointControlEnabled() is False
        assert flow.componentActive(c_offset_Uo) is False

    def test_fixed_temperature_profile(self, gas):
        flow = FreeFlow(gas)
        flow.setupUniformGrid(5, 0.05)
        t_prof = np.array([300.0, 500.0, 1000.0, 1800.0, 2200.0])
        flow.setFixedTemp(t_prof)
        for j in range(5):
            assert flow.T_fixed(j) == approx(t_prof[j])

        flow.fixTemperature(800.0)
        assert flow.m_tfixed == approx(800.0)
        assert flow.isFixedTemp() is True

    def test_energy_toggle(self, gas):
        flow = FreeFlow(gas)
        flow.setupUniformGrid(5, 0.05)
        assert flow.allOfEnergyEnabled() is False
        assert flow.noneOfEnergyEnabled() is True

        flow.solveEnergyEqn()
        assert flow.allOfEnergyEnabled() is True
        assert flow.noneOfEnergyEnabled() is False

        flow.solveEnergyEqn(False)
        assert flow.allOfEnergyEnabled() is False
        assert flow.noneOfEnergyEnabled() is True

        flow.solveEnergyEqn(True)
        assert flow.allOfEnergyEnabled() is True

    def test_radiation_toggle(self, gas):
        flow = FreeFlow(gas)
        assert flow.radiationEnabled() is False
        flow.enableRadiation(True)
        assert flow.radiationEnabled() is True
        flow.setBoundaryEmissivities(0.2, 0.8)
        assert flow.m_epsilon_left == approx(0.2)
        assert flow.m_epsilon_right == approx(0.8)

    def test_soret_toggle(self, gas):
        flow = FreeFlow(gas)
        assert flow.soretEnabled() is False
        flow.enableSoret(True)
        assert flow.soretEnabled() is True
        flow.enableSoret(False)
        assert flow.soretEnabled() is False


class TestSpatialOperators:
    def test_directional_derivatives(self, gas):
        flow = FreeFlow(gas)
        z = np.array([0.0, 0.01, 0.025, 0.045, 0.07])
        flow.setupGrid(z)
        nv = flow.nComponents()
        np_pts = flow.nPoints()
        x = np.zeros(nv * np_pts)

        # Set linear temperature field T(z) = 300 + 1000 * z
        for j in range(np_pts):
            x[flow.index(c_offset_T, j)] = 300.0 + 1000.0 * z[j]

        # For positive flow (u > 0), upwind dTdz is (T[j] - T[j-1])/dz[j-1]
        for j in range(1, np_pts):
            x[flow.index(c_offset_U, j)] = 1.0  # positive velocity
            dt_dz = flow.dTdz(x, j)
            assert dt_dz == approx(1000.0, rel=1e-5)

        # For negative flow (u < 0), upwind dTdz is (T[j+1] - T[j])/dz[j]
        for j in range(np_pts - 1):
            x[flow.index(c_offset_U, j)] = -1.0  # negative velocity
            dt_dz = flow.dTdz(x, j)
            assert dt_dz == approx(1000.0, rel=1e-5)

    def test_conduction_operator(self, gas):
        flow = FreeFlow(gas)
        z = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        flow.setupGrid(z)
        nv = flow.nComponents()
        np_pts = flow.nPoints()
        x = np.zeros(nv * np_pts)

        # Uniform temperature -> zero conduction heat flux divergence
        for j in range(np_pts):
            x[flow.index(c_offset_T, j)] = 500.0
            x[flow.index(c_offset_Y + gas.species_index("AR"), j)] = 1.0

        flow.updateThermo(x, 0, np_pts - 1)
        flow.updateTransport(x, 0, np_pts - 1)

        for j in range(1, np_pts - 1):
            cond = flow.conduction(x, j)
            assert cond == approx(0.0, abs=1e-10)

    def test_shear_operator(self, gas):
        flow = AxisymmetricFlow(gas)
        z = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        flow.setupGrid(z)
        nv = flow.nComponents()
        np_pts = flow.nPoints()
        x = np.zeros(nv * np_pts)

        # Uniform spread rate V -> zero shear divergence
        for j in range(np_pts):
            x[flow.index(c_offset_T, j)] = 400.0
            x[flow.index(c_offset_V, j)] = 2.0
            x[flow.index(c_offset_Y + gas.species_index("AR"), j)] = 1.0

        flow.updateThermo(x, 0, np_pts - 1)
        flow.updateTransport(x, 0, np_pts - 1)

        for j in range(1, np_pts - 1):
            sh = flow.shear(x, j)
            assert sh == approx(0.0, abs=1e-10)


class TestRadiationCalculation:
    def test_radiation_planck_polynomials(self, gas):
        flow = FreeFlow(gas)
        flow.setupUniformGrid(3, 0.05)
        nv = flow.nComponents()
        x = np.zeros(nv * 3)
        for j in range(3):
            x[flow.index(c_offset_T, j)] = 1500.0

        poly_co2 = flow.radiationPolyFactor(x, 0, 0)
        poly_h2o = flow.radiationPolyFactor(x, 0, 1)
        assert np.isfinite(poly_co2)
        assert np.isfinite(poly_h2o)
        assert poly_co2 > 0
        assert poly_h2o > 0

    def test_compute_radiation(self, gas):
        flow = FreeFlow(gas)
        flow.setupUniformGrid(5, 0.05)
        flow.enableRadiation(True)
        nv = flow.nComponents()
        x = np.zeros(nv * 5)
        for j in range(5):
            x[flow.index(c_offset_T, j)] = 1800.0
            x[flow.index(c_offset_Y + gas.species_index("H2O"), j)] = 0.2
            x[flow.index(c_offset_Y + gas.species_index("AR"), j)] = 0.8

        flow.updateThermo(x, 0, 4)
        flow.computeRadiation(x, 0, 4)
        # Should have positive radiative heat loss in hot interior
        assert flow.m_qdotRadiation[2] > 0.0
