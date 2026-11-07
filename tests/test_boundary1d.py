"""Unit tests for 1D boundary conditions (Boundary1D and derived classes).

Tests all 8 canonical boundary conditions:
- Boundary1D (base)
- Inlet1D
- Outlet1D
- OutletRes1D
- Empty1D
- Symm1D / SymmetryPlane1D
- Surf1D / Surface1D
- ReactingSurf1D / ReactingSurface1D
"""

import numpy as np
import pytest
from pytest import approx

import cantera_python as ct
from cantera_python.oned import CallableFloat


class TestBoundary1DBase:
    def test_default_values(self):
        b = ct.Boundary1D()
        assert b.domain_type == "boundary"
        assert b.domainType() == "boundary"
        assert b.is_connector is True
        assert b.isConnector() is True
        assert b.n_points == 1
        assert b.n_components == 1
        assert b.nSpecies() == 0
        assert b.n_species == 0
        assert b.T == 0.0
        assert b.temperature == 0.0
        assert b.mdot == 0.0

    def test_callable_float_properties(self):
        b = ct.Boundary1D()
        b.temperature = 500.0
        b.mdot = 2.5
        # Can be called as a method
        assert b.temperature() == 500.0
        assert b.T() == 500.0
        assert b.mdot() == 2.5
        # Can be used directly in arithmetic
        assert b.temperature / 2.0 == 250.0
        assert b.T + 50.0 == 550.0
        assert b.mdot * 4.0 == 10.0
        assert isinstance(b.temperature, CallableFloat)
        assert isinstance(b.T, CallableFloat)
        assert isinstance(b.mdot, CallableFloat)

    def test_unexpected_kwargs(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ct.Boundary1D(unknown_arg="val")

    def test_unimplemented_base_methods(self):
        b = ct.Boundary1D()
        with pytest.raises(NotImplementedError):
            b.setMoleFractions("H2:1.0")
        with pytest.raises(NotImplementedError):
            b.massFraction(0)
        with pytest.raises(NotImplementedError):
            b.setSpreadRate(1.0)
        with pytest.raises(NotImplementedError):
            b.spreadRate()

    def test_init_without_container_raises(self):
        b = ct.Boundary1D()
        with pytest.raises(ct.CanteraError, match="install in container before calling init"):
            b.init()


class TestInlet1D:
    def test_instantiate_default(self):
        gas = ct.Solution("h2o2.yaml")
        inlet = ct.Inlet1D(gas, name="inlet_test")
        assert inlet.domain_type == "inlet"
        assert inlet.domainType() == "inlet"
        assert inlet.is_connector is True
        assert inlet.name == "inlet_test"
        assert inlet.n_points == 1
        assert inlet.n_species == gas.n_species
        assert inlet.nSpecies() == gas.n_species
        assert inlet.temperature == approx(gas.T)
        assert inlet.spread_rate == 0.0
        assert inlet.spreadRate() == 0.0

    def test_callable_spread_rate(self):
        gas = ct.Solution("h2o2.yaml")
        inlet = ct.Inlet1D(gas)
        inlet.spread_rate = 12.5
        assert inlet.spread_rate() == 12.5
        assert inlet.spreadRate() == 12.5
        assert inlet.spread_rate * 2.0 == 25.0
        assert isinstance(inlet.spread_rate, CallableFloat)

    def test_set_mole_fractions(self):
        gas = ct.Solution("h2o2.yaml")
        inlet = ct.Inlet1D(gas)
        inlet.setMoleFractions("H2:0.5, O2:0.5")
        ih2 = gas.species_index("H2")
        io2 = gas.species_index("O2")
        assert inlet.massFraction(ih2) > 0.0
        assert inlet.massFraction(io2) > 0.0
        assert inlet.massFraction(ih2) + inlet.massFraction(io2) == approx(1.0)

    def test_bad_phase_argument(self):
        with pytest.raises(TypeError, match="Argument 'phase' has incorrect type"):
            ct.Inlet1D(None)

    def test_temperature_bound_adjustment(self):
        gas = ct.Solution("h2o2.yaml")
        inlet = ct.Inlet1D(gas)
        flow = ct.FreeFlow(gas)
        sim = ct.Sim1D((inlet, flow))
        inlet.setTemperature(150.0)
        assert flow.lowerBound(flow.component_index("T")) <= 150.0


class TestOutlet1D:
    def test_instantiate(self):
        outlet = ct.Outlet1D(name="outlet_test")
        assert outlet.domain_type == "outlet"
        assert outlet.domainType() == "outlet"
        assert outlet.is_connector is True
        assert outlet.name == "outlet_test"
        assert outlet.n_points == 1

    def test_init_without_container_raises(self):
        outlet = ct.Outlet1D()
        with pytest.raises(ct.CanteraError, match="install in container before calling init"):
            outlet.init()

    def test_init_without_flow_raises(self):
        outlet = ct.Outlet1D()
        with pytest.raises(ct.CanteraError, match="Outlet1D is not connected"):
            ct.Sim1D((outlet,))

    def test_init_with_right_flow_raises(self):
        gas = ct.Solution("h2o2.yaml")
        flow = ct.FreeFlow(gas)
        outlet = ct.Outlet1D()
        with pytest.raises(ct.CanteraError, match="Left outlets with right-to-left flow are not supported"):
            ct.Sim1D((outlet, flow))


class TestOutletRes1D:
    def test_instantiate(self):
        outlet_res = ct.OutletRes1D(name="outlet_res_test")
        assert outlet_res.domain_type == "outlet-reservoir"
        assert outlet_res.domainType() == "outlet-reservoir"
        assert outlet_res.is_connector is True
        assert outlet_res.name == "outlet_res_test"

    def test_init_without_container_raises(self):
        outlet_res = ct.OutletRes1D()
        with pytest.raises(ct.CanteraError, match="install in container before calling init"):
            outlet_res.init()

    def test_init_without_flow_raises(self):
        outlet_res = ct.OutletRes1D()
        with pytest.raises(ct.CanteraError, match="no flow"):
            ct.Sim1D((outlet_res,))

    def test_reservoir_composition(self):
        gas = ct.Solution("h2o2.yaml")
        flow = ct.FreeFlow(gas)
        outlet_res = ct.OutletRes1D()
        outlet_res.setMoleFractions("H2:0.8, O2:0.2")
        sim = ct.Sim1D((flow, outlet_res))
        ih2 = gas.species_index("H2")
        io2 = gas.species_index("O2")
        assert outlet_res.massFraction(ih2) > 0.0
        assert outlet_res.massFraction(io2) > 0.0
        assert outlet_res.nSpecies() == gas.n_species


class TestEmpty1D:
    def test_instantiate_and_eval(self):
        empty = ct.Empty1D(name="empty_test")
        assert empty.domain_type == "empty"
        assert empty.domainType() == "empty"
        assert empty.is_connector is True
        assert empty.name == "empty_test"
        assert empty.n_points == 1
        # eval is a no-op and should not raise
        xg = np.zeros(10)
        rg = np.zeros(10)
        diagg = np.zeros(10)
        empty.eval(0, xg, rg, diagg, 0.0)
        assert np.all(rg == 0.0)

    def test_in_sim1d_container(self):
        empty = ct.Empty1D()
        sim = ct.Sim1D((empty,))
        assert sim.domain(0) is empty


class TestSymm1D:
    def test_alias(self):
        assert ct.SymmetryPlane1D is ct.Symm1D

    def test_instantiate(self):
        symm = ct.Symm1D(name="symm_test")
        assert symm.domain_type == "symmetry-plane"
        assert symm.domainType() == "symmetry-plane"
        assert symm.is_connector is True
        assert symm.name == "symm_test"
        assert symm.n_points == 1

    def test_in_sim1d_container(self):
        gas = ct.Solution("h2o2.yaml")
        symm = ct.Symm1D()
        flow = ct.FreeFlow(gas)
        sim = ct.Sim1D((symm, flow))
        assert sim.domain(0) is symm


class TestSurf1D:
    def test_alias(self):
        assert ct.Surface1D is ct.Surf1D

    def test_instantiate(self):
        surf = ct.Surf1D(name="surf_test")
        assert surf.domain_type == "surface"
        assert surf.domainType() == "surface"
        assert surf.is_connector is True
        assert surf.name == "surf_test"
        assert surf.n_points == 1
        surf.temperature = 1200.0
        assert surf.temperature == 1200.0
        assert surf.T == 1200.0


class TestReactingSurf1D:
    def test_alias(self):
        assert ct.ReactingSurface1D is ct.ReactingSurf1D

    def test_bad_phase_raises(self):
        with pytest.raises(TypeError, match="Argument 'phase' has incorrect type"):
            ct.ReactingSurf1D(None)

        gas = ct.Solution("h2o2.yaml")
        with pytest.raises(ct.CanteraError, match="Detected incompatible ThermoPhase type"):
            ct.ReactingSurf1D(gas)

    def test_instantiate_with_interface(self):
        gas = ct.Solution("diamond.yaml", "gas")
        solid = ct.Solution("diamond.yaml", "diamond")
        interface = ct.Solution("diamond.yaml", "diamond_100", (gas, solid))
        surf = ct.ReactingSurf1D(interface, name="diamond_surf")

        assert surf.domain_type == "reacting-surface"
        assert surf.domainType() == "reacting-surface"
        assert surf.is_connector is True
        assert surf.name == "diamond_surf"
        assert surf.n_species == 8
        assert surf.nSpecies() == 8
        assert surf.coverageEnabled() is True
        surf.enableCoverageEquations(False)
        assert surf.coverageEnabled() is False
        surf.enableCoverageEquations(True)
        assert surf.coverageEnabled() is True

    def test_component_names_and_indices(self):
        gas = ct.Solution("diamond.yaml", "gas")
        solid = ct.Solution("diamond.yaml", "diamond")
        interface = ct.Solution("diamond.yaml", "diamond_100", (gas, solid))
        surf = ct.ReactingSurf1D(interface)

        names = [surf.componentName(i) for i in range(surf.n_species)]
        assert len(names) == 8
        for i, name in enumerate(names):
            assert surf.componentIndex(name) == i

        with pytest.raises(IndexError, match="out of range"):
            surf.componentName(100)

    def test_reset_bad_values(self):
        gas = ct.Solution("diamond.yaml", "gas")
        solid = ct.Solution("diamond.yaml", "diamond")
        interface = ct.Solution("diamond.yaml", "diamond_100", (gas, solid))
        surf = ct.ReactingSurf1D(interface)
        xg = np.array([2.0, 3.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        surf.resetBadValues(xg)
        assert np.sum(xg) == approx(1.0)
        assert xg[0] == approx(0.2)
        assert xg[1] == approx(0.3)
        assert xg[2] == approx(0.5)


class TestBoundaryCouplingAndTimeInteg:
    def test_time_integration_slicing(self):
        gas = ct.Solution("ptcombust.yaml", "gas")
        surf_phase = ct.Interface("ptcombust.yaml", "Pt_surf", [gas])
        flow = ct.FreeFlow(gas)
        surf = ct.ReactingSurf1D(surf_phase)
        sim = ct.Sim1D((flow, surf))
        # Test initTimeInteg local slicing and prevSoln on ReactingSurf1D
        x = np.zeros(sim.size())
        loc = surf.loc()
        sz = surf.size()
        assert sz == surf.n_species
        x[loc + 2] = 42.0
        surf.initTimeInteg(0.01, x)
        assert surf.prevSoln(2, 0) == 42.0

    def test_inlet_flow_residual_eval(self):
        gas = ct.Solution("h2o2.yaml")
        gas.TPX = 300, 101325, "H2:2, O2:1"
        inlet = ct.Inlet1D(gas)
        inlet.mdot = 0.1
        flow = ct.FreeFlow(gas)
        outlet = ct.Outlet1D()
        sim = ct.Sim1D((inlet, flow, outlet))

        xg = np.ones(sim.size()) * 300.0
        rg = np.zeros(sim.size())
        diagg = np.zeros(sim.size())
        inlet.eval(0, xg, rg, diagg, 0.0)
        outlet.eval(flow.nPoints() - 1, xg, rg, diagg, 0.0)
        assert np.all(np.isfinite(rg))
