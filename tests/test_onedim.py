"""Unit tests for 1D reacting flow and boundary domains.

Ported from Cantera native test suite test/python/test_onedim.py.
"""
import copy
import pickle
import numpy as np
import pytest
from pytest import approx

import cantera_python as ct
from cantera_python.numerics import BandMatrix


class TestOnedim:
    def test_finite_difference_jacobian_is_solved_in_python_banded_storage(self):
        matrix = np.array([
            [4.0, -1.0, 0.0, 0.0],
            [-2.0, 5.0, 1.0, 0.0],
            [0.0, 2.0, 6.0, -1.0],
            [0.0, 0.0, 3.0, 7.0],
        ])

        class LinearResidual:
            def eval(self, _domain, state, residual, rdt=0.0):
                residual[:] = matrix @ state + rdt * state

        system = LinearResidual()
        jac = ct.Sim1D.eval_jacobian(system, np.array([1.0, 2.0, 3.0, 4.0]))
        assert isinstance(jac, BandMatrix)
        assert (jac._kl, jac._ku) == (1, 1)
        jac.factor()
        expected_step = np.array([0.5, -1.0, 2.0, 0.25])
        rhs = matrix @ expected_step
        assert np.allclose(jac.solve(rhs), expected_step, rtol=1e-8, atol=1e-8)

    def test_instantiate(self):
        gas = ct.Solution("h2o2.yaml")
        free = ct.FreeFlow(gas)
        axi = ct.AxisymmetricFlow(gas)
        assert free.domain_type == "free-flow"
        assert axi.domain_type == "axisymmetric-flow"

    def test_badInstantiate(self):
        solid = ct.Solution("diamond.yaml", "diamond")
        with pytest.raises(ct.CanteraError,
                           match="An appropriate transport model\nshould be set when instantiating"):
            ct.FreeFlow(solid)

    def test_instantiateSurface(self):
        gas = ct.Solution("diamond.yaml", "gas")
        solid = ct.Solution("diamond.yaml", "diamond")
        interface = ct.Solution("diamond.yaml", "diamond_100", (gas, solid))

        surface = ct.ReactingSurface1D(phase=interface)
        assert surface.n_species == 8

    def test_boundaryProperties(self):
        gas1 = ct.Solution("h2o2.yaml")
        gas2 = ct.Solution("h2o2.yaml")
        inlet = ct.Inlet1D(gas1, name="spam")
        flame = ct.FreeFlow(gas1, name="eggs")
        sim = ct.Sim1D((inlet, flame))

        assert sim.domain_index("spam") == 0
        assert sim.domain_index(inlet) == 0
        assert sim.domain_index("eggs") == 1
        assert inlet.name == "spam"
        with pytest.raises(ct.CanteraError, match="Domain 'bacon' not found"):
            sim.domain_index("bacon")

        gas2.TPX = 400, 101325, 'H2:0.3, O2:0.5, AR:0.2'
        Xref = gas2.X
        Yref = gas2.Y
        inlet.Y = Yref

        assert inlet.Y == approx(Yref)
        assert inlet.X == approx(Xref)

        gas2.TPX = 400, 101325, 'H2:0.5, O2:0.2, AR:0.3'
        Xref = gas2.X
        Yref = gas2.Y
        inlet.X = Xref
        assert inlet.X == approx(Xref)
        assert inlet.Y == approx(Yref)

        inlet.X = {'H2': 0.3, 'O2': 0.5, 'AR': 0.2}
        assert inlet.X[gas2.species_index('H2')] == approx(0.3)

    def test_grid_check(self):
        gas = ct.Solution("h2o2.yaml")
        flame = ct.FreeFlow(gas)

        with pytest.raises(ct.CanteraError, match='monotonically'):
            flame.grid = [0, 0.1, 0.1, 0.2]

        with pytest.raises(ct.CanteraError, match='monotonically'):
            flame.grid = [0, 0.1, 0.2, 0.05]

    def test_unpicklable(self):
        gas = ct.Solution("h2o2.yaml")
        flame = ct.FreeFlow(gas)
        with pytest.raises(NotImplementedError):
            pickle.dumps(flame)

    def test_uncopyable(self):
        gas = ct.Solution("h2o2.yaml")
        flame = ct.FreeFlow(gas)
        with pytest.raises(NotImplementedError):
            copy.copy(flame)

    def test_exceptions(self):
        with pytest.raises(TypeError, match="Argument 'phase' has incorrect type"):
            ct.Inlet1D(None)
        gas = ct.Solution("h2o2.yaml")
        with pytest.raises(ct.CanteraError, match="incompatible ThermoPhase type"):
            ct.ReactingSurface1D(gas)
        interface = ct.Solution("diamond.yaml", "diamond_100")
        with pytest.raises(TypeError, match="unexpected keyword"):
            ct.ReactingSurface1D(interface, foo="bar")
        surf = ct.ReactingSurface1D(interface)
        assert surf.n_species == 8

    def test_invalid_property(self):
        gas1 = ct.Solution("h2o2.yaml")
        inlet = ct.Inlet1D(name='something', phase=gas1)
        flame = ct.FreeFlow(gas1)
        sim = ct.Sim1D((inlet, flame))

        for x in (inlet, flame, sim):
            with pytest.raises(AttributeError):
                x.foobar = 300
            with pytest.raises(AttributeError):
                _ = x.foobar

    def test_tolerances(self):
        gas = ct.Solution("h2o2.yaml")
        left = ct.Inlet1D(gas)
        flame = ct.FreeFlow(gas)
        right = ct.Outlet1D(gas)
        sim = ct.Sim1D((left, flame, right))

        with pytest.raises(ct.CanteraError, match="Component 'foobar' not found"):
            flame.set_steady_tolerances(foobar=(3e-4, 3e-6))

        flame.set_steady_tolerances(default=(5e-3, 5e-5),
                                    T=(3e-4, 3e-6),
                                    Y=(7e-7, 7e-9))
        flame.set_transient_tolerances(default=(6e-3, 6e-5),
                                       T=(4e-4, 4e-6),
                                       Y=(2e-7, 2e-9))

        # Flow domain
        atol_ss = set(flame.steady_abstol())
        atol_ts = set(flame.transient_abstol())
        rtol_ss = set(flame.steady_reltol())
        rtol_ts = set(flame.transient_reltol())

        assert atol_ss == set((5e-5, 3e-6, 7e-9))
        assert atol_ts == set((6e-5, 4e-6, 2e-9))
        assert rtol_ss == set((5e-3, 3e-4, 7e-7))
        assert rtol_ts == set((6e-3, 4e-4, 2e-7))

    def test_time_step_control_option_validation(self):
        gas = ct.Solution("h2o2.yaml")
        left = ct.Inlet1D(gas)
        flame = ct.FreeFlow(gas)
        right = ct.Outlet1D(gas)
        sim = ct.Sim1D((left, flame, right))

        with pytest.raises(ct.CanteraError, match=">= 1.0"):
            sim.time_step_growth_factor = 0.9

        with pytest.raises(ct.CanteraError, match="finite and >= 1.0"):
            sim.time_step_growth_factor = float("inf")

        with pytest.raises(ct.CanteraError, match="finite and >= 1.0"):
            sim.time_step_growth_factor = float("nan")

        with pytest.raises(ct.CanteraError, match="must be one of"):
            sim.time_step_growth_strategy = "not-a-strategy"

        with pytest.raises(ct.CanteraError, match=">= 0"):
            sim.time_step_regrid = -1

    def test_switch_transport(self):
        gas = ct.Solution('h2o2.yaml')
        gas.set_equivalence_ratio(0.9, 'H2', 'O2:0.21, N2:0.79')
        flame = ct.FreeFlame(gas, width=0.1)
        flame.set_initial_guess()

        assert gas.transport_model == flame.transport_model == 'mixture-averaged'

        flame.transport_model = 'unity-Lewis-number'
        assert gas.transport_model == flame.transport_model == 'unity-Lewis-number'
        Dkm_flame = flame.mix_diff_coeffs
        assert all(Dkm_flame[1, :] == Dkm_flame[2, :])

        gas.transport_model = 'multicomponent'
        assert flame.transport_model == 'multicomponent'

        with pytest.raises(ct.CanteraError, match="Invalid Transport model"):
            flame.transport_model = 'none'

        gas.transport_model = 'unity-Lewis-number'
        with pytest.raises(ct.CanteraError, match="Invalid Transport model"):
            gas.transport_model = 'none'

    def test_width_grid(self):
        gas = ct.Solution('h2o2.yaml')
        for cls in ct.FlameBase.__subclasses__():
            with pytest.raises(ValueError, match="mutually exclusive"):
                _ = cls(gas, grid=[0, 0.1, 0.2], width=0.4)


class TestFreeFlame:
    def create_sim(self, p, Tin, reactants, width=0.05, mech="h2o2.yaml"):
        self.gas = ct.Solution(mech)
        self.gas.TPX = Tin, p, reactants
        self.sim = ct.FreeFlame(self.gas, width=width)
        self.sim.inlet.T = Tin
        self.sim.inlet.X = reactants

    def test_flow_type(self):
        Tin = 300
        p = ct.OneAtm
        reactants = 'H2:0.65, O2:0.5, AR:2'
        self.create_sim(p, Tin, reactants, width=0.0001)
        assert self.sim.flame.domain_type == "free-flow"

    def test_fixed_temperature(self):
        Tin = 300
        p = ct.OneAtm
        reactants = 'H2:0.65, O2:0.5, AR:2'
        self.create_sim(p, Tin, reactants, width=0.0001)
        self.sim.set_initial_guess()
        zvec = self.sim.grid
        tvec = self.sim.T
        tfixed = 900.0
        self.sim.fixed_temperature = tfixed
        zfixed = np.interp(tfixed, tvec, zvec)
        assert self.sim.fixed_temperature == approx(tfixed)
        assert self.sim.fixed_temperature_location == approx(zfixed)

        assert np.allclose(self.sim.T, self.sim.flame.temperature)
        zeros = self.sim.T * 0
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("spread_rate", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("spreadRate", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("radial-pressure-gradient", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("L", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("Lambda", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("electric-field", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("E", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("eField", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("oxidizer-velocity", zeros)
        with pytest.raises(ct.CanteraError, match="is not used by 'free-flow'"):
            self.sim.flame.set_values("Uo", zeros)

    def test_refine_criteria_boundscheck(self):
        self.create_sim(ct.OneAtm, 300.0, 'H2:1.1, O2:1, AR:5')
        good = [3.0, 0.1, 0.2, 0.05]
        bad = [1.2, 1.1, -2, 0.3]

        self.sim.set_refine_criteria(*good)
        for i in range(4):
            with pytest.raises(ct.CanteraError, match='Refiner::setCriteria'):
                vals = list(good)
                vals[i] = bad[i]
                self.sim.set_refine_criteria(*vals)

    def test_refine_criteria(self):
        self.create_sim(ct.OneAtm, 300.0, 'H2:1.1, O2:1, AR:5')
        vals = {'ratio': 3.0, 'slope': 0.1, 'curve': 0.2, 'prune': 0.05}
        self.sim.set_refine_criteria(**vals)
        check = self.sim.get_refine_criteria()
        assert vals == check

    def test_settings(self):
        self.create_sim(p=ct.OneAtm, Tin=400, reactants='H2:0.8, O2:0.5', width=0.1)
        self.sim.set_initial_guess()
        sim = self.sim

        new_keys = {
            "type", "points", "tolerances", "transport-model", "phase",
            "radiation-enabled", "energy-enabled", "Soret-enabled",
            "refine-criteria", "fixed-point"}
        settings = sim.flame.settings
        for k in new_keys:
            assert k in settings

        assert settings["radiation-enabled"] is False

        sim.flame.boundary_emissivities = 0.12, 0.34
        sim.flame.radiation_enabled = True
        sim.flame.set_steady_tolerances(default=(3e-4, 7e-9))
        sim.transport_model = 'unity-Lewis-number'

        new_settings = sim.flame.settings
        assert new_settings["radiation-enabled"] is True
        assert new_settings["emissivity-left"] == 0.12
        assert new_settings["emissivity-right"] == 0.34
        assert new_settings["transport-model"] == "unity-Lewis-number"

        tolerances = new_settings["tolerances"]
        assert tolerances["steady-reltol"] == approx(3e-4)
        assert tolerances["steady-abstol"] == approx(7e-9)

        assert "fixed-point" in new_settings
        assert "location" in new_settings["fixed-point"]

    def test_solution_array_output(self):
        self.create_sim(ct.OneAtm, 300, 'H2:1.0, O2:0.5, AR:2', width=0.1)
        self.sim.set_initial_guess()

        flow = self.sim.to_array(normalize=True)
        assert self.sim.grid == approx(flow.grid)
        assert self.sim.T == approx(flow.T)

        f2 = ct.FreeFlame(self.gas)
        f2.from_array(flow)
        assert self.sim.grid == approx(f2.grid)
        assert self.sim.T == approx(f2.T)

        inlet = self.sim.to_array(self.sim.inlet)
        f2.from_array(inlet, f2.inlet)
        assert self.sim.inlet.T == approx(f2.inlet.T)
        assert self.sim.inlet.mdot == approx(f2.inlet.mdot)
        assert self.sim.inlet.Y == approx(f2.inlet.Y)

    def test_solve_fixed_temperature_continuity(self):
        self.create_sim(ct.OneAtm, 300.0, 'H2:1.0, O2:0.5, AR:2', width=0.05)
        self.sim.set_initial_guess()
        self.sim.energy_enabled = False
        self.sim.solve(loglevel=0, refine_grid=False)

        assert not self.sim.energy_enabled
        rhou = self.sim.inlet.mdot
        for rhou_j in self.sim.density * self.sim.velocity:
            assert rhou_j == approx(rhou, rel=1e-3)
