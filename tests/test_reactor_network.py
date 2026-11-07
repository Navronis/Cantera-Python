"""Verification tests for 0D reactor networks, flow devices, walls, and reservoirs."""
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import cantera_python as ct
from cantera_python.reactor import (
    IdealGasReactor, IdealGasConstPressureReactor, ReactorNet,
    Reservoir, Wall, MassFlowController, Valve, PressureController,
)

GRI30 = Path(__file__).parents[2] / 'cantera-original' / 'data' / 'gri30.yaml'


class PurePythonDenseSolveTests(unittest.TestCase):
    def test_reactor_network_default_component_tolerances_match_pinned(self):
        gas = ct.Solution(GRI30)
        reactor = IdealGasReactor(gas)
        network = ReactorNet([reactor])
        network.initialize()
        self.assertTrue(np.all(network._atol_array == 1e-15))

        network.setAbsoluteTolerance(2e-12)
        self.assertTrue(np.all(network._atol_array == 2e-12))
        network.atol = 3e-12
        self.assertTrue(np.all(network._atol_array == 3e-12))
        network.clearAbsoluteTolerance()
        self.assertTrue(np.all(network._atol_array == 1e-15))

        cp_reactor = IdealGasConstPressureReactor(ct.Solution(GRI30))
        cp_network = ReactorNet([cp_reactor])
        cp_network.initialize()
        self.assertEqual(cp_reactor.neq(), len(cp_reactor.get_state()))
        self.assertEqual(cp_network._atol_array.size, len(cp_reactor.get_state()))

    def test_reactor_preconditioner_uses_ported_lu(self):
        network = ReactorNet()
        network._precon_mat = [[0.0, 2.0], [1.0, 3.0]]
        rhs = [4.0, 7.0]
        out = [0.0, 0.0]
        with patch('numpy.linalg.solve', side_effect=AssertionError('compiled solve used')):
            network.preconditionerSolve(rhs, out)
        self.assertAlmostEqual(out[0], 1.0)
        self.assertAlmostEqual(out[1], 2.0)

    def test_steady_newton_step_uses_ported_lu(self):
        network = ReactorNet()
        network._get_state = lambda: np.array([1.0])
        network._eval_rhs = lambda t, y, out: out.__setitem__(0, y[0] - 2.0)
        network.finiteDifferenceJacobian = lambda: np.array([[1.0]])
        updated = []
        network.updateState = lambda y: updated.append(y.copy())
        with patch('numpy.linalg.solve', side_effect=AssertionError('compiled solve used')):
            self.assertEqual(network.solveSteady(), 0)
        np.testing.assert_allclose(updated[-1], [2.0])


class ReservoirAndFlowDeviceTests(unittest.TestCase):
    def test_reactor_state_updates_reject_nonpositive_trial_temperature(self):
        gas = ct.Solution(GRI30)
        gas.TPX = 300.0, ct.OneAtm, 'N2:1'
        reactors = (
            IdealGasReactor(gas),
            IdealGasConstPressureReactor(gas),
        )
        for reactor in reactors:
            with self.subTest(reactor=type(reactor).__name__):
                state = reactor.get_state()
                temperature_index = 2 if isinstance(reactor, IdealGasReactor) else 1
                state[temperature_index] = -1.0
                with self.assertRaises(ct.CanteraError):
                    reactor.update_state(state)

    def test_reservoir_state_constant(self):
        gas = ct.Solution(GRI30)
        gas.TPX = 1200.0, ct.OneAtm, 'CH4:1, O2:2, N2:7.52'
        res = Reservoir(gas, name='inlet_reservoir')
        self.assertEqual(len(res.get_state()), 0)
        self.assertAlmostEqual(res.temperature, 1200.0)
        self.assertAlmostEqual(res.pressure, ct.OneAtm)
        self.assertAlmostEqual(res.density, gas.density)

    def test_mfc_mass_transfer_to_reactor(self):
        gas_in = ct.Solution(GRI30)
        gas_in.TPX = 300.0, ct.OneAtm, 'N2:1'
        inlet = Reservoir(gas_in)

        gas_r = ct.Solution(GRI30)
        gas_r.TPX = 300.0, ct.OneAtm, 'N2:1'
        r = IdealGasReactor(gas_r, volume=1.0)
        m0 = r.mass

        # Mass flow rate = 0.05 kg/s
        mdot = 0.05
        mfc = MassFlowController(inlet, r, mdot=mdot)

        sim = ReactorNet(r)
        dt = 0.01  # 10 ms
        sim.advance(dt)

        # Expected mass increase: mdot * dt
        expected_mass = m0 + mdot * dt
        self.assertAlmostEqual(r.mass, expected_mass, delta=expected_mass * 1e-4)

    def test_valve_pressure_driven_flow(self):
        # High pressure reactor -> Low pressure reactor via Valve
        gas1 = ct.Solution(GRI30)
        gas1.TPX = 300.0, 2.0 * ct.OneAtm, 'N2:1'
        r1 = IdealGasReactor(gas1, volume=1.0)

        gas2 = ct.Solution(GRI30)
        gas2.TPX = 300.0, 1.0 * ct.OneAtm, 'N2:1'
        r2 = IdealGasReactor(gas2, volume=1.0)

        p1_0 = r1.pressure
        p2_0 = r2.pressure
        m_total_0 = r1.mass + r2.mass

        Kv = 1e-5  # kg / (s Pa)
        valve = Valve(r1, r2, Kv=Kv)

        sim = ReactorNet([r1, r2])
        sim.advance(0.005)

        # Total mass must be conserved between the two closed reactors
        self.assertAlmostEqual(r1.mass + r2.mass, m_total_0, delta=m_total_0 * 1e-5)
        # Flow should reduce pressure in r1 and increase pressure in r2
        self.assertLess(r1.pressure, p1_0)
        self.assertGreater(r2.pressure, p2_0)


class WallTests(unittest.TestCase):
    def test_wall_heat_transfer_between_reactors(self):
        # Hot reactor (1000 K) and cold reactor (300 K) separated by Wall
        gas1 = ct.Solution(GRI30)
        gas1.TPX = 1000.0, ct.OneAtm, 'N2:1'
        r1 = IdealGasReactor(gas1, volume=1.0)

        gas2 = ct.Solution(GRI30)
        gas2.TPX = 300.0, ct.OneAtm, 'N2:1'
        r2 = IdealGasReactor(gas2, volume=1.0)

        u1_0 = r1.mass * gas1.int_energy_mass
        u2_0 = r2.mass * gas2.int_energy_mass
        u_total_0 = u1_0 + u2_0

        U = 500.0  # W / (m2 K)
        wall = Wall(r1, r2, area=1.0, U=U)

        sim = ReactorNet([r1, r2])
        sim.advance(0.01)

        # Hot reactor cools down, cold reactor heats up
        self.assertLess(r1.temperature, 1000.0)
        self.assertGreater(r2.temperature, 300.0)
        # Total internal energy conserved across the closed wall
        u1_f = r1.mass * gas1.int_energy_mass
        u2_f = r2.mass * gas2.int_energy_mass
        self.assertAlmostEqual(u1_f + u2_f, u_total_0, delta=abs(u_total_0) * 1e-4)

    def test_moving_wall_expansion_work(self):
        # Reactor 1 has higher pressure, pushes wall into Reactor 2
        gas1 = ct.Solution(GRI30)
        gas1.TPX = 300.0, 2.0 * ct.OneAtm, 'N2:1'
        r1 = IdealGasReactor(gas1, volume=1.0)

        gas2 = ct.Solution(GRI30)
        gas2.TPX = 300.0, 1.0 * ct.OneAtm, 'N2:1'
        r2 = IdealGasReactor(gas2, volume=1.0)

        v_total_0 = r1.volume + r2.volume
        K_exp = 1e-7  # m / (s Pa)
        wall = Wall(r1, r2, area=1.0, K=K_exp)

        sim = ReactorNet([r1, r2])
        sim.advance(0.01)

        # Total volume must remain constant: V1 + V2 = 2.0
        self.assertAlmostEqual(r1.volume + r2.volume, v_total_0, places=6)
        # High pressure reactor expands, low pressure contracts
        self.assertGreater(r1.volume, 1.0)
        self.assertLess(r2.volume, 1.0)


if __name__ == '__main__':
    unittest.main()
