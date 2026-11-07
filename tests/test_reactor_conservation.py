"""Reactor conservation tests: independent physics invariants.

These checks are independent of the pinned native comparison and act as
error detectors for the reactor governing equations:

* inert closed constant-volume reactor: mass, composition constant
* reacting constant-volume reactor: mass + element conservation
* constant-pressure reactor: pressure consistency, enthalpy conservation
  (adiabatic), volume evolution
"""
import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

import cantera_python as ct
from cantera_python.constants import OneAtm
from cantera_python.reactor import (
    IdealGasConstPressureReactor, IdealGasReactor, ReactorNet,
)

GRI30 = Path(__file__).parents[2] / 'cantera-original' / 'data' / 'gri30.yaml'
PINNED_LINEAR_REACTOR = json.loads(
    (Path(__file__).parents[1] / 'provenance' /
     'pinned_native_reactor_first_order.json').read_text(encoding='utf-8'))
PINNED_GRI_REACTOR = json.loads(
    (Path(__file__).parents[1] / 'provenance' /
     'pinned_native_gri_reactor_t2e-5.json').read_text(encoding='utf-8'))


def element_counts(gas):
    """Atom totals [kmol] in the reactor contents."""
    from cantera_python.mechanism import load_mechanism
    mech = load_mechanism(GRI30)
    comp = {s.name: s.composition for s in mech.species}
    Y = gas.Y
    mw = gas.molecular_weights
    moles = np.asarray(Y) / np.asarray(mw)  # per unit mass basis
    totals = {}
    for i, name in enumerate(gas.species_names):
        for el, n in comp[name].items():
            totals[el] = totals.get(el, 0.0) + n * moles[i]
    return totals


class InertReactorTests(unittest.TestCase):
    def test_named_solver_backends_never_silently_fallback(self):
        gas = ct.Solution(GRI30)
        gas.TPX = 1000.0, OneAtm, 'N2:1'
        reactor = IdealGasReactor(gas)
        network = ReactorNet(reactor, solver_backend='native_python')
        self.assertEqual(network.solver_backend, 'native_python')
        self.assertAlmostEqual(network.advance(1e-6), 1e-6)
        self.assertEqual(network._last_solver_stats['backend'],
                         'in-package-python-cv-bdf')
        self.assertAlmostEqual(gas.temperature, 1000.0, places=10)
        with self.assertRaisesRegex(ValueError, 'unknown reactor solver backend'):
            ReactorNet(reactor, solver_backend='cvode-ish')

    def test_parity_backend_uses_standalone_sundials(self):
        try:
            import sksundae
        except ImportError:
            self.skipTest('optional scikit-sundae parity dependency is absent')
        self.assertEqual(sksundae.SUNDIALS_VERSION, '7.5.0')
        gas = ct.Solution(GRI30)
        gas.TPX = 1000.0, OneAtm, 'N2:1'
        network = ReactorNet(IdealGasReactor(gas), solver_backend='parity')
        network.advance(1e-4)
        self.assertEqual(network._last_solver_stats['backend'], 'sksundae-cvode')
        self.assertEqual(network._last_solver_stats['sundials_version'], '7.5.0')
        self.assertAlmostEqual(gas.temperature, 1000.0, places=10)

    def test_parity_sensitivity_does_not_silently_substitute(self):
        gas = ReactingReactorTests._two_species_solution()
        reactor = IdealGasConstPressureReactor(gas)
        reactor.add_sensitivity_reaction(0)
        network = ReactorNet(reactor, solver_backend='parity')
        with self.assertRaisesRegex(NotImplementedError, 'CVODES simultaneous'):
            network.advance(1e-3)

    def test_parity_backend_integrates_reacting_rhs(self):
        try:
            import sksundae  # noqa: F401
        except ImportError:
            self.skipTest('optional scikit-sundae parity dependency is absent')
        gas = ReactingReactorTests._two_species_solution()
        # Equal formation enthalpies make this an isothermal first-order case.
        gas._set_species_enthalpy_offset('B', 1.0e7)
        reactor = IdealGasConstPressureReactor(gas)
        network = ReactorNet(reactor, solver_backend='parity')
        network.advance(1e-2)
        expected_b = 1.0 - math.exp(-0.1)
        self.assertAlmostEqual(gas.Y[gas.species_index('B')], expected_b,
                               delta=5e-9)

    def test_inert_closed_cv_state_constant(self):
        gas = ct.Solution(GRI30)
        gas.TPX = 1000.0, OneAtm, 'N2:1'
        r = IdealGasReactor(gas, volume=2.5)
        mass0 = gas.density * r.volume
        sim = ReactorNet(r)
        sim.advance(1e-4)
        self.assertAlmostEqual(gas.temperature, 1000.0, places=6)
        self.assertAlmostEqual(gas.X[gas.species_index('N2')], 1.0, places=10)
        self.assertAlmostEqual(gas.density * r.volume, mass0, places=12)


class ReactingReactorTests(unittest.TestCase):
    @staticmethod
    def _two_species_solution():
        mechanism = {
            'phases': [{
                'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'gas',
                'transport': 'none', 'species': ['A', 'B'],
                'reactions': 'all',
            }],
            'species': [
                {'name': 'A', 'composition': {'H': 1},
                 'thermo': {'model': 'constant-cp', 'h0': 0.0,
                            's0': 0.0, 'cp0': 30000.0}},
                {'name': 'B', 'composition': {'H': 1},
                 'thermo': {'model': 'constant-cp', 'h0': -1.0e7,
                            's0': 0.0, 'cp0': 30000.0}},
            ],
            'reactions': [{
                'equation': 'A => B',
                'rate-constant': {'A': 10.0, 'b': 0.0, 'Ea': 0.0},
            }],
        }
        gas = ct.Solution(mechanism)
        gas.TPX = 1000.0, OneAtm, 'A:1'
        return gas

    def test_ideal_gas_reactor_trial_mass_fractions_are_not_normalized(self):
        gas = self._two_species_solution()
        reactor = IdealGasReactor(gas)
        state = reactor.get_state()
        state[3:] = [0.4, 0.4]
        reactor.update_state(state)
        np.testing.assert_array_equal(gas.Y, [0.4, 0.4])

    def test_const_pressure_reactor_trial_mass_fractions_are_not_normalized(self):
        gas = self._two_species_solution()
        reactor = IdealGasConstPressureReactor(gas)
        state = reactor.get_state()
        state[2:] = [0.4, 0.4]
        reactor.update_state(state)
        np.testing.assert_array_equal(gas.Y, [0.4, 0.4])

    def test_cv_mass_and_elements_conserved(self):
        gas = ct.Solution(GRI30)
        gas.TPX = 1500.0, OneAtm, 'H2:2, O2:1, N2:4'
        el0 = element_counts(gas)
        rho0 = gas.density
        r = IdealGasReactor(gas)
        sim = ReactorNet(r)
        sim.setTolerances(1e-8, 1e-12)
        sim.advance(2e-5)
        # mass conserved: rho*V constant with V fixed => rho constant
        self.assertAlmostEqual(gas.density, rho0, places=6)
        # elements conserved
        el1 = element_counts(gas)
        for el, n0 in el0.items():
            n1 = el1.get(el, 0.0)
            self.assertAlmostEqual(n1, n0, delta=abs(n0) * 1e-6 + 1e-12,
                                   msg=f'element {el} not conserved')
        self.assertTrue(all(y >= -1e-20 for y in gas.Y))

    def test_native_python_backend_integrates_reacting_rhs(self):
        gas = self._two_species_solution()
        # Equal formation enthalpies make this an isothermal first-order case.
        gas._set_species_enthalpy_offset('B', 1.0e7)
        reactor = IdealGasConstPressureReactor(gas)
        network = ReactorNet(reactor, solver_backend='native_python')
        network.advance(1e-2)
        expected_b = 1.0 - math.exp(-0.1)
        self.assertAlmostEqual(gas.Y[gas.species_index('B')], expected_b,
                               delta=5e-9)
        native = PINNED_LINEAR_REACTOR['native_result']
        tolerance = PINNED_LINEAR_REACTOR['python_comparison_tolerances']
        self.assertAlmostEqual(gas.T, native['temperature_K'],
                               delta=tolerance['temperature_K_atol'])
        self.assertAlmostEqual(gas.P, native['pressure_Pa'],
                               delta=tolerance['pressure_Pa_atol'])
        self.assertAlmostEqual(gas.Y[gas.species_index('A')], native['Y_A'],
                               delta=tolerance['mass_fraction_atol'])
        self.assertAlmostEqual(gas.Y[gas.species_index('B')], native['Y_B'],
                               delta=tolerance['mass_fraction_atol'])
        self.assertEqual(network._last_solver_stats['backend'],
                         'in-package-python-cv-bdf')

    def test_native_python_bdf_matches_pinned_gri_reactor_checkpoint(self):
        reference = PINNED_GRI_REACTOR
        gas = ct.Solution(GRI30)
        gas.TPX = (reference['initial_T_K'], reference['initial_P_Pa'],
                   reference['initial_mole_fractions'])
        network = ReactorNet(IdealGasReactor(gas), solver_backend='native_python')
        network.setTolerances(reference['rtol'], reference['atol'])
        network.advance(reference['t_s'])

        self.assertAlmostEqual(gas.T, reference['T_K'], delta=0.01)
        self.assertAlmostEqual(gas.P, reference['P_Pa'], delta=1.0)
        self.assertEqual(list(gas.species_names), reference['species_names'])
        np.testing.assert_allclose(
            gas.Y, reference['mass_fractions'], rtol=0.0, atol=2e-7)
        self.assertEqual(network._last_solver_stats['backend'],
                         'in-package-python-cv-bdf')

    def test_native_python_backend_keeps_solver_history_between_advances(self):
        def make_network():
            gas = self._two_species_solution()
            gas._set_species_enthalpy_offset('B', 1.0e7)
            return gas, ReactorNet(
                IdealGasConstPressureReactor(gas), solver_backend='native_python')

        split_gas, split = make_network()
        single_gas, single = make_network()
        split.advance(4e-3)
        split.advance(1e-2)
        single.advance(1e-2)
        np.testing.assert_allclose(split_gas.Y, single_gas.Y,
                                   rtol=0.0, atol=2e-9)
        self.assertIsNotNone(split._native_python_integrator)

    def test_cp_pressure_and_enthalpy_conserved(self):
        gas = ct.Solution(GRI30)
        gas.TPX = 1500.0, OneAtm, 'H2:2, O2:1, N2:4'
        el0 = element_counts(gas)
        h0 = gas.enthalpy_mass
        r = IdealGasConstPressureReactor(gas)
        sim = ReactorNet(r)
        sim.setTolerances(1e-8, 1e-12)
        sim.advance(2e-5)
        # pressure held by construction (setState_TP at m_pressure)
        self.assertAlmostEqual(gas.pressure, OneAtm, delta=OneAtm * 1e-6)
        # adiabatic closed CP reactor conserves total enthalpy
        self.assertAlmostEqual(gas.enthalpy_mass, h0, delta=abs(h0) * 1e-6)
        el1 = element_counts(gas)
        for el, n0 in el0.items():
            self.assertAlmostEqual(el1.get(el, 0.0), n0, delta=abs(n0) * 1e-6 + 1e-12)

    def test_reaction_sensitivity_matrix_and_component_lookup(self):
        gas = self._two_species_solution()
        reaction = 0
        reactor = IdealGasConstPressureReactor(gas, name='combustor')
        network = ReactorNet(reactor)
        reactor.add_sensitivity_reaction(reaction)
        reactor.add_sensitivity_species_enthalpy('A')

        network.advance(1e-2)
        matrix = network.sensitivities()
        self.assertEqual(matrix.shape, (network.n_vars, 2))
        self.assertEqual(network.n_sensitivity_params, 2)
        self.assertEqual(network.sensitivity_parameter_name(0),
                         f'combustor: reaction {reaction}')
        self.assertEqual(network.sensitivity_parameter_name(1),
                         'combustor: A enthalpy')
        self.assertTrue(math.isfinite(network.sensitivity('temperature', 0)))
        self.assertNotEqual(network.sensitivity('temperature', 0), 0.0)
        self.assertTrue(math.isfinite(network.sensitivity('temperature', 1)))
        self.assertNotEqual(network.sensitivity('temperature', 1), 0.0)
        # Both parameter columns are propagated in one augmented integration;
        # production code no longer runs one complete trajectory per parameter.
        self.assertGreater(network._sensitivity_rhs_evaluations, 0)
        with self.assertRaisesRegex(ValueError, 'already registered'):
            reactor.add_sensitivity_species_enthalpy('A')
        with self.assertRaisesRegex(IndexError, 'out of range'):
            reactor.add_sensitivity_species_enthalpy(99)

    def test_augmented_sensitivity_matches_independent_trajectory_difference(self):
        gas = self._two_species_solution()
        reactor = IdealGasConstPressureReactor(gas)
        network = ReactorNet(reactor)
        reactor.add_sensitivity_reaction(0)
        initial = copy.deepcopy(network)

        network.advance(1e-2)
        augmented = network._sensitivity_derivatives[:, 0]

        # A centered difference cancels the O(step) truncation error of the
        # former forward-difference reference. A step sweep showed 1e-3 gives
        # a stable derivative here while smaller steps amplify integration
        # error in the differenced trajectories.
        step = 1e-3
        upper = copy.deepcopy(initial)
        upper._reactors[0].kinetics.set_multiplier(1.0 + step, 0)
        upper.advance(1e-2, _compute_sensitivities=False)
        lower = copy.deepcopy(initial)
        lower._reactors[0].kinetics.set_multiplier(1.0 - step, 0)
        lower.advance(1e-2, _compute_sensitivities=False)
        finite_trajectory = (
            np.asarray(upper._get_global_state())
            - np.asarray(lower._get_global_state())) / (2.0 * step)

        np.testing.assert_allclose(augmented, finite_trajectory,
                                   rtol=2e-6, atol=1e-8)

    def test_energy_and_chemistry_switches_control_rhs_terms(self):
        for reactor_type in (IdealGasReactor, IdealGasConstPressureReactor):
            gas = self._two_species_solution()
            reactor = reactor_type(gas)
            state = reactor.get_state()
            self.assertTrue(reactor.chemistry_enabled)
            self.assertTrue(reactor.energy_enabled)

            active = np.asarray(reactor.rhs(0.0, state))
            species_offset = 3 if reactor_type is IdealGasReactor else 2
            temperature_index = 2 if reactor_type is IdealGasReactor else 1
            self.assertTrue(np.any(np.abs(active[species_offset:]) > 0.0))
            self.assertNotEqual(active[temperature_index], 0.0)

            reactor.setChemistryEnabled(False)
            inactive = np.asarray(reactor.rhs(0.0, state))
            self.assertFalse(reactor.chemistryEnabled())
            np.testing.assert_array_equal(inactive[species_offset:], 0.0)
            self.assertEqual(inactive[temperature_index], 0.0)

            reactor.set_chemistry_enabled()
            reactor.energy_enabled = False
            isothermal = np.asarray(reactor.rhs(0.0, state))
            self.assertTrue(reactor.chemistry_enabled)
            self.assertFalse(reactor.energyEnabled())
            self.assertEqual(isothermal[temperature_index], 0.0)
            self.assertTrue(np.any(np.abs(isothermal[species_offset:]) > 0.0))


if __name__ == '__main__':
    unittest.main()
