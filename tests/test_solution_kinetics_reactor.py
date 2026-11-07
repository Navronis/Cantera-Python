import math
from pathlib import Path
import unittest

import numpy as np
import cantera_python as ct
from cantera_python.constants import OneAtm
from cantera_python.ideal_gas import IdealGasPhase
from cantera_python.kinetics import GasKinetics, _VectorizedKinetics
from cantera_python.transport import MixTransport
from cantera_python.reactor import IdealGasReactor, IdealGasConstPressureReactor, ReactorNet
from cantera_python.solution import Solution


DATA_DIR = Path(__file__).parents[2] / 'cantera-original' / 'data'
GRI30_PATH = DATA_DIR / 'gri30.yaml'


class SolutionKineticsReactorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gas = ct.Solution(GRI30_PATH)

    def test_ideal_gas_phase_state_mutators(self):
        gas = IdealGasPhase.from_mechanism(GRI30_PATH)
        gas.TPX = 1000.0, 2.0 * OneAtm, 'CH4:0.1, O2:0.2, N2:0.7'
        self.assertAlmostEqual(gas.T, 1000.0)
        self.assertAlmostEqual(gas.P, 2.0 * OneAtm)
        
        # Test HP inverter
        h0 = gas.enthalpy_mass
        p0 = gas.pressure
        gas.setState_HP(h0, p0)
        self.assertAlmostEqual(gas.T, 1000.0, places=4)

        # Test UV inverter
        u0 = gas.int_energy_mass
        v0 = gas.specific_volume
        gas.setState_UV(u0, v0)
        self.assertAlmostEqual(gas.T, 1000.0, places=4)

        # Test SP inverter
        s0 = gas.entropy_mass
        gas.setState_SP(s0, p0)
        self.assertAlmostEqual(gas.T, 1000.0, places=4)

        # Test SV inverter
        gas.setState_SV(s0, v0)
        self.assertAlmostEqual(gas.T, 1000.0, places=4)

        # Test TD setter
        rho0 = gas.density
        gas.setState_TD(1000.0, rho0)
        self.assertAlmostEqual(gas.pressure, p0, places=4)

        # Test DP setter
        gas.setState_DP(rho0, p0)
        self.assertAlmostEqual(gas.temperature, 1000.0, places=4)
        self.assertEqual(gas.internal_pressure, 0.0)

        solution = ct.Solution(GRI30_PATH)
        self.assertEqual(solution.internal_pressure, 0.0)

    def test_vectorized_mass_action_matches_native_negative_trial_semantics(self):
        # CVODE evaluates trial states that can have tiny negative species
        # concentrations. Cantera's C1/C2/C3 fast stoichiometry paths preserve
        # the sign of direct multiplication, while C_AnyN returns zero if a
        # participating concentration is non-positive.
        stoich_matrix = np.array([
            [1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [0.5, 0.0],
            [1.0, 0.0],
        ])
        orders = stoich_matrix.copy()
        orders[3, 0] = 0.25
        orders[4, 0] = -1.0
        general = _VectorizedKinetics._general_stoich_products(
            stoich_matrix, orders)
        rates = _VectorizedKinetics._activity_products(
            object.__new__(_VectorizedKinetics),
            np.array([-1e-30, 2.0]), orders, general)
        self.assertEqual(general.tolist(), [False, False, True, True, True])
        self.assertAlmostEqual(rates[0], -1e-30)
        self.assertAlmostEqual(rates[1], 1e-60)
        self.assertEqual(rates[2], 0.0)
        self.assertEqual(rates[3], 0.0)
        zero_conc_rate = _VectorizedKinetics._activity_products(
            object.__new__(_VectorizedKinetics),
            np.array([0.0, 2.0]), orders[4:5], general[4:5])
        self.assertEqual(zero_conc_rate[0], 0.0)

    def test_solution_phase_factory_and_initial_state(self):
        gas = ct.Solution(GRI30_PATH, phase_name='gri30')
        self.assertEqual(gas.name, 'gri30')
        self.assertEqual(gas.thermo_model, 'ideal-gas')
        self.assertAlmostEqual(gas.T, 300.0)
        self.assertAlmostEqual(gas.P, OneAtm)

        h2o2 = DATA_DIR / 'h2o2.yaml'
        gas_rk = ct.Solution(h2o2, phase_name='ohmech-RK')
        self.assertEqual(gas_rk.name, 'ohmech-RK')
        self.assertEqual(gas_rk.thermo_model, 'Redlich-Kwong')
        self.assertAlmostEqual(gas_rk.T, 300.0)
        self.assertAlmostEqual(gas_rk.P, OneAtm)

        nasa9 = ct.Solution(DATA_DIR / 'airNASA9.yaml', phase_name='airNASA9')
        self.assertEqual(nasa9.n_species, 11)
        self.assertTrue(math.isfinite(nasa9.cp_mole))

        unit_state = ct.load_mechanism({
            'units': {'pressure': 'bar'},
            'phases': [{'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'none',
                        'transport': 'none', 'species': ['H2'],
                        'state': {'T': 400.0, 'P': 2.0}}],
            'species': [{'name': 'H2', 'composition': {'H': 2},
                         'thermo': {'model': 'constant-cp', 'T0': 298.15,
                                    'h0': 0.0, 's0': 0.0, 'cp0': 29000.0}}],
        })
        gas = ct.Solution(unit_state, phase_name='gas')
        self.assertEqual(gas.T, 400.0)
        self.assertEqual(gas.P, 2.0e5)

        mass_state = ct.load_mechanism({
            'phases': [{'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'none',
                        'transport': 'none', 'species': ['H2', 'O2'],
                        'state': {'T': 500.0, 'P': '3 atm',
                                  'Y': {'H2': 0.25, 'O2': 0.75}}}],
            'species': [
                {'name': 'H2', 'composition': {'H': 2},
                 'thermo': {'model': 'constant-cp'}},
                {'name': 'O2', 'composition': {'O': 2},
                 'thermo': {'model': 'constant-cp'}},
            ],
        })
        gas_y = ct.Solution(mass_state, phase_name='gas')
        self.assertAlmostEqual(gas_y.T, 500.0)
        self.assertAlmostEqual(gas_y.P, 3.0 * OneAtm)
        self.assertAlmostEqual(gas_y.Y[0], 0.25)
        self.assertAlmostEqual(gas_y.Y[1], 0.75)

        aliases = ct.load_mechanism({
            'phases': [{'name': 'gas', 'thermo': 'IdealGas', 'kinetics': 'Gas',
                        'transport': 'Mix', 'species': ['H2'], 'reactions': 'none'}],
            'species': [{'name': 'H2', 'composition': {'H': 2},
                         'thermo': {'model': 'constant-cp'},
                         'transport': {'geometry': 'linear', 'diameter': 2.92,
                                       'well-depth': 38.0}}],
        })
        alias_gas = ct.Solution(aliases, phase_name='gas')
        self.assertIsInstance(alias_gas.transport, MixTransport)
        self.assertEqual(alias_gas.n_species, 1)

    def test_bundled_linear_burke_phase(self):
        path = DATA_DIR / 'example_data' / 'ammonia-CO-H2-Alzueta-2023.yaml'
        gas = ct.Solution(path, phase_name='linear-Burke')
        gas.TPX = 1000.0, OneAtm, 'NH3:0.05, O2:0.1, N2:0.85'
        expected = (
            151912438.38841602,
            54834620.65122594,
            99.26298897447236,
            9.704495488783856e-10,
            1385547364.2182477,
            9.267852186955833,
        )
        for actual, reference in zip(gas.forward_rate_constants[-6:], expected):
            self.assertAlmostEqual(actual / reference, 1.0, places=13)

    def test_bundled_ionized_gas_transport(self):
        gas = ct.Solution(DATA_DIR / 'gri30_ion.yaml', phase_name='gas')
        gas.TPX = (1200.0, OneAtm,
                   'N2:0.7, O2:0.2, H2O:0.09999998, HCO+:1e-8, E:1e-8')
        self.assertEqual((gas.n_species, gas.n_reactions), (56, 331))
        charged = {species.name: species.charge for species in gas.species
                   if species.charge}
        self.assertEqual(charged, {'HCO+': 1.0, 'H3O+': 1.0, 'E': -1.0})

        # Supplementary Cantera 3.2.0 reference values. The ion-specific
        # equations are close; neutral transport retains the bounded port's
        # approximate collision-integral implementation.
        references = {
            'viscosity': 4.800450085901067e-05,
            'thermal_conductivity': 0.08597788208263807,
            'electrical_conductivity': 0.003929680764453163,
        }
        self.assertLess(abs(gas.viscosity / references['viscosity'] - 1.0), 0.02)
        self.assertLess(abs(gas.thermal_conductivity /
                            references['thermal_conductivity'] - 1.0), 0.02)
        self.assertLess(abs(gas.electrical_conductivity /
                            references['electrical_conductivity'] - 1.0), 1e-5)
        index = {name: gas.species_index(name) for name in ('HCO+', 'H3O+', 'E')}
        mobility_refs = {'HCO+': 0.001046558236970873,
                         'H3O+': 0.0011986053202214142, 'E': 0.4}
        diffusion_refs = {'HCO+': 0.00010822249214547713,
                          'H3O+': 0.00012394537792953799,
                          'E': 0.04136319965829686}
        for name in index:
            self.assertLess(abs(gas.mobilities[index[name]] / mobility_refs[name] - 1.0),
                            5e-4)
            self.assertLess(abs(gas.mix_diff_coeffs[index[name]] /
                                diffusion_refs[name] - 1.0), 5e-4)

    def test_array_getters(self):
        gas = IdealGasPhase.from_mechanism(GRI30_PATH)
        gas.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
        
        mu = gas.getChemPotentials()
        self.assertEqual(len(mu), gas.n_species)
        
        mu0 = gas.getStandardChemPotentials()
        self.assertEqual(len(mu0), gas.n_species)
        
        hbar = gas.getPartialMolarEnthalpies()
        self.assertEqual(len(hbar), gas.n_species)
        
        sbar = gas.getPartialMolarEntropies()
        self.assertEqual(len(sbar), gas.n_species)
        
        cpbar = gas.getPartialMolarCp()
        self.assertEqual(len(cpbar), gas.n_species)
        
        vbar = gas.getPartialMolarVolumes()
        self.assertEqual(len(vbar), gas.n_species)
        self.assertAlmostEqual(vbar[0], 1.0 / gas.molar_density)

    def test_kinetics_rates_and_conservation(self):
        gas = self.gas
        gas.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
        
        self.assertEqual(gas.n_reactions, 325)
        self.assertEqual(gas.n_species, 53)
        
        wdot = gas.net_production_rates
        self.assertEqual(len(wdot), 53)
        
        # Mass conservation: sum_k (wdot_k * mw_k) should be zero
        mw = gas.molecular_weights
        net_mass_rate = sum(w * m for w, m in zip(wdot, mw))
        self.assertAlmostEqual(net_mass_rate, 0.0, places=10)

        # Heat release rate: matches native Cantera -2068.555 W/m3 to 6 digits
        q_chem = gas.heat_release_rate
        self.assertAlmostEqual(q_chem, -2068.5552802, places=4)

        # Forward, reverse, net ROP consistency: net = fwd - rev
        fwd = gas.forward_rates_of_progress
        rev = gas.reverse_rates_of_progress
        net = gas.net_rates_of_progress
        for f, r, n in zip(fwd, rev, net):
            self.assertAlmostEqual(n, f - r, places=12)

    def test_reaction_multipliers_scale_scalar_and_vectorized_rates(self):
        gas = ct.Solution(GRI30_PATH)
        gas.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
        index = 10
        base = gas.net_rates_of_progress[index]

        gas.set_multiplier(2.5, index)
        self.assertEqual(gas.multiplier(index), 2.5)
        self.assertAlmostEqual(gas.net_rates_of_progress[index], 2.5 * base)
        scalar = gas.kinetics.net_production_rates()
        vectorized = gas.kinetics.net_production_rates_fast()
        for actual, expected in zip(vectorized, scalar):
            self.assertAlmostEqual(actual, expected,
                                   delta=max(1e-20, abs(expected) * 1e-12))

        gas.set_multiplier(0.0)
        self.assertTrue(all(rate == 0.0 for rate in gas.net_rates_of_progress))
        self.assertTrue(all(rate == 0.0 for rate in
                            gas.kinetics.net_production_rates_fast()))
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            gas.set_multiplier(-1.0, index)

    def test_vectorized_zero_default_third_body_efficiencies_match_scalar(self):
        gas = ct.Solution(GRI30_PATH)
        gas.TPX = 1500.0, OneAtm, 'H2:2, O2:1, N2:4'

        scalar = gas.kinetics.net_production_rates()
        vectorized = gas.kinetics.net_production_rates_fast()
        self.assertEqual(len(vectorized), len(scalar))
        for actual, expected in zip(vectorized, scalar):
            self.assertAlmostEqual(
                actual, expected,
                delta=max(2e-15, abs(expected) * 2e-13))

    def test_transport_properties(self):
        gas = self.gas
        gas.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
        
        # Pinned Cantera 4.0.0a2 mixture viscosity at this state.
        mu = gas.viscosity
        self.assertAlmostEqual(mu, 4.6872182846913704e-05, delta=1e-18)
        
        # Pinned Cantera 4.0.0a2 reference at this state. The former rounded
        # estimate came from a different release and predates the source fit.
        lambda_val = gas.thermal_conductivity
        self.assertAlmostEqual(lambda_val, 0.08965525008638553, delta=2e-14)
        
        d_mix = gas.mix_diff_coeffs
        self.assertEqual(len(d_mix), gas.n_species)
        self.assertTrue(all(d > 0.0 for d in d_mix))

    def test_constant_pressure_reactor_integration(self):
        sol = ct.Solution(GRI30_PATH)
        sol.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
        
        r = IdealGasConstPressureReactor(sol)
        sim = ReactorNet(r)
        
        t0 = sim.time
        T0 = sol.T
        
        # Advance 10 microseconds
        t_target = 1e-5
        sim.advance(t_target)
        
        self.assertGreaterEqual(sim.time, t_target)
        self.assertTrue(math.isfinite(sol.T))
        self.assertTrue(all(y >= 0.0 for y in sol.Y))

    def test_constant_volume_reactor_integration(self):
        sol = ct.Solution(GRI30_PATH)
        sol.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
        
        r = IdealGasReactor(sol)
        sim = ReactorNet(r)
        
        t_target = 1e-5
        sim.advance(t_target)
        
        self.assertGreaterEqual(sim.time, t_target)
        self.assertTrue(math.isfinite(sol.T))


if __name__ == '__main__':
    unittest.main()
