"""Unit tests for Interface, InterfaceKinetics, FixedStoichiometryPhase, and coverage solvers."""
import math
import unittest
from pathlib import Path

from cantera_python import (
    Interface,
    FixedStoichiometryPhase,
    IdealGasPhase,
    load_mechanism,
)
from cantera_python.constants import OneAtm, GasConstant


class InterfaceKineticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = Path(__file__).resolve().parents[2] / 'cantera-original' / 'data'
        cls.diamond_yaml = cls.data_dir / 'diamond.yaml'
        cls.ptcombust_yaml = cls.data_dir / 'ptcombust.yaml'

    def test_fixed_stoichiometry_phase_diamond(self):
        d = FixedStoichiometryPhase.from_mechanism(self.diamond_yaml, 'diamond')
        self.assertEqual(d.name, 'diamond')
        self.assertEqual(d.species_names, ('C(d)',))
        self.assertEqual(d.n_species, 1)
        self.assertAlmostEqual(d.density, 3520.0, places=4)
        self.assertAlmostEqual(d.standard_concentration(0), 1.0)
        self.assertEqual(d.activity_concentrations, (1.0,))

        # Test pressure correction on enthalpy: (P - P0) / rho_molar
        d.TP = 1200.0, 2666.45
        p0 = OneAtm
        expected_h = (2666.45 - p0) / d.density_mole
        self.assertAlmostEqual(d.enthalpy_mole, expected_h, places=4)
        self.assertAlmostEqual(d.gibbs_mole, expected_h, places=4)

    def test_diamond_interface_resolution_and_rates(self):
        surf = Interface.from_mechanism(self.diamond_yaml, 'diamond_100')
        self.assertEqual(surf.name, 'diamond_100')
        self.assertEqual(surf.n_phases, 3)
        self.assertEqual([p.name for p in surf.phases], ['diamond_100', 'gas', 'diamond'])
        self.assertEqual(surf.n_species, 8)
        self.assertEqual(surf.n_total_species, 13)
        self.assertEqual(surf.n_reactions, 20)

        # Evaluate at T=1200, P=2666.45 Pa
        surf.TP = 1200.0, 2666.45
        for p in surf.phases[1:]:
            p.TP = 1200.0, 2666.45

        kf = surf.forward_rate_constants
        kc = surf.equilibrium_constants
        kr = surf.reverse_rate_constants
        self.assertEqual(len(kf), 20)
        self.assertEqual(len(kc), 20)
        self.assertEqual(len(kr), 20)

        # Pinned reference values for first 3 reactions
        self.assertAlmostEqual(kf[0], 6087772725.679118, delta=1.0)
        self.assertAlmostEqual(kc[0], 873.4330048917985, delta=1e-6)
        self.assertAlmostEqual(kr[0], 6969936.66552969, delta=1e-3)

        # Net production rates
        sdot = surf.net_production_rates
        self.assertEqual(len(sdot), 13)
        # Verify creation and destruction rates
        creat = surf.creation_rates
        destr = surf.destruction_rates
        for k in range(13):
            self.assertAlmostEqual(sdot[k], creat[k] - destr[k], places=10)

    def test_ptcombust_sticking_rates_and_coverage_dependencies(self):
        surf = Interface.from_mechanism(self.ptcombust_yaml, 'Pt_surf')
        self.assertEqual(surf.name, 'Pt_surf')
        self.assertEqual(surf.n_phases, 2)
        self.assertEqual(surf.n_species, 11)
        self.assertEqual(surf.n_reactions, 24)

        surf.TP = 900.0, 101325.0
        for p in surf.phases[1:]:
            p.TP = 900.0, 101325.0

        kf = surf.forward_rate_constants
        # Reaction 3: H + PT(S) => H(S) (sticking, order 1 on PT(S))
        self.assertAlmostEqual(kf[2], 40164485627.496864, delta=1e-3)
        # Reaction 5: O2 + 2 PT(S) => 2 O(S) (sticking, order 2 on PT(S))
        self.assertAlmostEqual(kf[4], 6058469933898197.0, delta=1e6)

        # Net rates of progress
        ropnet = surf.net_rates_of_progress
        ropf = surf.forward_rates_of_progress
        ropr = surf.reverse_rates_of_progress
        for f, r, net in zip(ropf, ropr, ropnet):
            self.assertAlmostEqual(net, f - r, places=10)

    def test_advance_coverages_ptcombust(self):
        surf = Interface.from_mechanism(self.ptcombust_yaml, 'Pt_surf')
        surf.TP = 900.0, 101325.0
        surf.coverages = {'PT(S)': 0.5, 'H(S)': 0.5}

        # Advance over 1e-4 s
        surf.advance_coverages(1e-4, rtol=1e-7, atol=1e-14)
        cov = surf.coverages
        self.assertAlmostEqual(sum(cov), 1.0, places=7)
        # Oracle verified reference: PT(S) ~ 0.08562, O(S) ~ 0.91403
        self.assertAlmostEqual(cov[0], 0.0856216964, places=6)
        self.assertAlmostEqual(cov[10], 0.9140297814, places=6)

    def test_advance_coverages_to_steady_state(self):
        surf = Interface.from_mechanism(self.ptcombust_yaml, 'Pt_surf')
        surf.TP = 900.0, 101325.0
        for p in surf.phases[1:]:
            p.TP = 900.0, 101325.0
        surf.coverages = {'PT(S)': 0.5, 'H(S)': 0.5}

        surf.advance_coverages_to_steady_state()
        cov = surf.coverages
        self.assertAlmostEqual(sum(cov), 1.0, places=7)
        # Residual test
        sdot = surf.net_production_rates[:surf.n_species]
        max_res = max(abs(s) for s in sdot) / surf.site_density
        self.assertLess(max_res, 1e-8)


if __name__ == '__main__':
    unittest.main()
