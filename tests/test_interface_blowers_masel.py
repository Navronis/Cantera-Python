"""Pinned interface/sticking Blowers-Masel state-update regressions."""
import math
from types import SimpleNamespace
import unittest

from cantera_python import (
    BlowersMaselRate,
    Interface,
    InterfaceReaction,
    SurfaceSpecies,
)
from cantera_python.constants import GasConstant as R


class _ZeroSurfaceThermo:
    def properties(self, temperature):
        return 0.0, 0.0, 0.0


class _MutableBulk:
    """Small adjacent phase exposing the state read by InterfaceData upstream."""

    name = 'bulk'
    species_names = ('A', 'B')
    species = (
        SimpleNamespace(molecular_weight=2.0),
        SimpleNamespace(molecular_weight=4.0),
    )
    n_species = 2

    def __init__(self, delta_h=0.0):
        self.delta_h = float(delta_h)

    @property
    def partial_molar_enthalpies(self):
        return 0.0, self.delta_h


def _reaction(rate_type, params):
    # Global species order is S, A, B. S cancels, leaving Delta H = h_B-h_A.
    return InterfaceReaction(
        equation_str='A + S => B + S',
        rate_type=rate_type,
        reversible=False,
        reactants=((1, 1.0, 1.0), (0, 1.0, 1.0)),
        products=((2, 1.0), (0, 1.0)),
        net_stoich={1: -1.0, 2: 1.0, 0: 0.0},
        rate_params=params,
    )


class InterfaceBlowersMaselStateTests(unittest.TestCase):
    def _surface(self, reaction, delta_h=0.0):
        bulk = _MutableBulk(delta_h)
        surface = Interface(
            (SurfaceSpecies('S', 1.0, _ZeroSurfaceThermo()),),
            2.5e-8,
            adjacent={'bulk': bulk},
            reactions=(reaction,),
            temperature=500.0,
        )
        return surface, bulk

    def test_live_enthalpy_covers_all_three_barrier_branches(self):
        params = {
            'A': 3.0, 'b': 0.0, 'Ea0': 10.0 * R, 'w': 100.0 * R,
            'cov_deps': {},
        }
        surface, bulk = self._surface(
            _reaction('interface-Blowers-Masel', params))

        # Exothermic branch: Delta H <= -4 E0, so Ea = 0.
        bulk.delta_h = -50.0 * R
        self.assertAlmostEqual(surface.forward_rate_constants[0], 3.0)

        # Middle branch at Delta H = 0 reduces to the intrinsic barrier E0.
        bulk.delta_h = 0.0
        self.assertAlmostEqual(
            surface.forward_rate_constants[0], 3.0 * math.exp(-10.0 / 500.0))

        # Endothermic branch: Delta H > 4 E0, so Ea = Delta H.
        bulk.delta_h = 50.0 * R
        self.assertAlmostEqual(
            surface.forward_rate_constants[0], 3.0 * math.exp(-50.0 / 500.0))

        # Every access reads the current phase state; temperature is live too.
        surface.T = 1000.0
        self.assertAlmostEqual(
            surface.forward_rate_constants[0], 3.0 * math.exp(-50.0 / 1000.0))

    def test_sticking_preserves_coverage_then_motz_wise_order(self):
        params = {
            'A': 0.2, 'b': 0.0, 'Ea0': 10.0 * R, 'w': 100.0 * R,
            'cov_deps': {'S': {'a': 0.25, 'm': 0.0, 'E': 0.0}},
            'motz_wise': True,
            'multiplier': 2.0,
            'surface_order': 1.0,
        }
        surface, bulk = self._surface(
            _reaction('sticking-Blowers-Masel', params), -50.0 * R)

        coverage_factor = 10.0 ** 0.25
        gamma = 0.2 * coverage_factor
        corrected = gamma / (1.0 - 0.5 * gamma)
        expected = (corrected * 2.0 * surface.site_density ** -1.0
                    * math.sqrt(surface.T))
        self.assertTrue(math.isclose(surface.forward_rate_constants[0], expected,
                                     rel_tol=2e-15))

        # A state update must change the BM barrier without dropping either
        # the coverage term or the Motz-Wise correction.
        bulk.delta_h = 50.0 * R
        gamma = 0.2 * math.exp(-50.0 / surface.T) * coverage_factor
        corrected = gamma / (1.0 - 0.5 * gamma)
        expected = (corrected * 2.0 * surface.site_density ** -1.0
                    * math.sqrt(surface.T))
        self.assertTrue(math.isclose(surface.forward_rate_constants[0], expected,
                                     rel_tol=2e-15))

    def test_mechanism_factory_constructs_both_interface_rate_families(self):
        nasa = {
            'model': 'NASA7',
            'temperature-ranges': [200.0, 1000.0, 3000.0],
            'data': [[0.0] * 7, [0.0] * 7],
        }
        document = {
            'units': {'length': 'm', 'quantity': 'kmol',
                      'activation-energy': 'J/kmol'},
            'phases': [
                {'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'none',
                 'species': ['H', 'H2']},
                {'name': 'surface', 'thermo': 'ideal-surface',
                 'kinetics': 'surface', 'adjacent-phases': ['gas'],
                 'species': ['S', 'H(S)'], 'site-density': 2.5e-8,
                 'Motz-Wise': True, 'reactions': ['surface-reactions']},
            ],
            'species': [
                {'name': 'H', 'composition': {'H': 1}, 'thermo': nasa},
                {'name': 'H2', 'composition': {'H': 2}, 'thermo': nasa},
                {'name': 'S', 'composition': {'Pt': 1}, 'thermo': nasa},
                {'name': 'H(S)', 'composition': {'H': 1, 'Pt': 1},
                 'thermo': nasa},
            ],
            'surface-reactions': [
                {'equation': '2 H(S) => H2 + 2 S', 'type': 'Blowers-Masel',
                 'rate-constant': {'A': 3.7e20, 'b': 0.0,
                                   'Ea0': 67.4e6, 'w': 1.0e9}},
                {'equation': 'H + S => H(S)', 'type': 'Blowers-Masel',
                 'sticking-coefficient': {'A': 1.0, 'b': 0.0,
                                          'Ea0': 0.0, 'w': 1.0e9}},
                {'equation': 'H2 + 2 S => 2 H(S)', 'type': 'Blowers-Masel',
                 'Motz-Wise': False,
                 'sticking-coefficient': {'A': 0.25, 'b': 0.0,
                                          'Ea0': 0.0, 'w': 1.0e9}},
            ],
        }
        surface = Interface.from_mechanism(document, 'surface')
        self.assertEqual(surface.n_reactions, 3)
        self.assertEqual(surface.reactions[0].rate_type,
                         'interface-Blowers-Masel')
        self.assertEqual(surface.reactions[1].rate_type,
                         'sticking-Blowers-Masel')
        self.assertAlmostEqual(surface.reactions[0].rate_params['Ea0'], 67.4e6)
        self.assertAlmostEqual(surface.reactions[0].rate_params['w'], 1.0e9)

        # The phase-level Motz-Wise setting applies only when
        # a reaction does not provide an explicit local override.
        self.assertTrue(surface.reactions[1].rate_params['motz_wise'])
        self.assertFalse(surface.reactions[2].rate_params['motz_wise'])


if __name__ == '__main__':
    unittest.main()
