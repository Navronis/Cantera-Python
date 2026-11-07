import math
from pathlib import Path
import unittest

import cantera_python as ct
from cantera_python.constants import GasConstant


class IdealSurfaceTests(unittest.TestCase):
    def _mechanism(self):
        return ct.load_mechanism({
            'units': {'length': 'cm', 'quantity': 'mol'},
            'phases': [
                {'name': 'gas', 'thermo': 'ideal-gas', 'kinetics': 'none',
                 'species': ['G'], 'reactions': 'none'},
                {'name': 'surf', 'thermo': 'ideal-surface', 'kinetics': 'none',
                 'transport': 'none', 'adjacent-phases': ['gas'],
                 'species': ['S', 'SS'], 'reactions': 'none',
                 'site-density': 2.5e-9,
                 'state': {'T': 500.0, 'coverages': {'S': 0.25, 'SS': 0.75}}},
            ],
            'species': [
                {'name': 'G', 'composition': {'H': 1},
                 'thermo': {'model': 'constant-cp'}},
                {'name': 'S', 'composition': {}, 'size': 1,
                 'thermo': {'model': 'constant-cp', 'h0': 1.0e6,
                            's0': 1000.0, 'cp0': 2000.0}},
                {'name': 'SS', 'composition': {}, 'size': 2,
                 'thermo': {'model': 'constant-cp', 'h0': 2.0e6,
                            's0': 2000.0, 'cp0': 3000.0}},
            ],
        })

    def test_factory_coverages_and_site_density(self):
        surface = ct.Solution(self._mechanism(), phase_name='surf')
        self.assertEqual(surface.species_names, ('S', 'SS'))
        self.assertEqual(surface.T, 500.0)
        self.assertAlmostEqual(surface.thermo.site_density, 2.5e-8)
        for actual, expected in zip(surface.thermo.coverages, (0.25, 0.75)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(surface.X), 1.0)
        for actual, expected in zip(surface.thermo.standard_concentrations,
                                    (2.5e-8, 1.25e-8)):
            self.assertAlmostEqual(actual, expected)
        for activity, coverage in zip(surface.thermo.activities,
                                      surface.thermo.coverages):
            self.assertEqual(activity, coverage)

    def test_chemical_potential_uses_coverage(self):
        surface = ct.IdealSurfacePhase.from_mechanism(self._mechanism(), 'surf')
        for mu, mu0, theta in zip(surface.chemical_potentials,
                                  surface.standard_chemical_potentials,
                                  surface.coverages):
            self.assertAlmostEqual(mu, mu0 + GasConstant * surface.T * math.log(theta))

    def test_adjacent_phase_metadata_round_trip(self):
        mechanism = self._mechanism()
        phase = mechanism.phase('surf')
        self.assertEqual(phase.adjacent_phase_names, ('gas',))
        restored = ct.load_mechanism(mechanism.to_yaml()).phase('surf')
        self.assertEqual(restored.adjacent_phase_names, ('gas',))
        self.assertEqual(restored.site_density, 2.5e-9)

    def test_bundled_diamond_surface_reference_state(self):
        path = Path(__file__).parents[2] / 'cantera-original' / 'data' / 'diamond.yaml'
        surface = ct.IdealSurfacePhase.from_mechanism(path, 'diamond_100')
        self.assertEqual(surface.n_species, 8)
        self.assertAlmostEqual(surface.site_density, 3.0e-8)
        self.assertAlmostEqual(surface.enthalpy_mole, 64559120.0)
        self.assertAlmostEqual(surface.entropy_mole, 89939.2902300613)


if __name__ == '__main__':
    unittest.main()
