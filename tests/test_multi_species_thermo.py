"""Focused source-semantic tests for the MultiSpeciesThermo translation."""

import unittest

from cantera_python.multi_species_thermo import (
    CONSTANT_CP,
    NASA1,
    MultiSpeciesThermo,
)
from cantera_python.thermo import GAS_CONSTANT, ConstCpPoly, NasaPoly1


class MultiSpeciesThermoTests(unittest.TestCase):
    @staticmethod
    def constant(cp_r, *, low=200, high=3000, pressure=101325):
        return ConstCpPoly(
            cp0=cp_r * GAS_CONSTANT,
            min_temp=low,
            max_temp=high,
            reference_pressure=pressure,
        )

    def test_sparse_install_readiness_and_source_type_ids(self):
        manager = MultiSpeciesThermo()
        manager.install_STIT(2, self.constant(3))
        self.assertFalse(manager.ready(3))
        self.assertEqual(manager.reportType(2), CONSTANT_CP)
        self.assertEqual(manager.reportType(1), -1)
        manager.install_STIT(0, NasaPoly1(
            [4, 0, 0, 0, 0, 0, 0], min_temp=300, max_temp=4000))
        self.assertEqual(manager.reportType(0), NASA1)
        self.assertFalse(manager.ready(3))
        manager.install_STIT(1, self.constant(5))
        self.assertTrue(manager.ready(3))

    def test_update_mutates_only_installed_positions(self):
        manager = MultiSpeciesThermo()
        manager.install_STIT(0, self.constant(3))
        manager.install_STIT(2, NasaPoly1([4, 0, 0, 0, 0, 0, 0]))
        cp_r = [-9.0, -9.0, -9.0]
        h_rt = [-9.0, -9.0, -9.0]
        s_r = [-9.0, -9.0, -9.0]
        manager.update(500, cp_r, h_rt, s_r)
        self.assertAlmostEqual(cp_r[0], 3.0)
        self.assertEqual(cp_r[1], -9.0)
        self.assertEqual(cp_r[2], 4.0)
        self.assertEqual(manager.update_single(2, 500),
                         NasaPoly1([4, 0, 0, 0, 0, 0, 0]).properties(500))
        self.assertIsNone(manager.update_single(7, 500))

    def test_aggregate_and_species_temperature_bounds(self):
        manager = MultiSpeciesThermo()
        self.assertEqual(manager.minTemp(), 0.0)
        self.assertEqual(manager.maxTemp(), 1.0e30)
        manager.install_STIT(0, self.constant(3, low=200, high=5000))
        manager.install_STIT(1, self.constant(4, low=400, high=3000))
        self.assertEqual(manager.minTemp(), 400)
        self.assertEqual(manager.maxTemp(), 3000)
        self.assertEqual(manager.minTemp(0), 200)
        self.assertEqual(manager.maxTemp(0), 5000)
        # Missing species indices fall back to aggregate values in the source.
        self.assertEqual(manager.minTemp(99), 400)
        self.assertEqual(manager.maxTemp(99), 3000)

    def test_install_invariants(self):
        manager = MultiSpeciesThermo()
        manager.install_STIT(0, self.constant(3))
        self.assertEqual(manager.refPressure(), 101325)
        with self.assertRaises(ValueError):
            manager.install_STIT(0, self.constant(4))
        # The source comparison is strictly greater than 1e-6 Pa.
        manager.install_STIT(1, self.constant(4, pressure=101325 + 1e-6))
        with self.assertRaises(ValueError):
            manager.install_STIT(2, self.constant(4, pressure=101325.001))
        with self.assertRaises(ValueError):
            manager.install_STIT(3, None)

    def test_modify_preserves_type_and_aggregate_validity_envelope(self):
        manager = MultiSpeciesThermo()
        original = self.constant(3, low=300, high=3000)
        manager.install_STIT(0, original)
        replacement = self.constant(7, low=200, high=4000, pressure=2e5)
        manager.modifySpecies(0, replacement)
        self.assertIs(manager.provideSTIT(0), replacement)
        # Upstream modifySpecies does not impose the installation pressure check.
        self.assertEqual(manager.refPressure(), 101325)
        self.assertEqual(manager.update_single(0, 500)[0], 7.0)
        with self.assertRaises(ValueError):
            manager.modifySpecies(0, NasaPoly1([3] + [0] * 6))
        with self.assertRaises(ValueError):
            manager.modifySpecies(0, self.constant(3, low=301, high=4000))
        with self.assertRaises(ValueError):
            manager.modifySpecies(0, self.constant(3, low=200, high=2999))
        with self.assertRaises(KeyError):
            manager.modifySpecies(9, self.constant(3))


if __name__ == "__main__":
    unittest.main()
