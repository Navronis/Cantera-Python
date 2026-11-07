"""Regression for the pinned set_xy initial-state branch conditions."""

import unittest

from cantera_python.tpx import HFC134a, PropertyPair, TPXError, Undef


class TestSetXYUndefinedTemperature(unittest.TestCase):
    def test_does_not_repair_undefined_temperature_when_density_is_set(self):
        target = HFC134a()
        target.Set(PropertyPair.TV, 300.0, 0.02)
        target_enthalpy = target.h()

        phase = HFC134a()
        phase.Rho = 1.0 / target.v()
        density_before = phase.Rho

        # Pinned Substance::set_xy initializes only when both T and Rho are
        # undefined, or when Rho alone is undefined. With finite Rho and
        # undefined T it takes the ordinary starting-state path and propagates
        # the out-of-range temperature error rather than inventing Tcrit * 1.1.
        with self.assertRaisesRegex(TPXError, "Temperature out of range"):
            phase.Set(PropertyPair.TH, 300.0, target_enthalpy)

        self.assertEqual(phase.T, Undef)
        self.assertEqual(phase.Rho, density_before)
