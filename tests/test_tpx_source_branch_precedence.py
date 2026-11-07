"""Boundary branch precedence copied from the pinned TPX implementation."""

import unittest

from cantera_python.tpx import Heptane, Water


class TestVaporFractionBranchPrecedence(unittest.TestCase):
    def test_vapor_boundary_precedes_liquid_when_saturation_densities_coincide(self):
        phase = Water()
        phase.T = phase.Tcrit() - 1e-8
        phase.Rhv = phase.Rhf = 100.0
        phase.Tslast = phase.T  # Keep update_sat from replacing the exact edge.
        phase.Rho = 100.0

        # Sub.cpp tests Rho <= Rhv before testing Rho >= Rhf, so equality
        # belongs to the vapor branch even when both boundaries coincide.
        self.assertEqual(phase.x(), 1.0)

    def test_tsat_out_of_range_start_uses_pinned_temperature_difference(self):
        phase = Heptane()
        tmin = phase.Tmin()
        phase.T = phase.Tcrit() + 10.0
        original_temperature = phase.T
        observed_temperatures = []
        original_ps = phase.Ps

        def record_ps_temperature():
            observed_temperatures.append(phase.T)
            return original_ps()

        phase.Ps = record_ps_temperature
        saturation_temperature = phase.Tsat(1.0e5)

        # The first call evaluates Pmin at Tmin. The second is the first
        # iteration after the pinned half-(Tcrit - Tmin) seed is clipped to
        # Tmin. A half-sum seed incorrectly starts near 360 K for Heptane.
        self.assertEqual(observed_temperatures[0], tmin)
        self.assertEqual(observed_temperatures[1], tmin)
        self.assertGreaterEqual(saturation_temperature, tmin)
        self.assertEqual(phase.T, original_temperature)
