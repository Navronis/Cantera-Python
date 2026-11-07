"""Pinned ``set_TPp`` loop-count and failure-state semantics."""

import unittest

from cantera_python.tpx import PropertyPair, TPXError, Water


class TestSetTPpIterationLimit(unittest.TestCase):
    def test_active_bracket_moves_base_to_upper_bound_and_halves_damping(self):
        phase = Water()
        phase.kbr = 1

        v_here, dv, dvbf = phase._bracket_slope(
            pressure=16.0, v_here=2.0, vmin=1.0, vmax=3.0,
            pmin=10.0, pmax=14.0, dvbf=1.0,
        )

        self.assertEqual(v_here, 3.0)
        self.assertEqual(dv, 1.0)
        self.assertEqual(dvbf, 0.5)

    def test_throws_after_201_updates_and_restores_the_input_volume(self):
        phase = Water()
        phase.Set(PropertyPair.TV, 300.0, 0.01)
        original_volume = 1.0 / phase.Rho
        calls = 0

        def constant_pressure():
            nonlocal calls
            calls += 1
            return 1.0

        phase.Pp = constant_pressure
        with self.assertRaisesRegex(TPXError, "did not converge"):
            phase.set_TPp(350.0, 2.0)

        # Pinned Sub.cpp evaluates pressure and a finite-difference pressure
        # slope once each per update, then throws when LoopCount exceeds 200.
        self.assertEqual(calls, 402)
        self.assertEqual(phase.T, 350.0)
        self.assertAlmostEqual(1.0 / phase.Rho, original_volume, places=14)
