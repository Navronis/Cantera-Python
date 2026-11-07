import math
import unittest

from cantera_python.validation import (
    compare_scalar, compare_sequence, summarize, ulp_distance,
)


class ErrorMetricTests(unittest.TestCase):
    def test_ulp_distance_and_signed_zero(self):
        self.assertEqual(ulp_distance(0.0, -0.0), 0)
        one_up = math.nextafter(1.0, math.inf)
        one_down = math.nextafter(1.0, -math.inf)
        self.assertEqual(ulp_distance(1.0, one_up), 1)
        self.assertEqual(ulp_distance(one_down, one_up), 2)
        self.assertIsNone(ulp_distance(math.nan, 1.0))

    def test_mixed_tolerance_and_near_zero_reporting(self):
        result = compare_scalar(2e-30, 1e-30, atol=2e-30, rtol=1e-10,
                                relative_floor=1e-300, near_zero_floor=1e-20)
        self.assertAlmostEqual(result.relative_error, 1.0)
        self.assertTrue(result.near_zero)
        self.assertIsNone(result.meaningful_relative_error)
        self.assertLess(result.mixed_error, 1.0)
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.log10_error, math.log10(2.0))

    def test_sequence_summary_and_nonfinite_behavior(self):
        metrics = compare_sequence([1.0, math.inf], [1.0, math.inf],
                                   atol=1e-12, rtol=1e-9)
        summary = summarize(metrics)
        self.assertEqual(summary['count'], 2)
        self.assertEqual(summary['failed'], 0)
        bad = compare_scalar(math.inf, 1.0, atol=1e-12, rtol=1e-9)
        self.assertFalse(bad.passed)
        self.assertTrue(math.isinf(bad.absolute_error))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            compare_scalar(1.0, 1.0, atol=-1.0, rtol=1e-9)
        with self.assertRaises(ValueError):
            compare_sequence([1.0], [1.0, 2.0], atol=0.0, rtol=0.0)


if __name__ == '__main__':
    unittest.main()
