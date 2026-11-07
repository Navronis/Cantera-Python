"""Verification tests for PlogRate, ChebyshevRate, and LinearBurkeRate models."""
import math
import unittest
from pathlib import Path

import cantera_python as ct
from cantera_python.rates import (
    ArrheniusRate, PlogRate, ChebyshevRate, LinearBurkeCollider, LinearBurkeRate,
)

DATA_DIR = Path(__file__).parents[2] / 'cantera-original' / 'data'


class PlogRateTests(unittest.TestCase):
    def setUp(self):
        # 3 pressures: 1 atm, 10 atm, 100 atm
        # P1 = 101325: k1 = 1e11 * T**0.5 * exp(-1e7 / RT)
        # P2 = 1013250: k2 = 2e11 * T**0.5 * exp(-1e7 / RT)
        # P3 = 10132500: k3 = 4e11 * T**0.5 * exp(-1e7 / RT)
        self.r1 = ArrheniusRate(1e11, 0.5, 1e7)
        self.r2 = ArrheniusRate(2e11, 0.5, 1e7)
        self.r3 = ArrheniusRate(4e11, 0.5, 1e7)
        self.p1 = 101325.0
        self.p2 = 1013250.0
        self.p3 = 10132500.0
        self.plog = PlogRate([
            (self.p1, self.r1),
            (self.p2, self.r2),
            (self.p3, self.r3),
        ])

    def test_exact_pressures(self):
        T = 1000.0
        self.assertAlmostEqual(self.plog(T, self.p1) / self.r1(T), 1.0, places=12)
        self.assertAlmostEqual(self.plog(T, self.p2) / self.r2(T), 1.0, places=12)
        self.assertAlmostEqual(self.plog(T, self.p3) / self.r3(T), 1.0, places=12)

    def test_log_linear_interpolation(self):
        T = 1000.0
        p_mid = math.sqrt(self.p1 * self.p2)
        k_expected = math.sqrt(self.r1(T) * self.r2(T))
        self.assertAlmostEqual(self.plog(T, p_mid) / k_expected, 1.0, places=12)

    def test_pressure_clamping_extrapolation(self):
        T = 1000.0
        self.assertAlmostEqual(self.plog(T, 0.5 * self.p1) / self.r1(T), 1.0, places=12)
        self.assertAlmostEqual(self.plog(T, 1e-3) / self.r1(T), 1.0, places=12)
        self.assertAlmostEqual(self.plog(T, 2.0 * self.p3) / self.r3(T), 1.0, places=12)
        self.assertAlmostEqual(self.plog(T, 1e9) / self.r3(T), 1.0, places=12)

    def test_multi_rate_at_same_pressure(self):
        r_b = ArrheniusRate(5e10, 0.5, 1e7)
        plog2 = PlogRate([
            (self.p1, self.r1),
            (self.p1, r_b),
            (self.p2, self.r2),
        ])
        T = 1200.0
        expected = self.r1(T) + r_b(T)
        self.assertAlmostEqual(plog2(T, self.p1) / expected, 1.0, places=12)

    def test_validation(self):
        self.plog.validate()


class ChebyshevRateTests(unittest.TestCase):
    def setUp(self):
        self.tmin, self.tmax = 300.0, 2500.0
        self.pmin, self.pmax = 1000.0, 10000000.0
        # 2x2 coefficient matrix
        # phi_0(x) = 1, phi_1(x) = x
        # log10(k) = a00 + a01 * Pr + a10 * Tr + a11 * Tr * Pr
        self.coeffs = [
            [10.0, 0.5],
            [0.2, -0.1],
        ]
        self.cheb = ChebyshevRate(self.tmin, self.tmax, self.pmin, self.pmax, self.coeffs)

    def test_reduced_coordinates(self):
        # At boundary coordinates:
        # T = Tmin => Tr = -1, T = Tmax => Tr = +1
        # P = Pmin => Pr = -1, P = Pmax => Pr = +1
        tr_min, pr_min = self.cheb.reduced_coordinates(self.tmin, self.pmin)
        self.assertAlmostEqual(tr_min, -1.0, places=10)
        self.assertAlmostEqual(pr_min, -1.0, places=10)

        tr_max, pr_max = self.cheb.reduced_coordinates(self.tmax, self.pmax)
        self.assertAlmostEqual(tr_max, 1.0, places=10)
        self.assertAlmostEqual(pr_max, 1.0, places=10)

    def test_evaluation(self):
        T = 1000.0
        P = 101325.0
        Tr, Pr = self.cheb.reduced_coordinates(T, P)
        expected_log10 = (self.coeffs[0][0]
                          + self.coeffs[0][1] * Pr
                          + self.coeffs[1][0] * Tr
                          + self.coeffs[1][1] * Tr * Pr)
        expected_k = 10.0 ** expected_log10
        self.assertAlmostEqual(self.cheb.log10_rate(T, P), expected_log10, places=10)
        self.assertAlmostEqual(self.cheb(T, P) / expected_k, 1.0, places=12)


class LinearBurkeRateTests(unittest.TestCase):
    def test_solution_linear_burke_kinetics(self):
        path = DATA_DIR / 'example_data' / 'ammonia-CO-H2-Alzueta-2023.yaml'
        if not path.is_file():
            self.skipTest(f"Data file {path} not found")
        gas = ct.Solution(path, phase_name='linear-Burke')
        gas.TPX = 1000.0, ct.OneAtm, 'NH3:0.05, O2:0.1, N2:0.85'
        expected = (
            151912438.38841602,
            54834620.65122594,
            99.26298897447236,
            9.704495488783856e-10,
            1385547364.2182477,
            9.267852186955833,
        )
        for actual, reference in zip(gas.forward_rate_constants[-6:], expected):
            self.assertAlmostEqual(actual / reference, 1.0, places=12)


if __name__ == '__main__':
    unittest.main()
