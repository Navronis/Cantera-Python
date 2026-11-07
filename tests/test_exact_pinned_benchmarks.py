"""Exact formula benchmarks against the pinned Cantera commit (726522be4e2a13454d8415b7ef799d621f665cf3).

These tests evaluate exact mathematical expressions derived directly from the C++ headers
and implementation files of the pinned commit, providing deterministic verification
independent of any external binary build.
"""
import math
import unittest

from cantera_python.constants import GasConstant as R, OneAtm, Avogadro, ElectronMass, ElectronCharge
from cantera_python.rates import (
    ArrheniusRate, BlowersMaselRate, TwoTempPlasmaRate,
    LindemannRate, TroeRate, SriRate, TsangRate,
    PlogRate, ChebyshevRate, LinearBurkeCollider, LinearBurkeRate,
    ConstantVibrationalRelaxationRate, CastelaVibrationalRelaxationRate,
    StarikovskiyVibrationalRelaxationRate, MultiStateResolvedVibrationalRelaxationRate,
    InterfaceArrheniusRate, StickingArrheniusRate,
)
from cantera_python.thermo import NasaPoly2, ShomatePoly2, ConstCpPoly
from cantera_python.functions import Sin1, Cos1, Exp1, Log1, Pow1, Poly1, Gaussian1, simpson, trapezoidal


class ExactPinnedBenchmarkTests(unittest.TestCase):
    def assertExactRelative(self, actual, expected, tol=1e-13):
        denom = max(abs(actual), abs(expected), 1e-300)
        err = abs(actual - expected) / denom
        self.assertLessEqual(err, tol, f"Relative error {err:.2e} exceeded tolerance {tol:.2e}: actual={actual}, expected={expected}")

    def test_pinned_arrhenius_exact(self):
        # include/cantera/kinetics/Arrhenius.h: k = A * T^b * exp(-Ea / (R * T))
        A, b, Ea = 3.5e11, 0.75, 4.2e7
        T = 1250.0
        rate = ArrheniusRate(A, b, Ea)
        expected = A * (T ** b) * math.exp(-Ea / (R * T))
        self.assertExactRelative(rate(T), expected)

    def test_pinned_blowers_masel_exact_branches(self):
        # include/cantera/kinetics/BlowersMaselRate.h
        A, b, E0, w = 1.0e10, 0.0, 8.0e7, 1.0e8
        rate = BlowersMaselRate(A, b, E0, w)
        # Branch 1: delta_H <= -4*E0 -> Ea = 0
        self.assertEqual(rate.activation_energy(-3.5e8), 0.0)
        # Branch 2: delta_H >= 4*E0 -> Ea = delta_H
        self.assertEqual(rate.activation_energy(3.5e8), 3.5e8)
        # Branch 3: delta_H == 0 -> Ea = E0
        self.assertEqual(rate.activation_energy(0.0), E0)
        # Continuous formula:
        dh = 5.0e7
        vp = 2.0 * w * ((w + E0) / (w - E0))
        vp_2w_dh = vp - 2.0 * w + dh
        expected_ea = (w + dh / 2.0) * (vp_2w_dh * vp_2w_dh) / (vp * vp - 4.0 * w * w + dh * dh)
        self.assertExactRelative(rate.activation_energy(dh), expected_ea)

    def test_pinned_troe_falloff_exact(self):
        # include/cantera/kinetics/Falloff.h: Troe falloff
        low = ArrheniusRate(1e15, 0.0, 5e7)
        high = ArrheniusRate(1e12, 0.0, 8e7)
        a, T3, T1, T2 = 0.7346, 94.0, 1756.0, 5182.0
        troe = TroeRate(low, high, (a, T3, T1, T2))
        T = 1500.0
        M = 0.05
        k0 = low(T)
        kinf = high(T)
        Pr = k0 * M / kinf
        Fcent = (1.0 - a) * math.exp(-T / T3) + a * math.exp(-T / T1) + math.exp(-T2 / T)
        logPr = math.log10(max(Pr, 1e-300))
        c = -0.4 - 0.67 * math.log10(Fcent)
        n = 0.75 - 1.27 * math.log10(Fcent)
        f1 = (logPr + c) / (n - 0.14 * (logPr + c))
        F = 10.0 ** (math.log10(Fcent) / (1.0 + f1 * f1))
        expected = kinf * (Pr / (1.0 + Pr)) * F
        self.assertExactRelative(troe(T, M), expected)

    def test_pinned_vibrational_relaxation_exact(self):
        # Castela formulation: R * T / p0 * exp(18.42 + a*b - a*T^(-1/3))
        a, b = 20.0, 0.04
        cast = CastelaVibrationalRelaxationRate(a, b, OneAtm)
        T = 800.0
        expected = (R * T / OneAtm) * math.exp(18.42 + a * b - a * (T ** (-1.0 / 3.0)))
        self.assertExactRelative(cast(T), expected)

        # Constant: k_f = A
        const_v = ConstantVibrationalRelaxationRate(9.87e3)
        self.assertEqual(const_v(500.0), 9.87e3)

    def test_pinned_linear_burke_exact(self):
        # src/kinetics/LinearBurkeRate.cpp
        arr = ArrheniusRate(1e10, 0.0, 5e7)
        ref_rate = PlogRate([(1e4, arr), (1e6, arr)])
        eff_rate = ArrheniusRate(2.0, 0.0, 0.0)
        collider = LinearBurkeCollider(efficiency=eff_rate)
        lbr = LinearBurkeRate(reference_rate=ref_rate, colliders={'H2O': collider})
        T = 1000.0
        P = 101325.0
        X = {'H2O': 0.3, 'N2': 0.7}
        self.assertExactRelative(lbr(T, P, X), arr(T))

    def test_pinned_nasa7_exact(self):
        # include/cantera/thermo/NasaPoly2.h
        coeffs_low = [3.5, 1.0e-3, -2.0e-6, 3.0e-9, -1.0e-12, -1000.0, 5.0]
        coeffs_high = [3.0, 1.5e-3, -1.0e-6, 2.0e-9, -5.0e-13, -800.0, 4.0]
        poly = NasaPoly2(1000.0, coeffs_low, coeffs_high, min_temp=200.0, max_temp=3500.0, reference_pressure=OneAtm)
        T = 500.0
        c = coeffs_low
        cp_R_expected = c[0] + c[1]*T + c[2]*T**2 + c[3]*T**3 + c[4]*T**4
        h_RT_expected = c[0] + c[1]*T/2.0 + c[2]*T**2/3.0 + c[3]*T**3/4.0 + c[4]*T**4/5.0 + c[5]/T
        s_R_expected = c[0]*math.log(T) + c[1]*T + c[2]*T**2/2.0 + c[3]*T**3/3.0 + c[4]*T**4/4.0 + c[6]
        cp_R, h_RT, s_R = poly.properties(T)
        self.assertExactRelative(cp_R, cp_R_expected)
        self.assertExactRelative(h_RT, h_RT_expected)
        self.assertExactRelative(s_R, s_R_expected)

    def test_pinned_quadrature_exact(self):
        # Simpson's rule on exact quadratic y = x^2 on [0, 2]: exact integral is 8/3
        x = [0.0, 0.5, 1.0, 1.5, 2.0]
        y = [xi**2 for xi in x]
        self.assertExactRelative(simpson(y, x), 8.0 / 3.0)


if __name__ == '__main__':
    unittest.main()
