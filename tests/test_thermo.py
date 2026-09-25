"""Independent analytical and thermodynamic-identity checks for the port."""

import math
import unittest

from cantera_python.thermo import (
    GAS_CONSTANT as R,
    ConstCpPoly,
    Mu0Poly,
    Nasa9Poly1,
    Nasa9PolyMultiTempRegion,
    NasaPoly1,
    NasaPoly2,
    ShomatePoly,
    ShomatePoly2,
)


class ThermoTests(unittest.TestCase):
    def assert_close(self, actual, expected, relative=1e-12, absolute=1e-10):
        self.assertTrue(math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute),
                        f"{actual!r} != {expected!r}")

    def assert_properties(self, actual, expected):
        for first, second in zip(actual, expected):
            self.assert_close(first, second)

    def test_nasa7_constant_cp_limit(self):
        model = NasaPoly1([3.5, 0, 0, 0, 0, 1200, -2])
        for temperature in (100, 298.15, 1000, 6000):
            self.assert_properties(model.properties(temperature), (
                3.5, 3.5 + 1200 / temperature, 3.5 * math.log(temperature) - 2))

    def test_nasa7_all_terms(self):
        model = NasaPoly1([1, 0.002, 3e-6, 4e-9, 5e-12, 600, 7])
        self.assert_properties(model.properties(1000), (
            15, 5.6, math.log(1000) + 2 + 1.5 + 4 / 3 + 1.25 + 7))

    def test_nasa7_boundary_and_packed_order(self):
        low = [2, 0, 0, 0, 0, 0, 0]
        high = [4, 0, 0, 0, 0, 0, 0]
        model = NasaPoly2.from_coeffs([1000] + high + low, min_temp=200, max_temp=3000)
        self.assertEqual(model.properties(1000)[0], 2)
        self.assertEqual(model.properties(math.nextafter(1000, math.inf))[0], 4)
        self.assertEqual(model.properties(100)[0], 2)
        self.assertEqual(model.properties(4000)[0], 4)

    def test_nasa9_inverse_temperature_terms(self):
        model = Nasa9Poly1([1e6, 2000, 0, 0, 0, 0, 0, 500, 7])
        self.assert_properties(model.properties(1000), (
            3, -1 + 2 * math.log(1000) + 0.5, -0.5 - 2 + 7))

    def test_nasa9_reduces_to_nasa7(self):
        nasa7 = NasaPoly1([1, 0.002, 3e-6, 4e-9, 5e-12, 600, 7])
        nasa9 = Nasa9Poly1([0, 0, 1, 0.002, 3e-6, 4e-9, 5e-12, 600, 7])
        for temperature in (200, 500, 1000, 5000):
            self.assert_properties(nasa9.properties(temperature), nasa7.properties(temperature))

    def test_nasa9_multi_region_boundaries_and_extrapolation(self):
        data = [[0, 0, cp, 0, 0, 0, 0, 0, 0] for cp in (2, 4, 6)]
        model = Nasa9PolyMultiTempRegion([200, 1000, 6000, 20000], data)
        for temperature, expected_cp in ((100, 2), (999, 2), (1000, 4),
                                         (5999, 4), (6000, 6), (30000, 6)):
            self.assertEqual(model.properties(temperature)[0], expected_cp)
        packed = [3]
        for low, high, row in zip([200, 1000, 6000], [1000, 6000, 20000], data):
            packed.extend([low, high] + row)
        unpacked = Nasa9PolyMultiTempRegion.from_coeffs(packed)
        for temperature in (100, 1000, 6000, 30000):
            self.assert_properties(unpacked.properties(temperature), model.properties(temperature))

    def test_nasa9_single_region(self):
        coefficients = [2, 3, 4, 5e-3, 6e-6, 7e-9, 8e-12, 9, 10]
        direct = Nasa9Poly1(coefficients)
        multi = Nasa9PolyMultiTempRegion([200, 5000], [coefficients])
        for temperature in (100, 200, 1000, 5000, 6000):
            self.assert_properties(multi.properties(temperature), direct.properties(temperature))

    def test_shomate_units_and_all_terms(self):
        # At reduced t=1, the dimensional polynomials simplify independently.
        model = ShomatePoly([10, 20, 30, 40, 50, 60, 70])
        self.assert_close(model.cp(1000), 150000)
        self.assert_close(model.h(1000), (10 + 10 + 10 + 10 - 50 + 60) * 1e6)
        self.assert_close(model.s(1000), (20 + 15 + 40 / 3 - 25 + 70) * 1000)

    def test_shomate_boundary_and_packed_order(self):
        low = [20, 0, 0, 0, 0, 0, 0]
        high = [40, 0, 0, 0, 0, 0, 0]
        model = ShomatePoly2.from_coeffs([1000] + low + high)
        self.assert_close(model.cp(1000), 20000)
        self.assert_close(model.cp(math.nextafter(1000, math.inf)), 40000)

    def test_constant_cp_reference_and_change(self):
        model = ConstCpPoly(t0=300, h0=1e6, s0=2e4, cp0=3e4)
        self.assert_close(model.h(300), 1e6)
        self.assert_close(model.s(300), 2e4)
        self.assert_close(model.h(600), 1e6 + 3e4 * 300)
        self.assert_close(model.s(600), 2e4 + 3e4 * math.log(2))
        self.assert_close(model.cp(600), 3e4)

    def test_dimensional_reference_pressure_does_not_add_correction(self):
        first = NasaPoly1([3, 0, 0, 0, 0, 0, 0], reference_pressure=101325)
        second = NasaPoly1([3, 0, 0, 0, 0, 0, 0], reference_pressure=1e6)
        self.assertEqual(first.properties(400), second.properties(400))
        self.assert_close(first.cp(400), 3 * R)
        self.assert_close(first.h(400), 3 * R * 400)

    def test_enthalpy_and_entropy_derivatives(self):
        # Independent identities dh/dT = cp and ds/dT = cp/T.
        models = (
            NasaPoly1([1, 0.002, 3e-6, 4e-9, 5e-12, 600, 7]),
            Nasa9Poly1([1e6, 2000, 1, 0.002, 3e-6, 4e-9, 5e-12, 600, 7]),
            ShomatePoly([10, 20, 30, 40, 50, 60, 70]),
            ConstCpPoly(298.15, 1e6, 1e4, 3e4),
        )
        for model in models:
            for temperature in (400, 1000, 3000):
                step = temperature * 1e-5
                dh_dt = (model.h(temperature + step) - model.h(temperature - step)) / (2 * step)
                ds_dt = (model.s(temperature + step) - model.s(temperature - step)) / (2 * step)
                self.assert_close(dh_dt, model.cp(temperature), relative=1e-8)
                self.assert_close(ds_dt, model.cp(temperature) / temperature, relative=1e-8)

    @staticmethod
    def piecewise_state(temperature):
        """Analytical h/R, s/R and cp/R anchored at 298.15 K."""
        if temperature <= 298.15:
            return (1000 + 2 * (temperature - 298.15),
                    2 + 2 * math.log(temperature / 298.15), 2)
        if temperature <= 500:
            return (1000 + 3 * (temperature - 298.15),
                    2 + 3 * math.log(temperature / 298.15), 3)
        h500 = 1000 + 3 * (500 - 298.15)
        s500 = 2 + 3 * math.log(500 / 298.15)
        return h500 + 4 * (temperature - 500), s500 + 4 * math.log(temperature / 500), 4

    def test_mu0_piecewise_cp_and_exact_gibbs_knots(self):
        points = {}
        for temperature in (200, 298.15, 500, 1000):
            h_r, s_r, _ = self.piecewise_state(temperature)
            points[temperature] = (h_r - temperature * s_r) * R
        model = Mu0Poly(1000 * R, points)
        for temperature in (100, 200, 298.15, 299, 400, 500, 501, 800, 1000, 1500):
            h_r, s_r, cp_r = self.piecewise_state(temperature)
            self.assert_properties(model.properties(temperature), (cp_r, h_r / temperature, s_r))
        for temperature, expected in points.items():
            self.assert_close(model.h(temperature) - temperature * model.s(temperature), expected)

    def test_mu0_anchor_first_or_last_and_two_points(self):
        # Exercises both source recurrence directions and terminal extrapolation.
        for temperatures in ((298.15, 500), (200, 298.15)):
            points = {}
            for temperature in temperatures:
                h_r = 1000 + 2.5 * (temperature - 298.15)
                s_r = 2 + 2.5 * math.log(temperature / 298.15)
                points[temperature] = (h_r - temperature * s_r) * R
            model = Mu0Poly(1000 * R, points)
            for temperature in (100, 250, 298.15, 400, 1000):
                self.assert_properties(model.properties(temperature), (
                    2.5, (1000 + 2.5 * (temperature - 298.15)) / temperature,
                    2 + 2.5 * math.log(temperature / 298.15)))

    def test_invalid_temperature_domain(self):
        models = (NasaPoly1([0] * 7), Nasa9Poly1([0] * 9),
                  ShomatePoly([0] * 7), ConstCpPoly())
        for model in models:
            for temperature in (0, -1, math.nan, math.inf):
                with self.subTest(model=type(model).__name__, temperature=temperature):
                    with self.assertRaises(ValueError):
                        model.properties(temperature)

    def test_invalid_input_data(self):
        constructors = (
            lambda: NasaPoly1([0] * 6),
            lambda: Nasa9Poly1([0] * 8),
            lambda: ShomatePoly([0] * 8),
            lambda: ConstCpPoly(t0=0),
            lambda: Mu0Poly(0, {200: 0, 500: 0}),
            lambda: Mu0Poly(0, {298.15: 0}),
            lambda: Nasa9PolyMultiTempRegion([200, 1000, 500], [[0] * 9, [0] * 9]),
            lambda: Nasa9PolyMultiTempRegion([200, 1000], []),
            lambda: Nasa9PolyMultiTempRegion.from_coeffs([2, 200, 1000] + [0] * 9),
            lambda: NasaPoly1([math.nan] * 7),
        )
        for constructor in constructors:
            with self.assertRaises(ValueError):
                constructor()


if __name__ == "__main__":
    unittest.main()
