"""Source regressions and independent mathematical checks of functions.py.

Golden cases are adapted from upstream test/general/test_numerics.cpp at
726522be4e2a13454d8415b7ef799d621f665cf3. Finite-difference and polynomial
integral checks independently test derivative and quadrature equations.
These tests do not constitute validation of the rest of Cantera.
"""

import math
import random
import unittest

from cantera_python import functions as fn


class ScalarFunctorTests(unittest.TestCase):
    def test_source_basic_values(self):
        cases = [
            (fn.Sin1(2), math.sin, 2),
            (fn.Cos1(2), math.cos, 2),
            (fn.Exp1(2), math.exp, 2),
            (fn.Log1(2), math.log, 2),
        ]
        for function, reference, factor in cases:
            for x in (0.1, 0.5, 2.0):
                with self.subTest(kind=function.type(), x=x):
                    self.assertEqual(function(x), reference(factor * x))
        self.assertEqual(fn.Pow1(0.5)(0.25), 0.5)
        self.assertEqual(fn.Const1(1.25)(-100), 1.25)

    def test_single_parameter_overloads(self):
        for cls in (fn.Sin1, fn.Cos1, fn.Exp1, fn.Log1, fn.Pow1, fn.Const1):
            self.assertEqual(cls([2.0])(0.4), cls(2.0)(0.4))
            with self.assertRaises(ValueError):
                cls([])
            with self.assertRaises(ValueError):
                cls([1, 2])

    def test_base_and_structural_identity(self):
        base = fn.Func1()
        self.assertEqual(base(1), 0)
        self.assertEqual(base.type(), "functor")
        self.assertFalse(base.isIdentical(base))
        self.assertTrue(fn.Sin1(2).isIdentical(fn.Sin1(2)))
        self.assertFalse(fn.Sin1(2).isIdentical(fn.Sin1(3)))
        self.assertFalse(fn.Sin1(2).isIdentical(fn.Cos1(2)))
        self.assertFalse(fn.Sin1().isIdentical(object()))
        f = fn.Product1(fn.Sin1(2), fn.Cos1(3))
        self.assertTrue(f.isIdentical(fn.Product1(fn.Sin1(2), fn.Cos1(3))))
        self.assertFalse(f.isIdentical(fn.Product1(fn.Sin1(2), fn.Cos1(4))))
        self.assertEqual(f.func1_shared().c(), 2)
        self.assertEqual(f.func2_shared().c(), 3)
        self.assertEqual(fn.TimesConstant1(fn.Sin1(), 4).c(), 4)

    def test_analytic_derivatives_against_finite_differences(self):
        functions = [
            fn.Sin1(2.7), fn.Cos1(-1.8), fn.Exp1(0.25), fn.Log1(1),
            fn.Pow1(2.3), fn.Const1(4),
            fn.Sum1(fn.Sin1(2), fn.Exp1(0.3)),
            fn.Diff1(fn.Cos1(2), fn.Pow1(3)),
            fn.Product1(fn.Sin1(2), fn.Exp1(0.5)),
            fn.Ratio1(fn.Sin1(1), fn.Exp1(0.2)),
            fn.Composite1(fn.Sin1(2), fn.Cos1(0.7)),
            fn.TimesConstant1(fn.Sin1(1.2), -3),
            fn.PlusConstant1(fn.Pow1(2.1), -5),
        ]
        for f in functions:
            derivative = f.derivative()
            for x in (0.13, 0.7, 1.4):
                h = 1e-5
                # Fourth-order centered difference, independent of the
                # port's symbolic derivative and simplification machinery.
                reference = (f(x - 2*h) - 8*f(x - h) + 8*f(x + h) - f(x + 2*h))/(12*h)
                with self.subTest(kind=f.type(), x=x):
                    self.assertAlmostEqual(derivative(x), reference, delta=2e-9)

    def test_pinned_log_derivative_discrepancy(self):
        # Upstream ctfunc/log explicitly tests a/x. Preserve and expose this
        # inconsistency instead of presenting it as a verified analytic rule.
        f = fn.Log1(2)
        self.assertEqual(f.derivative()(0.5), 4)
        h = 1e-5
        analytic_numerical = (f(0.5+h) - f(0.5-h))/(2*h)
        self.assertAlmostEqual(analytic_numerical, 2, delta=1e-8)
        self.assertGreater(abs(f.derivative()(0.5) - analytic_numerical), 1)

    def test_power_derivative_edges_and_repeated_derivatives(self):
        self.assertEqual(fn.Pow1(0).derivative()(0), 0)
        self.assertEqual(fn.Pow1(1).derivative()(0), 1)
        f = fn.Pow1(4)
        for order, expected in enumerate((16, 32, 48, 48, 24, 0)):
            self.assertEqual(f(2), expected, f"derivative order {order}")
            f = f.derivative()

    def test_compound_source_golden_values(self):
        s, c, x = fn.Sin1(2), fn.Cos1(2), 0.6
        ss, cc = math.sin(2*x), math.cos(2*x)
        cases = [
            (fn.Sum1(s, c), ss+cc, 2*(cc-ss)),
            (fn.Diff1(s, c), ss-cc, 2*(cc+ss)),
            (fn.Product1(s, c), ss*cc, 2*(cc*cc-ss*ss)),
            (fn.Ratio1(s, c), ss/cc, 2/(cc*cc)),
            (fn.Composite1(s, c), math.sin(2*cc), -4*ss*math.cos(2*cc)),
        ]
        for f, value, derivative in cases:
            with self.subTest(kind=f.type()):
                self.assertAlmostEqual(f(x), value, places=14)
                self.assertAlmostEqual(f.derivative()(x), derivative, places=13)

    def test_polynomial_order_and_zero_polynomial(self):
        f = fn.Poly1([0.125, 0.25, 0.5])
        self.assertEqual(f(0), 0.5)
        self.assertEqual(f(0.5), 0.65625)
        self.assertEqual(fn.Poly1([0])(123), 0)
        self.assertEqual(fn.Poly1([1, 0, -1])(3), 8)
        with self.assertRaises(ValueError):
            fn.Poly1([])

    def test_gaussian_peak_fwhm_symmetry(self):
        f = fn.Gaussian1(3, 2, 0.8)
        g = fn.Gaussian1([3, 2, 0.8])
        self.assertEqual(f(2), 3)
        for t in (1.6, 2.4):
            self.assertAlmostEqual(f(t), 1.5, places=14)
            self.assertEqual(f(t), g(t))
        self.assertAlmostEqual(f(1), f(3), places=14)
        with self.assertRaises(ValueError):
            fn.Gaussian1([1, 2])

    def test_fourier_harmonics_and_constructor_order(self):
        f = fn.Fourier1([4, 3, -2, 1, 5, 7])
        g = fn.Fourier1(1, 4, [3, -2], [5, 7])
        self.assertEqual(f(0), 3)
        self.assertAlmostEqual(f(math.pi), -3, places=13)
        self.assertAlmostEqual(f(math.pi/2), 9, places=13)
        for t in (-0.3, 0, 0.7):
            self.assertEqual(f(t), g(t))
        self.assertEqual(fn.Fourier1(2, 6, [], [])(10), 3)
        for params in ([], [1, 2], [1, 2, 3, 4, 5]):
            with self.assertRaises(ValueError):
                fn.Fourier1(params)
        with self.assertRaises(ValueError):
            fn.Fourier1(1, 2, [1], [2, 3])

    def test_arrhenius_terms_and_temperature_units(self):
        self.assertEqual(fn.Arrhenius1([2, 2, 0, 3, 1, 0])(5), 65)
        f = fn.Arrhenius1([1, 0, 1000])
        self.assertAlmostEqual(f(1000), 1/math.e, places=15)
        self.assertAlmostEqual(f(500)/f(1000), 1/math.e, places=15)
        for params in ([], [1, 2], [1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                fn.Arrhenius1(params)

    def test_periodic_truncation_for_negative_times(self):
        f = fn.Periodic1(fn.Pow1(1), 2)
        self.assertEqual(f(0.5), 0.5)
        self.assertEqual(f(4.5), 0.5)
        self.assertEqual(f(4), 0)
        self.assertEqual(f(-0.5), -0.5)
        self.assertEqual(f(-4.5), -0.5)

    def test_original_unsupported_derivatives(self):
        functions = [fn.Func1(), fn.Poly1([1, 2]), fn.Gaussian1(1, 2, 3),
                     fn.Fourier1([1, 2, 3, 4]), fn.Arrhenius1([1, 2, 3]),
                     fn.Periodic1(fn.Sin1(), 2)]
        for f in functions:
            with self.subTest(kind=f.type()):
                with self.assertRaisesRegex(NotImplementedError, "pinned source"):
                    f.derivative()
        for f in functions[1:-1]:
            self.assertFalse(f.isIdentical(f))

    def test_latex_source_regressions(self):
        self.assertEqual(fn.Sin1(2).write("t"), r"\sin(2t)")
        self.assertEqual(fn.Cos1().write("t"), r"\cos(t)")
        self.assertEqual(fn.Exp1().write(), r"\exp(x)")
        self.assertEqual(fn.Log1(2).write(), r"\log(2x)")
        self.assertEqual(fn.Pow1(0.5).write(), r"\sqrt{x}")
        self.assertEqual(fn.Pow1(-0.5).write(), r"\frac{1}{\sqrt{x}}")
        self.assertEqual(fn.Pow1(2).write(), r"\left(x\right)^{2}")
        self.assertEqual(fn.Pow1(1).write(), "x")
        self.assertEqual(fn.Poly1([1, 0, -1]+[0]*8).write(), "x^{10} - x^8")
        self.assertEqual(fn.Poly1([1, -1, 2]).write(), "x^2 - x + 2")
        self.assertEqual(fn.Ratio1(fn.Sin1(), fn.Cos1()).write(), r"\frac{\sin(x)}{\cos(x)}")
        self.assertEqual(fn.Composite1(fn.Sin1(), fn.Exp1()).write(), r"\sin(\exp(x))")
        self.assertEqual(fn.Sum1(fn.Sin1(), fn.Const1(-2)).write(), r"\sin(x) - 2")
        self.assertEqual(fn.Diff1(fn.Sin1(), fn.Const1(-2)).write(), r"\sin(x) + 2")
        self.assertEqual(fn.TimesConstant1(fn.Const1(2), 3).write(), r"3\left(2\right)")
        self.assertEqual(fn.PlusConstant1(fn.Sin1(), 0).write(), r"\sin(x)")
        self.assertEqual(fn.Product1(fn.Sum1(fn.Sin1(), fn.Cos1()), fn.Pow1(1)).write(),
                         r"\left(\sin(x) + \cos(x)\right) x")

    def test_python_domain_errors_are_explicit(self):
        with self.assertRaises(ValueError):
            fn.Log1()(-1)
        with self.assertRaises(ValueError):
            fn.Pow1(0.5)(-1)
        with self.assertRaises(OverflowError):
            fn.Exp1()(10000)
        with self.assertRaises(ZeroDivisionError):
            fn.Ratio1(fn.Const1(1), fn.Pow1(1))(0)


class SimplificationTests(unittest.TestCase):
    def test_zero_one_and_proportional_sum_difference(self):
        f = fn.Sin1()
        self.assertIs(fn.newSumFunction(f, fn.Const1(0)), f)
        self.assertIs(fn.newSumFunction(fn.Const1(0), f), f)
        self.assertIs(fn.newDiffFunction(f, fn.Const1(0)), f)
        self.assertEqual(fn.newDiffFunction(fn.Const1(0), f)(0.5), -math.sin(0.5))
        self.assertEqual(fn.newSumFunction(f, f).c(), 2)
        self.assertEqual(fn.newDiffFunction(f, f).c(), 0)
        times = fn.newTimesConstFunction(f, 2.5)
        for a, b in ((f, times), (times, f)):
            self.assertEqual(fn.newSumFunction(a, b).c(), 3.5)
        self.assertEqual(fn.newDiffFunction(f, times).c(), -1.5)
        self.assertEqual(fn.newDiffFunction(times, f).c(), 1.5)
        self.assertEqual(fn.newSumFunction(times, times).c(), 5)
        minus = fn.newTimesConstFunction(f, -2.5)
        self.assertEqual(fn.newSumFunction(times, minus).type(), "constant")
        self.assertEqual(fn.newSumFunction(times, minus)(0.4), 0)
        self.assertEqual(fn.newDiffFunction(times, minus).c(), 5)

    def test_product_ratio_and_composition_simplification(self):
        f = fn.Sin1()
        self.assertIs(fn.newProdFunction(f, fn.Const1(1)), f)
        self.assertIs(fn.newProdFunction(fn.Const1(1), f), f)
        self.assertEqual(fn.newProdFunction(f, fn.Const1(0)).c(), 0)
        self.assertEqual(fn.newProdFunction(fn.Const1(2), fn.Const1(2.5)).c(), 5)
        self.assertEqual(fn.newProdFunction(fn.Pow1(2), fn.Pow1(3)).c(), 5)
        self.assertEqual(fn.newProdFunction(fn.Exp1(2), fn.Exp1(3)).c(), 5)
        self.assertEqual(fn.newRatioFunction(fn.Pow1(3), fn.Pow1(2)).c(), 1)
        self.assertEqual(fn.newRatioFunction(fn.Exp1(3), fn.Exp1(2)).c(), 1)
        self.assertIs(fn.newRatioFunction(f, fn.Const1(1)), f)
        self.assertEqual(fn.newRatioFunction(f, f)(0.5), 1)
        with self.assertRaises(ZeroDivisionError):
            fn.newRatioFunction(f, fn.Const1(0))
        # The original checks zero numerator before zero denominator.
        self.assertEqual(fn.newRatioFunction(fn.Const1(0), fn.Const1(0))(1), 0)
        self.assertIs(fn.newCompositeFunction(fn.Pow1(1), f), f)
        self.assertEqual(fn.newCompositeFunction(fn.Pow1(0), f)(2), 1)
        self.assertEqual(fn.newCompositeFunction(fn.Pow1(2), fn.Pow1(3)).c(), 6)
        self.assertEqual(fn.newCompositeFunction(fn.Const1(3), f)(2), 3)
        self.assertEqual(fn.newCompositeFunction(fn.Const1(0), f)(2), 0)

    def test_nested_constant_modifiers(self):
        f = fn.Sin1()
        self.assertIs(fn.newTimesConstFunction(f, 1), f)
        self.assertEqual(fn.newTimesConstFunction(f, 0).type(), "constant")
        self.assertIs(fn.newPlusConstFunction(f, 0), f)
        times = fn.newTimesConstFunction(fn.TimesConstant1(f, 2), 3)
        self.assertIs(times.func1_shared(), f)
        self.assertEqual(times.c(), 6)
        plus = fn.newPlusConstFunction(fn.PlusConstant1(f, 2), 3)
        self.assertIs(plus.func1_shared(), f)
        self.assertEqual(plus.c(), 5)
        self.assertEqual(fn.newPlusConstFunction(fn.Const1(2), 3).c(), 5)

    def test_simplifiers_preserve_sampled_values(self):
        rng = random.Random(2061)
        functions = [fn.Sin1(1.2), fn.Cos1(0.4), fn.Exp1(-0.5), fn.Pow1(2),
                     fn.Const1(3), fn.TimesConstant1(fn.Sin1(1.2), -2),
                     fn.PlusConstant1(fn.Cos1(0.4), 3)]
        for first in functions:
            for second in functions:
                for _ in range(3):
                    x = rng.uniform(0.1, 1)
                    f1, f2 = first(x), second(x)
                    cases = [(fn.newSumFunction, f1+f2), (fn.newDiffFunction, f1-f2),
                             (fn.newProdFunction, f1*f2), (fn.newRatioFunction, f1/f2)]
                    for helper, expected in cases:
                        with self.subTest(helper=helper.__name__, first=first.type(), second=second.type()):
                            self.assertAlmostEqual(helper(first, second)(x), expected,
                                                   delta=1e-12 * max(1, abs(expected)))


class TabulatedTests(unittest.TestCase):
    def test_linear_interpolation_clamping_and_derivative(self):
        f = fn.Tabulated1([0, 1, 2], [1, 0, 1])
        g = fn.Tabulated1([0, 1, 2, 1, 0, 1])
        for x, expected in [(-1, 1), (0, 1), (0.5, 0.5), (1, 0), (1.2, 0.2), (2, 1), (3, 1)]:
            self.assertAlmostEqual(f(x), expected, places=14)
            self.assertEqual(f(x), g(x))
        df = f.derivative()
        for x, expected in [(-1, -1), (0.5, -1), (1, -1), (1.5, 1), (2, 0), (3, 0)]:
            self.assertEqual(df(x), expected)
        self.assertFalse(f.isIdentical(f))
        self.assertEqual(f.write(), r"\mathrm{Tabulated}(x)")

    def test_previous_at_knots_and_next_floating_value(self):
        f = fn.Tabulated1([0, 1, 2], [1, 0, 1], "previous")
        self.assertEqual(f(1), 1)  # left value AT the interior knot
        self.assertEqual(f(math.nextafter(1, 2)), 0)
        self.assertEqual(f(math.nextafter(2, 1)), 0)
        self.assertEqual(f(2), 1)  # final endpoint has a separate branch
        for x in (-1, 0, 1, 1.5, 2, 3):
            self.assertEqual(f.derivative()(x), 0)
        f.setMethod("linear")
        self.assertEqual(f.type(), "tabulated-linear")
        self.assertEqual(f(0.5), 0.5)

    def test_one_point_and_duplicate_knots(self):
        f = fn.Tabulated1([1], [2])
        for x in (-10, 1, 10):
            self.assertEqual(f(x), 2)
            self.assertEqual(f.derivative()(x), 0)
        g = fn.Tabulated1([0, 1, 1, 2], [0, 1, 3, 4])
        self.assertEqual(g(1), 1)
        self.assertAlmostEqual(g(1.5), 3.5)
        # Upstream produces a nonfinite slope for a duplicate time;
        # Python deliberately raises at that same division by zero.
        with self.assertRaises(ZeroDivisionError):
            g.derivative()

    def test_invalid_table_inputs(self):
        for times, values in [([], []), ([0, 1], [1]), ([0, 2, 1], [1, 2, 3])]:
            with self.assertRaises(ValueError):
                fn.Tabulated1(times, values)
        for params in ([], [0, 1], [0, 1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                fn.Tabulated1(params)
        with self.assertRaises(ValueError):
            fn.Tabulated1([0, 1], [2, 3], "spline")


class SampledQuadratureTests(unittest.TestCase):
    def test_linear_interp(self):
        x, y = [0, 0.3, 1, 1.2], [1, 2, 5, 0]
        for coordinate, expected in [(-1, 1), (0, 1), (0.15, 1.5), (0.3, 2),
                                     (0.65, 3.5), (1, 5), (1.1, 2.5), (2, 0)]:
            self.assertAlmostEqual(fn.linearInterp(coordinate, x, y), expected, places=14)
        self.assertEqual(fn.linearInterp(10, [1], [2]), 2)
        for grid, data in [([], []), ([1], []), ([1, 2], [1])]:
            with self.assertRaises(ValueError):
                fn.linearInterp(0, grid, data)

    def test_source_quadrature_golden_values(self):
        x, y = [0, 0.3, 1, 1.2], [1, 2, 5, 0]
        self.assertAlmostEqual(fn.trapezoidal(y, x), 3.4, places=14)
        self.assertAlmostEqual(fn.simpson(y[:3], x[:3]), 2.84127, delta=1e-5)
        self.assertAlmostEqual(fn.simpson(y, x), 3.34127, delta=1e-5)
        self.assertEqual(fn.numericalQuadrature("simpson", y, x), fn.simpson(y, x))
        self.assertEqual(fn.numericalQuadrature("trapezoidal", y, x), fn.trapezoidal(y, x))

    def test_quadratic_exact_integral_on_unequal_grids(self):
        grids = [[0, 0.2, 1], [-2, -1, -0.2, 0.1, 3],
                 [-1, -0.8, 0, 0.3, 1, 1.7, 2]]
        for x in grids:
            y = [3*t*t - 2*t + 5 for t in x]
            primitive = lambda t: t**3 - t**2 + 5*t
            expected = primitive(x[-1]) - primitive(x[0])
            self.assertAlmostEqual(fn.simpson(y, x), expected, places=12)

    def test_affine_exact_trapezoidal_integral(self):
        x = [-3, -2.9, 0.1, 1.2, 4]
        y = [2*t+3 for t in x]
        primitive = lambda t: t*t+3*t
        self.assertAlmostEqual(fn.trapezoidal(y, x), primitive(x[-1])-primitive(x[0]), places=13)
        self.assertEqual(fn.trapezoidal([3], [1]), 0)

    def test_even_simpson_uses_final_trapezoid(self):
        # Integral x^2 from 0 to 1 is 1/3; the final trapezoid on [1,2]
        # contributes 2.5. This checks Cantera's even-count convention.
        self.assertAlmostEqual(fn.simpson([0, 0.25, 1, 4], [0, 0.5, 1, 2]), 1/3+2.5)
        self.assertEqual(fn.simpson([2, 4], [1, 3]), 6)

    def test_quadrature_input_validation(self):
        for rule in (fn.trapezoidal, fn.simpson, fn.basicSimpson):
            for f, x in [([], []), ([1, 2], [0]), ([1, 2, 3], [0, 1, 1]),
                         ([1, 2, 3], [2, 1, 0])]:
                with self.subTest(rule=rule.__name__, f=f, x=x):
                    with self.assertRaises(ValueError):
                        rule(f, x)
        with self.assertRaises(ValueError):
            fn.basicSimpson([1, 2, 3, 4], [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            fn.simpson([1], [0])
        with self.assertRaises(ValueError):
            fn.numericalQuadrature("unknown", [1, 2], [0, 1])


if __name__ == "__main__":
    unittest.main()
