"""Independent identities, branch checks, and derivative tests for rate ports."""

import math
import unittest

from cantera_python.rates import (
    GAS_CONSTANT as R, SMALL_NUMBER, ArrheniusRate, TwoTempPlasmaRate,
    BlowersMaselRate, ElectronCollisionPlasmaRate, LindemannRate, TroeRate, SriRate, TsangRate,
    PlogRate, ChebyshevRate, LinearBurkeCollider, LinearBurkeRate,
    CustomFunc1Rate,
    VibrationalRelaxationRate, ConstantVibrationalRelaxationRate,
    CastelaVibrationalRelaxationRate, StarikovskiyVibrationalRelaxationRate,
    MultiStateResolvedVibrationalRelaxationRate,
    InterfaceArrheniusRate, StickingArrheniusRate,
    InterfaceBlowersMaselRate, StickingBlowersMaselRate,
)
from cantera_python.functions import Const1


class RateTests(unittest.TestCase):
    def assertRelative(self, actual, expected, tol=2e-12):
        self.assertTrue(math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-300),
                        f"{actual!r} != {expected!r}")

    def test_arrhenius_energy_units_and_derivative(self):
        rate = ArrheniusRate(2.0, 1.5, 1000.0 * R)
        self.assertTrue(math.isnan(rate.order))
        self.assertIs(rate.set_rate_order(2.5), rate)
        self.assertEqual(rate.order, 2.5)
        self.assertRelative(rate(1000.0), 2.0 * 1000.0**1.5 / math.e)
        self.assertRelative(math.exp(rate.log_rate(1000.0)), rate(1000.0))
        T, step = 750.0, 1e-3
        numerical = (rate(T + step) - rate(T - step)) / (2 * step) / rate(T)
        self.assertRelative(rate.ddT_scaled(T), numerical, 1e-9)

    def test_blowers_masel_live_enthalpy_state(self):
        class Thermo:
            species_names = ('A', 'B')
            partial_molar_enthalpies = (0.0, 0.0)

        thermo = Thermo()
        rate = BlowersMaselRate(3.0, 0.0, 10.0 * R, 100.0 * R)
        rate.set_rate_order(1.0)
        self.assertEqual(rate.order, 1.0)

        # Exercise all pinned effective-barrier branches through live thermo
        # updates, rather than supplying delta H directly to evaluation.
        for delta_h, expected_ea in ((-50.0 * R, 0.0),
                                     (50.0 * R, 50.0 * R),
                                     (0.0, 10.0 * R)):
            thermo.partial_molar_enthalpies = (0.0, delta_h)
            rate.update_from_thermo(thermo, {'A': 1.0}, {'B': 1.0})
            self.assertEqual(rate.delta_enthalpy, delta_h)
            self.assertRelative(rate.activation_energy(), expected_ea)
            self.assertRelative(rate(1000.0),
                                3.0 * math.exp(-expected_ea / R / 1000.0))

    def test_arrhenius_zero_and_negative_A(self):
        self.assertEqual(ArrheniusRate(0.0, 0.0)(500.0), 0.0)
        self.assertTrue(math.isnan(ArrheniusRate(0.0, 0.0).log_rate(500.0)))
        with self.assertRaises(ValueError):
            ArrheniusRate(-1.0, 0.0)
        negative = ArrheniusRate(-2.0, 0.0, allow_negative_A=True)
        self.assertEqual(negative(700.0), -2.0)
        self.assertTrue(math.isnan(negative.log_rate(700.0)))

    def test_plasma_equal_temperatures_reduce_to_arrhenius(self):
        plasma = TwoTempPlasmaRate(2.3e5, 0.7, 1.3e7, 7.2e7, bg=0.2)
        arrhenius = ArrheniusRate(2.3e5, 0.9, 1.3e7)
        for T in (300.0, 900.0, 3500.0):
            self.assertRelative(plasma(T, T), arrhenius(T))

    def test_plasma_all_terms_and_partial_derivative(self):
        rate = TwoTempPlasmaRate(4.0, 0.5, 2.0 * R, 5.0 * R, 0.3, 5000.0)
        T, Te = 300.0, 10000.0
        expected = (4.0 * T**0.3 * Te**0.5 * math.exp(-2.0 / T)
                    * math.exp(5.0 * (Te - T) / (T * Te)) * math.exp(-T / 5000.0))
        self.assertRelative(rate(T, Te), expected)
        step = 1e-3
        numerical = (rate(T + step, Te) - rate(T - step, Te)) / (2 * step) / rate(T, Te)
        self.assertRelative(rate.ddT_scaled(T, Te), numerical, 2e-9)

    def test_electron_collision_forward_simpson_and_interpolation(self):
        rate = ElectronCollisionPlasmaRate((0.0, 1.0, 2.0), (0.0, 2.0, 4.0),
                                           kind="effective", target="N2")
        prefactor = math.sqrt(2.0 * 1.602176634e-19 / 9.1093837015e-31) * 6.02214076e26
        # Simpson integral of epsilon * sigma over [0, 1, 2] is 16/3.
        self.assertRelative(rate((0.0, 1.0, 2.0), (1.0, 1.0, 1.0)), prefactor * 16.0 / 3.0)
        self.assertEqual(rate.interpolated_cross_sections((0.0, 0.5, 2.0)), (0.0, 1.0, 4.0))

    def test_electron_collision_superelastic_shift_and_inputs(self):
        rate = ElectronCollisionPlasmaRate((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
        prefactor = math.sqrt(2.0 * 1.602176634e-19 / 9.1093837015e-31) * 6.02214076e26
        # Offset grid [0,1,2]; Simpson of (eps+1)*sigma_super is 26/3.
        self.assertRelative(rate.superelastic_rate((0.0, 1.0, 2.0), (1.0, 1.0, 1.0)),
                            prefactor * 26.0 / 3.0)
        with self.assertRaises(ValueError):
            ElectronCollisionPlasmaRate((0.0, 1.0), (1.0,))
        with self.assertRaises(ValueError):
            rate((0.0, 0.0), (1.0, 1.0))

    def test_custom_func1_rate_callable_func1_and_unconfigured_paths(self):
        custom = CustomFunc1Rate(lambda T: 38.7 * T**2.7 * math.exp(-3150.15428 / T))
        expected = 38.7 * 900.0**2.7 * math.exp(-3150.15428 / 900.0)
        self.assertRelative(custom(900.0), expected)
        custom.set_rate_function(Const1(3.25))
        self.assertEqual(custom(500.0), 3.25)
        empty = CustomFunc1Rate()
        self.assertTrue(math.isnan(empty(500.0)))
        with self.assertRaisesRegex(ValueError, 'not configured'):
            empty.validate('A => B')
        with self.assertRaises(TypeError):
            CustomFunc1Rate('not a function')

    def test_blowers_masel_three_branches_and_boundaries(self):
        Ea0 = 1e8
        rate = BlowersMaselRate(3.0, 0.0, Ea0, 1e9)
        self.assertEqual(rate.activation_energy(-5 * Ea0), 0.0)
        self.assertEqual(rate.activation_energy(-4 * Ea0), 0.0)
        self.assertRelative(rate.activation_energy(0.0), Ea0)
        # Exactly +4Ea0 still uses the rational branch. For w=10Ea0 its
        # dimensionless value is 4332/1081; it is not exactly 4.
        self.assertRelative(rate.activation_energy(4 * Ea0), (4332 / 1081) * Ea0)
        self.assertRelative(rate.activation_energy(5 * Ea0), 5 * Ea0)
        self.assertEqual(rate(800.0, -4 * Ea0), 3.0)
        # The symmetric forward/reverse barriers differ by reaction enthalpy.
        for dh in (-0.8 * Ea0, 0.4 * Ea0, 2.0 * Ea0):
            self.assertRelative(rate.activation_energy(dh) - rate.activation_energy(-dh), dh)

    def test_blowers_masel_partial_derivative(self):
        rate = BlowersMaselRate(4.2, 0.4, 1e7, 1e9)
        T, dh, step = 900.0, -2e6, 1e-3
        numerical = (rate(T + step, dh) - rate(T - step, dh)) / (2 * step) / rate(T, dh)
        self.assertRelative(rate.ddT_scaled(T, dh), numerical, 1e-9)

    def test_lindemann_falloff_and_chemically_activated(self):
        low, high = ArrheniusRate(10.0, 0.0), ArrheniusRate(100.0, 0.0)
        normal = LindemannRate(low, high)
        activated = LindemannRate(low, high, chemically_activated=True)
        self.assertEqual(normal(1000.0, 0.0), 0.0)
        self.assertEqual(activated(1000.0, 0.0), 10.0)
        self.assertEqual(normal(1000.0, 10.0), 50.0)
        self.assertEqual(activated(1000.0, 10.0), 5.0)
        self.assertEqual(normal.falloff_coefficient(1000.0, 7.0), 1.0)
        self.assertRelative(normal(1000.0, 1e-10), 1e-9, 2e-11)
        self.assertRelative(normal(1000.0, 1e12), 100.0, 2e-11)

    def test_falloff_negative_A_and_signs(self):
        negative = ArrheniusRate(-10.0, 0.0, allow_negative_A=True)
        positive = ArrheniusRate(10.0, 0.0)
        with self.assertRaises(ValueError):
            LindemannRate(negative, positive, allow_negative_A=True)
        rate = LindemannRate(negative, negative, allow_negative_A=True)
        self.assertEqual(rate(500.0, 1.0), -5.0)

    def test_troe_known_center_and_omitted_temperature_terms(self):
        low, high = ArrheniusRate(1.0, 0.0), ArrheniusRate(1.0, 0.0)
        T, Fcent = 800.0, 0.5
        # Only the a*exp(-T/T1) term remains and equals 1/2.
        rate = TroeRate(low, high, (1.0, 0.0, T / math.log(2.0)))
        log_Fcent = math.log10(Fcent)
        center_Pr = 10.0 ** (0.4 + 0.67 * log_Fcent)
        self.assertRelative(rate.falloff_function(T, center_Pr), Fcent)
        four = TroeRate(low, high, (1.0, 0.0, T / math.log(2.0), 0.0))
        self.assertEqual(rate.falloff_function(T, 2.0), four.falloff_function(T, 2.0))
        tiny = TroeRate(low, high, (1.0, -1e-310, T / math.log(2.0)))
        self.assertEqual(rate.falloff_function(T, 2.0), tiny.falloff_function(T, 2.0))
        self.assertEqual(rate.falloff_function(T, 0.0), rate.falloff_function(T, SMALL_NUMBER))

    def test_troe_t2_term(self):
        low, high = ArrheniusRate(1.0, 0.0), ArrheniusRate(1.0, 0.0)
        T = 700.0
        # T3 and T1 vanish; T2 alone produces Fcent=1/2.
        rate = TroeRate(low, high, (0.3, 0.0, 0.0, T * math.log(2.0)))
        center_Pr = 10.0 ** (0.4 + 0.67 * math.log10(0.5))
        self.assertRelative(rate.falloff_function(T, center_Pr), 0.5)

    def test_sri_branches_and_extra_parameters(self):
        low, high = ArrheniusRate(1.0, 0.0), ArrheniusRate(1.0, 0.0)
        base = SriRate(low, high, (2.0, 300.0, 0.0))
        self.assertRelative(base.falloff_function(300.0, 1.0), 2.0 / math.e)
        self.assertRelative(base.falloff_function(300.0, 10.0), math.sqrt(2.0 / math.e))
        extended = SriRate(low, high, (2.0, 300.0, 600.0, 3.0, 0.5))
        expected = (2.0 / math.e + math.exp(-0.5)) * 3.0 * math.sqrt(300.0)
        self.assertRelative(extended.falloff_function(300.0, 1.0), expected)
        zero_d = SriRate(low, high, (1.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(zero_d.falloff_function(300.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            SriRate(low, high, (1.0, 0.0, -1.0))
        with self.assertRaises(ValueError):
            SriRate(low, high, (1.0, 0.0, 1.0, -1.0, 0.0))

    def test_tsang_center_linear_and_constant(self):
        low, high = ArrheniusRate(1.0, 0.0), ArrheniusRate(1.0, 0.0)
        constant = TsangRate(low, high, (1.0,))
        for Pr in (0.0, 1e-8, 1.0, 1e8):
            self.assertEqual(constant.falloff_function(500.0, Pr), 1.0)
        linear = TsangRate(low, high, (0.2, 0.001))
        Fcent = 0.7
        center_Pr = 10.0 ** (0.4 + 0.67 * math.log10(Fcent))
        self.assertRelative(linear.falloff_function(500.0, center_Pr), Fcent)

    def test_plog_groups_interpolation_and_clamping(self):
        rate = PlogRate([
            (1e5, ArrheniusRate(2.0, 0.0)),
            (1e7, ArrheniusRate(900.0, 0.0)),
            (1e5, ArrheniusRate(8.0, 0.0)),
            (1e5, ArrheniusRate(-1.0, 0.0, allow_negative_A=True)),
        ])
        rate.validate()
        self.assertRelative(rate(500.0, 1e4), 9.0)
        self.assertRelative(rate(500.0, 1e5), 9.0)
        self.assertRelative(rate(500.0, 1e6), 90.0)
        self.assertRelative(rate(500.0, 1e7), 900.0)
        self.assertRelative(rate(500.0, 1e8), 900.0)
        only = PlogRate([(1e5, ArrheniusRate(4.0, 0.0))])
        self.assertRelative(only(400.0, 1e10), 4.0)

    def test_plog_floor_and_invalid_negative_groups(self):
        floor = PlogRate([(1e5, ArrheniusRate(0.0, 0.0))])
        self.assertRelative(floor(1000.0, 1e5), SMALL_NUMBER)
        with self.assertRaises(ValueError):
            floor.validate()
        bad = PlogRate([(1e5, ArrheniusRate(-1.0, 0.0, allow_negative_A=True))])
        with self.assertRaises(ValueError):
            bad(1000.0, 1e5)
        with self.assertRaises(ValueError):
            bad.validate()

    def test_chebyshev_coordinates_and_explicit_polynomial(self):
        coeffs = ((2.0, 0.3, 0.1), (0.7, -0.2, 0.4), (0.1, 0.3, -0.1))
        rate = ChebyshevRate(300.0, 1500.0, 1e4, 1e6, coeffs)
        for T, P in ((300.0, 1e4), (1500.0, 1e6), (500.0, 1e5), (800.0, 3e5)):
            Tr, Pr = rate.reduced_coordinates(T, P)
            t_basis, p_basis = (1.0, Tr, 2 * Tr**2 - 1), (1.0, Pr, 2 * Pr**2 - 1)
            expected = sum(coeffs[i][j] * t_basis[i] * p_basis[j]
                           for i in range(3) for j in range(3))
            self.assertRelative(rate.log10_rate(T, P), expected)
            self.assertRelative(rate(T, P), 10.0**expected)
        self.assertRelative(rate.reduced_coordinates(300.0, 1e4)[0], -1.0)
        self.assertRelative(rate.reduced_coordinates(1500.0, 1e6)[0], 1.0)
        self.assertAlmostEqual(rate.reduced_coordinates(500.0, 1e5)[0], 0.0)
        self.assertAlmostEqual(rate.reduced_coordinates(500.0, 1e5)[1], 0.0)

    def test_chebyshev_constant_and_extrapolation(self):
        constant = ChebyshevRate(300.0, 1500.0, 1e4, 1e6, ((3.0,),))
        self.assertEqual(constant(200.0, 1e9), 1000.0)
        linear = ChebyshevRate(300.0, 1500.0, 1e4, 1e6, ((0.0, 1.0),))
        self.assertRelative(linear(1000.0, 1e7), 100.0)

    def test_linear_burke_explicit_and_reference_collider_rule(self):
        reference = PlogRate(((1.0, ArrheniusRate(10.0, 0.0)),
                              (10.0, ArrheniusRate(100.0, 0.0))))
        special = PlogRate(((1.0, ArrheniusRate(4.0, 0.0)),
                            (10.0, ArrheniusRate(40.0, 0.0))))
        mixture = LinearBurkeRate(reference, {
            'A': LinearBurkeCollider(ArrheniusRate(2.0, 0.0), special)
        })
        expected = (special(900.0, 0.625) * 2.0 * 0.25 / 1.25
                    + reference(900.0, 1.25) * 0.75 / 1.25)
        self.assertRelative(mixture(900.0, 1.0, {'A': 0.25, 'B': 0.75}), expected)

    def test_linear_burke_troe_reference_pressure_conversion_and_validation(self):
        reference = TroeRate(ArrheniusRate(10.0, 0.0), ArrheniusRate(100.0, 0.0),
                             (0.5, 1000.0, 2000.0))
        mixture = LinearBurkeRate(reference)
        T, P = 800.0, 10.0 * R * 800.0
        self.assertRelative(mixture(T, P, {'N2': 1.0}), reference(T, 10.0))
        with self.assertRaises(ValueError):
            mixture(T, P, {'N2': -1.0})

    def test_input_validation(self):
        for T in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(T=T), self.assertRaises(ValueError):
                ArrheniusRate(1.0, 0.0)(T)
        with self.assertRaises(ValueError):
            ChebyshevRate(300.0, 1500.0, 1e4, 1e6, ((1.0,), (1.0, 2.0)))
        with self.assertRaises(ValueError):
            ChebyshevRate(300.0, 300.0, 1e4, 1e6, ((1.0,),))
        with self.assertRaises(ValueError):
            PlogRate([])
        with self.assertRaises(ValueError):
            TroeRate(ArrheniusRate(1, 0), ArrheniusRate(1, 0), (0.5, 100.0))

    def test_vibrational_relaxation_models(self):
        # Constant vibrational relaxation
        const_rate = ConstantVibrationalRelaxationRate(A=1.23e4)
        self.assertRelative(const_rate(300.0), 1.23e4)
        self.assertRelative(const_rate(1500.0), 1.23e4)

        # Starikovskiy vibrational relaxation
        star_rate = StarikovskiyVibrationalRelaxationRate(A=1e3, b=0.5, C0=1.0, C13=-10.0,
                                                        Cm=2.0, m=1.0, Cn=0.5, n=2.0)
        T = 1000.0
        expected = 1e3 * (T**0.5) * math.exp(1.0 - 10.0 * (T**(-1/3)) + 2.0 / T + 0.5 / (T**2))
        self.assertRelative(star_rate(T), expected)

        # Castela vibrational relaxation
        cast_rate = CastelaVibrationalRelaxationRate(a=15.0, b=0.05, reference_pressure=101325.0)
        expected_cast = (R * T / 101325.0) * math.exp(18.42 + 15.0 * 0.05 - 15.0 * (T**(-1/3)))
        self.assertRelative(cast_rate(T), expected_cast)

        # Multi-state resolved vibrational relaxation
        multi_rate = MultiStateResolvedVibrationalRelaxationRate(A=500.0, b=0.0, C0=0.0,
                                                               C13=-5.0, C23=2.0)
        expected_multi = 500.0 * math.exp(-5.0 * (T**(-1/3)) + 2.0 * (T**(-2/3)))
        self.assertRelative(multi_rate(T), expected_multi)

    def test_interface_and_sticking_rates(self):
        # Interface Arrhenius with linear coverage dependency
        # k_f = A * T^b * exp(-Ea/RT) * 10^(a*theta) * theta^m * exp(-E_cov*theta / (RT))
        cov_deps = {'PT(S)': (0.5, 1.0, 1000.0 * R)}
        irate = InterfaceArrheniusRate(A=1e5, b=0.0, Ea=5000.0 * R, coverage_dependencies=cov_deps)
        T = 500.0
        theta = 0.4
        base = 1e5 * math.exp(-5000.0 / T)
        cov_factor = (10.0**(0.5 * theta)) * (theta**1.0) * math.exp(-1000.0 * theta / T)
        self.assertRelative(irate(T, {'PT(S)': theta}), base * cov_factor)
        # Without coverages should give base rate
        self.assertRelative(irate(T), base)

        # Sticking Arrhenius rate
        srate = StickingArrheniusRate(A=0.1, b=0.0, Ea=0.0, sticking_species_weight=0.028,
                                     site_density=1.0e-5, surface_order=1.0, motz_wise=True)
        gamma = 0.1
        gamma_mw = gamma / (1.0 - 0.5 * gamma)
        v_th = math.sqrt(R * T / (2.0 * math.pi * 0.028))
        expected_s = gamma_mw * (1.0e-5**(-1.0)) * v_th
        self.assertRelative(srate(T), expected_s)


if __name__ == "__main__":
    unittest.main()
