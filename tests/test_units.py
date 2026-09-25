import unittest

from cantera_python.constants import GasConstant, OneAtm
from cantera_python.units import (
    UnitError, UnitStack, UnitSystem, Units, convert, convert_activation_energy,
    parse_units,
)


class UnitsTests(unittest.TestCase):
    def test_units_value_object_source_semantics(self):
        """Pinned Units.cpp:112-132, 134-200, 226-235, and 237-304."""
        self.assertEqual(Units('').str(), '1')
        self.assertEqual(Units('1.').str(False), '1.0')
        self.assertEqual(Units('kg / m^3').str(), 'kg / m^3')
        self.assertEqual(Units('1 / s').str(), '1 / s')
        self.assertEqual(Units('1.0 kg^0.5').str(), 'kg^0.5')
        self.assertEqual(Units('kg').pow(2).str(), 'kg^2')

        pressure = Units(1.0, (1, -1, -2, 0, 0, 0))
        energy = Units(1.0, (1, 2, -2, 0, 0, 0))
        self.assertEqual(Units(2.0, 1, -1, -2), Units(2.0, (1, -1, -2, 0, 0, 0)))
        self.assertEqual(pressure.pressure_dimension, 1.0)
        self.assertEqual(energy.energy_dimension, 1.0)
        pressure.scale(2.5)
        self.assertEqual(pressure.factor, 2.5)
        with self.assertRaisesRegex(UnitError, 'non-unity'):
            Units('cal', True)

    def test_source_factor_and_dimensional_conversions(self):
        self.assertAlmostEqual(convert(1, 'atm', 'Pa'), OneAtm)
        self.assertAlmostEqual(convert(1, 'cal/mol/K', 'J/kmol/K'), 4.184e3)
        self.assertAlmostEqual(convert(1, 'cm^3', 'm^3'), 1e-6)
        self.assertTrue(parse_units('kg/m/s^2').convertible(parse_units('Pa')))

    def test_prefixes_and_incompatible_dimensions(self):
        self.assertAlmostEqual(convert(3, 'ms', 's'), 0.003)
        with self.assertRaisesRegex(UnitError, 'not convertible'):
            convert(1, 'm', 's')
        with self.assertRaisesRegex(UnitError, 'Unknown'):
            parse_units('widget')

    def test_activation_energy_special_cases(self):
        self.assertAlmostEqual(convert_activation_energy(1, 'K', 'J/kmol'), GasConstant)
        self.assertAlmostEqual(convert_activation_energy(GasConstant, 'J/kmol', 'K'), 1.0)
        self.assertAlmostEqual(convert_activation_energy(1, 'eV', 'J/kmol'),
                               convert_activation_energy(1, 'eV', 'eV') * 96485332.12331002,
                               places=3)

    def test_activation_energy_overload_semantics_and_source_aliases(self):
        """Pinned Units.cpp:708-740 and Units.h:188-264."""
        system = UnitSystem({'activation-energy': 'J/kmol'})
        # Unlike explicit src/dest conversion, the pinned To/From overloads
        # ignore metric scaling on temperature units (Units.cpp:718-719,733-734).
        self.assertAlmostEqual(system.convertActivationEnergyTo(GasConstant, 'mK'), 1.0)
        self.assertAlmostEqual(system.convertActivationEnergyFrom(1.0, 'mK'), GasConstant)
        self.assertAlmostEqual(
            system.convertActivationEnergy(1.0, 'mK', 'J/kmol'),
            GasConstant * 1.0e-3,
        )
        system.setDefaults({'length': 'cm'})
        self.assertEqual(system.convertTo(1.0, 'm'), 0.01)
        self.assertEqual(system.convertFrom(1.0, 'm'), 100.0)
        self.assertEqual(
            system.getDelta(UnitSystem()),
            {'length': 'cm', 'activation-energy': 'J/kmol'},
        )

    def test_unit_system_defaults_objects_and_decoded_vectors(self):
        system = UnitSystem({'length': 'cm', 'quantity': 'mol', 'energy': 'cal'})
        self.assertEqual(system.defaults['length'], 'cm')
        self.assertEqual(system.defaults['activation-energy'], 'cal / mol')
        self.assertAlmostEqual(system.convert_to(1.0, 'm'), 0.01)
        self.assertAlmostEqual(system.convert_from(1.0, parse_units('m')), 100.0)
        self.assertEqual(system.convert([1.0, '2 m'], parse_units('cm')), [1.0, 200.0])
        self.assertAlmostEqual(
            system.convert_activation_energy_to(1.0, 'J/kmol'), 4184.0)

    def test_unit_system_input_failures_include_vector_location(self):
        system = UnitSystem({'length': 'cm'})
        with self.assertRaisesRegex(UnitError, 'space-separated'):
            system.convert('2cm', 'm')
        with self.assertRaisesRegex(UnitError, 'index 1'):
            system.convert([1.0, True], 'm')
        with self.assertRaisesRegex(UnitError, 'unknown dimension'):
            system.set_defaults({'widget': 'm'})

    def test_undefined_rate_coefficient_unit_path(self):
        undefined = Units(0.0)
        base = UnitSystem()
        self.assertEqual(base.convert_rate_coeff(3.0, undefined), 3.0)
        self.assertEqual(base.convert_rate_coeff('4 m^3/kmol/s', undefined), 4.0)

        nondefault = UnitSystem({'length': 'cm', 'quantity': 'mol'})
        with self.assertRaisesRegex(UnitError, 'undefined units'):
            nondefault.convert_rate_coeff(3.0, undefined)
        with self.assertRaisesRegex(UnitError, 'undefined units'):
            base.convert_rate_coeff('4 cm^3/mol/s', undefined)

    def test_unit_stack_empty_and_standard_unit_semantics(self):
        """Pinned Units.cpp:326-363 and Units.h:106-120."""
        import math
        empty = UnitStack()
        self.assertEqual(empty.size(), 0)
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.standard_units(), Units(0.0))
        self.assertEqual(empty.product(), Units(0.0))
        self.assertTrue(math.isnan(empty.standard_exponent()))
        with self.assertRaisesRegex(UnitError, 'Standard unit is not defined'):
            empty.join(1.0)

        concentration = parse_units('kmol/m^3')
        empty.set_standard_units(concentration)
        self.assertEqual(empty.standardUnits(), concentration)
        self.assertEqual(empty.standardExponent(), 0.0)
        self.assertEqual(empty.size(), 1)

    def test_unit_stack_set_standard_join_and_redefinition_error(self):
        """Pinned Units.cpp:334-363."""
        stack = UnitStack(parse_units('kmol/m^3'))
        stack.setStandardUnits(parse_units('mol/cm^3'))
        self.assertEqual(stack.standard_units(), parse_units('mol/cm^3'))
        stack.join(0.25)
        stack.join(-0.75)
        self.assertEqual(stack.standard_exponent(), -0.5)
        with self.assertRaisesRegex(UnitError, 'already defined'):
            stack.set_standard_units(parse_units('kmol/m^3'))

    def test_unit_stack_update_fractional_exponents_and_product(self):
        """Pinned Units.cpp:365-389."""
        concentration = parse_units('kmol/m^3')
        seconds = parse_units('s')
        stack = UnitStack([(concentration, 1.0), (seconds, -1.0)])
        stack.update(concentration, -2.5)
        stack.update(parse_units('K'), 0.5)
        self.assertEqual(stack.size(), 3)
        self.assertEqual(stack.standard_exponent(), -1.5)
        expected = (concentration.power(-1.5) * seconds.power(-1.0)
                    * parse_units('K').power(0.5))
        self.assertEqual(stack.product(), expected)
        self.assertEqual(stack.product().dimension('quantity'), -1.5)

    def test_unit_stack_product_reinfers_powered_pseudo_dimensions(self):
        """Pinned Units::pow reconstructs pressure/energy pseudo-dimensions."""
        pressure_temperature = parse_units('Pa*K')
        self.assertEqual(pressure_temperature.pressure_dimension, 1.0)
        product = UnitStack([(pressure_temperature, 2.0)]).product()
        self.assertEqual(product.pressure_dimension, 0.0)
        self.assertEqual(product.energy_dimension, 0.0)

    def test_unit_stack_rate_order_dimension_rule(self):
        """Pinned Arrhenius.h:126-132 and Reaction.cpp:576-610."""
        concentration = parse_units('kmol/m^3')
        stack = UnitStack(concentration)
        stack.join(1.0)
        stack.update(parse_units('s'), -1.0)
        stack.update(concentration, -1.75)
        self.assertAlmostEqual(
            1.0 - stack.product().dimension('quantity'), 1.75)
        stack.join(-1.0)
        self.assertAlmostEqual(
            1.0 - stack.product().dimension('quantity'), 2.75)


if __name__ == '__main__':
    unittest.main()
