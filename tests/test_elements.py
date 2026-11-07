import unittest

from cantera_python.elements import COMMON_ELEMENTS, Element, ElementError, ElementRegistry


class ElementTests(unittest.TestCase):
    def test_source_molecular_weight_equation(self):
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'H': 2, 'O': 1}),
                               2 * 1.008 + 15.999)
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'carbon': 1, 'O': 2}),
                               12.011 + 2 * 15.999)

    def test_ordered_elemental_amounts(self):
        registry = ElementRegistry([Element('H', 1), Element('O', 16)])
        self.assertEqual(registry.element_amounts({'O': 1, 'H': 2}), (2.0, 1.0))

    def test_invalid_compositions_are_not_silent(self):
        with self.assertRaisesRegex(ElementError, 'not found'):
            COMMON_ELEMENTS.molecular_weight({'Xx': 1})
        with self.assertRaisesRegex(ElementError, 'nonnegative'):
            COMMON_ELEMENTS.molecular_weight({'H': -1})
        with self.assertRaisesRegex(ElementError, 'Duplicate'):
            ElementRegistry([Element('H', 1), Element('H', 2)])

    def test_extended_elements_and_electron(self):
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'Pt': 1}), 195.084)
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'Ru': 1}), 101.07)
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'niobium': 1}), 92.90637)
        # Test electron and case insensitivity
        e_weight = COMMON_ELEMENTS.get('e').atomic_weight
        self.assertGreater(e_weight, 0.0)
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'E': 1}), e_weight)
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'e': 1}), e_weight)
        self.assertAlmostEqual(COMMON_ELEMENTS.molecular_weight({'D': 2, 'O': 1}), 2 * 2.0141017781 + 15.999)

    def test_ion_and_vacancy_compositions(self):
        # Cation N2+ has deficit of 1 electron: E: -1
        e_weight = COMMON_ELEMENTS.get('E').atomic_weight
        n2_plus_weight = COMMON_ELEMENTS.molecular_weight({'N': 2, 'E': -1})
        self.assertAlmostEqual(n2_plus_weight, 2 * 14.007 - e_weight)
        # Empty vacancy / surface site
        self.assertEqual(COMMON_ELEMENTS.molecular_weight({}), 0.0)


if __name__ == '__main__':
    unittest.main()
