import unittest

from cantera_python.composition import CompositionError, SpeciesRegistry, parse_comp_string


class CompositionParserTests(unittest.TestCase):
    def test_separator_syntax_and_known_names(self):
        self.assertEqual(parse_comp_string('ice:1   snow:2; fire:3', ['fire', 'ice', 'snow']),
                         {'fire': 3.0, 'ice': 1.0, 'snow': 2.0})

    def test_colon_in_species_name(self):
        # Upstream performs its known-name check before the retry that treats
        # the first colon as part of a name, so this form is supported only
        # when no name filter is supplied.
        self.assertEqual(parse_comp_string('phase:ice: 2'), {'phase:ice': 2.0})

    def test_errors_and_source_duplicate_behavior(self):
        with self.assertRaisesRegex(CompositionError, 'unknown species'):
            parse_comp_string('CO:1', ['O2'])
        with self.assertRaisesRegex(CompositionError, 'Duplicate key'):
            parse_comp_string('O2:1, O2:2')
        self.assertEqual(parse_comp_string('O2:0, O2:2'), {'O2': 2.0})
        with self.assertRaisesRegex(CompositionError, 'non-key:value'):
            parse_comp_string('O2:1 stray')


class SpeciesRegistryTests(unittest.TestCase):
    def test_lookup_alias_and_ordered_amounts(self):
        registry = SpeciesRegistry(['A', 'B'])
        registry.add_alias('A', 'alpha')
        self.assertEqual(registry.species_index('alpha'), 0)
        self.assertEqual(registry.amounts('alpha: 1, B: 3'), (1.0, 3.0))

    def test_optional_case_insensitive_lookup_is_ambiguous_when_needed(self):
        registry = SpeciesRegistry(['CO', 'co'], case_sensitive=False)
        with self.assertRaisesRegex(CompositionError, 'not unique'):
            registry.species_index('cO')
        self.assertIsNone(registry.species_index('N2', raise_on_missing=False))


if __name__ == '__main__':
    unittest.main()
