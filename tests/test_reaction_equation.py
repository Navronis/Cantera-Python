import unittest

from cantera_python.reaction_equation import (ReactionEquationError, are_duplicate_reactions,
    duplicate_stoichiometry_ratio, parse_reaction_equation)


class ReactionEquationTests(unittest.TestCase):
    def test_parse_coefficients_arrows_and_accumulation(self):
        parsed = parse_reaction_equation('2 H2 + O2 <=> 2 H2O', ['H2', 'O2', 'H2O'])
        self.assertTrue(parsed.reversible)
        self.assertEqual(parsed.reactants, {'H2': 2.0, 'O2': 1.0})
        self.assertEqual(parsed.products, {'H2O': 2.0})
        self.assertEqual(parse_reaction_equation('A + A => B').reactants, {'A': 2.0})

    def test_falloff_third_body_whitespace(self):
        parsed = parse_reaction_equation('H2O2 (+ M) <=> OH + OH (+ M)', ['H2O2', 'OH'])
        self.assertTrue(parsed.reversible)
        self.assertEqual(parsed.reactants, {'H2O2': 1.0, '(+M)': 1.0})
        self.assertEqual(parsed.products, {'OH': 2.0, '(+M)': 1.0})

    def test_parse_errors(self):
        with self.assertRaisesRegex(ReactionEquationError, 'one arrow'):
            parse_reaction_equation('A + B')
        with self.assertRaisesRegex(ReactionEquationError, 'Unknown'):
            parse_reaction_equation('A => C', ['A', 'B'])
        with self.assertRaisesRegex(ReactionEquationError, 'coefficient'):
            parse_reaction_equation('two A => B')

    def test_source_duplicate_direction_and_multiples(self):
        forward = parse_reaction_equation('H2 + O2 <=> H2O2')
        multiple = parse_reaction_equation('2 H2 + 2 O2 <=> 2 H2O2')
        reverse = parse_reaction_equation('H2O2 <=> H2 + O2')
        irreversible_reverse = parse_reaction_equation('H2O2 => H2 + O2')
        self.assertAlmostEqual(duplicate_stoichiometry_ratio(forward.net_stoichiometry,
                                                             multiple.net_stoichiometry), 2.0)
        self.assertAlmostEqual(duplicate_stoichiometry_ratio(forward.net_stoichiometry,
                                                             reverse.net_stoichiometry), -1.0)
        self.assertTrue(are_duplicate_reactions(forward, reverse))
        self.assertFalse(are_duplicate_reactions(
            parse_reaction_equation('H2 + O2 => H2O2'), irreversible_reverse))
