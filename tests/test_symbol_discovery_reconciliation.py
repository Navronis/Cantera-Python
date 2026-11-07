import json
import unittest
from pathlib import Path

from tools.forensic_audit import match_method_declaration
from tools.reconcile_symbol_discovery import OUTPUT, build


class SymbolDiscoveryReconciliationTests(unittest.TestCase):
    def test_statement_guard(self):
        self.assertIsNone(match_method_declaration(
            "    return convert(value, Units(src), Units(dest));"))
        self.assertIsNone(match_method_declaration(
            "    return newSumFunction(a, b);"))
        self.assertIsNone(match_method_declaration(
            "        : m_rxn(rxn), m_ic0(ic0), m_ic1(ic1) {}"))
        self.assertIsNone(match_method_declaration(
            "                                              : clip(minTemp(), 0.0, 100.0);"))
        self.assertIsNotNone(match_method_declaration(
            "double convert(double value, const Units& src);"))

    def test_checked_in_reconciliation_is_current(self):
        actual = build()
        expected = json.loads(OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(actual, expected)
        self.assertEqual(actual["legacy_denominator_preserved"], 5784)
        # The 71-row body review exposed 27 examples. Applying the same rule to
        # the full frozen inventory reveals 754 executable return/throw rows
        # plus five initializer-list / continued-expression rows.
        self.assertEqual(actual["rejected_false_positive_count"], 759)
        self.assertEqual(actual["corrected_lexical_candidate_count"], 5025)
        self.assertTrue(all(row["canonical_disposition"] == "NON_RUNTIME_CPP_ONLY_PROVEN"
                            for row in actual["rejected_rows"]))
        self.assertTrue(all(len(row["non_runtime_proof"]) == 4
                            for row in actual["rejected_rows"]))


if __name__ == "__main__":
    unittest.main()
