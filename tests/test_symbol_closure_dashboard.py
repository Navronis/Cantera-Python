import json
from pathlib import Path
import unittest

from tools.build_symbol_closure_dashboard import (
    CANONICAL_TOTAL, FINAL_STATUSES, TEMPORARY_STATUSES, build_dashboard,
)


class SymbolClosureDashboardTests(unittest.TestCase):
    def test_dashboard_enforces_canonical_denominator_and_gate(self):
        dashboard = build_dashboard()
        self.assertEqual(dashboard['TOTAL'], CANONICAL_TOTAL)
        accounted = sum(dashboard[name] for name in
                        FINAL_STATUSES | TEMPORARY_STATUSES)
        self.assertEqual(accounted, CANONICAL_TOTAL)
        self.assertEqual(
            dashboard['fully_dispositioned']
            + dashboard['remaining_without_final_disposition'],
            CANONICAL_TOTAL)
        self.assertFalse(dashboard['FULL_PORT_COMPLETE'])
        self.assertEqual(dashboard['fully_dispositioned'], 3246)
        self.assertEqual(dashboard['remaining_without_final_disposition'], 2538)

    def test_checked_in_dashboard_matches_generator(self):
        path = Path(__file__).parents[1] / 'provenance' / 'symbol_closure_dashboard.json'
        if not path.exists():
            self.skipTest('dashboard has not been generated yet')
        self.assertEqual(json.loads(path.read_text(encoding='utf-8')),
                         build_dashboard())


if __name__ == '__main__':
    unittest.main()
