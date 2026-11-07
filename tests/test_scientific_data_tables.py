import json
from pathlib import Path
import unittest


class ScientificDataCatalogTests(unittest.TestCase):
    def test_catalog_is_parseable_and_bounded(self):
        catalog = Path(__file__).parents[1] / 'provenance' / 'scientific_data_tables.jsonl'
        rows = [json.loads(line) for line in catalog.read_text(encoding='utf-8').splitlines()]
        by_kind = {row['kind']: row for row in rows}
        self.assertEqual(by_kind['catalog-metadata']['schema_version'], 1)
        self.assertIn('not represented', by_kind['catalog-metadata']['scope'])
        self.assertEqual(by_kind['physical-constants']['values']['GasConstant'], 8314.46261815324)
        self.assertEqual(by_kind['element-atomic-weights']['scope'].split(':')[0], 'COMMON_ELEMENTS only')
        self.assertIn('Pa', {row['name'] for row in by_kind['unit-definitions']['values']})


if __name__ == '__main__':
    unittest.main()
