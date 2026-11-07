import json
from pathlib import Path
import unittest


class ModelRegistryMatrixTests(unittest.TestCase):
    def test_registry_records_source_models_and_unported_entries(self):
        path = Path(__file__).parents[1] / 'provenance' / 'model_registry_matrix.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rates = {row['factory_key']: row for row in rows if row['kind'] == 'reaction-rate-model'}
        thermo = {row['factory_key']: row for row in rows if row['kind'] == 'species-thermo-model'}
        self.assertEqual(rates['Arrhenius']['python_symbol'], 'ArrheniusRate')
        self.assertEqual(rates['linear-Burke']['status'], 'partial')
        self.assertEqual(rates['linear-Burke']['python_symbol'], 'LinearBurkeRate')
        self.assertEqual(thermo['NASA7']['python_symbol'], 'NasaPoly2')
        self.assertNotIn('unported', {row['status'] for row in rates.values()})
        self.assertNotIn('unported', {row['status'] for row in thermo.values()})


if __name__ == '__main__':
    unittest.main()
