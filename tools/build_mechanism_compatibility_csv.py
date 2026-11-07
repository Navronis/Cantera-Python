"""Build a reproducible 34-file bundled-mechanism parser compatibility table."""
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'cantera-original' / 'data'


def main():
    records = json.loads((ROOT / 'audit' / 'mechanism_parser_audit.json').read_text())
    fields = [
        'file', 'sha256', 'native_load_result', 'python_load_result',
        'phases_expected', 'phases_loaded', 'species_expected', 'species_loaded',
        'reactions_expected', 'reactions_loaded', 'models_encountered',
        'unsupported_models', 'ignored_fields', 'exact_failure',
        'navronis_relevance', 'numerically_validated', 'validation_scope',
    ]
    output = ROOT / 'provenance' / 'mechanism_compatibility.csv'
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            path = DATA / Path(record['file'])
            name = record['file'].replace('\\', '/')
            navronis = ('high' if any(token in name.lower() for token in
                                      ('gri30', 'h2o2', 'ammonia', 'nasa9', 'n-dodecane'))
                        else 'medium' if record['reactions_count'] else 'low')
            writer.writerow({
                'file': name,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'native_load_result': 'NOT_RUN_THIS_RECONCILIATION',
                'python_load_result': 'LOADED' if record['success'] else 'FAILED',
                'phases_expected': 'NOT_MEASURED',
                'phases_loaded': record['phases_count'],
                'species_expected': 'NOT_MEASURED',
                'species_loaded': record['species_count'],
                'reactions_expected': 'NOT_MEASURED',
                'reactions_loaded': record['reactions_count'],
                'models_encountered': ';'.join(record['thermo_models']),
                'unsupported_models': 'NOT_AUDITED',
                'ignored_fields': 'NOT_AUDITED',
                'exact_failure': record['error'] or '',
                'navronis_relevance': navronis,
                'numerically_validated': 'false',
                'validation_scope': ('parser/factory reachability only; numerical validation '
                                     'is tracked separately'),
            })
    print(json.dumps({
        'files': len(records),
        'parser_loaded': sum(bool(r['success']) for r in records),
        'parser_failed': sum(not r['success'] for r in records),
        'output': str(output.relative_to(ROOT)),
    }, indent=2))


if __name__ == '__main__':
    main()
