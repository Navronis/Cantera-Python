"""Generate PORT_STATUS.json from current machine-readable evidence."""
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    audit = json.loads((ROOT / 'provenance' / 'forensic_audit_summary.json').read_text())
    thermo = json.loads((ROOT / 'provenance' / 'thermo_10050_reconciliation.json').read_text())
    test_run = json.loads((ROOT / 'provenance' / 'run_manifest_2026-09-17.json').read_text())
    coverage = json.loads((ROOT / 'provenance' / 'coverage_current.json').read_text())
    mechanisms = list(csv.DictReader((ROOT / 'provenance' / 'mechanism_compatibility.csv').open()))
    navronis = list(csv.DictReader((ROOT / 'provenance' / 'navronis_readiness.csv').open()))
    loaded = sum(row['python_load_result'] == 'LOADED' for row in mechanisms)
    runtime_summaries = [value['summary'] for name, value in coverage['files'].items()
                         if name.startswith('cantera_python')]
    runtime_covered = sum(row['covered_lines'] for row in runtime_summaries)
    runtime_statements = sum(row['num_statements'] for row in runtime_summaries)
    readiness = sum(float(row['credit']) for row in navronis)
    status = {
        'PORT_COMPLETE': False,
        'pinned_source_commit': audit['source_commit'],
        'status_date': '2026-09-17',
        'runtime_native_cantera_dependency': False,
        'tests_expected': test_run['collected'],
        'tests_passed': test_run['passed'],
        'test_duration_seconds': test_run['duration_seconds'],
        'scoped_runtime_statement_coverage_percent': 100.0 * runtime_covered / runtime_statements,
        'pinned_thermo_states': thermo['states'],
        'pinned_thermo_components': thermo['scalar_and_vector_components_compared'],
        'pinned_thermo_failed_components': thermo['failed_components'],
        'source_symbols': {
            'total': audit['symbols'],
            'equivalent_proven': audit['symbol_status_counts']['EQUIVALENT_PROVEN'],
            'partially_ported': audit['symbol_status_counts']['PARTIALLY_PORTED'],
            'not_ported': audit['symbol_status_counts']['NOT_PORTED'],
        },
        'new_capability': 'Ideal-gas TP and HP chemical equilibrium',
        'supported_equilibrium_modes': ['TP', 'HP'],
        'unsupported_equilibrium_modes': ['UV', 'SV', 'SP', 'multiphase VCS'],
        'bundled_mechanism_parser': {'loaded': loaded, 'total': len(mechanisms)},
        'navronis_weighted_score': readiness,
        'production_ready': False,
        'blocking_domains': [
            'nonideal and real-fluid thermodynamics', 'complete transport',
            'stiff reactor and sensitivity equivalence',
            'surface and multiphase chemistry', 'one-dimensional flames',
            'complete public API and upstream tests',
        ],
    }
    (ROOT / 'PORT_STATUS.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({'generated': 'PORT_STATUS.json', 'port_complete': False,
                      'tests_passed': status['tests_passed']}, indent=2))


if __name__ == '__main__':
    main()
