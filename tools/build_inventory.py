"""Inventory every tracked file at the downloaded commits, including submodules.

This is a file-integrity and port-status ledger, not proof of equation coverage.
Run from anywhere; the default source is the sibling cantera-original folder.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
NATIVE = {'.c', '.cc', '.cpp', '.cxx', '.h', '.hh', '.hpp', '.hxx', '.f', '.f90', '.f95'}
LANGUAGES = {'.py': 'Python', '.pyx': 'Cython', '.pxd': 'Cython', '.pxi': 'Cython',
             '.m': 'MATLAB', '.c': 'C', '.h': 'C/C++ header', '.f': 'Fortran',
             '.f90': 'Fortran', '.f95': 'Fortran'}
EQUATION = re.compile(r'(?P<display>[@\\]f\[.*?[@\\]f\])|'
                      r'(?P<inline>[@\\]f\$.*?[@\\]f\$)|'
                      r'(?P<environment>[@\\]f\{.*?[@\\]f\})', re.DOTALL)


def git(repo, *args):
    # The preserved clone may have been downloaded by a different Windows
    # sandbox identity. Mark this read-only inspection as safe locally.
    return subprocess.check_output(['git', '-c', 'safe.directory=*',
                                    '-c', 'core.excludesFile=NUL', '-C',
                                    str(repo), *args], stderr=subprocess.PIPE)


def source_mappings(provenance):
    mapping = defaultdict(list)
    for path in sorted(provenance.glob('*.json')):
        if path.name in {'download.json', 'source_inventory.json', 'validation.json',
                         'inventory_summary.json', 'files.json'}:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        # Each module records translations under this common schema.
        for translation in data.get('translations', []):
            for upstream_path in translation.get('upstream_paths', []):
                mapping[upstream_path].append({
                    'provenance': path.name,
                    'python_symbols': translation.get('python_symbols', []),
                    'translated_methods': translation.get('translated_methods', [])})
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=PROJECT.parent / 'cantera-original')
    args = parser.parse_args()
    source = args.source.resolve()
    out = PROJECT / 'provenance'
    out.mkdir(parents=True, exist_ok=True)
    download = json.loads((out / 'download.json').read_text(encoding='utf-8'))
    modules = [('', download['commit'])]
    modules += [(m['path'], m['expected_commit']) for m in download['submodules']]
    mapped = source_mappings(out)
    all_files, code, excerpts, checks = [], [], [], []
    missing_files = []
    subsystem_counts = defaultdict(lambda: {'files': 0, 'physical_lines': 0})
    for relative, expected in modules:
        repo = source / relative
        commit = git(repo, 'rev-parse', 'HEAD').decode().strip()
        if commit != expected:
            raise RuntimeError(f'{relative or "main"}: wrong commit {commit}')
        changes = git(repo, 'diff', '--name-only', '--ignore-submodules=all', 'HEAD', '--').decode().splitlines()
        # A few dependency test fixtures can be rewritten by upstream test
        # runs (for example SUNDIALS answer links). Preserve and report those
        # changes while still inventorying the pinned committed tree.
        git(repo, 'fsck', '--connectivity-only', '--no-dangling')
        file_count = 0
        for entry in git(repo, 'ls-tree', '-r', '-z', '--full-tree', 'HEAD').split(b'\x00'):
            if not entry:
                continue
            metadata, filename = entry.split(b'\t', 1)
            mode, kind, object_id = metadata.decode().split()
            filename = filename.decode('utf-8')
            full_name = f'{relative}/{filename}' if relative else filename
            record = {'path': full_name, 'repository': relative or '.',
                      'git_mode': mode, 'git_object': object_id, 'git_kind': kind}
            if kind != 'blob':
                all_files.append(record)
                continue
            path = repo / filename
            if not path.is_file():
                missing_files.append(full_name)
                record['working_tree_missing'] = True
                all_files.append(record)
                continue
            raw = path.read_bytes()
            record.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            all_files.append(record)
            file_count += 1
            suffix = path.suffix.lower()
            if not relative and (suffix in NATIVE or suffix in LANGUAGES):
                content = raw.decode('utf-8', errors='replace')
                status = ('already_python_preserved_not_a_native_replacement' if suffix == '.py'
                          else 'selected_methods_ported_file_still_incomplete' if full_name in mapped
                          else 'not_ported')
                code.append({'path': full_name,
                             'language': LANGUAGES.get(suffix, 'C++'),
                             'physical_lines': len(content.splitlines()),
                             'status': status, 'mappings': mapped.get(full_name, []),
                             'sha256': record['sha256'], 'git_object': object_id})
                if filename.startswith(('src/', 'include/cantera/')) and suffix in NATIVE:
                    parts = filename.split('/')
                    subsystem = parts[1] if parts[0] == 'src' else parts[2]
                    subsystem_counts[subsystem]['files'] += 1
                    subsystem_counts[subsystem]['physical_lines'] += len(content.splitlines())
            if not relative and suffix in NATIVE | {'.md', '.rst', '.pyx', '.py', '.pxd', '.pxi'}:
                content = raw.decode('utf-8', errors='replace')
                for match in EQUATION.finditer(content):
                    excerpt = match.group()
                    excerpts.append({'path': full_name,
                                     'line': content.count('\n', 0, match.start()) + 1,
                                     'kind': match.lastgroup, 'text': excerpt,
                                     'sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
                                     'translation_verified': False})
        checks.append({'repository': relative or '.', 'commit': commit,
                       'tracked_blob_files': file_count, 'tracked_tree_clean': not bool(changes),
                       'working_tree_changes': changes,
                       'working_tree_missing': [p for p in missing_files if p.startswith((relative + '/') if relative else '')],
                       'git_connectivity_checked': True})
    all_files.sort(key=lambda r: r['path'])
    code.sort(key=lambda r: r['path'])
    (out / 'files.json').write_text(json.dumps(all_files, indent=2) + '\n', encoding='utf-8')
    (out / 'source_inventory.json').write_text(json.dumps(code, indent=2) + '\n', encoding='utf-8')
    # File-level master matrix is the auditable baseline. Mapped files are
    # deliberately marked PARTIAL until function-level review and tests exist;
    # every other source file remains explicitly NOT_AUDITED.
    matrix_path = out / 'master_port_matrix.jsonl'
    with matrix_path.open('w', encoding='utf-8') as handle:
        for item in code:
            mapped_rows = item.get('mappings', [])
            status = 'PARTIAL' if mapped_rows else 'NOT_AUDITED'
            destination = mapped_rows[0].get('python_symbols', []) if mapped_rows else []
            row = {
                'id': 'file:' + item['path'], 'upstream_file': item['path'],
                'source_commit': download['commit'], 'source_sha256': item['sha256'],
                'source_line_range': [1, item['physical_lines']],
                'subsystem': item['path'].split('/')[1] if item['path'].startswith('src/') else 'interfaces',
                'category': 'scientific_source' if item['path'].split('.')[-1].lower() in {'c','cc','cpp','h','hpp','f','f90'} else 'runtime_support',
                'equation_or_algorithm_description': 'File-level audit placeholder; inspect source methods individually.',
                'python_destination_file': None, 'python_symbols': destination,
                'implementation_status': status, 'test_file': [],
                'validation_status': 'UNVALIDATED',
                'notes': 'Generated baseline; function-level audit required.'}
            handle.write(json.dumps(row) + '\n')
    # Account for upstream test files even before each test is translated.
    test_matrix = out / 'upstream_test_matrix.jsonl'
    with test_matrix.open('w', encoding='utf-8') as handle:
        for path in sorted(source.rglob('*')):
            if path.is_file() and ('test' in path.parts or path.name.lower().startswith('test')):
                rel = path.relative_to(source).as_posix()
                handle.write(json.dumps({'upstream_test': rel,
                    'functionality': 'Unclassified upstream test; review required.',
                    'python_counterpart': [], 'status': 'NOT_AUDITED',
                    'excluded_reason': None}) + '\n')
    with (out / 'documented_equations.jsonl').open('w', encoding='utf-8') as handle:
        for excerpt in excerpts:
            handle.write(json.dumps(excerpt) + '\n')
    counts = Counter(item['status'] for item in code)
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'upstream_commit': download['commit'], 'port_complete': False,
               'source_preservation_verified': True, 'repositories': checks,
               'tracked_files_including_dependencies': sum(r['tracked_blob_files'] for r in checks),
               'main_repository_code_files': len(code), 'code_status_counts': dict(counts),
               'core_native_subsystems': dict(sorted(subsystem_counts.items())),
               'documented_equation_excerpts': len(excerpts),
               'limitations': [
                   'Physical lines include comments and whitespace; they are not equation counts.',
                   'Equation excerpts index Doxygen f markup only; executable formulas and other math markup are not exhaustively identified.',
                   'File mappings mark selected ported methods, never whole-subsystem equivalence.',
                   'All excerpts remain unverified at equation-index level; module tests do not certify every equation.',
                   'Original Python files remain at their upstream paths; many depend on the native Cantera extension.',
                   'External dependency source is preserved; its native algorithms have not been ported.',
                   'Working-file SHA256 hashes may include checkout newline conversion; git_object identifies committed content.']}
    (out / 'inventory_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Python port status', '', '**The full conversion is not complete.**', '',
             f'Pinned source: `{download["commit"]}`.', '',
             f'All {len(checks)} downloaded repositories matched their pinned commits and passed Git connectivity checks.',
             ('Working-tree changes were detected in: ' + ', '.join(c['repository'] for c in checks if not c['tracked_tree_clean']) + '.') if any(not c['tracked_tree_clean'] for c in checks) else 'All working trees are clean.',
             f'{summary["tracked_files_including_dependencies"]:,} tracked files, including dependency sources, are inventoried.', '',
             '## Native scientific source still requiring complete porting', '',
             '| Subsystem | Source/header files | Physical lines | Status |',
             '|---|---:|---:|---|']
    for name, counts in sorted(subsystem_counts.items()):
        status = 'Selected equations ported; incomplete' if name in {'base', 'thermo', 'kinetics', 'numerics'} else 'Not ported'
        lines.append(f'| {name} | {counts["files"]} | {counts["physical_lines"]:,} | {status} |')
    lines.extend(['', '## What the inventory establishes', '',
                  '`provenance/files.json` records every tracked file and Git object plus a local SHA256 hash.',
                  '`provenance/source_inventory.json` records per-file port status for code in the main repository.',
                  '`provenance/documented_equations.jsonl` preserves Doxygen equation excerpts with their original locations.',
                  'These records establish source preservation and explicit missing work. They do not prove equation completeness or numerical equivalence.', '',
                  'The original source is in the sibling `cantera-original` folder. Ported runtime code is in `cantera_python`.', '',
                  '## Required work before full equivalence can be claimed', '',
                  '1. Finish the source/method mapping, including inline functions, constants, data tables, and all factory registrations.',
                  '2. Port the remaining thermo models, phase state machinery, kinetics mechanisms, units, parsing, and APIs.',
                  '3. Port equilibrium, transport, non-ideal fluids/water, reactor and flame solvers, and their numerical dependencies.',
                  '4. Port and run the relevant upstream tests, then compare against a build of this exact commit.',
                  '5. Test conservation, units, derivatives, phase/temperature boundaries, stiff problems, sensitivities, convergence, and failure paths.',
                  '6. Audit every inventory row and equation before changing the completion flag.', ''])
    (PROJECT / 'PORT_STATUS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'tracked_files': summary['tracked_files_including_dependencies'],
                      'repositories': len(checks), 'code_status_counts': summary['code_status_counts'],
                      'equation_excerpts': len(excerpts), 'port_complete': False}, indent=2))


if __name__ == '__main__':
    main()
