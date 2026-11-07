import json
from collections import Counter, defaultdict

# 1. Inspect canonical inventory for kinetics
kinetics_rows = []
with open('provenance/canonical_5784_symbol_inventory.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row.get('subsystem') == 'kinetics':
            kinetics_rows.append(row)

print(f"Total canonical kinetics symbols: {len(kinetics_rows)}")
kin_statuses = Counter(r['current_status'] for r in kinetics_rows)
print(f"Kinetics current statuses: {dict(kin_statuses)}")

# Verify each has exactly one disposition
invalid_status = [r for r in kinetics_rows if r['current_status'] not in ('PINNED_NATIVE_VALIDATED', 'NON_RUNTIME_CPP_ONLY_PROVEN')]
print(f"Unresolved kinetics rows: {len(invalid_status)}")

# 2. Check implementation evidence
with open('provenance/symbol_implementation_evidence.jsonl', 'r', encoding='utf-8') as f:
    evidence_rows = [json.loads(line) for line in f]

evidence_by_id = defaultdict(list)
for ev in evidence_rows:
    evidence_by_id[ev.get('canonical_id')].append(ev)

validated_kin = [r for r in kinetics_rows if r['current_status'] == 'PINNED_NATIVE_VALIDATED']
missing_ev = [r for r in validated_kin if r['canonical_index'] not in evidence_by_id]
print(f"Validated kinetics symbols without implementation evidence: {len(missing_ev)}")

# 3. Check Family manifests and overlaps
manifest_files = [
    ('Family 1', 'audit/kinetics_family1_frozen_manifest.json'),
    ('Family 2', 'audit/kinetics_family2_frozen_manifest.json'),
    ('Family 3', 'audit/kinetics_family3_frozen_manifest.json'),
    ('Family 4', 'audit/kinetics_family4_frozen_manifest.json'),
    ('Family 5', 'audit/kinetics_family5_frozen_manifest.json'),
]

family_symbols = {}
for fam_name, path in manifest_files:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Some manifests store symbol ids or closed counts
            symbols_closed = data.get('active_closure_symbols_closed', 0)
            family = data.get('active_closure_family')
            print(f"{fam_name} ({family}): closed {symbols_closed}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Check symbol set files for overlaps
symbol_set_files = [
    ('Family 1', 'audit/kinetics_family1_closure_symbol_set.json'),
    ('Family 2', 'audit/kinetics_family2_closure_symbol_set.json'),
    ('Family 3', 'audit/kinetics_family3_closure_symbol_set.json'),
    ('Family 4', 'audit/kinetics_family4_closure_symbol_set.json'),
    ('Family 5', 'audit/kinetics_family5_closure_symbol_set.json'),
]

family_sym_ids = {}
for fam_name, path in symbol_set_files:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # data can be list or dict
            if isinstance(data, dict):
                syms = data.get('symbols', data.get('closure_symbols', []))
            elif isinstance(data, list):
                syms = data
            ids = set()
            for s in syms:
                if isinstance(s, dict):
                    ids.add(s.get('canonical_index') or s.get('canonical_id'))
                else:
                    ids.add(s)
            family_sym_ids[fam_name] = ids
            print(f"{fam_name} symbol set count: {len(ids)}")
    except Exception as e:
        print(f"Error reading {path}: {e}")

# Check pairwise overlap
fam_names = list(family_sym_ids.keys())
for i in range(len(fam_names)):
    for j in range(i + 1, len(fam_names)):
        overlap = family_sym_ids[fam_names[i]] & family_sym_ids[fam_names[j]]
        if overlap:
            print(f"WARNING: Overlap between {fam_names[i]} and {fam_names[j]}: {overlap}")
        else:
            print(f"Disjoint: {fam_names[i]} and {fam_names[j]} (0 overlap)")

# 4. Check P-log and Chebyshev symbols distinction
plog_cheb = [r for r in kinetics_rows if 'Plog' in (r.get('signature') or '') or 'Chebyshev' in (r.get('signature') or '') or 'Plog' in (r.get('class') or '') or 'Chebyshev' in (r.get('class') or '')]
print(f"\nTotal P-log and Chebyshev symbols: {len(plog_cheb)}")
for r in plog_cheb[:10]:
    print(f"  #{r['canonical_index']} | class={r.get('class')} | {r['signature']} | {r['current_status']}")

# 5. Check Non-runtime classification
non_runtime_kin = [r for r in kinetics_rows if r['current_status'] == 'NON_RUNTIME_CPP_ONLY_PROVEN']
print(f"\nNon-runtime C++ only kinetics symbols: {len(non_runtime_kin)}")
sample_non_rt = [(r['canonical_index'], r.get('class'), r.get('signature')) for r in non_runtime_kin[:5]]
print(f"Sample non-runtime: {sample_non_rt}")

# 6. Check Family 5 temperature derivative max relative error
with open('audit/kinetics_family5_differential_summary.json', 'r', encoding='utf-8') as f:
    fam5_sum = json.load(f)
print(f"\nFamily 5 max global rel error: {fam5_sum.get('max_global_relative_error')}")
for cr in fam5_sum.get('case_reports', []):
    if cr.get('max_rel_err', 0) > 1e-7:
        print(f"  Case with error > 1e-7: {cr.get('case_id')}, category: {cr.get('category')}, max_rel_err: {cr.get('max_rel_err')}")
