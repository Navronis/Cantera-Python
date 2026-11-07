import json
from collections import Counter

subsystems = Counter()
status_by_sub = {}
open_transport = []

with open('provenance/canonical_5784_symbol_inventory.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        sub = row.get('subsystem', 'UNKNOWN')
        status = row.get('current_status', 'UNKNOWN')
        subsystems[sub] += 1
        if sub not in status_by_sub:
            status_by_sub[sub] = Counter()
        status_by_sub[sub][status] += 1
        if sub == 'transport' and status not in ('PINNED_NATIVE_VALIDATED', 'NON_RUNTIME_CPP_ONLY_PROVEN'):
            open_transport.append(row)

print('=== Subsystem Totals and Current Status ===')
for sub, cnt in sorted(subsystems.items()):
    print(f'{sub:15s}: {cnt:4d} -> {dict(status_by_sub[sub])}')

print(f'\nOpen Transport count: {len(open_transport)}')
for r in open_transport:
    idx = r.get('canonical_index')
    sym_id = r.get('stable_symbol_id')
    sig = r.get('signature')
    st = r.get('current_status')
    cls = r.get('class')
    print(f'  #{idx} | class={cls} | {sym_id} | sig={sig} | status={st}')
