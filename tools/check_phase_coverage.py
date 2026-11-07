import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
from cantera_python.phases import Phase

closure = json.load(open('audit/thermo_active_closure_symbol_set.json', 'r', encoding='utf-8'))
missing = []
present = []

p = Phase()
phase_attrs = set(dir(p))

for s in closure['symbols']:
    meth = s['function_or_method']
    if meth in phase_attrs:
        present.append(s)
    else:
        missing.append(s)

print(f"Total Family 1 symbols: {len(closure['symbols'])}")
print(f"Present on Phase: {len(present)}")
print(f"Missing on Phase: {len(missing)}")
print("\nSample missing methods:")
for s in missing[:20]:
    print(f"  {s['canonical_index']}: {s['stable_symbol_id']}")
