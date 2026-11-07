import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
with open(repo_root / "audit" / "flow1d_closure_symbol_set.json", "r") as f:
    syms = json.load(f)

print(f"Total symbols: {len(syms)}")
for sid, d in sorted(syms.items(), key=lambda x: x[1]["canonical_index"]):
    print(f"{d['canonical_index']:4d} | {d['current_status']:<25} | {sid}")
