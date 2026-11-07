import json
from collections import Counter

def main():
    with open("provenance/canonical_5784_symbol_inventory.jsonl", "r", encoding="utf-8") as f:
        symbols = [json.loads(line) for line in f]

    vcs_symbols = [
        s for s in symbols
        if "vcs" in s["stable_symbol_id"].lower()
        or "vcs" in s.get("source_file", "").lower()
        or (s.get("class") and "vcs" in s["class"].lower())
    ]
    print(f"Total VCS symbols: {len(vcs_symbols)}")
    counts = Counter(s.get("current_status") for s in vcs_symbols)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    for s in sorted(vcs_symbols, key=lambda x: x["stable_symbol_id"]):
        cls = s.get("class") or ""
        fn = s.get("function_or_method") or ""
        status = s.get("current_status") or ""
        sid = s.get("stable_symbol_id") or ""
        print(f"{sid:65} | {cls}::{fn} | {status}")

if __name__ == "__main__":
    main()
