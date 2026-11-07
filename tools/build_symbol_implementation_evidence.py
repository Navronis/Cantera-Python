"""Validate the curated symbol-implementation evidence file.

This tool deliberately does not infer mappings or promote statuses from class
names, source paths, test presence, or broad subsystem results.  Those
heuristics previously produced false whole-library completion claims.  New
rows must be added by a source review and remain ``PARTIAL`` until a separate
independent closure review supplies direct symbol-linked evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
NATIVE_ROOT = ROOT.parent / "cantera-original"
SOURCE = PROVENANCE / "original_cantera_symbols.jsonl"
DESTINATION = PROVENANCE / "symbol_implementation_evidence.jsonl"
PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def build() -> list[dict]:
    source_rows = _jsonl(SOURCE)
    source = {row["id"]: row for row in source_rows}
    rows = _jsonl(DESTINATION)
    ids = [row["stable_symbol_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate curated implementation-evidence IDs")

    for row in rows:
        symbol_id = row["stable_symbol_id"]
        if symbol_id not in source:
            raise RuntimeError(f"unknown canonical symbol ID: {symbol_id}")
        original = source[symbol_id]
        if row.get("current_status") != "PARTIAL":
            raise RuntimeError(
                f"curated evidence may only assign PARTIAL: {symbol_id}")
        if row.get("source_commit") != PINNED_COMMIT:
            raise RuntimeError(f"wrong pinned commit: {symbol_id}")
        if row.get("source_hash") != original["source_sha256"]:
            raise RuntimeError(f"source hash mismatch: {symbol_id}")
        if not row.get("python_mapping") or not row.get("validation_evidence"):
            raise RuntimeError(f"incomplete curated evidence: {symbol_id}")
        native = NATIVE_ROOT / original["source_file"]
        if hashlib.sha256(native.read_bytes()).hexdigest() != original["source_sha256"]:
            raise RuntimeError(f"pinned source changed: {original['source_file']}")
    return rows


def main() -> None:
    rows = build()
    print(json.dumps({
        "validated_rows": len(rows),
        "status_counts": dict(Counter(row["current_status"] for row in rows)),
        "output": str(DESTINATION),
        "mutated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
