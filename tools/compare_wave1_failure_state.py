#!/usr/bin/env python3
"""Compare native and pure-Python Wave 1 post-failure states from raw records."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
FIELDS = ("T", "P", "density", "enthalpy_mass", "int_energy_mass", "entropy_mass")


def number(value):
    if isinstance(value, (int, float)):
        return float(value)
    if value == "NaN":
        return math.nan
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    raise ValueError(f"Not a numeric state field: {value!r}")


def equal(a, b):
    x, y = number(a), number(b)
    if math.isnan(x) and math.isnan(y):
        return True
    if x == y:
        return True
    return math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 1e-8 + 1e-8*abs(x)


def load(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = {r["case_id"]: r for r in rows}
    if len(mapping) != len(rows):
        raise SystemExit(f"Duplicate case IDs in {path}")
    return mapping


def main():
    native = load(AUDIT / "raw_native_wave1_failure_state.jsonl")
    python = load(AUDIT / "raw_python_wave1_failure_state.jsonl")
    if native.keys() != python.keys():
        raise SystemExit("Native and Python Wave 1 failure-state case IDs differ")
    cases = []
    for cid in sorted(native):
        n, p = native[cid], python[cid]
        item = {"case_id": cid, "solver": n["solver"], "native_status": n["status"], "python_status": p["status"]}
        if n["status"] != "FAILURE" or p["status"] != "FAILURE":
            item["classification"] = "NOT_OBSERVABLE"
            item["reason"] = "Both engines did not fail in the same case; requested post-failure state comparison is not applicable."
        else:
            unavailable = [key for key in FIELDS if isinstance(n["after"].get(key), dict) or isinstance(p["after"].get(key), dict)]
            if unavailable:
                item["classification"] = "NOT_OBSERVABLE"
                item["unavailable_fields"] = unavailable
            else:
                differences = [key for key in FIELDS if not equal(n["after"][key], p["after"][key])]
                item["after_state_differences"] = differences
                if not differences:
                    item["classification"] = "POST_FAILURE_STATE_MATCH"
                else:
                    native_mutated = any(not equal(n["after"][key], n["before"][key]) for key in FIELDS)
                    python_restored = all(equal(p["after"][key], p["before"][key]) for key in FIELDS)
                    item["classification"] = (
                        "INTENTIONAL_BEHAVIOR_DIFFERENCE" if native_mutated and python_restored
                        else "POST_FAILURE_STATE_DIFFERENT"
                    )
                item["native_before"] = n["before"]
                item["native_after"] = n["after"]
                item["python_before"] = p["before"]
                item["python_after"] = p["after"]
        cases.append(item)
    counts = {}
    for case in cases:
        key = case["classification"]
        counts[key] = counts.get(key, 0) + 1
    out = {
        "classification_counts": counts,
        "wave1_post_failure_state_parity": counts.get("POST_FAILURE_STATE_DIFFERENT", 0) == 0 and counts.get("INTENTIONAL_BEHAVIOR_DIFFERENCE", 0) == 0 and counts.get("NOT_OBSERVABLE", 0) == 0,
        "cases": cases,
    }
    (AUDIT / "wave1_failure_state_comparison.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, sort_keys=True))
    return 0 if not counts.get("NOT_OBSERVABLE") and not counts.get("POST_FAILURE_STATE_DIFFERENT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
