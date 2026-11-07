#!/usr/bin/env python3
"""Capture Wave 1 thermodynamic state immediately before/after setter failure.

Run once with ``--engine native`` in the pinned native Cantera environment and
once with ``--engine python`` in the pure-Python project environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "audit"
FIELDS = ("T", "P", "density", "enthalpy_mass", "int_energy_mass", "entropy_mass")


def safe(value):
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def state(phase):
    values = {}
    for name in FIELDS:
        try:
            value = getattr(phase, name)
            if callable(value):
                value = value()
            values[name] = safe(value)
        except Exception as exc:
            values[name] = {"unavailable": f"{type(exc).__name__}: {exc}"}
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("native", "python"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.engine == "native":
        import cantera as ct
    else:
        import cantera_python as ct

    corpus = [json.loads(line) for line in (AUDIT / "corpus_wave1_failure_state.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for case in corpus:
        phase = ct.Solution("gri30.yaml")
        phase.TPX = case["initial_T"], case["initial_P"], case["initial_X"]
        before = state(phase)
        status, error = "SUCCESS", None
        try:
            solver = case["solver"]
            if solver == "TPX_INVALID":
                phase.TPX = case["target"], case["constraint"], case["initial_X"]
            else:
                setattr(phase, solver, (case["target"], case["constraint"]))
        except Exception as exc:
            status = "FAILURE"
            error = {"type": type(exc).__name__, "message": str(exc)}
        after = state(phase)
        results.append({
            "case_id": case["case_id"],
            "engine": args.engine,
            "solver": case["solver"],
            "failure_intent": case["failure_intent"],
            "status": status,
            "error": error,
            "before": before,
            "after": after,
        })
    output = args.output or AUDIT / f"raw_{args.engine}_wave1_failure_state.jsonl"
    output.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in results), encoding="utf-8")
    print(f"Wrote {len(results)} {args.engine} before/after records to {output}")


if __name__ == "__main__":
    main()
