"""Build centralized error and ULP distributions from the focused pinned replay."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import numbers
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cantera_python.validation import compare_scalar, summarize


SOURCE = ROOT / "provenance" / "pinned_native_differential_results.jsonl"
ERROR_OUTPUT = ROOT / "audit" / "error_distribution.json"
ULP_OUTPUT = ROOT / "audit" / "ulp_distribution.json"


def _ulp_bucket(value):
    if value is None:
        return "not_applicable"
    if value <= 4:
        return str(value)
    if value <= 16:
        return "5-16"
    if value <= 256:
        return "17-256"
    if value <= 4096:
        return "257-4096"
    return ">4096"


def build():
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    cases = [row for row in records if "summary" not in row]
    scalar_cases = [row for row in cases
                    if isinstance(row.get("python_value"), numbers.Real)
                    and isinstance(row.get("oracle_value"), numbers.Real)]
    aggregate_only = [row for row in cases if row not in scalar_cases]
    metrics = []
    by_suite = defaultdict(list)
    enriched = []
    for row in scalar_cases:
        is_rate = row.get("quantity") in {"kf", "kr", "ropf", "ropr"}
        policy = {
            "atol": 1e-300 if is_rate else 1e-12,
            "rtol": 1e-8 if is_rate else 1e-10,
            "relative_floor": 1e-300,
            "near_zero_floor": 1e-20,
        }
        metric = compare_scalar(row["python_value"], row["oracle_value"], **policy)
        metrics.append(metric)
        by_suite[row["suite"]].append(metric)
        enriched.append({
            "suite": row["suite"], "case": row["case"],
            "quantity": row["quantity"], "policy": policy,
            **metric.as_dict(),
        })

    error = {
        "schema_version": 1,
        "source": "provenance/pinned_native_differential_results.jsonl",
        "scope_limit": (
            "Focused archived-fixture replay only; this does not characterize "
            "unrepresented solvers or subsystems."),
        "scalar_case_count": len(scalar_cases),
        "aggregate_vector_case_count": len(aggregate_only),
        "aggregate_vector_limitation": (
            "The archived reconciliation file stores only a maximum relative error "
            "for vector quantities, not vector elements. Absolute, mixed, logarithmic, "
            "and ULP distributions therefore exclude those aggregate-only records."),
        "aggregate_vector_records": [
            {key: row.get(key) for key in
             ("suite", "case", "quantity", "max_rel_error_group", "note")}
            for row in aggregate_only
        ],
        "overall": summarize(metrics),
        "by_suite": {name: summarize(values) for name, values in sorted(by_suite.items())},
        "cases": enriched,
    }
    ulp = {
        "schema_version": 1,
        "source": "provenance/pinned_native_differential_results.jsonl",
        "applicability": (
            "Binary64 ULP distance for direct scalar algebraic outputs. Iterative "
            "trajectory and ill-conditioned comparisons require mixed tolerances."),
        "histogram": dict(sorted(Counter(
            _ulp_bucket(item.ulp_distance) for item in metrics).items())),
        "by_suite": {
            name: dict(sorted(Counter(_ulp_bucket(item.ulp_distance)
                                     for item in values).items()))
            for name, values in sorted(by_suite.items())
        },
        "max_ulp_distance": max((item.ulp_distance for item in metrics
                                 if item.ulp_distance is not None), default=None),
    }
    return error, ulp


def main():
    error, ulp = build()
    ERROR_OUTPUT.write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
    ULP_OUTPUT.write_text(json.dumps(ulp, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"error_output": str(ERROR_OUTPUT),
                      "ulp_output": str(ULP_OUTPUT),
                      "cases": error["overall"]["count"],
                      "failures": error["overall"]["failed"]}))


if __name__ == "__main__":
    main()
