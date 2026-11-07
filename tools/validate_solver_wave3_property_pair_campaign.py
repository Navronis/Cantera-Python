#!/usr/bin/env python3
"""Validate coverage and integrity of the TPX property-pair stress corpus.

This checks the Python self-consistency campaign only. It does not compare
results with the pinned native implementation or certify numerical parity.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
CASES = AUDIT / "solver_wave3_property_pair_validation.jsonl"
SUMMARY = AUDIT / "solver_wave3_property_pair_campaign_summary.json"
OUTPUT = AUDIT / "solver_wave3_property_pair_campaign_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    assert summary["schema"] == "solver-wave3-property-pair-campaign-v2"
    assert summary["case_file_sha256"] == sha256(CASES)
    assert summary["runtime_tpx_sha256"] == sha256(ROOT / "cantera_python/tpx.py")
    assert summary["campaign_runner_sha256"] == sha256(
        ROOT / "tools/run_solver_wave3_property_pair_campaign.py")
    assert len(rows) == summary["total_records"] == 22_400

    trials_per_pair = int(summary["requested_strategy_trials_per_fluid_pair"])
    pair_counts = Counter((row["fluid"], row["pair"]) for row in rows)
    expected_pairs = {(fluid, pair) for fluid in summary["fluids"]
                      for pair in summary["property_pairs"]}
    assert summary["fluid_pair_combinations"] == len(expected_pairs)
    assert set(pair_counts) == expected_pairs
    assert all(pair_counts[pair] == trials_per_pair for pair in expected_pairs)

    strategy_counts = Counter((row["fluid"], row["pair"], row["guess_strategy"])
                              for row in rows)
    strategies = summary["initial_guess_strategies"]
    assert trials_per_pair % len(strategies) == 0
    trials_per_strategy = trials_per_pair // len(strategies)
    assert all(strategy_counts[(fluid, pair, strategy)] == trials_per_strategy
               for fluid, pair in expected_pairs for strategy in strategies)

    statuses = Counter(row["status"] for row in rows)
    assert set(statuses) <= {"SUCCESS", "INVERSION_FAILURE"}
    assert statuses["SUCCESS"] == summary["totals"]["success"]
    assert statuses["INVERSION_FAILURE"] == summary["totals"]["failure"]
    assert statuses["SUCCESS"] + statuses["INVERSION_FAILURE"] == len(rows)
    assert summary["totals"]["attempts"] == len(rows)
    assert summary["totals"]["start_guess_failures"] == 0
    assert summary["totals"]["target_generation_failures"] == 0
    assert summary["totals"]["invalid_target_samples"] == 0
    assert summary["all_combinations_completed_requested_strategy_trials"] is True

    successes = [row for row in rows if row["status"] == "SUCCESS"]
    target_matches = sum(row["target_residual_match"] for row in successes)
    full_state_matches = sum(row["full_state_match"] for row in successes)
    failures = [row for row in rows if row["status"] == "INVERSION_FAILURE"]
    failure_clusters = Counter((row["fluid"], row["pair"], row["guess_strategy"])
                               for row in failures)
    pair_summary = summary["pair_summary"]
    assert set(pair_summary) == {f"{fluid}:{pair}" for fluid, pair in expected_pairs}
    for fluid, pair in expected_pairs:
        key = f"{fluid}:{pair}"
        pair_rows = [row for row in rows if row["fluid"] == fluid and row["pair"] == pair]
        assert pair_summary[key]["requested_strategy_trials"] == trials_per_pair
        assert pair_summary[key]["success"] == sum(row["status"] == "SUCCESS" for row in pair_rows)
        assert pair_summary[key]["failure"] == sum(
            row["status"] == "INVERSION_FAILURE" for row in pair_rows)
        assert pair_summary[key]["full_state_match"] == sum(
            row.get("full_state_match", False) for row in pair_rows)
        assert pair_summary[key]["target_residual_match"] == sum(
            row.get("target_residual_match", False) for row in pair_rows)

    result = {
        "schema": "solver-wave3-property-pair-campaign-verification-v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "PURE_PYTHON_SELF_CONSISTENCY_ONLY_NOT_NATIVE_PARITY",
        "pinned_source_commit_label": summary["pinned_source_commit"],
        "case_file": CASES.relative_to(ROOT).as_posix(),
        "case_file_sha256": sha256(CASES),
        "campaign_summary_sha256": sha256(SUMMARY),
        "campaign_runner_sha256": sha256(ROOT / "tools/run_solver_wave3_property_pair_campaign.py"),
        "runtime_tpx_sha256": sha256(ROOT / "cantera_python/tpx.py"),
        "trials": len(rows),
        "fluid_pair_combinations": len(expected_pairs),
        "trials_per_fluid_pair": trials_per_pair,
        "records_per_fluid_pair": dict(sorted((f"{fluid}:{pair}", count)
                                               for (fluid, pair), count in pair_counts.items())),
        "trials_per_strategy_per_fluid_pair": trials_per_strategy,
        "strategy_coverage_complete": True,
        "status_counts": dict(statuses),
        "successful_target_pair_residual_matches": target_matches,
        "successful_full_state_matches": full_state_matches,
        "solver_failure_rate": len(failures) / len(rows),
        "failure_clusters": [
            {"fluid": fluid, "pair": pair, "guess_strategy": strategy, "count": count}
            for (fluid, pair, strategy), count in failure_clusters.most_common()
        ],
        "native_parity": "NOT_TESTED",
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "trials", "fluid_pair_combinations", "strategy_coverage_complete",
        "status_counts", "successful_target_pair_residual_matches",
        "successful_full_state_matches", "solver_failure_rate", "native_parity",
        "output")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
