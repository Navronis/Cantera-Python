"""Build the strict 5,784-symbol closure dashboard.

Legacy ``EQUIVALENT_PROVEN`` rows are deliberately reported as ``PARTIAL``
until the independent revalidation artifact gives them a final disposition.
This prevents historical name/body heuristics from satisfying the closure gate.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
CANONICAL_TOTAL = 5784
FINAL_STATUSES = {
    "PINNED_NATIVE_VALIDATED",
    "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED",
    "NON_RUNTIME_CPP_ONLY_PROVEN",
}
TEMPORARY_STATUSES = {
    "PARTIAL", "NOT_PORTED", "INCORRECT", "BLOCKED", "NOT_AUDITED",
}


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _canonical_status(row):
    status = row.get("current_status", row.get("status", "NOT_AUDITED"))
    aliases = {
        "PARTIALLY_PORTED": "PARTIAL",
        # Revalidation, rather than the legacy label alone, is required before
        # a symbol can enter either validated final category.
        "EQUIVALENT_PROVEN": "PARTIAL",
        "STRUCTURAL_MATCH": "PARTIAL",
        "EQUATION_MATCH": "PARTIAL",
        "BEHAVIOR_MATCH": "PARTIAL",
    }
    status = aliases.get(status, status)
    if status not in FINAL_STATUSES | TEMPORARY_STATUSES:
        status = "NOT_AUDITED"
    return status


def build_dashboard(inventory_path=None):
    inventory_path = Path(inventory_path or
        (PROVENANCE / "canonical_5784_symbol_inventory.jsonl"))
    if not inventory_path.exists():
        inventory_path = PROVENANCE / "original_cantera_symbols.jsonl"
    rows = _load_jsonl(inventory_path)
    if len(rows) != CANONICAL_TOTAL:
        raise RuntimeError(
            f"canonical inventory must contain exactly {CANONICAL_TOTAL} rows; "
            f"found {len(rows)} in {inventory_path}")
    ids = [row.get("stable_symbol_id", row.get("id")) for row in rows]
    if any(symbol_id is None for symbol_id in ids):
        raise RuntimeError("every canonical inventory row must have a stable symbol ID")
    duplicates = [symbol_id for symbol_id, count in Counter(ids).items() if count != 1]
    if duplicates:
        raise RuntimeError(f"canonical inventory contains duplicate IDs: {duplicates[:10]}")

    counts = Counter(_canonical_status(row) for row in rows)
    relevance = Counter(
        str(row.get("scientific_runtime_relevance", "NOT_AUDITED")).upper()
        for row in rows)
    final_count = sum(counts[status] for status in FINAL_STATUSES)
    open_count = CANONICAL_TOTAL - final_count
    runtime_set = {
        "SCIENTIFIC", "RUNTIME", "SCIENTIFIC_RUNTIME", "REQUIRED",
        "SCIENTIFIC_RUNTIME_CANDIDATE", "RUNTIME_UTILITY_CANDIDATE",
    }
    runtime_required = sum(
        1 for row in rows
        if str(row.get("scientific_runtime_relevance", "")).upper() in runtime_set
        and _canonical_status(row) != "NON_RUNTIME_CPP_ONLY_PROVEN"
    )
    runtime_validated = sum(
        1 for row in rows
        if str(row.get("scientific_runtime_relevance", "")).upper() in runtime_set
        and _canonical_status(row) in {
            "PINNED_NATIVE_VALIDATED", "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED"}
    )
    blockers_zero = all(counts[name] == 0 for name in TEMPORARY_STATUSES)
    complete = final_count == CANONICAL_TOTAL and blockers_zero
    return {
        "schema_version": 1,
        "pinned_source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "inventory": str(inventory_path.relative_to(ROOT)).replace("\\", "/"),
        "TOTAL": CANONICAL_TOTAL,
        "PINNED_NATIVE_VALIDATED": counts["PINNED_NATIVE_VALIDATED"],
        "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED": counts[
            "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED"],
        "NON_RUNTIME_CPP_ONLY_PROVEN": counts["NON_RUNTIME_CPP_ONLY_PROVEN"],
        "PARTIAL": counts["PARTIAL"],
        "NOT_PORTED": counts["NOT_PORTED"],
        "INCORRECT": counts["INCORRECT"],
        "BLOCKED": counts["BLOCKED"],
        "NOT_AUDITED": counts["NOT_AUDITED"],
        "fully_dispositioned": final_count,
        "remaining_without_final_disposition": open_count,
        "scientific_runtime_relevance_counts": dict(sorted(relevance.items())),
        "required_scientific_runtime_symbols": runtime_required,
        "validated_scientific_runtime_symbols": runtime_validated,
        "validated_scientific_runtime_percent": (
            100.0 * runtime_validated / runtime_required if runtime_required else 0.0),
        "FULL_PORT_COMPLETE": complete,
        "closure_gate": {
            "all_5784_final": final_count == CANONICAL_TOTAL,
            "temporary_statuses_zero": blockers_zero,
            "permitted_final_statuses": sorted(FINAL_STATUSES),
        },
        "status_policy": (
            "Legacy EQUIVALENT_PROVEN is counted as PARTIAL until independent "
            "revalidation assigns an allowed final status."),
    }


def main():
    dashboard = build_dashboard()
    destination = PROVENANCE / "symbol_closure_dashboard.json"
    destination.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "TOTAL": dashboard["TOTAL"],
        "fully_dispositioned": dashboard["fully_dispositioned"],
        "remaining": dashboard["remaining_without_final_disposition"],
        "FULL_PORT_COMPLETE": dashboard["FULL_PORT_COMPLETE"],
    }))


if __name__ == "__main__":
    main()
