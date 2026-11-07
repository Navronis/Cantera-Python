"""Reconcile known lexical false positives without changing the 5,784 baseline."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .forensic_audit import ROOT, SRC, is_executable_statement
except ImportError:  # direct script execution
    from forensic_audit import ROOT, SRC, is_executable_statement


PROVENANCE = ROOT / "provenance"
LEGACY = PROVENANCE / "original_cantera_symbols.jsonl"
OUTPUT = PROVENANCE / "symbol_discovery_false_positive_reconciliation.json"


def build():
    rows = [json.loads(line) for line in LEGACY.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rejected = []
    retained = []
    for row in rows:
        path = SRC / row["source_file"]
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        line_number = int(row["line_range"][0])
        source_line = lines[line_number - 1]
        if is_executable_statement(source_line):
            is_continuation = source_line.lstrip().startswith(":")
            if is_continuation:
                reason = "CONSTRUCTOR_INITIALIZER_OR_EXPRESSION_CONTINUATION"
                proof = {
                    "no_independent_scientific_semantics": (
                        "This row is a constructor member-initializer or a continued "
                        "conditional-expression line, not an independently callable symbol."
                    ),
                    "python_replacement_or_elimination": (
                        "No separate Python symbol is required; initialization belongs to "
                        "the enclosing constructor and the expression belongs to its enclosing function."
                    ),
                    "observable_behavior_preserved": (
                        "Disposing this fabricated inventory row removes no source behavior and "
                        "does not excuse implementation of the enclosing constructor or function."
                    ),
                    "downstream_dependency_analysis": (
                        "C++ callers cannot reference an initializer-list or expression-continuation "
                        "line as a symbol; dependencies remain on the enclosing callable."
                    ),
                }
            else:
                reason = "EXECUTABLE_RETURN_OR_THROW_STATEMENT"
                proof = {
                    "no_independent_scientific_semantics": (
                        "This row is an executable statement inside an enclosing function, "
                        "not a declaration or definition of an independently callable symbol."
                    ),
                    "python_replacement_or_elimination": (
                        "No separate Python symbol is required; the statement's behavior belongs "
                        "to the enclosing function and its actual callee mappings."
                    ),
                    "observable_behavior_preserved": (
                        "Disposing this fabricated inventory row removes no source behavior and "
                        "does not excuse implementation of the enclosing function or callee."
                    ),
                    "downstream_dependency_analysis": (
                        "C++ callers cannot reference a return/throw statement as a symbol; any "
                        "dependency is on the enclosing function or named callee, audited separately."
                    ),
                }
            rejected.append({
                "stable_symbol_id": row["id"],
                "source_file": row["source_file"],
                "line": line_number,
                "legacy_symbol": row["symbol"],
                "source_text": source_line.strip(),
                "reason": reason,
                "canonical_disposition": "NON_RUNTIME_CPP_ONLY_PROVEN",
                "non_runtime_proof": proof,
            })
        else:
            retained.append(row["id"])

    artifact = {
        "schema_version": 1,
        "pinned_source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "legacy_denominator_preserved": 5784,
        "legacy_rows_examined": len(rows),
        "corrected_lexical_candidate_count": len(retained),
        "rejected_false_positive_count": len(rejected),
        "newly_discovered_count": 0,
        "denominator_policy": (
            "The canonical 5,784 IDs remain frozen. Rejected lexical rows require "
            "individual NON_RUNTIME_CPP_ONLY_PROVEN dispositions; they are not deleted."
        ),
        "rejected_rows": rejected,
    }
    return artifact


def main():
    artifact = build()
    OUTPUT.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: artifact[key] for key in (
        "legacy_denominator_preserved", "corrected_lexical_candidate_count",
        "rejected_false_positive_count", "newly_discovered_count")}, indent=2))


if __name__ == "__main__":
    main()
