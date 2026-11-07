#!/usr/bin/env python3
"""Revalidate legacy ``EQUIVALENT_PROVEN`` source-symbol claims.

This audit is intentionally strict. A matching name, class, source-file stem, or
subsystem benchmark is useful evidence of a port, but is not sufficient to prove
that one particular C++ symbol is equivalent. The script records source and
Python bodies and searches exact-pinned artifacts for explicit symbol-level
evidence. Claims without all three links are downgraded.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
NATIVE = ROOT.parent / "cantera-original"
PYTHON = ROOT / "cantera_python"
INPUT = PROVENANCE / "original_cantera_symbols.jsonl"
OUTPUT = PROVENANCE / "revalidation_421.jsonl"
SUMMARY = PROVENANCE / "revalidation_421_summary.json"
PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snake(name: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def source_fragment(path: Path, line: int) -> tuple[str, str, list[int]]:
    """Return a declaration/definition fragment beginning at a recorded line."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, line - 1)
    collected: list[str] = []
    braces = 0
    saw_brace = False
    form = "unknown"
    for index in range(start, min(len(lines), start + 400)):
        current = lines[index]
        collected.append(current)
        # This is a lexical body locator, not a C++ parser. It is sufficient to
        # distinguish the audit's header declarations from inline definitions.
        code = re.sub(r"//.*$", "", current)
        opens = code.count("{")
        closes = code.count("}")
        if opens:
            saw_brace = True
            form = "definition"
        braces += opens - closes
        if saw_brace and braces <= 0:
            return "\n".join(collected), form, [start + 1, index + 1]
        if not saw_brace and ";" in code:
            return "\n".join(collected), "declaration", [start + 1, index + 1]
    end = start + len(collected)
    return "\n".join(collected), form, [start + 1, end]


def python_definitions() -> list[dict]:
    definitions: list[dict] = []
    for path in sorted(PYTHON.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scope: list[str] = []

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
                qualname = ".".join([*self.scope, node.name])
                segment = ast.get_source_segment(text, node) or ""
                definitions.append({
                    "name": node.name,
                    "qualname": qualname,
                    "module": path.relative_to(ROOT).as_posix(),
                    "line_range": [node.lineno, getattr(node, "end_lineno", node.lineno)],
                    "body_sha256": digest(segment),
                })
                self.scope.append(node.name)
                self.generic_visit(node)
                self.scope.pop()

            visit_FunctionDef = _function
            visit_AsyncFunctionDef = _function

        Visitor().visit(tree)
    return definitions


def native_definition_index() -> dict[tuple[str, str], list[dict]]:
    """Index lexically identifiable out-of-line ``Class::method`` bodies."""
    index: dict[tuple[str, str], list[dict]] = {}
    qualified = re.compile(
        r"\b(?P<class>[A-Za-z_]\w*)(?:\s*<[^;{}]*>)?\s*::\s*"
        r"(?P<symbol>~?[A-Za-z_]\w*|operator\s*[^\s(]+)\s*\("
    )
    suffixes = {".h", ".hh", ".hpp", ".cpp", ".cc", ".cxx"}
    for path in sorted(NATIVE.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in suffixes
            or any(part in {"ext", "build"} for part in path.parts)
        ):
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for number, line in enumerate(lines, 1):
            for match in qualified.finditer(line):
                fragment, form, line_range = source_fragment(path, number)
                if form != "definition":
                    continue
                key = (match.group("class"), re.sub(r"\s+", "", match.group("symbol")))
                candidate = {
                    "source_file": path.relative_to(NATIVE).as_posix(),
                    "line_range": line_range,
                    "body_sha256": digest(fragment),
                }
                index.setdefault(key, []).append(candidate)
    return index


def resolve_mapping(row: dict, definitions: list[dict]) -> tuple[str, list[dict]]:
    claimed = row.get("python_equivalent") or ""
    symbol = row["symbol"]
    tail = claimed.rsplit(".", 1)[-1] if claimed else symbol
    names = {symbol, snake(symbol), tail, snake(tail)}
    candidates = [item for item in definitions if item["name"] in names]

    if "." in claimed:
        exact = [item for item in candidates if item["qualname"].endswith(claimed)]
        if exact:
            candidates = exact
    # Limit artifact size while preserving ambiguity evidence.
    candidates = sorted(candidates, key=lambda item: (item["module"], item["line_range"]))
    if not candidates:
        return "UNRESOLVED", []
    if len(candidates) == 1:
        return "RESOLVED_UNIQUE_NAME", candidates
    return "AMBIGUOUS_NAME_MATCH", candidates[:25]


def load_exact_pinned_terms() -> tuple[set[str], list[str]]:
    """Collect labels present in exact-pinned behavioral artifacts.

    These labels establish related behavioral coverage only. They are not
    promoted to symbol-level proof because the artifacts do not identify a C++
    symbol id, source-body hash, and Python-body hash together.
    """
    terms: set[str] = set()
    refs: list[str] = []
    baseline = PROVENANCE / "oracle_baseline_pinned.json"
    if baseline.exists():
        data = json.loads(baseline.read_text(encoding="utf-8"))
        commit = str(data.get("oracle_commit", ""))
        if PINNED_COMMIT.startswith(commit):
            refs.append(baseline.relative_to(ROOT).as_posix())

    differential = PROVENANCE / "pinned_native_differential_results.jsonl"
    if differential.exists() and refs:
        refs.append(differential.relative_to(ROOT).as_posix())
        for line in differential.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # Only output property labels can be related to a source symbol.
            # Domain names and arbitrary JSON keys such as ``type`` would create
            # misleading matches.
            if row.get("property"):
                terms.add(str(row["property"]).lower())
    return terms, refs


def main() -> None:
    definitions = python_definitions()
    native_definitions = native_definition_index()
    evidence_terms, evidence_files = load_exact_pinned_terms()
    inventory = [
        json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    legacy = [row for row in inventory if row.get("status") == "EQUIVALENT_PROVEN"]
    if len(legacy) != 421:
        raise RuntimeError(f"Expected 421 legacy claims, found {len(legacy)}")

    results: list[dict] = []
    for row in legacy:
        path = NATIVE / row["source_file"]
        source_exists = path.is_file()
        source_hash_matches = False
        fragment = ""
        source_form = "missing"
        line_range = row.get("line_range", [0, 0])
        fragment_range = line_range
        if source_exists:
            source_hash_matches = hashlib.sha256(path.read_bytes()).hexdigest() == row["source_sha256"]
            fragment, source_form, fragment_range = source_fragment(path, int(line_range[0]))

        source_body_candidates: list[dict] = []
        if source_form == "definition":
            source_body_candidates.append({
                "source_file": row["source_file"],
                "line_range": fragment_range,
                "body_sha256": digest(fragment),
            })
            source_body_status = "DEFINITION_AT_INVENTORY_ROW"
        else:
            key = (row.get("class") or "", re.sub(r"\s+", "", row["symbol"]))
            source_body_candidates = native_definitions.get(key, [])
            source_body_status = (
                "OUT_OF_LINE_DEFINITION_RESOLVED"
                if source_body_candidates
                else "DECLARATION_ONLY_NO_DEFINITION_RESOLVED"
            )

        mapping_status, candidates = resolve_mapping(row, definitions)
        normalized = {row["symbol"].lower(), snake(row["symbol"])}
        related_terms = sorted(normalized & evidence_terms)
        related_evidence = evidence_files if related_terms else []

        # No current evidence artifact binds all of source id/body, Python body,
        # exact pinned input/output, tolerances, and result. Therefore none of
        # the legacy rows meets a reproducible per-symbol proof standard.
        if mapping_status == "UNRESOLVED":
            status = "NOT_PORTED_MAPPING_UNRESOLVED"
        elif related_evidence:
            status = "RELATED_BEHAVIOR_VALIDATED_MAPPING_UNREVIEWED"
        else:
            status = "PARTIAL_MAPPING_REQUIRES_REVIEW"

        results.append({
            "id": row["id"],
            "source_commit": row.get("source_commit"),
            "source_file": row["source_file"],
            "source_symbol": row["symbol"],
            "source_class": row.get("class"),
            "legacy_status": "EQUIVALENT_PROVEN",
            "revalidated_status": status,
            "source_file_exists": source_exists,
            "source_file_sha256_matches_inventory": source_hash_matches,
            "source_fragment_form": source_form,
            "source_fragment_line_range": fragment_range,
            "source_fragment_sha256": digest(fragment) if fragment else None,
            "source_body_status": source_body_status,
            "source_body_candidates": source_body_candidates,
            "claimed_python_equivalent": row.get("python_equivalent"),
            "python_mapping_status": mapping_status,
            "python_candidates": candidates,
            "related_exact_pinned_terms": related_terms,
            "related_exact_pinned_evidence": related_evidence,
            "explicit_per_symbol_exact_pinned_evidence": [],
            "decision": (
                "Downgraded: no evidence record binds this exact C++ symbol/body "
                "to a concrete Python body and exact-pinned result."
            ),
        })

    counts = Counter(row["revalidated_status"] for row in results)
    mapping_counts = Counter(row["python_mapping_status"] for row in results)
    source_forms = Counter(row["source_fragment_form"] for row in results)
    source_bodies = Counter(row["source_body_status"] for row in results)
    summary = {
        "schema_version": 1,
        "source_commit": PINNED_COMMIT,
        "legacy_claims_revalidated": len(results),
        "legacy_equivalent_proven": 421,
        "equivalent_proven_retained": 0,
        "revalidated_status_counts": dict(sorted(counts.items())),
        "python_mapping_counts": dict(sorted(mapping_counts.items())),
        "source_fragment_form_counts": dict(sorted(source_forms.items())),
        "source_body_status_counts": dict(sorted(source_bodies.items())),
        "source_files_missing": sum(not row["source_file_exists"] for row in results),
        "source_hash_mismatches": sum(
            row["source_file_exists"] and not row["source_file_sha256_matches_inventory"]
            for row in results
        ),
        "rows_with_related_exact_pinned_behavior": sum(
            bool(row["related_exact_pinned_evidence"]) for row in results
        ),
        "rows_with_explicit_per_symbol_exact_pinned_evidence": 0,
        "proof_standard": (
            "EQUIVALENT_PROVEN requires a stable C++ symbol id and source-body hash, "
            "a concrete Python definition and body hash, and exact-pinned differential "
            "evidence explicitly linking both implementations with inputs, tolerances, "
            "and pass/fail results."
        ),
        "legacy_421_finding": (
            "All 421 claims were produced by broad name/class/file heuristics. None has "
            "an explicit per-symbol exact-pinned evidence link, so none is retained as "
            "EQUIVALENT_PROVEN by this revalidation."
        ),
        "unsupported_782_reconciliation": {
            "claimed_count": 782,
            "matching_row_level_audit_found": False,
            "authoritative_inventory_total": 5784,
            "authoritative_legacy_counts": {
                "EQUIVALENT_PROVEN": 421,
                "PARTIALLY_PORTED": 336,
                "NOT_PORTED": 5027,
            },
            "note": (
                "No 782-row equivalent-proven audit or derivation is present. Even "
                "421 + 336 equals 757, not 782. The 782 claim is unsupported and must "
                "not be used as a conversion numerator."
            ),
        },
        "artifacts": {
            "rows": OUTPUT.relative_to(ROOT).as_posix(),
            "input": INPUT.relative_to(ROOT).as_posix(),
            "related_exact_pinned_files": evidence_files,
        },
    }

    OUTPUT.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
