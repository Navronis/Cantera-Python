"""Freeze the canonical 5,784-symbol audit inventory.

This is a schema-normalization step, not a fresh semantic audit.  In
particular, the legacy audit's EQUIVALENT_PROVEN status was assigned by a
class/name heuristic.  Both that bucket and PARTIALLY_PORTED are therefore
kept as historical evidence but mapped to the temporary PARTIAL status until
per-symbol pinned-native evidence exists.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
NATIVE_ROOT = ROOT.parent / "cantera-original"
SOURCE = PROVENANCE / "original_cantera_symbols.jsonl"
DESTINATION = PROVENANCE / "canonical_5784_symbol_inventory.jsonl"
DISCOVERY_RECONCILIATION = (
    PROVENANCE / "symbol_discovery_false_positive_reconciliation.json"
)
IMPLEMENTATION_EVIDENCE = PROVENANCE / "symbol_implementation_evidence.jsonl"
PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"
CANONICAL_TOTAL = 5784

CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}
FORTRAN_SUFFIXES = {".f", ".f90", ".f95"}
NAMESPACE_BEFORE_BRACE = re.compile(
    r"(?:^|\s)(?:inline\s+)?namespace(?:\s+([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*))?\s*$")
CLASS_BEFORE_BRACE = re.compile(
    r"(?:^|\s)(class|struct)\s+([A-Za-z_]\w*)"
    r"(?:\s+final)?(?:\s*:[^{}]*)?\s*$")
ACCESS_LABEL = re.compile(r"\b(public|protected|private)\s*:")


def sanitize_cpp_lines(text: str) -> list[str]:
    """Remove comments and literals while preserving line/column positions."""
    output = []
    current = []
    block_comment = False
    line_comment = False
    quote = None
    escaped = False
    i = 0
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if char == "\n":
            output.append("".join(current))
            current = []
            line_comment = False
            if quote is not None:
                quote = None
                escaped = False
            i += 1
            continue
        if line_comment:
            current.append(" ")
            i += 1
            continue
        if block_comment:
            if char == "*" and nxt == "/":
                current.extend("  ")
                block_comment = False
                i += 2
            else:
                current.append(" ")
                i += 1
            continue
        if quote is not None:
            current.append(" ")
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char == "/" and nxt == "*":
            current.extend("  ")
            block_comment = True
            i += 2
        elif char == "/" and nxt == "/":
            current.extend("  ")
            line_comment = True
            i += 2
        elif char in {'"', "'"}:
            quote = char
            current.append(" ")
            i += 1
        else:
            current.append(char)
            i += 1
    output.append("".join(current))
    return output


def source_line_signature(line: str, symbol: str) -> tuple[str | None, str]:
    """Return a literal one-line declaration when the legacy symbol is present."""
    declaration = " ".join(line.strip().split())
    if not declaration or not re.search(rf"\b{re.escape(symbol)}\s*\(", declaration):
        return None, "NOT_EXTRACTED"
    declaration = re.sub(r"\s*[;{]\s*$", "", declaration).strip()
    return declaration, "SOURCE_LINE_DECLARATION"


def qualified_class(line: str, symbol: str) -> str | None:
    match = re.search(
        rf"\b((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)::{re.escape(symbol)}\s*\(",
        line,
    )
    return match.group(1).split("::")[-1] if match else None


def extract_cpp_scope(text: str, target_lines: set[int]) -> tuple[dict[int, dict], bool]:
    """Extract lexical namespace/class/visibility state at selected lines."""
    sanitized = sanitize_cpp_lines(text)
    scopes: list[dict] = []
    segment = ""
    result = {}
    underflow = False

    for line_number, line in enumerate(sanitized, start=1):
        class_scopes = [scope for scope in scopes if scope["kind"] == "class"]
        access_match = ACCESS_LABEL.search(line)
        if access_match and class_scopes:
            class_scopes[-1]["visibility"] = access_match.group(1)

        if line_number in target_lines:
            namespaces = [scope["name"] for scope in scopes if scope["kind"] == "namespace"]
            classes = [scope for scope in scopes if scope["kind"] == "class"]
            result[line_number] = {
                "namespace": "::".join(namespaces) if namespaces else None,
                "class": classes[-1]["name"] if classes else None,
                "visibility": classes[-1]["visibility"] if classes else None,
                "sanitized_line": line,
            }

        for char in line + "\n":
            if char == "{":
                namespace_match = NAMESPACE_BEFORE_BRACE.search(segment)
                class_match = CLASS_BEFORE_BRACE.search(segment)
                if namespace_match:
                    name = namespace_match.group(1) or "<anonymous>"
                    scopes.append({"kind": "namespace", "name": name})
                elif class_match:
                    kind, name = class_match.groups()
                    scopes.append({
                        "kind": "class",
                        "name": name,
                        "visibility": "public" if kind == "struct" else "private",
                    })
                else:
                    scopes.append({"kind": "block", "name": None})
                segment = ""
            elif char == "}":
                if scopes:
                    scopes.pop()
                else:
                    underflow = True
                segment = ""
            elif char == ";":
                segment = ""
            else:
                segment += char
                if len(segment) > 8192:
                    segment = segment[-8192:]
    return result, underflow or bool(scopes)


def relevance_candidate(row: dict) -> tuple[str, str]:
    path = row["source_file"].replace("\\", "/")
    if path.startswith("doc/"):
        return "NON_LIBRARY_EXAMPLE_CANDIDATE", "SOURCE_PATH_CLASSIFICATION"
    if path.startswith("interfaces/cython/") or path.startswith("interfaces/fortran/"):
        return "BINDING_RUNTIME_CANDIDATE", "SOURCE_PATH_CLASSIFICATION"
    if path.startswith("interfaces/clib/"):
        return "BINDING_RUNTIME_CANDIDATE", "SOURCE_PATH_CLASSIFICATION"
    if path.startswith("samples/"):
        return "NON_LIBRARY_SAMPLE_CANDIDATE", "SOURCE_PATH_CLASSIFICATION"
    if path.startswith("test/") or path.startswith("test_problems/"):
        return "VALIDATION_ONLY_CANDIDATE", "SOURCE_PATH_CLASSIFICATION"
    if path.startswith("src/") or path.startswith("include/"):
        if row.get("algorithm_category") == "numeric_utility_evaluation":
            return "RUNTIME_UTILITY_CANDIDATE", "SOURCE_PATH_AND_LEGACY_CATEGORY"
        return "SCIENTIFIC_RUNTIME_CANDIDATE", "SOURCE_PATH_AND_LEGACY_CATEGORY"
    if path.startswith("ext/"):
        return "EXTENSION_RUNTIME_CANDIDATE", "SOURCE_PATH_CLASSIFICATION"
    return "NOT_AUDITED", "UNRESOLVED"


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_status(legacy_status: str) -> str:
    if legacy_status in {"EQUIVALENT_PROVEN", "PARTIALLY_PORTED"}:
        return "PARTIAL"
    if legacy_status == "NOT_PORTED":
        return "NOT_PORTED"
    return "NOT_AUDITED"


FINAL_STATUSES = {
    "PINNED_NATIVE_VALIDATED",
    "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED",
    "NON_RUNTIME_CPP_ONLY_PROVEN",
}
ALLOWED_IMPLEMENTATION_STATUSES = {
    "PARTIAL",
    "IMPLEMENTED_UNVALIDATED",
    "PINNED_NATIVE_VALIDATED",
    "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED",
    "NON_RUNTIME_CPP_ONLY_PROVEN",
}


def next_action(status: str) -> str:
    if status in FINAL_STATUSES:
        return "NONE_FINAL_DISPOSITION"
    if status == "PARTIAL":
        return "VERIFY_MAPPING_COMPLETE_IMPLEMENTATION_AND_RUN_PINNED_ORACLE"
    if status == "IMPLEMENTED_UNVALIDATED":
        return "VALIDATE_SOURCE_MAPPING_AND_RUN_PINNED_ORACLE"
    if status == "NOT_PORTED":
        return "CLASSIFY_RELEVANCE_THEN_IMPLEMENT_OR_PROVE_NON_RUNTIME"
    return "AUDIT_SOURCE_BODY_AND_ASSIGN_DISPOSITION"


def build() -> list[dict]:
    legacy = read_jsonl(SOURCE)
    if len(legacy) != CANONICAL_TOTAL:
        raise RuntimeError(
            f"expected {CANONICAL_TOTAL} source rows, found {len(legacy)}")

    ids = [row["id"] for row in legacy]
    duplicate_ids = [key for key, count in Counter(ids).items() if count != 1]
    if duplicate_ids:
        raise RuntimeError(f"duplicate legacy symbol IDs: {duplicate_ids[:10]}")

    reconciliation = json.loads(
        DISCOVERY_RECONCILIATION.read_text(encoding="utf-8"))
    if reconciliation["legacy_denominator_preserved"] != CANONICAL_TOTAL:
        raise RuntimeError("symbol discovery reconciliation changed the denominator")
    non_symbol_rows = {
        row["stable_symbol_id"]: row for row in reconciliation["rejected_rows"]
        if row.get("canonical_disposition") == "NON_RUNTIME_CPP_ONLY_PROVEN"
    }
    implementation_rows = (
        read_jsonl(IMPLEMENTATION_EVIDENCE)
        if IMPLEMENTATION_EVIDENCE.is_file() else []
    )
    implementation_by_id = {}
    for evidence in implementation_rows:
        symbol_id = evidence["stable_symbol_id"]
        if symbol_id in implementation_by_id:
            raise RuntimeError(f"duplicate implementation evidence for {symbol_id}")
        if evidence["current_status"] not in ALLOWED_IMPLEMENTATION_STATUSES:
            raise RuntimeError(
                f"implementation evidence has invalid status '{evidence['current_status']}' for {symbol_id}")
        implementation_by_id[symbol_id] = evidence
    unknown_evidence = set(implementation_by_id) - set(ids)
    if unknown_evidence:
        raise RuntimeError(f"implementation evidence references unknown IDs: {unknown_evidence}")
    legacy_by_id = {row["id"]: row for row in legacy}
    for symbol_id, evidence in implementation_by_id.items():
        source = legacy_by_id[symbol_id]
        if evidence.get("source_commit") != PINNED_COMMIT:
            raise RuntimeError(f"wrong implementation-evidence commit for {symbol_id}")
        if evidence.get("source_hash") != source["source_sha256"]:
            raise RuntimeError(f"implementation-evidence hash mismatch for {symbol_id}")
        if not evidence.get("python_mapping"):
            raise RuntimeError(f"missing Python mapping for {symbol_id}")
        if not evidence.get("validation_evidence"):
            raise RuntimeError(f"missing validation evidence for {symbol_id}")
    conflicting_evidence = set(implementation_by_id) & set(non_symbol_rows)
    if conflicting_evidence:
        raise RuntimeError(
            f"implementation evidence conflicts with non-symbol proof: {conflicting_evidence}")

    hashes_by_file: dict[str, set[str]] = defaultdict(set)
    for row in legacy:
        if row.get("source_commit") != PINNED_COMMIT:
            raise RuntimeError(f"wrong source commit for {row['id']}")
        hashes_by_file[row["source_file"]].add(row["source_sha256"])

    for source_file, recorded_hashes in hashes_by_file.items():
        if len(recorded_hashes) != 1:
            raise RuntimeError(f"conflicting hashes for {source_file}")
        native_path = NATIVE_ROOT / source_file
        if not native_path.is_file():
            raise RuntimeError(f"pinned source file is missing: {source_file}")
        actual_hash = file_sha256(native_path)
        if actual_hash not in recorded_hashes:
            raise RuntimeError(f"source hash mismatch: {source_file}")

    rows_by_file: dict[str, list[dict]] = defaultdict(list)
    for row in legacy:
        rows_by_file[row["source_file"]].append(row)

    source_metadata: dict[str, dict] = {}
    for source_file, file_rows in rows_by_file.items():
        path = NATIVE_ROOT / source_file
        text = path.read_text(encoding="utf-8", errors="ignore")
        suffix = path.suffix.lower()
        if suffix in CPP_SUFFIXES:
            targets = {row["line_range"][0] for row in file_rows}
            scope, unbalanced = extract_cpp_scope(text, targets)
            source_metadata[source_file] = {
                "language": "C_OR_CPP",
                "lines": text.splitlines(),
                "scope": scope,
                "scope_unbalanced": unbalanced,
            }
        elif suffix in FORTRAN_SUFFIXES:
            source_metadata[source_file] = {
                "language": "FORTRAN",
                "lines": text.splitlines(),
                "scope": {},
                "scope_unbalanced": False,
            }
        else:
            source_metadata[source_file] = {
                "language": "UNKNOWN",
                "lines": text.splitlines(),
                "scope": {},
                "scope_unbalanced": False,
            }

    canonical = []
    for index, row in enumerate(legacy, start=1):
        status = canonical_status(row["status"])
        non_symbol_proof = non_symbol_rows.get(row["id"])
        implementation = implementation_by_id.get(row["id"])
        if non_symbol_proof is not None:
            status = "NON_RUNTIME_CPP_ONLY_PROVEN"
        elif implementation is not None:
            status = implementation["current_status"]
        mapping = (implementation["python_mapping"] if implementation is not None
                   else row.get("python_equivalent"))
        metadata = source_metadata[row["source_file"]]
        source_line_number = row["line_range"][0]
        source_line = metadata["lines"][source_line_number - 1]
        relevance, relevance_evidence = relevance_candidate(row)
        if non_symbol_proof is not None:
            relevance = "NON_SYMBOL_INVENTORY_ARTIFACT"
            relevance_evidence = "SOURCE_LINE_IS_EXECUTABLE_STATEMENT"

        namespace = None
        extracted_class = None
        visibility = "NOT_AUDITED"
        namespace_status = "NOT_AUDITED"
        class_status = "NOT_AUDITED"
        visibility_status = "NOT_AUDITED"
        signature, signature_status = source_line_signature(source_line, row["symbol"])

        if metadata["language"] == "C_OR_CPP":
            scope = metadata["scope"].get(source_line_number, {})
            namespace = scope.get("namespace")
            lexical_class = scope.get("class")
            qualifier_class = qualified_class(scope.get("sanitized_line", source_line), row["symbol"])
            extracted_class = lexical_class or qualifier_class
            if metadata["scope_unbalanced"]:
                namespace_status = "LEXICAL_SCOPE_FILE_UNBALANCED"
                class_status = "LEXICAL_SCOPE_FILE_UNBALANCED"
                visibility_status = "LEXICAL_SCOPE_FILE_UNBALANCED"
                namespace = None
                extracted_class = None
                visibility = "NOT_AUDITED"
            else:
                namespace_status = (
                    "LEXICAL_SCOPE" if namespace is not None
                    else "DETERMINISTIC_NO_NAMESPACE_AT_SYMBOL"
                )
                if lexical_class is not None:
                    class_status = "LEXICAL_SCOPE"
                    visibility = scope["visibility"]
                    visibility_status = "LEXICAL_CLASS_ACCESS"
                elif qualifier_class is not None:
                    class_status = "QUALIFIED_OUT_OF_CLASS_DEFINITION"
                    visibility = "NOT_AUDITED"
                    visibility_status = "DECLARATION_REQUIRED"
                else:
                    class_status = "DETERMINISTIC_NO_CLASS_AT_SYMBOL"
                    visibility = "NOT_APPLICABLE"
                    visibility_status = "FREE_OR_NAMESPACE_FUNCTION"
        elif metadata["language"] == "FORTRAN":
            namespace_status = "NOT_APPLICABLE_LANGUAGE"
            class_status = "NOT_APPLICABLE_LANGUAGE"
            visibility = "NOT_AUDITED"
            visibility_status = "LANGUAGE_SPECIFIC_AUDIT_REQUIRED"
        canonical.append({
            "schema_version": 1,
            "canonical_index": index,
            "stable_symbol_id": row["id"],
            "subsystem": row["subsystem"],
            "source_file": row["source_file"],
            "line_range": row["line_range"],
            "namespace": namespace,
            "class": extracted_class,
            "legacy_class_candidate": row.get("class"),
            "function_or_method": row["symbol"],
            "signature": signature,
            "visibility": visibility,
            "symbol_type": row["type"],
            "source_hash": row["source_sha256"],
            "source_commit": row["source_commit"],
            "scientific_runtime_relevance": relevance,
            "scientific_runtime_relevance_evidence": relevance_evidence,
            "callers": None,
            "callees": None,
            "model_factory_registration": {
                "applicability": "NOT_AUDITED",
                "registrations": None,
            },
            "python_mapping": mapping,
            "python_mapping_confidence": (
                implementation.get("python_mapping_confidence", "SOURCE_REVIEWED_MAPPING")
                if implementation is not None else
                ("LEGACY_NAME_OR_CLASS_CANDIDATE_UNVERIFIED"
                 if mapping is not None else "UNASSIGNED")
            ),
            "current_status": status,
            "legacy_status": row["status"],
            "legacy_algorithm_category": row.get("algorithm_category"),
            "legacy_scientific_purpose": row.get("scientific_purpose"),
            "legacy_notes": row.get("notes"),
            "metadata_extraction_status": {
                "namespace": namespace_status,
                "class": class_status,
                "signature": signature_status,
                "visibility": visibility_status,
                "call_graph": "NOT_AUDITED",
                "registration": "NOT_AUDITED",
            },
            "required_next_action": next_action(status),
            "final_disposition": (
                "NON_RUNTIME_CPP_ONLY_PROVEN" if non_symbol_proof is not None
                else (status if status in FINAL_STATUSES else None)
            ),
            "non_runtime_proof": (
                non_symbol_proof["non_runtime_proof"] if non_symbol_proof is not None
                else (implementation.get("non_runtime_proof") if implementation is not None else None)
            ),
            "validation_evidence": ([
                "provenance/symbol_discovery_false_positive_reconciliation.json"
            ] if non_symbol_proof is not None else
                implementation.get("validation_evidence", [])
                if implementation is not None else []),
        })

    if len(canonical) != CANONICAL_TOTAL:
        raise AssertionError("canonical row count changed during normalization")
    if len({row["stable_symbol_id"] for row in canonical}) != CANONICAL_TOTAL:
        raise AssertionError("canonical stable IDs are not unique")
    return canonical


def main() -> None:
    rows = build()
    DESTINATION.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(DESTINATION),
        "rows": len(rows),
        "current_status_counts": dict(Counter(
            row["current_status"] for row in rows)),
        "legacy_status_counts": dict(Counter(
            row["legacy_status"] for row in rows)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
