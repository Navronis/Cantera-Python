"""Derive the active full-port queue and dependency artifacts from source inventories.

Unknown source call-graph fields stay explicitly unknown; this tool never fills
missing dependency evidence with guessed edges or completion statuses.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
SOLVERS = ROOT / "audit" / "complete_solver_inventory.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def python_ast_index() -> dict:
    index = defaultdict(list)
    for path in sorted((ROOT / "cantera_python").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as err:
            index["__syntax_errors__"].append({"file": rel, "error": str(err)})
            continue
        class IndexVisitor(ast.NodeVisitor):
            def __init__(self):
                self.classes = []

            def visit_ClassDef(self, node):
                index["class:" + node.name].append((rel, node, None))
                self.classes.append(node)
                self.generic_visit(node)
                self.classes.pop()

            def visit_FunctionDef(self, node):
                index["function:" + node.name].append((rel, node, None))
                if self.classes:
                    parent = self.classes[-1]
                    index[f"method:{parent.name}.{node.name}"].append((rel, node, parent))
                self.generic_visit(node)

            visit_AsyncFunctionDef = visit_FunctionDef

        IndexVisitor().visit(tree)
    return index


def implementation_shape(record) -> str:
    if not record:
        return "NO_AST_TARGET"
    _, node, _ = record
    body = node.body
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = []
    if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
        return "EMPTY_OR_DOCSTRING_ONLY"
    if any(isinstance(x, ast.Raise) and isinstance(x.exc, ast.Call)
           and getattr(x.exc.func, "id", "") == "NotImplementedError"
           for x in ast.walk(node)):
        return "RAISES_NOT_IMPLEMENTED"
    return "NONTRIVIAL_AST_BODY_UNVALIDATED"


def main() -> None:
    symbols = read_jsonl(SYMBOLS)
    solver_rows = read_jsonl(SOLVERS)
    index = python_ast_index()
    source_hash = digest(SYMBOLS)
    open_rows = [r for r in symbols if r.get("current_status") in {"PARTIAL", "NOT_PORTED"}]
    solver_by_class = defaultdict(list)
    for solver in solver_rows:
        solver_by_class[solver.get("native_class")].append(solver)

    queue = []
    target_rows = []
    groups = defaultdict(list)
    for row in open_rows:
        cls, name = row.get("class"), row.get("function_or_method")
        candidates = []
        if cls and name:
            for n in (name, snake(name)):
                candidates += index.get(f"method:{cls}.{n}", [])
        if not cls and name:
            for n in (name, snake(name)):
                candidates += index.get(f"function:{n}", [])
        unique = {(rec[0], rec[1].name, getattr(rec[2], "name", None)): rec
                  for rec in candidates}
        candidates = list(unique.values())
        mapping = row.get("python_mapping")
        if mapping and mapping.startswith("cantera_python."):
            target_validity = "EXPLICIT_MAPPING_AST_VERIFICATION_REQUIRED"
        elif len(candidates) == 1:
            target_validity = implementation_shape(candidates[0])
        elif len(candidates) > 1:
            target_validity = "AMBIGUOUS_NAME_MATCH"
        else:
            target_validity = "NO_CURRENT_AST_TARGET"

        family_candidates = solver_by_class.get(cls, [])
        if not family_candidates:
            family_candidates = [s for s in solver_rows
                                 if s.get("native_file") == row.get("source_file")]
        block = f"{row.get('subsystem')}::{cls or '<free>'}::{row.get('source_file')}"
        groups[block].append(row["stable_symbol_id"])
        target = ([{"file": rec[0], "class": getattr(rec[2], "name", None),
                    "function": rec[1].name, "body_check": implementation_shape(rec)}
                   for rec in candidates])
        record = {
            "stable_symbol_id": row["stable_symbol_id"],
            "source_file": row.get("source_file"), "source_hash": row.get("source_hash"),
            "class": cls, "function_or_method": name, "line_range": row.get("line_range"),
            "subsystem": row.get("subsystem"),
            "solver_family_candidates": sorted({s.get("subsystem") for s in family_candidates}),
            "current_status": row.get("current_status"),
            "existing_python_target": mapping,
            "python_target_candidates": target,
            "python_target_validity": target_validity,
            "direct_dependencies": row.get("callees"), "callers": row.get("callers"),
            "downstream_blockers": "UNKNOWN: canonical caller/callee extraction is NOT_AUDITED",
            "scientific_runtime_relevance": row.get("scientific_runtime_relevance"),
            "estimated_implementation_block": block,
            "evidence_currently_available": row.get("validation_evidence", []),
        }
        queue.append(record)
        target_rows.append({"stable_symbol_id": row["stable_symbol_id"],
                            "target_candidates": target,
                            "verification": target_validity,
                            "reachability": "NOT_PROVEN_BY_CURRENT_CANONICAL_CALL_GRAPH"})

    out = ROOT / "audit"
    (out / "full_port_execution_queue.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in queue), encoding="utf-8")
    (out / "python_runtime_target_verification.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in target_rows), encoding="utf-8")

    graph_nodes = [{"block_id": key, "subsystem": key.split("::", 1)[0],
                    "symbol_count": len(ids), "symbols": ids}
                   for key, ids in sorted(groups.items())]
    graph = {
        "schema_version": 1, "source_inventory_sha256": source_hash,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "nodes": graph_nodes, "edges": [],
        "edge_status": "BLOCKED_INCOMPLETE_SOURCE_CALL_GRAPH",
        "evidence": "All canonical rows currently have callers=null and callees=null; no symbol dependency edges can be asserted without source call-graph extraction.",
        "dag_validated": False,
    }
    (out / "full_port_dependency_graph.json").write_text(
        json.dumps(graph, indent=2) + "\n", encoding="utf-8")

    base = {
        "schema_version": 1, "source_inventory_sha256": source_hash,
        "definition_status": "PROVISIONAL_BLOCKED_ON_DEPENDENCY_GRAPH",
        "selection_rule": "Symbols required by more than one solver family, ranked by graph centrality.",
        "selected_symbols": [], "reason": "Canonical symbol callers/callees are unpopulated; cross-family dependency centrality cannot be measured faithfully yet.",
        "next_required_step": "Extract pinned-source function call edges and connect them to Python runtime targets before declaring the shared base closed.",
    }
    (out / "cantera_python_base_definition.json").write_text(
        json.dumps(base, indent=2) + "\n", encoding="utf-8")

    counts = Counter(r.get("current_status") for r in symbols)
    selection = {
        "selected_solver_family": "MultiTransport",
        "selection_status": "ACTIVE_IN_PROGRESS_NOT_CLOSED",
        "scoring": {"dependency_ready": "existing Python implementation and pinned source block present",
                    "downstream_centrality": "not measurable until call graph extraction",
                    "implementation_completeness": "public matrix, flux, and cache parity remain open"},
        "rationale": "Existing Wave 4 artifacts identify MultiTransport as the active transport family, and its canonical source methods are present in Python; continue this block while dependency graph extraction remains open.",
    }
    (out / "next_solver_selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": "current dirty tree on full-port/continuous-closure-20260930",
        "implementation_block": "MultiTransport::getMassFluxes midpoint phase-state semantics",
        "symbols_touched": ["cantera_python.transport.MultiTransport.getMassFluxes"],
        "open_runtime_symbols_before": 5025,
        "open_runtime_symbols_after": counts["PARTIAL"] + counts["NOT_PORTED"],
        "open_runtime_symbol_delta": counts["PARTIAL"] + counts["NOT_PORTED"] - 5025,
        "status_counts": dict(counts),
        "tests": ["new regression tests added; execution status pending"],
        "native_comparisons": [],
        "solver_family_affected": "MultiTransport",
        "next_dependency_block": "finish MultiTransport call-path, binary/multicomponent coefficients, flux/cache semantics, and pinned differential validation",
        "dependency_graph_caveat": "Dependency graph is not a true DAG yet; callers/callees are absent in the canonical inventory.",
    }
    history = out / "full_port_progress_history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps({"open_queue_rows": len(queue), "status_counts": counts,
                      "dependency_edges": len(graph["edges"]),
                      "dependency_graph_ready": False}))


if __name__ == "__main__":
    main()
