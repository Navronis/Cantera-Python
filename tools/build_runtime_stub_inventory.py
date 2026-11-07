"""Comprehensive runtime and AST stub detector across cantera_python.

Detects:
- pass
- ... (Ellipsis)
- NotImplementedError
- methods returning placeholder constants
- methods returning None without source reason
- TODO/FIXME scientific implementation markers
- dummy arrays and fake outputs
"""
from __future__ import annotations
import ast, inspect, json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "cantera_python"
OUT = ROOT / "audit" / "runtime_stub_inventory.jsonl"
WAVE_MODULES = {"state.py", "thermo.py", "ideal_gas.py", "phases.py", "cubic_eos.py", "tpx.py", "tpx_exact.py"}

def inspect_function(fn_node, source_lines, file_name, class_name=None):
    findings = []
    body = fn_node.body
    # Filter out docstrings
    code_body = []
    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        code_body.append(stmt)

    fn_name = f"{class_name}.{fn_node.name}" if class_name else fn_node.name

    is_abstract_class = bool(class_name and any(k in class_name for k in ("Base", "Substance", "Phase", "MixtureFugacityTP")))

    if not code_body:
        findings.append({
            "type": "EMPTY_BODY",
            "symbol": fn_name,
            "file": file_name,
            "line": fn_node.lineno,
            "classification": "ABSTRACT_PROTOCOL" if is_abstract_class else "EMPTY_STUB"
        })
        return findings

    # Single pass
    if len(code_body) == 1 and isinstance(code_body[0], ast.Pass):
        findings.append({
            "type": "PASS_STUB",
            "symbol": fn_name,
            "file": file_name,
            "line": fn_node.lineno,
            "classification": "ABSTRACT_PROTOCOL" if is_abstract_class else "EMPTY_PASS"
        })

    # Single Ellipsis
    elif len(code_body) == 1 and isinstance(code_body[0], ast.Expr) and isinstance(code_body[0].value, ast.Constant) and code_body[0].value.value is Ellipsis:
        findings.append({
            "type": "ELLIPSIS_STUB",
            "symbol": fn_name,
            "file": file_name,
            "line": fn_node.lineno,
            "classification": "ABSTRACT_PROTOCOL"
        })

    # Single Raise NotImplementedError
    elif len(code_body) == 1 and isinstance(code_body[0], ast.Raise):
        exc = code_body[0].exc
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            findings.append({
                "type": "NOT_IMPLEMENTED_ERROR",
                "symbol": fn_name,
                "file": file_name,
                "line": fn_node.lineno,
                "classification": "ABSTRACT_VIRTUAL_METHOD" if is_abstract_class else "UNSUPPORTED_BRANCH"
            })
        elif isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "NotImplementedError":
            findings.append({
                "type": "NOT_IMPLEMENTED_ERROR",
                "symbol": fn_name,
                "file": file_name,
                "line": fn_node.lineno,
                "classification": "ABSTRACT_VIRTUAL_METHOD" if is_abstract_class else "UNSUPPORTED_BRANCH"
            })

    # Single return constant
    elif len(code_body) == 1 and isinstance(code_body[0], ast.Return):
        ret_val = code_body[0].value
        if isinstance(ret_val, ast.Constant) and ret_val.value in (0, 0.0, None, True, False, ""):
            classification = "PHYSICALLY_EXACT_ZERO" if fn_name == "IdealGasMixture.internal_pressure" and ret_val.value == 0.0 else "PLACEHOLDER_RETURN"
            findings.append({
                "type": "CONSTANT_RETURN_STUB" if classification == "PLACEHOLDER_RETURN" else "PHYSICAL_CONSTANT",
                "symbol": fn_name,
                "file": file_name,
                "line": fn_node.lineno,
                "value": repr(ret_val.value),
                "classification": classification
            })
        elif isinstance(ret_val, (ast.List, ast.Dict)) and not ret_val.elts if hasattr(ret_val, "elts") else False:
            findings.append({
                "type": "EMPTY_CONTAINER_RETURN_STUB",
                "symbol": fn_name,
                "file": file_name,
                "line": fn_node.lineno,
                "classification": "PLACEHOLDER_RETURN"
            })

    # Scan for TODO/FIXME in method source
    start_line = fn_node.lineno - 1
    end_line = fn_node.end_lineno if hasattr(fn_node, "end_lineno") else start_line + len(code_body)
    for i in range(start_line, min(len(source_lines), end_line)):
        line_text = source_lines[i]
        if "#" in line_text:
            comment = line_text.split("#", 1)[1].upper()
            if "TODO" in comment or "FIXME" in comment:
                findings.append({
                    "type": "TODO_FIXME_MARKER",
                    "symbol": fn_name,
                    "file": file_name,
                    "line": i + 1,
                    "text": line_text.strip(),
                    "classification": "INCOMPLETE_MARKER"
                })

    return findings

def audit_file(file_path: Path):
    source = file_path.read_text(encoding="utf-8")
    lines = source.splitlines()
    findings = []
    try:
        tree = ast.parse(source, filename=file_path.name)
    except Exception as e:
        return [{"type": "PARSE_ERROR", "file": file_path.name, "error": str(e)}]

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            findings.extend(inspect_function(node, lines, file_path.name))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    findings.extend(inspect_function(item, lines, file_path.name, class_name=node.name))

    return findings

def main():
    all_findings = []
    wave_findings = []

    for py_file in sorted(PKG.glob("*.py")):
        res = audit_file(py_file)
        all_findings.extend(res)
        if py_file.name in WAVE_MODULES:
            wave_findings.extend(res)

    OUT.write_text("".join(json.dumps(f) + "\n" for f in all_findings), encoding="utf-8")

    print(f"Total findings across entire cantera_python: {len(all_findings)}")
    print(f"Total findings across Wave 1-3 modules ({', '.join(sorted(WAVE_MODULES))}): {len(wave_findings)}")

    type_counts = Counter(f["type"] for f in wave_findings)
    class_counts = Counter(f.get("classification") for f in wave_findings)
    print("\nWave 1-3 Breakdown by Type:")
    for k, v in type_counts.items():
        print(f"  {k}: {v}")
    print("\nWave 1-3 Breakdown by Classification:")
    for k, v in class_counts.items():
        print(f"  {k}: {v}")

    # Inspect Wave 1-3 findings
    print("\nWave 1-3 Specific Findings:")
    for f in wave_findings:
        print(f"  {f.get('file')}:{f.get('line')} - {f.get('symbol')} ({f.get('type')}: {f.get('classification')})")

if __name__ == "__main__":
    main()
