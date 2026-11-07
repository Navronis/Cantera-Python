#!/usr/bin/env python3
"""Build complete Wave 4 source inventory from pinned Cantera C++ code."""
import re
import json
from pathlib import Path

ROOT = Path("outputs/cantera-original")
SRC_DIR = ROOT / "src/transport"
INC_DIR = ROOT / "include/cantera/transport"

files_to_parse = [
    ("include/cantera/transport/MultiTransport.h", INC_DIR / "MultiTransport.h", "MultiTransport"),
    ("src/transport/MultiTransport.cpp", SRC_DIR / "MultiTransport.cpp", "MultiTransport"),
    ("include/cantera/transport/GasTransport.h", INC_DIR / "GasTransport.h", "GasTransport"),
    ("src/transport/GasTransport.cpp", SRC_DIR / "GasTransport.cpp", "GasTransport"),
    ("src/transport/MMCollisionInt.h", SRC_DIR / "MMCollisionInt.h", "MMCollisionInt"),
    ("src/transport/MMCollisionInt.cpp", SRC_DIR / "MMCollisionInt.cpp", "MMCollisionInt"),
]

# Regex patterns for method and function declarations / implementations
# e.g., void MultiTransport::getThermalDiffCoeffs(span<double> dt)
# or virtual void getThermalDiffCoeffs(span<double> dt) override;

entries = []

for rel_path, full_path, primary_class in files_to_parse:
    lines = full_path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, 1):
        sline = line.strip()
        if not sline or sline.startswith("//") or sline.startswith("/*") or sline.startswith("*"):
            continue
        
        # Check for function/method declarations in headers
        if rel_path.endswith(".h"):
            # matches: [virtual] [type] name(args) [const] [override] [= 0];
            m = re.match(r'^(?:virtual\s+)?([A-Za-z0-9_:<>&*,\s]+?)\s+([A-Za-z0-9_]+)\s*\((.*?)\)(?:\s+const)?(?:\s+override)?(?:\s*=\s*0)?\s*[;{]', sline)
            if m:
                ret_type, name, args = m.groups()
                ret_type = ret_type.strip()
                # filter out macros / typedefs
                if name not in ("if", "for", "while", "switch", "catch", "return") and not ret_type.startswith("#"):
                    entries.append({
                        "file": rel_path,
                        "line": idx,
                        "symbol": f"{primary_class}::{name}",
                        "name": name,
                        "class": primary_class,
                        "return_type": ret_type,
                        "args": args.strip(),
                        "is_definition": False
                    })
        else: # .cpp
            # matches: type Class::name(args)
            m = re.match(r'^(?:[A-Za-z0-9_:<>&*,\s]+?)\s+([A-Za-z0-9_]+)::([A-Za-z0-9_]+)\s*\((.*?)\)', sline)
            if m:
                cls_name, name, args = m.groups()
                entries.append({
                    "file": rel_path,
                    "line": idx,
                    "symbol": f"{cls_name}::{name}",
                    "name": name,
                    "class": cls_name,
                    "args": args.strip(),
                    "is_definition": True
                })
            # Also check for free functions like Frot
            m_free = re.match(r'^(?:double|int|void|bool)\s+([A-Za-z0-9_]+)\s*\((.*?)\)', sline)
            if m_free:
                fname, args = m_free.groups()
                if fname not in ("if", "for", "while", "switch", "catch", "return"):
                    entries.append({
                        "file": rel_path,
                        "line": idx,
                        "symbol": f"::{fname}",
                        "name": fname,
                        "class": "free_function",
                        "args": args.strip(),
                        "is_definition": True
                    })

print(f"Extracted {len(entries)} raw symbol references from Wave 4 source files.")
out_json = Path("outputs/cantera-python/tools/parsed_raw_wave4_symbols.json")
out_json.write_text(json.dumps(entries, indent=2), encoding="utf-8")
