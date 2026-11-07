#!/usr/bin/env python3
"""
Generate Wave 4 Run-Level Provenance Manifest and derive wave4_run_id.

Cryptographically binds:
- Pinned native Cantera commit: 726522be4e2a13454d8415b7ef799d621f665cf3
- Native runner C++ source: tools/run_native_wave4.cpp
- Native compiled binary: tools/run_native_wave4
- Pinned native shared library: /home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4
- Corpus: audit/corpus_wave4.jsonl
- Python runner: tools/run_python_wave4_disk_chunked.py
- Python implementation: cantera_python/transport.py & cantera_python/solution.py
- Comparator: tools/compare_wave4_full_matrix.py
- Tolerance policy
"""

import hashlib
import json
import subprocess
from pathlib import Path

PINNED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"

def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def wsl_file_sha256(wsl_path: str) -> str:
    cmd = f"wsl sha256sum {wsl_path}"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to hash WSL file {wsl_path}: {res.stderr}")
    return res.stdout.split()[0]

def main():
    root = Path(__file__).resolve().parent.parent

    # File paths
    native_cpp = root / "tools" / "run_native_wave4.cpp"
    native_bin = root / "tools" / "run_native_wave4"
    corpus_file = root / "audit" / "corpus_wave4.jsonl"
    python_runner = root / "tools" / "run_python_wave4_disk_chunked.py"
    py_transport = root / "cantera_python" / "transport.py"
    py_solution = root / "cantera_python" / "solution.py"
    comparator = root / "tools" / "compare_wave4_full_matrix.py"
    wsl_lib = "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4"

    tolerances = {
        "viscosity": 1e-9,
        "sp_visc": 1e-9,
        "thermal_conductivity": 1e-9,
        "bdiff": 1e-9,
        "dmulti": 1e-8,
        "dt": 1e-7,
        "fluxes": 1e-8,
        "molar_fluxes": 1e-8,
    }

    manifest = {
        "pinned_native_commit": PINNED_COMMIT,
        "native_runner_cpp_sha256": file_sha256(native_cpp),
        "native_runner_binary_sha256": file_sha256(native_bin),
        "native_library_shared_sha256": wsl_file_sha256(wsl_lib),
        "corpus_sha256": file_sha256(corpus_file),
        "python_runner_sha256": file_sha256(python_runner),
        "python_transport_sha256": file_sha256(py_transport),
        "python_solution_sha256": file_sha256(py_solution),
        "comparator_sha256": file_sha256(comparator),
        "tolerance_policy": tolerances,
    }

    # Derive wave4_run_id from canonical json representation
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    manifest_hash = hashlib.sha256(serialized_manifest.encode("utf-8")).hexdigest()
    wave4_run_id = f"w4_{manifest_hash[:16]}"
    manifest["wave4_run_id"] = wave4_run_id

    out_file = root / "audit" / "solver_wave4_run_provenance.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 70)
    print("WAVE 4 PROVENANCE MANIFEST GENERATED")
    print("=" * 70)
    print(f"wave4_run_id: {wave4_run_id}")
    print(f"Manifest saved to: {out_file}")
    for k, v in manifest.items():
        if k != "tolerance_policy":
            print(f"  {k}: {v}")

    return wave4_run_id

if __name__ == "__main__":
    main()
