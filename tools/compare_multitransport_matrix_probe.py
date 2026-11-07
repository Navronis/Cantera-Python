"""Compare full pinned-native and Python MultiTransport block systems."""
import json
import sys
from pathlib import Path

import numpy as np


def load_probe(path: Path):
    values = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    nsp = int(values[0])
    size = 3 * nsp
    offset = 1
    matrix = values[offset:offset + size * size].reshape(size, size)
    offset += size * size
    rhs = values[offset:offset + size]
    offset += size
    solution = values[offset:offset + size]
    if len(solution) != size or not np.all(np.isfinite(values)):
        raise ValueError(f"Malformed or non-finite probe output: {path}")
    return matrix, rhs, solution


def errors(reference, candidate):
    delta = np.abs(reference - candidate)
    rel = delta / np.maximum(np.maximum(np.abs(reference), np.abs(candidate)), 1e-300)
    return {"max_abs": float(np.max(delta)), "max_rel": float(np.max(rel))}


def load_diagnostics(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, values = line.split(maxsplit=1)
        result[name] = np.fromstring(values, sep=" ")
    return result


def main():
    native_path = Path(sys.argv[1])
    python_path = Path(sys.argv[2])
    native = load_probe(native_path)
    python = load_probe(python_path)
    if native[0].shape != python[0].shape:
        raise ValueError(f"Matrix shape differs: {native[0].shape} vs {python[0].shape}")
    result = {
        "case": (sys.argv[3] if len(sys.argv) > 3 else
                 "GRI-3.0, 1200 K, 100 kPa, fixed 53-species composition"),
        "matrix": errors(native[0], python[0]),
        "rhs": errors(native[1], python[1]),
        "solution": errors(native[2], python[2]),
        "native_condition_2": float(np.linalg.cond(native[0])),
        "python_condition_2": float(np.linalg.cond(python[0])),
        "native_residual_inf": float(np.linalg.norm(native[0] @ native[2] - native[1], ord=np.inf)),
        "python_residual_inf": float(np.linalg.norm(python[0] @ python[2] - python[1], ord=np.inf)),
        "native_matrix_python_solution_residual_inf": float(
            np.linalg.norm(native[0] @ python[2] - native[1], ord=np.inf)),
        "python_matrix_native_solution_residual_inf": float(
            np.linalg.norm(python[0] @ native[2] - python[1], ord=np.inf)),
    }
    nsp = native[1].size // 3
    blocks = {}
    for i, row_name in enumerate(("00", "10", "01")):
        for j, col_name in enumerate(("00", "10", "01")):
            sl = (slice(i * nsp, (i + 1) * nsp),
                  slice(j * nsp, (j + 1) * nsp))
            blocks[f"L{row_name}_{col_name}"] = errors(
                native[0][sl], python[0][sl])
    result["matrix_blocks"] = blocks
    delta = np.abs(native[0] - python[0])
    worst = np.unravel_index(np.argmax(delta), delta.shape)
    result["worst_matrix_entry"] = {
        "row": int(worst[0]), "column": int(worst[1]),
        "native": float(native[0][worst]), "python": float(python[0][worst]),
    }
    native_diag = load_diagnostics(Path(str(native_path) + ".diag"))
    python_diag = load_diagnostics(Path(str(python_path) + ".diag"))
    inputs = {}
    for name in native_diag:
        native_values, python_values = native_diag[name], python_diag[name]
        record = errors(native_values, python_values)
        i = int(np.argmax(np.abs(native_values - python_values)))
        record["worst_index"] = i
        record["native_at_worst"] = float(native_values[i])
        record["python_at_worst"] = float(python_values[i])
        inputs[name] = record
    result["transport_inputs"] = inputs
    result["native_fit_vs_python_raw_viscosity"] = errors(
        native_diag["viscosity"], python_diag["viscosity_raw"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
