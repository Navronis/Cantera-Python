"""Compare Python BandMatrix against the pinned Cantera BandMatrix probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cantera_python.numerics import BandMatrix

CASES = ((6, 1, 1), (8, 2, 1), (10, 1, 3), (16, 3, 2),
         (4, 0, 2), (5, 2, 0), (30, 4, 3))


def coefficient(case_id: int, i: int, j: int, kl: int, ku: int) -> float:
    value = (((i + 2) * (j + 5) + case_id * 7) % 13 - 6) / 7.0
    if i == j:
        value += 3.0 + kl + ku
    if case_id == 0 and i == 0 and j == 0 and kl > 0:
        value = 0.0
    if case_id == 0 and i == 1 and j == 0:
        value = 1.0
    return value


def rhs_value(case_id: int, i: int, column: int) -> float:
    return ((i * 7 + column * 3 + case_id) % 11 - 5) / 3.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("native_output", nargs="?", type=Path,
                        default=ROOT / "audit" / "native_band_matrix_probe.txt")
    args = parser.parse_args()
    native_rows = [line.split() for line in args.native_output.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    if len(native_rows) != len(CASES):
        raise RuntimeError(f"Pinned probe returned {len(native_rows)} cases; expected {len(CASES)}")

    worst_solution = 0.0
    worst_residual = 0.0
    pivot_cases = 0
    for case_id, ((n, kl, ku), row) in enumerate(zip(CASES, native_rows)):
        observed_id, observed_n, observed_kl, observed_ku = map(int, row[:4])
        if (observed_id, observed_n, observed_kl, observed_ku) != (case_id, n, kl, ku):
            raise RuntimeError(f"Pinned probe case header mismatch: {row[:4]}")
        native = np.asarray([float(value) for value in row[4:]], dtype=float).reshape(n, 2)
        matrix = np.zeros((n, n), dtype=float)
        band = BandMatrix(n, kl, ku)
        for i in range(n):
            for j in range(max(0, i - kl), min(n, i + ku + 1)):
                value = coefficient(case_id, i, j, kl, ku)
                matrix[i, j] = value
                band[i, j] = value
        rhs = np.asarray([[rhs_value(case_id, i, col) for col in range(2)]
                          for i in range(n)], dtype=float)
        python = band.solve(rhs)
        difference = float(np.max(np.abs(python - native)))
        scale = max(1.0, float(np.max(np.abs(native))))
        relative = difference / scale
        residual = float(np.max(np.abs(matrix @ python - rhs)))
        worst_solution = max(worst_solution, relative)
        worst_residual = max(worst_residual, residual)
        pivot_cases += int(case_id == 0 and int(band._pivots[0]) == 1)
        if not np.allclose(python, native, rtol=2e-13, atol=2e-13):
            raise AssertionError(
                f"case {case_id} ({n},{kl},{ku}) differs: max abs={difference:.3e}"
            )
    print(f"Pinned BandMatrix cases: {len(CASES)}; forced pivot cases: {pivot_cases}")
    print(f"Max scaled solution difference: {worst_solution:.3e}")
    print(f"Max absolute equation residual: {worst_residual:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
