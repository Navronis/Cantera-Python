"""Diagnostic blocked DGETRF translation experiment for the saved 159x159 matrix."""
from pathlib import Path
import sys

import numpy as np


def blocked_lu_solve(matrix, rhs, block_size):
    a = np.asarray(matrix, dtype=float).tolist()
    n = len(a)
    pivots = []
    for j0 in range(0, n, block_size):
        j1 = min(j0 + block_size, n)
        # DGETF2 panel factorization, retaining updates within the panel.
        for k in range(j0, j1):
            pivot = max(range(k, n), key=lambda i: abs(a[i][k]))
            pivots.append(pivot)
            if pivot != k:
                a[k], a[pivot] = a[pivot], a[k]
            if a[k][k] == 0.0:
                raise ArithmeticError(f"zero pivot at {k}")
            for i in range(k + 1, n):
                a[i][k] /= a[k][k]
            for i in range(k + 1, n):
                multiplier = a[i][k]
                for j in range(k + 1, j1):
                    a[i][j] -= multiplier * a[k][j]
        # DTRSM: solve the block row against the unit-lower panel.
        for j in range(j1, n):
            for i in range(j0, j1):
                value = a[i][j]
                for k in range(j0, i):
                    value -= a[i][k] * a[k][j]
                a[i][j] = value
        # DGEMM: update the trailing block.
        for i in range(j1, n):
            for j in range(j1, n):
                product = 0.0
                for k in range(j0, j1):
                    product += a[i][k] * a[k][j]
                a[i][j] -= product
    x = [float(v) for v in rhs]
    for k, pivot in enumerate(pivots):
        if pivot != k:
            x[k], x[pivot] = x[pivot], x[k]
    for k in range(n):
        for i in range(k + 1, n):
            x[i] -= a[i][k] * x[k]
    for k in range(n - 1, -1, -1):
        x[k] /= a[k][k]
        for i in range(k):
            x[i] -= a[i][k] * x[k]
    return np.asarray(x)


def load(path):
    data = np.fromstring(path.read_text(encoding="utf-8"), sep=" ")
    nsp = int(data[0])
    n = 3 * nsp
    matrix = data[1:1 + n * n].reshape(n, n)
    rhs = data[1 + n * n:1 + n * n + n]
    solution = data[1 + n * n + n:1 + n * n + 2 * n]
    return matrix, rhs, solution


root = Path(__file__).resolve().parents[2] / "audit"
matrix, rhs, native = load(root / "wave4_internal_matrix_native.txt")
for block_size in (1, 16, 32, 64):
    solution = blocked_lu_solve(matrix, rhs, block_size)
    difference = np.abs(solution - native)
    relative = difference / np.maximum(np.maximum(np.abs(solution), np.abs(native)), 1e-300)
    residual = np.linalg.norm(matrix @ solution - rhs, ord=np.inf)
    print(block_size, "max_abs_to_native", difference.max(),
          "max_rel_to_native", relative.max(), "residual", residual)
