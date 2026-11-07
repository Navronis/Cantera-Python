"""Numerical linear algebra, banded solvers, Jacobians, and ODE integrators.

Provides:
- GeneralMatrix: Abstract base class for matrix operations.
- BandMatrix: Compact banded matrix storage with LU factorization and forward/back solve.
- DenseMatrix: Dense matrix storage with LU factorization, inversion, and condition numbers.
- SystemJacobian: Finite-difference and sparse block Jacobian evaluator.
- AdaptivePreconditioner: Block-diagonal and incomplete factorized preconditioners.
- FuncEval: Abstract ODE RHS evaluator.
- Integrator: Abstract base class for ODE/DAE integrators.
- CVodesIntegrator: Python variable-step BDF kernel (CVODE controller parity in progress).
- IdasIntegrator: DAE integrator for residual differential-algebraic systems.
"""

from __future__ import annotations
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np

from .constants import CanteraError


class NumericsError(CanteraError):
    """Exception for numerical algebra and integration errors."""
    pass


class _RecoverableRhsError(Exception):
    """A CanteraError raised while evaluating an integrator trial state."""


class GeneralMatrix:
    """Abstract base class for matrices."""

    def nRows(self) -> int:
        raise NotImplementedError

    def nColumns(self) -> int:
        raise NotImplementedError

    def factor(self) -> None:
        raise NotImplementedError

    def solve(self, b: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def mult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def leftMult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def zero(self) -> None:
        raise NotImplementedError

    def oneNorm(self) -> float:
        raise NotImplementedError

    def rcond(self, a1norm: float = 0.0) -> float:
        raise NotImplementedError

    def factorAlgorithm(self) -> int:
        return 0
    def _value(self, i: int, j: int) -> float:
        return self.value(i, j)

    def bfill(self, val: float = 0.0) -> None:
        self._data.fill(val)

    def checkColumns(self, n: int) -> bool:
        return n == self.nColumns()

    def checkRows(self, n: int) -> bool:
        return n == self.nRows()

    def info(self) -> int:
        return 0

    def ldim(self) -> int:
        return self._kl + self._ku + 1

    def operator(self, i: int, j: int) -> float:
        return self.value(i, j)
  # 0 for LU
    def factorQR(self) -> None:
        pass

    def rcondQR(self) -> float:
        return 1.0

    def useFactorAlgorithm(self, algo: int) -> None:
        pass

    def clearFactorFlag(self) -> None:
        pass

    def factored(self) -> bool:
        return True

    def col(self, j: int) -> np.ndarray:
        return np.zeros(self.nRows(), dtype=float)



class DenseMatrix(GeneralMatrix):
    def ipiv(self) -> np.ndarray:
        return getattr(self, '_ipiv', np.arange(self.nRows(), dtype=int))

    """Full dense matrix with LU decomposition, inversion, and linear solvers."""

    def __init__(self, n: int = 0, m: int = 0, v: float = 0.0):
        self._n = int(n)
        self._m = int(m) if m > 0 else int(n)
        self._data = np.full((self._n, self._m), float(v), dtype=float)
        self._factored = False
        self._lu: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._ipiv: Optional[np.ndarray] = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    def _clear_factorization(self) -> None:
        self._factored = False
        self._lu = None
        self._ipiv = None

    def appendColumn(self, col: np.ndarray) -> None:
        if self._data.size == 0:
            self._data = np.asarray(col, dtype=float).reshape(-1, 1)
        else:
            self._data = np.hstack([self._data, np.asarray(col, dtype=float).reshape(-1, 1)])
        self._n, self._m = self._data.shape
        self._clear_factorization()

    def setRow(self, i: int, row: np.ndarray) -> None:
        self._data[i, :] = row
        self._clear_factorization()

    def getRow(self, i: int, row: np.ndarray) -> None:
        row[:] = self._data[i, :]

    def setColumn(self, j: int, col: np.ndarray) -> None:
        self._data[:, j] = col
        self._clear_factorization()

    def getColumn(self, j: int, col: np.ndarray) -> None:
        col[:] = self._data[:, j]

    def col(self, j: int) -> np.ndarray:
        return self._data[:, j]

    def zero(self) -> None:
        self._data.fill(0.0)

    def operator(self, *args):
        return self._data

    def nRows(self) -> int:
        return self._n

    def nColumns(self) -> int:
        return self._m

    def resize(self, n: int, m: int, v: float = 0.0):
        self._n = int(n)
        self._m = int(m)
        self._data = np.full((self._n, self._m), float(v), dtype=float)
        self._clear_factorization()

    def value(self, i: int, j: int) -> float:
        return float(self._data[i, j])

    def setValue(self, i: int, j: int, v: float):
        self._data[i, j] = float(v)
        self._clear_factorization()

    def __getitem__(self, key: Union[Tuple[int, int], int]):
        if isinstance(key, tuple):
            i, j = key
            return self._data[i, j]
        return self._data[key]

    def __setitem__(self, key: Union[Tuple[int, int], int], val: Any):
        if isinstance(key, tuple):
            i, j = key
            self._data[i, j] = float(val)
        else:
            self._data[key] = val
        self._clear_factorization()

    def __call__(self, i: int, j: int) -> float:
        return float(self._data[i, j])

    def zero(self):
        self._data.fill(0.0)
        self._clear_factorization()

    def factor(self):
        """Factor a square matrix using Python partial-pivoting LU.

        The factorization follows the DGETRF convention: each pivot records a
        row interchange, and the strict lower triangle stores multipliers for
        a unit-diagonal L matrix while the upper triangle stores U.
        """
        if self._n == 0 or self._m == 0:
            return
        if self._n != self._m:
            raise NumericsError("LU factorization requires a square matrix")
        lu = [[float(self._data[i, j]) for j in range(self._n)]
              for i in range(self._n)]
        pivots = []
        for k in range(self._n):
            pivot = max(range(k, self._n), key=lambda i: abs(lu[i][k]))
            pivots.append(pivot)
            if lu[pivot][k] == 0.0:
                raise NumericsError(
                    f"LU factorization found an exactly singular pivot at column {k}"
                )
            if pivot != k:
                lu[k], lu[pivot] = lu[pivot], lu[k]
            diagonal = lu[k][k]
            for i in range(k + 1, self._n):
                multiplier = lu[i][k] / diagonal
                lu[i][k] = multiplier
                for j in range(k + 1, self._n):
                    lu[i][j] -= multiplier * lu[k][j]
        self._lu = np.asarray(lu, dtype=float)
        self._ipiv = np.asarray(pivots, dtype=int)
        self._factored = True

    @staticmethod
    def _solve_factored(lu: np.ndarray, pivots: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Solve one or more right-hand sides using Python forward/back substitution."""
        values = np.asarray(rhs, dtype=float)
        was_vector = values.ndim == 1
        if was_vector:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[0] != lu.shape[0]:
            raise NumericsError("Right-hand side dimensions do not match the factored matrix")
        result = [[float(values[i, col]) for col in range(values.shape[1])]
                  for i in range(values.shape[0])]
        n = lu.shape[0]
        for k, pivot in enumerate(pivots):
            pivot = int(pivot)
            if pivot != k:
                result[k], result[pivot] = result[pivot], result[k]
        for i in range(n):
            for j in range(i):
                coefficient = float(lu[i, j])
                for col in range(values.shape[1]):
                    result[i][col] -= coefficient * result[j][col]
        for i in range(n - 1, -1, -1):
            diagonal = float(lu[i, i])
            if diagonal == 0.0:
                raise NumericsError(f"LU solve found an exactly singular pivot at row {i}")
            for j in range(i + 1, n):
                coefficient = float(lu[i, j])
                for col in range(values.shape[1]):
                    result[i][col] -= coefficient * result[j][col]
            for col in range(values.shape[1]):
                result[i][col] /= diagonal
        answer = np.asarray(result, dtype=float)
        return answer[:, 0] if was_vector else answer

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A * x = b with the Python LU factorization."""
        if not self._factored or self._lu is None or self._ipiv is None:
            self.factor()
        return self._solve_factored(self._lu, self._ipiv, b)

    def invert(self) -> np.ndarray:
        """Compute the matrix inverse by solving for the identity columns."""
        if self._n != self._m:
            raise NumericsError("Matrix must be square to invert")
        if self._n == 0:
            return np.empty((0, 0), dtype=float)
        lu = self._data.copy()
        factor = DenseMatrix(self._n, self._n)
        factor._data[:, :] = lu
        factor.factor()
        return self._solve_factored(
            factor._lu, factor._ipiv, np.eye(self._n, dtype=float)
        )

    def mult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix-vector multiply: prod = A * b."""
        b_arr = np.asarray(b, dtype=float)
        res = self._data @ b_arr
        if prod is not None:
            prod[:] = res
            return prod
        return res

    def leftMult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        """Vector-matrix multiply: prod = b * A (or A^T * b)."""
        b_arr = np.asarray(b, dtype=float)
        res = b_arr @ self._data
        if prod is not None:
            prod[:] = res
            return prod
        return res

    def oneNorm(self) -> float:
        return _matrix_one_norm(self._data)

    def rcond(self, a1norm: float = 0.0) -> float:
        try:
            return _reciprocal_condition_one_norm(self._data, a1norm)
        except Exception:
            return 0.0


def _factor_dense_matrix(values: np.ndarray) -> DenseMatrix:
    """Copy and factor a dense matrix with the package's Python LU kernel."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NumericsError("Dense factorization requires a square matrix")
    result = DenseMatrix(*matrix.shape)
    result._data[:, :] = matrix
    result.factor()
    return result


def _matrix_one_norm(values: np.ndarray) -> float:
    """Matrix 1-norm computed with Python summation, not a native linalg call."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 0.0
    return max(math.fsum(abs(float(matrix[i, j])) for i in range(matrix.shape[0]))
               for j in range(matrix.shape[1]))


def _python_l2_norm(values: np.ndarray) -> float:
    """Stable Euclidean norm using Python's scaled hypot operation."""
    norm = 0.0
    for value in np.asarray(values, dtype=float).reshape(-1):
        norm = math.hypot(norm, float(value))
    return norm


def _reciprocal_condition_one_norm(values: np.ndarray, a1norm: float = 0.0) -> float:
    """Compute reciprocal 1-norm condition using the Python LU solver."""
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NumericsError("Condition estimate requires a square matrix")
    norm = float(a1norm) if a1norm > 0.0 else _matrix_one_norm(matrix)
    if norm == 0.0:
        return 0.0
    factor = _factor_dense_matrix(matrix)
    inverse = factor.solve(np.eye(matrix.shape[0], dtype=float))
    inverse_norm = _matrix_one_norm(inverse)
    product = norm * inverse_norm
    return 0.0 if not math.isfinite(product) or product == 0.0 else 1.0 / product


def _householder_rank_and_least_squares(
        values: np.ndarray, rhs: Optional[np.ndarray] = None, *,
        rcond: Optional[float] = None, tolerance: Optional[float] = None
        ) -> Tuple[int, Optional[np.ndarray]]:
    """Rank-revealing column-pivoted Householder QR using Python scalar loops.

    The least-squares solve returns a pivoted basic solution for rank-deficient
    systems. The converted call sites use a square Newton system or a tall
    element-potential system; underdetermined minimum-norm solves are rejected
    instead of silently returning a different solution convention.
    """
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise NumericsError("QR factorization requires a two-dimensional matrix")
    nrows, ncols = matrix.shape
    if rhs is not None:
        target = np.asarray(rhs, dtype=float)
        if target.ndim != 1 or target.shape[0] != nrows:
            raise NumericsError("QR right-hand side dimensions do not match the matrix")
        if nrows < ncols:
            return nrows, _least_squares_householder_wide(matrix, target, rcond)
        transformed_rhs = [float(value) for value in target]
    else:
        transformed_rhs = None

    work = [[float(matrix[i, j]) for j in range(ncols)] for i in range(nrows)]
    permutation = list(range(ncols))
    leading_norm = 0.0
    rank = 0
    max_steps = min(nrows, ncols)
    for k in range(max_steps):
        pivot = max(
            range(k, ncols),
            key=lambda j: math.fsum(work[i][j] * work[i][j]
                                    for i in range(k, nrows)),
        )
        if pivot != k:
            for row in work:
                row[k], row[pivot] = row[pivot], row[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]

        norm = math.sqrt(math.fsum(work[i][k] * work[i][k]
                                   for i in range(k, nrows)))
        if k == 0:
            leading_norm = norm
        if tolerance is not None:
            threshold = float(tolerance)
        else:
            relative = (float(rcond) if rcond is not None
                        else np.finfo(float).eps * max(nrows, ncols))
            threshold = relative * leading_norm
        if not math.isfinite(norm) or norm <= threshold:
            break

        alpha = -math.copysign(norm, work[k][k])
        reflector = [work[i][k] for i in range(k, nrows)]
        reflector[0] -= alpha
        denominator = math.fsum(value * value for value in reflector)
        if denominator == 0.0 or not math.isfinite(denominator):
            break
        factor = 2.0 / denominator
        for j in range(k + 1, ncols):
            projection = factor * math.fsum(
                reflector[offset] * work[k + offset][j]
                for offset in range(len(reflector)))
            for offset, value in enumerate(reflector):
                work[k + offset][j] -= projection * value
        if transformed_rhs is not None:
            projection = factor * math.fsum(
                reflector[offset] * transformed_rhs[k + offset]
                for offset in range(len(reflector)))
            for offset, value in enumerate(reflector):
                transformed_rhs[k + offset] -= projection * value
        work[k][k] = alpha
        for i in range(k + 1, nrows):
            work[i][k] = 0.0
        rank += 1

    if transformed_rhs is None:
        return rank, None

    pivoted = [0.0] * ncols
    for i in range(rank - 1, -1, -1):
        remainder = transformed_rhs[i] - math.fsum(
            work[i][j] * pivoted[j] for j in range(i + 1, ncols))
        pivoted[i] = remainder / work[i][i]
    solution = [0.0] * ncols
    for pivoted_index, original_index in enumerate(permutation):
        solution[original_index] = pivoted[pivoted_index]
    return rank, np.asarray(solution, dtype=float)


def _least_squares_householder_wide(matrix: np.ndarray, rhs: np.ndarray,
                                    rcond: Optional[float]) -> np.ndarray:
    """Minimum-norm solution for a full-row-rank underdetermined system.

    QR is applied to the transpose. Column pivoting reorders the equations;
    solving R.T z = P.T b followed by x = Q z gives the minimum-norm solution.
    Rank-deficient wide systems are rejected explicitly until their SVD-style
    minimum-norm semantics are implemented.
    """
    nrows, ncols = matrix.shape
    work = [[float(matrix[j, i]) for j in range(nrows)] for i in range(ncols)]
    permutation = list(range(nrows))
    reflectors = []
    leading_norm = 0.0
    relative = (float(rcond) if rcond is not None
                else np.finfo(float).eps * max(nrows, ncols))
    for k in range(nrows):
        pivot = max(
            range(k, nrows),
            key=lambda j: math.fsum(work[i][j] * work[i][j]
                                    for i in range(k, ncols)),
        )
        if pivot != k:
            for row in work:
                row[k], row[pivot] = row[pivot], row[k]
            permutation[k], permutation[pivot] = permutation[pivot], permutation[k]

        norm = math.sqrt(math.fsum(work[i][k] * work[i][k]
                                   for i in range(k, ncols)))
        if k == 0:
            leading_norm = norm
        if not math.isfinite(norm):
            raise NumericsError("minimum-norm QR encountered a non-finite column")
        if norm <= relative * leading_norm:
            return _minimum_norm_wide_least_squares(matrix, rhs, relative)
        alpha = -math.copysign(norm, work[k][k])
        reflector = [work[i][k] for i in range(k, ncols)]
        reflector[0] -= alpha
        denominator = math.fsum(value * value for value in reflector)
        if denominator == 0.0 or not math.isfinite(denominator):
            raise NumericsError("degenerate minimum-norm QR reflector")
        factor = 2.0 / denominator
        for j in range(k + 1, nrows):
            projection = factor * math.fsum(
                reflector[offset] * work[k + offset][j]
                for offset in range(len(reflector)))
            for offset, value in enumerate(reflector):
                work[k + offset][j] -= projection * value
        work[k][k] = alpha
        for i in range(k + 1, ncols):
            work[i][k] = 0.0
        reflectors.append((k, reflector, factor))

    # Forward substitution in the lower-triangular R.T system.
    permuted_rhs = [float(rhs[permutation[i]]) for i in range(nrows)]
    z = [0.0] * nrows
    for i in range(nrows):
        remainder = permuted_rhs[i] - math.fsum(
            work[j][i] * z[j] for j in range(i))
        z[i] = remainder / work[i][i]

    # Apply Q = H0 H1 ... H(m-1) to [z, 0], applying reflectors in reverse.
    result = z + [0.0] * (ncols - nrows)
    for k, reflector, factor in reversed(reflectors):
        projection = factor * math.fsum(
            reflector[offset] * result[k + offset]
            for offset in range(len(reflector)))
        for offset, value in enumerate(reflector):
            result[k + offset] -= projection * value
    return np.asarray(result, dtype=float)


def _minimum_norm_wide_least_squares(matrix: np.ndarray, rhs: np.ndarray,
                                     relative_tolerance: float) -> np.ndarray:
    """Rank-deficient wide minimum-norm solve using a Python Jacobi eigensolver.

    The symmetric Gram matrix is small in the intended VCS element-potential
    use case. Its eigenvectors form the left singular vectors of A; discarding
    eigenvalues below the squared relative singular-value threshold gives the
    Moore-Penrose solution without a native SVD call.
    """
    nrows, ncols = matrix.shape
    gram = [[math.fsum(float(matrix[i, k]) * float(matrix[j, k])
                       for k in range(ncols)) for j in range(nrows)]
            for i in range(nrows)]
    eigenvectors = [[1.0 if i == j else 0.0 for j in range(nrows)]
                    for i in range(nrows)]
    for _ in range(max(1, 50 * nrows * nrows)):
        p, q, largest = 0, 0, 0.0
        diagonal_scale = max((abs(gram[i][i]) for i in range(nrows)), default=0.0)
        for i in range(nrows):
            for j in range(i + 1, nrows):
                magnitude = abs(gram[i][j])
                if magnitude > largest:
                    p, q, largest = i, j, magnitude
        if largest <= 4.0 * np.finfo(float).eps * diagonal_scale:
            break

        apq = gram[p][q]
        theta = 0.5 * (gram[q][q] - gram[p][p]) / apq
        tangent = (1.0 if theta >= 0.0 else -1.0) / (
            abs(theta) + math.sqrt(1.0 + theta * theta))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        tau = sine / (1.0 + cosine)
        shift = tangent * apq
        gram[p][p] -= shift
        gram[q][q] += shift
        gram[p][q] = gram[q][p] = 0.0
        for k in range(nrows):
            if k != p and k != q:
                gkp, gkq = gram[k][p], gram[k][q]
                gram[k][p] = gram[p][k] = gkp - sine * (gkq + tau * gkp)
                gram[k][q] = gram[q][k] = gkq + sine * (gkp - tau * gkq)
            vkp, vkq = eigenvectors[k][p], eigenvectors[k][q]
            eigenvectors[k][p] = vkp - sine * (vkq + tau * vkp)
            eigenvectors[k][q] = vkq + sine * (vkp - tau * vkq)
    else:
        raise NumericsError("Jacobi eigensolver did not converge for rank-deficient QR")

    largest_eigenvalue = max((max(0.0, gram[i][i]) for i in range(nrows)), default=0.0)
    cutoff = (relative_tolerance * relative_tolerance) * largest_eigenvalue
    transformed = [0.0] * nrows
    for j in range(nrows):
        eigenvalue = gram[j][j]
        if eigenvalue <= cutoff or eigenvalue <= 0.0:
            continue
        projection = math.fsum(eigenvectors[i][j] * float(rhs[i])
                               for i in range(nrows))
        scale = projection / eigenvalue
        for i in range(nrows):
            transformed[i] += eigenvectors[i][j] * scale

    return np.asarray([
        math.fsum(float(matrix[i, j]) * transformed[i] for i in range(nrows))
        for j in range(ncols)
    ], dtype=float)


def _matrix_rank_householder(values: np.ndarray, tolerance: float) -> int:
    """Return numerical rank with an absolute tolerance using Python QR."""
    rank, _ = _householder_rank_and_least_squares(
        values, tolerance=float(tolerance))
    return rank


def _least_squares_householder(values: np.ndarray, rhs: np.ndarray,
                               rcond: Optional[float] = None) -> np.ndarray:
    """Solve a tall or square least-squares system using Python pivoted QR."""
    _, solution = _householder_rank_and_least_squares(
        values, rhs, rcond=rcond)
    assert solution is not None
    return solution


class BandMatrix(GeneralMatrix):
    """Compact banded matrix storage compatible with LAPACK and SUNDIALS band solvers.

    Storage layout:
    - Square matrix size: n x n
    - Lower bandwidth: kl
    - Upper bandwidth: ku
    - Storage dimensions: (ldim, n) where ldim = 2*kl + ku + 1.
    - LAPACK band storage reserves kl additional rows for pivot-induced fill.
    - Element A(i, j) within band is at row: kl + ku + i - j, column: j.
    - Flat 1D index (column-major): (2*kl + ku)*j + kl + ku + i.
    """

    def __init__(self, n: int = 0, kl: int = 0, ku: int = 0, v: float = 0.0):
        self._n = int(n)
        self._kl = int(kl)
        self._ku = int(ku)
        self._nrows_alloc = 2 * self._kl + self._ku + 1
        self._data = np.full((self._nrows_alloc, self._n), float(v), dtype=float, order="F")
        self._ludata = self._data.copy(order="F")
        self._factored = False
        self._pivots: Optional[np.ndarray] = None
        self._info: int = 0

    def nRows(self) -> int:
        return self._n

    def nColumns(self) -> int:
        return self._n

    def nSubDiagonals(self) -> int:
        return self._kl

    def nSuperDiagonals(self) -> int:
        return self._ku

    def ldim(self) -> int:
        return self._nrows_alloc

    def resize(self, n: int, kl: int, ku: int, v: float = 0.0):
        self._n = int(n)
        self._kl = int(kl)
        self._ku = int(ku)
        self._nrows_alloc = 2 * self._kl + self._ku + 1
        self._data = np.full((self._nrows_alloc, self._n), float(v), dtype=float, order="F")
        self._ludata = self._data.copy(order="F")
        self._factored = False
        self._pivots = None
        self._info = 0

    def bfill(self, v: float = 0.0):
        self._data.fill(float(v))
        self._factored = False
        self._pivots = None
        self._info = 0

    def zero(self):
        self.bfill(0.0)

    def index(self, i: int, j: int) -> int:
        """Returns the location in the internal 1D array corresponding to entry (i, j)."""
        return (2 * self._kl + self._ku) * j + self._kl + self._ku + i

    def _band_row(self, i: int, j: int) -> int:
        return self._kl + self._ku + i - j

    def _value(self, i: int, j: int) -> float:
        """Return the value of element (i, j) within bandwidth without range checks."""
        return float(self._data[self._kl + self._ku + i - j, j])

    def value(self, i: int, j: int) -> float:
        """Return the value of element (i, j) or 0.0 if outside the band."""
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Indices ({i}, {j}) out of range for {self._n}x{self._n} matrix")
        if j > i + self._ku or i > j + self._kl:
            return 0.0
        return float(self._data[self._kl + self._ku + i - j, j])

    def setValue(self, i: int, j: int, v: float):
        """Set element (i, j) to value v."""
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Indices ({i}, {j}) out of range for {self._n}x{self._n} matrix")
        if j > i + self._ku or i > j + self._kl:
            return
        self._data[self._kl + self._ku + i - j, j] = float(v)
        self._factored = False
        self._pivots = None

    def __getitem__(self, key: Tuple[int, int]) -> float:
        i, j = key
        return self.value(i, j)

    def __setitem__(self, key: Tuple[int, int], val: float):
        i, j = key
        self.setValue(i, j, float(val))

    def __call__(self, i: int, j: int) -> float:
        return self.value(i, j)

    def to_dense(self) -> np.ndarray:
        """Convert banded representation to a full dense NxN NumPy matrix."""
        dense = np.zeros((self._n, self._n), dtype=float)
        for j in range(self._n):
            for i in range(max(0, j - self._ku), min(self._n, j + self._kl + 1)):
                dense[i, j] = self._value(i, j)
        return dense

    @classmethod
    def from_dense(cls, dense: np.ndarray, kl: int, ku: int) -> "BandMatrix":
        """Construct a BandMatrix from an NxN dense NumPy array."""
        arr = np.asarray(dense, dtype=float)
        n = arr.shape[0]
        if arr.shape != (n, n):
            raise ValueError(f"Dense matrix must be square, got {arr.shape}")
        mat = cls(n, kl, ku)
        for j in range(n):
            for i in range(max(0, j - ku), min(n, j + kl + 1)):
                mat.setValue(i, j, arr[i, j])
        return mat

    def mult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        """Matrix-vector multiplication: prod = A * b."""
        b_arr = np.asarray(b, dtype=float)
        if len(b_arr) != self._n:
            raise CanteraError("BandMatrix::mult", f"Array size mismatch: b has {len(b_arr)}, expected {self._n}")
        if self._n == 0:
            res = np.empty(0, dtype=float)
            if prod is not None:
                prod[:] = res
                return prod
            return res

        try:
            import scipy.linalg.blas as blas
            ab_band = np.asfortranarray(self._data[self._kl:])
            b_f = np.ascontiguousarray(b_arr, dtype=float)
            res = blas.dgbmv(self._n, self._n, self._kl, self._ku, 1.0, ab_band, b_f, trans=0)
        except Exception:
            res = np.zeros(self._n, dtype=float)
            for m in range(self._n):
                val_sum = 0.0
                start = max(0, m - self._kl)
                stop = min(m + self._ku + 1, self._n)
                for j in range(start, stop):
                    val_sum += self._value(m, j) * b_arr[j]
                res[m] = val_sum

        if prod is not None:
            prod_arr = np.asarray(prod)
            if len(prod_arr) != self._n:
                raise CanteraError("BandMatrix::mult", f"Array size mismatch: prod has {len(prod_arr)}, expected {self._n}")
            prod[:] = res
            return prod
        return res

    def leftMult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        """Transposed matrix-vector multiplication: prod = b * A."""
        b_arr = np.asarray(b, dtype=float)
        if len(b_arr) != self._n:
            raise CanteraError("BandMatrix::leftMult", f"Array size mismatch: b has {len(b_arr)}, expected {self._n}")
        if self._n == 0:
            res = np.empty(0, dtype=float)
            if prod is not None:
                prod[:] = res
                return prod
            return res

        try:
            import scipy.linalg.blas as blas
            ab_band = np.asfortranarray(self._data[self._kl:])
            b_f = np.ascontiguousarray(b_arr, dtype=float)
            res = blas.dgbmv(self._n, self._n, self._kl, self._ku, 1.0, ab_band, b_f, trans=1)
        except Exception:
            res = np.zeros(self._n, dtype=float)
            for n_col in range(self._n):
                val_sum = 0.0
                start = max(0, n_col - self._ku)
                stop = min(n_col + self._kl + 1, self._n)
                for i in range(start, stop):
                    val_sum += self._value(i, n_col) * b_arr[i]
                res[n_col] = val_sum

        if prod is not None:
            prod_arr = np.asarray(prod)
            if len(prod_arr) != self._n:
                raise CanteraError("BandMatrix::leftMult", f"Array size mismatch: prod has {len(prod_arr)}, expected {self._n}")
            prod[:] = res
            return prod
        return res

    def _lu_value(self, i: int, j: int) -> float:
        """Read the expanded band used by LU, including pivot-created fill."""
        row = self._kl + self._ku + i - j
        if row < 0 or row >= self._nrows_alloc:
            return 0.0
        return float(self._ludata[row, j])

    def _set_lu_value(self, i: int, j: int, value: float) -> None:
        row = self._kl + self._ku + i - j
        if row < 0 or row >= self._nrows_alloc:
            if value != 0.0:
                raise NumericsError("Banded LU fill exceeded allocated storage")
            return
        self._ludata[row, j] = value

    def _factor_python(self):
        """Pure-Python fallback for banded LU decomposition with partial pivoting."""
        self._ludata = self._data.copy(order="F")
        pivots = []
        upper_fill = self._kl + self._ku
        last_active_column = -1
        for k in range(self._n):
            last_row = min(self._n - 1, k + self._kl)
            pivot = max(range(k, last_row + 1),
                        key=lambda i: abs(self._lu_value(i, k)))
            pivots.append(pivot)
            pivot_value = self._lu_value(pivot, k)
            if pivot_value == 0.0:
                self._info = k + 1
                raise NumericsError(
                    f"Factorization failed with DGBTRF error code {self._info}: "
                    f"exactly singular pivot at column {k}"
                )

            last_active_column = max(
                last_active_column,
                min(self._n - 1, k + self._ku + (pivot - k)),
            )
            if pivot != k:
                for j in range(k, last_active_column + 1):
                    upper = self._lu_value(k, j)
                    lower = self._lu_value(pivot, j)
                    self._set_lu_value(k, j, lower)
                    self._set_lu_value(pivot, j, upper)

            diagonal = self._lu_value(k, k)
            for i in range(k + 1, last_row + 1):
                multiplier = self._lu_value(i, k) / diagonal
                self._set_lu_value(i, k, multiplier)
                for j in range(k + 1, last_active_column + 1):
                    updated = self._lu_value(i, j) - multiplier * self._lu_value(k, j)
                    self._set_lu_value(i, j, updated)

        self._pivots = np.asarray(pivots, dtype=np.int32)
        self._factored = True
        self._info = 0

    def factor(self):
        """Factor the band matrix using LAPACK DGBTRF with partial pivoting."""
        if self._n == 0:
            self._ludata = self._data.copy(order="F")
            self._pivots = np.empty(0, dtype=np.int32)
            self._factored = True
            self._info = 0
            return

        try:
            import scipy.linalg.lapack as lapack
            ab = np.asfortranarray(self._data.copy(order="F"))
            lu, ipiv, info = lapack.dgbtrf(ab, self._kl, self._ku)
            self._info = int(info)
            if self._info != 0:
                raise NumericsError(
                    f"Factorization failed with DGBTRF error code {self._info}: "
                    f"exactly singular pivot at column {self._info - 1}"
                )
            self._ludata = lu
            self._pivots = ipiv
            self._factored = True
        except ImportError:
            self._factor_python()

    def _solve_python(self, b_mat: np.ndarray) -> np.ndarray:
        """Pure-Python fallback for banded linear solve."""
        result = [[float(b_mat[i, col]) for col in range(b_mat.shape[1])]
                  for i in range(self._n)]
        upper_fill = self._kl + self._ku

        # Apply pivots and solve L*y = P*b
        for k, pivot in enumerate(self._pivots):
            pivot = int(pivot)
            if pivot != k:
                result[k], result[pivot] = result[pivot], result[k]
            for i in range(k + 1, min(self._n, k + self._kl + 1)):
                multiplier = self._lu_value(i, k)
                for col in range(b_mat.shape[1]):
                    result[i][col] -= multiplier * result[k][col]

        # Back-substitute through U
        for i in range(self._n - 1, -1, -1):
            diagonal = self._lu_value(i, i)
            if diagonal == 0.0:
                raise NumericsError(f"Banded LU solve found an exactly singular pivot at row {i}")
            for j in range(i + 1, min(self._n, i + upper_fill + 1)):
                coefficient = self._lu_value(i, j)
                for col in range(b_mat.shape[1]):
                    result[i][col] -= coefficient * result[j][col]
            for col in range(b_mat.shape[1]):
                result[i][col] /= diagonal

        return np.asarray(result, dtype=float)

    def solve(self, b: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Solve one or more right-hand sides using the banded LU factors.

        Signatures supported:
        - solve(b): returns solution x
        - solve(b, x): writes solution into x and returns x
        - solve(b, nrhs, ldb): in-place solve on b
        """
        if not self._factored or self._pivots is None:
            self.factor()

        out_x = None
        if len(args) >= 1 and isinstance(args[0], np.ndarray):
            out_x = args[0]

        b_arr = np.asarray(b, dtype=float)
        was_vector = b_arr.ndim == 1
        if was_vector:
            if len(b_arr) != self._n:
                raise CanteraError("BandMatrix::solve", f"RHS vector dimension {len(b_arr)} does not match matrix size {self._n}")
            b_mat = b_arr.reshape(-1, 1)
        else:
            if b_arr.shape[0] != self._n:
                raise CanteraError("BandMatrix::solve", f"RHS matrix rows {b_arr.shape[0]} do not match matrix size {self._n}")
            b_mat = b_arr

        try:
            import scipy.linalg.lapack as lapack
            b_f = np.asfortranarray(b_mat.copy())
            lu_f = np.asfortranarray(self._ludata)
            x_sol, info = lapack.dgbtrs(lu_f, self._kl, self._ku, b_f, self._pivots)
            self._info = int(info)
            if self._info != 0:
                raise CanteraError("BandMatrix::solve", f"Linear solve failed with DGBTRS error code {self._info}.")
            result = x_sol[:, 0] if was_vector else x_sol
        except ImportError:
            result = self._solve_python(b_mat)[:, 0] if was_vector else self._solve_python(b_mat)

        if out_x is not None:
            out_x[:] = result
            return out_x
        return result

    def oneNorm(self) -> float:
        """Compute the 1-norm (maximum absolute column sum) in O(N*(kl+ku))."""
        val_max = 0.0
        for j in range(self._n):
            start = max(0, j - self._ku)
            stop = min(self._n, j + self._kl + 1)
            col_sum = 0.0
            for i in range(start, stop):
                col_sum += abs(self._value(i, j))
            val_max = max(val_max, col_sum)
        return float(val_max)

    def rcond(self, a1norm: float = 0.0) -> float:
        """Return reciprocal 1-norm condition number estimate using LAPACK DGBCON."""
        if not self._factored:
            self.factor()
        if a1norm <= 0.0:
            a1norm = self.oneNorm()
        if a1norm == 0.0:
            return 0.0
        try:
            import scipy.linalg.lapack as lapack
            lu_f = np.asfortranarray(self._ludata)
            rcond_val, rinfo = lapack.dgbcon(self._kl, self._ku, lu_f, self._pivots, a1norm, norm=b"1")
            if rinfo != 0:
                raise CanteraError("BandMatrix::rcond", f"DGBCON returned INFO = {rinfo}")
            return float(rcond_val)
        except Exception:
            dense = self.to_dense()
            return _reciprocal_condition_one_norm(dense, a1norm)

    def factorAlgorithm(self) -> int:
        """Factor algorithm used: returns 0 (LU) for band matrices."""
        return 0

    def info(self) -> int:
        """LAPACK info flag after last factor/solve operation."""
        return int(self._info)

    def checkRows(self, valueSmall: Optional[float] = None) -> Tuple[int, float]:
        """Check for zero/near-zero rows. Returns (row_index, max_entry_in_that_row)."""
        val_small = 1.0e300
        i_small = -1
        for i in range(self._n):
            start = max(0, i - self._kl)
            stop = min(self._n, i + self._ku + 1)
            val_s = 0.0
            for j in range(start, stop):
                val_s = max(abs(self._value(i, j)), val_s)
            if val_s < val_small:
                i_small = i
                val_small = val_s
                if val_small == 0.0:
                    return (i_small, val_small)
        return (i_small, val_small)

    def checkColumns(self, valueSmall: Optional[float] = None) -> Tuple[int, float]:
        """Check for zero/near-zero columns. Returns (col_index, max_entry_in_that_col)."""
        val_small = 1.0e300
        j_small = -1
        for j in range(self._n):
            start = max(0, j - self._ku)
            stop = min(self._n, j + self._kl + 1)
            val_s = 0.0
            for i in range(start, stop):
                val_s = max(abs(self._value(i, j)), val_s)
            if val_s < val_small:
                j_small = j
                val_small = val_s
                if val_small == 0.0:
                    return (j_small, val_small)
        return (j_small, val_small)

    def copy(self) -> "BandMatrix":
        mat = BandMatrix(self._n, self._kl, self._ku)
        mat._data = self._data.copy(order="F")
        mat._ludata = self._ludata.copy(order="F")
        mat._factored = self._factored
        mat._info = self._info
        if self._pivots is not None:
            mat._pivots = self._pivots.copy()
        return mat

    def __str__(self) -> str:
        lines = []
        for i in range(self.nRows()):
            row_vals = [f"{self.value(i, j):.17g}" for j in range(self.nColumns())]
            lines.append(", ".join(row_vals))
        return "\n".join(lines) + "\n"

    def m(self) -> str:
        return str(self)


class FuncEval:
    def __init__(self):
        self._errors: List[str] = []
        self._suppress_errors = False

    def clearErrors(self) -> None:
        self._errors = []

    def evalNoThrow(self, t: float, y: np.ndarray, ydot: np.ndarray) -> bool:
        try:
            self.eval(t, y, ydot)
            return True
        except Exception:
            return False

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray,
                residual: np.ndarray) -> None:
        raise NotImplementedError("FuncEval::evalDae")

    def evalDaeNoThrow(self, t: float, y: np.ndarray, ydot: np.ndarray,
                       residual: np.ndarray) -> int:
        """Evaluate an IDAS residual using Cantera's 0/1/-1 status contract."""
        try:
            self.evalDae(t, y, ydot, residual)
        except CanteraError as err:
            if getattr(self, "_suppress_errors", False):
                self._errors = getattr(self, "_errors", [])
                self._errors.append(str(err))
            return 1
        except Exception as err:
            if getattr(self, "_suppress_errors", False):
                self._errors = getattr(self, "_errors", [])
                self._errors.append(str(err))
            return -1
        return 0

    def evalRootFunctions(self, t: float, y: np.ndarray, gout: np.ndarray) -> None:
        pass

    def evalRootFunctionsNoThrow(self, t: float, y: np.ndarray, gout: np.ndarray) -> bool:
        return True

    def getConstraints(self, c: np.ndarray) -> None:
        raise NotImplementedError("FuncEval::getConstraints")

    def getErrors(self) -> list:
        return getattr(self, '_errors', [])

    def getState(self, y: np.ndarray) -> None:
        pass

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        pass

    def nRootFunctions(self) -> int:
        return 0

    def nparams(self) -> int:
        return 0

    def preconditionerSetup(self, t: float, y: np.ndarray, gamma: float) -> None:
        pass

    def preconditionerSolve(self, rhs: np.ndarray, output: np.ndarray) -> None:
        output[:] = rhs

    def preconditioner_setup_nothrow(self, t: float, y: np.ndarray, gamma: float) -> bool:
        return True

    def preconditioner_solve_nothrow(self, rhs: np.ndarray, output: np.ndarray) -> bool:
        output[:] = rhs
        return True

    def suppressErrors(self, suppress: bool = True) -> None:
        self._suppress_errors = bool(suppress)

    def updatePreconditioner(self, gamma: float, atol: float) -> None:
        pass

    """Abstract ODE/DAE right-hand side function evaluator."""

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray) -> None:
        raise NotImplementedError

    def eval_f(self, t: float, y: np.ndarray, ydot: np.ndarray) -> None:
        self.eval(t, y, ydot)

    def neq(self) -> int:
        raise NotImplementedError

    def nEquations(self) -> int:
        return self.neq()

    def getState(self, y: np.ndarray) -> None:
        pass

    def getInitialConditions(self, t0: float, len_y: int, y: np.ndarray) -> None:
        self.getState(y)


class SystemJacobian:
    def initialize(self, n: int = 0) -> None:
        if n > 0:
            self._n = int(n)
            self._mat = np.zeros((self._n, self._n), dtype=float)

    def reset(self) -> None:
        if hasattr(self, '_mat') and isinstance(self._mat, np.ndarray):
            self._mat.fill(0.0)

    def updatePreconditioner(self, gamma: float = 1.0, atol: float = 1e-12) -> None:
        pass

    def updateTransient(self, gamma: float = 1.0, ydot: Optional[np.ndarray] = None) -> None:
        pass

    def nEvals(self) -> int:
        return getattr(self, '_evals', 0)

    def age(self) -> int:
        return getattr(self, '_age', 0)

    def setAge(self, age: int) -> None:
        self._age = int(age)

    def incrementAge(self) -> None:
        self._age = getattr(self, '_age', 0) + 1

    def clearStats(self) -> None:
        self._evals = 0
        self._age = 0
        self._elapsed = 0.0

    def elapsedTime(self) -> float:
        return getattr(self, '_elapsed', 0.0)

    def updateElapsed(self, dt: float) -> None:
        self._elapsed = getattr(self, '_elapsed', 0.0) + float(dt)

    def gamma(self) -> float:
        return getattr(self, '_gamma', 1.0)

    def setGamma(self, gamma: float) -> None:
        self._gamma = float(gamma)

    def incrementEvals(self) -> None:
        self._evals = getattr(self, '_evals', 0) + 1

    def info(self) -> int:
        return 0

    def preconditionerSide(self) -> str:
        return getattr(self, '_side', 'none')

    def setPreconditionerSide(self, side: str) -> None:
        self._side = str(side)

    def printPreconditioner(self) -> None:
        pass

    def setAbsoluteTolerance(self, atol: float) -> None:
        self._atol = float(atol)

    def setBandwidth(self, kl: int, ku: int) -> None:
        self._kl = int(kl)
        self._ku = int(ku)

    def setValue(self, i: int, j: int, val: float) -> None:
        if hasattr(self, '_matrix') and self._matrix is not None:
            self._matrix[i, j] = float(val)

    def stateAdjustment(self, y: np.ndarray) -> None:
        pass

    def factorize(self) -> None:
        pass

    """Finite-difference and block Jacobian evaluator for reactive systems."""

    def __init__(self, sys: Optional[Any] = None, n: int = 0):
        if sys is not None and not isinstance(sys, (int, float)):
            self._sys = sys
            if hasattr(sys, "neq"):
                self._n = sys.neq()
            elif hasattr(sys, "nEquations"):
                self._n = sys.nEquations()
            else:
                self._n = int(n)
        else:
            self._sys = None
            self._n = int(sys) if sys is not None else int(n)
        self._evals = 0
        self._precon_side = "none"
        self._mat = np.zeros((self._n, self._n), dtype=float)

    def type(self) -> str:
        return "Dense"

    def preconditionerSide(self) -> str:
        return self._precon_side

    def setPreconditionerSide(self, preconSide: str):
        self._precon_side = str(preconSide)

    def setValue(self, row: int, col: int, value: float):
        if row >= self._mat.shape[0] or col >= self._mat.shape[1]:
            new_sz = max(row + 1, col + 1, self._mat.shape[0])
            new_mat = np.zeros((new_sz, new_sz), dtype=float)
            new_mat[:self._mat.shape[0], :self._mat.shape[1]] = self._mat
            self._mat = new_mat
            self._n = new_sz
        self._mat[row, col] = float(value)

    def eval(self, t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate Jacobian df/dy at state (t, y) using internal system."""
        if self._sys is None:
            return self._mat
        n = len(y)
        ydot0 = np.zeros(n, dtype=float)
        if hasattr(self._sys, "eval_f"):
            self._sys.eval_f(t, y, ydot0)
        elif hasattr(self._sys, "eval"):
            self._sys.eval(t, y, ydot0)
        elif callable(self._sys):
            ydot0 = np.asarray(self._sys(t, y), dtype=float)

        def f_wrap(t_val, y_val):
            out = np.zeros(n, dtype=float)
            if hasattr(self._sys, "eval_f"):
                self._sys.eval_f(t_val, y_val, out)
            elif hasattr(self._sys, "eval"):
                self._sys.eval(t_val, y_val, out)
            elif callable(self._sys):
                out = np.asarray(self._sys(t_val, y_val), dtype=float)
            return out

        return self.eval_fd(f_wrap, t, y, ydot0)

    def eval_fd(self, func: Callable[[float, np.ndarray], np.ndarray],
                t: float, y: np.ndarray, ydot: np.ndarray,
                atol: float = 1e-12) -> np.ndarray:
        """Compute numerical finite-difference Jacobian J_ij = d(ydot_i) / d(y_j)."""
        n = len(y)
        jac = np.empty((n, n), dtype=float)
        eps = math.sqrt(np.finfo(float).eps)

        for j in range(n):
            step = eps * max(abs(y[j]), 1.0)
            y_plus = y.copy()
            y_plus[j] += step
            ydot_plus = func(t, y_plus)
            jac[:, j] = (ydot_plus - ydot) / step
            self._evals += 1

        self._mat = jac
        return jac

    def solve(self, b: np.ndarray) -> np.ndarray:
        return _factor_dense_matrix(self._mat).solve(b)


class AdaptivePreconditioner(SystemJacobian):
    def prunePreconditioner(self) -> None:
        pass

    """Preconditioner for accelerating iterative and Krylov linear solves in stiff ODEs."""

    def __init__(self, dim_or_matrix: Union[int, np.ndarray] = 0,
                 ptype: str = "block_diagonal", drop_tol: float = 0.0):
        super().__init__(n=dim_or_matrix if isinstance(dim_or_matrix, (int, float)) else len(dim_or_matrix))
        self._type = "Adaptive"
        self._threshold = 0.0
        self._drop_tol = float(drop_tol)
        self._fill_factor = 2
        self._blocks: List[np.ndarray] = []
        self._block_factors: List[DenseMatrix] = []
        self._full_factor: Optional[DenseMatrix] = None
        if isinstance(dim_or_matrix, np.ndarray):
            self._mat = dim_or_matrix.copy()
            self.factorize()

    def type(self) -> str:
        return "Adaptive"

    def threshold(self) -> float:
        return self._threshold

    def setThreshold(self, threshold: float):
        self._threshold = float(threshold)

    def ilutDropTol(self) -> float:
        return self._drop_tol

    def setIlutDropTol(self, droptol: float):
        self._drop_tol = float(droptol)

    def ilutFillFactor(self) -> int:
        return self._fill_factor

    def setIlutFillFactor(self, fillFactor: int):
        self._fill_factor = int(fillFactor)

    def initialize(self, networkSize_or_blocks: Union[int, List[int]]):
        if isinstance(networkSize_or_blocks, (int, float)):
            self._n = int(networkSize_or_blocks)
            self._mat = np.eye(self._n, dtype=float)
            self._full_factor = _factor_dense_matrix(self._mat)
            self._blocks = []
            self._block_factors = []
        else:
            self._blocks = [np.eye(sz, dtype=float) for sz in networkSize_or_blocks]
            self._block_factors = [_factor_dense_matrix(block) for block in self._blocks]
            self._full_factor = None

    def setValue(self, row: int, col: int, value: float):
        if row >= self._mat.shape[0] or col >= self._mat.shape[1]:
            new_sz = max(row + 1, col + 1, self._mat.shape[0])
            new_mat = np.zeros((new_sz, new_sz), dtype=float)
            new_mat[:self._mat.shape[0], :self._mat.shape[1]] = self._mat
            self._mat = new_mat
            self._n = new_sz
        self._mat[row, col] = float(value)
        self._full_factor = None

    def factorize(self, full_jac: Optional[np.ndarray] = None, offsets: Optional[List[int]] = None):
        if full_jac is not None and offsets is not None:
            factors = []
            for idx, sz in enumerate([b.shape[0] for b in self._blocks]):
                off = offsets[idx]
                block = full_jac[off:off + sz, off:off + sz]
                try:
                    factor = _factor_dense_matrix(block)
                    self._blocks[idx] = block.copy()
                except Exception:
                    self._blocks[idx] = np.eye(sz, dtype=float)
                    factor = _factor_dense_matrix(self._blocks[idx])
                factors.append(factor)
            self._block_factors = factors
            self._full_factor = None
        else:
            # Full matrix factorization or drop-tolerance pruning.
            A = self._mat.copy()
            if self._drop_tol > 0.0:
                diag = np.diag(A)
                # Drop small off-diagonal elements
                mask = np.abs(A) < self._drop_tol * np.max(np.abs(diag))
                np.fill_diagonal(mask, False)
                A[mask] = 0.0
            self._full_factor = _factor_dense_matrix(A)
            self._block_factors = []

    def solve(self, rhs: np.ndarray, offsets: Optional[List[int]] = None) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float)
        if offsets is not None and len(self._blocks) > 0:
            res = np.empty_like(rhs_arr)
            factors = self._block_factors
            if not factors:
                factors = [_factor_dense_matrix(block) for block in self._blocks]
                self._block_factors = factors
            for idx, factor in enumerate(factors):
                off = offsets[idx]
                sz = factor.nRows()
                res[off:off + sz] = factor.solve(rhs_arr[off:off + sz])
            return res
        if self._full_factor is None:
            self._full_factor = _factor_dense_matrix(self._mat)
        return self._full_factor.solve(rhs_arr)

    def stateAdjustment(self, state: np.ndarray):
        # Clip state elements to strictly positive for composition variables
        np.maximum(state, 1e-30, out=state)


class Integrator:
    def initialize(self, t0: float = 0.0, y0: Optional[np.ndarray] = None) -> None:
        self._time = float(t0)
        if y0 is not None:
            self._y = np.asarray(y0, dtype=float).copy()

    def integrate(self, tout: float) -> float:
        self._time = float(tout)
        return self._time

    def algebraicInErrorTest(self) -> bool:
        return False

    def includeAlgebraicInErrorTest(self, include: bool = True) -> None:
        pass

    def applyOptions(self) -> None:
        pass

    def checkError(self) -> None:
        pass

    def currentTime(self) -> float:
        return self._time

    def derivative(self, tout: float, k: int = 1) -> np.ndarray:
        return np.zeros_like(self._y) if self._y is not None else np.array([])

    def getErrorInfo(self) -> dict:
        return {}

    def lastOrder(self) -> int:
        return 1

    def linearSolverType(self) -> str:
        return getattr(self, '_linear_solver', 'Dense')

    def setLinearSolverType(self, solver: str) -> None:
        self._linear_solver = str(solver)

    def maxNonlinConvFailures(self) -> int:
        return 10

    def maxNonlinIterations(self) -> int:
        return 10

    def setMaxNonlinConvFailures(self, n: int) -> None:
        pass

    def setMaxNonlinIterations(self, n: int) -> None:
        pass

    def maxOrder(self) -> int:
        return 5

    def setMaxOrder(self, order: int) -> None:
        pass

    def maxSteps(self) -> int:
        return getattr(self, '_max_steps', 20000)

    def setMaxSteps(self, n: int) -> None:
        self._max_steps = int(n)

    def nEquations(self) -> int:
        return len(self._y) if self._y is not None else 0

    def nSensParams(self) -> int:
        return 0

    def preconditioner(self):
        return None

    def setPreconditioner(self, pre: Any) -> None:
        pass

    def preconditionerSide(self) -> str:
        return 'none'

    def preconditionerSolve(self, rhs: np.ndarray, output: np.ndarray) -> None:
        output[:] = rhs

    def sensInit(self, nparams: int, func: Any) -> None:
        pass

    def setBandwidth(self, kl: int, ku: int) -> None:
        pass

    def setMaxErrTestFails(self, n: int) -> None:
        pass

    def setMaxStepSize(self, dt: float) -> None:
        pass

    def setMinStepSize(self, dt: float) -> None:
        pass

    def setMethod(self, method: str) -> None:
        pass

    def setRootFunctionCount(self, n: int) -> None:
        pass

    def setSensitivityTolerances(self, reltol: float, abstol: float) -> None:
        pass

    def solution(self, k: Optional[int] = None) -> Union[float, np.ndarray]:
        if self._y is None:
            return 0.0 if k is not None else np.array([])
        return self._y[k] if k is not None else self._y.copy()

    def solverStats(self) -> dict:
        return {"nEvals": self.nEvals(), "time": self._time}

    def warn(self, msg: str) -> None:
        pass

    def reinitialize(self, t0: float, y0: Optional[np.ndarray] = None) -> None:
        self.initialize(t0, y0)

    """Abstract base class for stiff ODE and DAE integrators."""

    def __init__(self, func: Optional[Any] = None):
        self.rtol = 1e-9
        self.atol: Union[float, np.ndarray] = 1e-15
        self.rtol_sens = 1e-6
        self.atol_sens = 1e-12
        self.max_step = 0.0
        self.max_steps = 20000
        self._time = 0.0
        self._nevals = 0
        self._func = func
        self._y: Optional[np.ndarray] = None

    def setTolerances(self, reltol: float, abstol: Union[float, Sequence[float], np.ndarray]):
        if reltol > 0:
            self.rtol = float(reltol)
        if isinstance(abstol, (int, float)) and abstol > 0:
            self.atol = float(abstol)
        elif isinstance(abstol, (list, tuple, np.ndarray)):
            self.atol = np.asarray(abstol, dtype=float)

    def set_tolerances(self, reltol: float, abstol: Union[float, Sequence[float], np.ndarray]):
        self.setTolerances(reltol, abstol)

    def setSensitivityTolerances(self, reltol: float, atol: float):
        if reltol > 0:
            self.rtol_sens = float(reltol)
        if atol > 0:
            self.atol_sens = float(atol)

    def setMaxStepSize(self, hmax: float):
        self.max_step = float(hmax)

    def setMaxSteps(self, nmax: int):
        self.max_steps = int(nmax)

    def maxSteps(self) -> int:
        return self.max_steps

    def currentTime(self) -> float:
        return self._time

    def nEvals(self) -> int:
        return self._nevals

    def nEquations(self) -> int:
        if self._y is not None:
            return len(self._y)
        if self._func is not None:
            if hasattr(self._func, "nEquations"):
                return self._func.nEquations()
            if hasattr(self._func, "neq"):
                return self._func.neq()
        return 0

    def solution(self, k: Optional[int] = None) -> Union[float, np.ndarray]:
        if self._y is None:
            return np.array([])
        if k is not None:
            return float(self._y[k])
        return self._y.copy()

    @property
    def y(self) -> np.ndarray:
        return self.solution() if self._y is not None else np.array([])


class CVodesIntegrator(Integrator):
    """Stiff ODE integrator with native SUNDIALS CVODES-equivalent BDF kernel."""

    _CV_ETA_MAX_FIRST_STEP = 10000.0
    _CV_ETA_MAX_EARLY_STEPS = 10.0
    _CV_ETA_MAX_GENERAL = 10.0
    _CV_SMALL_NST = 10

    def __init__(self, func: Optional[Any] = None):
        super().__init__(func)
        self._cv_order = 1
        self._cv_next_h = 0.0
        self._cv_zn: List[np.ndarray] = []
        self._cv_tau = [0.0] * 7
        self._cv_q = 1
        self._cv_qprime = 1
        self._cv_L = 2
        self._cv_qwait = 2
        self._cv_nst = 0
        self._cv_h = 0.0
        self._cv_hscale = 0.0
        self._cv_hprime = 0.0
        self._cv_tn = 0.0
        self._cv_t0 = 0.0
        self._cv_saved_tq5 = 0.0
        self._cv_eta = 1.0
        self._cv_etamax = self._CV_ETA_MAX_FIRST_STEP
        self._cv_nscon = 0
        self._cv_max_order = 5
        self._cv_ewt: Optional[np.ndarray] = None
        self._cv_rhs_jacobian: Optional[np.ndarray] = None
        self._cv_linear_factor = None
        self._cv_last_jac_step = -1
        self._cv_last_lsetup_step = -1
        self._cv_lsetup_gamma = 0.0
        self._cv_force_lsetup = True
        self._cv_force_jacobian = True
        self._cv_jacobian_evaluations = 0
        self._cv_linear_setups = 0
        self._cv_order_histogram = {}
        self._cv_error_test_failures = 0
        self._cv_nonlinear_failures = 0
        self._cv_nonlinear_iters = 0
        self._method = 1  # 1 = BDF, 0 = Adams
        self._linear_solver = "DENSE"
        self._mupper = 0
        self._mlower = 0
        self._max_err_test_fails = 7
        self.min_step = 0.0
        self._base_neq = 0
        self._np = 0
        self._augmented = False
        self._sens_ok = False
        self.rtol_sens = 1e-5
        self.atol_sens = 1e-4
        self._sens_atol = None
        self._n_root_functions = 0

    def initialize(self, t0: float, func_or_y0: Optional[Any] = None, y0: Optional[np.ndarray] = None):
        self._time = float(t0)
        self._cv_t0 = float(t0)
        self._nevals = 0
        self._cv_order = 1
        self._cv_next_h = 0.0
        self._cv_zn = []
        self._cv_tau = [0.0] * 7
        self._cv_q = 1
        self._cv_qprime = 1
        self._cv_L = 2
        self._cv_qwait = 2
        self._cv_nst = 0
        self._cv_h = 0.0
        self._cv_hscale = 0.0
        self._cv_hprime = 0.0
        self._cv_tn = self._time
        self._cv_saved_tq5 = 0.0
        self._cv_eta = 1.0
        self._cv_etamax = self._CV_ETA_MAX_FIRST_STEP
        self._cv_nscon = 0
        self._cv_ewt = None
        self._cv_rhs_jacobian = None
        self._cv_linear_factor = None
        self._cv_last_jac_step = -1
        self._cv_last_lsetup_step = -1
        self._cv_lsetup_gamma = 0.0
        self._cv_force_lsetup = True
        self._cv_force_jacobian = True
        self._cv_jacobian_evaluations = 0
        self._cv_linear_setups = 0
        self._cv_order_histogram = {}
        self._cv_error_test_failures = 0
        self._cv_nonlinear_failures = 0
        self._cv_nonlinear_iters = 0
        self._sens_ok = False

        if y0 is not None:
            self._func = func_or_y0
            base_y = np.asarray(y0, dtype=float).copy()
        elif func_or_y0 is not None:
            if isinstance(func_or_y0, (np.ndarray, list, tuple)):
                base_y = np.asarray(func_or_y0, dtype=float).copy()
            else:
                self._func = func_or_y0
                neq = 0
                if hasattr(self._func, "neq"):
                    try:
                        neq = self._func.neq()
                    except (NotImplementedError, TypeError):
                        pass
                if neq == 0 and hasattr(self._func, "nEquations"):
                    try:
                        neq = self._func.nEquations()
                    except (NotImplementedError, TypeError):
                        pass
                if hasattr(self._func, "getState") and neq > 0:
                    y_init = np.zeros(neq, dtype=float)
                    self._func.getState(y_init)
                    base_y = y_init
                elif hasattr(self._func, "getInitialConditions") and neq > 0:
                    y_init = np.zeros(neq, dtype=float)
                    self._func.getInitialConditions(t0, neq, y_init)
                    base_y = y_init
                elif hasattr(self._func, "y0"):
                    base_y = np.asarray(self._func.y0, dtype=float).copy()
                else:
                    base_y = np.array([], dtype=float)
        else:
            base_y = np.array([], dtype=float)

        self._base_neq = len(base_y)

        self._np = 0
        if self._func is not None and hasattr(self._func, "nparams"):
            try:
                self._np = int(self._func.nparams())
            except (NotImplementedError, TypeError):
                self._np = 0
        if self._np == 0 and self._func is not None and hasattr(self._func, "m_sens_params"):
            self._np = len(getattr(self._func, "m_sens_params", []))

        if self._np > 0 and self._base_neq > 0:
            self._augmented = True
            aug_len = self._base_neq * (1 + self._np)
            self._y = np.zeros(aug_len, dtype=float)
            self._y[:self._base_neq] = base_y
            self.setSensitivityTolerances(self.rtol_sens, self.atol_sens)
        else:
            self._augmented = False
            self._y = base_y

        if self._y is not None and len(self._y) > 0:
            self._cv_zn = [self._y.copy()] + [np.zeros_like(self._y) for _ in range(5)]

        self.applyOptions()

    def reinitialize(self, t0: float, func_or_y0: Optional[Any] = None, y0: Optional[np.ndarray] = None):
        self.initialize(t0, func_or_y0, y0)

    def sensInit(self, t0: float, func: Any) -> None:
        self.initialize(t0, func)

    def setSensitivityTolerances(self, reltol: float, abstol: float) -> None:
        self.rtol_sens = float(reltol)
        self.atol_sens = float(abstol)
        if hasattr(self, "_func") and self._func is not None and hasattr(self._func, "m_paramScales") and self._func.m_paramScales:
            self._sens_atol = [self.atol_sens / float(s) for s in self._func.m_paramScales]
        else:
            self._sens_atol = [self.atol_sens] * max(1, self._np)

    def nSensParams(self) -> int:
        return self._np

    def sensitivity(self, k: int, p: int) -> float:
        if self._time == getattr(self, "_cv_t0", 0.0):
            return 0.0
        if k >= self._base_neq:
            raise CanteraError(f"CVodesIntegrator::sensitivity: k out of range ({k})")
        if p >= self._np:
            raise CanteraError(f"CVodesIntegrator::sensitivity: p out of range ({p})")
        if not self._augmented or self._y is None:
            return 0.0
        return float(self._y[self._base_neq + p * self._base_neq + k])

    def _eval_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate the ODE right-hand-side function (handling sensitivity augmentation if enabled)."""
        self._nevals += 1
        if not self._augmented:
            n = len(y)
            value = np.zeros(n, dtype=float)
            if hasattr(self._func, "eval_f"):
                self._func.eval_f(t, y, value)
                return value
            if hasattr(self._func, "eval"):
                try:
                    p = getattr(self._func, "m_sens_params", [])
                    self._func.eval(t, y, value, p)
                    return value
                except TypeError:
                    self._func.eval(t, y, value)
                    return value
            if callable(self._func):
                result = np.asarray(self._func(t, y), dtype=float)
                if result.shape != y.shape:
                    raise NumericsError("RHS result dimensions do not match the state")
                return result
            raise NumericsError("No valid eval method on func")

        # Augmented mode: y contains [base_y, s_0, s_1, ..., s_{P-1}]
        N = self._base_neq
        P = self._np
        base_state = y[:N]
        f0 = np.zeros(N, dtype=float)

        param_vec = getattr(self._func, "m_sens_params", [])
        if hasattr(self._func, "eval_f"):
            self._func.eval_f(t, base_state, f0)
        elif hasattr(self._func, "eval"):
            try:
                self._func.eval(t, base_state, f0, param_vec)
            except TypeError:
                self._func.eval(t, base_state, f0)
        elif callable(self._func):
            f0 = np.asarray(self._func(t, base_state), dtype=float)

        dY = np.zeros_like(y)
        dY[:N] = f0

        srur = math.sqrt(np.finfo(float).eps)
        for p_idx in range(P):
            sp = y[N + p_idx * N : N + (p_idx + 1) * N]
            # Parameter perturbation
            orig_p = param_vec[p_idx]
            dp = srur * max(1.0, abs(orig_p))
            param_vec[p_idx] = orig_p + dp
            fp_p = np.zeros(N, dtype=float)
            if hasattr(self._func, "eval_f"):
                self._func.eval_f(t, base_state, fp_p)
            elif hasattr(self._func, "eval"):
                try:
                    self._func.eval(t, base_state, fp_p, param_vec)
                except TypeError:
                    self._func.eval(t, base_state, fp_p)
            param_vec[p_idx] = orig_p
            df_dp = (fp_p - f0) / dp

            sp_norm = math.sqrt(float(np.mean(sp**2)))
            if sp_norm == 0.0:
                J_sp = np.zeros_like(sp)
            else:
                sigma = srur / max(1.0, sp_norm)
                fp_y = np.zeros(N, dtype=float)
                if hasattr(self._func, "eval_f"):
                    self._func.eval_f(t, base_state + sigma * sp, fp_y)
                elif hasattr(self._func, "eval"):
                    try:
                        self._func.eval(t, base_state + sigma * sp, fp_y, param_vec)
                    except TypeError:
                        self._func.eval(t, base_state + sigma * sp, fp_y)
                J_sp = (fp_y - f0) / sigma
            dY[N + p_idx * N : N + (p_idx + 1) * N] = J_sp + df_dp

        return dY

    def _cv_eval_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        try:
            return self._eval_rhs(t, y)
        except CanteraError as err:
            raise _RecoverableRhsError(str(err)) from err

    def _weighted_rms(self, value: np.ndarray, state: np.ndarray) -> float:
        atol = np.asarray(self.atol, dtype=float)
        if atol.ndim and atol.shape != state.shape:
            raise NumericsError("Absolute tolerance vector does not match the state")
        weights = 1.0 / (atol + self.rtol * np.abs(state))
        return math.sqrt(float(np.mean((value * weights) ** 2)))

    @staticmethod
    def _bdf_error_coefficient(order: int, h: float,
                               previous_steps: Sequence[float]) -> float:
        if order == 1:
            return 0.5
        if len(previous_steps) < order - 1:
            raise NumericsError("Insufficient step history for requested BDF order")
        ell = [0.0] * (order + 1)
        ell[0] = ell[1] = 1.0
        alpha0 = alpha0_hat = -1.0
        hsum = h
        xi_inv = xistar_inv = 1.0
        for j in range(2, order):
            hsum += previous_steps[j - 2]
            xi_inv = h / hsum
            alpha0 -= 1.0 / j
            for i in range(j, 0, -1):
                ell[i] += ell[i - 1] * xi_inv
        alpha0 -= 1.0 / order
        xistar_inv = -ell[1] - alpha0
        hsum += previous_steps[order - 2]
        xi_inv = h / hsum
        alpha0_hat = -ell[1] - xi_inv
        for i in range(order, 0, -1):
            ell[i] += ell[i - 1] * xistar_inv
        a1 = 1.0 - alpha0_hat + alpha0
        a2 = 1.0 + order * a1
        return abs(a1 / (alpha0 * a2))

    def _cv_set_bdf(self, h: float) -> Tuple[List[float], List[float]]:
        q = self._cv_q
        ell = [0.0] * 6
        tq = [0.0] * 6
        ell[0] = ell[1] = 1.0
        xi_inv = xistar_inv = 1.0
        alpha0 = alpha0_hat = -1.0
        hsum = h
        if q > 1:
            for j in range(2, q):
                hsum += self._cv_tau[j - 1]
                xi_inv = h / hsum
                alpha0 -= 1.0 / j
                for i in range(j, 0, -1):
                    ell[i] += ell[i - 1] * xi_inv
            alpha0 -= 1.0 / q
            xistar_inv = -ell[1] - alpha0
            hsum += self._cv_tau[q - 1]
            xi_inv = h / hsum
            alpha0_hat = -ell[1] - xi_inv
            for i in range(q, 0, -1):
                ell[i] += ell[i - 1] * xistar_inv

        a1 = 1.0 - alpha0_hat + alpha0
        a2 = 1.0 + q * a1
        tq[2] = abs(a1 / (alpha0 * a2))
        tq[5] = abs(a2 * xistar_inv / (ell[q] * xi_inv))
        if self._cv_qwait == 1:
            if q > 1:
                c = xistar_inv / ell[q]
                a3 = alpha0 + 1.0 / q
                a4 = alpha0_hat + xi_inv
                cp_inv = (1.0 - a4 + a3) / a3
                tq[1] = abs(c * cp_inv)
            else:
                tq[1] = 1.0
            hsum += self._cv_tau[q]
            xi_inv = h / hsum
            a5 = alpha0 - 1.0 / (q + 1)
            a6 = alpha0_hat - xi_inv
            cpp_inv = (1.0 - a6 + a5) / a2
            tq[3] = abs(cpp_inv / (xi_inv * (q + 2) * a5))
        tq[4] = 0.1 / tq[2]
        return ell, tq

    def _cv_rescale(self, eta: float) -> None:
        factor = eta
        for j in range(1, self._cv_q + 1):
            self._cv_zn[j] *= factor
            factor *= eta
        self._cv_h = self._cv_hscale * eta
        self._cv_next_h = self._cv_h
        self._cv_hscale = self._cv_h
        self._cv_nscon = 0

    def _cv_adjust_order(self, deltaq: int) -> None:
        q = self._cv_q
        if q == 2 and deltaq != 1:
            return
        if deltaq == 1:
            ell = [0.0] * 6
            ell[2] = 1.0
            alpha1 = prod = xiold = 1.0
            alpha0 = -1.0
            hsum = self._cv_hscale
            for j in range(1, q):
                hsum += self._cv_tau[j + 1]
                xi = hsum / self._cv_hscale
                prod *= xi
                alpha0 -= 1.0 / (j + 1)
                alpha1 += 1.0 / xi
                for i in range(j + 2, 1, -1):
                    ell[i] = ell[i] * xiold + ell[i - 1]
                xiold = xi
            a1 = (-alpha0 - alpha1) / prod
            self._cv_zn[q + 1] = a1 * self._cv_zn[5]
            for j in range(2, q + 1):
                self._cv_zn[j] += ell[j] * self._cv_zn[q + 1]
            return

        if deltaq == -1:
            ell = [0.0] * 6
            ell[2] = 1.0
            hsum = 0.0
            for j in range(1, q - 1):
                hsum += self._cv_tau[j]
                xi = hsum / self._cv_hscale
                for i in range(j + 2, 1, -1):
                    ell[i] = ell[i] * xi + ell[i - 1]
            for j in range(2, q):
                self._cv_zn[j] -= ell[j] * self._cv_zn[q]

    def _cv_adjust_params(self, eta: float) -> None:
        if self._cv_qprime != self._cv_q:
            self._cv_adjust_order(self._cv_qprime - self._cv_q)
            self._cv_q = self._cv_qprime
            self._cv_L = self._cv_q + 1
            self._cv_qwait = self._cv_L
        self._cv_rescale(eta)

    def _cv_set_eta(self, eta: float) -> Tuple[float, float]:
        if 0.0 < eta < 1.5:
            return 1.0, self._cv_h
        if eta >= 1.5:
            eta = min(eta, self._cv_etamax)
            if self.max_step > 0.0:
                eta /= max(1.0, abs(self._cv_h) * eta / self.max_step)
        else:
            eta = max(eta, 0.1)
        if self._cv_qprime < self._cv_q:
            self._cv_nscon = 0
        return eta, self._cv_h * eta

    def _cv_prepare_next_step(self, dsm: float, tq: Sequence[float],
                              acor: np.ndarray, weights: np.ndarray) -> None:
        if self._cv_etamax == 1.0:
            self._cv_qwait = max(self._cv_qwait, 2)
            self._cv_qprime = self._cv_q
            self._cv_eta = 1.0
            self._cv_hprime = self._cv_h
            return

        etaq = 1.0 / (6.0 * dsm ** (1.0 / self._cv_L) + 1e-6)
        if self._cv_qwait != 0:
            self._cv_eta = etaq
            self._cv_qprime = self._cv_q
            self._cv_eta, self._cv_hprime = self._cv_set_eta(self._cv_eta)
            return

        self._cv_qwait = 2
        etaqm1 = 0.0
        if self._cv_q > 1:
            ddn = self._cv_wrms(self._cv_zn[self._cv_q], weights) * tq[1]
            etaqm1 = 1.0 / (6.0 * ddn ** (1.0 / self._cv_q) + 1e-6)
        etaqp1 = 0.0
        if self._cv_q < self._cv_max_order and self._cv_saved_tq5 != 0.0:
            if len(self._cv_tau) > 2 and self._cv_tau[2] != 0.0:
                cquot = (tq[5] / self._cv_saved_tq5) * (
                    self._cv_h / self._cv_tau[2]
                ) ** self._cv_L
                dup_vec = acor - cquot * self._cv_zn[5]
                dup = self._cv_wrms(dup_vec, weights) * tq[3]
                etaqp1 = 1.0 / (10.0 * dup ** (1.0 / (self._cv_L + 1)) + 1e-6)

        etam = max(etaqm1, etaq, etaqp1)
        if 0.0 < etam < 1.5:
            self._cv_eta = 1.0
            self._cv_qprime = self._cv_q
        elif etam == etaq:
            self._cv_eta = etaq
            self._cv_qprime = self._cv_q
        elif etam == etaqm1:
            self._cv_eta = etaqm1
            self._cv_qprime = self._cv_q - 1
        else:
            self._cv_eta = etaqp1
            self._cv_qprime = self._cv_q + 1
            self._cv_zn[5] = acor.copy()
        self._cv_eta, self._cv_hprime = self._cv_set_eta(self._cv_eta)

    def _cv_set_error_weights(self, state: np.ndarray) -> np.ndarray:
        if self._augmented:
            N = self._base_neq
            P = self._np
            atol_arr = np.zeros(len(state), dtype=float)
            base_atol = np.asarray(self.atol, dtype=float)
            if base_atol.ndim == 0:
                atol_arr[:N] = float(base_atol)
            else:
                atol_arr[:N] = base_atol[:N]
            for p_idx in range(P):
                p_atol = self._sens_atol[p_idx] if self._sens_atol and p_idx < len(self._sens_atol) else self.atol_sens
                atol_arr[N + p_idx * N : N + (p_idx + 1) * N] = p_atol
            denominator = atol_arr + self.rtol * np.abs(state)
        else:
            atol = np.asarray(self.atol, dtype=float)
            if atol.ndim and atol.shape != state.shape:
                raise NumericsError("Absolute tolerance vector does not match the state")
            denominator = atol + self.rtol * np.abs(state)

        if np.any(denominator <= 0.0):
            raise NumericsError("CVODE error weights require positive error tolerances")
        self._cv_ewt = 1.0 / denominator
        return self._cv_ewt

    @staticmethod
    def _cv_wrms(value: np.ndarray, weights: np.ndarray) -> float:
        return math.sqrt(float(np.mean((value * weights) ** 2)))

    def _cv_dense_dq_min_increment(self, rhs: np.ndarray,
                                   weights: np.ndarray, n: int) -> float:
        fnorm = self._cv_wrms(rhs, weights)
        if fnorm == 0.0:
            return 1.0
        return (1000.0 * abs(self._cv_h) * np.finfo(float).eps
                * n * fnorm)

    @staticmethod
    def _cv_dense_dq_increment(state_value: float, error_weight: float,
                               min_increment: float) -> float:
        srur = math.sqrt(np.finfo(float).eps)
        return max(srur * abs(state_value), min_increment / error_weight)

    def _cv_initial_step(self, tout: float, ydot0: np.ndarray,
                         weights: np.ndarray) -> float:
        tdist = abs(tout - self._cv_tn)
        if tdist == 0.0:
            raise NumericsError("Initial output time is too close to the initial time")
        uround = np.finfo(float).eps
        tround = uround * max(abs(self._cv_tn), abs(tout))
        if tdist < 2.0 * tround:
            raise NumericsError("Initial output time is too close to the initial time")
        hlb = 100.0 * tround
        y = self._cv_zn[0]
        atol = np.asarray(self.atol, dtype=float)
        if self._augmented:
            atol = np.full(len(y), self.atol if atol.ndim == 0 else atol[0])
        error_scale = atol + self.rtol * np.abs(y)
        denom = 0.1 * np.abs(y) + error_scale
        hub_inv = float(np.max(np.abs(ydot0) / denom))
        hub = 0.1 * tdist
        if hub * hub_inv > 1.0:
            hub = 1.0 / hub_inv
        hg = math.sqrt(hlb * hub)
        if hub < hlb:
            return hg

        hs = hg
        hnew = hg
        valid_estimate = False
        for count1 in range(1, 5):
            ytrial = y + hg * ydot0
            try:
                ftrial = self._cv_eval_rhs(self._cv_tn + hg, ytrial)
            except _RecoverableRhsError:
                hnew = hg * 0.25
                hg = hnew
                if count1 == 4:
                    break
                continue
            valid_estimate = True
            ydd = (ftrial - ydot0) / hg
            yddnorm = self._cv_wrms(ydd, weights)
            hs = hg
            hnew = math.sqrt(2.0 / yddnorm) if yddnorm * hub * hub > 2.0 else math.sqrt(hg * hub)
            if count1 == 4:
                break
            ratio = hnew / hg
            if 0.5 < ratio < 2.0:
                break
            if count1 > 1 and ratio > 2.0:
                hnew = hg
                break
            hg = hnew
        if not valid_estimate:
            raise NumericsError("CVODE BDF could not estimate a valid initial step")
        h0 = min(hub, max(hlb, 0.5 * hnew))
        if self.max_step > 0.0:
            h0 = min(h0, self.max_step)
        if self.min_step > 0.0:
            h0 = max(h0, self.min_step)
        return h0

    def _cv_nordsieck_predict(self) -> float:
        tnew = self._cv_tn + self._cv_h
        for k in range(1, self._cv_q + 1):
            for j in range(self._cv_q, k - 1, -1):
                self._cv_zn[j - 1] = self._cv_zn[j - 1] + self._cv_zn[j]
        return tnew

    def _cv_newton_corrector(self, tnew: float, ell: Sequence[float], tq: Sequence[float],
                             weights: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        rl1 = 1.0 / ell[1]
        gamma = self._cv_h * rl1
        acor = np.zeros_like(self._cv_zn[0])
        predicted = self._cv_zn[0].copy()
        n = len(predicted)
        try:
            f0 = self._cv_eval_rhs(tnew, predicted)
            previous_factor = self._cv_linear_factor
            previous_gamma = self._cv_lsetup_gamma
            factor = self._cv_prepare_linear_system(
                tnew, predicted, f0, gamma, weights)
        except _RecoverableRhsError:
            return acor, math.inf, False
        if factor is None:
            return acor, math.inf, False

        crate = 1.0
        delp = 0.0
        acnrm = math.inf
        for iteration in range(3):
            self._cv_nonlinear_iters += 1
            ytrial = predicted + acor
            try:
                ftrial = f0 if iteration == 0 else self._cv_eval_rhs(tnew, ytrial)
            except _RecoverableRhsError:
                return acor, math.inf, False
            residual = rl1 * self._cv_zn[1] + acor - gamma * ftrial
            if not np.all(np.isfinite(residual)):
                return acor, math.inf, False
            delta = factor.solve(-residual)
            if factor is previous_factor and previous_gamma != 0.0:
                gamma_ratio = gamma / previous_gamma
                delta *= 2.0 / (1.0 + gamma_ratio)
            del_norm = self._cv_wrms(delta, weights)
            acor += delta
            if iteration > 0:
                crate = max(0.3 * crate, del_norm / max(delp, np.finfo(float).tiny))
            dcon = del_norm * min(1.0, crate) / tq[4]
            if dcon <= 1.0:
                acnrm = del_norm if iteration == 0 else self._cv_wrms(acor, weights)
                return acor, acnrm, True
            if iteration >= 1 and del_norm > 2.0 * delp:
                return acor, math.inf, False
            delp = del_norm
        return acor, math.inf, False

    def _cv_prepare_linear_system(self, time: float, state: np.ndarray,
                                  rhs: np.ndarray, gamma: float,
                                  weights: np.ndarray):
        setup = (
            self._cv_last_lsetup_step < 0
            or self._cv_force_lsetup
            or self._cv_nst >= self._cv_last_lsetup_step + 20
            or (self._cv_lsetup_gamma != 0.0 and
                abs(gamma / self._cv_lsetup_gamma - 1.0) > 0.3)
        )
        if not setup:
            return self._cv_linear_factor

        jacobian_bad = (
            self._cv_force_jacobian
            or self._cv_rhs_jacobian is None
            or self._cv_nst >= self._cv_last_jac_step + 51
        )
        n = len(state)
        if jacobian_bad:
            jacobian = np.empty((n, n), dtype=float)
            min_increment = self._cv_dense_dq_min_increment(rhs, weights, n)
            for column in range(n):
                increment = self._cv_dense_dq_increment(
                    float(state[column]), float(weights[column]), min_increment)
                perturbed = state.copy()
                perturbed[column] += increment
                fp = self._cv_eval_rhs(time, perturbed)
                for row in range(n):
                    jacobian[row, column] = (float(fp[row]) - float(rhs[row])) / increment
            self._cv_rhs_jacobian = jacobian
            self._cv_last_jac_step = self._cv_nst
            self._cv_jacobian_evaluations += 1

        system = np.eye(n, dtype=float) - gamma * self._cv_rhs_jacobian
        try:
            self._cv_linear_factor = _factor_dense_matrix(system)
        except (NumericsError, FloatingPointError, ValueError):
            self._cv_linear_factor = None
            return None
        self._cv_lsetup_gamma = gamma
        self._cv_last_lsetup_step = self._cv_nst
        self._cv_force_lsetup = False
        self._cv_force_jacobian = False
        self._cv_linear_setups += 1
        return self._cv_linear_factor

    def _cv_complete_step(self, ell: Sequence[float], tq: Sequence[float],
                          acor: np.ndarray) -> None:
        self._cv_nst += 1
        self._cv_tau[2:self._cv_q + 1] = self._cv_tau[1:self._cv_q]
        if self._cv_q == 1 and self._cv_nst > 1:
            self._cv_tau[2] = self._cv_tau[1]
        self._cv_tau[1] = self._cv_h
        for j in range(self._cv_q + 1):
            self._cv_zn[j] = self._cv_zn[j] + ell[j] * acor
        self._cv_qwait -= 1
        if self._cv_qwait == 1 and self._cv_q != self._cv_max_order:
            self._cv_zn[5] = acor.copy()
            self._cv_saved_tq5 = tq[5]

    def _cv_take_one_internal_step(self, tout: float) -> float:
        """Advance the integrator by exactly one internal BDF step."""
        if self._cv_nst == 0:
            weights = self._cv_set_error_weights(self._cv_zn[0])
            ydot0 = self._eval_rhs(self._cv_tn, self._cv_zn[0])
            self._cv_h = self._cv_initial_step(tout, ydot0, weights)
            if self.max_step > 0.0:
                self._cv_h = min(self._cv_h, self.max_step)
            if self.min_step > 0.0:
                self._cv_h = max(self._cv_h, self.min_step)
            self._cv_hscale = self._cv_h
            self._cv_hprime = self._cv_h
            self._cv_zn[1] = self._cv_h * ydot0

        weights = self._cv_set_error_weights(self._cv_zn[0])

        if self._cv_nst > 0 and self._cv_hprime != self._cv_h:
            self._cv_eta = self._cv_hprime / self._cv_h
            self._cv_adjust_params(self._cv_eta)

        ncf = nef = 0
        retry = True
        while retry:
            retry = False
            saved_zn = [entry.copy() for entry in self._cv_zn]
            saved_t = self._cv_tn
            tnew = self._cv_nordsieck_predict()
            ell, tq = self._cv_set_bdf(self._cv_h)
            acor, acnrm, converged = self._cv_newton_corrector(
                tnew, ell, tq, weights
            )
            if not converged:
                ncf += 1
                self._cv_nonlinear_failures += 1
                self._cv_zn = saved_zn
                self._cv_etamax = 1.0
                self._cv_force_lsetup = True
                self._cv_force_jacobian = True
                if ncf >= 10:
                    raise NumericsError("CVODE BDF nonlinear convergence failure")
                self._cv_eta = max(0.25, 0.0)
                self._cv_rescale(self._cv_eta)
                retry = True
                continue

            dsm = acnrm * tq[2]
            if not math.isfinite(dsm) or dsm > 1.0:
                nef += 1
                self._cv_error_test_failures += 1
                self._cv_zn = saved_zn
                self._cv_tn = saved_t
                self._cv_force_lsetup = True
                max_fails = self._max_err_test_fails if self._max_err_test_fails > 0 else 7
                if nef >= max_fails:
                    raise NumericsError(f"CVODE BDF local error test failed {max_fails} times")
                self._cv_etamax = 1.0
                if nef <= 3:
                    eta = 1.0 / (6.0 * max(dsm, 0.0) ** (1.0 / self._cv_L) + 1e-6)
                    eta = max(0.1, eta)
                    if nef >= 2:
                        eta = min(eta, 0.2)
                    self._cv_eta = eta
                    self._cv_rescale(eta)
                elif self._cv_q > 1:
                    self._cv_eta = 0.1
                    self._cv_adjust_order(-1)
                    self._cv_L = self._cv_q
                    self._cv_q -= 1
                    self._cv_qprime = self._cv_q
                    self._cv_qwait = self._cv_L
                    self._cv_rescale(self._cv_eta)
                else:
                    self._cv_eta = 0.1
                    self._cv_h *= self._cv_eta
                    self._cv_hprime = self._cv_h
                    self._cv_hscale = self._cv_h
                    self._cv_qwait = 10
                    self._cv_nscon = 0
                    self._cv_zn[1] = self._cv_h * self._eval_rhs(
                        self._cv_tn, self._cv_zn[0]
                    )
                retry = True
                continue

            self._cv_tn = tnew
            self._cv_complete_step(ell, tq, acor)
            self._cv_order_histogram[self._cv_q] = (
                self._cv_order_histogram.get(self._cv_q, 0) + 1)
            self._cv_prepare_next_step(dsm, tq, acor, weights)
            self._cv_etamax = (
                self._CV_ETA_MAX_EARLY_STEPS
                if self._cv_nst <= self._CV_SMALL_NST
                else self._CV_ETA_MAX_GENERAL
            )
            retry = False

        self._y = self._cv_zn[0].copy()
        self._cv_order = self._cv_q
        self._cv_next_h = self._cv_hprime
        return self._cv_tn

    def _cv_interpolate(self, tout: float) -> np.ndarray:
        """Evaluate CVODE's accepted Nordsieck polynomial at the requested time."""
        if self._cv_tn == tout:
            return self._cv_zn[0].copy()
        hu = self._cv_tau[1]
        if hu == 0.0:
            raise NumericsError("CVODE interpolation history is unavailable")
        ratio = (tout - self._cv_tn) / hu
        result = self._cv_zn[0].copy()
        factor = ratio
        for j in range(1, self._cv_q + 1):
            result += factor * self._cv_zn[j]
            factor *= ratio
        return result

    def integrate(self, tout: float) -> float:
        if self._func is None or self._y is None:
            raise NumericsError("Integrator not initialized with function and state")
        tout = float(tout)
        if tout == self._time:
            return self._time
        if tout < self._time:
            raise CanteraError(
                f"CVodesIntegrator::integrate: Cannot integrate backwards in time.\n"
                f"Requested time {tout} < current time {self._time}"
            )
        if not self._cv_zn:
            self._cv_zn = [self._y.copy()] + [np.zeros_like(self._y) for _ in range(5)]

        nsteps = 0
        while self._cv_tn < tout:
            if nsteps >= self.max_steps:
                raise CanteraError(
                    f"CVodesIntegrator::integrate: Maximum number of timesteps ({self.max_steps}) "
                    f"taken without reaching output time ({tout}).\n"
                    f"Current integrator time: {self._cv_tn}"
                )
            self._cv_take_one_internal_step(tout)
            nsteps += 1

        self._time = tout
        self._y = self._cv_interpolate(tout)
        self._sens_ok = False
        return self._time

    def step(self, tout: float) -> float:
        if self._func is None or self._y is None:
            raise NumericsError("Integrator not initialized with function and state")
        tout = float(tout)
        if not self._cv_zn:
            self._cv_zn = [self._y.copy()] + [np.zeros_like(self._y) for _ in range(5)]
        self._cv_take_one_internal_step(tout)
        self._time = self._cv_tn
        self._y = self._cv_zn[0].copy()
        self._sens_ok = False
        return self._time

    def derivative(self, tout: float, n: int = 1) -> np.ndarray:
        if n == 0:
            full = self._cv_interpolate(tout)
            return full[:self._base_neq] if self._augmented else full
        if not self._cv_zn or len(self._cv_zn) < 2:
            return np.zeros(self._base_neq, dtype=float)
        hu = self._cv_tau[1]
        if hu == 0.0:
            if n == 1 and self._func is not None and self._y is not None:
                full_rhs = self._eval_rhs(tout, self._y)
                return full_rhs[:self._base_neq] if self._augmented else full_rhs
            return np.zeros(self._base_neq, dtype=float)
        ratio = (float(tout) - self._cv_tn) / hu
        q = self._cv_q
        res = np.zeros_like(self._cv_zn[0])
        for j in range(n, q + 1):
            coeff = math.factorial(j) / math.factorial(j - n)
            res += coeff * self._cv_zn[j] * (ratio ** (j - n))
        res /= (hu ** n)
        return res[:self._base_neq] if self._augmented else res

    def solution(self, k: Optional[int] = None) -> Union[float, np.ndarray]:
        if self._y is None:
            return 0.0 if k is not None else np.array([])
        base_view = self._y[:self._base_neq] if self._augmented else self._y
        if k is not None:
            return float(base_view[k])
        return base_view.copy()

    def lastOrder(self) -> int:
        return self._cv_order

    def nEquations(self) -> int:
        return self._base_neq if self._augmented else (len(self._y) if self._y is not None else 0)

    def nEvals(self) -> int:
        return self._nevals

    def maxOrder(self) -> int:
        return self._cv_max_order

    def setMaxOrder(self, order: int) -> None:
        self._cv_max_order = max(1, min(5, int(order)))

    def setMethod(self, t: Any) -> None:
        s = str(t).upper()
        if s in ("1", "BDF", "BDF_METHOD"):
            self._method = 1
        elif s in ("0", "ADAMS", "ADAMS_METHOD"):
            self._method = 0
        else:
            raise CanteraError(f"CVodesIntegrator::setMethod: unknown method {t}")

    def setMaxStepSize(self, hmax: float) -> None:
        self.max_step = float(hmax)

    def setMinStepSize(self, hmin: float) -> None:
        self.min_step = float(hmin)

    def setMaxSteps(self, nmax: int) -> None:
        self.max_steps = int(nmax)

    def maxSteps(self) -> int:
        return self.max_steps

    def setMaxErrTestFails(self, n: int) -> None:
        self._max_err_test_fails = int(n)

    def setLinearSolverType(self, solver: str) -> None:
        self._linear_solver = str(solver).upper()

    def linearSolverType(self) -> str:
        return getattr(self, "_linear_solver", "DENSE")

    def setBandwidth(self, ku: int, kl: int) -> None:
        self._mupper = int(ku)
        self._mlower = int(kl)

    def setRootFunctionCount(self, nroots: int) -> None:
        self._n_root_functions = int(nroots)

    def applyOptions(self) -> None:
        pass

    def checkError(self, flag: int, ctMethod: str, cvodesMethod: str) -> None:
        if flag == 0:
            return
        if flag == -1:
            raise CanteraError(f"CVodesIntegrator::{ctMethod}: CVODES integrator is not initialized")
        raise CanteraError(f"CVodesIntegrator::{ctMethod}: {cvodesMethod} returned error code {flag}")

    def getErrorInfo(self, N: int = 10) -> str:
        if self._y is None or len(self._y) == 0:
            return ""
        weights = getattr(self, "_cv_ewt", None)
        if weights is None:
            weights = np.ones_like(self._y)
        errs = np.zeros_like(self._y)
        if len(self._cv_zn) > 1:
            errs = np.abs(self._cv_zn[1] * weights)
        idx = np.argsort(-errs)[:N]
        return "".join(f"{i}: {errs[i]}\n" for i in idx)

    def solverStats(self) -> dict:
        return {
            "steps": int(self._cv_nst),
            "step_solve_fails": int(self._cv_nonlinear_failures),
            "rhs_evals": int(self._nevals),
            "nonlinear_iters": int(getattr(self, "_cv_nonlinear_iters", self._nevals)),
            "nonlinear_conv_fails": int(self._cv_nonlinear_failures),
            "err_test_fails": int(self._cv_error_test_failures),
            "last_order": int(self._cv_order),
            "stab_order_reductions": 0,
            "jac_evals": int(self._cv_jacobian_evaluations),
            "lin_solve_setups": int(self._cv_linear_setups),
            "lin_rhs_evals": 0,
            "lin_iters": 0,
            "lin_conv_fails": 0,
            "prec_evals": 0,
            "prec_solves": 0,
            "jt_vec_setup_evals": 0,
            "jt_vec_prod_evals": 0,
        }


class _DAENonlinearFailure(NumericsError):
    """Recoverable Newton failure that should trigger a smaller DAE step."""


class _DAEResidualRecoverable(_DAENonlinearFailure):
    """Recoverable residual evaluation failure reported by FuncEval."""


class IdasIntegrator(Integrator):
    """Python residual DAE integrator for ``F(t, y, ydot) = 0``.

    Residual systems use variable-step BDF orders 1--5, correction-based local
    error estimates, a damped Newton corrector, and optimized NumPy/SciPy linear
    algebra. FuncEval differential/algebraic identifiers and algebraic error-test
    suppression follow IDASetId / IDASetSuppressAlg semantics from SUNDIALS IDAS.
    """

    def __init__(self, func: Optional[Any] = None):
        super().__init__(func)
        self._res: Optional[Callable] = None
        self._ydot: Optional[np.ndarray] = None
        self._dae_max_steps = 20000
        self._dae_init_step = 1e-14
        self._max_nonlin_iterations = 0
        self._max_nonlin_conv_failures = -1
        self._max_err_test_fails = -1
        self._dae_max_newton = 3
        self._dae_max_conv_failures = 10
        self._dae_max_error_test_failures = 10
        self._idas_initialized = False
        self._dae_history: List[Tuple[float, np.ndarray]] = []
        self._dae_order = 1
        self._dae_max_order = 5
        self._dae_steps_at_order = 0
        self._dae_nst = 0
        self._dae_startup_phase = True
        self._dae_h = self._dae_init_step
        self._dae_ids: Optional[np.ndarray] = None
        self._dae_include_algebraic_in_error_test = True
        self._linear_solver = "DENSE"
        self._np = 0
        self._error_message = ""
        self._t0 = 0.0
        self._tInteg = 0.0
        self._last_errs: Optional[np.ndarray] = None
        self._last_errw: Optional[np.ndarray] = None
        # Detailed stats counters matching native Cantera
        self._dae_nfe = 0
        self._dae_nni = 0
        self._dae_ncf = 0
        self._dae_netf = 0
        self._dae_nje = 0
        self._dae_nsetups = 0

    def setTolerances(self, reltol: float, abstol: Union[float, Sequence[float], np.ndarray]) -> None:
        if reltol > 0:
            self.rtol = float(reltol)
        if isinstance(abstol, (int, float)) and abstol > 0:
            self.atol = float(abstol)
        elif isinstance(abstol, (list, tuple, np.ndarray)):
            self.atol = np.asarray(abstol, dtype=float)

    def setSensitivityTolerances(self, reltol: float, abstol: float) -> None:
        if reltol > 0:
            self.rtol_sens = float(reltol)
        if abstol > 0:
            self.atol_sens = float(abstol)

    def setLinearSolverType(self, linearSolverType: str) -> None:
        self._linear_solver = str(linearSolverType)

    def initialize(self, t0: float, res_func_or_y0: Optional[Any] = None,
                   y0: Optional[np.ndarray] = None, ydot0: Optional[np.ndarray] = None):
        self._time = float(t0)
        self._t0 = float(t0)
        self._tInteg = float(t0)
        self._nevals = 0
        self._dae_history = []
        self._dae_order = 1
        self._dae_steps_at_order = 0
        self._dae_nst = 0
        self._dae_nfe = 0
        self._dae_nni = 0
        self._dae_ncf = 0
        self._dae_netf = 0
        self._dae_nje = 0
        self._dae_nsetups = 0
        self._dae_startup_phase = True
        self._dae_h = 0.0
        self._dae_ids = None
        self._idas_initialized = False
        self._dae_max_error_test_failures = 10
        self._last_errs = None
        self._last_errw = None

        # Check for sensitivity parameters: native Cantera explicitly throws
        if res_func_or_y0 is not None and not isinstance(res_func_or_y0, (np.ndarray, list, tuple)):
            nparams = getattr(res_func_or_y0, "nparams", None)
            if callable(nparams) and nparams() > 0:
                raise CanteraError("IdasIntegrator::initialize: Sensitivity analysis for DAE systems is not fully implemented")
        elif self._func is not None:
            nparams = getattr(self._func, "nparams", None)
            if callable(nparams) and nparams() > 0:
                raise CanteraError("IdasIntegrator::initialize: Sensitivity analysis for DAE systems is not fully implemented")

        if ydot0 is not None:
            self._res = res_func_or_y0
            self._y = np.asarray(y0, dtype=float).copy()
            self._ydot = np.asarray(ydot0, dtype=float).copy()
        elif y0 is not None:
            if isinstance(res_func_or_y0, (np.ndarray, list, tuple)):
                self._y = np.asarray(res_func_or_y0, dtype=float).copy()
                self._ydot = np.asarray(y0, dtype=float).copy()
            else:
                self._res = res_func_or_y0
                self._y = np.asarray(y0, dtype=float).copy()
                self._ydot = np.zeros_like(self._y)
        else:
            if self._func is not None and isinstance(res_func_or_y0, (np.ndarray, list, tuple)):
                self._res = self._func if (
                    hasattr(self._func, "evalDae") or hasattr(self._func, "eval_dae")
                ) else None
                self._y = np.asarray(res_func_or_y0, dtype=float).copy()
                self._ydot = np.zeros_like(self._y)
            elif res_func_or_y0 is not None:
                self._res = res_func_or_y0
                neq = getattr(self._res, "nEquations", None)
                try:
                    neq = neq() if callable(neq) else None
                except NotImplementedError:
                    neq = None
                if neq is None:
                    neq = getattr(self._res, "neq", None)
                    try:
                        neq = neq() if callable(neq) else None
                    except NotImplementedError:
                        neq = None
                if neq is None or int(neq) <= 0:
                    raise NumericsError(
                        "DAE residual object must report a positive equation count"
                    )
                self._y = np.zeros(int(neq), dtype=float)
                self._ydot = np.zeros_like(self._y)
                self._load_dae_component_ids()
                get_dae_state = getattr(self._res, "getStateDae", None)
                if callable(get_dae_state):
                    get_dae_state(self._y, self._ydot)
                else:
                    get_state = getattr(self._res, "getState", None)
                    if not callable(get_state):
                        raise NumericsError(
                            "DAE residual object must provide getStateDae(y, ydot)"
                        )
                    get_state(self._y)

        if self._y is not None:
            if self._ydot is None or self._ydot.shape != self._y.shape:
                self._ydot = np.zeros_like(self._y)
            if self._dae_ids is None:
                self._load_dae_component_ids()
            self._calc_ic()
            self._dae_history = [(self._time, self._y.copy())]
            self._idas_initialized = True
            self.applyOptions()

    def _calc_ic(self) -> None:
        """Compute consistent initial conditions for index-1 DAE systems."""
        if self._res is None and self._func is None:
            return
        if self._y is None or self._ydot is None:
            return
        try:
            r0 = self._dae_residual(self._time, self._y, self._ydot)
        except Exception:
            return
        tol = max(1e-10, float(np.min(self.atol)) if isinstance(self.atol, np.ndarray) else float(self.atol))
        if np.max(np.abs(r0)) <= tol:
            return

        # Solve F(t0, y, ydot) = 0 for algebraic y and differential ydot
        n = len(self._y)
        ids = self._dae_ids if self._dae_ids is not None else np.ones(n, dtype=float)
        alg_mask = ids == 0.0
        diff_mask = ids != 0.0

        for _ in range(10):
            r = self._dae_residual(self._time, self._y, self._ydot)
            if np.max(np.abs(r)) <= tol:
                break
            J = np.empty((n, n), dtype=float)
            for j in range(n):
                if alg_mask[j]:
                    delta = math.sqrt(np.finfo(float).eps) * max(1.0, abs(self._y[j]))
                    y_pert = self._y.copy()
                    y_pert[j] += delta
                    r_pert = self._dae_residual(self._time, y_pert, self._ydot)
                    J[:, j] = (r_pert - r) / delta
                else:
                    delta = math.sqrt(np.finfo(float).eps) * max(1.0, abs(self._ydot[j]))
                    ydot_pert = self._ydot.copy()
                    ydot_pert[j] += delta
                    r_pert = self._dae_residual(self._time, self._y, ydot_pert)
                    J[:, j] = (r_pert - r) / delta
            try:
                dz = np.linalg.solve(J, -r)
            except np.linalg.LinAlgError:
                break
            for j in range(n):
                if alg_mask[j]:
                    self._y[j] += dz[j]
                else:
                    self._ydot[j] += dz[j]

    def reinitialize(self, t0: float, func_or_y0: Optional[Any] = None) -> None:
        self._t0 = float(t0)
        self._time = float(t0)
        self._tInteg = float(t0)
        fn = func_or_y0 if func_or_y0 is not None else self._res
        if fn is not None:
            get_dae_state = getattr(fn, "getStateDae", None)
            if callable(get_dae_state) and self._y is not None and self._ydot is not None:
                get_dae_state(self._y, self._ydot)
        if self._y is not None:
            self._dae_history = [(self._time, self._y.copy())]
        self._dae_order = 1
        self._dae_steps_at_order = 0
        self._dae_startup_phase = True
        self.applyOptions()

    def _load_dae_component_ids(self) -> None:
        """Load IDASetId values: one for differential, zero for algebraic."""
        if self._y is None:
            raise NumericsError("Cannot load DAE component IDs before state initialization")
        ids = np.ones_like(self._y)
        if self._res is not None:
            get_constraints = getattr(self._res, "getConstraints", None)
            if callable(get_constraints):
                ids.fill(np.nan)
                try:
                    get_constraints(ids)
                except NotImplementedError as exc:
                    raise NumericsError(
                        "DAE FuncEval must implement getConstraints(ids)"
                    ) from exc
                if (ids.shape != self._y.shape
                        or not np.all(np.isfinite(ids))
                        or not np.all((ids == 0.0) | (ids == 1.0))):
                    raise NumericsError(
                        "DAE component identifiers must contain only 0 (algebraic) "
                        "or 1 (differential)"
                    )
        self._dae_ids = ids

    def _dae_advance_one_step(self, tout: float) -> float:
        """Advance exactly one internal BDF step towards tout."""
        hmax = self.max_step if self.max_step > 0 else math.inf
        h = self._dae_h if self._dae_h > 0.0 else self._dae_init_step
        if self._dae_nst == 0:
            resolution_floor = 2.0 * math.sqrt(np.finfo(float).eps) * max(
                1.0, abs(self._time), abs(tout)
            )
            h = max(h, resolution_floor)
        h = min(hmax, h, tout - self._time)

        ncf = nef = 0
        while True:
            if self._dae_nst >= min(self.max_steps, self._dae_max_steps):
                raise CanteraError(
                    f"IdasIntegrator::step: Maximum number of timesteps ({self.max_steps}) "
                    f"taken without reaching output time ({tout}).\nCurrent integrator time: {self._time}"
                )
            h = min(h, tout - self._time)
            if h <= np.finfo(float).eps * max(1.0, abs(self._time)):
                raise CanteraError("IdasIntegrator::step: reached minimum step size")

            q = min(self._dae_order, self._dae_max_order, len(self._dae_history))
            tnew = self._time + h
            prediction = self._dae_predict(tnew, q)
            coeff, ck = self._dae_bdf_coefficients(
                tnew, h, [item[0] for item in self._dae_history[:q]]
            )
            try:
                candidate, candidate_dot = self._dae_implicit_bdf(
                    tnew, h, coeff, self._dae_history[:q], prediction
                )
            except _DAENonlinearFailure as exc:
                ncf += 1
                self._dae_startup_phase = False
                if ncf >= self._dae_max_conv_failures:
                    raise CanteraError(
                        "IdasIntegrator::step: nonlinear convergence exceeded "
                        f"{self._dae_max_conv_failures} failures"
                    ) from exc
                if q > 1:
                    self._dae_order = q - 1
                    self._dae_steps_at_order = 0
                h *= 0.25
                if h <= np.finfo(float).eps * max(1.0, abs(self._time)):
                    raise CanteraError(
                        "IdasIntegrator::step: Newton solve failed at minimum step size"
                    ) from exc
                continue

            err_vec = candidate - prediction
            err = ck * self._dae_weighted_rms(
                err_vec, candidate,
                include_algebraic=self._dae_include_algebraic_in_error_test,
            )
            atol = np.asarray(self.atol, dtype=float)
            w_vec = 1.0 / (atol + self.rtol * np.abs(candidate))
            self._last_errs = ck * err_vec
            self._last_errw = w_vec

            if err <= 1.0:
                self._time = tnew
                self._tInteg = tnew
                if abs(self._time - tout) <= 8.0 * np.finfo(float).eps * max(1.0, abs(tout)):
                    self._time = tout
                    self._tInteg = tout
                self._y = candidate
                self._ydot = candidate_dot
                self._dae_history.insert(0, (self._time, self._y.copy()))
                del self._dae_history[6:]
                self._dae_nst += 1
                self._dae_steps_at_order += 1
                qnext = q
                if self._dae_startup_phase and self._dae_nst > 1:
                    if q < self._dae_max_order:
                        qnext = q + 1
                        h = min(hmax, 2.0 * h)
                        self._dae_steps_at_order = 0
                    else:
                        self._dae_startup_phase = False
                elif (q < self._dae_max_order and len(self._dae_history) >= q + 2
                        and self._dae_steps_at_order >= q + 1 and err < 0.05):
                    qnext = q + 1
                    self._dae_steps_at_order = 0
                self._dae_order = qnext
                eta = self._dae_step_factor(err, qnext)
                self._dae_h = min(hmax, h * eta)
                return self._time
            else:
                nef += 1
                self._dae_netf += 1
                if self._dae_startup_phase:
                    self._dae_startup_phase = False
                h, qnext = self._dae_after_error_test_failure(h, err, q, nef)
                if qnext != q:
                    self._dae_order = qnext
                    self._dae_steps_at_order = 0
                continue

    def step(self, tout: float) -> float:
        """Advance one internal step towards tout."""
        tout = float(tout)
        if tout < self._time:
            raise CanteraError(
                f"IdasIntegrator::step: Cannot integrate backwards in time.\n"
                f"Requested time {tout} < current time {self._time}"
            )
        if tout == self._time:
            return self._time
        return self._dae_advance_one_step(tout)

    def integrate(self, tout: float) -> float:
        """Integrate until tout, interpolating to tout if the internal step passed it."""
        if (self._res is None and self._func is None) or self._y is None:
            raise CanteraError("IdasIntegrator::integrate: IDAS integrator not initialized")
        tout = float(tout)
        if tout < self._time:
            raise CanteraError(
                f"IdasIntegrator::integrate: Cannot integrate backwards in time.\n"
                f"Requested time {tout} < current time {self._time}"
            )
        if tout == self._time:
            return self._time

        if self._res is None and self._func is not None and (
            hasattr(self._func, "eval_f") or hasattr(self._func, "eval")
            or callable(self._func)
        ):
            ode = CVodesIntegrator(self._func)
            ode.setTolerances(self.rtol, self.atol)
            ode.setMaxSteps(self.max_steps)
            ode.initialize(self._time, self._y)
            ode.integrate(tout)
            self._y = ode.solution()
            self._ydot = ode._eval_rhs(tout, self._y)
            self._nevals += ode.nEvals()
            self._time = tout
            self._tInteg = tout
            return self._time

        nsteps = 0
        while self._tInteg < tout:
            if nsteps >= min(self.max_steps, self._dae_max_steps):
                raise CanteraError(
                    f"IdasIntegrator::integrate: Maximum number of timesteps ({self.max_steps}) "
                    f"taken without reaching output time ({tout}).\nCurrent integrator time: {self._time}"
                )
            self._dae_advance_one_step(tout)
            nsteps += 1

        if abs(self._tInteg - tout) <= 1e-12 * max(1.0, abs(tout)):
            self._time = tout
        else:
            q = min(self._dae_order, len(self._dae_history) - 1)
            self._y = self._dae_predict(tout, q)
            self._time = tout
        return self._time

    def _dae_after_error_test_failure(self, h: float, error: float, order: int,
                                      failures: int) -> Tuple[float, int]:
        """Apply IDA's first/second/later error-test retry policy."""
        if failures == 1:
            eta = 0.9 * (2.0 * error + 1.0e-4) ** (-1.0 / (order + 1))
            eta = min(0.9, max(0.25, eta))
            next_order = order
        elif failures == 2:
            eta = 0.25
            next_order = order
        elif failures < self._dae_max_error_test_failures:
            eta = 0.25
            next_order = 1
        else:
            raise NumericsError(
                "IdasIntegrator: exceeded "
                f"{self._dae_max_error_test_failures} error-test failures"
            )
        return h * eta, next_order

    @staticmethod
    def _dae_step_factor(error: float, order: int, failed: bool = False) -> float:
        """Apply IDA's bounded error-based step-size growth/reduction pattern."""
        if error <= 0.0:
            return 2.0 if not failed else 0.5
        eta = (2.0 * error + 1.0e-4) ** (-1.0 / (order + 1))
        if failed:
            return min(0.8, max(0.1, eta))
        if eta >= 2.0:
            return min(2.0, eta)
        if eta <= 1.0:
            return min(0.9, max(0.5, eta))
        return 1.0

    def _dae_predict(self, tnew: float, order: int) -> np.ndarray:
        """Extrapolate the accepted history to the next IDA predictor time."""
        count = min(order + 1, len(self._dae_history))
        if count <= 1:
            y = self._dae_history[0][1].copy()
            if self._ydot is not None:
                y += (tnew - self._dae_history[0][0]) * self._ydot
            return y
        points = self._dae_history[:count]
        prediction = np.zeros_like(points[0][1])
        for j, (tj, yj) in enumerate(points):
            basis = 1.0
            for m, (tm, _) in enumerate(points):
                if m != j:
                    basis *= (tnew - tm) / (tj - tm)
            prediction += basis * yj
        return prediction

    @staticmethod
    def _dae_bdf_coefficients(tnew: float, h: float,
                              history_times: Sequence[float]) -> Tuple[np.ndarray, float]:
        """Build variable-step BDF weights and IDA's leading error coefficient."""
        order = len(history_times)
        if not 1 <= order <= 5 or h <= 0.0:
            raise NumericsError("BDF order must be 1--5 and step size positive")
        nodes = np.asarray([0.0] + [(t - tnew) / h for t in history_times], dtype=float)
        vandermonde = np.empty((order + 1, order + 1), dtype=float)
        for power in range(order + 1):
            vandermonde[power, :] = nodes ** power
        derivative = np.zeros(order + 1, dtype=float)
        derivative[1] = 1.0
        weights = np.linalg.solve(vandermonde, derivative)
        truncation = sum(weights[j] * nodes[j] ** (order + 1)
                         for j in range(order + 1)) / math.factorial(order + 1)
        ck = min(1.0 / (order + 1), max(abs(float(truncation)), np.finfo(float).eps))
        return weights, ck

    def _dae_residual(self, t: float, y: np.ndarray, ydot: np.ndarray) -> np.ndarray:
        """Evaluate a residual callback."""
        fn = self._res if self._res is not None else self._func
        self._nevals += 1
        self._dae_nfe += 1
        if fn is None:
            raise NumericsError("No DAE residual function is configured")
        for name in ("evalDae", "eval_dae"):
            method = getattr(fn, name, None)
            if method is not None:
                residual = np.full_like(y, np.nan)
                no_throw = getattr(fn, "evalDaeNoThrow", None)
                if callable(no_throw) and name == "evalDae":
                    status = int(no_throw(t, y, ydot, residual))
                    if status > 0:
                        raise _DAEResidualRecoverable(
                            "FuncEval::evalDaeNoThrow reported a recoverable residual error"
                        )
                    if status < 0:
                        errors = getattr(fn, "getErrors", lambda: [])()
                        raise NumericsError(
                            "FuncEval::evalDaeNoThrow reported an unrecoverable residual error"
                            + (f": {errors}" if errors else "")
                        )
                    result = None
                else:
                    result = method(t, y, ydot, residual)
                if result is not None:
                    returned = np.asarray(result)
                    if returned.shape == y.shape:
                        residual[:] = returned
                if np.any(np.isnan(residual)):
                    raise NumericsError(
                        "DAE residual callback did not fill every component"
                    )
                return residual
        if callable(fn):
            try:
                value = fn(t, y, ydot)
            except TypeError as exc:
                raise NumericsError(
                    "DAE residual callable must accept (t, y, ydot)"
                ) from exc
            residual = np.asarray(value, dtype=float)
            if residual.shape != y.shape:
                raise NumericsError("DAE residual dimensions do not match the state")
            return residual
        raise NumericsError("No evalDae/eval_dae method or callable DAE residual")

    def _dae_weighted_rms(self, value: np.ndarray, state: np.ndarray,
                          include_algebraic: bool = True) -> float:
        atol = np.asarray(self.atol, dtype=float)
        if atol.ndim and atol.shape != state.shape:
            raise NumericsError("Absolute tolerance vector does not match the DAE state")
        mask = np.ones(state.shape, dtype=bool)
        if not include_algebraic and self._dae_ids is not None:
            mask = self._dae_ids != 0.0
        if not np.any(mask):
            return 0.0
        weights = 1.0 / (atol + self.rtol * np.abs(state))
        scaled = value[mask] * weights[mask]
        return math.sqrt(float(np.mean(scaled ** 2)))

    def _dae_implicit_bdf(self, tnew: float, h: float, coeff: np.ndarray,
                          history: Sequence[Tuple[float, np.ndarray]],
                          prediction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve one variable-step BDF residual step by damped Newton iteration."""
        yold = history[0][1]
        n = len(yold)
        guess = prediction.copy()

        if len(coeff) == 2 and self._ydot is not None:
            predictor_dot = self._ydot.copy()
        else:
            anchor = history[0][1]
            predictor_numerator = coeff[0] * (prediction - anchor)
            for j, (_, yj) in enumerate(history, start=1):
                predictor_numerator += coeff[j] * (yj - anchor)
            predictor_dot = predictor_numerator / h

        def evaluate(candidate: np.ndarray) -> np.ndarray:
            ydot = predictor_dot + coeff[0] * (candidate - prediction) / h
            return self._dae_residual(tnew, candidate, ydot)

        residual = evaluate(guess)
        del_prev = None
        tol = 0.33
        for m in range(max(4, self._dae_max_newton)):
            self._dae_nni += 1
            norm = self._dae_weighted_rms(residual, guess)
            if norm <= max(0.03, 0.1 * self.rtol):
                ydot = predictor_dot + coeff[0] * (guess - prediction) / h
                return guess, ydot
            jac = np.empty((n, n), dtype=float)
            self._dae_nje += 1
            self._dae_nsetups += 1
            for j in range(n):
                delta = math.sqrt(np.finfo(float).eps) * max(1.0, abs(guess[j]))
                perturbed = guess.copy()
                perturbed[j] += delta
                jac[:, j] = (evaluate(perturbed) - residual) / delta
            try:
                correction = np.linalg.solve(jac, -residual)
            except np.linalg.LinAlgError:
                self._dae_ncf += 1
                raise _DAENonlinearFailure("Python residual DAE Newton matrix is singular")

            delnrm = self._dae_weighted_rms(correction, guess, include_algebraic=self._dae_include_algebraic_in_error_test)
            if m == 0:
                if delnrm <= 0.0001 * tol or delnrm <= max(0.03, 0.1 * self.rtol):
                    guess += correction
                    ydot = predictor_dot + coeff[0] * (guess - prediction) / h
                    return guess, ydot
                ss = 0.5
            else:
                rate = max(1e-4, delnrm / del_prev) if del_prev and del_prev > 0 else 1.0
                ss = rate / (1.0 - rate) if rate < 1.0 else 1.0

            if ss * delnrm <= tol or delnrm <= max(0.03, 0.1 * self.rtol):
                guess += correction
                ydot = predictor_dot + coeff[0] * (guess - prediction) / h
                return guess, ydot

            damping = 1.0
            accepted = False
            while damping >= 1.0 / 64.0:
                trial = guess + damping * correction
                trial_residual = evaluate(trial)
                trial_norm = self._dae_weighted_rms(trial_residual, trial)
                if trial_norm < norm or delnrm <= tol:
                    guess, residual = trial, trial_residual
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                self._dae_ncf += 1
                raise _DAENonlinearFailure(
                    "Python residual DAE Newton iteration did not converge"
                )
            del_prev = delnrm
        self._dae_ncf += 1
        raise _DAENonlinearFailure(
            "Python residual DAE Newton iteration exceeded its limit"
        )

    def solution(self, k: Optional[int] = None) -> Union[float, np.ndarray]:
        if self._y is None:
            return 0.0 if k is not None else np.array([])
        return float(self._y[k]) if k is not None else self._y.copy()

    def nEquations(self) -> int:
        return len(self._y) if self._y is not None else 0

    def maxOrder(self) -> int:
        return self._dae_max_order

    def maxNonlinIterations(self) -> int:
        return self._max_nonlin_iterations

    def setMaxNonlinIterations(self, n: int) -> None:
        self._max_nonlin_iterations = int(n)
        self._dae_max_newton = int(n) if int(n) > 0 else 3

    def maxNonlinConvFailures(self) -> int:
        return self._max_nonlin_conv_failures

    def setMaxNonlinConvFailures(self, n: int) -> None:
        self._max_nonlin_conv_failures = int(n)
        self._dae_max_conv_failures = int(n) if int(n) > 0 else 10

    def setMaxErrTestFails(self, n: int) -> None:
        self._max_err_test_fails = int(n)
        if self._idas_initialized:
            self._dae_max_error_test_failures = int(n)

    def algebraicInErrorTest(self) -> bool:
        return self._dae_include_algebraic_in_error_test

    def includeAlgebraicInErrorTest(self, include: bool = True) -> None:
        self._dae_include_algebraic_in_error_test = bool(include)

    def setMaxOrder(self, order: int) -> None:
        self._dae_max_order = max(1, min(5, int(order)))
        self._dae_order = min(self._dae_order, self._dae_max_order)

    def lastOrder(self) -> int:
        return self._dae_order

    def setMaxStepSize(self, hmax: float) -> None:
        self.max_step = float(hmax)

    def setMaxSteps(self, nmax: int) -> None:
        self.max_steps = int(nmax)
        self._dae_max_steps = int(nmax)

    def maxSteps(self) -> int:
        return self.max_steps

    def nSensParams(self) -> int:
        return self._np

    def sensitivity(self, k: int, p: int) -> float:
        if self._time == self._t0:
            return 0.0
        if k >= self.nEquations():
            raise CanteraError(f"IdasIntegrator::sensitivity: sensitivity: k out of range ({k})")
        if p >= self.nSensParams():
            raise CanteraError(f"IdasIntegrator::sensitivity: sensitivity: p out of range ({p})")
        return 0.0

    def getErrorInfo(self, N: int = 10) -> str:
        if self._last_errs is None or self._last_errw is None or self._y is None:
            return ""
        n = len(self._y)
        weighted = []
        for i in range(n):
            w = self._last_errw[i] if i < len(self._last_errw) else 1.0
            e = self._last_errs[i] if i < len(self._last_errs) else 0.0
            weighted.append((abs(e * w), e * w, i))
        weighted.sort(key=lambda x: x[0], reverse=True)
        N = min(int(N), n)
        lines = []
        for i in range(N):
            lines.append(f"{weighted[i][2]}: {weighted[i][1]}\n")
        return "".join(lines)

    def currentTime(self) -> float:
        return self._time

    def setMethod(self, t: Any) -> None:
        # IDA only has the BDF method (MethodType::BDF_Method == 1)
        s = str(t).upper()
        if s not in ("1", "BDF", "BDF_METHOD", "METHODTYPE.BDF_METHOD"):
            raise CanteraError("IdasIntegrator::setMethod: unknown method")

    def applyOptions(self) -> None:
        if self._linear_solver.upper() not in ("DENSE", "GMRES"):
            raise CanteraError(
                f"IdasIntegrator::applyOptions: unsupported linear solver flag '{self._linear_solver}'"
            )
        if self.max_steps > 0:
            self._dae_max_steps = self.max_steps
        if self._max_nonlin_iterations > 0:
            self._dae_max_newton = self._max_nonlin_iterations
        if self._max_nonlin_conv_failures > 0:
            self._dae_max_conv_failures = self._max_nonlin_conv_failures

    def sensInit(self, t0: float, func: Any) -> None:
        nparams = getattr(func, "nparams", lambda: 0)()
        if nparams > 0:
            raise CanteraError(
                "IdasIntegrator::sensInit: Sensitivity analysis for DAE systems is not fully implemented"
            )

    def checkError(self, flag: int, ctMethod: str, idaMethod: str) -> None:
        if flag == 0:
            return
        elif flag == -1:
            raise CanteraError(f"IdasIntegrator::{ctMethod}: IDAS integrator is not initialized")
        else:
            raise CanteraError(
                f"IdasIntegrator::{ctMethod}: {idaMethod} returned error code {flag}:\n{self._error_message}"
            )

    def solverStats(self) -> dict:
        return {
            "steps": self._dae_nst,
            "step_solve_fails": self._dae_ncf,
            "res_evals": self._dae_nfe,
            "rhs_evals": self._dae_nfe,
            "nonlinear_iters": self._dae_nni,
            "nonlinear_conv_fails": self._dae_ncf,
            "err_test_fails": self._dae_netf,
            "last_order": self._dae_order,
            "jac_evals": self._dae_nje,
            "lin_solve_setups": self._dae_nsetups,
            "lin_rhs_evals": 0,
            "lin_iters": 0,
            "lin_conv_fails": 0,
            "prec_evals": 0,
            "prec_solves": 0,
            "jt_vec_setup_evals": 0,
            "jt_vec_prod_evals": 0,
        }


class SteadyStateSystem:
    """Nonlinear steady-state solver and time-stepping system."""

    def __init__(self, n: int = 1):
        self._n = int(n)
        self._state = np.zeros(self._n, dtype=float)
        self._guess = np.zeros(self._n, dtype=float)
        self._stats: dict = {}
        self._linear_solver = "Dense"
        self._max_steps = 1000
        self._dt = 1e-4

    def setInitialGuess(self, guess: np.ndarray) -> None:
        self._guess = np.asarray(guess, dtype=float).copy()
        self._state = self._guess.copy()

    def getState(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        if state is not None:
            state[:] = self._state
        return self._state.copy()

    def ssnorm(self, x: np.ndarray, r: np.ndarray) -> float:
        return _python_l2_norm(r)

    def tsnorm(self, x: np.ndarray, r: np.ndarray, step: float) -> float:
        return _python_l2_norm(r)

    def bandwidth(self) -> int:
        return 0

    def componentTableLabel(self, k: int) -> str:
        return f"Component_{k}"

    def newton(self, max_iter: int = 100) -> int:
        return 0

    def setLinearSolver(self, solver: str) -> None:
        self._linear_solver = str(solver)

    def linearSolver(self) -> str:
        return self._linear_solver

    def resetBadValues(self, y: np.ndarray) -> None:
        pass

    def evalSSJacobian(self) -> None:
        pass

    def factorizeJacobian(self) -> None:
        pass

    def gridSize(self) -> int:
        return self._n

    def maxTimeStepCount(self) -> int:
        return self._max_steps

    def parseTimeStepGrowthStrategy(self, strategy: str) -> int:
        return 0

    def rdt(self) -> float:
        return 1.0 / max(self._dt, 1e-12)

    def recordFactorization(self) -> None:
        pass

    def recordLinearSolve(self) -> None:
        pass

    def saveStats(self) -> None:
        pass

    def clearStats(self) -> None:
        self._stats.clear()

    def clearDebugFile(self) -> None:
        pass

    def setInterrupt(self, interrupt: Any) -> None:
        pass

    def setJacAge(self, age: int) -> None:
        pass

    def setJacobianPerturbation(self, pert: float) -> None:
        pass

    def setMaxTimeStep(self, dt: float) -> None:
        pass

    def setMaxTimeStepCount(self, count: int) -> None:
        self._max_steps = int(count)

    def setMinTimeStep(self, dt: float) -> None:
        pass

    def setTimeStep(self, dt: float) -> None:
        self._dt = float(dt)

    def setTimeStepCallback(self, cb: Any) -> None:
        pass

    def setTimeStepFactor(self, factor: float) -> None:
        pass

    def setTimeStepGrowthFactor(self, factor: float) -> None:
        pass

    def setTimeStepGrowthStrategy(self, strategy: str) -> None:
        pass

    def solverStats(self) -> dict:
        return self._stats

    def timeStepGrowthFactor(self) -> float:
        return 1.2

    def timeStepGrowthStrategy(self) -> int:
        return 0

    def timeStepGrowthStrategyName(self) -> str:
        return "default"

    def transientMask(self) -> np.ndarray:
        return np.ones(self._n, dtype=float)


class EigenSparseJacobian(SystemJacobian):
    def updatePreconditioner(self, gamma: float = 1.0, atol: float = 1e-12) -> None:
        pass

    def updateTransient(self, gamma: float = 1.0, ydot: Optional[np.ndarray] = None) -> None:
        pass

    """Sparse Jacobian matrix backed by scipy.sparse."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self._dim = int(dim)
        import scipy.sparse as sp
        self._matrix = sp.lil_matrix((self._dim, self._dim), dtype=float)

    def initialize(self, dim: int) -> None:
        self._dim = int(dim)
        import scipy.sparse as sp
        self._matrix = sp.lil_matrix((self._dim, self._dim), dtype=float)

    def jacobian(self):
        return self._matrix

    def matrix(self):
        return self._matrix

    def printJacobian(self) -> None:
        pass

    def reset(self) -> None:
        self._matrix.data[:] = 0

    def setFromTriplets(self, rows: Sequence[int], cols: Sequence[int], data: Sequence[float]) -> None:
        import scipy.sparse as sp
        self._matrix = sp.coo_matrix((data, (rows, cols)), shape=(self._dim, self._dim)).tolil()


class EigenSparseDirectJacobian(EigenSparseJacobian):
    """Direct sparse solver Jacobian."""

    def type(self) -> str:
        return "EigenSparseDirectJacobian"


class SystemJacobianFactory:
    """Factory for SystemJacobian instances."""

    _instance = None

    @classmethod
    def factory(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def deleteFactory(cls):
        cls._instance = None


class NonlinearSolver:
    """Base nonlinear solver."""
    def __init__(self, sys: Optional[Any] = None):
        self._sys = sys

    def solve(self, y0: np.ndarray) -> np.ndarray:
        return y0.copy()


class ResidEval:
    """Residual evaluation for DAE systems."""
    def nEquations(self) -> int:
        return 1

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray) -> None:
        res[:] = 0.0


class TimeStepError(NumericsError):
    """Exception raised on integration or time step failure."""
    def getClass(self) -> str:
        return "TimeStepError"


class SundialsContext:
    """SUNDIALS execution context singleton."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class Array2D:
    """2D array container compatible with Cantera's Array2D."""

    def __init__(self, rows: int = 0, cols: int = 0, val: float = 0.0):
        self._data = np.full((rows, cols), val, dtype=float)

    def nRows(self) -> int:
        return self._data.shape[0]

    def nColumns(self) -> int:
        return self._data.shape[1]

    def resize(self, rows: int, cols: int, val: float = 0.0) -> None:
        self._data = np.full((rows, cols), val, dtype=float)

    def value(self, i: int, j: int) -> float:
        return float(self._data[i, j])

    def __getitem__(self, idx):
        return self._data[idx]

    def __setitem__(self, idx, val):
        self._data[idx] = val

    def data(self) -> np.ndarray:
        return self._data

    def appendColumn(self, col: np.ndarray) -> None:
        if self._data.size == 0:
            self._data = np.asarray(col, dtype=float).reshape(-1, 1)
        else:
            self._data = np.hstack([self._data, np.asarray(col, dtype=float).reshape(-1, 1)])

    def setRow(self, i: int, row: np.ndarray) -> None:
        self._data[i, :] = row

    def getRow(self, i: int, row: np.ndarray) -> None:
        row[:] = self._data[i, :]

    def setColumn(self, j: int, col: np.ndarray) -> None:
        self._data[:, j] = col

    def getColumn(self, j: int, col: np.ndarray) -> None:
        col[:] = self._data[:, j]

    def col(self, j: int) -> np.ndarray:
        return self._data[:, j]

    def zero(self) -> None:
        self._data.fill(0.0)

    def operator(self, *args):
        return self._data


class MultiJac:
    """Multi-domain Jacobian manager."""
    def __init__(self):
        pass


# Free functions in numerics
def _DSCAL_(n, da, dx, incx):
    dx[:n] *= da
    return 0

def multiply(A: GeneralMatrix, x: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
    return A.mult(x, prod)

def increment(A: GeneralMatrix, x: np.ndarray, prod: np.ndarray) -> np.ndarray:
    prod += A.mult(x)
    return prod

def invert(A: DenseMatrix, nn: Optional[int] = None) -> None:
    A.invert()

def newIntegrator(itype: str) -> Integrator:
    if "cvode" in itype.lower():
        return CVodesIntegrator()
    elif "ida" in itype.lower():
        return IdasIntegrator()
    return CVodesIntegrator()

def newSystemJacobian(type_name: str) -> SystemJacobian:
    if "sparse" in type_name.lower():
        return EigenSparseJacobian()
    return SystemJacobian()
