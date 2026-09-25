"""Numerical linear algebra, banded solvers, Jacobians, and ODE integrators.

Provides:
- GeneralMatrix: Abstract base class for matrix operations.
- BandMatrix: Compact banded matrix storage with LU factorization and forward/back solve.
- DenseMatrix: Dense matrix storage with LU factorization, inversion, and condition numbers.
- SystemJacobian: Finite-difference and sparse block Jacobian evaluator.
- AdaptivePreconditioner: Block-diagonal and incomplete factorized preconditioners.
- FuncEval: Abstract ODE RHS evaluator.
- Integrator: Abstract base class for ODE/DAE integrators.
- CVodesIntegrator: Stiff ODE integrator wrapping variable-step BDF/Radau algorithms.
- IdasIntegrator: DAE integrator for residual differential-algebraic systems.
"""

from __future__ import annotations
import math
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.linalg

from .constants import CanteraError


class NumericsError(CanteraError):
    """Exception for numerical algebra and integration errors."""
    pass


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
        return getattr(self, '_pivots', np.arange(self.nRows(), dtype=int))

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

    def nRows(self) -> int:
        return self._n

    def nColumns(self) -> int:
        return self._m

    def resize(self, n: int, m: int, v: float = 0.0):
        self._n = int(n)
        self._m = int(m)
        self._data = np.full((self._n, self._m), float(v), dtype=float)
        self._factored = False
        self._lu = None

    def value(self, i: int, j: int) -> float:
        return float(self._data[i, j])

    def setValue(self, i: int, j: int, v: float):
        self._data[i, j] = float(v)
        self._factored = False

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
        self._factored = False

    def __call__(self, i: int, j: int) -> float:
        return float(self._data[i, j])

    def zero(self):
        self._data.fill(0.0)
        self._factored = False

    def factor(self):
        """In-place LU factorization with partial row pivoting."""
        if self._n == 0 or self._m == 0:
            return
        if self._n != self._m:
            raise NumericsError("LU factorization requires a square matrix")
        self._lu, self._ipiv = scipy.linalg.lu_factor(self._data)
        self._factored = True

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A * x = b using pre-computed LU factorization."""
        b_arr = np.asarray(b, dtype=float)
        if not self._factored or self._lu is None or self._ipiv is None:
            self.factor()
        return scipy.linalg.lu_solve((self._lu, self._ipiv), b_arr)

    def invert(self) -> np.ndarray:
        """Compute the matrix inverse."""
        if self._n != self._m:
            raise NumericsError("Matrix must be square to invert")
        return np.linalg.inv(self._data)

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
        return float(np.linalg.norm(self._data, ord=1))

    def rcond(self, a1norm: float = 0.0) -> float:
        try:
            return float(1.0 / np.linalg.cond(self._data, p=1))
        except Exception:
            return 0.0


class BandMatrix(GeneralMatrix):
    """Compact banded matrix storage compatible with LAPACK and SUNDIALS band solvers.

    Storage:
    - Number of rows: n
    - Lower bandwidth: kl
    - Upper bandwidth: ku
    - Storage layout is (2*kl + ku + 1, n) to accommodate fill-in during LU factorization.
    """

    def __init__(self, n: int = 0, kl: int = 0, ku: int = 0, v: float = 0.0):
        self._n = int(n)
        self._kl = int(kl)
        self._ku = int(ku)
        # SciPy/LAPACK band storage requires (2*kl + ku + 1, n) for LU factorization with pivoting
        self._nrows_alloc = 2 * self._kl + self._ku + 1
        self._ludata = np.full((self._nrows_alloc, self._n), float(v), dtype=float)
        self._factored = False
        self._pivots: Optional[np.ndarray] = None

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
        self._ludata = np.full((self._nrows_alloc, self._n), float(v), dtype=float)
        self._factored = False

    def bfill(self, v: float = 0.0):
        self._ludata.fill(float(v))
        self._factored = False

    def zero(self):
        self.bfill(0.0)

    def _band_row(self, i: int, j: int) -> int:
        # Row index in LAPACK band storage (with extra kl rows for pivots)
        return self._kl + self._ku + i - j

    def value(self, i: int, j: int) -> float:
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Indices ({i}, {j}) out of range for {self._n}x{self._n} matrix")
        row = self._band_row(i, j)
        if row < 0 or row >= self._nrows_alloc or j - i > self._ku or i - j > self._kl:
            return 0.0
        return float(self._ludata[row, j])

    def setValue(self, i: int, j: int, v: float):
        if i < 0 or i >= self._n or j < 0 or j >= self._n:
            raise IndexError(f"Indices ({i}, {j}) out of range")
        if j - i > self._ku or i - j > self._kl:
            if abs(v) > 1e-15:
                raise ValueError(f"Entry ({i}, {j}) is outside band ({self._kl}, {self._ku})")
            return
        row = self._band_row(i, j)
        self._ludata[row, j] = float(v)
        self._factored = False

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
                dense[i, j] = self.value(i, j)
        return dense

    def mult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.zeros(self._n, dtype=float)
        for j in range(self._n):
            bj = b[j]
            for i in range(max(0, j - self._ku), min(self._n, j + self._kl + 1)):
                res[i] += self.value(i, j) * bj
        if prod is not None:
            prod[:] = res
            return prod
        return res

    def leftMult(self, b: np.ndarray, prod: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.zeros(self._n, dtype=float)
        for i in range(self._n):
            bi = b[i]
            for j in range(max(0, i - self._kl), min(self._n, i + self._ku + 1)):
                res[j] += self.value(i, j) * bi
        if prod is not None:
            prod[:] = res
            return prod
        return res

    def factor(self):
        """In-place LU factorization with partial row pivoting."""
        if self._n == 0:
            return
        # Pack into SciPy / LAPACK gbtrf format
        # gbtrf expects ab of shape (2*kl + ku + 1, n)
        ab = self._ludata.copy()
        lu, piv, info = scipy.linalg.lapack.dgbtrf(ab, self._kl, self._ku)
        if info != 0:
            raise NumericsError(f"dgbtrf LU factorization failed with info={info}")
        self._ludata = lu
        self._pivots = piv
        self._factored = True

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Solve A * x = b using banded LU factorization (LAPACK dgbtrs)."""
        if not self._factored or self._pivots is None:
            self.factor()
        b_arr = np.asarray(b, dtype=float).copy()
        x, info = scipy.linalg.lapack.dgbtrs(self._ludata, self._kl, self._ku, b_arr, self._pivots)
        if info != 0:
            raise NumericsError(f"dgbtrs banded solve failed with info={info}")
        return x

    def oneNorm(self) -> float:
        dense = self.to_dense()
        return float(np.linalg.norm(dense, ord=1))

    def rcond(self, a1norm: float = 0.0) -> float:
        try:
            dense = self.to_dense()
            return float(1.0 / np.linalg.cond(dense, p=1))
        except Exception:
            return 0.0


class FuncEval:
    def clearErrors(self) -> None:
        self._errors = []

    def evalNoThrow(self, t: float, y: np.ndarray, ydot: np.ndarray) -> bool:
        try:
            self.eval(t, y, ydot)
            return True
        except Exception:
            return False

    def evalRootFunctions(self, t: float, y: np.ndarray, gout: np.ndarray) -> None:
        pass

    def evalRootFunctionsNoThrow(self, t: float, y: np.ndarray, gout: np.ndarray) -> bool:
        return True

    def getConstraints(self, c: np.ndarray) -> None:
        c.fill(0)

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
        return self.nEquations()

    def nEquations(self) -> int:
        raise NotImplementedError

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
        return np.linalg.solve(self._mat, b)


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
        else:
            self._blocks = [np.eye(sz, dtype=float) for sz in networkSize_or_blocks]

    def factorize(self, full_jac: Optional[np.ndarray] = None, offsets: Optional[List[int]] = None):
        if full_jac is not None and offsets is not None:
            for idx, sz in enumerate([b.shape[0] for b in self._blocks]):
                off = offsets[idx]
                block = full_jac[off:off + sz, off:off + sz]
                try:
                    self._blocks[idx] = np.linalg.inv(block)
                except Exception:
                    self._blocks[idx] = np.eye(sz, dtype=float)
        else:
            # Full matrix factorization or drop-tolerance pruning
            A = self._mat.copy()
            if self._drop_tol > 0.0:
                diag = np.diag(A)
                # Drop small off-diagonal elements
                mask = np.abs(A) < self._drop_tol * np.max(np.abs(diag))
                np.fill_diagonal(mask, False)
                A[mask] = 0.0
            self._inv_mat = np.linalg.inv(A)

    def solve(self, rhs: np.ndarray, offsets: Optional[List[int]] = None) -> np.ndarray:
        rhs_arr = np.asarray(rhs, dtype=float)
        if offsets is not None and len(self._blocks) > 0:
            res = np.empty_like(rhs_arr)
            for idx, inv_b in enumerate(self._blocks):
                off = offsets[idx]
                sz = inv_b.shape[0]
                res[off:off + sz] = inv_b @ rhs_arr[off:off + sz]
            return res
        if hasattr(self, "_inv_mat"):
            return self._inv_mat @ rhs_arr
        return np.linalg.solve(self._mat, rhs_arr)

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
    """Variable-order stiff BDF ODE integrator."""

    def __init__(self, func: Optional[Any] = None):
        super().__init__(func)

    def initialize(self, t0: float, func_or_y0: Optional[Any] = None, y0: Optional[np.ndarray] = None):
        self._time = float(t0)
        self._nevals = 0

        if y0 is not None:
            self._func = func_or_y0
            self._y = np.asarray(y0, dtype=float).copy()
        elif func_or_y0 is not None:
            if isinstance(func_or_y0, (np.ndarray, list, tuple)):
                self._y = np.asarray(func_or_y0, dtype=float).copy()
            else:
                self._func = func_or_y0
                # Extract initial conditions from FuncEval if available
                if hasattr(self._func, "getState"):
                    neq = self._func.nEquations() if hasattr(self._func, "nEquations") else self._func.neq()
                    y_init = np.zeros(neq, dtype=float)
                    self._func.getState(y_init)
                    self._y = y_init
                elif hasattr(self._func, "getInitialConditions"):
                    neq = self._func.nEquations() if hasattr(self._func, "nEquations") else self._func.neq()
                    y_init = np.zeros(neq, dtype=float)
                    self._func.getInitialConditions(t0, neq, y_init)
                    self._y = y_init
                elif hasattr(self._func, "y0"):
                    self._y = np.asarray(self._func.y0, dtype=float).copy()

    def reinitialize(self, t0: float, func_or_y0: Optional[Any] = None, y0: Optional[np.ndarray] = None):
        self.initialize(t0, func_or_y0, y0)

    def integrate(self, tout: float) -> float:
        if self._func is None or self._y is None:
            raise NumericsError("Integrator not initialized with function and state")
        from scipy.integrate import solve_ivp

        n = len(self._y)

        def rhs(t, y):
            self._nevals += 1
            if hasattr(self._func, "eval_f"):
                ydot = np.zeros(n, dtype=float)
                self._func.eval_f(t, y, ydot)
                return ydot
            elif hasattr(self._func, "eval"):
                ydot = np.zeros(n, dtype=float)
                self._func.eval(t, y, ydot)
                return ydot
            elif callable(self._func):
                return self._func(t, y)
            raise NumericsError("No valid eval method on func")

        sol = solve_ivp(
            rhs, (self._time, tout), self._y,
            method="BDF", rtol=self.rtol, atol=self.atol,
            max_step=self.max_step if self.max_step > 0 else np.inf
        )
        if not sol.success:
            raise NumericsError(f"Integration failed: {sol.message}")
        self._y = sol.y[:, -1]
        self._time = float(sol.t[-1])
        return self._time

    def step(self, tout: float) -> float:
        return self.integrate(tout)


class IdasIntegrator(Integrator):
    """DAE integrator for implicit residual systems F(t, y, ydot) = 0."""

    def __init__(self, func: Optional[Any] = None):
        super().__init__(func)
        self._res: Optional[Callable] = None
        self._ydot: Optional[np.ndarray] = None

    def initialize(self, t0: float, res_func_or_y0: Optional[Any] = None,
                   y0: Optional[np.ndarray] = None, ydot0: Optional[np.ndarray] = None):
        self._time = float(t0)
        self._nevals = 0
        if ydot0 is not None:
            self._res = res_func_or_y0
            self._y = np.asarray(y0, dtype=float).copy()
            self._ydot = np.asarray(ydot0, dtype=float).copy()
        elif y0 is not None:
            # func passed in constructor or first arg
            if isinstance(res_func_or_y0, (np.ndarray, list, tuple)):
                self._y = np.asarray(res_func_or_y0, dtype=float).copy()
                self._ydot = np.asarray(y0, dtype=float).copy()
            else:
                self._res = res_func_or_y0
                self._y = np.asarray(y0, dtype=float).copy()
                self._ydot = np.zeros_like(self._y)
        else:
            if self._func is not None and isinstance(res_func_or_y0, (np.ndarray, list, tuple)):
                self._res = self._func
                self._y = np.asarray(res_func_or_y0, dtype=float).copy()
                self._ydot = np.zeros_like(self._y)

    def integrate(self, tout: float) -> float:
        if (self._res is None and self._func is None) or self._y is None:
            raise NumericsError("IDAS integrator not initialized")
        res_fn = self._res if self._res is not None else self._func
        from scipy.integrate import solve_ivp

        n = len(self._y)
        ydot_curr = self._ydot if self._ydot is not None else np.zeros(n, dtype=float)

        def rhs(t, y):
            self._nevals += 1
            if hasattr(res_fn, "eval_f"):
                res = np.zeros(n, dtype=float)
                res_fn.eval_f(t, y, res)
                return res
            elif hasattr(res_fn, "eval"):
                res = np.zeros(n, dtype=float)
                res_fn.eval(t, y, res)
                return res
            elif callable(res_fn):
                return -np.asarray(res_fn(t, y, ydot_curr), dtype=float)
            raise NumericsError("No valid eval method on res_fn")

        sol = solve_ivp(rhs, (self._time, tout), self._y, method="Radau", rtol=self.rtol, atol=self.atol)
        if not sol.success:
            raise NumericsError(f"DAE integration failed: {sol.message}")
        self._y = sol.y[:, -1]
        self._time = float(sol.t[-1])
        return self._time

    def step(self, tout: float) -> float:
        return self.integrate(tout)


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
        return float(np.linalg.norm(r))

    def tsnorm(self, x: np.ndarray, r: np.ndarray, step: float) -> float:
        return float(np.linalg.norm(r))

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
