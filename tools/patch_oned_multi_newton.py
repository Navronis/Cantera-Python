"""Patch oned.py with complete MultiJac and MultiNewton implementations."""
from pathlib import Path


def main():
    oned_path = Path("outputs/cantera-python/cantera_python/oned.py")
    text = oned_path.read_text(encoding="utf-8")

    old_import = """from typing import Any, Callable, Literal
import numpy as np

from .constants import CanteraError, OneAtm
from .numerics import BandMatrix"""

    new_import = """from typing import Any, Callable, Literal, Optional
import math
import time
import numpy as np

from .constants import CanteraError, OneAtm
from .numerics import BandMatrix, SystemJacobian"""

    if old_import not in text:
        # Check CRLF
        old_import_crlf = old_import.replace("\n", "\r\n")
        new_import_crlf = new_import.replace("\n", "\r\n")
        assert old_import_crlf in text, "Import string not found"
        text = text.replace(old_import_crlf, new_import_crlf, 1)
    else:
        text = text.replace(old_import, new_import, 1)

    old_classes = """class MultiNewton:
    \"\"\"Newton solver for one-dimensional problems.\"\"\"
    def __init__(self, sys=None):
        pass

    def lastIterations(self) -> int:
        return 1

    def setOptions(self, max_iter: int = 100) -> None:
        pass


class MultiJac:
    \"\"\"Banded block Jacobian for 1D multi-domain networks.\"\"\"
    def __init__(self, sys=None):
        pass

    def factorize(self) -> None:
        pass

    def info(self) -> int:
        return 0

    def initialize(self, n: int = 1) -> None:
        pass

    def reset(self) -> None:
        pass

    def setBandwidth(self, kl: int, ku: int) -> None:
        pass

    def setValue(self, i: int, j: int, val: float) -> None:
        pass

    def type(self) -> str:
        return "MultiJac"

    def updateTransient(self, gamma: float, ydot: np.ndarray) -> None:
        pass

    def value(self, i: int, j: int) -> float:
        return 0.0"""

    new_classes = """class MultiJac(SystemJacobian):
    \"\"\"Banded block Jacobian for 1D multi-domain networks.

    Direct reconstruction of Cantera's C++ MultiJac class.
    \"\"\"

    def __init__(self, sys=None):
        super().__init__(sys)
        self.m_mat = BandMatrix()
        self.m_ssdiag = np.zeros(0, dtype=float)
        self.m_dim = 0
        self._age = 100000

    def reset(self) -> None:
        self.m_mat.bfill(0.0)
        self._age = 10000

    def type(self) -> str:
        return "banded-direct"

    def setValue(self, row: int, col: int, value: float) -> None:
        self.m_mat.setValue(row, col, value)
        if row == col and row < len(self.m_ssdiag):
            self.m_ssdiag[row] = float(value)

    def initialize(self, nVars: int = 1) -> None:
        self.m_dim = int(nVars)
        kl = self.m_mat.nSubDiagonals()
        ku = self.m_mat.nSuperDiagonals()
        self.m_mat.resize(self.m_dim, kl, ku)
        self.m_ssdiag = np.zeros(self.m_dim, dtype=float)

    def setBandwidth(self, bw: int, ku: Optional[int] = None) -> None:
        ku_val = bw if ku is None else ku
        self.m_mat.resize(self.m_dim, bw, ku_val)

    def updateTransient(self, rdt: float, mask: Sequence[int]) -> None:
        for n in range(self.m_dim):
            m_val = mask[n] if n < len(mask) else 1
            self.m_mat.setValue(n, n, self.m_ssdiag[n] - m_val * rdt)
        self.factorize()

    def factorize(self) -> None:
        self.m_mat.factor()

    def value(self, i: int, j: int) -> float:
        return self.m_mat.value(i, j)

    def solve(self, b: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
        return self.m_mat.solve(b, x)

    def info(self) -> int:
        return self.m_mat.info()


class MultiNewton:
    \"\"\"Newton iterator for multi-domain, one-dimensional problems.

    Reconstructs Cantera's C++ MultiNewton solver implementing bounded damped
    Newton iteration (Kee et al., 2003).
    \"\"\"

    def __init__(self, sz: int = 0):
        self.m_n = int(sz)
        self.m_maxAge = 5
        self.m_dampFactor = math.sqrt(2.0)
        self.m_maxDampIter = 7
        self.m_elapsed = 0.0
        self.m_lastIterations = 0
        self.m_x = np.zeros(self.m_n, dtype=float)
        self.m_stp = np.zeros(self.m_n, dtype=float)
        self.m_stp1 = np.zeros(self.m_n, dtype=float)

    def size(self) -> int:
        return self.m_n

    def lastIterations(self) -> int:
        return self.m_lastIterations

    def setOptions(self, maxJacAge: int = 5) -> None:
        self.m_maxAge = int(maxJacAge)

    def resize(self, sz: int) -> None:
        self.m_n = int(sz)
        self.m_x = np.zeros(self.m_n, dtype=float)
        self.m_stp = np.zeros(self.m_n, dtype=float)
        self.m_stp1 = np.zeros(self.m_n, dtype=float)

    def step(self, x: Sequence[float], step: Sequence[float], r: Any, loglevel: int = 0) -> None:
        \"\"\"Compute undamped Newton step delta_x = -J^{-1} F(x).\"\"\"
        x_arr = np.asarray(x, dtype=float)
        step_arr = np.asarray(step, dtype=float)
        r.eval(x_arr, step_arr)
        step_arr[:] = -step_arr[:]

        jac = r.linearSolver()
        try:
            t0 = time.perf_counter()
            jac.solve(step_arr, step_arr)
            if hasattr(r, "recordLinearSolve"):
                r.recordLinearSolve(time.perf_counter() - t0)
        except Exception as err:
            info = getattr(jac, "info", lambda: 0)()
            if info > 0:
                row = int(info - 1)
                comp = r.componentName(row) if hasattr(r, "componentName") else f"component_{row}"
                raise CanteraError(
                    f"MultiNewton::step: Jacobian is singular for matrix row {row}:\\n{comp}"
                ) from err
            raise
        if step is not step_arr:
            step[:] = step_arr[:]

    def boundStep(self, x0: Sequence[float], step0: Sequence[float], r: Any, loglevel: int = 0) -> float:
        \"\"\"Return multiplier in [0, 1] keeping all components between lower and upper bounds.\"\"\"
        fbound = 1.0
        n = self.size()
        for i in range(n):
            above = r.upperBound(i)
            below = r.lowerBound(i)
            val = float(x0[i])
            newval = val + float(step0[i])

            if newval > above:
                fbound = max(0.0, min(fbound, (above - val) / (newval - val)))
            elif newval < below:
                fbound = min(fbound, (val - below) / (val - newval))
        return float(fbound)

    def dampStep(self, x0: Sequence[float], step0: Sequence[float],
                 x1: Sequence[float], step1: Sequence[float],
                 s1: Any, r: Any, loglevel: int = 0, writetitle: bool = False) -> int:
        \"\"\"Performs a damped Newton step seeking alpha_k where norm(step_{k+1}) < norm(step_k).\"\"\"
        s0 = r.weightedNorm(step0)
        fbound = self.boundStep(x0, step0, r, loglevel - 1)

        if fbound < 1.0e-10:
            return -3

        alpha = fbound * 1.0
        computed_s1 = s0
        m = 0
        for m in range(self.m_maxDampIter):
            for j in range(self.m_n):
                x1[j] = x0[j] + alpha * step0[j]

            try:
                self.step(x1, step1, r, loglevel - 1)
            except Exception:
                alpha /= self.m_dampFactor
                continue

            computed_s1 = r.weightedNorm(step1)
            if computed_s1 < 1.0 or computed_s1 < s0:
                break
            alpha /= self.m_dampFactor
        else:
            m = self.m_maxDampIter

        if isinstance(s1, (list, np.ndarray)):
            s1[0] = computed_s1

        if m < self.m_maxDampIter:
            if computed_s1 > 1.0:
                return 0
            else:
                return 1
        else:
            return -2

    def solve(self, x0: Sequence[float], x1: Sequence[float], r: Any, loglevel: int = 0) -> int:
        \"\"\"Find solution to F(x) = 0 by damped Newton iteration.\"\"\"
        t0 = time.perf_counter()
        status = 0
        forceNewJac = False
        write_header = True
        s1 = 1.0e30

        if len(self.m_x) != len(x0):
            self.resize(len(x0))

        self.m_x[:] = np.asarray(x0, dtype=float)
        self.m_lastIterations = 0

        rdt = getattr(r, "rdt", lambda: 0.0)()
        nJacReeval = 0
        jac = r.linearSolver()

        while True:
            self.m_lastIterations += 1
            if jac.age() > self.m_maxAge:
                forceNewJac = True

            if forceNewJac:
                r.evalJacobian(self.m_x)
                try:
                    r.factorizeJacobian()
                except Exception as err:
                    if rdt == 0.0:
                        return -4
                    else:
                        raise
                forceNewJac = False

            self.step(self.m_x, self.m_stp, r, loglevel - 1)
            jac.incrementAge()

            s1_box = [s1]
            status = self.dampStep(self.m_x, self.m_stp, x1, self.m_stp1, s1_box, r, loglevel - 1, write_header)
            s1 = s1_box[0]
            write_header = False

            if status == 0:
                self.m_x[:] = x1[:]
            elif status == 1:
                if rdt == 0.0:
                    jac.setAge(0)
                break
            elif status < 0:
                if jac.age() > 1:
                    forceNewJac = True
                    if nJacReeval > 3:
                        break
                    nJacReeval += 1
                else:
                    break

        if status < 0:
            x1[:] = self.m_x[:]

        self.m_elapsed += time.perf_counter() - t0
        return status"""

    if old_classes not in text:
        old_classes_crlf = old_classes.replace("\n", "\r\n")
        new_classes_crlf = new_classes.replace("\n", "\r\n")
        assert old_classes_crlf in text, "Classes string not found"
        text = text.replace(old_classes_crlf, new_classes_crlf, 1)
    else:
        text = text.replace(old_classes, new_classes, 1)

    oned_path.write_text(text, encoding="utf-8")
    print("Successfully patched oned.py!")


if __name__ == "__main__":
    main()
