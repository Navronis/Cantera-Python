"""Unit tests for OneDim multi-domain residual container."""

import math
import numpy as np
import pytest

from cantera_python.oned import OneDim, Domain1D
from cantera_python.state import CanteraError


class MockDomain(Domain1D):
    def __init__(self, name: str, dtype: str, nv: int, np_pts: int, bw: int = 1):
        super().__init__(nv=nv, points=np_pts, name=name)
        self.m_id = name
        self.name = name
        self.m_domain_type = dtype
        self._names = [f"{name}_var_{i}" for i in range(self.m_nv)]
        self.setBandwidth(bw)
        self.m_min = [-100.0 - 5.0 * i for i in range(self.m_nv)]
        self.m_max = [10000.0 + 10.0 * i for i in range(self.m_nv)]
        self.m_rtol_ss = [1e-4] * self.m_nv
        self.m_atol_ss = [1e-9] * self.m_nv

    def nComponents(self) -> int:
        return self.m_nv

    def nPoints(self) -> int:
        return self.m_points

    def component_name(self, k: int) -> str:
        return self._names[int(k)]

    def componentName(self, k: int) -> str:
        return self._names[int(k)]

    def index(self, k: int, pt: int) -> int:
        return int(pt) * self.m_nv + int(k)

    def size(self) -> int:
        return self.m_points * self.m_nv

    def setTolerances(self, rtol: list[float], atol: list[float]) -> None:
        for i in range(self.m_nv):
            self.m_rtol_ss[i] = rtol[i % len(rtol)]
            self.m_atol_ss[i] = atol[i % len(atol)]

    def eval(self, j: int | None, x: np.ndarray, r: np.ndarray, mask: np.ndarray, rdt: float = 0.0) -> None:
        j0 = 0
        j1 = self.nPoints() - 1
        if j is not None and j != -1:
            fp = self.firstPoint()
            lp = self.lastPoint()
            if j < fp or j > lp:
                if j + 1 < fp or (fp > 0 and j > lp + 1):
                    return
            if fp <= j <= lp:
                jloc = j - fp
                j0 = (jloc - 1) if jloc > 0 else 0
                j1 = min(self.nPoints() - 1, jloc + 1)
            elif j + 1 == fp:
                j0 = 0
                j1 = 0
            elif j == lp + 1:
                j0 = self.nPoints() - 1
                j1 = self.nPoints() - 1

        for pt in range(j0, j1 + 1):
            for k in range(self.m_nv):
                idx = self.index(k, pt)
                g_idx = self.loc() + idx
                val = x[g_idx]
                lapl = 0.0
                if 0 < pt < self.nPoints() - 1:
                    v_left = x[self.loc() + self.index(k, pt - 1)]
                    v_right = x[self.loc() + self.index(k, pt + 1)]
                    lapl = v_right - 2.0 * val + v_left
                elif pt == 0 and self.m_left:
                    v_left = x[self.m_left.loc() + self.m_left.index(k % self.m_left.nComponents(), self.m_left.nPoints() - 1)]
                    v_right = x[self.loc() + self.index(k, 1)] if self.nPoints() > 1 else val
                    lapl = v_right - 2.0 * val + v_left
                elif pt == self.nPoints() - 1 and self.m_right:
                    v_left = x[self.loc() + self.index(k, pt - 1)] if self.nPoints() > 1 else val
                    v_right = x[self.m_right.loc() + self.m_right.index(k % self.m_right.nComponents(), 0)]
                    lapl = v_right - 2.0 * val + v_left
                r[g_idx] = -0.5 * lapl + 0.1 * val * val - 0.05 * val
                if rdt > 0.0:
                    r[g_idx] += rdt * val
                    mask[g_idx] = 1


def test_onedim_empty():
    oned = OneDim()
    assert oned.nDomains() == 0
    assert oned.points() == 0
    assert oned.size() == 0
    assert oned.bandwidth() == 0


def test_onedim_single_domain():
    oned = OneDim()
    dom = MockDomain("test_dom", "flow", nv=3, np_pts=5, bw=2)
    oned.addDomain(dom)
    oned.init()
    oned.resize()

    assert oned.nDomains() == 1
    assert oned.points() == 5
    assert oned.size() == 15
    assert oned.start(0) == 0
    assert oned.domain(0) is dom
    assert oned.domainIndex("test_dom") == 0


def test_onedim_multi_domain_linking():
    oned = OneDim()
    inlet = MockDomain("inlet", "inlet", nv=2, np_pts=1, bw=1)
    flow = MockDomain("flow", "flow", nv=4, np_pts=10, bw=3)
    outlet = MockDomain("outlet", "outlet", nv=2, np_pts=1, bw=1)

    oned.addDomain(inlet)
    oned.addDomain(flow)
    oned.addDomain(outlet)
    oned.init()
    oned.resize()

    assert oned.nDomains() == 3
    assert oned.points() == 12
    assert oned.size() == 2 + 40 + 2  # 44
    assert oned.domain(0) is inlet
    assert oned.domain(1) is flow
    assert oned.domain(2) is outlet

    # Check offsets
    assert oned.start(0) == 0
    assert oned.start(1) == 2
    assert oned.start(2) == 42

    # Check bidirectional linking
    assert inlet.m_right is flow
    assert flow.m_left is inlet
    assert flow.m_right is outlet
    assert outlet.m_left is flow


def test_onedim_check_domain_index_exceptions():
    oned = OneDim()
    dom = MockDomain("d1", "flow", nv=2, np_pts=3)
    oned.addDomain(dom)
    oned.init()

    oned.checkDomainIndex(0)
    with pytest.raises((IndexError, CanteraError)):
        oned.checkDomainIndex(1)
    with pytest.raises((IndexError, CanteraError)):
        oned.checkDomainIndex(-1)


def test_onedim_component_naming_and_bounds():
    oned = OneDim()
    dom1 = MockDomain("inlet", "inlet", nv=2, np_pts=1)
    dom2 = MockDomain("flame", "flow", nv=3, np_pts=2)
    oned.addDomain(dom1)
    oned.addDomain(dom2)
    oned.init()
    oned.resize()

    # Total size = 2 + 6 = 8 components
    name0 = oned.componentName(0)
    assert "inlet" in name0
    name2 = oned.componentName(2)
    assert "flame" in name2

    # Bounds
    ub0 = oned.upperBound(0)
    lb0 = oned.lowerBound(0)
    assert ub0 == 10000.0
    assert lb0 == -100.0


def test_onedim_weighted_norm():
    oned = OneDim()
    dom = MockDomain("flow", "flow", nv=2, np_pts=4)
    dom.setTolerances([1e-3, 1e-3], [1e-6, 1e-6])
    oned.addDomain(dom)
    oned.init()
    oned.resize()

    sz = oned.size()
    state = [10.0] * sz
    step = [0.1] * sz
    oned.setInitialGuess(state)

    wn = oned.weightedNorm(step)
    # Expected ewt = 1e-3 * 10.0 + 1e-6 = 0.010001
    # f = 0.1 / 0.010001 = 9.99900009999
    # sqrt(sum(f^2)/size) = f
    assert abs(wn - (0.1 / 0.010001)) < 1e-8


def test_onedim_eval_steady_and_transient():
    oned = OneDim()
    dom = MockDomain("flow", "flow", nv=2, np_pts=4)
    oned.addDomain(dom)
    oned.init()
    oned.resize()

    sz = oned.size()
    x = np.ones(sz, dtype=float) * 2.0
    r_steady = np.zeros(sz, dtype=float)
    oned.eval(-1, x, r_steady, rdt=0.0, count=1)

    r_transient = np.zeros(sz, dtype=float)
    oned.eval(-1, x, r_transient, rdt=10.0, count=1)

    # Difference in transient mode should be rdt * x = 10.0 * 2.0 = 20.0
    diff = r_transient - r_steady
    np.testing.assert_allclose(diff, 20.0, rtol=1e-12)


def test_onedim_eval_jacobian():
    oned = OneDim()
    dom = MockDomain("flow", "flow", nv=2, np_pts=4, bw=1)
    oned.addDomain(dom)
    oned.init()
    oned.resize()

    x = [1.0] * oned.size()
    oned.evalJacobian(x)
    mjac = oned.linearSolver()
    assert mjac is not None
    # Check that diagonal entries are non-zero
    for k in range(oned.size()):
        assert abs(mjac.value(k, k)) > 0.0
