"""Unit tests for Sim1D multi-domain simulation manager and solve orchestration."""

import math
import numpy as np
import pytest

from cantera_python.oned import Sim1D, Domain1D, newSim1D
from cantera_python.state import CanteraError


class MockSimDomain(Domain1D):
    def __init__(self, name: str, dtype: str, nv: int, np_pts: int, bw: int, grid: list[float]):
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
        if grid:
            self.setupGrid(grid)

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

    def initialValue(self, n: int, j: int) -> float:
        return 1.0

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


def test_sim1d_instantiation():
    d = MockSimDomain("flow", "flow", nv=2, np_pts=5, bw=1, grid=[0.0, 0.25, 0.5, 0.75, 1.0])
    sim = Sim1D([d])
    assert sim.nDomains() == 1
    assert sim.points() == 5
    assert sim.size() == 10

    sim_factory = newSim1D([d])
    assert isinstance(sim_factory, Sim1D)
    assert sim_factory.nDomains() == 1


def test_sim1d_value_access():
    d = MockSimDomain("flow", "flow", nv=2, np_pts=3, bw=1, grid=[0.0, 0.5, 1.0])
    sim = Sim1D([d])

    sim._setValue(0, 0, 1, 42.5)
    sim._setValue(0, 1, 2, -13.0)

    assert sim._value(0, 0, 1) == 42.5
    assert sim._value(0, 1, 2) == -13.0


def test_sim1d_refine_slope():
    z = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    d = MockSimDomain("flow", "flow", nv=1, np_pts=6, bw=1, grid=z)
    sim = Sim1D([d])
    sim.setRefineCriteria(0, ratio=10.0, slope=0.2, curve=0.8, prune=0.0)

    # Steep step between pt 2 (0.4) and pt 3 (0.6)
    for j in range(6):
        val = 1.0 if j <= 2 else 100.0
        sim._setValue(0, 0, j, val)

    added = sim.refine(0)
    assert added == 1
    assert sim.points() > 6
    # Midpoint should be inserted at z = 0.5
    grid_after = list(sim.domain(0).grid)
    assert any(abs(zj - 0.5) < 1e-10 for zj in grid_after)


def test_sim1d_refine_flat():
    z = [0.1 * i for i in range(11)]
    d = MockSimDomain("flow", "flow", nv=1, np_pts=11, bw=1, grid=z)
    sim = Sim1D([d])
    sim.setRefineCriteria(0, ratio=10.0, slope=0.8, curve=0.8, prune=0.5)

    # Completely flat profile has zero dynamic range; eliminated from refinement/pruning
    for j in range(11):
        sim._setValue(0, 0, j, 5.0)

    res = sim.refine(0)
    assert res == 0
    assert sim.domain(0).nPoints() == 11


def test_sim1d_refine_prune():
    z = [0.1 * i for i in range(11)]
    d = MockSimDomain("flow", "flow", nv=1, np_pts=11, bw=1, grid=z)
    sim = Sim1D([d])
    sim.setRefineCriteria(0, ratio=10.0, slope=0.8, curve=0.8, prune=0.05)

    # Flat baseline with ramp at the end: flat region points are pruned
    for j in range(11):
        val = 0.0 if j < 6 else float(j - 5) * 2.0
        sim._setValue(0, 0, j, val)

    res = sim.refine(0)
    assert res == 1


def test_sim1d_criteria_get_set():
    d1 = MockSimDomain("inlet", "inlet", nv=1, np_pts=1, bw=1, grid=[0.0])
    d2 = MockSimDomain("flow", "flow", nv=2, np_pts=5, bw=1, grid=[0.0, 0.25, 0.5, 0.75, 1.0])
    sim = Sim1D([d1, d2])

    sim.setRefineCriteria(-1, 8.0, 0.4, 0.3, 0.05)
    crit0 = sim.getRefineCriteria(0)
    assert np.allclose(crit0, [8.0, 0.4, 0.3, 0.05])

    sim.setRefineCriteria(1, 4.0, 0.1, 0.2, 0.01)
    crit1 = sim.getRefineCriteria(1)
    assert np.allclose(crit1, [4.0, 0.1, 0.2, 0.01])

    with pytest.raises(CanteraError):
        sim.getRefineCriteria(-1)


def test_sim1d_gridmin_and_max_points():
    d = MockSimDomain("flow", "flow", nv=1, np_pts=5, bw=1, grid=[0.0, 0.25, 0.5, 0.75, 1.0])
    sim = Sim1D([d])

    sim.setMaxGridPoints(0, 50)
    assert sim.maxGridPoints(0) == 50

    sim.setGridMin(0, 0.01)
    assert sim.domain(0).refiner().gridMin() == 0.01


def test_sim1d_restore_solutions():
    z = [0.0, 0.25, 0.5, 0.75, 1.0]
    d = MockSimDomain("flow", "flow", nv=1, np_pts=5, bw=1, grid=z)
    sim = Sim1D([d])

    # Unset steady solution should raise
    with pytest.raises(CanteraError):
        sim.restoreSteadySolution()

    # Unset time stepping solution should raise
    with pytest.raises(CanteraError):
        sim.restoreTimeSteppingSolution()

    # Refine snapshots steady solution
    for j in range(5):
        sim._setValue(0, 0, j, float(j))
    sim.refine(0)

    # Tamper with state
    sim._setValue(0, 0, 0, 999.0)
    assert sim._value(0, 0, 0) == 999.0

    # Restore restores original snapshot
    sim.restoreSteadySolution()
    assert sim._value(0, 0, 0) == 0.0


def test_sim1d_timestep_regrid_max():
    d = MockSimDomain("flow", "flow", nv=1, np_pts=3, bw=1, grid=[0.0, 0.5, 1.0])
    sim = Sim1D([d])

    assert sim.timeStepRegridMax() == 10
    sim.setTimeStepRegridMax(30)
    assert sim.timeStepRegridMax() == 30


def test_sim1d_eval_ss_jacobian_and_solve_adjoint():
    inlet = MockSimDomain("inlet", "inlet", nv=2, np_pts=1, bw=1, grid=[0.0])
    flow = MockSimDomain("flow", "flow", nv=2, np_pts=4, bw=2, grid=[0.0, 0.33, 0.67, 1.0])
    outlet = MockSimDomain("outlet", "outlet", nv=2, np_pts=1, bw=1, grid=[1.0])
    sim = Sim1D([inlet, flow, outlet])

    for i in range(sim.size()):
        sim.m_state[i] = 1.0 + 0.1 * i

    sim.evalSSJacobian()
    mjac = sim.linearSolver()
    assert mjac is not None

    b = [1.0] * sim.size()
    lambda_out = [0.0] * sim.size()
    sim.solveAdjoint(b, lambda_out)

    # Check non-trivial solution
    assert any(abs(x) > 1e-12 for x in lambda_out)


def test_sim1d_control_points_exceptions():
    d = MockSimDomain("flow", "flow", nv=1, np_pts=3, bw=1, grid=[0.0, 0.5, 1.0])
    sim = Sim1D([d])

    with pytest.raises(CanteraError):
        sim.setLeftControlPoint(1000.0)

    with pytest.raises(CanteraError):
        sim.setRightControlPoint(1000.0)
