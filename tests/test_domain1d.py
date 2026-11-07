"""Unit tests for Domain1D base class and multi-domain topology management."""

import pytest
import numpy as np

from cantera_python.oned import Domain1D
from cantera_python.state import CanteraError


class MockDomain(Domain1D):
    def eval(self, j: int | None, x: np.ndarray, r: np.ndarray, mask: np.ndarray, rdt: float = 0.0) -> None:
        pass

    def initialValue(self, n: int, j: int) -> float:
        return 10.0 * (n + 1) + float(j)


def test_domain1d_construction_and_defaults():
    d = MockDomain(nv=1, points=1)
    assert d.nComponents() == 1
    assert d.nPoints() == 1
    assert d.size() == 1
    assert d.bandwidth() == -1
    assert d.domainType() == "domain"
    assert d.domainIndex() == 0
    assert not d.isConnector()
    assert d.zmin() == 0.0
    assert d.zmax() == 0.0
    assert d.steady_rtol(0) == 1e-4
    assert d.steady_atol(0) == 1e-9
    assert d.transient_rtol(0) == 1e-4
    assert d.transient_atol(0) == 1e-11


def test_domain1d_resize():
    d = MockDomain(nv=1, points=1)
    d.resize(nv=3, np_pts=8)
    assert d.nComponents() == 3
    assert d.nPoints() == 8
    assert d.size() == 24
    for n in range(3):
        assert d.steady_rtol(n) == 1e-4
        assert d.steady_atol(n) == 1e-9
        assert d.transient_rtol(n) == 1e-4
        assert d.transient_atol(n) == 1e-11
        assert d.lowerBound(n) == 0.0
        assert d.upperBound(n) == 0.0


def test_domain1d_component_names_and_indices():
    d = MockDomain(nv=3, points=4)
    assert d.componentName(0) == "component 0"
    assert d.componentName(1) == "component 1"

    d.setComponentName(0, "u")
    d.setComponentName(1, "V")
    d.setComponentName(2, "T")

    assert d.componentName(0) == "u"
    assert d.componentName(1) == "V"
    assert d.componentName(2) == "T"

    assert d.componentIndex("u") == 0
    assert d.componentIndex("V") == 1
    assert d.componentIndex("T") == 2
    assert d.hasComponent("u")
    assert d.hasComponent("T")

    with pytest.raises(CanteraError):
        d.componentIndex("nonexistent")

    with pytest.raises(CanteraError):
        d.hasComponent("nonexistent")

    with pytest.raises(IndexError):
        d.componentName(5)


def test_domain1d_bounds():
    d = MockDomain(nv=2, points=3)
    d.setBounds(0, -50.0, 150.0)
    d.setBounds(1, 200.0, 4000.0)

    assert d.lowerBound(0) == -50.0
    assert d.upperBound(0) == 150.0
    assert d.lowerBound(1) == 200.0
    assert d.upperBound(1) == 4000.0


def test_domain1d_tolerances_and_modes():
    d = MockDomain(nv=2, points=4)
    d.setSteadyTolerances(1e-5, 1e-10, -1)
    d.setTransientTolerances(2e-4, 3e-11, -1)
    d.setSteadyTolerances(1e-6, 1e-12, 1)

    assert d.steady_rtol(0) == 1e-5
    assert d.steady_atol(0) == 1e-10
    assert d.steady_rtol(1) == 1e-6
    assert d.steady_atol(1) == 1e-12

    assert d.steady()
    assert not d.transient()
    assert d.rtol(0) == 1e-5
    assert d.atol(0) == 1e-10

    d.initTimeInteg(0.01, [1.0] * d.size())
    assert not d.steady()
    assert d.transient()
    assert d.rtol(0) == 2e-4
    assert d.atol(0) == 3e-11

    d.setSteadyMode()
    assert d.steady()
    assert not d.transient()


def test_domain1d_prev_soln():
    d = MockDomain(nv=2, points=3)
    state = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    d.initTimeInteg(0.02, state)

    for j in range(3):
        for n in range(2):
            expected = state[d.index(n, j)]
            assert d.prevSoln(n, j) == expected


def test_domain1d_grids():
    d = MockDomain(nv=1, points=1)
    z_custom = [0.0, 0.1, 0.3, 0.6, 1.0]
    d.setupGrid(z_custom)

    assert d.nPoints() == 5
    assert d.zmin() == 0.0
    assert d.zmax() == 1.0
    for j in range(5):
        assert d.z(j) == z_custom[j]
    assert np.allclose(d.grid, z_custom)

    d.setupUniformGrid(points=11, length=0.05, start=0.01)
    assert d.nPoints() == 11
    assert abs(d.zmin() - 0.01) < 1e-12
    assert abs(d.zmax() - 0.06) < 1e-12
    assert abs(d.z(5) - (0.01 + 5 * 0.005)) < 1e-12


def test_domain1d_topology_linking():
    d0 = MockDomain(nv=1, points=1)
    d1 = MockDomain(nv=3, points=5)
    d2 = MockDomain(nv=1, points=1)

    d0.append(d1)
    d1.append(d2)

    assert d0.left() is None
    assert d0.right() is d1
    assert d1.left() is d0
    assert d1.right() is d2
    assert d2.left() is d1
    assert d2.right() is None

    assert d0.loc() == 0
    assert d0.firstPoint() == 0
    assert d0.lastPoint() == 0

    assert d1.loc() == 1
    assert d1.firstPoint() == 1
    assert d1.lastPoint() == 5

    assert d2.loc() == 16
    assert d2.firstPoint() == 6
    assert d2.lastPoint() == 6

    assert d1.globalComponentIndex(1, 2) == 1 + (2 * 3 + 1)


def test_domain1d_refiner_delegation():
    d = MockDomain(nv=2, points=5)
    r = d.refiner()
    assert r is not None

    d.setRefineCriteria(ratio=12.0, slope=0.3, curve=0.4, prune=0.01)
    crit = d.getRefineCriteria()
    assert np.allclose(crit, [12.0, 0.3, 0.4, 0.01])


def test_domain1d_jacobian_mode():
    d = MockDomain(nv=1, points=2)
    assert d.jacobianMode() == "auto"

    d.setJacobianMode("analytic")
    assert d.jacobianMode() == "analytic"

    d.setJacobianMode("finite-difference")
    assert d.jacobianMode() == "finite-difference"

    d.setJacobianMode("auto")
    assert d.jacobianMode() == "auto"

    with pytest.raises(CanteraError):
        d.setJacobianMode("invalid_mode")


def test_domain1d_meta():
    d = MockDomain(nv=2, points=3)
    d.setComponentName(0, "u")
    d.setComponentName(1, "T")
    d.setSteadyTolerances(1e-4, 1e-9, -1)
    d.setTransientTolerances(1e-4, 1e-11, -1)

    meta = d.getMeta()
    assert meta["type"] == "domain"
    assert meta["points"] == 3
    assert meta["tolerances"]["steady-reltol"] == 1e-4

    # Nonuniform tolerances
    d.setSteadyTolerances(5e-5, 2e-9, 1)
    meta2 = d.getMeta()
    assert isinstance(meta2["tolerances"]["steady-reltol"], dict)
    assert meta2["tolerances"]["steady-reltol"]["u"] == 1e-4
    assert meta2["tolerances"]["steady-reltol"]["T"] == 5e-5


def test_domain1d_element_access():
    d = MockDomain(nv=2, points=3)
    x = [0.0] * d.size()

    for j in range(3):
        for n in range(2):
            d.setValue(n, j, 10.0 * (n + 1) + float(j), x)

    for j in range(3):
        for n in range(2):
            assert d.value(x, n, j) == 10.0 * (n + 1) + float(j)

    d._getInitialSoln(x)
    for j in range(3):
        for n in range(2):
            assert d.value(x, n, j) == 10.0 * (n + 1) + float(j)

    shared = [0.0] * d.size()
    d.setData(shared)
    d.setFlatProfile(1, 99.0)
    for j in range(3):
        assert shared[d.index(1, j)] == 99.0
