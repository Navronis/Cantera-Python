"""Comprehensive pytest suite for Refiner (1D adaptive grid refinement)."""

import json
import math
from pathlib import Path
import pytest

from cantera_python.oned import Refiner
from cantera_python.state import CanteraError


class MockDomain:
    def __init__(self, nv: int, np: int, is_free: bool = False, zfixed: float = -1.0):
        self.n_components = nv
        self.n_points = np
        self._names = [f"component_{i}" for i in range(nv)]
        self._is_free = is_free
        self.m_zfixed = zfixed
        self._two_point_control = False
        self._left_control = -1.0
        self._right_control = -1.0

    def nPoints(self) -> int:
        return self.n_points

    def nComponents(self) -> int:
        return self.n_components

    def componentName(self, n: int) -> str:
        return self._names[n]

    def component_name(self, n: int) -> str:
        return self._names[n]

    def index(self, n: int, j: int) -> int:
        return j * self.n_components + n

    def isFree(self) -> bool:
        return self._is_free

    def is_free(self) -> bool:
        return self._is_free

    def twoPointControlEnabled(self) -> bool:
        return self._two_point_control

    def two_point_control_enabled(self) -> bool:
        return self._two_point_control

    def leftControlPointCoordinate(self) -> float:
        return self._left_control

    def rightControlPointCoordinate(self) -> float:
        return self._right_control


def test_refiner_initialization():
    dom = MockDomain(nv=3, np=10)
    ref = Refiner(dom)

    assert ref.maxRatio() == 10.0
    assert ref.maxDelta() == 0.8
    assert ref.maxSlope() == 0.8
    assert ref.prune() == -0.001
    assert ref.maxPoints() == 1000
    assert ref.gridMin() == 1e-10
    assert ref.nNewPoints() == 0
    assert ref.getCriteria() == [10.0, 0.8, 0.8, -0.001]


def test_refiner_criteria_validation():
    dom = MockDomain(nv=1, np=5)
    ref = Refiner(dom)

    # Valid criteria
    ref.setCriteria(5.0, 0.5, 0.5, 0.1)
    assert ref.getCriteria() == [5.0, 0.5, 0.5, 0.1]
    assert ref.maxRatio() == 5.0
    assert ref.maxDelta() == 0.5
    assert ref.maxSlope() == 0.5
    assert ref.prune() == 0.1

    # Invalid ratio < 2.0
    with pytest.raises(CanteraError, match="ratio.*must be greater than 2.0"):
        ref.setCriteria(ratio=1.5)

    # Invalid slope < 0 or > 1
    with pytest.raises(CanteraError, match="slope.*must be between 0.0 and 1.0"):
        ref.setCriteria(slope=-0.1)
    with pytest.raises(CanteraError, match="slope.*must be between 0.0 and 1.0"):
        ref.setCriteria(slope=1.5)

    # Invalid curve < 0 or > 1
    with pytest.raises(CanteraError, match="curve.*must be between 0.0 and 1.0"):
        ref.setCriteria(curve=-0.1)
    with pytest.raises(CanteraError, match="curve.*must be between 0.0 and 1.0"):
        ref.setCriteria(curve=1.5)

    # Invalid prune > curve or prune > slope
    with pytest.raises(CanteraError, match="prune.*must be less than"):
        ref.setCriteria(slope=0.3, curve=0.5, prune=0.4)


def test_refiner_setters_and_getters():
    dom = MockDomain(nv=2, np=10)
    ref = Refiner(dom)

    ref.setMaxPoints(500)
    assert ref.maxPoints() == 500

    ref.setGridMin(1e-8)
    assert ref.gridMin() == 1e-8

    ref.setActive(0, False)
    assert not ref.m_active[0]
    ref.setActive(0, True)
    assert ref.m_active[0]


def test_refiner_value_access():
    dom = MockDomain(nv=2, np=5)
    ref = Refiner(dom)

    x = [10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0]
    assert ref.value(x, 0, 0) == 10.0
    assert ref.value(x, 1, 0) == 20.0
    assert ref.value(x, 0, 2) == 12.0
    assert ref.value(x, 1, 2) == 22.0


def test_refiner_uniform_constant():
    n = 10
    dom = MockDomain(nv=1, np=n)
    ref = Refiner(dom)
    ref.setCriteria(10.0, 0.8, 0.8, -0.1)

    z = [i / (n - 1) for i in range(n)]
    x = [1.0] * n

    n_new = ref.analyze(n, z, x)
    assert n_new == 0
    assert ref.nNewPoints() == 0
    for j in range(n):
        assert not ref.newPointNeeded(j)
        assert ref.keepPoint(j)


def test_refiner_slope_insertion():
    n = 12
    dom = MockDomain(nv=1, np=n)
    ref = Refiner(dom)
    ref.setCriteria(10.0, 0.4, 0.8, -0.1)

    z = [i / (n - 1) for i in range(n)]
    x = [math.tanh(25.0 * (zi - 0.5)) for zi in z]

    n_new = ref.analyze(n, z, x)
    assert n_new > 0
    assert ref.nNewPoints() == n_new
    assert ref.newPointNeeded(5)


def test_refiner_curvature_insertion():
    n = 15
    dom = MockDomain(nv=1, np=n)
    ref = Refiner(dom)
    ref.setCriteria(10.0, 0.8, 0.4, -0.1)

    z = [i / (n - 1) for i in range(n)]
    x = [math.exp(-120.0 * (zi - 0.5) ** 2) for zi in z]

    n_new = ref.analyze(n, z, x)
    assert n_new > 0
    assert ref.newPointNeeded(5)
    assert ref.newPointNeeded(6)


def test_refiner_grid_ratio():
    z = [0.0, 0.01, 0.02, 0.3, 0.6, 1.0]
    n = len(z)
    dom = MockDomain(nv=1, np=n)
    ref = Refiner(dom)
    ref.setCriteria(10.0, 0.8, 0.8, -0.1)
    x = [0.5] * n

    n_new = ref.analyze(n, z, x)
    assert n_new == 1
    assert ref.newPointNeeded(2)
    assert ref.keepPoint(1)
    assert ref.keepPoint(2)
    assert ref.keepPoint(3)
    assert ref.keepPoint(4)


def test_refiner_pruning_and_adjacent_protection():
    n = 15
    dom = MockDomain(nv=1, np=n)
    ref = Refiner(dom)
    ref.setCriteria(10.0, 0.8, 0.8, 0.1)
    z = [i / (n - 1) for i in range(n)]
    x = [1.0] * n

    ref.analyze(n, z, x)
    # Boundaries must always be kept
    assert ref.keepPoint(0)
    assert ref.keepPoint(n - 1)

    # Adjacent pruning protection: never two consecutive REMOVE
    for j in range(2, n - 1):
        if not ref.keepPoint(j):
            assert ref.keepPoint(j - 1)


def test_refiner_max_points_exception():
    n = 10
    dom = MockDomain(nv=1, np=n)
    ref = Refiner(dom)
    ref.setMaxPoints(10)
    z = [i / 9 for i in range(10)]
    x = [1.0] * 10

    with pytest.raises(CanteraError, match="max number of grid points reached"):
        ref.analyze(n, z, x)


def test_refiner_mismatched_size_exception():
    dom = MockDomain(nv=1, np=10)
    ref = Refiner(dom)
    z = [i / 7 for i in range(8)]
    x = [1.0] * 8

    with pytest.raises(CanteraError, match="number of grid points provided does not match domain size"):
        ref.analyze(8, z, x)


def test_refiner_flame_fixed_temperature():
    z = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n = len(z)
    dom = MockDomain(nv=1, np=n, is_free=True, zfixed=0.4)
    ref = Refiner(dom)
    ref.setCriteria(10.0, 0.8, 0.8, 0.5)  # aggressive prune
    x = [0.0] * n

    ref.analyze(n, z, x)
    # Point at zfixed=0.4 (index 2) must be protected
    assert ref.keepPoint(2)


def test_refiner_differential_campaign_parity():
    repo_root = Path(__file__).resolve().parent.parent
    diff_file = repo_root / "audit" / "refiner_differential_summary.json"
    assert diff_file.exists(), "refiner_differential_summary.json must exist"

    with open(diff_file, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["parity_achieved"] is True
    assert summary["passed_cases"] == summary["total_cases"] == 20
    assert summary["pass_rate_pct"] == 100.0
    assert summary["metrics"]["insertion_flag_mismatches"] == 0
    assert summary["metrics"]["retention_flag_mismatches"] == 0
    assert summary["metrics"]["new_point_count_mismatches"] == 0
