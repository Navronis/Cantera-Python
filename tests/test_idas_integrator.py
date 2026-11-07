"""Comprehensive test suite for IdasIntegrator DAE solver family."""

import math
import pytest
import numpy as np

from cantera_python.constants import CanteraError
from cantera_python.numerics import IdasIntegrator, FuncEval


class SimpleDAESystem(FuncEval):
    """Simple Index-1 DAE:
    y'[0] = -y[0] + y[1]
    0     = y[0]**2 - y[1]
    with y[0](0) = 1.0, y[1](0) = 1.0.
    """

    def __init__(self):
        super().__init__()
        self._y = np.array([1.0, 1.0], dtype=float)
        self._ydot = np.array([0.0, 0.0], dtype=float)

    def neq(self) -> int:
        return 2

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        y[:] = self._y
        ydot[:] = self._ydot

    def getConstraints(self, c: np.ndarray) -> None:
        c[0] = 1.0  # differential
        c[1] = 0.0  # algebraic

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray) -> None:
        res[0] = ydot[0] - (-y[0] + y[1])
        res[1] = y[0] ** 2 - y[1]

    def setState(self, y0: float, y1: float, yd0: float, yd1: float) -> None:
        self._y[0] = y0
        self._y[1] = y1
        self._ydot[0] = yd0
        self._ydot[1] = yd1


class SensitivityDAESystem(SimpleDAESystem):
    def nparams(self) -> int:
        return 2


def test_idas_integrator_initialization_and_options():
    """Verify IdasIntegrator configuration and option setters/getters."""
    sys = SimpleDAESystem()
    idas = IdasIntegrator()
    idas.setTolerances(1e-6, 1e-8)
    idas.setTolerances(1e-6, [1e-8, 1e-8])
    idas.setSensitivityTolerances(1e-6, 1e-8)
    idas.setLinearSolverType("DENSE")
    idas.setMaxOrder(4)
    assert idas.maxOrder() == 4
    idas.setMaxStepSize(0.05)
    idas.setMaxSteps(5000)
    assert idas.maxSteps() == 5000
    idas.setMaxErrTestFails(15)
    idas.setMaxNonlinIterations(5)
    assert idas.maxNonlinIterations() == 5
    idas.setMaxNonlinConvFailures(12)
    assert idas.maxNonlinConvFailures() == 12
    idas.includeAlgebraicInErrorTest(False)
    assert idas.algebraicInErrorTest() is False
    idas.includeAlgebraicInErrorTest(True)
    assert idas.algebraicInErrorTest() is True

    idas.initialize(0.0, sys)
    assert idas.nEquations() == 2
    assert np.isclose(idas.currentTime(), 0.0)
    assert np.allclose(idas.solution(), [1.0, 1.0])
    assert np.isclose(idas.solution(0), 1.0)
    assert np.isclose(idas.solution(1), 1.0)
    assert idas.nSensParams() == 0


def test_idas_integrator_advance():
    """Verify IdasIntegrator integrates index-1 DAE forward in time preserving constraints."""
    sys = SimpleDAESystem()
    idas = IdasIntegrator()
    idas.setTolerances(1e-7, 1e-9)
    idas.initialize(0.0, sys)

    t_target = 0.1
    t_end = idas.integrate(t_target)
    assert np.isclose(t_end, t_target)
    sol = idas.solution()
    assert abs(sol[0] ** 2 - sol[1]) < 1e-5

    # Check solver stats
    stats = idas.solverStats()
    assert "steps" in stats and stats["steps"] > 0
    assert "res_evals" in stats and stats["res_evals"] > 0
    assert "nonlinear_iters" in stats
    assert "last_order" in stats and stats["last_order"] >= 1
    assert idas.lastOrder() == stats["last_order"]


def test_idas_integrator_single_step():
    """Verify IdasIntegrator step() single-step execution."""
    sys = SimpleDAESystem()
    idas = IdasIntegrator()
    idas.initialize(0.0, sys)
    t1 = idas.step(0.1)
    assert 0.0 < t1 <= 0.1
    assert np.isclose(idas.currentTime(), t1)


def test_idas_integrator_reinitialize():
    """Verify IdasIntegrator reinitialize() resets time and applies options."""
    sys = SimpleDAESystem()
    idas = IdasIntegrator()
    idas.initialize(0.0, sys)
    idas.integrate(0.1)
    assert np.isclose(idas.currentTime(), 0.1)

    sys.setState(0.5, 0.25, 0.0, 0.0)
    idas.reinitialize(0.1, sys)
    assert np.isclose(idas.currentTime(), 0.1)
    assert np.allclose(idas.solution(), [0.5, 0.25])


def test_idas_integrator_sensitivity_and_errors():
    """Verify IdasIntegrator sensitivity methods and error handling."""
    sys = SimpleDAESystem()
    idas = IdasIntegrator()
    idas.initialize(0.0, sys)

    # At t0, sensitivity queries return 0.0
    assert idas.sensitivity(0, 0) == 0.0

    # Advance so time > t0, then test out of range indices
    idas.integrate(0.01)
    with pytest.raises(CanteraError):
        idas.sensitivity(10, 0)
    with pytest.raises(CanteraError):
        idas.sensitivity(0, 5)

    # Method validation
    idas.setMethod(1)
    idas.setMethod("BDF")
    with pytest.raises(CanteraError):
        idas.setMethod("Adams")

    # Linear solver validation
    idas.applyOptions()
    idas.setLinearSolverType("INVALID_SOLVER")
    with pytest.raises(CanteraError):
        idas.applyOptions()

    # Backward integration raises CanteraError
    idas.setLinearSolverType("DENSE")
    idas.integrate(0.05)
    with pytest.raises(CanteraError):
        idas.integrate(0.01)
    with pytest.raises(CanteraError):
        idas.step(0.01)

    # getErrorInfo format
    info = idas.getErrorInfo(5)
    assert isinstance(info, str)

    # checkError
    idas.checkError(0, "test", "idaMethod")
    with pytest.raises(CanteraError):
        idas.checkError(-1, "test", "idaMethod")
    with pytest.raises(CanteraError):
        idas.checkError(2, "test", "idaMethod")


def test_idas_integrator_sensitivity_unsupported_exception():
    """Verify initialize throws when FuncEval requests DAE sensitivities (matching native Cantera)."""
    sens_sys = SensitivityDAESystem()
    idas = IdasIntegrator()
    with pytest.raises(CanteraError, match="Sensitivity analysis for DAE systems is not fully implemented"):
        idas.initialize(0.0, sens_sys)


def test_idas_consistent_initialization():
    """Verify IdasIntegrator computes consistent ICs when initial algebraic state is perturbed."""
    sys = SimpleDAESystem()
    # Inconsistent initial algebraic state: y0=1.0, y1=2.0 (constraint y0^2 - y1 = -1 != 0)
    sys.setState(1.0, 2.0, 0.0, 0.0)
    idas = IdasIntegrator()
    idas.initialize(0.0, sys)
    sol = idas.solution()
    # Consistent IC should have restored y1 = y0^2 = 1.0
    assert np.isclose(sol[0], 1.0, atol=1e-6)
    assert np.isclose(sol[1], 1.0, atol=1e-6)
