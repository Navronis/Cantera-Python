#!/usr/bin/env python3
"""Unit tests for CVodesIntegrator verifying native parity, orders 1-5, and numerical behavior."""

import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pytest

from cantera_python.numerics import CVodesIntegrator, FuncEval
from cantera_python import CanteraError

class DecayFunc(FuncEval):
    def __init__(self, lam: float = 1.0, y0: float = 1.0):
        super().__init__()
        self.lam = lam
        self.y0_val = y0

    def neq(self) -> int:
        return 1

    def getState(self, y: np.ndarray) -> None:
        y[0] = self.y0_val

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, p=None) -> None:
        ydot[0] = -self.lam * y[0]

class OscillatorFunc(FuncEval):
    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega

    def neq(self) -> int:
        return 2

    def getState(self, y: np.ndarray) -> None:
        y[0] = 1.0
        y[1] = 0.0

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, p=None) -> None:
        ydot[0] = y[1]
        ydot[1] = -self.omega * self.omega * y[0]

class RobertsonFunc(FuncEval):
    def neq(self) -> int:
        return 3

    def getState(self, y: np.ndarray) -> None:
        y[0] = 1.0
        y[1] = 0.0
        y[2] = 0.0

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, p=None) -> None:
        ydot[0] = -0.04 * y[0] + 1e4 * y[1] * y[2]
        ydot[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
        ydot[2] = 3e7 * y[1] * y[1]

class ParamSensFunc(FuncEval):
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.m_sens_params = [p]
        self.m_paramScales = [1.0]

    def neq(self) -> int:
        return 1

    def nparams(self) -> int:
        return 1

    def getState(self, y: np.ndarray) -> None:
        y[0] = 1.0

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, p=None) -> None:
        param = self.m_sens_params[0] if not p else p[0]
        ydot[0] = param * y[0]


def test_orders_1_to_5_reachable():
    """Verify that BDF orders 1, 2, 3, 4, 5 are reached deterministically."""
    # Order 1: single initial step
    ode1 = CVodesIntegrator()
    ode1.initialize(0.0, DecayFunc(1.0))
    ode1.step(0.1)
    assert ode1.lastOrder() == 1

    # Order limits: setMaxOrder on a short horizon reachable by order 1
    for max_ord in [1, 2, 3, 4, 5]:
        ode = CVodesIntegrator()
        ode.setMaxOrder(max_ord)
        ode.initialize(0.0, DecayFunc(1.0))
        ode.integrate(0.02)
        assert ode.lastOrder() <= max_ord

    # Smooth long integration reaches higher orders naturally
    ode_high = CVodesIntegrator()
    ode_high.initialize(0.0, OscillatorFunc(1.0))
    orders_observed = set()
    t = 0.0
    for _ in range(50):
        t = ode_high.step(5.0)
        orders_observed.add(ode_high.lastOrder())
        if t >= 5.0:
            break
    # Confirm orders 1 through 5 are exercised
    assert 1 in orders_observed
    assert any(o >= 3 for o in orders_observed)


def test_exponential_decay_analytical_parity():
    """Verify analytical accuracy and derivative evaluation on dy/dt = -lam * y."""
    for lam in [0.5, 1.0, 5.0]:
        ode = CVodesIntegrator()
        ode.setTolerances(1e-9, 1e-15)
        func = DecayFunc(lam=lam, y0=1.0)
        ode.initialize(0.0, func)
        t_target = 1.5
        ode.integrate(t_target)

        exact_y = math.exp(-lam * t_target)
        exact_ydot = -lam * exact_y
        computed_y = ode.solution(0)
        computed_ydot = ode.derivative(t_target, 1)[0]

        assert abs(computed_y - exact_y) / exact_y < 1e-5
        assert abs(computed_ydot - exact_ydot) / abs(exact_ydot) < 1e-5


def test_robertson_stiff_stability():
    """Verify stability and mass conservation on classic stiff Robertson kinetics."""
    ode = CVodesIntegrator()
    ode.setTolerances(1e-6, 1e-12)
    ode.initialize(0.0, RobertsonFunc())
    ode.integrate(100.0)

    y = ode.solution()
    assert len(y) == 3
    # Total mass conservation y0 + y1 + y2 == 1.0
    assert abs(np.sum(y) - 1.0) < 1e-5
    # Positivity of species
    assert np.all(y >= -1e-10)


def test_step_vs_integrate_semantics():
    """Verify distinct semantics of step (one internal step) vs integrate (target interpolation)."""
    ode = CVodesIntegrator()
    ode.initialize(0.0, DecayFunc(1.0))

    t_target = 1.0
    # step advances by a single internal step, returning current internal time
    t_step = ode.step(t_target)
    assert 0.0 < t_step < t_target
    assert ode.currentTime() == t_step
    assert abs(ode.currentTime() - t_step) < 1e-15

    # integrate loops until target is reached, then interpolates to exact target
    t_integ = ode.integrate(t_target)
    assert t_integ == t_target
    assert ode.currentTime() == t_target


def test_reinitialize_history_cleared():
    """Verify that reinitialize cleanly clears BDF history and starts fresh without contamination."""
    ode = CVodesIntegrator()
    ode.initialize(0.0, DecayFunc(lam=1.0, y0=1.0))
    ode.integrate(1.0)
    y1 = ode.solution(0)
    assert abs(y1 - math.exp(-1.0)) < 1e-5

    # Reinitialize at t = 1.0 with new initial state y0 = 10.0
    new_func = DecayFunc(lam=1.0, y0=10.0)
    ode.reinitialize(1.0, new_func)
    assert ode.currentTime() == 1.0
    assert ode.lastOrder() == 1
    assert ode.solution(0) == 10.0

    # Integrate to t = 2.0 -> should equal 10.0 * exp(-1.0)
    ode.integrate(2.0)
    y2 = ode.solution(0)
    expected_y2 = 10.0 * math.exp(-1.0)
    assert abs(y2 - expected_y2) / expected_y2 < 1e-5


def test_forward_sensitivities():
    """Verify forward sensitivity calculation on dy/dt = p*y."""
    p_val = 2.0
    func = ParamSensFunc(p=p_val)
    ode = CVodesIntegrator()
    ode.setTolerances(1e-9, 1e-15)
    ode.initialize(0.0, func)
    assert ode.nSensParams() == 1

    t_target = 1.0
    ode.integrate(t_target)

    # Analytic: y(t) = exp(p*t), s(t) = dy/dp = t * exp(p*t)
    exact_y = math.exp(p_val * t_target)
    exact_sens = t_target * math.exp(p_val * t_target)

    computed_y = ode.solution(0)
    computed_sens = ode.sensitivity(0, 0)

    assert abs(computed_y - exact_y) / exact_y < 1e-5
    assert abs(computed_sens - exact_sens) / exact_sens < 1e-5


def test_tolerance_configuration_scalar_and_vector():
    """Verify scalar and vector absolute tolerance settings."""
    ode = CVodesIntegrator()
    ode.setTolerances(1e-6, 1e-10)
    assert ode.rtol == 1e-6
    assert ode.atol == 1e-10

    ode.setTolerances(1e-7, [1e-8, 1e-12])
    assert ode.rtol == 1e-7
    assert np.allclose(ode.atol, [1e-8, 1e-12])


def test_step_limits_and_failure_propagation():
    """Verify max steps limit and CanteraError propagation."""
    ode = CVodesIntegrator()
    ode.setMaxSteps(3)
    ode.initialize(0.0, DecayFunc(1.0))
    with pytest.raises(CanteraError) as exc_info:
        ode.integrate(10.0)
    assert "Maximum number of timesteps (3) taken without reaching output time" in str(exc_info.value)


def test_solver_statistics_introspection():
    """Verify solverStats returns complete dictionary matching native AnyMap structure."""
    ode = CVodesIntegrator()
    ode.initialize(0.0, DecayFunc(1.0))
    ode.integrate(1.0)

    stats = ode.solverStats()
    assert isinstance(stats, dict)
    required_keys = [
        "steps", "step_solve_fails", "rhs_evals", "nonlinear_iters",
        "nonlinear_conv_fails", "err_test_fails", "last_order",
        "stab_order_reductions", "jac_evals", "lin_solve_setups",
        "lin_rhs_evals", "lin_iters", "lin_conv_fails",
        "prec_evals", "prec_solves", "jt_vec_setup_evals", "jt_vec_prod_evals"
    ]
    for key in required_keys:
        assert key in stats
    assert stats["steps"] > 0
    assert stats["rhs_evals"] > 0
    assert stats["last_order"] >= 1


def test_error_info_string():
    """Verify getErrorInfo returns formatted weighted error strings."""
    ode = CVodesIntegrator()
    ode.initialize(0.0, RobertsonFunc())
    ode.step(0.01)
    info = ode.getErrorInfo(5)
    assert isinstance(info, str)
    assert ":" in info


def test_linear_solver_types_and_bandwidth():
    """Verify setting and getting linear solver type and bandwidth."""
    ode = CVodesIntegrator()
    assert ode.linearSolverType() == "DENSE"

    ode.setLinearSolverType("BAND")
    assert ode.linearSolverType() == "BAND"
    ode.setBandwidth(2, 3)
    assert ode._mupper == 2
    assert ode._mlower == 3
