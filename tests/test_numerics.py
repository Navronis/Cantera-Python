import numpy as np
import pytest
from cantera_python.numerics import (
    BandMatrix,
    DenseMatrix,
    SystemJacobian,
    AdaptivePreconditioner,
    FuncEval,
    CanteraError,
    CVodesIntegrator,
    IdasIntegrator,
    NumericsError,
    _least_squares_householder,
    _matrix_rank_householder,
    _python_l2_norm,
)

def test_band_matrix_basic_and_solve():
    n = 5
    kl = 1
    ku = 1
    A = BandMatrix(n, kl, ku)
    # Construct tridiagonal system
    # A = [ 2 -1  0  0  0]
    #     [-1  2 -1  0  0]
    #     [ 0 -1  2 -1  0]
    #     [ 0  0 -1  2 -1]
    #     [ 0  0  0 -1  2]
    dense_expected = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2.0
        dense_expected[i, i] = 2.0
        if i > 0:
            A[i, i - 1] = -1.0
            dense_expected[i, i - 1] = -1.0
        if i < n - 1:
            A[i, i + 1] = -1.0
            dense_expected[i, i + 1] = -1.0

    b = np.array([1.0, 0.0, 0.0, 0.0, 6.0])
    x_expected = np.linalg.solve(dense_expected, b)

    # Matrix-vector multiply
    b_test = A.mult(x_expected)
    assert np.allclose(b_test, b, atol=1e-12)

    # LU Factor and solve
    A.factor()
    x_computed = A.solve(b)
    assert np.allclose(x_computed, x_expected, atol=1e-12)


def test_band_matrix_python_lu_pivoting_multiple_rhs_and_source_matrix():
    matrix = np.array([
        [0.0, 2.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 4.0, 1.0],
        [0.0, 0.0, 1.0, 3.0],
    ])
    exact = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.25], [3.0, 1.5]])
    rhs = matrix @ exact
    band = BandMatrix(4, 1, 1)
    for i in range(4):
        for j in range(max(0, i - 1), min(4, i + 2)):
            band[i, j] = matrix[i, j]

    assert np.array_equal(band.to_dense(), matrix)
    band.factor()
    assert band._pivots[0] == 1
    assert np.allclose(band.solve(rhs), exact, rtol=1e-13, atol=1e-13)
    assert np.array_equal(band.to_dense(), matrix)

    # Mutating the source matrix invalidates cached LU data before another solve.
    updated = matrix.copy()
    updated[0, 0] = 4.0
    band[0, 0] = 4.0
    assert np.allclose(band.solve(updated @ exact), exact, rtol=1e-13, atol=1e-13)


def test_band_matrix_python_lu_rejects_exact_singularity():
    band = BandMatrix(2, 1, 1)
    band[0, 0] = 1.0
    band[0, 1] = 2.0
    band[1, 0] = 2.0
    band[1, 1] = 4.0
    with pytest.raises(NumericsError, match="exactly singular"):
        band.factor()

def test_dense_matrix_factor_and_invert():
    n = 4
    np.random.seed(42)
    mat = np.random.randn(n, n) + 5.0 * np.eye(n)
    D = DenseMatrix(n, n)
    for i in range(n):
        for j in range(n):
            D[i, j] = mat[i, j]

    b = np.array([1.0, 2.0, 3.0, 4.0])
    x_expected = np.linalg.solve(mat, b)

    D.factor()
    x_computed = D.solve(b)
    assert np.allclose(x_computed, x_expected, atol=1e-12)

    inv_expected = np.linalg.inv(mat)
    D_inv = DenseMatrix(n, n)
    for i in range(n):
        for j in range(n):
            D_inv[i, j] = mat[i, j]
    inv_computed = D_inv.invert()
    assert np.allclose(inv_computed, inv_expected, atol=1e-12)


def test_dense_matrix_python_lu_pivoting_and_multiple_rhs():
    # The first diagonal entry is zero, forcing the same row-pivoting path used
    # by DGETRF. Exercise both vector and matrix right-hand sides.
    matrix = np.array([[0.0, 2.0, 1.0], [1.0, -2.0, -3.0], [2.0, 3.0, 1.0]])
    expected = np.array([[1.0, -2.0], [0.5, 3.0], [-1.0, 0.25]])
    rhs = matrix @ expected
    dense = DenseMatrix(*matrix.shape)
    dense.data[:, :] = matrix

    dense.factor()
    assert dense.ipiv()[0] == 2
    assert np.allclose(dense.solve(rhs), expected, rtol=1e-13, atol=1e-13)
    assert np.allclose(matrix @ dense.solve(rhs), rhs, rtol=1e-13, atol=1e-13)

    # Public row/column mutation must discard cached factors before the next
    # solve, or the result would silently use the old operator.
    updated = matrix.copy()
    updated[1, :] = [3.0, 1.0, -1.0]
    dense.setRow(1, updated[1, :])
    updated_rhs = updated @ expected
    assert np.allclose(dense.solve(updated_rhs), expected, rtol=1e-13, atol=1e-13)


def test_dense_matrix_python_lu_rejects_exact_singularity():
    dense = DenseMatrix(2, 2)
    dense.data[:, :] = [[1.0, 2.0], [2.0, 4.0]]
    with pytest.raises(NumericsError, match="exactly singular"):
        dense.factor()


def test_python_norms_and_lu_based_reciprocal_condition():
    matrix = np.array([[4.0, 1.0], [2.0, 3.0]])
    dense = DenseMatrix(2, 2)
    dense.data[:, :] = matrix
    assert dense.oneNorm() == 6.0
    assert np.isclose(dense.rcond(), 1.0 / np.linalg.cond(matrix, p=1))
    assert np.isclose(dense.rcond(6.0), 1.0 / np.linalg.cond(matrix, p=1))

    band = BandMatrix(2, 1, 1)
    for i in range(2):
        for j in range(2):
            band[i, j] = matrix[i, j]
    assert band.oneNorm() == 6.0
    assert np.isclose(band.rcond(), 1.0 / np.linalg.cond(matrix, p=1))
    assert _python_l2_norm([1e308, 1e308]) == pytest.approx(np.hypot(1e308, 1e308))

def test_system_jacobian():
    # f(y) = [y[0]^2 + y[1], y[0] * y[1]^3]
    class SimpleSystem(FuncEval):
        def neq(self):
            return 2
        def eval_f(self, t, y, ydot):
            ydot[0] = y[0]**2 + y[1]
            ydot[1] = y[0] * (y[1]**3)

    sys = SimpleSystem()
    jac_eval = SystemJacobian(sys)
    y0 = np.array([2.0, 3.0])
    J = jac_eval.eval(0.0, y0)

    # Analytical J:
    # df0/dy0 = 2*y[0] = 4
    # df0/dy1 = 1
    # df1/dy0 = y[1]^3 = 27
    # df1/dy1 = 3*y[0]*y[1]^2 = 3 * 2 * 9 = 54
    J_exact = np.array([[4.0, 1.0], [27.0, 54.0]])
    assert np.allclose(J, J_exact, rtol=1e-5, atol=1e-6)


def test_system_jacobian_solve_uses_python_lu():
    jac_eval = SystemJacobian(n=2)
    jac_eval._mat[:, :] = [[0.0, 2.0], [1.0, 3.0]]
    expected = np.array([1.5, -0.25])
    rhs = jac_eval._mat @ expected
    assert np.allclose(jac_eval.solve(rhs), expected, rtol=1e-14, atol=1e-14)


def test_python_pivoted_householder_least_squares_and_rank():
    matrix = np.array([[1.0, 1000.0], [2.0, -1.0], [3.0, 1.0], [4.0, 2.0]])
    expected = np.array([1.25, -0.75])
    rhs = matrix @ expected + np.array([1e-4, -2e-4, 3e-4, -1e-4])
    actual = _least_squares_householder(matrix, rhs, rcond=1e-12)
    reference, *_ = np.linalg.lstsq(matrix, rhs, rcond=1e-12)
    assert np.allclose(actual, reference, rtol=1e-11, atol=1e-11)

    dependent = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert _matrix_rank_householder(dependent, 1e-12) == 1

    # Rank-deficient solves need not choose NumPy's minimum-norm vector, but
    # they must minimize the residual and report their numerical rank.
    dependent_rhs = np.array([1.0, 2.0, 3.0])
    rank_deficient_solution = _least_squares_householder(
        dependent, dependent_rhs, rcond=1e-12)
    assert np.isfinite(rank_deficient_solution).all()
    assert np.linalg.norm(dependent @ rank_deficient_solution - dependent_rhs) < 1e-10

    wide = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    wide_rhs = np.array([2.0, 3.0])
    wide_solution = _least_squares_householder(wide, wide_rhs)
    wide_reference, *_ = np.linalg.lstsq(wide, wide_rhs, rcond=None)
    assert np.allclose(wide_solution, wide_reference, rtol=1e-13, atol=1e-13)

    wide_rank_deficient = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    inconsistent_rhs = np.array([1.0, 3.0])
    deficient_solution = _least_squares_householder(
        wide_rank_deficient, inconsistent_rhs, rcond=1e-12)
    deficient_reference, *_ = np.linalg.lstsq(
        wide_rank_deficient, inconsistent_rhs, rcond=1e-12)
    assert np.allclose(deficient_solution, deficient_reference,
                       rtol=1e-12, atol=1e-12)

def test_adaptive_preconditioner():
    A = np.array([[10.0, 0.1, 0.0],
                  [0.05, 5.0, 0.2],
                  [0.0, 0.1, 2.0]])
    prec = AdaptivePreconditioner(A, drop_tol=0.2)
    r = np.array([10.0, 5.0, 2.0])
    z = prec.solve(r)
    # The threshold is 0.2 * max(abs(diag(A))) = 2, so all off-diagonal
    # terms are discarded and the diagonal system has the exact solution 1.
    assert np.allclose(z, np.ones(3), rtol=0.0, atol=1e-14)


def test_adaptive_preconditioner_python_lu_full_and_block_solves():
    matrix = np.array([[4.0, 1.0, 0.0], [2.0, 3.0, 0.0],
                       [0.0, 0.0, 5.0]])
    expected = np.array([2.0, -1.0, 3.0])
    rhs = matrix @ expected

    full = AdaptivePreconditioner(matrix)
    assert np.allclose(full.solve(rhs), expected, rtol=1e-14, atol=1e-14)

    blocks = AdaptivePreconditioner(3)
    blocks.initialize([2, 1])
    blocks.factorize(matrix, [0, 2])
    assert np.allclose(blocks.solve(rhs, [0, 2]), expected,
                       rtol=1e-14, atol=1e-14)

    # Updating an entry after a solve must invalidate and rebuild the factor.
    updated = matrix.copy()
    updated[0, 0] = 6.0
    full.setValue(0, 0, 6.0)
    updated_expected = np.array([-1.0, 2.0, 0.5])
    assert np.allclose(full.solve(updated @ updated_expected), updated_expected,
                       rtol=1e-14, atol=1e-14)

def test_cvodes_integrator():
    # Stiff decay: dy/dt = -100 * y, y(0) = 1.0 => y(t) = exp(-100*t)
    class Decay(FuncEval):
        def neq(self):
            return 1
        def eval_f(self, t, y, ydot):
            ydot[0] = -100.0 * y[0]

    sys = Decay()
    cv = CVodesIntegrator(sys)
    cv.initialize(0.0, np.array([1.0]))
    cv.set_tolerances(reltol=1e-6, abstol=1e-10)

    tout = 0.05
    cv.integrate(tout)
    y_final = cv.solution()
    y_exact = np.exp(-100.0 * tout)
    assert np.isclose(y_final[0], y_exact, rtol=1e-4)


def test_cvodes_python_bdf_variable_order_stiff_decay():
    """The in-package BDF path handles stiff decay without SciPy's integrator."""
    class Decay(FuncEval):
        def neq(self):
            return 1

        def eval_f(self, t, y, ydot):
            ydot[0] = -100.0 * y[0]

    cv = CVodesIntegrator(Decay())
    cv.initialize(0.0, np.array([1.0]))
    cv.set_tolerances(reltol=1e-6, abstol=1e-10)
    cv.integrate(0.05)

    exact = np.exp(-5.0)
    assert np.isclose(cv.solution()[0], exact, rtol=5e-6, atol=1e-12)
    assert cv._cv_order > 1
    assert cv.nEvals() < 5000


def test_cvode_dense_factor_reuse_scales_newton_correction():
    class IdentityFactor:
        def solve(self, rhs):
            return np.asarray(rhs, dtype=float).copy()

    cv = CVodesIntegrator(lambda time, state: np.zeros_like(state))
    cv.initialize(0.0, np.array([0.0]))
    factor = IdentityFactor()
    cv._cv_linear_factor = factor
    cv._cv_lsetup_gamma = 5.0 / 6.0
    cv._cv_last_lsetup_step = 0
    cv._cv_last_jac_step = 0
    cv._cv_nst = 1
    cv._cv_h = 1.0
    cv._cv_force_lsetup = False
    cv._cv_force_jacobian = False
    cv._cv_zn[1][0] = -1.0

    ell = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    tq = [0.0, 0.0, 1.0, 0.0, 10.0, 0.0]
    correction, _, converged = cv._cv_newton_corrector(
        1.0, ell, tq, np.ones(1))

    assert converged
    # gamma / gamma_prev = 1.2; CVLS scales by 2 / (1 + ratio).
    assert cv._cv_linear_factor is factor
    assert correction[0] == pytest.approx(2.0 / 2.2)


def test_cvodes_first_step_growth_uses_pinned_sundials_limit():
    """Pinned CVODE permits 10000x growth after the first accepted step."""
    cv = CVodesIntegrator(lambda t, y: np.ones_like(y))
    cv.initialize(0.0, np.array([1.0]))
    cv.set_tolerances(reltol=1e-9, abstol=1e-12)

    assert cv._cv_etamax == 10000.0
    cv.integrate(0.01)

    # The constant-derivative solution is exactly linear. The first-step
    # growth cap from SUNDIALS cvode_impl.h reaches tout in three BDF steps.
    assert cv.solution()[0] == pytest.approx(1.01, abs=1e-12)
    assert cv._cv_nst == 3


def test_cvodes_dense_difference_quotient_step_matches_pinned_cvls():
    """Zero components use CVLS's norm-scaled minInc, not sqrt(eps)*atol."""
    cv = CVodesIntegrator()
    cv._cv_h = 4.0e-5
    rhs = np.array([1.0e12, -2.0e12, 0.0])
    weights = np.array([1.0e2, 1.0e4, 1.0e15])
    fnorm = cv._cv_wrms(rhs, weights)
    min_increment = 1000.0 * abs(cv._cv_h) * np.finfo(float).eps * 3 * fnorm

    assert cv._cv_dense_dq_min_increment(rhs, weights, 3) == pytest.approx(
        min_increment, rel=1e-15)
    zero_state_increment = cv._cv_dense_dq_increment(
        0.0, weights[2], min_increment)
    assert zero_state_increment == pytest.approx(min_increment / weights[2])
    assert zero_state_increment > np.sqrt(np.finfo(float).eps) / weights[2]


def test_cvodes_python_bdf_retries_recoverable_rhs_error():
    """CanteraError from a trial RHS is recoverable by reducing the step."""
    class TransientTrialFailure(FuncEval):
        def __init__(self):
            self.failed = False

        def neq(self):
            return 1

        def eval_f(self, t, y, ydot):
            if t > 0.0 and not self.failed:
                self.failed = True
                raise CanteraError("transient invalid trial state")
            ydot[0] = -y[0]

    func = TransientTrialFailure()
    cv = CVodesIntegrator(func)
    cv.initialize(0.0, np.array([1.0]))
    cv.set_tolerances(reltol=1e-8, abstol=1e-12)
    cv.integrate(1.0)

    assert func.failed
    assert cv.solution()[0] == pytest.approx(np.exp(-1.0), rel=2e-6)


def test_cvodes_python_bdf_oscillator_and_bdf_error_coefficient():
    class Oscillator(FuncEval):
        def neq(self):
            return 2

        def eval_f(self, t, y, ydot):
            ydot[0] = y[1]
            ydot[1] = -y[0]

    cv = CVodesIntegrator(Oscillator())
    cv.initialize(0.0, np.array([0.0, 1.0]))
    cv.set_tolerances(reltol=1e-8, abstol=1e-10)
    cv.integrate(np.pi / 2.0)

    assert np.allclose(cv.solution(), np.array([1.0, 0.0]), rtol=0.0, atol=5e-7)
    assert cv._cv_order > 1
    # For equal adjacent steps and q=2, pinned cvSetBDF/cvSetTqBDF gives 2/9.
    assert CVodesIntegrator._bdf_error_coefficient(2, 0.25, [0.25]) == pytest.approx(2.0 / 9.0)

    cv._cv_q = 2
    cv._cv_qwait = 1
    cv._cv_tau[1:3] = [0.25, 0.25]
    ell, tq = cv._cv_set_bdf(0.25)
    assert ell[0] == pytest.approx(1.0)
    assert ell[1] == pytest.approx(1.5)
    assert ell[2] == pytest.approx(0.5)
    assert tq[1] == pytest.approx(1.0)
    assert tq[2] == pytest.approx(2.0 / 9.0)

def test_idas_integrator_semi_explicit():
    # ODE disguised as DAE:
    # y0' = y1
    # y1' = -y0
    class Oscillator(FuncEval):
        def neq(self):
            return 2
        def eval_f(self, t, y, ydot):
            ydot[0] = y[1]
            ydot[1] = -y[0]

    sys = Oscillator()
    idas = IdasIntegrator(sys)
    idas.initialize(0.0, np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    t_target = np.pi / 2.0
    idas.integrate(t_target)
    y = idas.solution()
    # at t = pi/2, y0 = sin(pi/2) = 1, y1 = cos(pi/2) = 0
    assert np.isclose(y[0], 1.0, atol=1e-3)
    assert np.isclose(y[1], 0.0, atol=1e-3)


def test_idas_integrator_python_residual_dae_with_algebraic_variable():
    # Index-1 semi-explicit DAE: x' = -x + z, 0 = z - x**2.
    # Eliminating z gives x' = -x*(1-x), x(0)=1/2.
    def residual(t, y, ydot):
        return np.array([ydot[0] + y[0] - y[1], y[1] - y[0] ** 2])

    idas = IdasIntegrator()
    idas.setTolerances(2e-3, 1e-6)
    idas.setMaxStepSize(0.025)
    idas.initialize(0.0, residual, np.array([0.5, 0.25]), np.array([-0.25, 0.0]))
    idas.integrate(0.25)
    idas.integrate(0.5)

    y = idas.solution()
    expected_x = 1.0 / (1.0 + np.exp(0.5))
    assert np.isclose(y[0], expected_x, atol=2e-3)
    assert np.isclose(y[1], y[0] ** 2, atol=2e-4)
    assert idas.currentTime() == pytest.approx(0.5)
    assert 1 < idas.lastOrder() <= idas.maxOrder()


def test_idas_integrator_func_eval_dae_initialization_contract():
    class AlgebraicDae(FuncEval):
        def neq(self):
            return 2

        def getConstraints(self, ids):
            # IDASetId convention: 1 is differential and 0 is algebraic.
            ids[:] = [1.0, 0.0]

        def getStateDae(self, y, ydot):
            y[:] = [0.5, 0.25]
            ydot[:] = [-0.25, 0.0]

        def evalDae(self, t, y, ydot, residual):
            residual[0] = ydot[0] + y[0] - y[1]
            residual[1] = y[1] - y[0] ** 2

    idas = IdasIntegrator()
    idas.setTolerances(2e-3, 1e-6)
    idas.initialize(0.0, AlgebraicDae())
    assert np.array_equal(idas._dae_ids, [1.0, 0.0])
    idas.integrate(0.1)

    y = idas.solution()
    assert np.isfinite(y).all()
    assert np.isclose(y[1], y[0] ** 2, atol=2e-4)


def test_idas_algebraic_error_suppression_matches_component_ids():
    class AlgebraicDae(FuncEval):
        def neq(self):
            return 2

        def getConstraints(self, ids):
            ids[:] = [1.0, 0.0]

        def getStateDae(self, y, ydot):
            y[:] = [1.0, 0.0]
            ydot[:] = [0.0, 0.0]

        def evalDae(self, t, y, ydot, residual):
            residual[:] = [ydot[0], y[1] - y[0]]

    idas = IdasIntegrator()
    idas.initialize(0.0, AlgebraicDae())
    assert idas.algebraicInErrorTest()

    estimated_error = np.array([0.25, 1.0e6])
    state = np.zeros(2)
    included = idas._dae_weighted_rms(
        estimated_error, state,
        include_algebraic=idas.algebraicInErrorTest(),
    )
    idas.includeAlgebraicInErrorTest(False)
    differential_only = idas._dae_weighted_rms(
        estimated_error, state,
        include_algebraic=idas.algebraicInErrorTest(),
    )
    assert included > differential_only
    assert differential_only == pytest.approx(estimated_error[0] / 1e-15)


def test_idas_native_default_iteration_and_failure_limits():
    idas = IdasIntegrator()
    # Cantera leaves max nonlinear iterations unset (0); SUNDIALS Newton then
    # uses its pinned default of three. IDAS defaults both retry limits to ten.
    assert idas.maxNonlinIterations() == 0
    assert idas._dae_max_newton == 3
    assert idas.maxNonlinConvFailures() == -1
    assert idas._dae_max_conv_failures == 10
    assert idas._dae_max_error_test_failures == 10

    idas.setMaxNonlinIterations(6)
    idas.setMaxNonlinConvFailures(4)
    idas.setMaxErrTestFails(7)
    assert idas.maxNonlinIterations() == 6
    assert idas._dae_max_newton == 6
    assert idas.maxNonlinConvFailures() == 4
    assert idas._dae_max_conv_failures == 4
    # Pinned IdasIntegrator::applyOptions does not forward this saved option;
    # it becomes active only if set after initialization.
    assert idas._dae_max_error_test_failures == 10
    idas.initialize(0.0, lambda t, y, ydot: ydot,
                    np.array([1.0]), np.array([0.0]))
    idas.setMaxErrTestFails(7)
    assert idas._dae_max_error_test_failures == 7


def test_idas_error_test_retry_schedule_matches_pinned_ida_policy():
    idas = IdasIntegrator()
    first_h, first_order = idas._dae_after_error_test_failure(1.0, 1.0, 4, 1)
    assert first_order == 4
    assert first_h == pytest.approx(0.9 * (2.0 + 1e-4) ** (-1.0 / 5.0))

    second_h, second_order = idas._dae_after_error_test_failure(1.0, 3.0, 4, 2)
    assert second_order == 4
    assert second_h == pytest.approx(0.25)

    third_h, third_order = idas._dae_after_error_test_failure(1.0, 3.0, 4, 3)
    assert third_order == 1
    assert third_h == pytest.approx(0.25)

    idas.initialize(0.0, lambda t, y, ydot: ydot,
                    np.array([1.0]), np.array([0.0]))
    idas.setMaxErrTestFails(2)
    second_h, second_order = idas._dae_after_error_test_failure(1.0, 3.0, 4, 2)
    assert second_h == pytest.approx(0.25)
    assert second_order == 4
    with pytest.raises(NumericsError, match="error-test failures"):
        idas._dae_after_error_test_failure(1.0, 3.0, 4, 3)


def test_idas_func_eval_requires_explicit_dae_component_ids():
    class MissingIds(FuncEval):
        def neq(self):
            return 1

        def getStateDae(self, y, ydot):
            y[:] = [1.0]
            ydot[:] = [0.0]

        def evalDae(self, t, y, ydot, residual):
            residual[:] = ydot

    with pytest.raises(NumericsError, match="must implement getConstraints"):
        IdasIntegrator().initialize(0.0, MissingIds())


def test_func_eval_dae_no_throw_status_distinguishes_recoverable_and_fatal():
    class Recoverable(FuncEval):
        def evalDae(self, t, y, ydot, residual):
            raise CanteraError("recoverable chemistry state")

    class Fatal(FuncEval):
        def evalDae(self, t, y, ydot, residual):
            raise RuntimeError("fatal callback defect")

    y = np.array([1.0])
    ydot = np.array([0.0])
    residual = np.zeros(1)
    recoverable = Recoverable()
    recoverable.suppressErrors(True)
    assert recoverable.evalDaeNoThrow(0.0, y, ydot, residual) == 1
    assert "recoverable chemistry state" in recoverable.getErrors()[0]

    fatal = Fatal()
    fatal.suppressErrors(True)
    assert fatal.evalDaeNoThrow(0.0, y, ydot, residual) == -1
    assert "fatal callback defect" in fatal.getErrors()[0]

    # A return code from evalDae itself is not a residual vector; unlike
    # evalDaeNoThrow, the native virtual evalDae contract returns void.
    class InvalidReturn(FuncEval):
        def evalDae(self, t, y, ydot, residual):
            return 1

    idas = IdasIntegrator()
    idas._res = InvalidReturn()
    idas._y = y.copy()
    with pytest.raises(NumericsError, match="did not fill every component"):
        idas._dae_residual(0.0, y, ydot)


def test_idas_retries_after_recoverable_residual_callback_error():
    class RecoverOnce(FuncEval):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def neq(self):
            return 2

        def getConstraints(self, ids):
            ids[:] = [1.0, 0.0]

        def getStateDae(self, y, ydot):
            y[:] = [0.5, 0.25]
            ydot[:] = [-0.25, 0.0]

        def evalDae(self, t, y, ydot, residual):
            self.calls += 1
            if self.calls == 1:
                raise CanteraError("recoverable first residual evaluation")
            residual[:] = [ydot[0] + y[0] - y[1], y[1] - y[0] ** 2]

    func = RecoverOnce()
    idas = IdasIntegrator()
    idas.setTolerances(2e-3, 1e-6)
    idas.initialize(0.0, func)
    idas.integrate(0.05)

    y = idas.solution()
    assert func.calls > 1
    assert np.isfinite(y).all()
    assert np.isclose(y[1], y[0] ** 2, atol=2e-4)


def test_idas_python_variable_step_bdf_coefficients_orders_one_to_five():
    for order in range(1, 6):
        h = 0.125
        times = [-((j + 1) * h) for j in range(order)]
        coeff, error_coeff = IdasIntegrator._dae_bdf_coefficients(0.0, h, times)
        # Constant-step BDF coefficients from the IDAS backward-difference
        # formulas; the leading local-error coefficient is 1/(q+1).
        expected = {
            1: [1.0, -1.0],
            2: [1.5, -2.0, 0.5],
            3: [11.0 / 6.0, -3.0, 1.5, -1.0 / 3.0],
            4: [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 0.25],
            5: [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 1.25, -0.2],
        }[order]
        assert np.allclose(coeff, expected, rtol=0.0, atol=2e-13)
        assert error_coeff == pytest.approx(1.0 / (order + 1), abs=2e-13)


def test_idas_python_variable_order_bdf_stiff_residual_decay():
    rate = 20.0
    idas = IdasIntegrator()
    idas.setTolerances(1e-6, 1e-10)
    idas.initialize(
        0.0,
        lambda t, y, ydot: ydot + rate * y,
        np.array([1.0]),
        np.array([-rate]),
    )
    idas.integrate(0.5)

    assert np.isclose(idas.solution()[0], np.exp(-rate * 0.5), rtol=5e-5, atol=1e-12)
    assert 1 < idas.lastOrder() <= 5
    assert idas.nEvals() < 1000


def test_idas_ode_constructor_state_initialization_uses_python_bdf():
    idas = IdasIntegrator(lambda t, y: -y)
    idas.initialize(0.0, np.array([1.0]))
    idas.integrate(1.0)
    assert np.isclose(idas.solution()[0], np.exp(-1.0), rtol=1e-5)
