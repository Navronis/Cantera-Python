import numpy as np
import pytest
from cantera_python.numerics import (
    BandMatrix,
    DenseMatrix,
    SystemJacobian,
    AdaptivePreconditioner,
    FuncEval,
    CVodesIntegrator,
    IdasIntegrator,
    NumericsError,
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

def test_adaptive_preconditioner():
    A = np.array([[10.0, 0.1, 0.0],
                  [0.05, 5.0, 0.2],
                  [0.0, 0.1, 2.0]])
    prec = AdaptivePreconditioner(A, drop_tol=0.2)
    r = np.array([10.0, 5.0, 2.0])
    z = prec.solve(r)
    assert len(z) == 3
    assert np.all(np.isfinite(z))

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
