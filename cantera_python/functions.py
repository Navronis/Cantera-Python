"""Editable stdlib ports of Cantera's scalar functors and sampled quadrature.

Source: Cantera commit 726522be4e2a13454d8415b7ef799d621f665cf3,
include/cantera/numerics/{Func1.h,funcs.h} and src/numerics/{Func1.cpp,funcs.cpp}.
See provenance/functions.json for coverage and behavioral limits.

This is the C++ class API expressed in Python, not Cantera's Cython factory API.
In particular ``type()`` and ``c()`` are methods. Constructor sequences follow
the order in the original headers. Scalar operations use Python's real-valued
``math`` functions: domain/overflow errors raise Python exceptions instead of
returning the C++ floating-point NaN/Inf. Finite valid-domain results are the
target; bitwise equivalence to a particular C++ build is not claimed.

Pinned-source details intentionally retained:
* Log1.derivative() is a/x, although the analytic derivative of log(a*x) is 1/x.
* Tabulated1(method='previous') returns the left value AT an interior knot.
* Tabulated1's derivative clamps to its first slope below the first knot.
* Periodic1 truncates t/period toward zero, including at negative times.
* The upstream older functors have no derivative implementation; the same
  unsupported operation raises here. This is not a numerical approximation.

Derived from Cantera; distributed under the Cantera BSD license in LICENSE.txt.
"""

from bisect import bisect_left
import math
from numbers import Real


__all__ = [
    "Func1", "Sin1", "Cos1", "Exp1", "Log1", "Pow1", "Const1",
    "Tabulated1", "Sum1", "Diff1", "Product1", "Ratio1", "Composite1",
    "TimesConstant1", "PlusConstant1", "Periodic1", "Gaussian1", "Poly1",
    "Fourier1", "Arrhenius1", "newSumFunction", "newDiffFunction",
    "newProdFunction", "newRatioFunction", "newCompositeFunction",
    "newTimesConstFunction", "newPlusConstFunction", "linearInterp",
    "trapezoidal", "basicSimpson", "simpson", "numericalQuadrature",
]


def _single(value):
    """Resolve the scalar and span<double> constructor overloads."""
    if isinstance(value, Real):
        return float(value)
    params = tuple(value)
    if len(params) != 1:
        raise ValueError("Constructor needs exactly one parameter.")
    return float(params[0])


def _fmt(value):
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


class Func1:
    """Original scalar functor base; the base evaluation returns zero."""

    _type = "functor"
    _order = 3

    def __init__(self, f1=None, f2=None):
        self.m_c = 0.0
        self.m_f1 = f1
        self.m_f2 = None
        if isinstance(f2, Func1):
            self.m_f2 = f2
        elif f2 is not None:
            self.m_c = float(f2)

    def type(self):
        return self._type

    def typeName(self):
        return self._type

    def operator(self, t):
        return self(t)

    def __call__(self, t):
        return self.eval(t)

    def eval(self, t):
        return 0.0

    def derivative(self):
        # This exception is part of the original Func1 behavior. Older
        # functors (Poly1, Fourier1, Gaussian1, Arrhenius1, Periodic1) inherit it.
        raise NotImplementedError(
            f"{type(self).__name__}: derivative is unsupported by the pinned source."
        )

    def isIdentical(self, other):
        if not isinstance(other, Func1):
            return False
        if self.type() == "functor" or self.type() != other.type() or self.m_c != other.m_c:
            return False
        if self.m_f1 is not None:
            if other.m_f1 is None or not self.m_f1.isIdentical(other.m_f1):
                return False
        if self.m_f2 is not None:
            if other.m_f2 is None or not self.m_f2.isIdentical(other.m_f2):
                return False
        return True

    def c(self):
        return self.m_c

    def func1_shared(self):
        return self.m_f1

    def func2_shared(self):
        return self.m_f2

    def order(self):
        return self._order

    def write(self, arg="x"):
        return "\\mathrm{" + self.type() + "}(" + arg + ")"


class _Scalar1(Func1):
    def __init__(self, value):
        super().__init__()
        self.m_c = _single(value)

    def write(self, arg="x"):
        factor = "" if self.m_c == 1.0 else _fmt(self.m_c)
        return "\\" + self.type() + "(" + factor + arg + ")"


class Sin1(_Scalar1):
    """sin(omega*x)."""
    _type = "sin"

    def __init__(self, omega=1.0):
        super().__init__(omega)

    def eval(self, t):
        return math.sin(self.m_c * t)

    def derivative(self):
        return newTimesConstFunction(Cos1(self.m_c), self.m_c)


class Cos1(_Scalar1):
    """cos(omega*x)."""
    _type = "cos"

    def __init__(self, omega=1.0):
        super().__init__(omega)

    def eval(self, t):
        return math.cos(self.m_c * t)

    def derivative(self):
        return newTimesConstFunction(Sin1(self.m_c), -self.m_c)


class Exp1(_Scalar1):
    """exp(a*x)."""
    _type = "exp"

    def __init__(self, a=1.0):
        super().__init__(a)

    def eval(self, t):
        return math.exp(self.m_c * t)

    def derivative(self):
        f = Exp1(self.m_c)
        return newTimesConstFunction(f, self.m_c) if self.m_c != 1.0 else f


class Log1(_Scalar1):
    """log(a*x); derivative() preserves upstream's a/x, not analytic 1/x."""
    _type = "log"

    def __init__(self, a=1.0):
        super().__init__(a)

    def eval(self, t):
        return math.log(self.m_c * t)

    def derivative(self):
        f = Pow1(-1.0)
        return newTimesConstFunction(f, self.m_c) if self.m_c != 1.0 else f


class Pow1(_Scalar1):
    """x**n on the real-valued math.pow domain."""
    _type = "pow"

    def eval(self, t):
        return math.pow(t, self.m_c)

    def derivative(self):
        if self.m_c == 0.0:
            return Const1(0.0)
        if self.m_c == 1.0:
            return Const1(1.0)
        return newTimesConstFunction(Pow1(self.m_c - 1.0), self.m_c)

    def write(self, arg="x"):
        if self.m_c == 0.5:
            return "\\sqrt{" + arg + "}"
        if self.m_c == -0.5:
            return "\\frac{1}{\\sqrt{" + arg + "}}"
        if self.m_c == 1.0:
            return arg
        return "\\left(" + arg + "\\right)^{" + _fmt(self.m_c) + "}"


class Const1(_Scalar1):
    """Constant a."""
    _type = "constant"

    def eval(self, t):
        return self.m_c

    def derivative(self):
        return Const1(0.0)

    def write(self, arg="x"):
        return _fmt(self.m_c)


class Tabulated1(Func1):
    """Piecewise linear or previous-value table, clamped at both endpoints.

    ``Tabulated1(times, values, method='linear')`` or
    ``Tabulated1([t0, ..., tN, f0, ..., fN])``.
    """

    def __init__(self, tvals, fvals=None, method="linear"):
        super().__init__()
        if fvals is None:
            params = tuple(tvals)
            if len(params) < 4 or len(params) % 2:
                raise ValueError("Constructor needs an even number of at least 4 entries.")
            n = len(params) // 2
            tvals, fvals = params[:n], params[n:]
        self.m_tvec = tuple(float(v) for v in tvals)
        self.m_fvec = tuple(float(v) for v in fvals)
        if len(self.m_tvec) != len(self.m_fvec):
            raise ValueError("Expected matching time/value lengths.")
        # Upstream's two-vector overload fails to check empty input, despite
        # eval assuming nonempty. Reject that undefined-input case explicitly.
        if not self.m_tvec:
            raise ValueError("Time/value arrays must not be empty.")
        if any(a > b for a, b in zip(self.m_tvec, self.m_tvec[1:])):
            raise ValueError("Time values are not monotonically increasing.")
        self.setMethod(method)

    def type(self):
        return "tabulated-linear" if self.m_isLinear else "tabulated-previous"

    def setMethod(self, method):
        if method not in ("linear", "previous"):
            raise ValueError("Interpolation method must be 'linear' or 'previous'.")
        self.m_isLinear = method == "linear"

    def isIdentical(self, other):
        return False

    def eval(self, t):
        if t <= self.m_tvec[0]:
            return self.m_fvec[0]
        if t >= self.m_tvec[-1]:
            return self.m_fvec[-1]
        ix = 0
        while t > self.m_tvec[ix + 1]:
            ix += 1
        if self.m_isLinear:
            df = self.m_fvec[ix + 1] - self.m_fvec[ix]
            df /= self.m_tvec[ix + 1] - self.m_tvec[ix]
            df *= t - self.m_tvec[ix]
            return self.m_fvec[ix] + df
        return self.m_fvec[ix]

    def derivative(self):
        if self.m_isLinear:
            tvec = list(self.m_tvec[:-1])
            dvec = [
                (self.m_fvec[i] - self.m_fvec[i - 1])
                / (self.m_tvec[i] - self.m_tvec[i - 1])
                for i in range(1, len(self.m_tvec))
            ]
            tvec.append(self.m_tvec[-1])
            dvec.append(0.0)
        else:
            tvec = [self.m_tvec[0], self.m_tvec[-1]]
            dvec = [0.0, 0.0]
        return Tabulated1(tvec, dvec, "previous")

    def write(self, arg="x"):
        return "\\mathrm{Tabulated}(" + arg + ")"


class Sum1(Func1):
    """f1(x) + f2(x)."""
    _type = "sum"
    _order = 0

    def eval(self, t):
        return self.m_f1.eval(t) + self.m_f2.eval(t)

    def derivative(self):
        return newSumFunction(self.m_f1.derivative(), self.m_f2.derivative())

    def write(self, arg="x"):
        s1, s2 = self.m_f1.write(arg), self.m_f2.write(arg)
        return s1 + " - " + s2[1:] if s2.startswith("-") else s1 + " + " + s2


class Diff1(Func1):
    """f1(x) - f2(x)."""
    _type = "diff"
    _order = 0

    def eval(self, t):
        return self.m_f1.eval(t) - self.m_f2.eval(t)

    def derivative(self):
        return newDiffFunction(self.m_f1.derivative(), self.m_f2.derivative())

    def write(self, arg="x"):
        s1, s2 = self.m_f1.write(arg), self.m_f2.write(arg)
        return s1 + " + " + s2[1:] if s2.startswith("-") else s1 + " - " + s2


class Product1(Func1):
    """f1(x)*f2(x), including its product-rule derivative."""
    _type = "product"
    _order = 1

    def eval(self, t):
        return self.m_f1.eval(t) * self.m_f2.eval(t)

    def derivative(self):
        a1 = newProdFunction(self.m_f1, self.m_f2.derivative())
        a2 = newProdFunction(self.m_f2, self.m_f1.derivative())
        return newSumFunction(a1, a2)

    def write(self, arg="x"):
        s1, s2 = self.m_f1.write(arg), self.m_f2.write(arg)
        if self.m_f1.order() < self.order():
            s1 = "\\left(" + s1 + "\\right)"
        if self.m_f2.order() < self.order():
            s2 = "\\left(" + s2 + "\\right)"
        return s1 + " " + s2


class TimesConstant1(Func1):
    """a*f1(x)."""
    _type = "times-constant"
    _order = 0

    def eval(self, t):
        return self.m_f1.eval(t) * self.m_c

    def derivative(self):
        return newTimesConstFunction(self.m_f1.derivative(), self.m_c)

    def write(self, arg="x"):
        s = self.m_f1.write(arg)
        if self.m_f1.order() < self.order():
            s = "\\left(" + s + "\\right)"
        if self.m_c == 1.0:
            return s
        if self.m_c == -1.0:
            return "-" + s
        if s and "0" <= s[0] <= "9":
            s = "\\left(" + s + "\\right)"
        return _fmt(self.m_c) + s


class PlusConstant1(Func1):
    """f1(x)+a."""
    _type = "plus-constant"
    _order = 0

    def eval(self, t):
        return self.m_f1.eval(t) + self.m_c

    def derivative(self):
        return self.m_f1.derivative()

    def write(self, arg="x"):
        if self.m_c == 0.0:
            return self.m_f1.write(arg)
        return self.m_f1.write(arg) + " + " + _fmt(self.m_c)


class Ratio1(Func1):
    """f1(x)/f2(x), including its quotient-rule derivative."""
    _type = "ratio"
    _order = 1

    def eval(self, t):
        return self.m_f1.eval(t) / self.m_f2.eval(t)

    def derivative(self):
        a1 = newProdFunction(self.m_f1.derivative(), self.m_f2)
        a2 = newProdFunction(self.m_f1, self.m_f2.derivative())
        s = newDiffFunction(a1, a2)
        p = newProdFunction(self.m_f2, self.m_f2)
        return newRatioFunction(s, p)

    def write(self, arg="x"):
        return "\\frac{" + self.m_f1.write(arg) + "}{" + self.m_f2.write(arg) + "}"


class Composite1(Func1):
    """f1(f2(x)), including its chain-rule derivative."""
    _type = "composite"
    _order = 2

    def eval(self, t):
        return self.m_f1.eval(self.m_f2.eval(t))

    def derivative(self):
        d1, d2 = self.m_f1.derivative(), self.m_f2.derivative()
        return newProdFunction(newCompositeFunction(d1, self.m_f2), d2)

    def write(self, arg="x"):
        return self.m_f1.write(self.m_f2.write(arg))


class _Advanced1(Func1):
    def isIdentical(self, other):
        return False


class Gaussian1(_Advanced1):
    """A*exp(-((t-t0)/tau)**2), tau=fwhm/(2*sqrt(log(2)))."""
    _type = "Gaussian"

    def __init__(self, A, t0=None, fwhm=None):
        super().__init__()
        if t0 is None and fwhm is None:
            params = tuple(A)
            if len(params) != 3:
                raise ValueError("Constructor needs exactly 3 parameters.")
            A, t0, fwhm = params
        if t0 is None or fwhm is None:
            raise ValueError("Amplitude, center, and width are required.")
        self.m_A = float(A)
        self.m_t0 = float(t0)
        self.m_tau = float(fwhm) / (2.0 * math.sqrt(math.log(2.0)))

    def eval(self, t):
        x = (t - self.m_t0) / self.m_tau
        return self.m_A * math.exp(-x * x)


class Poly1(_Advanced1):
    """Horner polynomial; coefficients highest degree first: [a_n,...,a_0]."""
    _type = "polynomial3"

    def __init__(self, params):
        super().__init__()
        self.m_cpoly = tuple(float(v) for v in params)
        if not self.m_cpoly:
            raise ValueError("Constructor needs an array that is not empty.")

    def eval(self, t):
        result = self.m_cpoly[0]
        for c in self.m_cpoly[1:]:
            result *= t
            result += c
        return result

    def write(self, arg="x"):
        out = _fmt(self.m_cpoly[-1]) if self.m_cpoly[-1] != 0.0 else ""
        for n, coeff in enumerate(reversed(self.m_cpoly[:-1]), 1):
            if coeff == 0.0:
                continue
            term = ("" if coeff == 1.0 else "-" if coeff == -1.0 else _fmt(coeff)) + arg
            if n > 9:
                term += "^{" + str(n) + "}"
            elif n > 1:
                term += "^" + str(n)
            if not out:
                out = term
            elif out[0] == "-":
                out = term + " - " + out[1:]
            else:
                out = term + " + " + out
        return out


class Fourier1(_Advanced1):
    """a0/2 + sum(a_n*cos(n*omega*t)+b_n*sin(n*omega*t)).

    Accepts (omega, a0, a, b) or [a0,a1,...,aN,omega,b1,...,bN].
    """
    _type = "Fourier"

    def __init__(self, omega, a0=None, a=None, b=None):
        super().__init__()
        if a0 is None and a is None and b is None:
            params = tuple(omega)
            if len(params) < 4 or len(params) % 2:
                raise ValueError("Constructor needs an even number of at least 4 entries.")
            n = len(params) // 2 - 1
            omega, a0, a, b = params[n + 1], params[0], params[1:n + 1], params[n + 2:]
        if a0 is None or a is None or b is None:
            raise ValueError("omega, a0, and both coefficient arrays are required.")
        self.m_omega = float(omega)
        self.m_a0_2 = 0.5 * float(a0)
        self.m_ccos = tuple(float(v) for v in a)
        self.m_csin = tuple(float(v) for v in b)
        if len(self.m_ccos) != len(self.m_csin):
            raise ValueError("Expected matching sin/cos coefficient lengths.")

    def eval(self, t):
        result = self.m_a0_2
        for n, (a, b) in enumerate(zip(self.m_ccos, self.m_csin), 1):
            result += a * math.cos(self.m_omega * n * t) + b * math.sin(self.m_omega * n * t)
        return result


class Arrhenius1(_Advanced1):
    """sum(A_n*T**b_n*exp(-E_n/T)); E is already in temperature units.

    Parameters: [A1,b1,E1,A2,b2,E2,...]. This functor does not divide E by R.
    """
    _type = "Arrhenius"

    def __init__(self, params):
        super().__init__()
        params = tuple(float(v) for v in params)
        if len(params) < 3 or len(params) % 3:
            raise ValueError("Constructor needs at least 3 entries, in multiples of 3.")
        self.m_A, self.m_b, self.m_E = params[0::3], params[1::3], params[2::3]

    def eval(self, t):
        result = 0.0
        for A, b, E in zip(self.m_A, self.m_b, self.m_E):
            result += A * math.pow(t, b) * math.exp(-E / t)
        return result


class Periodic1(Func1):
    """f(t-trunc(t/T)*T); source behavior for negative t is preserved."""
    _type = "periodic"

    def eval(self, t):
        np = int(t / self.m_c)
        time = t - np * self.m_c
        return self.m_f1.eval(time)


def _is_constant(f):
    return f.type() == "constant"


def _is_zero(f):
    return _is_constant(f) and f.c() == 0.0


def _is_one(f):
    return _is_constant(f) and f.c() == 1.0


def _is_times_const(f):
    return f.type() == "times-constant"


def _is_proportional(f1, f2):
    tc1, tc2 = _is_times_const(f1), _is_times_const(f2)
    if not tc1 and not tc2:
        return (True, 1.0) if f1.isIdentical(f2) else (False, 0.0)
    if not tc1 and tc2:
        return (True, f2.c()) if f1.isIdentical(f2.func1_shared()) else (False, 0.0)
    if tc1 and not tc2:
        return (True, 1.0 / f1.c()) if f2.isIdentical(f1.func1_shared()) else (False, 0.0)
    if f2.func1_shared().isIdentical(f1.func1_shared()):
        return True, f2.c() / f1.c()
    return False, 0.0


def newSumFunction(f1, f2):
    if f1.isIdentical(f2):
        return newTimesConstFunction(f1, 2.0)
    if _is_zero(f1):
        return f2
    if _is_zero(f2):
        return f1
    if _is_constant(f2):
        return newPlusConstFunction(f1, f2.c())
    if _is_constant(f1):
        return newPlusConstFunction(f2, f1.c())
    prop, c = _is_proportional(f1, f2)
    if prop:
        return Const1(0.0) if c == -1.0 else newTimesConstFunction(f1, c + 1.0)
    return Sum1(f1, f2)


def newDiffFunction(f1, f2):
    if _is_zero(f2):
        return f1
    if _is_zero(f1):
        return newTimesConstFunction(f2, -1.0)
    if f1.isIdentical(f2):
        return Const1(0.0)
    if _is_constant(f2):
        return newPlusConstFunction(f1, -f2.c())
    prop, c = _is_proportional(f1, f2)
    if prop:
        return Const1(0.0) if c == 1.0 else newTimesConstFunction(f1, 1.0 - c)
    return Diff1(f1, f2)


def newProdFunction(f1, f2):
    if _is_one(f1):
        return f2
    if _is_one(f2):
        return f1
    if _is_zero(f1) or _is_zero(f2):
        return Const1(0.0)
    if _is_constant(f1) and _is_constant(f2):
        return Const1(f1.c() * f2.c())
    if _is_constant(f1):
        return newTimesConstFunction(f2, f1.c())
    if _is_constant(f2):
        return newTimesConstFunction(f1, f2.c())
    if f1.type() == "pow" and f2.type() == "pow":
        return Pow1(f1.c() + f2.c())
    if f1.type() == "exp" and f2.type() == "exp":
        return Exp1(f1.c() + f2.c())
    tc1, tc2 = _is_times_const(f1), _is_times_const(f2)
    if tc1 or tc2:
        c1, ff1 = (f1.c(), f1.func1_shared()) if tc1 else (1.0, f1)
        c2, ff2 = (f2.c(), f2.func1_shared()) if tc2 else (1.0, f2)
        p = newProdFunction(ff1, ff2)
        return newTimesConstFunction(p, c1 * c2) if c1 * c2 != 1.0 else p
    return Product1(f1, f2)


def newRatioFunction(f1, f2):
    if _is_one(f2):
        return f1
    if _is_zero(f1):
        return Const1(0.0)
    if _is_zero(f2):
        raise ZeroDivisionError("newRatioFunction: Division by zero.")
    if f1.isIdentical(f2):
        return Const1(1.0)
    if _is_constant(f2):
        return newTimesConstFunction(f1, 1.0 / f2.c())
    if f1.type() == "pow" and f2.type() == "pow":
        return Pow1(f1.c() - f2.c())
    if f1.type() == "exp" and f2.type() == "exp":
        return Exp1(f1.c() - f2.c())
    return Ratio1(f1, f2)


def newCompositeFunction(f1, f2):
    if _is_zero(f1):
        return Const1(0.0)
    if _is_constant(f1):
        return f1
    if f1.type() == "pow" and f1.c() == 1.0:
        return f2
    if f1.type() == "pow" and f1.c() == 0.0:
        return Const1(1.0)
    if f1.type() == "pow" and f2.type() == "pow":
        return Pow1(f1.c() * f2.c())
    return Composite1(f1, f2)


def newTimesConstFunction(f, c):
    if c == 0.0:
        return Const1(0.0)
    if c == 1.0:
        return f
    if f.type() == "times-constant":
        return TimesConstant1(f.func1_shared(), f.c() * c)
    return TimesConstant1(f, c)


def newPlusConstFunction(f, c):
    if c == 0.0:
        return f
    if _is_constant(f):
        return Const1(f.c() + c)
    if f.type() == "plus-constant":
        return PlusConstant1(f.func1_shared(), f.c() + c)
    return PlusConstant1(f, c)


def linearInterp(x, xpts, fpts):
    """Linear interpolation on an ordered grid, clamped to nearest endpoint."""
    if len(xpts) == 0 or len(fpts) == 0:
        raise ValueError("x and f(x) data must not be empty.")
    if len(xpts) != len(fpts):
        raise ValueError("Grid and value lengths need to be the same.")
    if x <= xpts[0]:
        return fpts[0]
    if x >= xpts[-1]:
        return fpts[-1]
    i = bisect_left(xpts, x) - 1
    return fpts[i] + (x - xpts[i]) * (fpts[i + 1] - fpts[i]) / (xpts[i + 1] - xpts[i])


def _quadrature_inputs(f, x, minimum):
    f, x = tuple(float(v) for v in f), tuple(float(v) for v in x)
    if len(f) != len(x):
        raise ValueError("Vector lengths need to be the same.")
    if len(f) < minimum:
        raise ValueError(f"At least {minimum} sample points are required.")
    h = tuple(x[i + 1] - x[i] for i in range(len(x) - 1))
    if any(v <= 0.0 for v in h):
        raise ValueError("Coordinates must be monotonically increasing.")
    return f, x, h


def trapezoidal(f, x):
    """sum((f[i+1]+f[i])*(x[i+1]-x[i]))/2 on an increasing grid."""
    f, x, h = _quadrature_inputs(f, x, 1)
    return sum((f[i + 1] + f[i]) * h[i] for i in range(len(h))) / 2.0


def basicSimpson(f, x):
    """Original unequal-spacing Simpson formula for an odd number of points."""
    f, x, h = _quadrature_inputs(f, x, 3)
    if len(f) % 2 == 0:
        raise ValueError("Vector lengths need to be an odd number.")
    result = 0.0
    for i in range(1, len(f) - 1, 2):
        h0, h1 = h[i - 1], h[i]
        hph, hdh, hmh = h1 + h0, h1 / h0, h1 * h0
        result += (hph / 6.0) * (
            (2.0 - hdh) * f[i - 1]
            + (math.pow(hph, 2) / hmh) * f[i]
            + (2.0 - 1.0 / hdh) * f[i + 1]
        )
    return result


def simpson(f, x):
    """Unequal-grid Simpson; for even sample count use a final trapezoid."""
    f, x, h = _quadrature_inputs(f, x, 2)
    if len(f) % 2 == 1:
        return basicSimpson(f, x)
    if len(f) == 2:
        return 0.5 * h[0] * (f[1] + f[0])
    return basicSimpson(f[:-1], x[:-1]) + 0.5 * h[-1] * (f[-1] + f[-2])


def numericalQuadrature(method, f, x):
    """Dispatch to the original 'simpson' or 'trapezoidal' sampled-data rule."""
    if method == "simpson":
        return simpson(f, x)
    if method == "trapezoidal":
        return trapezoidal(f, x)
    raise ValueError("Unknown quadrature method; use 'simpson' or 'trapezoidal'.")


class Func1Factory:
    _instance = None

    @classmethod
    def factory(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def deleteFactory(cls):
        cls._instance = None


class Math1FactoryA(Func1Factory):
    pass


class Math1FactoryB(Func1Factory):
    pass


def newFunc1(func1Type: str, *args, **kwargs) -> Func1:
    type_lower = func1Type.lower()
    if type_lower == "sin":
        return Sin1(*args, **kwargs)
    elif type_lower == "cos":
        return Cos1(*args, **kwargs)
    elif type_lower == "exp":
        return Exp1(*args, **kwargs)
    elif type_lower == "log":
        return Log1(*args, **kwargs)
    elif type_lower == "pow":
        return Pow1(*args, **kwargs)
    elif type_lower == "constant":
        return Const1(*args, **kwargs)
    return Func1()


def checkFunc1(func1Type: str) -> str:
    return str(func1Type)
