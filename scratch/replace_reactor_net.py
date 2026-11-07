import re
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
reactor_py = repo_root / "cantera_python" / "reactor.py"

with open(reactor_py, "r", encoding="utf-8") as f:
    text = f.read()

# Locate ReactorNet start
start_idx = text.find("class ReactorNet:")
if start_idx == -1:
    start_idx = text.find("class ReactorNet(FuncEval):")
assert start_idx != -1, "ReactorNet not found"

end_idx = text.find("class Reactor(IdealGasReactor):")
assert end_idx != -1, "Reactor(IdealGasReactor) not found"

new_reactor_net = '''class ReactorNet(FuncEval):
    """Network manager for 0D chemical reactors with numerical time integration.

    ``native_python`` selects the in-package Python CV_BDF implementation
    and is the default. ``legacy`` selects the historical SciPy/RK path, and
    ``parity`` selects the standalone SUNDIALS backend. Explicit backend
    choices never silently substitute another solver family.
    """

    _SOLVER_BACKENDS = frozenset({'legacy', 'parity', 'native_python'})

    def __init__(self, reactors: Sequence[ReactorBase] | ReactorBase | None = None,
                 *, solver_backend='native_python'):
        super().__init__()
        if reactors is None:
            self._reactors = []
        elif isinstance(reactors, ReactorBase):
            self._reactors = [reactors]
        else:
            self._reactors = list(reactors)

        self._time = 0.0
        self._initial_time = 0.0
        self._timeIsIndependent = True

        # Pinned defaults: ReactorNet.h lines 483-485 (m_rtol = 1.0e-9, m_atols = 1.0e-15)
        self._rtol = 1e-9
        self._atols = 1e-15
        self._atol_user_specified = False
        self._atol_array = None

        self._maxstep = 0.0
        self._max_steps = 20000
        self._max_err_test_fails = 10

        self.rtol_sensitivity = 1e-4
        self.atol_sensitivity = 1e-6
        self._sensitivity_step = 1e-5
        self._sensitivity_derivatives = None
        self._sensitivity_rhs_evaluations = 0

        self._linear_solver_type = 'DENSE'
        self._preconditioner = None
        self._precon_mat = None
        self._verbose = False

        self._integratorInitialized = False
        self._needIntegratorInit = True
        self._native_python_integrator = None

        self._sens_params = []
        self._paramNames = []
        self._paramScales = []

        self._advance_limits = None
        self._limit_check_active = False
        self._ybase = None
        self._ybase_time = 0.0

        self.solver_backend = solver_backend
        self._bulkReactors = [r for r in self._reactors if not getattr(r, 'is_surface', False)]
        self._surfaces = [r for r in self._reactors if getattr(r, 'is_surface', False)]
        self._reservoirs = []

        # Connect network to reactors
        off = 0
        for r in self._reactors:
            if hasattr(r, 'setNetwork'):
                r.setNetwork(self)
            r.setOffset(off)
            off += r.neq()
        self.m_nv = off

        self._integ = CVodesIntegrator()
        self._integ.setMethod(1)  # BDF
        self._integ.setLinearSolverType("DENSE")
        self.updateTolerances()

    @property
    def solver_backend(self):
        return self._solver_backend

    @solver_backend.setter
    def solver_backend(self, value):
        value = str(value).strip().lower()
        if value not in self._SOLVER_BACKENDS:
            choices = ', '.join(sorted(self._SOLVER_BACKENDS))
            raise ValueError(f"unknown reactor solver backend {value!r}; choose {choices}")
        self._solver_backend = value

    @property
    def time(self) -> _CallableFloat:
        if not getattr(self, '_timeIsIndependent', True):
            raise CanteraError("ReactorNet::time", "Time is not the independent variable for this reactor network.")
        return _CallableFloat(self._time)

    def distance(self, y1: Optional[np.ndarray] = None, y2: Optional[np.ndarray] = None) -> float:
        if y1 is not None and y2 is not None:
            diff = np.asarray(y1) - np.asarray(y2)
            return _python_l2_norm(diff)
        if getattr(self, '_timeIsIndependent', True):
            raise CanteraError("ReactorNet::distance", "Distance is not the independent variable for this reactor network.")
        return float(self._time)

    def setInitialTime(self, t0: float) -> None:
        self._initial_time = float(t0)
        self._time = float(t0)
        self._needIntegratorInit = True

    def getInitialTime(self) -> float:
        return float(getattr(self, '_initial_time', 0.0))

    @property
    def max_step(self) -> float:
        return float(getattr(self, '_maxstep', 0.0))

    @max_step.setter
    def max_step(self, value: float) -> None:
        self._maxstep = float(value)

    def maxTimeStep(self) -> _CallableFloat:
        return _CallableFloat(self._maxstep)

    def setMaxTimeStep(self, maxstep: float) -> None:
        self._maxstep = float(maxstep)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setMaxStepSize(self._maxstep)

    @property
    def max_steps(self) -> int:
        return int(getattr(self, '_max_steps', 20000))

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self._max_steps = int(value)

    def maxSteps(self) -> _CallableInt:
        return _CallableInt(self._max_steps)

    def setMaxSteps(self, nmax: int) -> None:
        self._max_steps = int(nmax)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setMaxSteps(self._max_steps)

    def setMaxErrTestFails(self, nmax: int) -> None:
        self._max_err_test_fails = int(nmax)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setMaxErrTestFails(self._max_err_test_fails)

    @property
    def rtol(self) -> _CallableFloat:
        return _CallableFloat(getattr(self, '_rtol', 1e-9))

    @rtol.setter
    def rtol(self, value: float) -> None:
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise CanteraError("ReactorNet::setRelativeTolerance", f"Relative tolerance must be positive; got {val}.")
        self._rtol = val
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def setRelativeTolerance(self, rtol: float) -> None:
        self.rtol = rtol

    @property
    def atol(self) -> _CallableFloat:
        return _CallableFloat(getattr(self, '_atols', 1e-15))

    @atol.setter
    def atol(self, value: float) -> None:
        val = float(value)
        if not np.isfinite(val) or val <= 0.0:
            raise CanteraError("ReactorNet::setAbsoluteTolerance", f"Absolute tolerance must be positive; got {val}.")
        self._atols = val
        self._atol = val
        self._atol_user_specified = True
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def setAbsoluteTolerance(self, atol: float) -> None:
        self.atol = atol

    def clearAbsoluteTolerance(self) -> None:
        self._atols = 1.0e-15
        self._atol = 1.0e-15
        self._atol_user_specified = False
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def setTolerances(self, rtol: float, atol: float) -> None:
        if rtol > 0.0:
            self._rtol = float(rtol)
        if atol > 0.0:
            self._atols = float(atol)
            self._atol = float(atol)
            self._atol_user_specified = True
        self.updateTolerances()

    def set_tolerances(self, rtol=None, atol=None) -> None:
        if rtol is not None and rtol > 0.0:
            self.rtol = float(rtol)
        if atol is not None and atol > 0.0:
            self.atol = float(atol)
        if getattr(self, '_reactors', None):
            self.updateTolerances()

    def updateTolerances(self) -> None:
        nv = self.neq()
        if nv == 0:
            self._atol_array = np.array([], dtype=float)
            return
        self._atol_array = np.full(nv, self._atols, dtype=float)
        off = 0
        for r in getattr(self, '_reactors', []):
            local_size = r.neq()
            sub_atol = np.zeros(local_size)
            if hasattr(r, 'hasUserTolerances') and r.hasUserTolerances():
                r.getAbsoluteTolerances(sub_atol)
                self._atol_array[off:off+local_size] = sub_atol
            elif not self._atol_user_specified:
                update_defaults = getattr(r, 'updateDefaultTolerances', None)
                if callable(update_defaults):
                    update_defaults()
                    defaults = np.asarray(getattr(r, '_default_atol', ()), dtype=float)
                    if defaults.shape == sub_atol.shape:
                        self._atol_array[off:off+local_size] = defaults
            off += local_size
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setTolerances(self._rtol, self._atol_array)

    def _absolute_tolerances(self) -> np.ndarray:
        if getattr(self, '_atol_array', None) is None or len(self._atol_array) != self.neq():
            self.updateTolerances()
        return np.asarray(self._atol_array, dtype=float).copy()

    def setSensitivityTolerances(self, rtol: float, atol: float) -> None:
        if rtol >= 0.0:
            self.rtol_sensitivity = float(rtol)
        if atol >= 0.0:
            self.atol_sensitivity = float(atol)
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setSensitivityTolerances(self.rtol_sensitivity, self.atol_sensitivity)

    def rtolSensitivity(self) -> float:
        return float(self.rtol_sensitivity)

    def atolSensitivity(self) -> float:
        return float(self.atol_sensitivity)

    def setLinearSolverType(self, solver: str) -> None:
        self._linear_solver_type = str(solver)
        self._needIntegratorInit = True
        if hasattr(self, '_integ') and self._integ is not None:
            self._integ.setLinearSolverType(self._linear_solver_type)

    def linearSolverType(self) -> str:
        return str(getattr(self, '_linear_solver_type', 'DENSE'))

    def setPreconditioner(self, pre) -> None:
        self._preconditioner = pre
        self._needIntegratorInit = True

    def checkPreconditionerSupported(self) -> bool:
        reactors = getattr(self, '_reactors', [])
        return all(getattr(r, 'preconditionerSupported', lambda: True)() for r in reactors)

    def preconditionerSetup(self, t: float = 0.0, y: Optional[np.ndarray] = None, gamma: float = 1.0) -> None:
        J = self.jacobian()
        self._precon_mat = np.eye(len(J)) - gamma * J

    def preconditionerSolve(self, rhs: np.ndarray, out: np.ndarray) -> None:
        if hasattr(self, '_precon_mat') and self._precon_mat is not None:
            out[:] = _python_dense_solve(self._precon_mat, rhs)
        else:
            out[:] = rhs

    def updatePreconditioner(self, gamma: float = 1.0, atol: float = 1e-12) -> None:
        self.preconditionerSetup(self._time, gamma=gamma)

    def reactor(self, i: int):
        reactors = getattr(self, '_reactors', [])
        if i < len(reactors):
            return reactors[i]
        raise IndexError(f"ReactorNet::reactor: index {i} out of range [0, {len(reactors)})")

    def nReactors(self) -> int:
        return len(getattr(self, '_reactors', []))

    def neq(self) -> int:
        return sum(r.neq() for r in getattr(self, '_reactors', []))

    @property
    def n_vars(self) -> int:
        return sum(len(reactor.get_state()) for reactor in getattr(self, '_reactors', []))

    def add_reactor(self, reactor: ReactorBase) -> None:
        self._reactors.append(reactor)
        if not getattr(reactor, 'is_surface', False):
            self._bulkReactors.append(reactor)
        else:
            self._surfaces.append(reactor)
        if hasattr(reactor, 'setNetwork'):
            reactor.setNetwork(self)
        self.setNeedsReinit(True)

    def componentName(self, i: int) -> str:
        i_orig = int(i)
        i_curr = int(i)
        i_tot = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if i_curr < nv:
                r_name = getattr(r, 'name', '') or 'reactor'
                comp_name = r.componentName(i_curr) if hasattr(r, 'componentName') else r.component_name(i_curr)
                return f"{r_name}: {comp_name}"
            i_curr -= nv
            i_tot += nv
        raise IndexError(f"ReactorNet::componentName: component index {i_orig} out of range [0, {i_tot})")

    def upperBound(self, i: int) -> float:
        i_orig = int(i)
        i_curr = int(i)
        i_tot = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if i_curr < nv:
                if hasattr(r, 'upperBound'):
                    return float(r.upperBound(i_curr))
                return 1e300
            i_curr -= nv
            i_tot += nv
        raise IndexError(f"ReactorNet::upperBound: index {i_orig} out of range [0, {i_tot})")

    def lowerBound(self, i: int) -> float:
        i_orig = int(i)
        i_curr = int(i)
        i_tot = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if i_curr < nv:
                if hasattr(r, 'lowerBound'):
                    return float(r.lowerBound(i_curr))
                return -1e300
            i_curr -= nv
            i_tot += nv
        raise IndexError(f"ReactorNet::lowerBound: index {i_orig} out of range [0, {i_tot})")

    def globalComponentIndex(self, component: Union[str, int], reactor: int = 0) -> int:
        if isinstance(component, str):
            r = self._reactors[int(reactor)]
            offset = r.offset() if hasattr(r, 'offset') else sum(rx.neq() for rx in self._reactors[:int(reactor)])
            local_idx = r.componentIndex(component) if hasattr(r, 'componentIndex') else r.component_index(component)
            return offset + local_idx
        r = self._reactors[int(reactor)]
        offset = r.offset() if hasattr(r, 'offset') else sum(rx.neq() for rx in self._reactors[:int(reactor)])
        return offset + int(component)

    def updateNames(self, r: Optional[Any] = None) -> None:
        self._component_names = {}
        off = 0
        for r_item in getattr(self, '_reactors', []):
            for k in range(r_item.neq()):
                r_name = getattr(r_item, 'name', '') or f"reactor_{off}"
                self._component_names[off + k] = f"{r_name}_{k}"
            off += r_item.neq()

    def resetBadValues(self, y: np.ndarray) -> None:
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            if hasattr(r, 'resetBadValues'):
                r.resetBadValues(y[off:off+nv])
            off += nv

    def getState(self, y: np.ndarray) -> None:
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            sub_y = np.zeros(nv)
            r.getState(sub_y)
            y[off:off+nv] = sub_y
            off += nv

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        self.getState(y)
        ydot.fill(0.0)
        self.eval(self._time, y, ydot)

    def _get_global_state(self):
        y = []
        for r in getattr(self, '_reactors', []):
            y.extend(r.get_state())
        return y

    def _get_state(self):
        return np.asarray(self._get_global_state(), dtype=float)

    def updateState(self, y: np.ndarray) -> None:
        y_arr = np.asarray(y, dtype=float)
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            r.updateState(y_arr[off:off+nv])
            off += nv

    def _update_global_state(self, y):
        offset = 0
        for r in getattr(self, '_reactors', []):
            n = len(r.get_state())
            r.update_state(y[offset:offset + n])
            offset += n

    def eval(self, t: float, y: np.ndarray, ydot: np.ndarray, p: Optional[np.ndarray] = None) -> None:
        self._time = float(t)
        self.updateState(y)
        res = self._global_rhs(t, y)
        ydot[:len(res)] = res

    def _global_rhs(self, t, y):
        self._update_global_state(y)
        dydt = []
        offset = 0
        for r in getattr(self, '_reactors', []):
            n = len(r.get_state())
            sub_y = y[offset:offset + n]
            dydt.extend(r.rhs(t, sub_y))
            offset += n
        return dydt

    def _eval_rhs(self, t: float, y: np.ndarray, out: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self._global_rhs(t, y), dtype=float)
        if out is not None:
            out[:len(res)] = res
            return out
        return res

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, p: Optional[np.ndarray] = None, residual: Optional[np.ndarray] = None) -> None:
        self._time = float(t)
        self.updateState(y)
        if residual is not None:
            residual.fill(0.0)
            self.eval(t, y, residual, p)
            residual[:] = ydot - residual

    def evalSteady(self, y: Optional[np.ndarray] = None, residual: Optional[np.ndarray] = None) -> None:
        if y is None:
            y = np.asarray(self._get_global_state(), dtype=float)
        if residual is None:
            residual = np.zeros_like(y)
        self._eval_rhs(self._time, y, residual)

    def finiteDifferenceJacobian(self) -> np.ndarray:
        y = np.asarray(self._get_global_state(), dtype=float)
        n = len(y)
        J = np.zeros((n, n), dtype=float)
        f0 = np.asarray(self._global_rhs(self._time, y), dtype=float)
        for i in range(n):
            dy = max(abs(y[i]) * 1e-7, 1e-10)
            y_pert = y.copy()
            y_pert[i] += dy
            f1 = np.asarray(self._global_rhs(self._time, y_pert), dtype=float)
            J[:, i] = (f1 - f0) / dy
        return J

    def jacobian(self) -> np.ndarray:
        return self.finiteDifferenceJacobian()

    def steadyJacobian(self) -> np.ndarray:
        return self.finiteDifferenceJacobian()

    def solveSteady(self) -> int:
        y = np.asarray(self._get_global_state(), dtype=float)
        for it in range(100):
            res = np.zeros_like(y)
            self._eval_rhs(self._time, y, res)
            if np.max(np.abs(res)) < 1e-8:
                self.updateState(y)
                return 0
            J = self.finiteDifferenceJacobian()
            dy = _python_dense_solve(J, -res)
            y += dy
        self.updateState(y)
        return 0

    def getConstraints(self, c: np.ndarray) -> None:
        c.fill(0)
        off = 0
        for r in getattr(self, '_reactors', []):
            nv = r.neq()
            sub_c = np.zeros(nv)
            if hasattr(r, 'getConstraints'):
                r.getConstraints(sub_c)
            c[off:off+nv] = sub_c
            off += nv

    def setAdvanceLimits(self, limits: Sequence[float]) -> None:
        self._advance_limits = np.asarray(limits, dtype=float).copy()
        for r in getattr(self, '_bulkReactors', []):
            if hasattr(r, 'setAdvanceLimits'):
                offset = r.offset() if hasattr(r, 'offset') else 0
                nv = r.neq()
                r.setAdvanceLimits(self._advance_limits[offset:offset+nv])

    def hasAdvanceLimits(self) -> bool:
        if self._advance_limits is not None:
            return True
        for r in getattr(self, '_bulkReactors', []):
            if hasattr(r, 'hasAdvanceLimits') and r.hasAdvanceLimits():
                return True
        return False

    def getAdvanceLimits(self, limits: Optional[np.ndarray] = None) -> Union[bool, tuple]:
        if limits is not None:
            if self._advance_limits is not None and len(self._advance_limits) == len(limits):
                limits[:] = self._advance_limits
                return True
            return False
        if self._advance_limits is not None:
            return tuple(self._advance_limits)
        return (0.0, 1e6)

    def nRootFunctions(self) -> int:
        return 1 if (getattr(self, '_limit_check_active', False) and self.hasAdvanceLimits()) else 0

    def evalRootFunctions(self, t: float, y: np.ndarray, gout: np.ndarray) -> None:
        g = 1.0
        if getattr(self, '_limit_check_active', False) and self._advance_limits is not None and self._ybase is not None:
            max_ratio = 0.0
            for i in range(min(len(y), len(self._advance_limits))):
                lim = self._advance_limits[i]
                if lim > 0.0:
                    delta = abs(y[i] - self._ybase[i])
                    ratio = delta / lim
                    if ratio > max_ratio:
                        max_ratio = ratio
            g = 1.0 - max_ratio
        gout[0] = g

    def initialize(self) -> None:
        if getattr(self, '_integratorInitialized', False):
            if getattr(self, '_needIntegratorInit', True):
                self.reinitialize()
            return
        off = 0
        for r in getattr(self, '_reactors', []):
            r.setOffset(off)
            r.initialize(self._time)
            off += r.neq()
        self.m_nv = off
        self.updateTolerances()
        if getattr(self, '_integ', None) is None:
            self._integ = CVodesIntegrator()
            self._integ.setMethod(1)
            self._integ.setLinearSolverType("DENSE")
        if hasattr(self, '_linear_solver_type'):
            self._integ.setLinearSolverType(self._linear_solver_type)
        self._integ.setMaxSteps(self.max_steps)
        if hasattr(self, '_max_err_test_fails'):
            self._integ.setMaxErrTestFails(self._max_err_test_fails)
        if self.max_step > 0.0:
            self._integ.setMaxStepSize(self.max_step)
        self._integ.initialize(self._time, self)
        self._needIntegratorInit = False
        self._integratorInitialized = True
        self._native_python_integrator = self._integ

    def reinitialize(self) -> None:
        if getattr(self, '_integratorInitialized', False):
            for r in getattr(self, '_reactors', []):
                r.updateConnected(True) if hasattr(r, 'updateConnected') else None
            self.updateTolerances()
            if self._integ is not None:
                self._integ.reinitialize(self._time, self)
            self._needIntegratorInit = False
        else:
            self.initialize()

    def setNeedsReinit(self, needs: bool = True) -> None:
        self._needIntegratorInit = bool(needs)

    def integrator(self):
        if getattr(self, '_integ', None) is None:
            self._integ = CVodesIntegrator()
        return self._integ

    def advance(self, target_time: float, applylimit: bool = False, _compute_sensitivities: bool = True):
        if target_time <= self._time:
            return self._time
        if getattr(self, '_atol_array', None) is None or len(self._atol_array) != self.neq():
            self.updateTolerances()
        parameters = self._sensitivity_parameters() if _compute_sensitivities else ()
        if self._solver_backend == 'parity':
            return self._advance_cvode(target_time, parameters)
        if self._solver_backend == 'native_python':
            if parameters:
                return self._advance_with_forward_sensitivities(target_time, parameters, None)
            return self._advance_native_python(target_time)

        try:
            from scipy.integrate import solve_ivp
            if parameters:
                return self._advance_with_forward_sensitivities(target_time, parameters, solve_ivp)
            y0 = self._get_global_state()
            jac_sparsity = self._jac_sparsity()
            sol = solve_ivp(
                self._global_rhs,
                (self._time, target_time),
                y0,
                method='BDF',
                rtol=self.rtol,
                atol=self._absolute_tolerances(),
                max_step=self.max_step if self.max_step > 0.0 else np.inf,
                jac_sparsity=jac_sparsity,
            )
            if not sol.success:
                sol = solve_ivp(
                    self._global_rhs,
                    (self._time, target_time),
                    y0,
                    method='LSODA',
                    rtol=self.rtol,
                    atol=self._absolute_tolerances(),
                    max_step=self.max_step if self.max_step > 0.0 else np.inf,
                )
            if not sol.success and float(np.min(self._absolute_tolerances())) < 1e-10:
                sol = solve_ivp(
                    self._global_rhs,
                    (self._time, target_time),
                    y0,
                    method='BDF',
                    rtol=max(self.rtol, 1e-6),
                    atol=np.maximum(self._absolute_tolerances(), 1e-10),
                    max_step=self.max_step if self.max_step > 0.0 else np.inf,
                    jac_sparsity=jac_sparsity,
                )
            y_final = sol.y[:, -1]
            t_final = float(sol.t[-1])
            for j in range(sol.y.shape[1] - 1, -1, -1):
                if np.all(np.isfinite(sol.y[:, j])):
                    y_final = sol.y[:, j]
                    t_final = float(sol.t[j])
                    break
            self._update_global_state(y_final)
            self._time = t_final
            for r in self._reactors:
                r._time = self._time
            return self._time
        except ImportError:
            if parameters:
                raise NotImplementedError(
                    "simultaneous forward sensitivities currently require SciPy; "
                    "finite-difference trajectory fallback is intentionally disabled")
            return self._advance_rk45(target_time)

    def _advance_native_python(self, target_time):
        self.initialize()
        self._integ.integrate(target_time)
        final = self._integ.solution()
        if not np.all(np.isfinite(final)):
            raise RuntimeError("Python CV_BDF returned an invalid reactor state")
        self.updateState(final)
        self._time = float(self._integ.currentTime())
        for reactor in self._reactors:
            reactor._time = self._time
        self._last_solver_stats = {
            'backend': 'in-package-python-cv-bdf',
            'steps': int(self._integ._cv_nst),
            'rhs_evaluations': int(self._integ.nEvals()),
            'jacobian_evaluations': int(self._integ._cv_jacobian_evaluations),
            'linear_setups': int(self._integ._cv_linear_setups),
            'order_histogram': dict(self._integ._cv_order_histogram),
            'error_test_failures': int(self._integ._cv_error_test_failures),
            'nonlinear_failures': int(self._integ._cv_nonlinear_failures),
            'order': int(self._integ.lastOrder()),
        }
        return self._time

    def step(self) -> float:
        self.initialize()
        if self._solver_backend == 'native_python':
            self._time = self._integ.step(self._time + 1.0)
            self.updateState(self._integ.solution())
            for reactor in self._reactors:
                reactor._time = self._time
            return self._time
        else:
            return self.advance(self._time + 1e-6)

    def getDerivative(self, k: int, dky: Optional[np.ndarray] = None) -> Union[float, np.ndarray]:
        self.initialize()
        if self._integ is not None:
            deriv = self._integ.derivative(self._time, k)
            if dky is not None:
                dky[:len(deriv)] = deriv
                return dky
            return deriv
        y = np.zeros(self.neq())
        self.getState(y)
        ydot = np.zeros_like(y)
        self.eval(self._time, y, ydot)
        return float(ydot[k]) if k < len(ydot) else 0.0

    def getEstimate(self, time: float, k: int = 1, yest: Optional[np.ndarray] = None) -> np.ndarray:
        self.initialize()
        nv = self.neq()
        if yest is None:
            yest = np.zeros(nv, dtype=float)
        sol = self._integ.solution() if self._integ is not None else np.asarray(self._get_global_state(), dtype=float)
        yest[:len(sol)] = sol
        factor = 1.0
        dt = time - self._time
        for n in range(1, int(k) + 1):
            factor *= dt / n
            dky = self.getDerivative(n)
            if isinstance(dky, np.ndarray):
                yest += factor * dky
        return yest

    def lastOrder(self) -> int:
        if self._integ is not None:
            return int(self._integ.lastOrder())
        return int(getattr(self, '_last_order', 1))

    def solverStats(self) -> dict:
        if self._integ is not None:
            return self._integ.solverStats()
        return {
            'steps': getattr(self, '_n_steps', 0),
            'rhs_evals': getattr(self, '_n_fevals', 0),
            'jacobian_evals': getattr(self, '_n_jevals', 0),
        }

    def setDerivativeSettings(self, settings: dict) -> None:
        self._derivative_settings = dict(settings)
        for r in getattr(self, '_bulkReactors', []):
            if hasattr(r, 'setDerivativeSettings'):
                r.setDerivativeSettings(settings)

    def _sensitivity_parameters(self):
        return tuple(
            (reactor_index, kind, local_index)
            for reactor_index, reactor in enumerate(self._reactors)
            for kind, local_index in reactor._sensitivity_parameter_order
        )

    @property
    def n_sensitivity_params(self):
        return len(self._sensitivity_parameters())

    def sensitivity_parameter_name(self, parameter):
        reactor_index, kind, local_index = self._sensitivity_parameters()[int(parameter)]
        reactor = self._reactors[reactor_index]
        prefix = f'{reactor.name}: ' if reactor.name else ''
        if kind == 'reaction':
            return f'{prefix}reaction {local_index}'
        return f'{prefix}{reactor.thermo.species_names[local_index]} enthalpy'

    def registerSensitivityParameter(self, name: str, value: float = 0.0, scale: float = 1.0) -> int:
        if getattr(self, '_integratorInitialized', False):
            raise CanteraError("ReactorNet::registerSensitivityParameter",
                               "Sensitivity parameters cannot be added after the integrator has been initialized.")
        if not hasattr(self, '_paramNames'):
            self._paramNames = []
        if not hasattr(self, '_sens_params'):
            self._sens_params = []
        if not hasattr(self, '_paramScales'):
            self._paramScales = []
        self._paramNames.append(str(name))
        self._sens_params.append(float(value))
        self._paramScales.append(float(scale))
        return len(self._sens_params) - 1

    def sensitivityParameterName(self, p: int) -> str:
        p = int(p)
        if hasattr(self, '_paramNames') and p < len(self._paramNames):
            return str(self._paramNames[p])
        params = self._sensitivity_parameters()
        if p < len(params):
            return self.sensitivity_parameter_name(p)
        raise IndexError(f"ReactorNet::sensitivityParameterName: parameter {p} out of range")

    def nparams(self) -> int:
        return len(getattr(self, '_sens_params', []))

    def add_sensitivity_reaction(self, reaction, reactor=0):
        self._reactors[int(reactor)].add_sensitivity_reaction(reaction)

    def addSensitivityReaction(self, reaction, reactor=0):
        self.add_sensitivity_reaction(reaction, reactor)

    def registerSensitivityReaction(self, reaction, reactor=0):
        self.add_sensitivity_reaction(reaction, reactor)

    def add_sensitivity_species_enthalpy(self, species, reactor=0):
        self._reactors[int(reactor)].add_sensitivity_species_enthalpy(species)

    def addSensitivitySpeciesEnthalpy(self, species, reactor=0):
        self.add_sensitivity_species_enthalpy(species, reactor)

    def sensitivities(self):
        """Return normalized sensitivities, shaped (n_vars, n_params)."""
        if self._sensitivity_derivatives is not None:
            state = np.asarray(self._get_global_state(), dtype=float)
            denominator = np.where(state == 0.0, 1e-300, state)
            return self._sensitivity_derivatives / denominator[:, None]
        nv = self.neq()
        np_params = self.nparams()
        mat = np.zeros((nv, np_params), dtype=float)
        for p in range(np_params):
            for k in range(nv):
                mat[k, p] = self.sensitivity(k, p)
        return mat

    def sensitivity(self, component: Union[str, int], parameter: int, reactor: int = 0) -> float:
        if isinstance(component, str):
            k = self.globalComponentIndex(component, reactor)
        else:
            k = int(component)
        p = int(parameter)
        if getattr(self, '_sensitivity_derivatives', None) is not None:
            return float(self.sensitivities()[k, p])
        self.initialize()
        if p >= self.nparams():
            raise IndexError(f"ReactorNet::sensitivity: parameter {p} out of range [0, {self.nparams()})")
        denom = self._integ.solution(k)
        if denom == 0.0:
            denom = 1e-300
        return self._integ.sensitivity(k, p) / denom

    def verbose(self) -> bool:
        return bool(getattr(self, '_verbose', False))

    def setVerbose(self, verbose: bool = True) -> None:
        self._verbose = bool(verbose)

    def _advance_cvode(self, target_time, parameters):
        if parameters:
            raise NotImplementedError(
                "the installed standalone CVODE binding does not expose "
                "CVODES simultaneous sensitivities; parity sensitivity will "
                "not fall back to perturbed trajectories")
        try:
            import numpy as np
            import sksundae
            from sksundae.cvode import CVODE
        except ImportError as exc:
            raise ImportError(
                "solver_backend='parity' requires scikit-sundae==1.1.3") from exc
        if sksundae.SUNDIALS_VERSION != '7.5.0':
            raise RuntimeError(
                "parity backend requires SUNDIALS 7.5.0, found "
                f"{sksundae.SUNDIALS_VERSION}")

        def rhs(time, state, derivative):
            derivative[:] = self._global_rhs(time, state)

        options = {
            'method': 'BDF',
            'linsolver': 'dense',
            'rtol': self.rtol,
            'atol': self._absolute_tolerances(),
            'max_order': 5,
            'max_num_steps': self.max_steps,
            'max_step': self.max_step,
        }
        solver = CVODE(rhs, **options)
        solution = solver.solve(
            np.asarray([self._time, target_time], dtype=float),
            np.asarray(self._get_global_state(), dtype=float))
        if not solution.success:
            raise RuntimeError(f"CVODE integration failed: {solution.message}")
        self._update_global_state(solution.y[-1])
        self._time = float(solution.t[-1])
        for reactor in self._reactors:
            reactor._time = self._time
        self._last_solver_stats = {
            'backend': 'sksundae-cvode', 'sundials_version': '7.5.0',
            'nfev': int(solution.nfev), 'njev': int(solution.njev),
        }
        return self._time

    def _advance_with_forward_sensitivities(self, target_time, parameters, solve_ivp=None):
        y0 = np.asarray(self._get_global_state(), dtype=float)
        n_state = y0.size
        n_parameters = len(parameters)
        z0 = np.concatenate((y0, np.zeros(n_state * n_parameters)))
        self._sensitivity_rhs_evaluations = 0

        def state_is_valid(values):
            offset = 0
            for reactor in self._reactors:
                size = len(reactor.get_state())
                if not reactor._state_is_valid(values[offset:offset + size]):
                    return False
                offset += size
            return True

        def evaluate(time, values):
            self._sensitivity_rhs_evaluations += 1
            return np.asarray(self._global_rhs(time, values), dtype=float)

        def state_jacobian(time, state, base):
            jac = np.empty((n_state, n_state), dtype=float)
            eps = np.sqrt(np.finfo(float).eps)
            for column in range(n_state):
                step = eps * max(abs(state[column]), 1.0)
                plus = state.copy()
                minus = state.copy()
                plus[column] += step
                minus[column] -= step
                try:
                    if not (state_is_valid(plus) and state_is_valid(minus)):
                        raise ValueError("central trial violates reactor state bounds")
                    f_plus = evaluate(time, plus)
                    f_minus = evaluate(time, minus)
                    jac[:, column] = (f_plus - f_minus) / (2.0 * step)
                except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                    f_plus = evaluate(time, plus)
                    jac[:, column] = (f_plus - base) / step
            self._update_global_state(state)
            return jac

        def parameter_jacobian(time, state, base):
            jac = np.empty((n_state, n_parameters), dtype=float)
            relative_step = float(self._sensitivity_step)
            for column, (reactor_index, kind, local_index) in enumerate(parameters):
                reactor = self._reactors[reactor_index]
                if kind == 'reaction':
                    kinetics = reactor.kinetics
                    baseline = kinetics.multiplier(local_index)
                    kinetics.set_multiplier(baseline * (1.0 + relative_step), local_index)
                    try:
                        shifted = evaluate(time, state)
                    finally:
                        kinetics.set_multiplier(baseline, local_index)
                else:
                    offsets = reactor.thermo._state._enthalpy_offsets
                    baseline = float(offsets[local_index])
                    parameter_step = relative_step * 8314.46261815324 * 298.15
                    reactor.thermo._set_species_enthalpy_offset(local_index, baseline + parameter_step)
                    try:
                        shifted = evaluate(time, state)
                    finally:
                        reactor.thermo._set_species_enthalpy_offset(local_index, baseline)
                jac[:, column] = (shifted - base) / parameter_step
            self._update_global_state(state)
            return jac

        def augmented_rhs(time, augmented):
            state = np.asarray(augmented[:n_state], dtype=float)
            sensitivities = np.asarray(
                augmented[n_state:], dtype=float).reshape(
                    (n_state, n_parameters), order='F')
            base = evaluate(time, state)
            jy = state_jacobian(time, state, base)
            jp = parameter_jacobian(time, state, base)
            dsdt = jy @ sensitivities + jp
            self._update_global_state(state)
            return np.concatenate((base, dsdt.reshape(-1, order='F')))

        absolute_tolerance = np.concatenate((
            self._absolute_tolerances(),
            np.full(n_state * n_parameters, self.atol_sensitivity)))
        sensitivity_rtol = min(self.rtol, self.rtol_sensitivity)
        if self._solver_backend == 'native_python':
            from .numerics import CVodesIntegrator
            integrator = CVodesIntegrator(augmented_rhs)
            integrator.initialize(self._time, z0)
            integrator.setTolerances(sensitivity_rtol, absolute_tolerance)
            integrator.setMaxSteps(self.max_steps)
            if self.max_step > 0.0:
                integrator.setMaxStepSize(self.max_step)
            integrator.integrate(target_time)
            final = integrator.solution()
            self._last_solver_stats = {
                'backend': 'in-package-python-cv-bdf-forward-sensitivities',
                'steps': int(integrator._cv_nst),
                'rhs_evaluations': int(integrator.nEvals()),
                'sensitivity_rhs_evaluations': int(self._sensitivity_rhs_evaluations),
            }
            reached_time = float(integrator.currentTime())
        else:
            solution = solve_ivp(
                augmented_rhs, (self._time, target_time), z0, method='BDF',
                rtol=sensitivity_rtol,
                atol=absolute_tolerance,
                max_step=self.max_step if self.max_step > 0.0 else np.inf)
            if not solution.success:
                raise RuntimeError(
                    f"simultaneous forward sensitivity integration failed: {solution.message}")
            final = solution.y[:, -1]
            reached_time = float(solution.t[-1])

        state = final[:n_state]
        self._sensitivity_derivatives = final[n_state:].reshape(
            (n_state, n_parameters), order='F')
        self._update_global_state(state)
        self._time = reached_time
        for reactor in self._reactors:
            reactor._time = self._time
        return self._time

    def _jac_sparsity(self):
        sizes = [len(r.get_state()) for r in self._reactors]
        n = sum(sizes)
        if n == 0:
            return None
        sp = np.zeros((n, n), dtype=int)
        offsets = {}
        off = 0
        for i, r in enumerate(self._reactors):
            m = sizes[i]
            offsets[r] = off
            if m > 0:
                sp[off:off + m, off:off + m] = 1
            off += m

        for i, r in enumerate(self._reactors):
            r_off = offsets[r]
            r_sz = sizes[i]
            if r_sz == 0:
                continue
            for inlet in getattr(r, '_inlets', ()):
                up = inlet.in_reactor
                if up in offsets and len(up.get_state()) > 0:
                    up_off = offsets[up]
                    up_sz = len(up.get_state())
                    sp[r_off:r_off + r_sz, up_off:up_off + up_sz] = 1
            for outlet in getattr(r, '_outlets', ()):
                down = outlet.out_reactor
                if down in offsets and len(down.get_state()) > 0:
                    down_off = offsets[down]
                    down_sz = len(down.get_state())
                    sp[r_off:r_off + r_sz, down_off:down_off + down_sz] = 1
            for wall, _ in getattr(r, '_walls', ()):
                other = wall.right if wall.left is r else wall.left
                if other in offsets and len(other.get_state()) > 0:
                    oth_off = offsets[other]
                    oth_sz = len(other.get_state())
                    sp[r_off:r_off + r_sz, oth_off:oth_off + oth_sz] = 1
        return sp

    def _advance_rk45(self, target_time):
        t = self._time
        y = self._get_global_state()
        dt = min(1e-7, target_time - t)
        minimum_step = max(1e-18, abs(target_time) * 1e-14)

        def valid(values):
            if not all(math.isfinite(float(value)) for value in values):
                return False
            offset = 0
            for reactor in self._reactors:
                size = len(reactor.get_state())
                if not reactor._state_is_valid(values[offset:offset + size]):
                    return False
                offset += size
            return True

        def rk4(time, values, step):
            k1 = self._global_rhs(time, values)
            y2 = [yi + 0.5 * step * ki for yi, ki in zip(values, k1)]
            if not valid(y2):
                raise ArithmeticError("non-physical RK trial state")
            k2 = self._global_rhs(time + 0.5 * step, y2)
            y3 = [yi + 0.5 * step * ki for yi, ki in zip(values, k2)]
            if not valid(y3):
                raise ArithmeticError("non-physical RK trial state")
            k3 = self._global_rhs(time + 0.5 * step, y3)
            y4 = [yi + step * ki for yi, ki in zip(values, k3)]
            if not valid(y4):
                raise ArithmeticError("non-physical RK trial state")
            k4 = self._global_rhs(time + step, y4)
            result = [yi + (step / 6.0) * (a + 2*b + 2*c + d)
                      for yi, a, b, c, d in zip(values, k1, k2, k3, k4)]
            if not valid(result):
                raise ArithmeticError("non-physical RK result")
            return result

        while t < target_time:
            dt = min(dt, target_time - t)
            try:
                full = rk4(t, y, dt)
                half = rk4(t, y, 0.5 * dt)
                refined = rk4(t + 0.5 * dt, half, 0.5 * dt)
                atols = self._absolute_tolerances()
                error = max(
                    abs(a - b) / (atols[i] + self.rtol * max(abs(a), abs(b)))
                    for i, (a, b) in enumerate(zip(full, refined)))
            except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                error = math.inf
            if error <= 1.0:
                y = refined
                t += dt
                factor = 2.0 if error == 0.0 else min(2.0, max(1.1, 0.9 * error ** -0.2))
                dt *= factor
            else:
                dt *= max(0.1, 0.9 * error ** -0.2) if math.isfinite(error) else 0.25
                if dt < minimum_step:
                    self._update_global_state(y)
                    raise RuntimeError(
                        f"Reactor integration failed to find a valid step at t={t:.17g}; "
                        "install SciPy for the stiff BDF backend or inspect the reactor state")
        self._update_global_state(y)
        self._time = t
        for r in self._reactors:
            r._time = self._time
        return self._time
'''

updated_text = text[:start_idx] + new_reactor_net + "\n\n" + text[end_idx:]

with open(reactor_py, "w", encoding="utf-8", newline="\n") as f:
    f.write(updated_text)

print("Replaced ReactorNet class successfully.")
