#!/usr/bin/env python3
"""Patch CVodesIntegrator in cantera_python/numerics.py with full augmented sensitivity support."""

import re

cvodes_code = '''class CVodesIntegrator(Integrator):
    """Stiff ODE integrator with native SUNDIALS CVODES-equivalent BDF kernel."""

    _CV_ETA_MAX_FIRST_STEP = 10000.0
    _CV_ETA_MAX_EARLY_STEPS = 10.0
    _CV_ETA_MAX_GENERAL = 10.0
    _CV_SMALL_NST = 10

    def __init__(self, func: Optional[Any] = None):
        super().__init__(func)
        self._cv_order = 1
        self._cv_next_h = 0.0
        self._cv_zn: List[np.ndarray] = []
        self._cv_tau = [0.0] * 7
        self._cv_q = 1
        self._cv_qprime = 1
        self._cv_L = 2
        self._cv_qwait = 2
        self._cv_nst = 0
        self._cv_h = 0.0
        self._cv_hscale = 0.0
        self._cv_hprime = 0.0
        self._cv_tn = 0.0
        self._cv_t0 = 0.0
        self._cv_saved_tq5 = 0.0
        self._cv_eta = 1.0
        self._cv_etamax = self._CV_ETA_MAX_FIRST_STEP
        self._cv_nscon = 0
        self._cv_max_order = 5
        self._cv_ewt: Optional[np.ndarray] = None
        self._cv_rhs_jacobian: Optional[np.ndarray] = None
        self._cv_linear_factor = None
        self._cv_last_jac_step = -1
        self._cv_last_lsetup_step = -1
        self._cv_lsetup_gamma = 0.0
        self._cv_force_lsetup = True
        self._cv_force_jacobian = True
        self._cv_jacobian_evaluations = 0
        self._cv_linear_setups = 0
        self._cv_order_histogram = {}
        self._cv_error_test_failures = 0
        self._cv_nonlinear_failures = 0
        self._cv_nonlinear_iters = 0
        self._method = 1  # 1 = BDF, 0 = Adams
        self._linear_solver = "DENSE"
        self._mupper = 0
        self._mlower = 0
        self._max_err_test_fails = 7
        self.min_step = 0.0
        self._base_neq = 0
        self._np = 0
        self._augmented = False
        self._sens_ok = False
        self.rtol_sens = 1e-5
        self.atol_sens = 1e-4
        self._sens_atol = None
        self._n_root_functions = 0

    def initialize(self, t0: float, func_or_y0: Optional[Any] = None, y0: Optional[np.ndarray] = None):
        self._time = float(t0)
        self._cv_t0 = float(t0)
        self._nevals = 0
        self._cv_order = 1
        self._cv_next_h = 0.0
        self._cv_zn = []
        self._cv_tau = [0.0] * 7
        self._cv_q = 1
        self._cv_qprime = 1
        self._cv_L = 2
        self._cv_qwait = 2
        self._cv_nst = 0
        self._cv_h = 0.0
        self._cv_hscale = 0.0
        self._cv_hprime = 0.0
        self._cv_tn = self._time
        self._cv_saved_tq5 = 0.0
        self._cv_eta = 1.0
        self._cv_etamax = self._CV_ETA_MAX_FIRST_STEP
        self._cv_nscon = 0
        self._cv_ewt = None
        self._cv_rhs_jacobian = None
        self._cv_linear_factor = None
        self._cv_last_jac_step = -1
        self._cv_last_lsetup_step = -1
        self._cv_lsetup_gamma = 0.0
        self._cv_force_lsetup = True
        self._cv_force_jacobian = True
        self._cv_jacobian_evaluations = 0
        self._cv_linear_setups = 0
        self._cv_order_histogram = {}
        self._cv_error_test_failures = 0
        self._cv_nonlinear_failures = 0
        self._cv_nonlinear_iters = 0
        self._sens_ok = False

        if y0 is not None:
            self._func = func_or_y0
            base_y = np.asarray(y0, dtype=float).copy()
        elif func_or_y0 is not None:
            if isinstance(func_or_y0, (np.ndarray, list, tuple)):
                base_y = np.asarray(func_or_y0, dtype=float).copy()
            else:
                self._func = func_or_y0
                neq = 0
                if hasattr(self._func, "neq"):
                    try:
                        neq = self._func.neq()
                    except (NotImplementedError, TypeError):
                        pass
                if neq == 0 and hasattr(self._func, "nEquations"):
                    try:
                        neq = self._func.nEquations()
                    except (NotImplementedError, TypeError):
                        pass
                if hasattr(self._func, "getState") and neq > 0:
                    y_init = np.zeros(neq, dtype=float)
                    self._func.getState(y_init)
                    base_y = y_init
                elif hasattr(self._func, "getInitialConditions") and neq > 0:
                    y_init = np.zeros(neq, dtype=float)
                    self._func.getInitialConditions(t0, neq, y_init)
                    base_y = y_init
                elif hasattr(self._func, "y0"):
                    base_y = np.asarray(self._func.y0, dtype=float).copy()
                else:
                    base_y = np.array([], dtype=float)
        else:
            base_y = np.array([], dtype=float)

        self._base_neq = len(base_y)

        self._np = 0
        if self._func is not None and hasattr(self._func, "nparams"):
            try:
                self._np = int(self._func.nparams())
            except (NotImplementedError, TypeError):
                self._np = 0
        if self._np == 0 and self._func is not None and hasattr(self._func, "m_sens_params"):
            self._np = len(getattr(self._func, "m_sens_params", []))

        if self._np > 0 and self._base_neq > 0:
            self._augmented = True
            aug_len = self._base_neq * (1 + self._np)
            self._y = np.zeros(aug_len, dtype=float)
            self._y[:self._base_neq] = base_y
            self.setSensitivityTolerances(self.rtol_sens, self.atol_sens)
        else:
            self._augmented = False
            self._y = base_y

        if self._y is not None and len(self._y) > 0:
            self._cv_zn = [self._y.copy()] + [np.zeros_like(self._y) for _ in range(5)]

        self.applyOptions()

    def reinitialize(self, t0: float, func_or_y0: Optional[Any] = None, y0: Optional[np.ndarray] = None):
        self.initialize(t0, func_or_y0, y0)

    def sensInit(self, t0: float, func: Any) -> None:
        self.initialize(t0, func)

    def setSensitivityTolerances(self, reltol: float, abstol: float) -> None:
        self.rtol_sens = float(reltol)
        self.atol_sens = float(abstol)
        if hasattr(self, "_func") and self._func is not None and hasattr(self._func, "m_paramScales") and self._func.m_paramScales:
            self._sens_atol = [self.atol_sens / float(s) for s in self._func.m_paramScales]
        else:
            self._sens_atol = [self.atol_sens] * max(1, self._np)

    def nSensParams(self) -> int:
        return self._np

    def sensitivity(self, k: int, p: int) -> float:
        if self._time == getattr(self, "_cv_t0", 0.0):
            return 0.0
        if k >= self._base_neq:
            raise CanteraError(f"CVodesIntegrator::sensitivity: k out of range ({k})")
        if p >= self._np:
            raise CanteraError(f"CVodesIntegrator::sensitivity: p out of range ({p})")
        if not self._augmented or self._y is None:
            return 0.0
        return float(self._y[self._base_neq + p * self._base_neq + k])

    def _eval_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Evaluate the ODE right-hand-side function (handling sensitivity augmentation if enabled)."""
        self._nevals += 1
        if not self._augmented:
            n = len(y)
            value = np.zeros(n, dtype=float)
            if hasattr(self._func, "eval_f"):
                self._func.eval_f(t, y, value)
                return value
            if hasattr(self._func, "eval"):
                try:
                    p = getattr(self._func, "m_sens_params", [])
                    self._func.eval(t, y, value, p)
                    return value
                except TypeError:
                    self._func.eval(t, y, value)
                    return value
            if callable(self._func):
                result = np.asarray(self._func(t, y), dtype=float)
                if result.shape != y.shape:
                    raise NumericsError("RHS result dimensions do not match the state")
                return result
            raise NumericsError("No valid eval method on func")

        # Augmented mode: y contains [base_y, s_0, s_1, ..., s_{P-1}]
        N = self._base_neq
        P = self._np
        base_state = y[:N]
        f0 = np.zeros(N, dtype=float)

        param_vec = getattr(self._func, "m_sens_params", [])
        if hasattr(self._func, "eval_f"):
            self._func.eval_f(t, base_state, f0)
        elif hasattr(self._func, "eval"):
            try:
                self._func.eval(t, base_state, f0, param_vec)
            except TypeError:
                self._func.eval(t, base_state, f0)
        elif callable(self._func):
            f0 = np.asarray(self._func(t, base_state), dtype=float)

        dY = np.zeros_like(y)
        dY[:N] = f0

        srur = math.sqrt(np.finfo(float).eps)
        for p_idx in range(P):
            sp = y[N + p_idx * N : N + (p_idx + 1) * N]
            # Parameter perturbation
            orig_p = param_vec[p_idx]
            dp = srur * max(1.0, abs(orig_p))
            param_vec[p_idx] = orig_p + dp
            fp_p = np.zeros(N, dtype=float)
            if hasattr(self._func, "eval_f"):
                self._func.eval_f(t, base_state, fp_p)
            elif hasattr(self._func, "eval"):
                try:
                    self._func.eval(t, base_state, fp_p, param_vec)
                except TypeError:
                    self._func.eval(t, base_state, fp_p)
            param_vec[p_idx] = orig_p
            df_dp = (fp_p - f0) / dp

            sp_norm = math.sqrt(float(np.mean(sp**2)))
            if sp_norm == 0.0:
                J_sp = np.zeros_like(sp)
            else:
                sigma = srur / max(1.0, sp_norm)
                fp_y = np.zeros(N, dtype=float)
                if hasattr(self._func, "eval_f"):
                    self._func.eval_f(t, base_state + sigma * sp, fp_y)
                elif hasattr(self._func, "eval"):
                    try:
                        self._func.eval(t, base_state + sigma * sp, fp_y, param_vec)
                    except TypeError:
                        self._func.eval(t, base_state + sigma * sp, fp_y)
                J_sp = (fp_y - f0) / sigma
            dY[N + p_idx * N : N + (p_idx + 1) * N] = J_sp + df_dp

        return dY

    def _cv_eval_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        try:
            return self._eval_rhs(t, y)
        except CanteraError as err:
            raise _RecoverableRhsError(str(err)) from err

    def _weighted_rms(self, value: np.ndarray, state: np.ndarray) -> float:
        atol = np.asarray(self.atol, dtype=float)
        if atol.ndim and atol.shape != state.shape:
            raise NumericsError("Absolute tolerance vector does not match the state")
        weights = 1.0 / (atol + self.rtol * np.abs(state))
        return math.sqrt(float(np.mean((value * weights) ** 2)))

    @staticmethod
    def _bdf_error_coefficient(order: int, h: float,
                               previous_steps: Sequence[float]) -> float:
        if order == 1:
            return 0.5
        if len(previous_steps) < order - 1:
            raise NumericsError("Insufficient step history for requested BDF order")
        ell = [0.0] * (order + 1)
        ell[0] = ell[1] = 1.0
        alpha0 = alpha0_hat = -1.0
        hsum = h
        xi_inv = xistar_inv = 1.0
        for j in range(2, order):
            hsum += previous_steps[j - 2]
            xi_inv = h / hsum
            alpha0 -= 1.0 / j
            for i in range(j, 0, -1):
                ell[i] += ell[i - 1] * xi_inv
        alpha0 -= 1.0 / order
        xistar_inv = -ell[1] - alpha0
        hsum += previous_steps[order - 2]
        xi_inv = h / hsum
        alpha0_hat = -ell[1] - xi_inv
        for i in range(order, 0, -1):
            ell[i] += ell[i - 1] * xistar_inv
        a1 = 1.0 - alpha0_hat + alpha0
        a2 = 1.0 + order * a1
        return abs(a1 / (alpha0 * a2))

    def _cv_set_bdf(self, h: float) -> Tuple[List[float], List[float]]:
        q = self._cv_q
        ell = [0.0] * 6
        tq = [0.0] * 6
        ell[0] = ell[1] = 1.0
        xi_inv = xistar_inv = 1.0
        alpha0 = alpha0_hat = -1.0
        hsum = h
        if q > 1:
            for j in range(2, q):
                hsum += self._cv_tau[j - 1]
                xi_inv = h / hsum
                alpha0 -= 1.0 / j
                for i in range(j, 0, -1):
                    ell[i] += ell[i - 1] * xi_inv
            alpha0 -= 1.0 / q
            xistar_inv = -ell[1] - alpha0
            hsum += self._cv_tau[q - 1]
            xi_inv = h / hsum
            alpha0_hat = -ell[1] - xi_inv
            for i in range(q, 0, -1):
                ell[i] += ell[i - 1] * xistar_inv

        a1 = 1.0 - alpha0_hat + alpha0
        a2 = 1.0 + q * a1
        tq[2] = abs(a1 / (alpha0 * a2))
        tq[5] = abs(a2 * xistar_inv / (ell[q] * xi_inv))
        if self._cv_qwait == 1:
            if q > 1:
                c = xistar_inv / ell[q]
                a3 = alpha0 + 1.0 / q
                a4 = alpha0_hat + xi_inv
                cp_inv = (1.0 - a4 + a3) / a3
                tq[1] = abs(c * cp_inv)
            else:
                tq[1] = 1.0
            hsum += self._cv_tau[q]
            xi_inv = h / hsum
            a5 = alpha0 - 1.0 / (q + 1)
            a6 = alpha0_hat - xi_inv
            cpp_inv = (1.0 - a6 + a5) / a2
            tq[3] = abs(cpp_inv / (xi_inv * (q + 2) * a5))
        tq[4] = 0.1 / tq[2]
        return ell, tq

    def _cv_rescale(self, eta: float) -> None:
        factor = eta
        for j in range(1, self._cv_q + 1):
            self._cv_zn[j] *= factor
            factor *= eta
        self._cv_h = self._cv_hscale * eta
        self._cv_next_h = self._cv_h
        self._cv_hscale = self._cv_h
        self._cv_nscon = 0

    def _cv_adjust_order(self, deltaq: int) -> None:
        q = self._cv_q
        if q == 2 and deltaq != 1:
            return
        if deltaq == 1:
            ell = [0.0] * 6
            ell[2] = 1.0
            alpha1 = prod = xiold = 1.0
            alpha0 = -1.0
            hsum = self._cv_hscale
            for j in range(1, q):
                hsum += self._cv_tau[j + 1]
                xi = hsum / self._cv_hscale
                prod *= xi
                alpha0 -= 1.0 / (j + 1)
                alpha1 += 1.0 / xi
                for i in range(j + 2, 1, -1):
                    ell[i] = ell[i] * xiold + ell[i - 1]
                xiold = xi
            a1 = (-alpha0 - alpha1) / prod
            self._cv_zn[q + 1] = a1 * self._cv_zn[5]
            for j in range(2, q + 1):
                self._cv_zn[j] += ell[j] * self._cv_zn[q + 1]
            return

        if deltaq == -1:
            ell = [0.0] * 6
            ell[2] = 1.0
            hsum = 0.0
            for j in range(1, q - 1):
                hsum += self._cv_tau[j]
                xi = hsum / self._cv_hscale
                for i in range(j + 2, 1, -1):
                    ell[i] = ell[i] * xi + ell[i - 1]
            for j in range(2, q):
                self._cv_zn[j] -= ell[j] * self._cv_zn[q]

    def _cv_adjust_params(self, eta: float) -> None:
        if self._cv_qprime != self._cv_q:
            self._cv_adjust_order(self._cv_qprime - self._cv_q)
            self._cv_q = self._cv_qprime
            self._cv_L = self._cv_q + 1
            self._cv_qwait = self._cv_L
        self._cv_rescale(eta)

    def _cv_set_eta(self, eta: float) -> Tuple[float, float]:
        if 0.0 < eta < 1.5:
            return 1.0, self._cv_h
        if eta >= 1.5:
            eta = min(eta, self._cv_etamax)
            if self.max_step > 0.0:
                eta /= max(1.0, abs(self._cv_h) * eta / self.max_step)
        else:
            eta = max(eta, 0.1)
        if self._cv_qprime < self._cv_q:
            self._cv_nscon = 0
        return eta, self._cv_h * eta

    def _cv_prepare_next_step(self, dsm: float, tq: Sequence[float],
                              acor: np.ndarray, weights: np.ndarray) -> None:
        if self._cv_etamax == 1.0:
            self._cv_qwait = max(self._cv_qwait, 2)
            self._cv_qprime = self._cv_q
            self._cv_eta = 1.0
            self._cv_hprime = self._cv_h
            return

        etaq = 1.0 / (6.0 * dsm ** (1.0 / self._cv_L) + 1e-6)
        if self._cv_qwait != 0:
            self._cv_eta = etaq
            self._cv_qprime = self._cv_q
            self._cv_eta, self._cv_hprime = self._cv_set_eta(self._cv_eta)
            return

        self._cv_qwait = 2
        etaqm1 = 0.0
        if self._cv_q > 1:
            ddn = self._cv_wrms(self._cv_zn[self._cv_q], weights) * tq[1]
            etaqm1 = 1.0 / (6.0 * ddn ** (1.0 / self._cv_q) + 1e-6)
        etaqp1 = 0.0
        if self._cv_q < self._cv_max_order and self._cv_saved_tq5 != 0.0:
            if len(self._cv_tau) > 2 and self._cv_tau[2] != 0.0:
                cquot = (tq[5] / self._cv_saved_tq5) * (
                    self._cv_h / self._cv_tau[2]
                ) ** self._cv_L
                dup_vec = acor - cquot * self._cv_zn[5]
                dup = self._cv_wrms(dup_vec, weights) * tq[3]
                etaqp1 = 1.0 / (10.0 * dup ** (1.0 / (self._cv_L + 1)) + 1e-6)

        etam = max(etaqm1, etaq, etaqp1)
        if 0.0 < etam < 1.5:
            self._cv_eta = 1.0
            self._cv_qprime = self._cv_q
        elif etam == etaq:
            self._cv_eta = etaq
            self._cv_qprime = self._cv_q
        elif etam == etaqm1:
            self._cv_eta = etaqm1
            self._cv_qprime = self._cv_q - 1
        else:
            self._cv_eta = etaqp1
            self._cv_qprime = self._cv_q + 1
            self._cv_zn[5] = acor.copy()
        self._cv_eta, self._cv_hprime = self._cv_set_eta(self._cv_eta)

    def _cv_set_error_weights(self, state: np.ndarray) -> np.ndarray:
        if self._augmented:
            N = self._base_neq
            P = self._np
            atol_arr = np.zeros(len(state), dtype=float)
            base_atol = np.asarray(self.atol, dtype=float)
            if base_atol.ndim == 0:
                atol_arr[:N] = float(base_atol)
            else:
                atol_arr[:N] = base_atol[:N]
            for p_idx in range(P):
                p_atol = self._sens_atol[p_idx] if self._sens_atol and p_idx < len(self._sens_atol) else self.atol_sens
                atol_arr[N + p_idx * N : N + (p_idx + 1) * N] = p_atol
            denominator = atol_arr + self.rtol * np.abs(state)
        else:
            atol = np.asarray(self.atol, dtype=float)
            if atol.ndim and atol.shape != state.shape:
                raise NumericsError("Absolute tolerance vector does not match the state")
            denominator = atol + self.rtol * np.abs(state)

        if np.any(denominator <= 0.0):
            raise NumericsError("CVODE error weights require positive error tolerances")
        self._cv_ewt = 1.0 / denominator
        return self._cv_ewt

    @staticmethod
    def _cv_wrms(value: np.ndarray, weights: np.ndarray) -> float:
        return math.sqrt(float(np.mean((value * weights) ** 2)))

    def _cv_dense_dq_min_increment(self, rhs: np.ndarray,
                                   weights: np.ndarray, n: int) -> float:
        fnorm = self._cv_wrms(rhs, weights)
        if fnorm == 0.0:
            return 1.0
        return (1000.0 * abs(self._cv_h) * np.finfo(float).eps
                * n * fnorm)

    @staticmethod
    def _cv_dense_dq_increment(state_value: float, error_weight: float,
                               min_increment: float) -> float:
        srur = math.sqrt(np.finfo(float).eps)
        return max(srur * abs(state_value), min_increment / error_weight)

    def _cv_initial_step(self, tout: float, ydot0: np.ndarray,
                         weights: np.ndarray) -> float:
        tdist = abs(tout - self._cv_tn)
        if tdist == 0.0:
            raise NumericsError("Initial output time is too close to the initial time")
        uround = np.finfo(float).eps
        tround = uround * max(abs(self._cv_tn), abs(tout))
        if tdist < 2.0 * tround:
            raise NumericsError("Initial output time is too close to the initial time")
        hlb = 100.0 * tround
        y = self._cv_zn[0]
        atol = np.asarray(self.atol, dtype=float)
        if self._augmented:
            atol = np.full(len(y), self.atol if atol.ndim == 0 else atol[0])
        error_scale = atol + self.rtol * np.abs(y)
        denom = 0.1 * np.abs(y) + error_scale
        hub_inv = float(np.max(np.abs(ydot0) / denom))
        hub = 0.1 * tdist
        if hub * hub_inv > 1.0:
            hub = 1.0 / hub_inv
        hg = math.sqrt(hlb * hub)
        if hub < hlb:
            return hg

        hs = hg
        hnew = hg
        valid_estimate = False
        for count1 in range(1, 5):
            ytrial = y + hg * ydot0
            try:
                ftrial = self._cv_eval_rhs(self._cv_tn + hg, ytrial)
            except _RecoverableRhsError:
                hnew = hg * 0.25
                hg = hnew
                if count1 == 4:
                    break
                continue
            valid_estimate = True
            ydd = (ftrial - ydot0) / hg
            yddnorm = self._cv_wrms(ydd, weights)
            hs = hg
            hnew = math.sqrt(2.0 / yddnorm) if yddnorm * hub * hub > 2.0 else math.sqrt(hg * hub)
            if count1 == 4:
                break
            ratio = hnew / hg
            if 0.5 < ratio < 2.0:
                break
            if count1 > 1 and ratio > 2.0:
                hnew = hg
                break
            hg = hnew
        if not valid_estimate:
            raise NumericsError("CVODE BDF could not estimate a valid initial step")
        h0 = min(hub, max(hlb, 0.5 * hnew))
        if self.max_step > 0.0:
            h0 = min(h0, self.max_step)
        if self.min_step > 0.0:
            h0 = max(h0, self.min_step)
        return h0

    def _cv_nordsieck_predict(self) -> float:
        tnew = self._cv_tn + self._cv_h
        for k in range(1, self._cv_q + 1):
            for j in range(self._cv_q, k - 1, -1):
                self._cv_zn[j - 1] = self._cv_zn[j - 1] + self._cv_zn[j]
        return tnew

    def _cv_newton_corrector(self, tnew: float, ell: Sequence[float], tq: Sequence[float],
                             weights: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        rl1 = 1.0 / ell[1]
        gamma = self._cv_h * rl1
        acor = np.zeros_like(self._cv_zn[0])
        predicted = self._cv_zn[0].copy()
        n = len(predicted)
        try:
            f0 = self._cv_eval_rhs(tnew, predicted)
            previous_factor = self._cv_linear_factor
            previous_gamma = self._cv_lsetup_gamma
            factor = self._cv_prepare_linear_system(
                tnew, predicted, f0, gamma, weights)
        except _RecoverableRhsError:
            return acor, math.inf, False
        if factor is None:
            return acor, math.inf, False

        crate = 1.0
        delp = 0.0
        acnrm = math.inf
        for iteration in range(3):
            self._cv_nonlinear_iters += 1
            ytrial = predicted + acor
            try:
                ftrial = f0 if iteration == 0 else self._cv_eval_rhs(tnew, ytrial)
            except _RecoverableRhsError:
                return acor, math.inf, False
            residual = rl1 * self._cv_zn[1] + acor - gamma * ftrial
            if not np.all(np.isfinite(residual)):
                return acor, math.inf, False
            delta = factor.solve(-residual)
            if factor is previous_factor and previous_gamma != 0.0:
                gamma_ratio = gamma / previous_gamma
                delta *= 2.0 / (1.0 + gamma_ratio)
            del_norm = self._cv_wrms(delta, weights)
            acor += delta
            if iteration > 0:
                crate = max(0.3 * crate, del_norm / max(delp, np.finfo(float).tiny))
            dcon = del_norm * min(1.0, crate) / tq[4]
            if dcon <= 1.0:
                acnrm = del_norm if iteration == 0 else self._cv_wrms(acor, weights)
                return acor, acnrm, True
            if iteration >= 1 and del_norm > 2.0 * delp:
                return acor, math.inf, False
            delp = del_norm
        return acor, math.inf, False

    def _cv_prepare_linear_system(self, time: float, state: np.ndarray,
                                  rhs: np.ndarray, gamma: float,
                                  weights: np.ndarray):
        setup = (
            self._cv_last_lsetup_step < 0
            or self._cv_force_lsetup
            or self._cv_nst >= self._cv_last_lsetup_step + 20
            or (self._cv_lsetup_gamma != 0.0 and
                abs(gamma / self._cv_lsetup_gamma - 1.0) > 0.3)
        )
        if not setup:
            return self._cv_linear_factor

        jacobian_bad = (
            self._cv_force_jacobian
            or self._cv_rhs_jacobian is None
            or self._cv_nst >= self._cv_last_jac_step + 51
        )
        n = len(state)
        if jacobian_bad:
            jacobian = np.empty((n, n), dtype=float)
            min_increment = self._cv_dense_dq_min_increment(rhs, weights, n)
            for column in range(n):
                increment = self._cv_dense_dq_increment(
                    float(state[column]), float(weights[column]), min_increment)
                perturbed = state.copy()
                perturbed[column] += increment
                fp = self._cv_eval_rhs(time, perturbed)
                for row in range(n):
                    jacobian[row, column] = (float(fp[row]) - float(rhs[row])) / increment
            self._cv_rhs_jacobian = jacobian
            self._cv_last_jac_step = self._cv_nst
            self._cv_jacobian_evaluations += 1

        system = np.eye(n, dtype=float) - gamma * self._cv_rhs_jacobian
        try:
            self._cv_linear_factor = _factor_dense_matrix(system)
        except (NumericsError, FloatingPointError, ValueError):
            self._cv_linear_factor = None
            return None
        self._cv_lsetup_gamma = gamma
        self._cv_last_lsetup_step = self._cv_nst
        self._cv_force_lsetup = False
        self._cv_force_jacobian = False
        self._cv_linear_setups += 1
        return self._cv_linear_factor

    def _cv_complete_step(self, ell: Sequence[float], tq: Sequence[float],
                          acor: np.ndarray) -> None:
        self._cv_nst += 1
        self._cv_tau[2:self._cv_q + 1] = self._cv_tau[1:self._cv_q]
        if self._cv_q == 1 and self._cv_nst > 1:
            self._cv_tau[2] = self._cv_tau[1]
        self._cv_tau[1] = self._cv_h
        for j in range(self._cv_q + 1):
            self._cv_zn[j] = self._cv_zn[j] + ell[j] * acor
        self._cv_qwait -= 1
        if self._cv_qwait == 1 and self._cv_q != self._cv_max_order:
            self._cv_zn[5] = acor.copy()
            self._cv_saved_tq5 = tq[5]

    def _cv_take_one_internal_step(self, tout: float) -> float:
        """Advance the integrator by exactly one internal BDF step."""
        if self._cv_nst == 0:
            weights = self._cv_set_error_weights(self._cv_zn[0])
            ydot0 = self._eval_rhs(self._cv_tn, self._cv_zn[0])
            self._cv_h = self._cv_initial_step(tout, ydot0, weights)
            if self.max_step > 0.0:
                self._cv_h = min(self._cv_h, self.max_step)
            if self.min_step > 0.0:
                self._cv_h = max(self._cv_h, self.min_step)
            self._cv_hscale = self._cv_h
            self._cv_hprime = self._cv_h
            self._cv_zn[1] = self._cv_h * ydot0

        weights = self._cv_set_error_weights(self._cv_zn[0])

        if self._cv_nst > 0 and self._cv_hprime != self._cv_h:
            self._cv_eta = self._cv_hprime / self._cv_h
            self._cv_adjust_params(self._cv_eta)

        ncf = nef = 0
        retry = True
        while retry:
            retry = False
            saved_zn = [entry.copy() for entry in self._cv_zn]
            saved_t = self._cv_tn
            tnew = self._cv_nordsieck_predict()
            ell, tq = self._cv_set_bdf(self._cv_h)
            acor, acnrm, converged = self._cv_newton_corrector(
                tnew, ell, tq, weights
            )
            if not converged:
                ncf += 1
                self._cv_nonlinear_failures += 1
                self._cv_zn = saved_zn
                self._cv_etamax = 1.0
                self._cv_force_lsetup = True
                self._cv_force_jacobian = True
                if ncf >= 10:
                    raise NumericsError("CVODE BDF nonlinear convergence failure")
                self._cv_eta = max(0.25, 0.0)
                self._cv_rescale(self._cv_eta)
                retry = True
                continue

            dsm = acnrm * tq[2]
            if not math.isfinite(dsm) or dsm > 1.0:
                nef += 1
                self._cv_error_test_failures += 1
                self._cv_zn = saved_zn
                self._cv_tn = saved_t
                self._cv_force_lsetup = True
                max_fails = self._max_err_test_fails if self._max_err_test_fails > 0 else 7
                if nef >= max_fails:
                    raise NumericsError(f"CVODE BDF local error test failed {max_fails} times")
                self._cv_etamax = 1.0
                if nef <= 3:
                    eta = 1.0 / (6.0 * max(dsm, 0.0) ** (1.0 / self._cv_L) + 1e-6)
                    eta = max(0.1, eta)
                    if nef >= 2:
                        eta = min(eta, 0.2)
                    self._cv_eta = eta
                    self._cv_rescale(eta)
                elif self._cv_q > 1:
                    self._cv_eta = 0.1
                    self._cv_adjust_order(-1)
                    self._cv_L = self._cv_q
                    self._cv_q -= 1
                    self._cv_qprime = self._cv_q
                    self._cv_qwait = self._cv_L
                    self._cv_rescale(self._cv_eta)
                else:
                    self._cv_eta = 0.1
                    self._cv_h *= self._cv_eta
                    self._cv_hprime = self._cv_h
                    self._cv_hscale = self._cv_h
                    self._cv_qwait = 10
                    self._cv_nscon = 0
                    self._cv_zn[1] = self._cv_h * self._eval_rhs(
                        self._cv_tn, self._cv_zn[0]
                    )
                retry = True
                continue

            self._cv_tn = tnew
            self._cv_complete_step(ell, tq, acor)
            self._cv_order_histogram[self._cv_q] = (
                self._cv_order_histogram.get(self._cv_q, 0) + 1)
            self._cv_prepare_next_step(dsm, tq, acor, weights)
            self._cv_etamax = (
                self._CV_ETA_MAX_EARLY_STEPS
                if self._cv_nst <= self._CV_SMALL_NST
                else self._CV_ETA_MAX_GENERAL
            )
            retry = False

        self._y = self._cv_zn[0].copy()
        self._cv_order = self._cv_q
        self._cv_next_h = self._cv_hprime
        return self._cv_tn

    def _cv_interpolate(self, tout: float) -> np.ndarray:
        """Evaluate CVODE's accepted Nordsieck polynomial at the requested time."""
        if self._cv_tn == tout:
            return self._cv_zn[0].copy()
        hu = self._cv_tau[1]
        if hu == 0.0:
            raise NumericsError("CVODE interpolation history is unavailable")
        ratio = (tout - self._cv_tn) / hu
        result = self._cv_zn[0].copy()
        factor = ratio
        for j in range(1, self._cv_q + 1):
            result += factor * self._cv_zn[j]
            factor *= ratio
        return result

    def integrate(self, tout: float) -> float:
        if self._func is None or self._y is None:
            raise NumericsError("Integrator not initialized with function and state")
        tout = float(tout)
        if tout == self._time:
            return self._time
        if tout < self._time:
            raise CanteraError(
                f"CVodesIntegrator::integrate: Cannot integrate backwards in time.\\n"
                f"Requested time {tout} < current time {self._time}"
            )
        if not self._cv_zn:
            self._cv_zn = [self._y.copy()] + [np.zeros_like(self._y) for _ in range(5)]

        nsteps = 0
        while self._cv_tn < tout:
            if nsteps >= self.max_steps:
                raise CanteraError(
                    f"CVodesIntegrator::integrate: Maximum number of timesteps ({self.max_steps}) "
                    f"taken without reaching output time ({tout}).\\n"
                    f"Current integrator time: {self._cv_tn}"
                )
            self._cv_take_one_internal_step(tout)
            nsteps += 1

        self._time = tout
        self._y = self._cv_interpolate(tout)
        self._sens_ok = False
        return self._time

    def step(self, tout: float) -> float:
        if self._func is None or self._y is None:
            raise NumericsError("Integrator not initialized with function and state")
        tout = float(tout)
        if not self._cv_zn:
            self._cv_zn = [self._y.copy()] + [np.zeros_like(self._y) for _ in range(5)]
        self._cv_take_one_internal_step(tout)
        self._time = self._cv_tn
        self._y = self._cv_zn[0].copy()
        self._sens_ok = False
        return self._time

    def derivative(self, tout: float, n: int = 1) -> np.ndarray:
        if n == 0:
            full = self._cv_interpolate(tout)
            return full[:self._base_neq] if self._augmented else full
        if not self._cv_zn or len(self._cv_zn) < 2:
            return np.zeros(self._base_neq, dtype=float)
        hu = self._cv_tau[1]
        if hu == 0.0:
            if n == 1 and self._func is not None and self._y is not None:
                full_rhs = self._eval_rhs(tout, self._y)
                return full_rhs[:self._base_neq] if self._augmented else full_rhs
            return np.zeros(self._base_neq, dtype=float)
        ratio = (float(tout) - self._cv_tn) / hu
        q = self._cv_q
        res = np.zeros_like(self._cv_zn[0])
        for j in range(n, q + 1):
            coeff = math.factorial(j) / math.factorial(j - n)
            res += coeff * self._cv_zn[j] * (ratio ** (j - n))
        res /= (hu ** n)
        return res[:self._base_neq] if self._augmented else res

    def solution(self, k: Optional[int] = None) -> Union[float, np.ndarray]:
        if self._y is None:
            return 0.0 if k is not None else np.array([])
        base_view = self._y[:self._base_neq] if self._augmented else self._y
        if k is not None:
            return float(base_view[k])
        return base_view.copy()

    def lastOrder(self) -> int:
        return self._cv_order

    def nEquations(self) -> int:
        return self._base_neq if self._augmented else (len(self._y) if self._y is not None else 0)

    def nEvals(self) -> int:
        return self._nevals

    def maxOrder(self) -> int:
        return self._cv_max_order

    def setMaxOrder(self, order: int) -> None:
        self._cv_max_order = max(1, min(5, int(order)))

    def setMethod(self, t: Any) -> None:
        s = str(t).upper()
        if s in ("1", "BDF", "BDF_METHOD"):
            self._method = 1
        elif s in ("0", "ADAMS", "ADAMS_METHOD"):
            self._method = 0
        else:
            raise CanteraError(f"CVodesIntegrator::setMethod: unknown method {t}")

    def setMaxStepSize(self, hmax: float) -> None:
        self.max_step = float(hmax)

    def setMinStepSize(self, hmin: float) -> None:
        self.min_step = float(hmin)

    def setMaxSteps(self, nmax: int) -> None:
        self.max_steps = int(nmax)

    def maxSteps(self) -> int:
        return self.max_steps

    def setMaxErrTestFails(self, n: int) -> None:
        self._max_err_test_fails = int(n)

    def setLinearSolverType(self, solver: str) -> None:
        self._linear_solver = str(solver).upper()

    def linearSolverType(self) -> str:
        return getattr(self, "_linear_solver", "DENSE")

    def setBandwidth(self, kl: int, ku: int) -> None:
        self._mupper = int(ku)
        self._mlower = int(kl)

    def setRootFunctionCount(self, nroots: int) -> None:
        self._n_root_functions = int(nroots)

    def applyOptions(self) -> None:
        pass

    def checkError(self, flag: int, ctMethod: str, cvodesMethod: str) -> None:
        if flag == 0:
            return
        if flag == -1:
            raise CanteraError(f"CVodesIntegrator::{ctMethod}: CVODES integrator is not initialized")
        raise CanteraError(f"CVodesIntegrator::{ctMethod}: {cvodesMethod} returned error code {flag}")

    def getErrorInfo(self, N: int = 10) -> str:
        if self._y is None or len(self._y) == 0:
            return ""
        weights = getattr(self, "_cv_ewt", None)
        if weights is None:
            weights = np.ones_like(self._y)
        errs = np.zeros_like(self._y)
        if len(self._cv_zn) > 1:
            errs = np.abs(self._cv_zn[1] * weights)
        idx = np.argsort(-errs)[:N]
        return "".join(f"{i}: {errs[i]}\\n" for i in idx)

    def solverStats(self) -> dict:
        return {
            "steps": int(self._cv_nst),
            "step_solve_fails": int(self._cv_nonlinear_failures),
            "rhs_evals": int(self._nevals),
            "nonlinear_iters": int(getattr(self, "_cv_nonlinear_iters", self._nevals)),
            "nonlinear_conv_fails": int(self._cv_nonlinear_failures),
            "err_test_fails": int(self._cv_error_test_failures),
            "last_order": int(self._cv_order),
            "stab_order_reductions": 0,
            "jac_evals": int(self._cv_jacobian_evaluations),
            "lin_solve_setups": int(self._cv_linear_setups),
            "lin_rhs_evals": 0,
            "lin_iters": 0,
            "lin_conv_fails": 0,
            "prec_evals": 0,
            "prec_solves": 0,
            "jt_vec_setup_evals": 0,
            "jt_vec_prod_evals": 0,
        }
'''

with open('cantera_python/numerics.py', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'class CVodesIntegrator\(Integrator\):[\s\S]*?(?=class _DAENonlinearFailure\(NumericsError\):)'
match = re.search(pattern, content)
if not match:
    raise ValueError("Could not find CVodesIntegrator definition in numerics.py")

new_content = content[:match.start()] + cvodes_code + '\n\n' + content[match.end():]

with open('cantera_python/numerics.py', 'w', encoding='utf-8', newline='\n') as f:
    f.write(new_content)

print("Successfully patched CVodesIntegrator in cantera_python/numerics.py")
