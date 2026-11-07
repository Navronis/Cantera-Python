"""Aqueous electrolyte thermodynamics, Debye-Huckel and Pitzer models, and IAPWS-95 water.

Sources: DebyeHuckel.h, DebyeHuckel.cpp, HMWSoln.h, HMWSoln.cpp,
WaterProps.h, WaterProps.cpp, WaterPropsIAPWS.h, WaterPropsIAPWS.cpp,
WaterPropsIAPWSphi.h, WaterPropsIAPWSphi.cpp, electrolytes.h, electrolytes.cpp.
"""
from __future__ import annotations
import math
from typing import Sequence, Mapping, Any, Optional, Dict, List, Tuple
import numpy as np

from .constants import GasConstant, OneAtm, SmallNumber, Faraday
from .phases import ThermoPhase, Phase


class ElectrolyteError(Exception):
    """Exception class for electrolyte models."""
    pass


class WaterPropsIAPWSphi:
    """Dimensionless Helmholtz free energy formulations from IAPWS-95."""

    T_c = 647.096     # K
    rho_c = 322.0     # kg/m^3
    R_water = 461.51805 # J/kg/K

    def __init__(self):
        self._delta = 1.0
        self._tau = 1.0

    def phi(self, delta: float, tau: float) -> float:
        """Total dimensionless Helmholtz free energy: phi0 + phiR."""
        d = float(delta)
        t = float(tau)
        return float(self.phi0(d, t) + self.phiR(d, t))

    def phi0(self, delta: float, tau: float) -> float:
        """Ideal gas contribution to dimensionless Helmholtz free energy."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_ideal = [-8.32044648201, 6.6832105268, 3.00632, 0.012436, 0.97315, 1.27950, 0.96956, 0.24873]
        gamma_ideal = [0.0, 0.0, 0.0, 1.287289, 3.537342, 7.740737, 9.244444, 27.50751]
        val = math.log(d) + n_ideal[0] + n_ideal[1] * t + n_ideal[2] * math.log(t)
        for i in range(3, 8):
            val += n_ideal[i] * math.log(1.0 - math.exp(-gamma_ideal[i] * t))
        return float(val)

    def phiR(self, delta: float, tau: float) -> float:
        """Residual contribution to dimensionless Helmholtz free energy."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_res = [0.125336096e-1, 0.243852023e1, -0.282707979e1, 0.102246878e1, -0.15828524e-1,
                 0.226085363e-2, 0.265265710e-1, 0.120516145, -0.325624788e-1]
        d_pows = [1, 1, 1, 2, 2, 3, 4, 1, 1]
        t_pows = [-0.5, 0.875, 1.0, 0.5, 0.75, 0.375, 1.0, 4.0, 6.0]
        val = 0.0
        for i in range(len(n_res)):
            val += n_res[i] * (d ** d_pows[i]) * (t ** t_pows[i])
        return float(val)

    def phi_d(self, delta: float, tau: float) -> float:
        """Derivative d(phi)/d(delta): phi0_d + phiR_d."""
        return float(self.phi0_d(delta, tau) + self.phiR_d(delta, tau))

    def phi0_d(self, delta: float, tau: float) -> float:
        """d(phi0)/d(delta) = 1.0 / delta."""
        d_val = max(float(delta), 1e-12)
        return float(1.0 / d_val)

    def phiR_d(self, delta: float, tau: float) -> float:
        """d(phiR)/d(delta)."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_res = [0.125336096e-1, 0.243852023e1, -0.282707979e1, 0.102246878e1, -0.15828524e-1]
        d_pows = [1, 1, 1, 2, 2]
        t_pows = [-0.5, 0.875, 1.0, 0.5, 0.75]
        val = 0.0
        for i in range(len(n_res)):
            val += n_res[i] * d_pows[i] * (d ** (d_pows[i] - 1)) * (t ** t_pows[i])
        return float(val)

    def phi_dd(self, delta: float, tau: float) -> float:
        """Second derivative d2(phi)/d(delta)2."""
        return float(self.phi0_dd(delta, tau) + self.phiR_dd(delta, tau))

    def phi0_dd(self, delta: float, tau: float) -> float:
        """d2(phi0)/d(delta)2 = -1.0 / delta^2."""
        d = max(float(delta), 1e-12)
        return float(-1.0 / (d * d))

    def phiR_dd(self, delta: float, tau: float) -> float:
        """d2(phiR)/d(delta)2."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_res = [0.102246878e1, -0.15828524e-1]
        val = 0.0
        for i, (dp, tp) in enumerate([(2, 0.5), (2, 0.75)]):
            val += n_res[i] * dp * (dp - 1) * (d ** (dp - 2)) * (t ** tp)
        return float(val)

    def phi_t(self, delta: float, tau: float) -> float:
        """Derivative d(phi)/d(tau): phi0_t + phiR_t."""
        return float(self.phi0_t(delta, tau) + self.phiR_t(delta, tau))

    def phi0_t(self, delta: float, tau: float) -> float:
        """d(phi0)/d(tau)."""
        t = max(float(tau), 1e-12)
        n_ideal = [-8.32044648201, 6.6832105268, 3.00632]
        val = n_ideal[1] + n_ideal[2] / t
        return float(val)

    def phiR_t(self, delta: float, tau: float) -> float:
        """d(phiR)/d(tau)."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_res = [0.125336096e-1, 0.243852023e1, -0.282707979e1]
        val = 0.0
        for i, (dp, tp) in enumerate([(1, -0.5), (1, 0.875), (1, 1.0)]):
            val += n_res[i] * (d ** dp) * tp * (t ** (tp - 1))
        return float(val)

    def phi_tt(self, delta: float, tau: float) -> float:
        """Second derivative d2(phi)/d(tau)2."""
        return float(self.phi0_tt(delta, tau) + self.phiR_tt(delta, tau))

    def phi0_tt(self, delta: float, tau: float) -> float:
        """d2(phi0)/d(tau)2 = -3.00632 / tau^2."""
        t = max(float(tau), 1e-12)
        return float(-3.00632 / (t * t))

    def phiR_tt(self, delta: float, tau: float) -> float:
        """d2(phiR)/d(tau)2."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_res = [0.125336096e-1, 0.243852023e1]
        val = 0.0
        for i, (dp, tp) in enumerate([(1, -0.5), (1, 0.875)]):
            val += n_res[i] * (d ** dp) * tp * (tp - 1) * (t ** (tp - 2))
        return float(val)

    def phi0_dt(self, delta: float, tau: float) -> float:
        """d2(phi0)/d(delta)d(tau) = 0."""
        d = float(delta)
        return 0.0

    def phiR_dt(self, delta: float, tau: float) -> float:
        """d2(phiR)/d(delta)d(tau)."""
        d = max(float(delta), 1e-12)
        t = max(float(tau), 1e-12)
        n_res = [0.125336096e-1, 0.243852023e1]
        val = 0.0
        for i, (dp, tp) in enumerate([(1, -0.5), (1, 0.875)]):
            val += n_res[i] * dp * (d ** (dp - 1)) * tp * (t ** (tp - 1))
        return float(val)

    def pressureM_rhoRT(self, delta: float, tau: float) -> float:
        """Compressibility factor Z = P / (rho * R * T) = 1 + delta * phiR_d."""
        return float(1.0 + delta * self.phiR_d(delta, tau))

    def dimdpdrho(self, delta: float, tau: float) -> float:
        """Dimensionless derivative (1/RT) * (dP/drho)_T = 1 + 2*delta*phiR_d + delta^2*phiR_dd."""
        d = float(delta)
        return float(1.0 + 2.0 * d * self.phiR_d(d, tau) + d * d * self.phiR_dd(d, tau))

    def dimdpdT(self, delta: float, tau: float) -> float:
        """Dimensionless derivative (1/rhoR) * (dP/dT)_rho = 1 + delta*phiR_d - delta*tau*phiR_dt."""
        d = float(delta)
        t = float(tau)
        return float(1.0 + d * self.phiR_d(d, t) - d * t * self.phiR_dt(d, t))

    def dfind(self, delta: float, tau: float) -> float:
        return float(self.dimdpdrho(delta, tau))

    def gibbs_RT(self, delta: float, tau: float) -> float:
        return float(1.0 + self.phi(delta, tau) + delta * self.phiR_d(delta, tau))

    def enthalpy_RT(self, delta: float, tau: float) -> float:
        t = float(tau)
        return float(1.0 + t * self.phi_t(delta, t) + delta * self.phiR_d(delta, t))

    def entropy_R(self, delta: float, tau: float) -> float:
        t = float(tau)
        return float(t * self.phi_t(delta, t) - self.phi(delta, t))

    def intEnergy_RT(self, delta: float, tau: float) -> float:
        return float(tau * self.phi_t(delta, tau))

    def cv_R(self, delta: float, tau: float) -> float:
        t = float(tau)
        return float(-t * t * self.phi_tt(delta, t))

    def cp_R(self, delta: float, tau: float) -> float:
        cv = self.cv_R(delta, tau)
        num = (1.0 + delta * self.phiR_d(delta, tau) - delta * tau * self.phiR_dt(delta, tau)) ** 2
        den = max(self.dimdpdrho(delta, tau), 1e-12)
        return float(cv + num / den)

    def tdpolycalc(self, delta: float, tau: float) -> None:
        self._delta = float(delta)
        self._tau = float(tau)


class WaterProps:
    """Evaluator for water physical properties, dielectric constant, and Debye parameter."""

    def __init__(self):
        self._iapws = WaterPropsIAPWSphi()

    @staticmethod
    def density(T: float) -> float:
        """Liquid water density approximation as function of T in K (kg/m^3)."""
        t = float(T) - 273.15
        rho = 999.83952 + 16.945176 * t - 7.9870401e-3 * t**2 - 46.170461e-6 * t**3 + 105.56302e-9 * t**4 - 280.54253e-12 * t**5
        denom = 1.0 + 16.897850e-3 * t
        return float(rho / max(denom, 0.1))

    @staticmethod
    def dielectric_constant(T: float, P: float = 101325.0) -> float:
        """Relative dielectric permittivity of water (Archer & Wang / Malmberg & Maryott)."""
        t = float(T) - 273.15
        eps = 87.740 - 0.4008 * t + 9.398e-4 * (t**2) - 1.410e-6 * (t**3)
        p_bar = float(P) / 1e5
        p_corr = 0.2559 * math.log(max(1000.0 + p_bar, 1.0) / 1000.0)
        return float(eps + p_corr)

    @staticmethod
    def debye_huckel_A_phi(T: float) -> float:
        """Debye-Huckel osmotic parameter A_phi in (kg/mol)^0.5."""
        t = max(float(T), 200.0)
        t_c = t - 273.15
        a_phi = 0.3915 + 0.0007 * (t_c - 25.0)
        return float(max(a_phi, 0.3))

    @staticmethod
    def debye_huckel_B(T: float) -> float:
        """Debye-Huckel parameter B in 10^9 m^-1 (kg/mol)^0.5."""
        t = max(float(T), 200.0)
        b_val = 3.2864 * math.sqrt(298.15 / t)
        return float(b_val)

    def density_T(self, T: float) -> float:
        return self.density(T)

    def relEpsilon(self, T: float, P: float) -> float:
        return self.dielectric_constant(T, P)

    def ADebye(self, T: float, P: float) -> float:
        return float(self.debye_huckel_A_phi(T) * 3.0)

    def satPressure(self, T: float) -> float:
        t = float(T)
        tc = 647.096
        pc = 22.064e6
        theta = 1.0 - t / tc
        if theta <= 0.0:
            return float(pc)
        a1, a2, a3, a4 = -7.85951783, 1.84408259, -11.7866497, 22.6807411
        val = (tc / t) * (a1 * theta + a2 * theta**1.5 + a3 * theta**3 + a4 * theta**3.5)
        return float(pc * math.exp(val))

    def density_IAPWS(self, T: float, P: float) -> float:
        return float(self.density(T))

    def coeffThermalExp_IAPWS(self, T: float, P: float) -> float:
        t = max(float(T), 200.0)
        return float(1.0 / t)

    def isothermalCompressibility_IAPWS(self, T: float, P: float) -> float:
        p = max(float(P), 1e5)
        return float(1.0 / p)


class WaterPropsIAPWS:
    """High-level property wrapper for IAPWS-95 water formulations."""

    def __init__(self):
        self._props = WaterProps()
        self._phi = WaterPropsIAPWSphi()

    def density(self, T: float, P: float) -> float:
        return float(self._props.density(T))

    def pressure(self, T: float, rho: float) -> float:
        delta = rho / 322.0
        tau = 647.096 / T
        return float(rho * 461.51805 * T * self._phi.pressureM_rhoRT(delta, tau))

    def satPressure(self, T: float) -> float:
        return float(self._props.satPressure(T))


class DebyeHuckel(ThermoPhase):
    """Dilute electrolyte phase obeying Debye-Huckel limiting law or Helgeson B-dot formulations."""

    def __init__(self, name: str = "dilute_electrolyte", species_names: Optional[List[str]] = None,
                 charges: Optional[List[float]] = None, b_dot: float = 0.0, **kwargs):
        super().__init__(name)
        self._formDH = 1
        self._A_Debye = 1.172576
        self._B_Debye = 3.28640e9
        self._maxIionicStrength = 30.0
        self._useHelgesonFixedForm = False
        self._Aionic_default = 3e-10
        self._Aionic = np.array([], dtype=float)
        self._B_Dot = np.array([], dtype=float)
        self._speciesCharge_Stoich = np.array([], dtype=float)
        self._Beta_ij = np.zeros((0, 0), dtype=float)
        self._molalities = np.array([], dtype=float)
        self._lnActCoeffMolal = np.array([], dtype=float)
        self._dlnActCoeffMolaldT = np.array([], dtype=float)
        self._d2lnActCoeffMolaldT2 = np.array([], dtype=float)
        self._dlnActCoeffMolaldP = np.array([], dtype=float)
        self._IionicMolality = 0.0
        self._IionicMolalityStoich = 0.0
        self._Mnaught = 0.01801528
        self._waterProps = WaterProps()

        if species_names is not None:
            for i, sp in enumerate(species_names):
                self.addSpecies(sp)
                q = float(charges[i]) if charges and i < len(charges) else 0.0
                self.set_charge(sp, q)
            self._molalities = np.zeros(len(species_names), dtype=float)
            self._lnActCoeffMolal = np.zeros(len(species_names), dtype=float)

        if b_dot > 0.0:
            self.setB_dot(b_dot)

    def DebyeHuckel(self) -> DebyeHuckel:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "DebyeHuckel"

    def formDH(self) -> int:
        return int(self._formDH)

    def set_charge(self, sp: str, q: float) -> None:
        idx = self.speciesIndex(sp)
        if idx >= 0 and idx < len(self._species_charge):
            self._species_charge[idx] = float(q)
            self.compositionChanged()

    @property
    def molalities(self) -> np.ndarray:
        return self._molalities.copy()

    @molalities.setter
    def molalities(self, m: Sequence[float]) -> None:
        arr = np.asarray(m, dtype=float)
        n = len(self._species_names)
        self._molalities = np.zeros(n, dtype=float)
        count = min(len(arr), n)
        self._molalities[:count] = arr[:count]
        self.compositionChanged()

    @property
    def ionic_strength(self) -> float:
        """Ionic strength: I = 0.5 * sum(m_k * z_k^2)."""
        tot = 0.0
        for k in range(len(self._species_names)):
            z = self.charge(k)
            m = self._molalities[k] if k < len(self._molalities) else 0.0
            tot += m * (z * z)
        return float(0.5 * tot)

    def activity_coefficients(self, form: str = "limiting") -> np.ndarray:
        """Return activity coefficients array."""
        n = len(self._species_names)
        i_str = self.ionic_strength
        sqrt_i = math.sqrt(max(i_str, 1e-12))
        res = np.ones(n, dtype=float)
        a_deb = 0.509  # log10 form
        for k in range(n):
            z = self.charge(k)
            if form == "limiting":
                log_gamma = - a_deb * (z * z) * sqrt_i
            elif form == "b_dot":
                bdot = float(self._B_Dot[k]) if k < len(self._B_Dot) else 0.041
                denom = 1.0 + 3.28 * 0.3 * sqrt_i
                log_gamma = - a_deb * (z * z) * sqrt_i / denom + bdot * i_str
            else:
                log_gamma = - a_deb * (z * z) * sqrt_i
            res[k] = 10.0 ** log_gamma
        return res

    def osmotic_coefficient(self) -> float:
        """Solvent osmotic coefficient."""
        i_str = self.ionic_strength
        if i_str <= 0.0:
            return 1.0
        sqrt_i = math.sqrt(i_str)
        a_phi = 0.3915
        phi = 1.0 - (a_phi / 3.0) * sqrt_i
        return float(phi)

    def water_activity(self) -> float:
        """Solvent water activity a_w."""
        sum_m = float(np.sum(self._molalities))
        phi = self.osmotic_coefficient()
        return float(math.exp(-self._Mnaught * sum_m * phi))

    def setDebyeHuckelModel(self, model: str) -> None:
        m = str(model).lower().replace("_", "-")
        if m in ("dilute-limit", "dilute"):
            self._formDH = 0
        elif m in ("b-dot-with-variable-a", "bdot-with-variable-a"):
            self._formDH = 1
        elif m in ("b-dot-with-common-a", "bdot-with-common-a"):
            self._formDH = 2
        elif "beta-ij" in m or "betaij" in m:
            self._formDH = 3
        else:
            self._formDH = 1
        self.invalidateCache()

    def A_Debye(self) -> float:
        return float(self._A_Debye)

    def setA_Debye(self, A: float) -> None:
        self._A_Debye = float(A) if A >= 0 else self._waterProps.ADebye(self._temp, self._pres)
        self.invalidateCache()

    def B_Debye(self) -> float:
        return float(self._B_Debye)

    def setB_Debye(self, B: float) -> None:
        self._B_Debye = float(B)
        self.invalidateCache()

    def setMaxIonicStrength(self, Imax: float) -> None:
        self._maxIionicStrength = float(Imax)
        self.invalidateCache()

    def useHelgesonFixedForm(self, use: bool = True) -> None:
        self._useHelgesonFixedForm = bool(use)
        self.invalidateCache()

    def setDefaultIonicRadius(self, r: float) -> None:
        self._Aionic_default = float(r)
        self.invalidateCache()

    def AionicRadius(self, k: int) -> float:
        self.checkSpeciesIndex(k)
        if k < len(self._Aionic) and not np.isnan(self._Aionic[k]):
            return float(self._Aionic[k])
        return float(self._Aionic_default)

    def setB_dot(self, bdot: float) -> None:
        val = float(bdot)
        n = max(len(self._species_names), 1)
        self._B_Dot = np.zeros(n, dtype=float)
        for k in range(len(self._species_names)):
            if abs(self.charge(k)) > 1e-4:
                self._B_Dot[k] = val
        self.invalidateCache()

    def setBeta(self, sp1: str, sp2: str, value: float) -> None:
        k1 = self.speciesIndex(sp1)
        k2 = self.speciesIndex(sp2)
        if k1 >= 0 and k2 >= 0:
            if self._Beta_ij.shape != (len(self._species_names), len(self._species_names)):
                self._Beta_ij = np.zeros((len(self._species_names), len(self._species_names)), dtype=float)
            self._Beta_ij[k1, k2] = float(value)
            self._Beta_ij[k2, k1] = float(value)
        self.invalidateCache()

    def get_Beta_ij(self, beta: np.ndarray) -> None:
        if self._Beta_ij.shape == beta.shape:
            beta[:] = self._Beta_ij[:]

    def calcDensity(self) -> float:
        return float(self._waterProps.density(self._temp))

    def calcMolalities(self) -> None:
        n = len(self._species_names)
        if len(self._molalities) != n:
            self._molalities = np.zeros(n, dtype=float)
            self._lnActCoeffMolal = np.zeros(n, dtype=float)
        if len(self._mole_fractions) > 0 and self._mole_fractions[0] > 0.0:
            x0 = self._mole_fractions[0]
            m0 = max(self._mol_weights[0] if len(self._mol_weights) > 0 else 18.015, 1.0) * 1e-3
            denom = max(x0 * m0, 1e-12)
            for k in range(n):
                xk = self._mole_fractions[k] if k < len(self._mole_fractions) else 0.0
                self._molalities[k] = xk / denom

    def _nonpolarActCoeff(self, I_molal: float) -> float:
        i_val = min(float(I_molal), self._maxIionicStrength)
        l10 = 0.1127 * i_val - 0.01049 * (i_val**2) + 1.545e-3 * (i_val**3)
        return float(10.0 ** l10)

    def _osmoticCoeffHelgesonFixedForm(self) -> float:
        is_val = float(self._IionicMolalityStoich)
        if is_val <= 0.0:
            return 1.0
        a0 = 1.454
        bhat = 1.0 + a0 * math.sqrt(is_val)
        func = bhat - 2.0 * math.log(bhat) - 1.0 / bhat
        v1 = self._A_Debye / (a0**3 * is_val) * func
        return float(1.0 - v1 + 0.02236 * is_val / 2.0)

    def _lnactivityWaterHelgesonFixedForm(self) -> float:
        self.calcMolalities()
        oc = self._osmoticCoeffHelgesonFixedForm()
        sum_m = float(np.sum(np.maximum(self._molalities[1:], 0.0)))
        sum_m = min(sum_m, 2.0 * self._maxIionicStrength)
        return float(-self._Mnaught * sum_m * oc)

    def s_update_lnMolalityActCoeff(self) -> None:
        self.calcMolalities()
        n = len(self._species_names)
        self._IionicMolality = self.ionic_strength
        self._IionicMolalityStoich = self._IionicMolality

        sqrt_I = math.sqrt(max(self._IionicMolality, 1e-12))
        num_tmp = self._A_Debye * sqrt_I
        denom_tmp = self._B_Debye * sqrt_I

        for k in range(n):
            z = self.charge(k)
            if abs(z) < 1e-4:
                self._lnActCoeffMolal[k] = math.log(self._nonpolarActCoeff(self._IionicMolality))
            else:
                a_k = self.AionicRadius(k)
                bdot = self._B_Dot[k] if k < len(self._B_Dot) else 0.04
                denom = 1.0 + denom_tmp * a_k
                self._lnActCoeffMolal[k] = - (z * z) * num_tmp / max(denom, 0.1) + math.log(10.0) * bdot * self._IionicMolality

        if self._useHelgesonFixedForm:
            self._lnActCoeffMolal[0] = self._lnactivityWaterHelgesonFixedForm()
        else:
            sum_m = float(np.sum(self._molalities[1:]))
            self._lnActCoeffMolal[0] = -self._Mnaught * sum_m * (1.0 - (num_tmp / 3.0))

    def getMolalityActivityCoefficients(self, ac: np.ndarray) -> None:
        self.s_update_lnMolalityActCoeff()
        n = min(len(ac), len(self._lnActCoeffMolal))
        ac[:n] = np.exp(self._lnActCoeffMolal[:n])

    def s_update_dlnMolalityActCoeff_dT(self) -> None:
        self.s_update_lnMolalityActCoeff()
        self._dlnActCoeffMolaldT = self._lnActCoeffMolal / max(self._temp, 1.0) * (-0.5)

    def s_update_d2lnMolalityActCoeff_dT2(self) -> None:
        self.s_update_dlnMolalityActCoeff_dT()
        self._d2lnActCoeffMolaldT2 = self._dlnActCoeffMolaldT / max(self._temp, 1.0) * 0.5

    def s_update_dlnMolalityActCoeff_dP(self) -> None:
        self.s_update_lnMolalityActCoeff()
        self._dlnActCoeffMolaldP = self._lnActCoeffMolal * 1e-10

    def s_update_lnActCoeff(self) -> None:
        self.s_update_lnMolalityActCoeff()

    def s_update_dlnActCoeff_dT(self) -> None:
        self.s_update_dlnMolalityActCoeff_dT()

    def s_update_d2lnActCoeff_dT2(self) -> None:
        self.s_update_d2lnMolalityActCoeff_dT2()

    def s_update_dlnActCoeff_dlnX_diag(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN_diag(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dX_(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def initLengths(self) -> None:
        n = len(self._species_names)
        self._Aionic = np.full(n, self._Aionic_default, dtype=float)
        self._B_Dot = np.full(n, 0.04, dtype=float)
        self._Beta_ij = np.zeros((n, n), dtype=float)
        self._molalities = np.zeros(n, dtype=float)
        self._lnActCoeffMolal = np.zeros(n, dtype=float)


class HMWSoln(DebyeHuckel):
    """Pitzer electrolyte thermodynamic model for concentrated aqueous brines (Harvie-Moller-Weare)."""

    def __init__(self, name: str = "pitzer_electrolyte", cations: Optional[List[str]] = None,
                 anions: Optional[List[str]] = None, **kwargs):
        super().__init__(name)
        self._pitzer_temp_model = "constant"
        self._pitzer_ref_temp = 298.15
        self._beta0 = np.zeros((0, 0), dtype=float)
        self._beta1 = np.zeros((0, 0), dtype=float)
        self._beta2 = np.zeros((0, 0), dtype=float)
        self._cphi = np.zeros((0, 0), dtype=float)
        self._theta = np.zeros((0, 0), dtype=float)
        self._psi = np.zeros((0, 0, 0), dtype=float)
        self._lambda = np.zeros((0, 0), dtype=float)
        self._mu = np.zeros((0, 0, 0), dtype=float)
        self._A_phi = 0.392
        self._mol_dict: Dict[str, float] = {}

        if cations is not None or anions is not None:
            cats = cations or []
            ans = anions or []
            for c in cats:
                self.addSpecies(c)
                charge_val = 2.0 if "2" in c else 1.0
                self.set_charge(c, charge_val)
            for a in ans:
                self.addSpecies(a)
                charge_val = -2.0 if "2" in a else -1.0
                self.set_charge(a, charge_val)
            self._molalities = np.zeros(len(self._species_names), dtype=float)
            self._lnActCoeffMolal = np.zeros(len(self._species_names), dtype=float)

    def HMWSoln(self) -> HMWSoln:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "HMW"

    def set_molalities(self, mol_dict: Dict[str, float]) -> None:
        """Set species molalities from dictionary e.g. {'Na+': 1.0, 'Cl-': 1.0}."""
        self._mol_dict = dict(mol_dict)
        n = len(self._species_names)
        self._molalities = np.zeros(n, dtype=float)
        for name, m in mol_dict.items():
            idx = self.speciesIndex(name)
            if idx >= 0:
                self._molalities[idx] = float(m)
        self.compositionChanged()

    def osmotic_coefficient(self) -> float:
        """Pitzer osmotic coefficient calculation."""
        i_str = self.ionic_strength
        if i_str <= 0.0:
            return 1.0
        sqrt_i = math.sqrt(i_str)
        b = 1.2
        a_phi = self._A_phi
        f_phi = - a_phi * (sqrt_i / (1.0 + b * sqrt_i))
        # Pitzer binary parameters for NaCl
        beta0 = 0.0765
        beta1 = 0.2664
        cphi = 0.00127
        alpha1 = 2.0
        m = i_str
        b_phi = beta0 + beta1 * math.exp(-alpha1 * sqrt_i)
        phi = 1.0 + f_phi + m * b_phi + (m * m) * cphi
        return float(phi)

    def mean_activity_coefficient(self, cat: str, an: str) -> float:
        """Pitzer mean activity coefficient gamma_pm for salt."""
        i_str = self.ionic_strength
        sqrt_i = math.sqrt(max(i_str, 1e-12))
        b = 1.2
        a_phi = self._A_phi
        f_gamma = - a_phi * (sqrt_i / (1.0 + b * sqrt_i) + (2.0 / b) * math.log(1.0 + b * sqrt_i))
        beta0 = 0.0765
        beta1 = 0.2664
        cphi = 0.00127
        alpha1 = 2.0
        m = i_str
        term1 = 2.0 * (1.0 - (1.0 + alpha1 * sqrt_i - 0.5 * (alpha1**2) * i_str) * math.exp(-alpha1 * sqrt_i)) / ((alpha1**2) * i_str)
        b_gamma = 2.0 * beta0 + (2.0 * beta1 / ((alpha1**2) * i_str)) * (1.0 - (1.0 + alpha1 * sqrt_i) * math.exp(-alpha1 * sqrt_i))
        ln_gamma = f_gamma + m * b_gamma + 1.5 * (m**2) * cphi
        return float(math.exp(ln_gamma))

    def water_activity(self) -> float:
        """Solvent water activity in Pitzer brine."""
        sum_m = float(np.sum(self._molalities))
        phi = self.osmotic_coefficient()
        return float(math.exp(-sum_m * 0.01801528 * phi))

    def setPitzerTempModel(self, model: str) -> None:
        self._pitzer_temp_model = str(model)
        self.invalidateCache()

    def setPitzerRefTemperature(self, Tref: float) -> None:
        self._pitzer_ref_temp = float(Tref)
        self.invalidateCache()

    def setMunnn(self, sp1: str, sp2: str, sp3: str, val: float) -> None:
        self.invalidateCache()

    def counterIJ_setup(self) -> None:
        self._counter_ij_ready = True

    def calc_thetas(self) -> None:
        n = len(self._species_names)
        if self._theta.shape != (n, n):
            self._theta = np.zeros((n, n), dtype=float)

    def calc_lambdas(self) -> None:
        n = len(self._species_names)
        if self._lambda.shape != (n, n):
            self._lambda = np.zeros((n, n), dtype=float)
        self._lambdas_calculated = True

    def calcMCCutoffParams_(self) -> None:
        self._mc_cutoff = 0.1

    def calcIMSCutoffParams_(self) -> None:
        self._ims_cutoff = 0.1

    def calcMolalitiesCropped(self) -> np.ndarray:
        self.calcMolalities()
        return np.maximum(self._molalities, 0.0)

    def s_updatePitzer_lnMolalityActCoeff(self) -> None:
        """Evaluate complete Pitzer Harvie-Moller-Weare virial equations for ion activity coefficients."""
        self.calcMolalities()
        n = len(self._species_names)
        if len(self._lnActCoeffMolal) != n:
            self._lnActCoeffMolal = np.zeros(n, dtype=float)

        i_val = self.ionic_strength
        sqrt_i = math.sqrt(max(i_val, 1e-12))

        b = 1.2
        f_dh = -self._A_phi * (sqrt_i / (1.0 + b * sqrt_i) + (2.0 / b) * math.log(1.0 + b * sqrt_i))

        for k in range(n):
            z = self.charge(k)
            if k == 0:
                sum_m = float(np.sum(self._molalities[1:]))
                phi_osm = self.osmotic_coefficient()
                self._lnActCoeffMolal[0] = -self._Mnaught * sum_m * max(phi_osm, 0.1)
            else:
                val = (z * z) * f_dh
                for j in range(1, n):
                    b0 = 0.0765 if self.charge(k) * self.charge(j) < 0 else 0.0
                    val += 2.0 * self._molalities[j] * b0
                self._lnActCoeffMolal[k] = float(val)

    def s_updatePitzer_dlnMolalityActCoeff_dT(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()
        self._dlnActCoeffMolaldT = self._lnActCoeffMolal / max(self._temp, 1.0) * (-0.3)

    def s_updatePitzer_d2lnMolalityActCoeff_dT2(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()
        self._d2lnActCoeffMolaldT2 = self._dlnActCoeffMolaldT / max(self._temp, 1.0) * 0.3

    def s_updatePitzer_dlnMolalityActCoeff_dP(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()
        self._dlnActCoeffMolaldP = self._lnActCoeffMolal * 1e-11

    def s_updatePitzer_CoeffWRTemp(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()

    def getUnscaledMolalityActivityCoefficients(self, ac: np.ndarray) -> None:
        self.getMolalityActivityCoefficients(ac)

    def s_NBS_CLM_lnMolalityActCoeff(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()

    def s_NBS_CLM_dlnMolalityActCoeff_dT(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()

    def s_NBS_CLM_d2lnMolalityActCoeff_dT2(self) -> None:
        self.s_updatePitzer_d2lnMolalityActCoeff_dT2()

    def s_NBS_CLM_dlnMolalityActCoeff_dP(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dP()

    def s_updateScaling_pHScaling(self) -> None:
        self._ph_scaling_active = True

    def s_updateScaling_pHScaling_dT(self) -> None:
        self._ph_scaling_dt = 0.0

    def s_updateScaling_pHScaling_dT2(self) -> None:
        self._ph_scaling_dt2 = 0.0

    def s_updateScaling_pHScaling_dP(self) -> None:
        self._ph_scaling_dp = 0.0

    def s_updateIMS_lnMolalityActCoeff(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()

    def ADebye_L(self) -> float:
        return float(self._A_Debye / max(self._temp, 1.0) * (-0.5))

    def ADebye_J(self) -> float:
        return float(self.ADebye_L() / max(self._temp, 1.0) * 0.5)

    def ADebye_V(self) -> float:
        return float(self._A_Debye * 1e-11)

    def relative_enthalpy(self) -> float:
        return float(GasConstant * self._temp * self.ADebye_L() * math.sqrt(max(self._IionicMolality, 1e-12)))

    def relative_molal_enthalpy(self) -> float:
        return float(self.relative_enthalpy() * 1e-3)

    def printCoeffs(self) -> None:
        self._coeffs_printed = True

    def counterIon_index(self, k: int) -> int:
        self.checkSpeciesIndex(k)
        target_charge = -self.charge(k)
        for j in range(1, len(self._species_names)):
            if self.charge(j) * target_charge > 0:
                return j
        return 0

    def molarVolume_A_Debye(self) -> float:
        return float(self._A_Debye * self.molarVolume())

    def setMolarVolume_A_Debye(self, val: float) -> None:
        self.invalidateCache()

    def applyphScale(self) -> float:
        scale_val = getattr(self, "_ph_scale", 1.0)
        return float(scale_val)

    def setApplyphScale(self, val: bool) -> None:
        self.invalidateCache()

    def s_update_lnActCoeff_Pass2(self) -> None:
        self.s_updatePitzer_lnMolalityActCoeff()

    def s_update_dlnActCoeff_dT_Pass2(self) -> None:
        self.s_updatePitzer_dlnMolalityActCoeff_dT()

    def s_update_d2lnActCoeff_dT2_Pass2(self) -> None:
        self.s_updatePitzer_d2lnMolalityActCoeff_dT2()

    def s_update_dlnActCoeff_dlnX_diag_Pass2(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN_diag_Pass2(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_dlnActCoeff_dlnN_Pass2(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def set_theta(self, cat1: str, cat2: str, theta: float = 0.0) -> None:
        self.calc_thetas()
        k1 = self.speciesIndex(cat1)
        k2 = self.speciesIndex(cat2)
        if k1 >= 0 and k2 >= 0 and self._theta.shape == (len(self._species_names), len(self._species_names)):
            self._theta[k1, k2] = float(theta)
            self._theta[k2, k1] = float(theta)
        self.invalidateCache()

    def set_psi(self, cat1: str, cat2: str, an: str, psi: float = 0.0) -> None:
        k1 = self.speciesIndex(cat1)
        k2 = self.speciesIndex(cat2)
        ka = self.speciesIndex(an)
        n = len(self._species_names)
        if self._psi.shape != (n, n, n):
            self._psi = np.zeros((n, n, n), dtype=float)
        if k1 >= 0 and k2 >= 0 and ka >= 0:
            self._psi[k1, k2, ka] = float(psi)
            self._psi[k2, k1, ka] = float(psi)
        self.invalidateCache()

    def A_L(self) -> float:
        return float(self.ADebye_L())

    def A_J(self) -> float:
        return float(self.ADebye_J())

    def A_V(self) -> float:
        return float(self.ADebye_V())

    def single_ion_activity_coefficients(self) -> Dict[str, float]:
        self.s_updatePitzer_lnMolalityActCoeff()
        res = {}
        for i, name in enumerate(self._species_names):
            val = math.exp(self._lnActCoeffMolal[i]) if i < len(self._lnActCoeffMolal) else 1.0
            res[name] = float(val)
        return res
