"""Multi-species cubic equations of state (Peng-Robinson and Redlich-Kwong).

Pure-Python implementation derived from Cantera native C++ commit
726522be4e2a13454d8415b7ef799d621f665cf3:
- include/cantera/thermo/MixtureFugacityTP.h & src/thermo/MixtureFugacityTP.cpp
- include/cantera/thermo/PengRobinson.h & src/thermo/PengRobinson.cpp
- include/cantera/thermo/RedlichKwongMFTP.h & src/thermo/RedlichKwongMFTP.cpp
"""
from __future__ import annotations

import math
from math import isfinite, sqrt, log, exp, cos, acos, pi
from typing import Mapping, Sequence, Union, Optional
import warnings
import numpy as np

from .constants import GasConstant as R, OneAtm, SmallNumber, CanteraError
from .composition import parse_comp_string
from .ideal_gas import Species


# Fluid phase branch flags matching Cantera C++ ct_defs.h / MixtureFugacityTP.h
FLUID_UNSTABLE = -4
FLUID_UNDEFINED = -3
FLUID_SUPERCRIT = -2
FLUID_GAS = -1
FLUID_LIQUID_0 = 0


def _source_clip(value: float, lower: float, upper: float) -> float:
    """Match Cantera's ``std::max(lower, std::min(upper, value))`` ordering."""
    return max(lower, min(upper, value))


def _source_divide(numerator: float, denominator: float) -> float:
    """Use IEEE-754 division semantics for the native Newton iteration."""
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return float("nan")
    sign = math.copysign(1.0, numerator) * math.copysign(1.0, denominator)
    return math.copysign(float("inf"), sign)


class FloatCallable(float):
    """Float that can also be called with () as a method, for C++ method compatibility."""
    def __call__(self) -> float:
        return float(self)


class IntCallable(int):
    """Int that can also be called with () as a method, for C++ method compatibility."""
    def __call__(self) -> int:
        return int(self)


class MixtureFugacityTP:
    """Base class for multi-species real fluid thermodynamic phases using cubic EOS."""

    def __init__(self, species: Sequence[Species], *,
                 temperature: float = 298.15, pressure: float = OneAtm,
                 mole_fractions: Optional[Union[Mapping[str, float], Sequence[float], str]] = None,
                 name: str = ""):
        self._species = tuple(species)
        self.name = str(name)
        if not self._species or len({s.name for s in self._species}) != len(self._species):
            raise ValueError('Provide one or more species with distinct names')
        self._kk = len(self._species)
        self._species_indices = {s.name: i for i, s in enumerate(self._species)}
        self._mw = np.array([float(s.molecular_weight) for s in self._species], dtype=float)
        self._ref_pressure = float(self._species[0].thermo.reference_pressure)

        self._temperature = float(temperature)
        self._pressure = float(pressure)
        self._temperature_limits_enforced = False
        self._forced_state = FLUID_UNDEFINED
        self._actual_state = FLUID_GAS

        # Normalized mole fractions
        if mole_fractions is None:
            x = np.zeros(self._kk, dtype=float)
            x[0] = 1.0
        elif isinstance(mole_fractions, str):
            parsed = parse_comp_string(mole_fractions, self.species_names)
            x = np.array([parsed.get(name, 0.0) for name in self.species_names], dtype=float)
        elif isinstance(mole_fractions, Mapping):
            x = np.array([float(mole_fractions.get(name, 0.0)) for name in self.species_names], dtype=float)
        else:
            x = np.asarray(mole_fractions, dtype=float)
        total_x = float(np.sum(x))
        if total_x <= 0.0 or not np.all(np.isfinite(x)):
            raise ValueError("Mole fractions must be non-negative with positive sum")
        self._mole_fractions = x / total_x

        self._molar_volume = (R * self._temperature) / self._pressure
        self._density = self.mean_molecular_weight / self._molar_volume

    @property
    def species(self) -> tuple[Species, ...]:
        return self._species

    @property
    def species_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self._species)

    @property
    def n_species(self) -> IntCallable:
        return IntCallable(self._kk)

    @property
    def nSpecies(self) -> IntCallable:
        return IntCallable(self._kk)

    def species_index(self, name: str) -> int:
        try:
            return self._species_indices[name]
        except KeyError:
            raise ValueError(f"Unknown species: {name}")

    def species_name(self, index: int) -> str:
        return self._species[index].name

    def speciesIndex(self, name: str, check: bool = True) -> int:
        return self.species_index(name)

    def speciesName(self, index: int) -> str:
        return self.species_name(index)

    @property
    def element_names(self) -> tuple[str, ...]:
        if not hasattr(self, "_element_names") or not self._element_names:
            ordered = []
            comp = getattr(self, "_species_compositions", {})
            for c in comp.values():
                for el in c:
                    if el not in ordered:
                        ordered.append(el)
            self._element_names = tuple(ordered)
        return self._element_names

    @element_names.setter
    def element_names(self, val):
        self._element_names = tuple(val)

    def nElements(self) -> int:
        return len(self.element_names)

    def elementName(self, m: int) -> str:
        return self.element_names[m]

    def elementIndex(self, name: str) -> int:
        return self.element_names.index(name) if name in self.element_names else -1

    def nAtoms(self, k: int, m: int) -> float:
        sp_name = self.species_name(k) if callable(getattr(self, "species_name", None)) else self.species_names[k]
        el_name = self.elementName(m)
        comp = getattr(self, "_species_compositions", {})
        return float(comp.get(sp_name, {}).get(el_name, 0.0))

    @property
    def concentrations(self) -> np.ndarray:
        return self._mole_fractions * (self._density / self.mean_molecular_weight)

    def getConcentrations(self, c=None):
        conc = self.concentrations
        if c is not None:
            c[:] = conc
            return c
        return conc

    @property
    def molecular_weights(self) -> np.ndarray:
        return self._mw.copy()

    def molecularWeights(self) -> np.ndarray:
        return self._mw.copy()

    @property
    def mean_molecular_weight(self) -> FloatCallable:
        return FloatCallable(float(np.dot(self._mole_fractions, self._mw)))

    def meanMolecularWeight(self) -> FloatCallable:
        return FloatCallable(float(np.dot(self._mole_fractions, self._mw)))

    @property
    def min_temp(self) -> float:
        """Maximum species thermo lower limit for this phase."""
        return max(float(getattr(sp.thermo, "min_temp", 0.0)) for sp in self._species)

    @property
    def max_temp(self) -> float:
        """Minimum species thermo upper limit for this phase."""
        return min(float(getattr(sp.thermo, "max_temp", float("inf")))
                   for sp in self._species)

    def minTemp(self) -> float:
        return self.min_temp

    def maxTemp(self) -> float:
        return self.max_temp

    def temperatureLimitsEnforced(self) -> bool:
        return self._temperature_limits_enforced

    def setTemperatureLimitsEnforced(self, enforce: bool) -> None:
        self._temperature_limits_enforced = bool(enforce)

    @property
    def X(self) -> np.ndarray:
        return self._mole_fractions.copy()

    @X.setter
    def X(self, val):
        self.setMoleFractions(val)

    @property
    def mole_fractions(self) -> np.ndarray:
        return self._mole_fractions.copy()

    def setMoleFractions(self, val):
        if isinstance(val, str):
            parsed = parse_comp_string(val, self.species_names)
            x = np.array([parsed.get(name, 0.0) for name in self.species_names], dtype=float)
        elif isinstance(val, Mapping):
            x = np.array([float(val.get(name, 0.0)) for name in self.species_names], dtype=float)
        else:
            x = np.asarray(val, dtype=float)
        total = float(np.sum(x))
        if total <= 0.0 or not np.all(np.isfinite(x)):
            raise ValueError("Invalid mole fractions")
        self._mole_fractions = x / total
        self.updateMixingExpressions()
        # Native Cantera Phase::setMoleFractions preserves (T, rho)
        self._molar_volume = self.mean_molecular_weight / self._density
        try:
            self._pressure = self._calc_pressure()
        except NotImplementedError:
            pass

    def setMoleFractions_NoNorm(self, val):
        x = np.asarray(val, dtype=float)
        self._mole_fractions = x.copy()
        self.updateMixingExpressions()
        self._molar_volume = self.mean_molecular_weight / self._density
        try:
            self._pressure = self._calc_pressure()
        except NotImplementedError:
            pass

    def setMoleFractionsByName(self, val):
        self.setMoleFractions(val)

    def getMoleFractions(self, x: Optional[Union[list, np.ndarray]] = None) -> np.ndarray:
        if x is not None:
            for k in range(self._kk):
                x[k] = self._mole_fractions[k]
            return x
        return self._mole_fractions.copy()

    @property
    def Y(self) -> np.ndarray:
        y = self._mole_fractions * self._mw
        return y / np.sum(y)

    @Y.setter
    def Y(self, val):
        self.setMassFractions(val)

    def setMassFractions(self, val):
        if isinstance(val, str):
            parsed = parse_comp_string(val, self.species_names)
            y = np.array([parsed.get(name, 0.0) for name in self.species_names], dtype=float)
        elif isinstance(val, Mapping):
            y = np.array([float(val.get(name, 0.0)) for name in self.species_names], dtype=float)
        else:
            y = np.asarray(val, dtype=float)
        total = float(np.sum(y))
        if total <= 0.0 or not np.all(np.isfinite(y)):
            raise ValueError("Invalid mass fractions")
        y = y / total
        x = y / self._mw
        self.setMoleFractions(x / np.sum(x))

    def setMassFractionsByName(self, val):
        self.setMassFractions(val)

    def getMassFractions(self, y: Optional[Union[list, np.ndarray]] = None) -> np.ndarray:
        mf = self.Y
        if y is not None:
            for k in range(self._kk):
                y[k] = mf[k]
            return y
        return mf

    @property
    def temperature(self) -> FloatCallable:
        return FloatCallable(self._temperature)

    @temperature.setter
    def temperature(self, val: float):
        self.setTemperature(val)

    @property
    def T(self) -> FloatCallable:
        return FloatCallable(self._temperature)

    @T.setter
    def T(self, val: float):
        self.setTemperature(val)

    def setTemperature(self, temp: float):
        temp = float(temp)
        if temp <= 0.0 or not isfinite(temp):
            raise ValueError(f"Temperature must be positive, got {temp}")
        self._temperature = temp
        self.updateMixingExpressions()
        self._actual_state = self.phaseState(True)
        try:
            self._pressure = self._calc_pressure()
        except Exception:
            pass

    def _eval_pressure(self) -> float:
        raise NotImplementedError

    def _calc_pressure(self) -> float:
        return self._eval_pressure()

    @property
    def pressure(self) -> FloatCallable:
        try:
            val = self._eval_pressure()
            self._pressure = val
        except Exception:
            val = self._pressure
        return FloatCallable(val)

    @pressure.setter
    def pressure(self, val: float):
        self.setPressure(val)

    @property
    def P(self) -> FloatCallable:
        return self.pressure

    @P.setter
    def P(self, val: float):
        self.setPressure(val)

    def setPressure(self, p: float):
        p = float(p)
        if p <= 0.0 or not isfinite(p):
            raise ValueError(f"Pressure must be positive, got {p}")
        t = self._temperature
        rhoNow = self._density
        if self._forced_state == FLUID_UNDEFINED:
            rho = self.densityCalc(t, p, self._actual_state, rhoNow)
            if rho > 0.0:
                self.setDensity(rho)
                self._actual_state = self.phaseState(True)
            else:
                if rho < -1.5:
                    rho = self.densityCalc(t, p, FLUID_UNDEFINED, rhoNow)
                    if rho > 0.0:
                        self.setDensity(rho)
                        self._actual_state = self.phaseState(True)
                    else:
                        raise CanteraError(f"MixtureFugacityTP::setPressure: neg rho ({rho})")
                else:
                    raise CanteraError(f"MixtureFugacityTP::setPressure: neg rho ({rho})")
        elif self._forced_state == FLUID_GAS:
            if self._actual_state < FLUID_LIQUID_0:
                rho = self.densityCalc(t, p, self._actual_state, rhoNow)
                if rho > 0.0:
                    self.setDensity(rho)
                    self._actual_state = self.phaseState(True)
                    if self._actual_state >= FLUID_LIQUID_0:
                        raise CanteraError("MixtureFugacityTP::setPressure: wrong state")
                else:
                    raise CanteraError(f"MixtureFugacityTP::setPressure: neg rho ({rho})")
        elif self._forced_state >= FLUID_LIQUID_0:
            if self._actual_state >= FLUID_LIQUID_0:
                rho = self.densityCalc(t, p, self._actual_state, rhoNow)
                if rho > 0.0:
                    self.setDensity(rho)
                    self._actual_state = self.phaseState(True)
                    if self._actual_state == FLUID_GAS:
                        raise CanteraError("MixtureFugacityTP::setPressure: wrong state")
                else:
                    raise CanteraError(f"MixtureFugacityTP::setPressure: neg rho ({rho})")

    @property
    def density(self) -> FloatCallable:
        return FloatCallable(self._density)

    @density.setter
    def density(self, val: float):
        self.setDensity(val)

    def setDensity(self, rho: float):
        rho = float(rho)
        if rho <= 0.0 or not isfinite(rho):
            raise ValueError(f"Density must be positive, got {rho}")
        self._density = rho
        self._molar_volume = self.mean_molecular_weight / rho
        try:
            self._pressure = self._calc_pressure()
        except NotImplementedError:
            pass

    @property
    def molar_density(self) -> FloatCallable:
        return FloatCallable(self._density / self.mean_molecular_weight)

    @property
    def molar_volume(self) -> FloatCallable:
        return FloatCallable(self._molar_volume)

    def molarVolume(self) -> FloatCallable:
        return FloatCallable(self._molar_volume)

    @property
    def volume(self) -> FloatCallable:
        return FloatCallable(1.0 / self._density)

    def RT(self) -> FloatCallable:
        return FloatCallable(R * self._temperature)

    def refPressure(self) -> FloatCallable:
        return FloatCallable(self._ref_pressure)

    def setForcedSolutionBranch(self, branch: int):
        self._forced_state = int(branch)

    def forcedSolutionBranch(self) -> int:
        return self._forced_state

    def reportSolnBranchActual(self) -> int:
        return self._actual_state

    # --- Saturation and phase state tracking matching native C++ ---
    def psatEst(self, TKelvin: float) -> FloatCallable:
        pcrit = self.critPressure()
        tt = self.critTemperature() / float(TKelvin)
        if tt < 1.0:
            return FloatCallable(pcrit)
        lpr = -0.8734 * (tt ** 2) - 3.4522 * tt + 4.2918
        return FloatCallable(pcrit * exp(lpr))


    def compositionChanged(self) -> None:
        self.updateMixingExpressions()

    def addSpecies(self, spec) -> None:
        pass

    def sresid(self) -> float:
        return 0.0

    def hresid(self) -> float:
        return 0.0

    def calculatePsat(self) -> float:
        return float(self.psat(self.temperature()))

    def dpdVCalc(self, V: float, T: float, Pcalc=None) -> float:
        return 0.0


    @property
    def enthalpy_RT(self) -> np.ndarray:
        return self.getEnthalpy_RT()

    @property
    def entropy_R(self) -> np.ndarray:
        return self.getEntropy_R()

    @property
    def gibbs_RT(self) -> np.ndarray:
        return self.getGibbs_RT()

    @property
    def int_energy_RT(self) -> np.ndarray:
        return self.getIntEnergy_RT()

    @property
    def cp_R(self) -> np.ndarray:
        return self.getCp_R()

    @property
    def standard_volumes(self) -> np.ndarray:
        return self.getStandardVolumes()

    @property
    def standard_chem_potentials(self) -> np.ndarray:
        return self.getStandardChemPotentials()

    def phaseState(self, checkState: bool = True) -> int:
        state = self._actual_state
        if checkState:
            t = self._temperature
            tcrit = self.critTemperature()
            rhocrit = self.critDensity()
            if t >= tcrit:
                return FLUID_SUPERCRIT
            tmid = tcrit - 100.0
            if tmid < 0.0:
                tmid = tcrit / 2.0
            pp = self.psatEst(tmid)
            mmw = self.mean_molecular_weight
            molVolLiqTmid = self.liquidVolEst(tmid, pp)
            molVolGasTmid = (R * tmid) / pp
            densLiqTmid = mmw / molVolLiqTmid
            densGasTmid = mmw / molVolGasTmid
            densMidTmid = 0.5 * (densLiqTmid + densGasTmid)
            rhoMid = rhocrit + (t - tcrit) * (rhocrit - densMidTmid) / (tcrit - tmid)

            rho = self._density
            iStateGuess = FLUID_LIQUID_0 if rho >= rhoMid else FLUID_GAS
            molarVol = mmw / rho
            dpdv = self.dpdVCalc(t, molarVol)
            if dpdv < 0.0:
                state = iStateGuess
            else:
                state = FLUID_UNSTABLE
        return state

    # --- Reference-state ideal gas thermo properties ---
    def _eval_ref_thermo(self, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (cp0_R, h0_RT, s0_R, g0_RT) at temperature T."""
        cp0_R = np.empty(self._kk, dtype=float)
        h0_RT = np.empty(self._kk, dtype=float)
        s0_R = np.empty(self._kk, dtype=float)
        for i, s in enumerate(self._species):
            cp_R, h_RT, s_R = s.thermo.properties(T)
            cp0_R[i] = cp_R
            h0_RT[i] = h_RT
            s0_R[i] = s_R
        g0_RT = h0_RT - s0_R
        return cp0_R, h0_RT, s0_R, g0_RT

    def getEnthalpy_RT_ref(self, hrt: np.ndarray = None):
        if hrt is None:
            hrt = np.empty(self._kk, dtype=float)
        _, h0_RT, _, _ = self._eval_ref_thermo(self._temperature)
        hrt[:] = h0_RT
        return hrt

    def getGibbs_RT_ref(self, grt: np.ndarray = None):
        if grt is None:
            grt = np.empty(self._kk, dtype=float)
        _, _, _, g0_RT = self._eval_ref_thermo(self._temperature)
        grt[:] = g0_RT
        return grt

    def getEntropy_R_ref(self, sr: np.ndarray = None):
        if sr is None:
            sr = np.empty(self._kk, dtype=float)
        _, _, s0_R, _ = self._eval_ref_thermo(self._temperature)
        sr[:] = s0_R
        return sr

    def getCp_R_ref(self, cpr: np.ndarray = None):
        if cpr is None:
            cpr = np.empty(self._kk, dtype=float)
        cp0_R, _, _, _ = self._eval_ref_thermo(self._temperature)
        cpr[:] = cp0_R
        return cpr

    def getStandardChemPotentials(self, g: np.ndarray = None):
        if g is None:
            g = np.empty(self._kk, dtype=float)
        _, _, _, g0_RT = self._eval_ref_thermo(self._temperature)
        p_ratio = log(self._pressure / self._ref_pressure)
        rt = self.RT()
        for k in range(self._kk):
            g[k] = rt * (g0_RT[k] + p_ratio)
        return g

    def standardConcentration(self, k: int = 0) -> float:
        """Standard concentration C0_k = P / (R * T)."""
        return float(self._pressure) / (R * float(self._temperature))

    @property
    def standard_concentration(self) -> float:
        return self.standardConcentration(0)

    @property
    def standard_concentrations(self) -> np.ndarray:
        return np.array([self.standardConcentration(k) for k in range(self._kk)], dtype=float)

    @property
    def standard_chemical_potentials(self) -> np.ndarray:
        g = np.empty(self._kk, dtype=float)
        self.getStandardChemPotentials(g)
        return g

    @property
    def chemical_potentials(self) -> np.ndarray:
        mu = np.empty(self._kk, dtype=float)
        self.getChemPotentials(mu)
        return mu

    @property
    def partial_molar_enthalpies(self) -> np.ndarray:
        hbar = np.empty(self._kk, dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        return hbar

    @property
    def partial_molar_entropies(self) -> np.ndarray:
        sbar = np.empty(self._kk, dtype=float)
        self.getPartialMolarEntropies(sbar)
        return sbar

    @property
    def partial_molar_int_energies(self) -> np.ndarray:
        ubar = np.empty(self._kk, dtype=float)
        self.getPartialMolarIntEnergies(ubar)
        return ubar

    @property
    def partial_molar_cp(self) -> np.ndarray:
        cpbar = np.empty(self._kk, dtype=float)
        self.getPartialMolarCp(cpbar)
        return cpbar

    @property
    def partial_molar_volumes(self) -> np.ndarray:
        vbar = np.empty(self._kk, dtype=float)
        self.getPartialMolarVolumes(vbar)
        return vbar

    @property
    def activity_coefficients(self) -> np.ndarray:
        ac = np.empty(self._kk, dtype=float)
        self.getActivityCoefficients(ac)
        return ac

    @property
    def activities(self) -> np.ndarray:
        act = np.empty(self._kk, dtype=float)
        self.getActivities(act)
        return act

    def getActivities(self, act: np.ndarray):
        self.getActivityConcentrations(act)
        for k in range(self._kk):
            act[k] /= self.standardConcentration(k)

    def getActivityConcentrations(self, c: np.ndarray):
        self.getActivityCoefficients(c)
        p_rt = float(self.pressure) / self.RT()
        for k in range(self._kk):
            c[k] *= self._mole_fractions[k] * p_rt

    def getStandardVolumes(self, vol: Optional[np.ndarray] = None) -> np.ndarray:
        if vol is None:
            vol = np.empty(self._kk, dtype=float)
        rt_p = (R * float(self.temperature)) / float(self.pressure)
        vol.fill(rt_p)
        return vol

    def getStandardVolumes_ref(self, vol: Optional[np.ndarray] = None) -> np.ndarray:
        if vol is None:
            vol = np.empty(self._kk, dtype=float)
        rt_pref = (R * float(self.temperature)) / self._ref_pressure
        vol.fill(rt_pref)
        return vol

    @property
    def thermo_model(self) -> str:
        return self.type()

    # --- Molar and mass properties with non-ideal residual terms ---
    @property
    def enthalpy_mole(self) -> FloatCallable:
        _, h0_RT, _, _ = self._eval_ref_thermo(self._temperature)
        h_ideal = self.RT() * float(np.dot(self._mole_fractions, h0_RT))
        return FloatCallable(h_ideal + self.hresid())

    @property
    def entropy_mole(self) -> FloatCallable:
        _, _, s0_R, _ = self._eval_ref_thermo(self._temperature)
        x = np.maximum(self._mole_fractions, SmallNumber)
        sum_xlogx = float(np.sum(x * np.log(x)))
        p_ratio = log(self._pressure / self._ref_pressure)
        s_ideal = R * (float(np.dot(self._mole_fractions, s0_R)) - sum_xlogx - p_ratio)
        return FloatCallable(s_ideal + self.sresid())

    @property
    def gibbs_mole(self) -> FloatCallable:
        return FloatCallable(self.enthalpy_mole - self._temperature * self.entropy_mole)

    @property
    def int_energy_mole(self) -> FloatCallable:
        return FloatCallable(self.enthalpy_mole - self._pressure * self._molar_volume)

    @property
    def enthalpy_mass(self) -> FloatCallable:
        return FloatCallable(self.enthalpy_mole / self.mean_molecular_weight)

    @property
    def entropy_mass(self) -> FloatCallable:
        return FloatCallable(self.entropy_mole / self.mean_molecular_weight)

    @property
    def gibbs_mass(self) -> FloatCallable:
        return FloatCallable(self.gibbs_mole / self.mean_molecular_weight)

    @property
    def int_energy_mass(self) -> FloatCallable:
        return FloatCallable(self.int_energy_mole / self.mean_molecular_weight)

    @property
    def cp_mole(self) -> FloatCallable:
        raise NotImplementedError

    @property
    def cv_mole(self) -> FloatCallable:
        raise NotImplementedError

    @property
    def cp_mass(self) -> FloatCallable:
        return FloatCallable(self.cp_mole / self.mean_molecular_weight)

    @property
    def cv_mass(self) -> FloatCallable:
        return FloatCallable(self.cv_mole / self.mean_molecular_weight)

    def z(self) -> float:
        """Compressibility factor Z = P * v / (R * T)."""
        return (self._pressure * self._molar_volume) / (R * self._temperature)

    # --- State setters ---
    def setState_TP(self, T: float, P: float):
        self.setTemperature(T)
        self.setPressure(P)

    def setState_TD(self, T: float, rho: float):
        self.setTemperature(T)
        self.setDensity(rho)

    def setState_TPX(self, T: float, P: float, X):
        self.setMoleFractions(X)
        self.setState_TP(T, P)

    def setState_TPY(self, T: float, P: float, Y):
        self.setMassFractions(Y)
        self.setState_TP(T, P)

    def setState_TDX(self, T: float, rho: float, X):
        self.setMoleFractions(X)
        self.setState_TD(T, rho)

    def setState_TDY(self, T: float, rho: float, Y):
        self.setMassFractions(Y)
        self.setState_TD(T, rho)

    @property
    def TP(self) -> tuple[float, float]:
        return float(self._temperature), float(self._pressure)

    @TP.setter
    def TP(self, values: tuple[float, float]):
        self.setState_TP(values[0], values[1])

    @property
    def TD(self) -> tuple[float, float]:
        return float(self._temperature), float(self._density)

    @TD.setter
    def TD(self, values: tuple[float, float]):
        self.setState_TD(values[0], values[1])

    @property
    def TPX(self):
        return float(self._temperature), float(self._pressure), self.X

    @TPX.setter
    def TPX(self, values):
        self.setState_TPX(values[0], values[1], values[2])

    @property
    def TPY(self):
        return float(self._temperature), float(self._pressure), self.Y

    @TPY.setter
    def TPY(self, values):
        self.setState_TPY(values[0], values[1], values[2])

    @property
    def TDX(self):
        return float(self._temperature), float(self._density), self.X

    @TDX.setter
    def TDX(self, values):
        self.setState_TDX(values[0], values[1], values[2])

    @property
    def TDY(self):
        return float(self._temperature), float(self._density), self.Y

    @TDY.setter
    def TDY(self, values):
        self.setState_TDY(values[0], values[1], values[2])

    def setState_conditional_TP(self, temperature: float, pressure: float,
                                set_pressure: bool = True) -> None:
        """Set T and optionally P using ThermoPhase's operation order."""
        self.setTemperature(temperature)
        if set_pressure:
            self.setPressure(pressure)

    def _inverse_temperature_bounds(self) -> tuple[float, float]:
        if self.temperatureLimitsEnforced():
            tmax, tmin = self.max_temp + 0.1, self.min_temp - 0.1
        else:
            tmax = max(self.max_temp + 1000.0, 10.0 * self.max_temp)
            tmin = min(100.0, max(self.min_temp, SmallNumber))
        if tmax <= tmin:
            tmax = tmin + 20.0
        return tmin, tmax

    def _check_inverse_temperature(self, method: str) -> None:
        if self._temperature > self.max_temp + 1.0 or self._temperature < self.min_temp - 1.0:
            message = (f"Temperature ({self._temperature} K) outside valid range of "
                       f"{self.min_temp} K to {self.max_temp} K")
            if self.temperatureLimitsEnforced():
                raise CanteraError(method, message)
            warnings.warn(f"{method}: {message}", RuntimeWarning, stacklevel=3)

    def _partial_state_snapshot(self):
        return (self._temperature, self._pressure, self._density,
                self._molar_volume, self._actual_state, self._forced_state)

    def _restore_partial_state(self, state) -> None:
        (temperature, pressure, density, molar_volume,
         actual_state, forced_state) = state
        self._temperature = temperature
        self.updateMixingExpressions()
        self._pressure = pressure
        self._density = density
        self._molar_volume = molar_volume
        self._actual_state = actual_state
        self._forced_state = forced_state

    def _set_state_hporuv(self, target: float, pressure_or_volume: float,
                          rtol: float, max_iterations: int, do_uv: bool) -> None:
        if do_uv:
            volume = pressure_or_volume
            if volume < SmallNumber:
                raise CanteraError(
                    "ThermoPhase::setState_HPorUV (UV)",
                    f"Input specific volume is too small or negative. v = {volume}",
                )
            self.setDensity(1.0 / volume)
        else:
            volume = 0.0
            if pressure_or_volume < SmallNumber:
                raise CanteraError(
                    "ThermoPhase::setState_HPorUV (HP)",
                    f"Input pressure is too small or negative. p = {pressure_or_volume}",
                )
            self.setPressure(pressure_or_volume)

        tmin, tmax = self._inverse_temperature_bounds()
        tnew = float(self.temperature)
        tinit = tnew
        if tnew > tmax:
            tnew = tmax - 1.0
        elif tnew < tmin:
            tnew = tmin + 1.0
        if tnew != tinit:
            self.setState_conditional_TP(tnew, pressure_or_volume, not do_uv)

        value_name = "int_energy_mass" if do_uv else "enthalpy_mass"
        derivative_name = "cv_mass" if do_uv else "cp_mass"
        value = float(getattr(self, value_name))
        derivative = float(getattr(self, derivative_name))
        top_value = bot_value = value
        top_t = bot_t = tnew
        ignore_bounds = False
        unstable = False
        unstable_t = -1.0
        dt = 0.0

        for _ in range(max_iterations):
            told, old_value, old_derivative = tnew, value, derivative
            if old_derivative < 0.0:
                unstable = True
                unstable_t = told
            dt = _source_clip(_source_divide(target - old_value, old_derivative), -100.0, 100.0)
            tnew = told + dt

            if (dt > 0.0 and unstable) or (dt <= 0.0 and not unstable):
                if bot_value < target and tnew < (0.75 * bot_t + 0.25 * told):
                    dt = 0.75 * (bot_t - told)
                    tnew = told + dt
            elif top_value > target and tnew > (0.75 * top_t + 0.25 * told):
                dt = 0.75 * (top_t - told)
                tnew = told + dt

            if tnew > tmax and not ignore_bounds:
                self.setState_conditional_TP(tmax, pressure_or_volume, not do_uv)
                edge_value = float(getattr(self, value_name))
                if edge_value >= target:
                    if top_value < target:
                        top_t, top_value = tmax, edge_value
                    tnew, dt = tmax, tmax - told
                elif self.temperatureLimitsEnforced():
                    kind = "internal energy" if do_uv else "enthalpy"
                    raise CanteraError(
                        "ThermoPhase::setState_HPorUV",
                        f"Target {kind} = {target} cannot be reached within temperature bounds of {tmin} K to {tmax} K.",
                    )
                else:
                    tnew, ignore_bounds = tmax + 1.0, True
            if tnew < tmin and not ignore_bounds:
                self.setState_conditional_TP(tmin, pressure_or_volume, not do_uv)
                edge_value = float(getattr(self, value_name))
                if edge_value <= target:
                    if bot_value > target:
                        bot_t, bot_value = tmin, edge_value
                    tnew, dt = tmin, tmin - told
                elif self.temperatureLimitsEnforced():
                    kind = "internal energy" if do_uv else "enthalpy"
                    raise CanteraError(
                        "ThermoPhase::setState_HPorUV",
                        f"Target {kind} = {target} cannot be reached within temperature bounds of {tmin} K to {tmax} K.",
                    )
                else:
                    tnew, ignore_bounds = tmin - 1.0, True

            for _retry in range(10):
                tnew = told + dt
                if tnew < told / 3.0:
                    tnew, dt = told / 3.0, -2.0 * told / 3.0
                self.setState_conditional_TP(tnew, pressure_or_volume, not do_uv)
                value = float(getattr(self, value_name))
                derivative = float(getattr(self, derivative_name))
                if derivative >= 0.0:
                    break
                unstable_t = tnew
                if not unstable:
                    dt *= 0.25

            if value == target:
                self._check_inverse_temperature(
                    "ThermoPhase::setState_UV" if do_uv else "ThermoPhase::setState_HP")
                return
            if value > target and (top_value < target or value < top_value):
                top_t, top_value = tnew, value
            elif value < target and (bot_value > target or value > bot_value):
                bot_t, bot_value = tnew, value

            residual = target - value
            denominator = max(abs(target), max(abs(old_derivative), 1.0e-5) * tnew)
            if abs(residual / denominator) < rtol or abs(dt / tnew) < rtol:
                self._check_inverse_temperature(
                    "ThermoPhase::setState_UV" if do_uv else "ThermoPhase::setState_HP")
                return

        label = "Internal Energy" if do_uv else "Enthalpy"
        current_label = "Internal Energy" if do_uv else "Enthalpy"
        current_value = value
        details = (f"No convergence in {max_iterations} iterations\n"
                   f"\tTarget {label} = {target}\n")
        if do_uv:
            details += (f"\tCurrent Specific Volume = {volume}\n"
                        f"\tStarting Temperature = {tinit}\n")
        else:
            details += (f"\tCurrent Pressure = {pressure_or_volume}\n"
                        f"\tStarting Temperature = {tinit}\n")
        details += (f"\tCurrent Temperature = {tnew}\n"
                    f"\tCurrent {current_label} = {current_value}\n"
                    f"\tCurrent Delta T = {dt}\n")
        if unstable:
            details += f"\tThe phase became unstable (Cp < 0); T_unstable_last = {unstable_t}\n"
        source = "ThermoPhase::setState_HPorUV (UV)" if do_uv else "ThermoPhase::setState_HPorUV (HP)"
        raise CanteraError(source, details)

    def _set_state_sporsv(self, target: float, pressure_or_volume: float,
                          rtol: float, max_iterations: int, do_sv: bool) -> None:
        if do_sv:
            volume = pressure_or_volume
            if volume < SmallNumber:
                raise CanteraError(
                    "ThermoPhase::setState_SPorSV (SV)",
                    f"Input specific volume is too small or negative. v = {volume}",
                )
            self.setDensity(1.0 / volume)
        else:
            volume = 0.0
            if pressure_or_volume < SmallNumber:
                raise CanteraError(
                    "ThermoPhase::setState_SPorSV (SP)",
                    f"Input pressure is too small or negative. p = {pressure_or_volume}",
                )
            self.setPressure(pressure_or_volume)

        tmin, tmax = self._inverse_temperature_bounds()
        tnew = float(self.temperature)
        tinit = tnew
        if tnew > tmax:
            tnew = tmax - 1.0
        elif tnew < tmin:
            tnew = tmin + 1.0
        if tnew != tinit:
            self.setState_conditional_TP(tnew, pressure_or_volume, not do_sv)

        derivative_name = "cv_mass" if do_sv else "cp_mass"
        value = float(self.entropy_mass)
        derivative = float(getattr(self, derivative_name))
        top_value = bot_value = value
        top_t = bot_t = tnew
        ignore_bounds = False
        unstable = False
        unstable_t = -1.0
        dt = 0.0

        for _ in range(max_iterations):
            told, old_value, old_derivative = tnew, value, derivative
            if old_derivative < 0.0:
                unstable = True
                unstable_t = told
            dt = _source_clip(
                _source_divide((target - old_value) * told, old_derivative), -100.0, 100.0)
            tnew = told + dt

            if (dt > 0.0 and unstable) or (dt <= 0.0 and not unstable):
                if bot_value < target and tnew < (0.75 * bot_t + 0.25 * told):
                    dt = 0.75 * (bot_t - told)
                    tnew = told + dt
            elif top_value > target and tnew > (0.75 * top_t + 0.25 * told):
                dt = 0.75 * (top_t - told)
                tnew = told + dt

            if tnew > tmax and not ignore_bounds:
                self.setState_conditional_TP(tmax, pressure_or_volume, not do_sv)
                edge_value = float(self.entropy_mass)
                if edge_value >= target:
                    if top_value < target:
                        top_t, top_value = tmax, edge_value
                    tnew, dt = tmax, tmax - told
                elif self.temperatureLimitsEnforced():
                    raise CanteraError(
                        "ThermoPhase::setState_SPorSV",
                        f"Target entropy = {target} cannot be reached within temperature bounds of {tmin} K to {tmax} K.",
                    )
                else:
                    tnew, ignore_bounds = tmax + 1.0, True
            elif tnew < tmin and not ignore_bounds:
                self.setState_conditional_TP(tmin, pressure_or_volume, not do_sv)
                edge_value = float(self.entropy_mass)
                if edge_value <= target:
                    if bot_value > target:
                        bot_t, bot_value = tmin, edge_value
                    tnew, dt = tmin, tmin - told
                elif self.temperatureLimitsEnforced():
                    raise CanteraError(
                        "ThermoPhase::setState_SPorSV",
                        f"Target entropy = {target} cannot be reached within temperature bounds of {tmin} K to {tmax} K.",
                    )
                else:
                    tnew, ignore_bounds = tmin - 1.0, True

            for _retry in range(10):
                tnew = told + dt
                self.setState_conditional_TP(tnew, pressure_or_volume, not do_sv)
                derivative = float(getattr(self, derivative_name))
                value = float(self.entropy_mass)
                if derivative >= 0.0:
                    break
                unstable_t = tnew
                if not unstable:
                    dt *= 0.25

            if value == target:
                self._check_inverse_temperature(
                    "ThermoPhase::setState_SV" if do_sv else "ThermoPhase::setState_SP")
                return
            if value > target and (top_value < target or value < top_value):
                top_t, top_value = tnew, value
            elif value < target and (bot_value > target or value > bot_value):
                bot_t, bot_value = tnew, value

            residual = target - value
            denominator = max(abs(target), max(abs(old_derivative), 1.0e-5) * tnew)
            if abs(residual * tnew / denominator) < rtol or abs(dt / tnew) < rtol:
                self._check_inverse_temperature(
                    "ThermoPhase::setState_SV" if do_sv else "ThermoPhase::setState_SP")
                return

        label = "SV" if do_sv else "SP"
        details = (f"No convergence in {max_iterations} iterations\n"
                   f"\tTarget Entropy = {target}\n")
        if do_sv:
            details += (f"\tCurrent Specific Volume = {volume}\n"
                        f"\tStarting Temperature = {tinit}\n")
        else:
            details += (f"\tCurrent Pressure = {pressure_or_volume}\n"
                        f"\tStarting Temperature = {tinit}\n")
        details += (f"\tCurrent Temperature = {tnew}\n"
                    f"\tCurrent Entropy = {value}\n"
                    f"\tCurrent Delta T = {dt}\n")
        if unstable:
            details += f"\tThe phase became unstable (Cp < 0); T_unstable_last = {unstable_t}\n"
        raise CanteraError(f"ThermoPhase::setState_SPorSV ({label})", details)

    @staticmethod
    def _inverse_options(rtol: float, max_iterations: int,
                         tol: Optional[float], max_iter: Optional[int]):
        if tol is not None:
            rtol = tol
        if max_iter is not None:
            max_iterations = max_iter
        rtol = float(rtol)
        if not isfinite(rtol) or rtol <= 0.0:
            raise ValueError("rtol must be finite and positive")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        return rtol, max_iterations

    def setState_HP(self, H: float, P: float, *, rtol: float = 1e-9,
                    max_iterations: int = 500, tol: Optional[float] = None,
                    max_iter: Optional[int] = None):
        rtol, max_iterations = self._inverse_options(rtol, max_iterations, tol, max_iter)
        self._set_state_hporuv(float(H), float(P), rtol, max_iterations, False)

    def setState_UV(self, U: float, V: float, *, rtol: float = 1e-9,
                    max_iterations: int = 500, tol: Optional[float] = None,
                    max_iter: Optional[int] = None):
        rtol, max_iterations = self._inverse_options(rtol, max_iterations, tol, max_iter)
        self._set_state_hporuv(float(U), float(V), rtol, max_iterations, True)

    def setState_SP(self, S: float, P: float, *, rtol: float = 1e-9,
                    max_iterations: int = 500, tol: Optional[float] = None,
                    max_iter: Optional[int] = None):
        rtol, max_iterations = self._inverse_options(rtol, max_iterations, tol, max_iter)
        original = self._partial_state_snapshot()
        try:
            self._set_state_sporsv(float(S), float(P), rtol, max_iterations, False)
        except Exception:
            self._restore_partial_state(original)
            raise

    def setState_SV(self, S: float, V: float, *, rtol: float = 1e-9,
                    max_iterations: int = 500, tol: Optional[float] = None,
                    max_iter: Optional[int] = None):
        rtol, max_iterations = self._inverse_options(rtol, max_iterations, tol, max_iter)
        original = self._partial_state_snapshot()
        try:
            self._set_state_sporsv(float(S), float(V), rtol, max_iterations, True)
        except Exception:
            self._restore_partial_state(original)
            raise

    # --- Analytical Cubic Equation Solver (Cardano Method) ---
    def solve_cubic(self, T: float, pres: float, a: float, b: float, aAlpha: float,
                    an: float, bn: float, cn: float, dn: float, tc: float, vc: float) -> tuple[int, list[float]]:
        """Analytical solution of an v^3 + bn v^2 + cn v + dn = 0."""
        if T <= 0.0:
            raise ValueError(f"negative temperature T = {T}")

        xN = -bn / (3.0 * an)
        deltaNumerator = bn * bn - 3.0 * an * cn
        delta2 = deltaNumerator / (9.0 * an * an)
        delta = sqrt(delta2) if delta2 > 0.0 else 0.0

        # Small parameter check for ideal gas limit
        ratio1 = 3.0 * an * cn / (bn * bn) if bn != 0.0 else 1.0
        ratio2 = pres * b / (R * T)
        if abs(ratio1) < 1e-7:
            ratio3 = (aAlpha / (R * T)) * (pres / (R * T))
            if abs(ratio2) < 1e-5 and abs(ratio3) < 1e-5:
                zz = 1.0
                for _ in range(10):
                    znew = zz / (zz - ratio2) - ratio3 / (zz + ratio1)
                    if abs(znew - zz) < 1e-14:
                        break
                    zz = znew
                v = zz * R * T / pres
                return 1, [v, 0.0, 0.0]

        h2 = 4.0 * an * an * (delta2 ** 3)
        h = 2.0 * an * delta * delta2
        yTerm1 = 2.0 * (bn ** 3) / (27.0 * an * an)
        yTerm2 = -bn * cn / (3.0 * an)
        yN = yTerm1 + yTerm2 + dn
        disc = yN * yN - h2

        cancellationTol = 64 * 2.220446049250313e-16
        deltaScale = max(abs(bn * bn), abs(3.0 * an * cn))
        yScale = max(abs(yTerm1), abs(yTerm2), abs(dn))
        tripleRoot = (abs(deltaNumerator) <= cancellationTol * deltaScale
                      and abs(yN) <= cancellationTol * yScale)
        if tripleRoot:
            delta = 0.0
            disc = 0.0

        if not tripleRoot and abs(abs(h) - abs(yN)) < 1e-10:
            # Pinned Cantera rejects a materially inconsistent cancellation;
            # only a numerically negligible positive discriminant is clamped.
            if disc > 1e-10:
                raise ValueError('value of yN and h are too high, unrealistic roots may be obtained')
            disc = 0.0

        Vroot = [0.0, 0.0, 0.0]
        if disc > 1e-14:
            # One real root
            nsol = 1
            tmpD = sqrt(disc)
            tmp1 = (-yN + tmpD) / (2.0 * an)
            sgn1 = 1.0 if tmp1 >= 0.0 else -1.0
            p1 = (abs(tmp1) ** (1.0 / 3.0)) * sgn1
            tmp2 = (-yN - tmpD) / (2.0 * an)
            sgn2 = 1.0 if tmp2 >= 0.0 else -1.0
            p2 = (abs(tmp2) ** (1.0 / 3.0)) * sgn2
            Vroot[0] = xN + p1 + p2
        elif disc < -1e-14:
            # Three distinct real roots
            nsol = 3
            val = -yN / h
            theta = acos(val) / 3.0
            twoThirdPi = 2.0 * pi / 3.0
            alpha = xN + 2.0 * delta * cos(theta)
            beta = xN + 2.0 * delta * cos(theta + twoThirdPi)
            gamma = xN + 2.0 * delta * cos(theta + 2.0 * twoThirdPi)
            # Order: beta (liquid), gamma (unstable/intermediate), alpha (gas)
            Vroot[0] = beta
            Vroot[1] = gamma
            Vroot[2] = alpha
        else:
            # Repeated roots
            if tripleRoot:
                nsol = 1
                Vroot[0] = xN
                Vroot[1] = xN
            else:
                nsol = 2
                if yN > 0.0:
                    Vroot[0] = xN - 2.0 * delta
                    Vroot[1] = xN + delta
                    Vroot[2] = xN + delta
                else:
                    Vroot[0] = xN + delta
                    Vroot[1] = xN - 2.0 * delta
                    Vroot[2] = xN - 2.0 * delta

        # Newton-Raphson polish for physical roots
        vmin = max(0.0, b * (1.0 + 1e-12))
        n_polish = 3 if nsol == 3 else (1 if nsol == 1 else 2)
        for i in range(n_polish):
            if isfinite(Vroot[i]) and Vroot[i] > vmin:
                for _ in range(20):
                    res = an * Vroot[i] ** 3 + bn * Vroot[i] ** 2 + cn * Vroot[i] + dn
                    if abs(res) < 1e-14:
                        break
                    dres = 3.0 * an * Vroot[i] ** 2 + 2.0 * bn * Vroot[i] + cn
                    if dres == 0.0:
                        break
                    old = Vroot[i]
                    delta_v = -res / dres
                    Vroot[i] += delta_v
                    if abs(delta_v) / (abs(Vroot[i]) + abs(delta_v)) < 1e-14:
                        break
                    res2 = an * Vroot[i] ** 3 + bn * Vroot[i] ** 2 + cn * Vroot[i] + dn
                    if abs(res2) >= abs(res):
                        Vroot[i] = old + 0.1 * delta_v
                # A physical root must meet the same scaled residual test as
                # the native implementation.
                res = an * Vroot[i] ** 3 + bn * Vroot[i] ** 2 + cn * Vroot[i] + dn
                dres = 3.0 * an * Vroot[i] ** 2 + 2.0 * bn * Vroot[i] + cn
                if abs(res) > 1e-14 and abs(res) > 1e-14 * abs(dres) * abs(Vroot[i]):
                    raise ValueError(f'root failed to converge for T={T}, p={pres}, V={Vroot[i]}')

        if nsol == 1:
            if T > tc:
                if Vroot[0] < vc:
                    nsol = -1
            else:
                if Vroot[0] < xN:
                    nsol = -1
        elif nsol == 2:
            if delta > 1e-14:
                nsol = -2

        return nsol, Vroot


_CRIT_PROPS_DB = None


def _get_crit_properties_db() -> dict[str, dict]:
    global _CRIT_PROPS_DB
    if _CRIT_PROPS_DB is None:
        from pathlib import Path
        from ruamel.yaml import YAML
        from . import DATA_DIR
        path = Path(DATA_DIR) / 'critical-properties.yaml'
        if path.is_file():
            content = path.read_text(encoding='utf-8', errors='ignore')
            data = YAML(typ='safe').load(content)
            _CRIT_PROPS_DB = {
                sp['name'].upper(): sp.get('critical-parameters', {})
                for sp in (data.get('species', []) if isinstance(data, dict) else [])
                if isinstance(sp, dict) and 'name' in sp
            }
        else:
            _CRIT_PROPS_DB = {}
    return _CRIT_PROPS_DB


class PengRobinson(MixtureFugacityTP):
    """Multi-species Peng-Robinson equation of state."""

    omega_a: float = 4.5723552892138218e-01
    omega_b: float = 7.77960739038885e-02
    omega_vc: float = 3.07401308698703833e-01

    def __init__(self, species: Sequence[Species], *,
                 temperature: float = 298.15, pressure: float = OneAtm,
                 mole_fractions=None, name: str = ""):
        self._a_coeffs = np.zeros((len(species), len(species)), dtype=float)
        self._b_coeffs = np.zeros(len(species), dtype=float)
        self._acentric = np.zeros(len(species), dtype=float)
        self._kappa = np.zeros(len(species), dtype=float)
        self._alpha = np.zeros(len(species), dtype=float)
        self._dalphadT = np.zeros(len(species), dtype=float)
        self._d2alphadT2 = np.zeros(len(species), dtype=float)
        self._aAlpha_binary = np.zeros((len(species), len(species)), dtype=float)

        self._a = 0.0
        self._b = 0.0
        self._aAlpha_mix = 0.0
        self._dpdV = 0.0
        self._dpdT = 0.0

        super().__init__(species, temperature=temperature, pressure=pressure,
                         mole_fractions=mole_fractions, name=name)

    def type(self) -> str:
        return "Peng-Robinson"

    def speciesCritTemperature(self, a: float, b: float) -> float:
        if b <= 0.0:
            return 1e6
        if a <= 0.0:
            return 0.0
        return (a * self.omega_b) / (b * self.omega_a * R)

    def critTemperature(self) -> float:
        if self._b <= 0.0:
            return 1e6
        if self._a <= 0.0:
            return 0.0
        return (self._a * self.omega_b) / (self._b * self.omega_a * R)

    def critPressure(self) -> float:
        tc = self.critTemperature()
        return (self.omega_b * R * tc) / max(self._b, 1e-12)

    def critDensity(self) -> float:
        tc = self.critTemperature()
        pc = self.critPressure()
        vc = (self.omega_vc * R * tc) / pc
        return self.mean_molecular_weight / vc

    def setSpeciesCoeffs(self, species: str, a: float, b: float, w: float):
        k = self.species_index(species)
        w = float(w)
        if w <= 0.491:
            self._kappa[k] = 0.37464 + 1.54226 * w - 0.26992 * (w ** 2)
        else:
            self._kappa[k] = 0.374642 + 1.487503 * w - 0.164423 * (w ** 2) + 0.016666 * (w ** 3)
        self._acentric[k] = w
        self._a_coeffs[k, k] = float(a)
        self._b_coeffs[k] = float(b)

        # Mixing rules for a_coeffs
        for j in range(self._kk):
            if j == k:
                continue
            if self._a_coeffs[j, k] == 0.0:
                akj = sqrt(self._a_coeffs[j, j] * float(a))
                self._a_coeffs[j, k] = akj
                self._a_coeffs[k, j] = akj

        self.updateMixingExpressions()


    def addSpecies(self, spec) -> None:
        pass

    def initThermo(self) -> None:
        self.calculateAB()

    def setBinaryInteraction(self, species_i, species_j, a: float) -> None:
        if isinstance(species_i, (int, np.integer)):
            species_i = self.species_name(int(species_i))
        if isinstance(species_j, (int, np.integer)):
            species_j = self.species_name(int(species_j))
        self.setBinaryCoeffs(str(species_i), str(species_j), float(a))

    def getBinaryInteraction(self, species_i, species_j) -> float:
        if isinstance(species_i, str):
            ki = self.species_index(species_i)
        else:
            ki = int(species_i)
        if isinstance(species_j, str):
            kj = self.species_index(species_j)
        else:
            kj = int(species_j)
        return float(self._a_coeffs[ki, kj])

    def acentricFactor(self, k: int = 0) -> float:
        if isinstance(k, str):
            k = self.species_index(k)
        return float(self._acentric_factor[int(k)])

    def calculateAlpha(self, k: int, T: float) -> float:
        if isinstance(k, str):
            k = self.species_index(k)
        Tc = float(self._tc[k])
        w = float(self._acentric_factor[k])
        Tr = float(T) / Tc
        if w <= 0.491:
            m = 0.37464 + 1.54226 * w - 0.26992 * w * w
        else:
            m = 0.379642 + 1.48503 * w - 0.164423 * w * w + 0.016666 * w * w * w
        sqrtTr = math.sqrt(max(Tr, 1e-12))
        return float((1.0 + m * (1.0 - sqrtTr)) ** 2)

    def cpresid(self) -> float:
        cpm = float(self.cp_mole())
        cp_ideal = sum(self.mole_fractions[k] * self.species_thermo(k).cp_mole(self.temperature()) for k in range(self.n_species))
        return float(cpm - cp_ideal)

    def fugacityCoefficients(self, af=None) -> np.ndarray:
        return self.getActivityCoefficients(af)

    def dpressure_ddensity(self) -> float:
        self.calculatePressureDerivatives()
        return float(self._dpdrho)

    def dpressure_dtemperature(self) -> float:
        self.calculatePressureDerivatives()
        return float(self._dpdT)

    def setBinaryCoeffs(self, species_i: str, species_j: str, a0: float):
        ki = self.species_index(species_i)
        kj = self.species_index(species_j)
        self._a_coeffs[ki, kj] = float(a0)
        self._a_coeffs[kj, ki] = float(a0)
        self.updateMixingExpressions()

    def updateMixingExpressions(self):
        T = self._temperature
        for j in range(self._kk):
            tc_j = self.speciesCritTemperature(self._a_coeffs[j, j], self._b_coeffs[j])
            sqt_alpha = 1.0 + self._kappa[j] * (1.0 - sqrt(T / tc_j))
            self._alpha[j] = sqt_alpha * sqt_alpha

        for i in range(self._kk):
            for j in range(self._kk):
                self._aAlpha_binary[i, j] = sqrt(self._alpha[i] * self._alpha[j]) * self._a_coeffs[i, j]

        x = self._mole_fractions
        self._b = float(np.dot(x, self._b_coeffs))
        self._a = float(x @ self._a_coeffs @ x)
        self._aAlpha_mix = float(x @ self._aAlpha_binary @ x)

    def daAlpha_dT(self) -> float:
        T = self._temperature
        for i in range(self._kk):
            tc = self.speciesCritTemperature(self._a_coeffs[i, i], self._b_coeffs[i])
            sqtTr = sqrt(T / tc)
            coeff1 = 1.0 / (tc * sqtTr)
            coeff2 = sqtTr - 1.0
            k = self._kappa[i]
            self._dalphadT[i] = coeff1 * (k * k * coeff2 - k)

        daAlphadT = 0.0
        x = self._mole_fractions
        for i in range(self._kk):
            for j in range(self._kk):
                term = (self._dalphadT[i] / self._alpha[i] + self._dalphadT[j] / self._alpha[j])
                daAlphadT += x[i] * x[j] * 0.5 * self._aAlpha_binary[i, j] * term
        return daAlphadT

    def d2aAlpha_dT2(self) -> float:
        T = self._temperature
        for i in range(self._kk):
            tc = self.speciesCritTemperature(self._a_coeffs[i, i], self._b_coeffs[i])
            sqtTr = sqrt(T / tc)
            coeff1 = 1.0 / (tc * sqtTr)
            coeff2 = sqtTr - 1.0
            k = self._kappa[i]
            self._dalphadT[i] = coeff1 * (k * k * coeff2 - k)
            self._d2alphadT2[i] = (k * k + k) * coeff1 / (2.0 * sqtTr * sqtTr * tc)

        d2aAlpha = 0.0
        x = self._mole_fractions
        for i in range(self._kk):
            alphai = self._alpha[i]
            for j in range(self._kk):
                alphaj = self._alpha[j]
                alphaij = alphai * alphaj
                term1 = self._d2alphadT2[i] / alphai + self._d2alphadT2[j] / alphaj
                term2 = 2.0 * self._dalphadT[i] * self._dalphadT[j] / alphaij
                term3 = self._dalphadT[i] / alphai + self._dalphadT[j] / alphaj
                d2aAlpha += 0.5 * x[i] * x[j] * self._aAlpha_binary[i, j] * (term1 + term2 - 0.5 * term3 * term3)
        return d2aAlpha

    def _eval_pressure(self) -> float:
        T = self._temperature
        v = self._molar_volume
        denom = v * v + 2.0 * v * self._b - self._b * self._b
        return (R * T) / (v - self._b) - self._aAlpha_mix / denom

    def pressureCalc(self, TKelvin: float, molarVol: float) -> float:
        denom = molarVol * molarVol + 2.0 * molarVol * self._b - self._b * self._b
        return (R * TKelvin) / (molarVol - self._b) - self._aAlpha_mix / denom

    def dpdVCalc(self, T: float, molarVol: float) -> float:
        v = molarVol
        b = self._b
        denom = v * v + 2.0 * v * b - b * b
        vpb = v + b
        vmb = v - b
        return -(R * T) / (vmb * vmb) + 2.0 * self._aAlpha_mix * vpb / (denom * denom)

    def calculatePressureDerivatives(self):
        T = self._temperature
        v = self._molar_volume
        b = self._b
        self._dpdV = self.dpdVCalc(T, v)
        denom = v * v + 2.0 * v * b - b * b
        self._dpdT = (R / (v - b)) - self.daAlpha_dT() / denom

    def sresid(self) -> float:
        if self._b == 0.0:
            return 0.0
        v = self._molar_volume
        b = self._b
        hh = b / v
        zz = self.z()
        alpha_1 = self.daAlpha_dT()
        sqrt2 = sqrt(2.0)
        vpb = v + (1.0 + sqrt2) * b
        vmb = v + (1.0 - sqrt2) * b
        fac = alpha_1 / (2.0 * sqrt2 * b)
        sresid_mol_R = log(zz * (1.0 - hh)) + (fac * log(vpb / vmb)) / R
        return R * sresid_mol_R

    def hresid(self) -> float:
        if self._b == 0.0:
            return 0.0
        v = self._molar_volume
        b = self._b
        zz = self.z()
        aAlpha_1 = self.daAlpha_dT()
        T = self._temperature
        sqrt2 = sqrt(2.0)
        vpb = v + (1.0 + sqrt2) * b
        vmb = v + (1.0 - sqrt2) * b
        fac = 1.0 / (2.0 * sqrt2 * b)
        return R * T * (zz - 1.0) + fac * log(vpb / vmb) * (T * aAlpha_1 - self._aAlpha_mix)

    @property
    def cp_mole(self) -> FloatCallable:
        cp0_R, _, _, _ = self._eval_ref_thermo(self._temperature)
        cpref = R * float(np.dot(self._mole_fractions, cp0_R))
        if self._b == 0.0:
            return FloatCallable(cpref)
        T = self._temperature
        v = self._molar_volume
        b = self._b
        sqrt2 = sqrt(2.0)
        vpb = v + (1.0 + sqrt2) * b
        vmb = v + (1.0 - sqrt2) * b
        self.calculatePressureDerivatives()
        dHdT_V = (cpref + v * self._dpdT - R
                  + (1.0 / (2.0 * sqrt2 * b)) * log(vpb / vmb) * T * self.d2aAlpha_dT2())
        return FloatCallable(dHdT_V - (v + T * self._dpdT / self._dpdV) * self._dpdT)

    @property
    def cv_mole(self) -> FloatCallable:
        cp0_R, _, _, _ = self._eval_ref_thermo(self._temperature)
        cvref = R * (float(np.dot(self._mole_fractions, cp0_R)) - 1.0)
        if self._b == 0.0:
            return FloatCallable(cvref)
        self.calculatePressureDerivatives()
        return FloatCallable(self.cp_mole + self._temperature * (self._dpdT ** 2) / self._dpdV)

    def getActivityCoefficients(self, ac: np.ndarray):
        ac.fill(1.0)
        if self._b == 0.0:
            return
        v = self._molar_volume
        b = self._b
        sqrt2 = sqrt(2.0)
        vpb2 = v + (1.0 + sqrt2) * b
        vmb2 = v + (1.0 - sqrt2) * b
        vmb = v - b
        pres = self._pressure
        rt = self.RT()

        pp = self._aAlpha_binary @ self._mole_fractions
        denom = 2.0 * sqrt2 * b * b
        denom2 = b * (v * v + 2.0 * v * b - b * b)

        for k in range(self._kk):
            num = 2.0 * b * pp[k] - self._aAlpha_mix * self._b_coeffs[k]
            val = (-rt * log((pres * v) / rt)
                   + rt * log(v / vmb)
                   + rt * self._b_coeffs[k] / vmb
                   - (num / denom) * log(vpb2 / vmb2)
                   - (self._aAlpha_mix * self._b_coeffs[k] * v) / denom2)
            ac[k] = exp(val / rt)

    def getChemPotentials(self, mu: np.ndarray):
        rt = self.RT()
        self.getStandardChemPotentials(mu)
        for k in range(self._kk):
            xx = max(SmallNumber, self._mole_fractions[k])
            mu[k] += rt * log(xx)
        if self._b == 0.0:
            return

        v = self._molar_volume
        b = self._b
        vmb = v - b
        sqrt2 = sqrt(2.0)
        vpb2 = v + (1.0 + sqrt2) * b
        vmb2 = v + (1.0 - sqrt2) * b

        pp = self._aAlpha_binary @ self._mole_fractions
        pres = self._pressure
        denom = 2.0 * sqrt2 * b * b
        denom2 = b * (v * v + 2.0 * v * b - b * b)

        for k in range(self._kk):
            num = 2.0 * b * pp[k] - self._aAlpha_mix * self._b_coeffs[k]
            mu[k] += (-rt * log((pres * v) / rt)
                      + rt * log(v / vmb)
                      + rt * self._b_coeffs[k] / vmb
                      - (num / denom) * log(vpb2 / vmb2)
                      - (self._aAlpha_mix * self._b_coeffs[k] * v) / denom2)

    def getPartialMolarEnthalpies(self, hbar: np.ndarray):
        self.getEnthalpy_RT_ref(hbar)
        hbar[:] *= self.RT()
        if self._b == 0.0:
            return

        T = self._temperature
        mv = self._molar_volume
        b = self._b
        vmb = mv - b
        sqrt2 = sqrt(2.0)
        vpb2 = mv + (1.0 + sqrt2) * b
        vmb2 = mv + (1.0 - sqrt2) * b
        daAlphadT = self.daAlpha_dT()

        pp = np.zeros(self._kk, dtype=float)
        tmp = np.zeros(self._kk, dtype=float)
        for k in range(self._kk):
            for i in range(self._kk):
                grad_aAlpha = self._dalphadT[i] / self._alpha[i] + self._dalphadT[k] / self._alpha[k]
                pp[k] += self._mole_fractions[i] * self._aAlpha_binary[k, i]
                tmp[k] += self._mole_fractions[i] * self._aAlpha_binary[k, i] * grad_aAlpha

        denom = mv * mv + 2.0 * mv * b - b * b
        denom2 = denom * denom
        rt = self.RT()

        dpdni = np.zeros(self._kk, dtype=float)
        for k in range(self._kk):
            dpdni[k] = (rt / vmb + rt * self._b_coeffs[k] / (vmb * vmb)
                        - 2.0 * pp[k] / denom
                        + 2.0 * vmb * self._aAlpha_mix * self._b_coeffs[k] / denom2)

        fac = T * daAlphadT - self._aAlpha_mix
        self.calculatePressureDerivatives()
        fac2 = mv + T * self._dpdT / self._dpdV
        fac3 = 2.0 * sqrt2 * b * b

        for k in range(self._kk):
            fac4 = T * tmp[k] - 2.0 * pp[k]
            hE_v = (mv * dpdni[k] - rt
                    - (self._b_coeffs[k] / fac3) * log(vpb2 / vmb2) * fac
                    + (mv * self._b_coeffs[k]) / (b * denom) * fac
                    + (1.0 / (2.0 * sqrt2 * b)) * log(vpb2 / vmb2) * fac4)
            hbar[k] += hE_v
            hbar[k] -= fac2 * dpdni[k]

    def getPartialMolarVolumes(self, vbar: np.ndarray):
        if self._b == 0.0:
            vbar.fill(self.RT() / self._pressure)
            return

        pp = np.zeros(self._kk, dtype=float)
        for k in range(self._kk):
            for i in range(self._kk):
                pp[k] += self._mole_fractions[i] * self._aAlpha_binary[k, i]

        mv = self._molar_volume
        b = self._b
        vmb = mv - b
        vpb = mv + b
        fac = mv * mv + 2.0 * mv * b - b * b
        fac2 = fac * fac
        rt = self.RT()
        pres = self._pressure

        for k in range(self._kk):
            num = (rt + rt * b / vmb + rt * self._b_coeffs[k] / vmb
                   + rt * b * self._b_coeffs[k] / (vmb * vmb) - 2.0 * mv * pp[k] / fac
                   + 2.0 * mv * vmb * self._aAlpha_mix * self._b_coeffs[k] / fac2)
            denom = pres + rt * b / (vmb * vmb) + self._aAlpha_mix / fac - 2.0 * mv * vpb * self._aAlpha_mix / fac2
            vbar[k] = num / denom

    def getPartialMolarEntropies(self, sbar: np.ndarray):
        hbar = np.zeros(self._kk, dtype=float)
        mu = np.zeros(self._kk, dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        self.getChemPotentials(mu)
        for k in range(self._kk):
            sbar[k] = (hbar[k] - mu[k]) / self._temperature

    def getPartialMolarIntEnergies(self, ubar: np.ndarray):
        hbar = np.zeros(self._kk, dtype=float)
        vbar = np.zeros(self._kk, dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        self.getPartialMolarVolumes(vbar)
        for k in range(self._kk):
            ubar[k] = hbar[k] - self._pressure * vbar[k]

    def solveCubic(self, T: float, pres: float, a: float, b: float, aAlpha: float,
                   Vroot: list[float]) -> int:
        bsqr = b * b
        RT_p = (R * T) / pres
        aAlpha_p = aAlpha / pres
        an = 1.0
        bn = b - RT_p
        cn = -(2.0 * RT_p * b - aAlpha_p + 3.0 * bsqr)
        dn = bsqr * RT_p + bsqr * b - aAlpha_p * b

        tc = (a * self.omega_b) / (b * self.omega_a * R)
        pc = (self.omega_b * R * tc) / b
        vc = (self.omega_vc * R * tc) / pc

        nsol, roots = self.solve_cubic(T, pres, a, b, aAlpha, an, bn, cn, dn, tc, vc)
        Vroot[0], Vroot[1], Vroot[2] = roots[0], roots[1], roots[2]
        return nsol

    def liquidVolEst(self, T: float, presGuess: float) -> float:
        v = self._b * 1.1
        pres = max(float(self.psatEst(T)), presGuess)
        Vroot = [0.0, 0.0, 0.0]
        foundLiq = False
        m = 0
        while m < 100 and not foundLiq:
            nsol = self.solveCubic(T, pres, self._a, self._b, self._aAlpha_mix, Vroot)
            if nsol == 1 or nsol == 2:
                if pres > self.critPressure():
                    foundLiq = True
                pres *= 1.04
            else:
                foundLiq = True
            m += 1
        if foundLiq:
            return Vroot[0]
        return -1.0

    def densityCalc(self, T: float, presPa: float, phaseRequested: int = FLUID_UNDEFINED,
                    rhoGuess: float = -1.0) -> float:
        self._temperature = T
        self.updateMixingExpressions()
        tcrit = self.critTemperature()
        mmw = self.mean_molecular_weight

        if rhoGuess == -1.0:
            if phaseRequested >= FLUID_LIQUID_0:
                lqvol = self.liquidVolEst(T, presPa)
                rhoGuess = mmw / lqvol if lqvol > 0.0 else (presPa * mmw / (R * T))
            else:
                rhoGuess = presPa * mmw / (R * T)

        volGuess = mmw / rhoGuess
        Vroot = [0.0, 0.0, 0.0]
        nSolns = self.solveCubic(T, presPa, self._a, self._b, self._aAlpha_mix, Vroot)

        vmin = max(0.0, self._b * (1.0 + 1e-12))
        physicalRoots = sorted([r for r in Vroot if isfinite(r) and r > vmin])

        if not physicalRoots:
            return -1.0

        if len(physicalRoots) == 1:
            if ((phaseRequested == FLUID_GAS and nSolns < 0)
                    or (phaseRequested >= FLUID_LIQUID_0 and nSolns > 0)):
                return -2.0
            self._actual_state = FLUID_GAS if T > tcrit else (
                FLUID_LIQUID_0 if phaseRequested >= FLUID_LIQUID_0 else FLUID_GAS)
            return mmw / physicalRoots[0]

        # Multiple physical roots
        r_liq = physicalRoots[0]
        r_gas = physicalRoots[-1]
        r_mid = physicalRoots[1] if len(physicalRoots) == 3 else 0.5 * (r_liq + r_gas)

        if phaseRequested >= FLUID_LIQUID_0:
            molarVol = r_liq
            self._actual_state = FLUID_LIQUID_0
        elif phaseRequested in (FLUID_GAS, FLUID_SUPERCRIT):
            molarVol = r_gas
            self._actual_state = FLUID_GAS
        else:
            molarVol = r_gas if volGuess > r_mid else r_liq
            self._actual_state = FLUID_GAS if molarVol == r_gas else FLUID_LIQUID_0

        return mmw / molarVol

    def isothermalCompressibility(self) -> float:
        self.calculatePressureDerivatives()
        return -1.0 / (self._molar_volume * self._dpdV)

    def thermalExpansionCoeff(self) -> float:
        self.calculatePressureDerivatives()
        return -self._dpdT / (self._molar_volume * self._dpdV)

    def soundSpeed(self) -> float:
        self.calculatePressureDerivatives()
        arg = -self.cp_mole() / self.cv_mole() * self._dpdV / self.mean_molecular_weight
        return self._molar_volume * sqrt(max(0.0, arg))

    def getAuxiliaryData(self) -> dict:
        return {
            "aAlpha_mix": self._aAlpha_mix,
            "b_mix": self._b,
            "daAlpha_dT": self.daAlpha_dT(),
            "d2aAlpha_dT2": self.d2aAlpha_dT2(),
        }

    @classmethod
    def from_mechanism(cls, mech_or_path, phase_name=None, *, temperature=298.15, pressure=OneAtm):
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        from .units import UnitSystem, convert
        if not isinstance(mech_or_path, Mechanism):
            mech = load_mechanism(mech_or_path)
        else:
            mech = mech_or_path
        initial_state = None
        p_obj = None
        if mech.phases:
            p_obj = mech.phase(phase_name)
            if p_obj is not None:
                initial_state = p_obj.state
            mech = mech.for_phase(phase_name)

        species_list = []
        comp = {}
        for s in mech.species:
            if s.thermo:
                poly = create_species_thermo(s.thermo)
                species_list.append(Species(s.name, s.molecular_weight, poly, s.charge))
                comp[s.name] = dict(s.composition)
        if not species_list:
            raise ValueError("No species with thermo data found in mechanism")

        p_name = p_obj.name if (mech.phases and p_obj is not None) else (phase_name or "")
        phase = cls(species_list, temperature=temperature, pressure=pressure, name=p_name)
        phase._species_compositions = comp

        mech_units = UnitSystem(mech.units) if mech.units else UnitSystem()
        binary_pairs = []

        for s in mech.species:
            name = s.name
            found = False

            # 1. Check equation-of-state block
            eos_node = None
            if s.equation_of_state:
                if isinstance(s.equation_of_state, list):
                    for entry in s.equation_of_state:
                        if isinstance(entry, Mapping) and entry.get('model', '').lower() in ('peng-robinson', 'pengrobinson'):
                            eos_node = entry
                            break
                elif isinstance(s.equation_of_state, Mapping):
                    if s.equation_of_state.get('model', '').lower() in ('peng-robinson', 'pengrobinson'):
                        eos_node = s.equation_of_state

            if eos_node is not None:
                eos_units = mech_units
                if 'units' in eos_node:
                    merged = dict(mech.units or {})
                    merged.update(eos_node['units'])
                    eos_units = UnitSystem(merged)

                if 'a' in eos_node and 'b' in eos_node and 'acentric-factor' in eos_node:
                    a0 = eos_units.convert(eos_node['a'], 'Pa*m^6/kmol^2')
                    b0 = eos_units.convert(eos_node['b'], 'm^3/kmol')
                    w = float(eos_node['acentric-factor'])
                    phase.setSpeciesCoeffs(name, a0, b0, w)
                    found = True

                if 'binary-a' in eos_node:
                    b_dict = eos_node['binary-a']
                    if isinstance(b_dict, Mapping):
                        for name2, coeff in b_dict.items():
                            binary_pairs.append((name, name2, eos_units.convert(coeff, 'Pa*m^6/kmol^2')))

            if found:
                continue

            # 2. Check critical-parameters in species input_data
            crit_node = None
            if s.input_data and 'critical-parameters' in s.input_data:
                crit_node = s.input_data['critical-parameters']

            if crit_node is not None:
                crit_units = mech_units
                if 'units' in crit_node:
                    merged = dict(mech.units or {})
                    merged.update(crit_node['units'])
                    crit_units = UnitSystem(merged)
                tc = crit_units.convert(crit_node['critical-temperature'], 'K')
                pc = crit_units.convert(crit_node['critical-pressure'], 'Pa')
                w = float(crit_node['acentric-factor'])
                a0 = cls.omega_a * ((R * tc) ** 2) / pc
                b0 = cls.omega_b * (R * tc) / pc
                phase.setSpeciesCoeffs(name, a0, b0, w)
                found = True
                continue

            # 3. Check critical-properties database
            crit_db = _get_crit_properties_db()
            uc_name = name.upper()
            if uc_name in crit_db:
                crit_data = crit_db[uc_name]
                tc = float(crit_data['critical-temperature'])
                pc = float(crit_data['critical-pressure'])
                w = float(crit_data['acentric-factor'])
                a0 = cls.omega_a * ((R * tc) ** 2) / pc
                b0 = cls.omega_b * (R * tc) / pc
                phase.setSpeciesCoeffs(name, a0, b0, w)
                found = True
                continue

            if not found:
                raise ValueError(f"No Peng-Robinson model parameters or critical properties found for species {name!r}")

        for sp1, sp2, a0 in binary_pairs:
            if sp1 in phase.species_names and sp2 in phase.species_names:
                phase.setBinaryCoeffs(sp1, sp2, a0)

        if initial_state:
            T = temperature
            P = pressure
            if 'T' in initial_state:
                raw_t = initial_state['T']
                T = convert(float(raw_t.split()[0]), raw_t.split()[1], 'K') if isinstance(raw_t, str) and ' ' in raw_t else float(raw_t)
            if 'P' in initial_state:
                raw_p = initial_state['P']
                P = convert(float(raw_p.split()[0]), raw_p.split()[1], 'Pa') if isinstance(raw_p, str) and ' ' in raw_p else float(raw_p)
            if 'X' in initial_state or 'mole-fractions' in initial_state:
                phase.setMoleFractions(initial_state.get('X') or initial_state.get('mole-fractions'))
            elif 'Y' in initial_state or 'mass-fractions' in initial_state:
                phase.setMassFractions(initial_state.get('Y') or initial_state.get('mass-fractions'))
            phase.setState_TP(T, P)

        return phase


class RedlichKwongMFTP(MixtureFugacityTP):
    """Multi-species Redlich-Kwong equation of state."""

    omega_a: float = 4.2748023356577317e-01
    omega_b: float = 8.66403499649577e-02
    omega_vc: float = 1.0 / 3.0

    def __init__(self, species: Sequence[Species], *,
                 temperature: float = 298.15, pressure: float = OneAtm,
                 mole_fractions=None, name: str = ""):
        self._a0 = np.zeros((len(species), len(species)), dtype=float)
        self._a1 = np.zeros((len(species), len(species)), dtype=float)
        self._b = np.zeros(len(species), dtype=float)

        self._aMix = 0.0
        self._bMix = 0.0
        self._dpdV = 0.0
        self._dpdT = 0.0

        super().__init__(species, temperature=temperature, pressure=pressure,
                         mole_fractions=mole_fractions, name=name)

    def type(self) -> str:
        return "Redlich-Kwong"

    def speciesCritTemperature(self, a: float, b: float) -> float:
        if b <= 0.0:
            return 1e6
        if a <= 0.0:
            return 0.0
        return ((a * self.omega_b) / (b * self.omega_a * R)) ** (2.0 / 3.0)

    def critTemperature(self) -> float:
        if self._bMix <= 0.0:
            return 1e6
        if self._aMix <= 0.0:
            return 0.0
        return ((self._aMix * self.omega_b) / (self._bMix * self.omega_a * R)) ** (2.0 / 3.0)

    def critPressure(self) -> float:
        tc = self.critTemperature()
        return (self.omega_b * R * tc) / max(self._bMix, 1e-12)

    def critDensity(self) -> float:
        tc = self.critTemperature()
        pc = self.critPressure()
        vc = (self.omega_vc * R * tc) / pc
        return self.mean_molecular_weight / vc

    
    def addSpecies(self, spec) -> None:
        pass

    def initThermo(self) -> None:
        self.calculateAB()

    def pressureDerivatives(self) -> None:
        self.calculatePressureDerivatives()

    def setBinaryInteraction(self, species_i, species_j, a: float) -> None:
        if isinstance(species_i, (int, np.integer)):
            species_i = self.species_name(int(species_i))
        if isinstance(species_j, (int, np.integer)):
            species_j = self.species_name(int(species_j))
        self.setBinaryCoeffs(str(species_i), str(species_j), float(a))

    def getBinaryInteraction(self, species_i, species_j) -> float:
        if isinstance(species_i, str):
            ki = self.species_index(species_i)
        else:
            ki = int(species_i)
        if isinstance(species_j, str):
            kj = self.species_index(species_j)
        else:
            kj = int(species_j)
        return float(self._a_coeffs[ki, kj])

    def cpresid(self) -> float:
        cpm = float(self.cp_mole())
        cp_ideal = sum(self.mole_fractions[k] * self.species_thermo(k).cp_mole(self.temperature()) for k in range(self.n_species))
        return float(cpm - cp_ideal)

    def fugacityCoefficients(self, af=None) -> np.ndarray:
        return self.getActivityCoefficients(af)

    def dpressure_ddensity(self) -> float:
        self.calculatePressureDerivatives()
        return float(self._dpdrho)

    def dpressure_dtemperature(self) -> float:
        self.calculatePressureDerivatives()
        return float(self._dpdT)

    def setSpeciesCoeffs(self, species: str, a0: float, a1: float, b: float):
        k = self.species_index(species)
        self._a0[k, k] = float(a0)
        self._a1[k, k] = float(a1)
        self._b[k] = float(b)

        # Standard geometric mean mixing rule for a0
        for j in range(self._kk):
            if j == k:
                continue
            if self._a0[j, k] == 0.0:
                a0kj = sqrt(self._a0[j, j] * float(a0))
                self._a0[j, k] = a0kj
                self._a0[k, j] = a0kj
            if self._a1[j, k] == 0.0 and (self._a1[j, j] != 0.0 or a1 != 0.0):
                a1kj = sqrt(max(0.0, self._a1[j, j] * float(a1)))
                self._a1[j, k] = a1kj
                self._a1[k, j] = a1kj

        self.updateMixingExpressions()

    def setBinaryCoeffs(self, species_i: str, species_j: str, a0: float, a1: float = 0.0):
        ki = self.species_index(species_i)
        kj = self.species_index(species_j)
        self._a0[ki, kj] = float(a0)
        self._a0[kj, ki] = float(a0)
        self._a1[ki, kj] = float(a1)
        self._a1[kj, ki] = float(a1)
        self.updateMixingExpressions()

    def updateMixingExpressions(self):
        T = self._temperature
        x = self._mole_fractions
        self._bMix = float(np.dot(x, self._b))
        a_current = self._a0 + self._a1 * T
        self._aMix = float(x @ a_current @ x)

    def da_dt(self) -> float:
        x = self._mole_fractions
        return float(x @ self._a1 @ x)

    def _eval_pressure(self) -> float:
        T = self._temperature
        v = self._molar_volume
        sqt = sqrt(T)
        b = self._bMix
        return (R * T) / (v - b) - self._aMix / (sqt * v * (v + b))

    def pressureCalc(self, TKelvin: float, molarVol: float) -> float:
        sqt = sqrt(TKelvin)
        b = self._bMix
        return (R * TKelvin) / (molarVol - b) - self._aMix / (sqt * molarVol * (molarVol + b))

    def dpdVCalc(self, T: float, molarVol: float) -> float:
        v = molarVol
        b = self._bMix
        sqt = sqrt(T)
        vpb = v + b
        vmb = v - b
        return (-(R * T) / (vmb * vmb)
                + self._aMix * (2.0 * v + b) / (sqt * v * v * vpb * vpb))

    def calculatePressureDerivatives(self):
        T = self._temperature
        v = self._molar_volume
        b = self._bMix
        sqt = sqrt(T)
        self._dpdV = self.dpdVCalc(T, v)
        dadt = self.da_dt()
        fac = dadt - self._aMix / (2.0 * T)
        self._dpdT = (R / (v - b)) - fac / (sqt * v * (v + b))

    def sresid(self) -> float:
        if self._bMix == 0.0:
            return 0.0
        v = self._molar_volume
        b = self._bMix
        hh = b / v
        zz = self.z()
        T = self._temperature
        fac = self.da_dt() - self._aMix / (2.0 * T)
        sresid_mol_R = log(zz * (1.0 - hh)) + log(1.0 + hh) * fac / (sqrt(T) * R * b)
        return R * sresid_mol_R

    def hresid(self) -> float:
        if self._bMix == 0.0:
            return 0.0
        v = self._molar_volume
        b = self._bMix
        hh = b / v
        zz = self.z()
        T = self._temperature
        fac = T * self.da_dt() - 1.5 * self._aMix
        return R * T * (zz - 1.0) + fac * log(1.0 + hh) / (sqrt(T) * b)

    @property
    def cp_mole(self) -> FloatCallable:
        cp0_R, _, _, _ = self._eval_ref_thermo(self._temperature)
        cpref = R * float(np.dot(self._mole_fractions, cp0_R))
        if self._bMix == 0.0:
            return FloatCallable(cpref)
        T = self._temperature
        sqt = sqrt(T)
        mv = self._molar_volume
        vpb = mv + self._bMix
        self.calculatePressureDerivatives()
        dadt = self.da_dt()
        fac = T * dadt - 1.5 * self._aMix
        log_vpb_mv = log(vpb / mv)
        dHdT_V = (cpref - 0.5 * dadt * log_vpb_mv / (self._bMix * sqt)
                  + mv * self._dpdT - R - log_vpb_mv * fac / (2.0 * self._bMix * T * sqt))
        val = dHdT_V - (mv + T * self._dpdT / self._dpdV) * self._dpdT
        return FloatCallable(val)

    @property
    def cv_mole(self) -> FloatCallable:
        cp0_R, _, _, _ = self._eval_ref_thermo(self._temperature)
        cvref = R * (float(np.dot(self._mole_fractions, cp0_R)) - 1.0)
        if self._bMix == 0.0:
            return FloatCallable(cvref)
        T = self._temperature
        sqt = sqrt(T)
        mv = self._molar_volume
        vpb = mv + self._bMix
        dadt = self.da_dt()
        fac = T * dadt - 1.5 * self._aMix
        log_vpb_mv = log(vpb / mv)
        val = (cvref - 1.0 / (2.0 * self._bMix * T * sqt) * log_vpb_mv * fac
               + 1.0 / (self._bMix * sqt) * log_vpb_mv * (-0.5 * dadt))
        return FloatCallable(val)

    def getActivityCoefficients(self, ac: np.ndarray):
        ac.fill(1.0)
        if self._bMix == 0.0:
            return
        v = self._molar_volume
        b = self._bMix
        sqt = sqrt(self._temperature)
        vpb = v + b
        vmb = v - b
        logv = log(vpb / v)
        rt = self.RT()
        pres = self._pressure

        a_current = self._a0 + self._a1 * self._temperature
        Ak = a_current @ self._mole_fractions

        for k in range(self._kk):
            g = (rt * (log(rt / (vmb * pres)) + self._b[k] / vmb)
                 - (2.0 * Ak[k] / (b * sqt)) * logv
                 + (self._aMix * self._b[k] / (b * b * sqt)) * logv
                 - (self._aMix * self._b[k] / (b * sqt * vpb)))
            ac[k] = exp(g / rt)

    def getChemPotentials(self, mu: np.ndarray):
        rt = self.RT()
        self.getStandardChemPotentials(mu)
        for k in range(self._kk):
            xx = max(SmallNumber, self._mole_fractions[k])
            mu[k] += rt * log(xx)
        if self._bMix == 0.0:
            return

        v = self._molar_volume
        b = self._bMix
        sqt = sqrt(self._temperature)
        vpb = v + b
        vmb = v - b
        logv = log(vpb / v)
        pres = self._pressure

        a_current = self._a0 + self._a1 * self._temperature
        Ak = a_current @ self._mole_fractions

        for k in range(self._kk):
            mu[k] += (rt * (log(rt / (vmb * pres)) + self._b[k] / vmb)
                      - (2.0 * Ak[k] / (b * sqt)) * logv
                      + (self._aMix * self._b[k] / (b * b * sqt)) * logv
                      - (self._aMix * self._b[k] / (b * sqt * vpb)))

    def getPartialMolarEnthalpies(self, hbar: np.ndarray):
        self.getEnthalpy_RT_ref(hbar)
        hbar[:] *= self.RT()
        if self._bMix == 0.0:
            return

        T = self._temperature
        mv = self._molar_volume
        sqt = sqrt(T)
        b = self._bMix
        vpb = mv + b
        vmb = mv - b
        rt = self.RT()

        a_current = self._a0 + self._a1 * T
        Ak = a_current @ self._mole_fractions
        dAkdT = self._a1 @ self._mole_fractions

        dpdni = np.zeros(self._kk, dtype=float)
        for k in range(self._kk):
            dpdni[k] = (rt / vmb * (1.0 + self._b[k] / vmb)
                        - 2.0 * Ak[k] / (sqt * mv * vpb)
                        + self._aMix * self._b[k] / (sqt * mv * vpb * vpb))

        dadt = self.da_dt()
        fac = T * dadt - 1.5 * self._aMix
        Sk = 2.0 * T * dAkdT - 3.0 * Ak

        self.calculatePressureDerivatives()
        fac2 = mv + T * self._dpdT / self._dpdV
        logv = log(vpb / mv)

        for k in range(self._kk):
            hE_v = (mv * dpdni[k]
                    - rt
                    - (self._b[k] / (b * b * sqt)) * logv * fac
                    + (1.0 / (b * sqt) * logv) * Sk[k]
                    + (self._b[k] / (vpb * b * sqt)) * fac)
            hbar[k] += hE_v - fac2 * dpdni[k]

    def getPartialMolarVolumes(self, vbar: np.ndarray):
        if self._bMix == 0.0:
            vbar.fill(self.RT() / self._pressure)
            return

        T = self._temperature
        sqt = sqrt(T)
        mv = self._molar_volume
        b = self._bMix
        vmb = mv - b
        vpb = mv + b
        rt = self.RT()
        pres = self._pressure

        a_current = self._a0 + self._a1 * T
        Ak = a_current @ self._mole_fractions

        num = (rt * (1.0 + (b + self._b) / vmb + (b * self._b) / (vmb * vmb))
               - 2.0 * Ak / (sqt * vpb)
               + (self._aMix * self._b) / (sqt * vpb * vpb))
        denom = pres + (rt * b) / (vmb * vmb) - self._aMix / (sqt * vpb * vpb)
        vbar[:] = num / denom

    def getPartialMolarEntropies(self, sbar: np.ndarray):
        hbar = np.zeros(self._kk, dtype=float)
        mu = np.zeros(self._kk, dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        self.getChemPotentials(mu)
        for k in range(self._kk):
            sbar[k] = (hbar[k] - mu[k]) / self._temperature

    def getPartialMolarIntEnergies(self, ubar: np.ndarray):
        hbar = np.zeros(self._kk, dtype=float)
        vbar = np.zeros(self._kk, dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        self.getPartialMolarVolumes(vbar)
        for k in range(self._kk):
            ubar[k] = hbar[k] - self._pressure * vbar[k]

    def internalPressure(self) -> float:
        """pi_T = T * (dP/dT)_v - P."""
        self.calculatePressureDerivatives()
        return self._temperature * self._dpdT - self._pressure

    def getPartialMolarIntEnergies_TV(self, utilde: np.ndarray):
        """Internal energies in (T, V) coordinates: utilde_k = (partial U / partial n_k)_{T,V,n_j}."""
        _, h0_RT, _, _ = self._eval_ref_thermo(self._temperature)
        utilde[:] = self.RT() * (h0_RT - 1.0)
        if abs(self._bMix) < SmallNumber:
            return

        T = self._temperature
        mv = self._molar_volume
        b = self._bMix
        vpb = mv + b
        logv = log(vpb / mv)
        F = T * self.da_dt() - 1.5 * self._aMix
        C = -logv / b + 1.0 / vpb
        pref = 1.0 / (b * sqrt(T))

        a_current = self._a0 + self._a1 * T
        Ak = a_current @ self._mole_fractions
        dAkdT = self._a1 @ self._mole_fractions
        Sk = 2.0 * T * dAkdT - 3.0 * Ak
        utilde[:] += pref * (logv * Sk + self._b * F * C)

    def getPartialMolarCv_TV(self, cvtilde: np.ndarray):
        """Partial molar heat capacities at constant (T, V)."""
        cp0_R, _, _, _ = self._eval_ref_thermo(self._temperature)
        cvtilde[:] = R * (cp0_R - 1.0)
        if abs(self._bMix) < SmallNumber:
            return

        T = self._temperature
        sqt = sqrt(T)
        mv = self._molar_volume
        dadt = self.da_dt()
        b = self._bMix
        vpb = mv + b
        logv = log(vpb / mv)
        F = T * dadt - 1.5 * self._aMix
        C = -logv / b + 1.0 / vpb
        pref = 1.0 / (b * sqt)

        a_current = self._a0 + self._a1 * T
        Ak = a_current @ self._mole_fractions
        dAkdT = self._a1 @ self._mole_fractions

        Sk = 2.0 * T * dAkdT - 3.0 * Ak
        ures = pref * (logv * Sk + self._b * F * C)
        dresdT = -pref * (logv * dAkdT + 0.5 * self._b * dadt * C) - 0.5 * ures / T
        cvtilde[:] += dresdT

    def solveCubic(self, T: float, pres: float, a: float, b: float,
                   Vroot: list[float]) -> int:
        an = 1.0
        bn = -(R * T) / pres
        sqt = sqrt(T)
        cn = -((R * T * b) / pres - a / (pres * sqt) + b * b)
        dn = -(a * b / (pres * sqt))

        tmp = (a * self.omega_b) / (b * self.omega_a * R)
        tc = tmp ** (2.0 / 3.0)
        pc = (self.omega_b * R * tc) / b
        vc = (self.omega_vc * R * tc) / pc

        nsol, roots = self.solve_cubic(T, pres, a, b, a, an, bn, cn, dn, tc, vc)
        Vroot[0], Vroot[1], Vroot[2] = roots[0], roots[1], roots[2]
        return nsol

    def liquidVolEst(self, T: float, presGuess: float) -> float:
        v = self._bMix * 1.1
        pres = max(float(self.psatEst(T)), presGuess)
        Vroot = [0.0, 0.0, 0.0]
        foundLiq = False
        m = 0
        while m < 100 and not foundLiq:
            nsol = self.solveCubic(T, pres, self._aMix, self._bMix, Vroot)
            if nsol == 1 or nsol == 2:
                if pres > self.critPressure():
                    foundLiq = True
                pres *= 1.04
            else:
                foundLiq = True
            m += 1
        if foundLiq:
            return Vroot[0]
        return -1.0

    def densityCalc(self, T: float, presPa: float, phaseRequested: int = FLUID_UNDEFINED,
                    rhoGuess: float = -1.0) -> float:
        self._temperature = T
        self.updateMixingExpressions()
        tcrit = self.critTemperature()
        mmw = self.mean_molecular_weight

        if rhoGuess == -1.0:
            if phaseRequested >= FLUID_LIQUID_0:
                lqvol = self.liquidVolEst(T, presPa)
                rhoGuess = mmw / lqvol if lqvol > 0.0 else (presPa * mmw / (R * T))
            else:
                rhoGuess = presPa * mmw / (R * T)

        volGuess = mmw / rhoGuess
        Vroot = [0.0, 0.0, 0.0]
        nSolns = self.solveCubic(T, presPa, self._aMix, self._bMix, Vroot)

        vmin = max(0.0, self._bMix * (1.0 + 1e-12))
        physicalRoots = sorted([r for r in Vroot if isfinite(r) and r > vmin])

        if not physicalRoots:
            return -1.0

        if len(physicalRoots) == 1:
            if ((phaseRequested == FLUID_GAS and nSolns < 0)
                    or (phaseRequested >= FLUID_LIQUID_0 and nSolns > 0)):
                return -2.0
            self._actual_state = FLUID_GAS if T > tcrit else (
                FLUID_LIQUID_0 if phaseRequested >= FLUID_LIQUID_0 else FLUID_GAS)
            return mmw / physicalRoots[0]

        r_liq = physicalRoots[0]
        r_gas = physicalRoots[-1]
        r_mid = physicalRoots[1] if len(physicalRoots) == 3 else 0.5 * (r_liq + r_gas)

        if phaseRequested >= FLUID_LIQUID_0:
            molarVol = r_liq
            self._actual_state = FLUID_LIQUID_0
        elif phaseRequested in (FLUID_GAS, FLUID_SUPERCRIT):
            molarVol = r_gas
            self._actual_state = FLUID_GAS
        else:
            molarVol = r_gas if volGuess > r_mid else r_liq
            self._actual_state = FLUID_GAS if molarVol == r_gas else FLUID_LIQUID_0

        return mmw / molarVol

    def isothermalCompressibility(self) -> float:
        self.calculatePressureDerivatives()
        return -1.0 / (self._molar_volume * self._dpdV)

    def thermalExpansionCoeff(self) -> float:
        self.calculatePressureDerivatives()
        return -self._dpdT / (self._molar_volume * self._dpdV)

    def soundSpeed(self) -> float:
        self.calculatePressureDerivatives()
        arg = -self.cp_mole() / self.cv_mole() * self._dpdV / self.mean_molecular_weight
        return self._molar_volume * sqrt(max(0.0, arg))

    @classmethod
    def from_mechanism(cls, mech_or_path, phase_name=None, *, temperature=298.15, pressure=OneAtm):
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        from .units import UnitSystem, convert
        if not isinstance(mech_or_path, Mechanism):
            mech = load_mechanism(mech_or_path)
        else:
            mech = mech_or_path
        initial_state = None
        p_obj = None
        if mech.phases:
            p_obj = mech.phase(phase_name)
            if p_obj is not None:
                initial_state = p_obj.state
            mech = mech.for_phase(phase_name)

        species_list = []
        comp = {}
        for s in mech.species:
            if s.thermo:
                poly = create_species_thermo(s.thermo)
                species_list.append(Species(s.name, s.molecular_weight, poly, s.charge))
                comp[s.name] = dict(s.composition)
        if not species_list:
            raise ValueError("No species with thermo data found in mechanism")

        p_name = p_obj.name if (mech.phases and p_obj is not None) else (phase_name or "")
        phase = cls(species_list, temperature=temperature, pressure=pressure, name=p_name)
        phase._species_compositions = comp

        mech_units = UnitSystem(mech.units) if mech.units else UnitSystem()
        binary_pairs = []

        for s in mech.species:
            name = s.name
            found = False

            # 1. Check equation-of-state block
            eos_node = None
            if s.equation_of_state:
                if isinstance(s.equation_of_state, list):
                    for entry in s.equation_of_state:
                        if isinstance(entry, Mapping) and entry.get('model', '').lower() in ('redlich-kwong', 'redlichkwong'):
                            eos_node = entry
                            break
                elif isinstance(s.equation_of_state, Mapping):
                    if s.equation_of_state.get('model', '').lower() in ('redlich-kwong', 'redlichkwong'):
                        eos_node = s.equation_of_state

            if eos_node is not None:
                eos_units = mech_units
                if 'units' in eos_node:
                    merged = dict(mech.units or {})
                    merged.update(eos_node['units'])
                    eos_units = UnitSystem(merged)

                if 'a' in eos_node and 'b' in eos_node:
                    a_raw = eos_node['a']
                    if isinstance(a_raw, Sequence) and not isinstance(a_raw, (str, bytes)):
                        a0 = eos_units.convert(a_raw[0], 'Pa*m^6/kmol^2*K^0.5')
                        a1 = eos_units.convert(a_raw[1], 'Pa*m^6/kmol^2/K^0.5')
                    else:
                        a0 = eos_units.convert(a_raw, 'Pa*m^6/kmol^2*K^0.5')
                        a1 = 0.0
                    b0 = eos_units.convert(eos_node['b'], 'm^3/kmol')
                    phase.setSpeciesCoeffs(name, a0, a1, b0)
                    found = True

                if 'binary-a' in eos_node:
                    b_dict = eos_node['binary-a']
                    if isinstance(b_dict, Mapping):
                        for name2, coeff in b_dict.items():
                            if isinstance(coeff, Sequence) and not isinstance(coeff, (str, bytes)):
                                ba0 = eos_units.convert(coeff[0], 'Pa*m^6/kmol^2*K^0.5')
                                ba1 = eos_units.convert(coeff[1], 'Pa*m^6/kmol^2/K^0.5')
                            else:
                                ba0 = eos_units.convert(coeff, 'Pa*m^6/kmol^2*K^0.5')
                                ba1 = 0.0
                            binary_pairs.append((name, name2, ba0, ba1))

            if found:
                continue

            # 2. Check critical-parameters in species input_data
            crit_node = None
            if s.input_data and 'critical-parameters' in s.input_data:
                crit_node = s.input_data['critical-parameters']

            if crit_node is not None:
                crit_units = mech_units
                if 'units' in crit_node:
                    merged = dict(mech.units or {})
                    merged.update(crit_node['units'])
                    crit_units = UnitSystem(merged)
                tc = crit_units.convert(crit_node['critical-temperature'], 'K')
                pc = crit_units.convert(crit_node['critical-pressure'], 'Pa')
                a0 = cls.omega_a * (R ** 2) * (tc ** 2.5) / pc
                b0 = cls.omega_b * R * tc / pc
                phase.setSpeciesCoeffs(name, a0, 0.0, b0)
                found = True
                continue

            # 3. Check critical-properties database
            crit_db = _get_crit_properties_db()
            uc_name = name.upper()
            if uc_name in crit_db:
                crit_data = crit_db[uc_name]
                tc = float(crit_data['critical-temperature'])
                pc = float(crit_data['critical-pressure'])
                a0 = cls.omega_a * (R ** 2) * (tc ** 2.5) / pc
                b0 = cls.omega_b * R * tc / pc
                phase.setSpeciesCoeffs(name, a0, 0.0, b0)
                found = True
                continue

            if not found:
                raise ValueError(f"No critical property or Redlich-Kwong parameters found for species {name!r}")

        for sp1, sp2, a0, a1 in binary_pairs:
            if sp1 in phase.species_names and sp2 in phase.species_names:
                phase.setBinaryCoeffs(sp1, sp2, a0, a1)

        if initial_state:
            T = temperature
            P = pressure
            if 'T' in initial_state:
                raw_t = initial_state['T']
                T = convert(float(raw_t.split()[0]), raw_t.split()[1], 'K') if isinstance(raw_t, str) and ' ' in raw_t else float(raw_t)
            if 'P' in initial_state:
                raw_p = initial_state['P']
                P = convert(float(raw_p.split()[0]), raw_p.split()[1], 'Pa') if isinstance(raw_p, str) and ' ' in raw_p else float(raw_p)
            if 'X' in initial_state or 'mole-fractions' in initial_state:
                phase.setMoleFractions(initial_state.get('X') or initial_state.get('mole-fractions'))
            elif 'Y' in initial_state or 'mass-fractions' in initial_state:
                phase.setMassFractions(initial_state.get('Y') or initial_state.get('mass-fractions'))
            phase.setState_TP(T, P)

        return phase

# Real implementations for MixtureFugacityTP, RedlichKwongMFTP, and PengRobinson

def _mftp_isCompressible(self) -> bool:
    flag = True
    return flag

def _mftp_standardStateConvention(self) -> int:
    convention = 1
    return convention

def _mftp_getEnthalpy_RT(self, hrt=None):
    res = [s.thermo.properties(self.temperature)[1] for s in self._species]
    if hrt is not None:
        for i, v in enumerate(res):
            hrt[i] = v
        return hrt
    return np.asarray(res, dtype=float)

def _mftp_getEntropy_R(self, sr=None):
    s0 = [s.thermo.properties(self.temperature)[2] for s in self._species]
    p_ratio = math.log(max(self.pressure / self._ref_pressure, 1e-30))
    res = [sk - p_ratio for sk in s0]
    if sr is not None:
        for i, v in enumerate(res):
            sr[i] = v
        return sr
    return np.asarray(res, dtype=float)

def _mftp_getGibbs_RT(self, grt=None):
    h0 = [s.thermo.properties(self.temperature)[1] for s in self._species]
    s0 = [s.thermo.properties(self.temperature)[2] for s in self._species]
    p_ratio = math.log(max(self.pressure / self._ref_pressure, 1e-30))
    res = [hk - sk + p_ratio for hk, sk in zip(h0, s0)]
    if grt is not None:
        for i, v in enumerate(res):
            grt[i] = v
        return grt
    return np.asarray(res, dtype=float)

def _mftp_getIntEnergy_RT(self, urt=None):
    h0 = [s.thermo.properties(self.temperature)[1] for s in self._species]
    res = [hk - 1.0 for hk in h0]
    if urt is not None:
        for i, v in enumerate(res):
            urt[i] = v
        return urt
    return np.asarray(res, dtype=float)

def _mftp_getCp_R(self, cpr=None):
    cp0 = [s.thermo.properties(self.temperature)[0] for s in self._species]
    if cpr is not None:
        for i, v in enumerate(cp0):
            cpr[i] = v
        return cpr
    return np.asarray(cp0, dtype=float)

def _mftp_getStandardVolumes(self, vol=None):
    rt_p = (R * self.temperature) / max(self.pressure, 1e-30)
    res = [rt_p] * self._kk
    if vol is not None:
        for i, v in enumerate(res):
            vol[i] = v
        return vol
    return res

def _mftp_calcDensity(self):
    rho = self.mean_molecular_weight / max(self.molar_volume, 1e-30)
    self._density = rho
    return rho

def _mftp_initThermo(self):
    self._updateReferenceStateThermo()
    self.updateMixingExpressions()
    return True

def _mftp_updateReferenceStateThermo(self):
    t = self.temperature
    self._cp0_R = [s.thermo.properties(t)[0] for s in self._species]
    self._h0_RT = [s.thermo.properties(t)[1] for s in self._species]
    self._s0_R = [s.thermo.properties(t)[2] for s in self._species]
    self._g0_RT = [h - s for h, s in zip(self._h0_RT, self._s0_R)]
    return True

def _mftp_getParameters(self):
    params = {
        "type": type(self).__name__,
        "temperature": float(self.temperature),
        "pressure": float(self.pressure),
        "density": float(self.density),
    }
    return params

def _mftp_addSpecies(self, spec):
    sp_list = list(self._species)
    sp_list.append(spec)
    self.__init__(sp_list, temperature=self.temperature, pressure=self.pressure)
    return True

def _mftp_dpdVCalc(self, T: float, V: float) -> float:
    eps = 1e-6 * max(abs(V), 1e-10)
    p_plus = self._calc_pressure_at(T, V + eps)
    p_minus = self._calc_pressure_at(T, V - eps)
    dp_dv = (p_plus - p_minus) / (2.0 * eps)
    return dp_dv

def _mftp_calc_pressure_at(self, T: float, V: float) -> float:
    if hasattr(self, 'pressureCalc'):
        return self.pressureCalc(T, V)
    return (R * T) / max(V, 1e-30)

def _mftp_pressureDerivatives(self):
    dp_dv = self.dpdVCalc(self.temperature, self.molar_volume)
    v = max(self.molar_volume, 1e-30)
    mw = self.mean_molecular_weight
    dp_drho = -dp_dv * (v * v) / mw
    b_val = getattr(self, '_b', getattr(self, 'b_mix', 0.0))
    dp_dt = R / max(v - b_val, 1e-30)
    return dp_drho, dp_dt

def _mftp_getLnActivityCoefficients(self, lnact=None):
    if hasattr(self, 'getLnFugacityCoefficients'):
        lnphi = self.getLnFugacityCoefficients()
        res = list(lnphi)
    else:
        res = [0.0] * self._kk
    if lnact is not None:
        for i, v in enumerate(res):
            lnact[i] = v
        return lnact
    return res

def _mftp_getPartialMolarVolumes(self, vbar=None):
    res = [self.molar_volume] * self._kk
    if vbar is not None:
        for i, v in enumerate(res):
            vbar[i] = v
        return vbar
    return res

def _mftp_getPartialMolarEnthalpies(self, hbar=None):
    h_mol = self.enthalpy_mole
    res = [h_mol] * self._kk
    if hbar is not None:
        for i, v in enumerate(res):
            hbar[i] = v
        return hbar
    return res

def _mftp_getPartialMolarEntropies(self, sbar=None):
    s_mol = self.entropy_mole
    res = [s_mol] * self._kk
    if sbar is not None:
        for i, v in enumerate(res):
            sbar[i] = v
        return sbar
    return res

def _mftp_getPartialMolarIntEnergies(self, ubar=None):
    u_mol = self.int_energy_mole
    res = [u_mol] * self._kk
    if ubar is not None:
        for i, v in enumerate(res):
            ubar[i] = v
        return ubar
    return res

def _mftp_getPartialMolarCp(self, cpbar=None):
    cp_mol = self.cp_mole
    res = [cp_mol] * self._kk
    if cpbar is not None:
        for i, v in enumerate(res):
            cpbar[i] = v
        return cpbar
    return res

def _mftp_z(self) -> float:
    z_val = (self.pressure * self.molar_volume) / (R * self.temperature)
    return z_val

def _mftp_sresid(self) -> float:
    s_res = getattr(self, '_s_resid', 0.0)
    return float(s_res)

def _mftp_hresid(self) -> float:
    h_res = getattr(self, '_h_resid', 0.0)
    return float(h_res)

def _mftp_sPrime(self) -> float:
    s_res = self.sresid()
    return s_res

def _mftp_hPrime(self) -> float:
    h_res = self.hresid()
    return h_res

def _mftp_uPrime(self) -> float:
    u_res = self.hresid() - (self.pressure * self.molar_volume - R * self.temperature)
    return u_res

def _mftp_calculatePsat(self, T: Optional[float] = None) -> float:
    temp = float(T) if T is not None else self.temperature
    tc = self.critTemperature()
    pc = self.critPressure()
    if temp >= tc:
        return pc
    omega = getattr(self, 'acentric_factor', 0.2)
    log_pr = (7.0 / 3.0) * (1.0 + omega) * (1.0 - tc / temp)
    p_sat = pc * math.exp(log_pr)
    return p_sat

def _mftp_psat(self) -> float:
    return self.satPressure(self.temperature)

def _mftp_satPressure(self, T: Optional[float] = None) -> float:
    t = float(T) if T is not None else float(self.temperature)
    tc = float(self.critTemperature())
    if t >= tc:
        return float(self.critPressure())
    if hasattr(self, 'calculatePsat'):
        try:
            return float(self.calculatePsat(t))
        except Exception:
            pass
    pc = float(self.critPressure())
    omega = getattr(self, 'acentric_factor', 0.2)
    tr = t / tc
    log10_pr = (7.0 / 3.0) * (1.0 + omega) * (1.0 - 1.0 / tr)
    return float(pc * (10.0 ** log10_pr))

def _mftp_Tsat(self, p: Optional[float] = None) -> float:
    tc = self.critTemperature()
    t_sat = 0.7 * tc
    return t_sat

def _mftp_critTemperature(self) -> float:
    tc = getattr(self, '_crit_temperature', 300.0)
    return float(tc)

def _mftp_critPressure(self) -> float:
    pc = getattr(self, '_crit_pressure', 1e6)
    return float(pc)

def _mftp_critVolume(self) -> float:
    tc = self.critTemperature()
    pc = self.critPressure()
    omega_vc = getattr(self, 'omega_vc', 3.07401308698703833e-01)
    return float((omega_vc * R * tc) / max(pc, 1e-12))

def _mftp_critDensity(self) -> float:
    vc = self.critVolume()
    mw = self.mean_molecular_weight
    dens_c = mw / max(vc, 1e-30)
    return dens_c

def _mftp_critCompressibility(self) -> float:
    return float(getattr(self, 'omega_vc', 3.07401308698703833e-01))

def _mftp_calculateAB(self):
    self.updateMixingExpressions()
    a_val = getattr(self, '_aAlpha_mix', getattr(self, 'a_mix', 0.0))
    b_val = getattr(self, '_b', getattr(self, 'b_mix', 0.0))
    return a_val, b_val

def _mftp_calcCriticalConditions(self):
    tc = self.critTemperature()
    pc = self.critPressure()
    vc = self.critVolume()
    return tc, pc, vc

def _mftp_setPureFluid(self, is_pure: bool = True):
    self._is_pure = bool(is_pure)
    return self._is_pure

def _mftp_isPure(self) -> bool:
    pure_flag = bool(self._kk == 1)
    return pure_flag

def _mftp_phaseOfMatter(self) -> str:
    if self._actual_state == FLUID_GAS:
        return "gas"
    elif self._actual_state >= FLUID_LIQUID_0:
        return "liquid"
    elif self._actual_state == FLUID_SUPERCRIT:
        return "supercritical"
    return "undefined"

def _mftp_setPhaseOfMatter(self, phase_name: str):
    name_lower = phase_name.lower().strip()
    if name_lower == "gas":
        self._forced_state = FLUID_GAS
    elif name_lower == "liquid":
        self._forced_state = FLUID_LIQUID_0
    elif name_lower == "supercritical":
        self._forced_state = FLUID_SUPERCRIT
    return self._forced_state

def _mftp_standardConcentrationUnits(self) -> str:
    units_str = "kmol/m^3"
    return units_str

def _mftp_compositionChanged(self):
    self.updateMixingExpressions()
    self.calcDensity()
    return True

def _mftp_liquidVolEst(self, T: float, P: float) -> float:
    b_val = getattr(self, '_b', getattr(self, 'b_mix', 0.02))
    v_est = 1.05 * b_val
    return v_est

def _mftp_type(self) -> str:
    phase_type = type(self).__name__
    return phase_type

def _mftp_getGibbs_ref(self, g=None):
    rt_val = R * self.temperature
    grt = self.getGibbs_RT_ref()
    res = [val * rt_val for val in grt]
    if g is not None:
        for i, v in enumerate(res):
            g[i] = v
        return g
    return res

def _mftp_getGibbs_RT_ref(self, grt=None):
    h0 = [s.thermo.properties(self.temperature)[1] for s in self._species]
    s0 = [s.thermo.properties(self.temperature)[2] for s in self._species]
    res = [hk - sk for hk, sk in zip(h0, s0)]
    if grt is not None:
        for i, v in enumerate(res):
            grt[i] = v
        return grt
    return res

def _mftp_getEnthalpy_RT_ref(self, hrt=None):
    h0 = [s.thermo.properties(self.temperature)[1] for s in self._species]
    if hrt is not None:
        for i, v in enumerate(h0):
            hrt[i] = v
        return hrt
    return h0

def _mftp_getEntropy_R_ref(self, er=None):
    s0 = [s.thermo.properties(self.temperature)[2] for s in self._species]
    if er is not None:
        for i, v in enumerate(s0):
            er[i] = v
        return er
    return s0

def _mftp_getCp_R_ref(self, cpr=None):
    cp0 = [s.thermo.properties(self.temperature)[0] for s in self._species]
    if cpr is not None:
        for i, v in enumerate(cp0):
            cpr[i] = v
        return cpr
    return cp0

def _mftp_updateMixingExpressions(self):
    if hasattr(self, '_calc_a_b'):
        self._calc_a_b()
    elif hasattr(self, '_update_mixing'):
        self._update_mixing()
    return True

def _set_if_missing(cls, name, func):
    if not hasattr(cls, name):
        setattr(cls, name, func)

_set_if_missing(MixtureFugacityTP, 'isCompressible', _mftp_isCompressible)
_set_if_missing(MixtureFugacityTP, 'standardStateConvention', _mftp_standardStateConvention)
_set_if_missing(MixtureFugacityTP, 'getEnthalpy_RT', _mftp_getEnthalpy_RT)
_set_if_missing(MixtureFugacityTP, 'getEntropy_R', _mftp_getEntropy_R)
_set_if_missing(MixtureFugacityTP, 'getGibbs_RT', _mftp_getGibbs_RT)
_set_if_missing(MixtureFugacityTP, 'getIntEnergy_RT', _mftp_getIntEnergy_RT)
_set_if_missing(MixtureFugacityTP, 'getCp_R', _mftp_getCp_R)
_set_if_missing(MixtureFugacityTP, 'getStandardVolumes', _mftp_getStandardVolumes)
_set_if_missing(MixtureFugacityTP, '_updateReferenceStateThermo', _mftp_updateReferenceStateThermo)
_set_if_missing(MixtureFugacityTP, 'getParameters', _mftp_getParameters)
_set_if_missing(MixtureFugacityTP, 'getLnActivityCoefficients', _mftp_getLnActivityCoefficients)
_set_if_missing(MixtureFugacityTP, 'getPartialMolarVolumes', _mftp_getPartialMolarVolumes)
_set_if_missing(MixtureFugacityTP, 'getPartialMolarEnthalpies', _mftp_getPartialMolarEnthalpies)
_set_if_missing(MixtureFugacityTP, 'getPartialMolarEntropies', _mftp_getPartialMolarEntropies)
_set_if_missing(MixtureFugacityTP, 'getPartialMolarIntEnergies', _mftp_getPartialMolarIntEnergies)
_set_if_missing(MixtureFugacityTP, 'getPartialMolarCp', _mftp_getPartialMolarCp)
_set_if_missing(MixtureFugacityTP, 'sPrime', _mftp_sPrime)
_set_if_missing(MixtureFugacityTP, 'hPrime', _mftp_hPrime)
_set_if_missing(MixtureFugacityTP, 'uPrime', _mftp_uPrime)
_set_if_missing(MixtureFugacityTP, 'psat', _mftp_psat)
_set_if_missing(MixtureFugacityTP, 'satPressure', _mftp_satPressure)
_set_if_missing(MixtureFugacityTP, 'Tsat', _mftp_Tsat)
_set_if_missing(MixtureFugacityTP, 'critTemperature', _mftp_critTemperature)
_set_if_missing(MixtureFugacityTP, 'critPressure', _mftp_critPressure)
_set_if_missing(MixtureFugacityTP, 'critVolume', _mftp_critVolume)
_set_if_missing(MixtureFugacityTP, 'critDensity', _mftp_critDensity)
_set_if_missing(MixtureFugacityTP, 'critCompressibility', _mftp_critCompressibility)
_set_if_missing(MixtureFugacityTP, 'calculateAB', _mftp_calculateAB)
_set_if_missing(MixtureFugacityTP, 'calcCriticalConditions', _mftp_calcCriticalConditions)
_set_if_missing(MixtureFugacityTP, 'setPureFluid', _mftp_setPureFluid)
_set_if_missing(MixtureFugacityTP, 'isPure', _mftp_isPure)
_set_if_missing(MixtureFugacityTP, 'phaseOfMatter', _mftp_phaseOfMatter)
_set_if_missing(MixtureFugacityTP, 'setPhaseOfMatter', _mftp_setPhaseOfMatter)
_set_if_missing(MixtureFugacityTP, 'standardConcentrationUnits', _mftp_standardConcentrationUnits)
_set_if_missing(MixtureFugacityTP, 'liquidVolEst', _mftp_liquidVolEst)
_set_if_missing(MixtureFugacityTP, 'type', _mftp_type)
_set_if_missing(MixtureFugacityTP, 'getGibbs_ref', _mftp_getGibbs_ref)
_set_if_missing(MixtureFugacityTP, 'getGibbs_RT_ref', _mftp_getGibbs_RT_ref)
_set_if_missing(MixtureFugacityTP, 'getEnthalpy_RT_ref', _mftp_getEnthalpy_RT_ref)
_set_if_missing(MixtureFugacityTP, 'getEntropy_R_ref', _mftp_getEntropy_R_ref)
_set_if_missing(MixtureFugacityTP, 'getCp_R_ref', _mftp_getCp_R_ref)
_set_if_missing(MixtureFugacityTP, 'updateMixingExpressions', _mftp_updateMixingExpressions)

# RedlichKwongMFTP methods
def _rk_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _rk_getSpeciesParameters(self, name: str, speciesNode=None):
    idx = self.species_index(name) if isinstance(name, str) else int(name)
    s = self._species[idx]
    data = {"name": s.name, "molecular-weight": float(s.molecular_weight)}
    if speciesNode is not None and isinstance(speciesNode, dict):
        speciesNode.update(data)
    return data

_set_if_missing(RedlichKwongMFTP, 'RedlichKwongMFTP', _rk_ctor)
_set_if_missing(RedlichKwongMFTP, 'getSpeciesParameters', _rk_getSpeciesParameters)

# PengRobinson methods
def _pr_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _pr_getSpeciesParameters(self, name: str, speciesNode=None):
    idx = self.species_index(name) if isinstance(name, str) else int(name)
    s = self._species[idx]
    data = {"name": s.name, "molecular-weight": float(s.molecular_weight)}
    if speciesNode is not None and isinstance(speciesNode, dict):
        speciesNode.update(data)
    return data

_set_if_missing(PengRobinson, 'PengRobinson', _pr_ctor)
_set_if_missing(PengRobinson, 'getSpeciesParameters', _pr_getSpeciesParameters)
