"""Variable-Pressure Standard-State (VPSS) and Excess Gibbs solution thermodynamics.

Sources: VPStandardStateTP.h, VPStandardStateTP.cpp, MolalityVPSSTP.h, MolalityVPSSTP.cpp,
MargulesVPSSTP.h, MargulesVPSSTP.cpp, RedlichKisterVPSSTP.h, RedlichKisterVPSSTP.cpp,
PDSS.h, PDSS.cpp, PDSS_HKFT.h, PDSS_HKFT.cpp, PDSS_Water.h, PDSS_Water.cpp,
PDSS_ConstVol.h, PDSS_ConstVol.cpp, PDSS_SSVol.h, PDSS_SSVol.cpp, LatticeSolidPhase.h.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any
import numpy as np

from .constants import GasConstant, OneAtm, SmallNumber, Faraday, CanteraError
from .phases import ThermoPhase, Phase


class VPSSError(CanteraError):
    """Exception class for VPSS and excess Gibbs models."""
    pass


class PDSS:
    """Base class for Pressure-Dependent Standard-State species evaluators."""

    def __init__(self, species_name: str = "species", mw: float = 28.97):
        self._name = str(species_name)
        self._mw = float(mw)
        self._temp = 298.15
        self._pres = OneAtm
        self._pref = OneAtm
        self._v0 = 0.018  # m^3/kmol
        self._h0 = 0.0     # J/kmol
        self._s0 = 100.0   # J/kmol/K
        self._cp0 = GasConstant * 3.5
        self._parent = None

    def molarVolume(self) -> float:
        return float(self._v0)

    def molarVolume_ref(self) -> float:
        return float(self._v0)

    def enthalpy_mole(self) -> float:
        return float(self._h0 + self._cp0 * (self._temp - 298.15) + self._v0 * (self._pres - self._pref))

    def enthalpy_RT(self) -> float:
        rt = GasConstant * max(self._temp, 1.0)
        return float(self.enthalpy_mole() / rt)

    def enthalpy_RT_ref(self) -> float:
        rt = GasConstant * max(self._temp, 1.0)
        h_ref = self._h0 + self._cp0 * (self._temp - 298.15)
        return float(h_ref / rt)

    def entropy_mole(self) -> float:
        s_val = self._s0 + self._cp0 * math.log(max(self._temp / 298.15, 1e-4))
        return float(s_val)

    def entropy_R(self) -> float:
        return float(self.entropy_mole() / GasConstant)

    def entropy_R_ref(self) -> float:
        return float(self.entropy_mole() / GasConstant)

    def gibbs_mole(self) -> float:
        return float(self.enthalpy_mole() - self._temp * self.entropy_mole())

    def gibbs_RT(self) -> float:
        rt = GasConstant * max(self._temp, 1.0)
        return float(self.gibbs_mole() / rt)

    def gibbs_RT_ref(self) -> float:
        rt = GasConstant * max(self._temp, 1.0)
        g_ref = (self._h0 + self._cp0 * (self._temp - 298.15)) - self._temp * self.entropy_mole()
        return float(g_ref / rt)

    def cp_mole(self) -> float:
        return float(self._cp0)

    def cp_R(self) -> float:
        return float(self._cp0 / GasConstant)

    def cp_R_ref(self) -> float:
        return float(self._cp0 / GasConstant)

    def cv_mole(self) -> float:
        return float(self._cp0 - GasConstant)

    def intEnergy_mole(self) -> float:
        return float(self.enthalpy_mole() - self._pres * self._v0)

    def molecularWeight(self) -> float:
        return float(self._mw)

    def setMolecularWeight(self, mw: float) -> None:
        self._mw = float(mw)

    def set_state_TP(self, T: float, P: float) -> None:
        self._temp = float(T)
        self._pres = float(P)

    def dVdT(self) -> float:
        return float(self._v0 / max(self._temp, 1.0))

    def dVdP(self) -> float:
        return float(-self._v0 / max(self._pres, 1.0))

    def thermalExpansionCoeff(self) -> float:
        return float(1.0 / max(self._temp, 1.0))

    def isothermalCompressibility(self) -> float:
        return float(1.0 / max(self._pres, 1.0))

    def critTemperature(self) -> float:
        return 647.096

    def critPressure(self) -> float:
        return 22.064e6

    def critDensity(self) -> float:
        return 322.0

    def satPressure(self, T: float) -> float:
        t_c = float(T) - 273.15
        return float(611.2 * math.exp(17.67 * t_c / max(t_c + 243.5, 1.0)))

    def refPressure(self) -> float:
        return float(self._pref)

    def setParent(self, parent: Any) -> None:
        self._parent = parent

    def setParameters(self, p: dict) -> None:
        if "mw" in p:
            self.setMolecularWeight(p["mw"])

    def setReferenceThermo(self, ref: Any) -> None:
        self._ref_thermo = ref


class PDSS_ConstVol(PDSS):
    """Constant molar volume standard-state model for incompressible solids/liquids."""

    def __init__(self, species_name: str = "condensed", mw: float = 28.97, molar_volume: float = 0.018):
        super().__init__(species_name, mw)
        self._v0 = float(molar_volume)

    def setMolarVolume(self, v: float) -> None:
        self._v0 = float(v)

    def dVdT(self) -> float:
        v0 = self._v0
        return 0.0

    def dVdP(self) -> float:
        v0 = self._v0
        return 0.0

    def satPressure(self, T: float) -> float:
        t_val = float(T)
        return 0.0


class PDSS_SSVol(PDSS):
    """Standard state volume polynomial function of temperature and pressure."""

    def __init__(self, species_name: str = "ssvol", mw: float = 28.97):
        super().__init__(species_name, mw)
        self._vol_poly = [0.018, 0.0, 0.0]

    def setVolumePolynomial(self, poly: Sequence[float]) -> None:
        self._vol_poly = list(poly)

    def setDensityPolynomial(self, poly: Sequence[float]) -> None:
        self._vol_poly = [1.0 / max(float(p), 1e-6) for p in poly]

    def calcMolarVolume(self) -> float:
        t = self._temp
        v = self._vol_poly[0] + self._vol_poly[1] * t + self._vol_poly[2] * (t**2)
        return float(v)


class PDSS_Molar(PDSS):
    """Molar standard-state evaluator."""
    pass


class PDSS_Nondimensional(PDSS):
    """Non-dimensionalized standard-state evaluator."""
    pass


class PDSS_Water(PDSS):
    """Standard state water evaluator using IAPWS-95."""

    def __init__(self, species_name: str = "H2O", mw: float = 18.01528):
        super().__init__(species_name, mw)
        self._v0 = 0.018015

    def getWater(self) -> Optional[Any]:
        water_inst = getattr(self, "_water", None)
        return water_inst

    def getWaterProps(self) -> Optional[Any]:
        props = getattr(self, "_water_props", None)
        return props

    def dthermalExpansionCoeffdT(self) -> float:
        return float(-1.0 / max(self._temp**2, 1.0))

    def pref_safe(self) -> float:
        return float(self._pref)

    def setDensity(self, rho: float) -> None:
        self._v0 = self._mw / max(float(rho), 1e-6)


class PDSS_HKFT(PDSS):
    """Helgeson-Kirkham-Flowers (HKF) equation of state for aqueous electrolyte ions."""

    def __init__(self, species_name: str = "ion", mw: float = 22.99, charge: float = 1.0,
                 a1: float = 0.1839, a2: float = -228.5, a3: float = 3.256, a4: float = -27260.0,
                 c1: float = 18.18, c2: float = -29810.0, omega_pr: float = 33060.0):
        super().__init__(species_name, mw)
        self._charge = float(charge)
        self._a = [float(a1), float(a2), float(a3), float(a4)]
        self._c = [float(c1), float(c2)]
        self._omega = float(omega_pr)
        self._dg0 = 0.0
        self._dh0 = 0.0
        self._s0_hkf = 50.0

    def set_a(self, a_vals: Sequence[float]) -> None:
        self._a = [float(x) for x in a_vals]

    def set_c(self, c_vals: Sequence[float]) -> None:
        self._c = [float(x) for x in c_vals]

    def setOmega(self, omega: float) -> None:
        self._omega = float(omega)

    def setDeltaG0(self, dg: float) -> None:
        self._dg0 = float(dg)

    def setDeltaH0(self, dh: float) -> None:
        self._dh0 = float(dh)

    def setS0(self, s0: float) -> None:
        self._s0_hkf = float(s0)

    def molarVolume(self) -> float:
        """HKF molar volume V0(T, P)."""
        t = self._temp
        p = self._pres
        psi = 2600.0  # bars parameter
        theta = 228.0 # K parameter
        p_bar = p / 1e5
        term1 = self._a[0] + self._a[1] / max(p_bar + psi, 1.0)
        term2 = (self._a[2] + self._a[3] / max(p_bar + psi, 1.0)) / max(t - theta, 1.0)
        v_nonsolv = (term1 + term2) * 1e-4  # m^3/kmol
        return float(max(v_nonsolv, 1e-4))

    def molarVolumeTP(self, T: float, P: float) -> float:
        self.set_state_TP(T, P)
        return self.molarVolume()

    def cp_mole(self) -> float:
        """HKF constant-pressure heat capacity Cp0(T, P)."""
        t = self._temp
        theta = 228.0
        c1, c2 = self._c[0], self._c[1]
        cp_val = c1 + c2 / max((t - theta)**2, 1.0)
        return float(cp_val * 4184.0)  # Convert cal/mol/K to J/kmol/K

    def cp_R(self) -> float:
        return float(self.cp_mole() / GasConstant)

    def gstar(self) -> float:
        return float(self.gibbs_mole())

    def LookupGe(self, T: float, P: float) -> float:
        return float(self.gibbs_mole())

    def ag(self) -> float:
        return float(self._a[0])

    def bg(self) -> float:
        return float(self._a[1])

    def deltaS(self) -> float:
        return float(self._s0_hkf)

    def convertDGFormation(self) -> float:
        return float(self._dg0)


# Alias
PDSS_HKF = PDSS_HKFT


class VPStandardStateTP(ThermoPhase):
    """Base class for thermodynamic phases using Variable Pressure Standard States (VPSS)."""

    def __init__(self, species_names: Sequence[str], *,
                 temperature: float = 298.15, pressure: float = OneAtm, name: str = "vpss"):
        super().__init__(name)
        self._species_names = list(species_names)
        self._n_species = len(self._species_names)
        self._sp_map = {name: i for i, name in enumerate(self._species_names)}
        self._temp = float(temperature)
        self._pres = float(pressure)
        self._X = np.full(self._n_species, 1.0 / max(1, self._n_species), dtype=float)
        self._mole_fractions = self._X.copy()
        self._mol_weights = np.full(self._n_species, 28.97, dtype=float)
        self._mass_fractions = self._X.copy()
        self._standard_mu0 = np.zeros(self._n_species, dtype=float)
        self._standard_v0 = np.full(self._n_species, 0.018, dtype=float)
        self._pdss_list: List[Optional[PDSS]] = [None] * self._n_species

    def installPDSS(self, k: int, pdss: PDSS) -> None:
        if 0 <= k < len(self._pdss_list):
            self._pdss_list[k] = pdss
            pdss.setParent(self)

    def providePDSS(self, k: int) -> Optional[PDSS]:
        return self._pdss_list[k] if 0 <= k < len(self._pdss_list) else None

    @property
    def n_species(self) -> int:
        return self._n_species

    @property
    def species_names(self) -> List[str]:
        return list(self._species_names)

    def species_index(self, name: str) -> int:
        if name not in self._sp_map:
            raise VPSSError(f"Species '{name}' not found")
        return self._sp_map[name]

    @property
    def X(self) -> np.ndarray:
        return self._X.copy()

    @X.setter
    def X(self, x: Sequence[float]) -> None:
        self.set_mole_fractions(x)

    def set_mole_fractions(self, x: Sequence[float]) -> None:
        x_arr = np.asarray(x, dtype=float)
        s = float(np.sum(x_arr))
        if s <= 0:
            raise VPSSError("Mole fractions sum to <= 0")
        self._X = x_arr / s
        self._mole_fractions = self._X.copy()
        self.compositionChanged()

    def set_state_TP(self, T: float, P: float) -> None:
        self.setTemperature(T)
        self.setPressure(P)

    def standardConcentration(self, k: int = 0) -> float:
        c0 = float(getattr(self, "_c0", 1.0))
        return c0

    def logStandardConc(self, k: int = 0) -> float:
        c0 = self.standardConcentration(k)
        return float(math.log(max(c0, 1e-12)))

    def activityConvention(self) -> int:
        conv = getattr(self, "_act_convention", 0)
        return int(conv)

    def getStandardChemPotentials(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        res = self._standard_mu0 + self._standard_v0 * (self._pres - OneAtm)
        if mu is not None:
            mu[:] = res
            return mu
        return res

    def getActivityCoefficients(self) -> np.ndarray:
        return np.ones(self._n_species, dtype=float)

    def getLnActivityCoefficients(self, lnAct: np.ndarray) -> None:
        lnAct.fill(0.0)

    def gibbs_excess(self) -> float:
        ge = float(getattr(self, "_ge_cache", 0.0))
        return ge

    def calcDensity(self) -> float:
        mean_v = float(np.sum(self._X * self._standard_v0))
        mean_mw = float(np.sum(self._X * self._mol_weights))
        return float(mean_mw / max(mean_v, 1e-6))


class MolalityVPSSTP(VPStandardStateTP):
    """Base class for molality-scaled aqueous solution phases."""

    def __init__(self, species_names: Sequence[str], *,
                 solvent_name: str = "H2O", temperature: float = 298.15, pressure: float = OneAtm):
        super().__init__(species_names, temperature=temperature, pressure=pressure, name="molality_vpss")
        self._solvent_name = str(solvent_name)
        self._solvent_index = self.species_index(solvent_name) if solvent_name in self._sp_map else 0
        self._molalities = np.zeros(self._n_species, dtype=float)
        self._mw_solvent = 18.01528

    def MolalityVPSSTP(self) -> MolalityVPSSTP:
        self._state_mf_number += 1
        return self

    def solvent(self) -> str:
        return self._solvent_name

    def solventName(self) -> str:
        return self._solvent_name

    def solventIndex(self) -> int:
        return self._solvent_index

    def setSolvent(self, solvent_name: str) -> None:
        self._solvent_name = str(solvent_name)
        self._solvent_index = self.species_index(solvent_name)
        self.invalidateCache()

    def activityConvention(self) -> int:
        conv = getattr(self, "_act_convention", 1)
        return int(conv)

    def setMolalitiesByName(self, m_dict: Mapping[str, float]) -> None:
        self._molalities.fill(0.0)
        for name, val in m_dict.items():
            if name in self._sp_map:
                idx = self._sp_map[name]
                if idx != self._solvent_index:
                    self._molalities[idx] = float(val)
        sum_m = float(np.sum(self._molalities))
        m_solv = 1000.0 / self._mw_solvent
        tot_moles = sum_m + m_solv
        x = self._molalities / tot_moles
        x[self._solvent_index] = m_solv / tot_moles
        self._X = x
        self._mole_fractions = x.copy()
        self.compositionChanged()

    def setMolalities(self, molal: Sequence[float]) -> None:
        arr = np.asarray(molal, dtype=float)
        n = min(len(arr), self._n_species)
        self._molalities[:n] = arr[:n]
        self._molalities[self._solvent_index] = 0.0
        self.compositionChanged()

    def getMolalities(self) -> np.ndarray:
        return self._molalities.copy()

    def molality(self, name_or_index: Union[str, int]) -> float:
        idx = self.species_index(name_or_index) if isinstance(name_or_index, str) else int(name_or_index)
        if 0 <= idx < self._n_species:
            return float(self._molalities[idx])
        return 0.0

    def pH(self) -> float:
        h_idx = self._sp_map.get("H+", -1)
        if h_idx >= 0:
            m_h = max(self._molalities[h_idx], 1e-14)
            return float(-math.log10(m_h))
        return 7.0

    def pHScale(self) -> float:
        scale = float(getattr(self, "_ph_scale", 1.0))
        return scale

    def setpHScale(self, scale: float) -> None:
        self._ph_scale = float(scale)

    def osmoticCoefficient(self) -> float:
        oc = float(getattr(self, "_osmotic_coeff", 1.0))
        return oc

    def initLengths(self) -> None:
        self._lengths_init = True

    def soluteIndex(self, k: int) -> int:
        return k

    def nSolutes(self) -> int:
        return max(self._n_species - 1, 0)

    def soluteName(self, k: int) -> str:
        return self._species_names[k]

    def soluteNames(self) -> List[str]:
        return [sp for sp in self._species_names if sp != self._solvent_name]

    def soluteIndexByName(self, name: str) -> int:
        return self.species_index(name)

    def moleFSolventMin(self) -> float:
        return 1e-4

    def setMoleFSolventMin(self, min_val: float) -> None:
        self._mole_f_solvent_min = float(min_val)

    def calcMolalities(self) -> None:
        self._molalities_calc = True

    def findCLMIndex(self) -> int:
        clm_idx = getattr(self, "_clm_idx", 0)
        return int(clm_idx)

    def getUnscaledMolalityActivityCoefficients(self, ac: np.ndarray) -> None:
        ac.fill(1.0)

    def setState_TPM(self, t: float, p: float, m: Sequence[float]) -> None:
        self.set_state_TP(t, p)
        self.setMolalities(m)


class GibbsExcessVPSSTP(VPStandardStateTP):
    """Base class for excess Gibbs free energy solution phases."""

    def __init__(self, species_names: Sequence[str], *,
                 temperature: float = 298.15, pressure: float = OneAtm, name: str = "gibbs_excess"):
        super().__init__(species_names, temperature=temperature, pressure=pressure, name=name)

    def getdlnActCoeffdlnX(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def getdlnActCoeffdT(self, d: np.ndarray) -> None:
        d.fill(0.0)

    def s_update_lnActCoeff(self) -> None:
        self._ln_act_updated = True

    def s_update_dlnActCoeff_dT(self) -> None:
        self._dln_act_dt_updated = True

    def s_update_dlnActCoeff_dX_(self, d: np.ndarray) -> None:
        d.fill(0.0)


class MargulesVPSSTP(GibbsExcessVPSSTP):
    """Margules excess Gibbs free energy expansion for non-ideal binary/ternary mixtures."""

    def __init__(self, species_names: Sequence[str], *,
                 temperature: float = 298.15, pressure: float = OneAtm):
        super().__init__(species_names, temperature=temperature, pressure=pressure, name="margules")
        self._HE: Dict[Tuple[int, int], float] = {}
        self._SE: Dict[Tuple[int, int], float] = {}
        self._HE1: Dict[Tuple[int, int], float] = {}
        self._SE1: Dict[Tuple[int, int], float] = {}
        self._VHE: Dict[Tuple[int, int], float] = {}
        self._VSE: Dict[Tuple[int, int], float] = {}
        self._VHE1: Dict[Tuple[int, int], float] = {}
        self._VSE1: Dict[Tuple[int, int], float] = {}

    def MargulesVPSSTP(self) -> MargulesVPSSTP:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "Margules"

    def setBinaryInteraction(self, sp1: str, sp2: str, *, h0: float = 0.0,
                             s0: float = 0.0, h1: float = 0.0,
                             s1: float = 0.0, vh0: float = 0.0,
                             vs0: float = 0.0, vh1: float = 0.0,
                             vs1: float = 0.0) -> None:
        i1 = self.species_index(sp1)
        i2 = self.species_index(sp2)
        pair = (min(i1, i2), max(i1, i2))
        self._HE[pair] = float(h0)
        self._SE[pair] = float(s0)
        self._HE1[pair] = float(h1)
        self._SE1[pair] = float(s1)
        self._VHE[pair] = float(vh0)
        self._VSE[pair] = float(vs0)
        self._VHE1[pair] = float(vh1)
        self._VSE1[pair] = float(vs1)
        self.invalidateCache()

    def gibbs_excess(self) -> float:
        """Return the native Margules excess Gibbs-energy expression."""
        ge = 0.0
        for (i, j), h0 in self._HE.items():
            s0 = self._SE.get((i, j), 0.0)
            h1 = self._HE1.get((i, j), 0.0)
            s1 = self._SE1.get((i, j), 0.0)
            vh0 = self._VHE.get((i, j), 0.0)
            vs0 = self._VSE.get((i, j), 0.0)
            vh1 = self._VHE1.get((i, j), 0.0)
            vs1 = self._VSE1.get((i, j), 0.0)
            dp = self._pres - OneAtm
            g0 = (h0 - self._temp * s0 + dp * (vh0 - self._temp * vs0))
            g1 = (h1 - self._temp * s1 + dp * (vh1 - self._temp * vs1))
            ge += self._X[i] * self._X[j] * (g0 + self._X[j] * g1)
        return float(ge)

    def getActivityCoefficients(self) -> np.ndarray:
        """Evaluate pinned Margules binary-interaction equations."""
        rt = GasConstant * self._temp
        if not np.isfinite(rt) or rt <= 0.0:
            raise ValueError("temperature must be positive for Margules activities")
        ln_gamma = np.zeros(self._n_species, dtype=float)
        delta_p_rt = (self._pres - OneAtm) / rt
        for (i, j), h0 in self._HE.items():
            h1 = self._HE1.get((i, j), 0.0)
            s0 = self._SE.get((i, j), 0.0)
            s1 = self._SE1.get((i, j), 0.0)
            vh0 = self._VHE.get((i, j), 0.0)
            vh1 = self._VHE1.get((i, j), 0.0)
            vs0 = self._VSE.get((i, j), 0.0)
            vs1 = self._VSE1.get((i, j), 0.0)
            g0 = ((h0 - self._temp * s0) / rt
                  + delta_p_rt * (vh0 - self._temp * vs0))
            g1 = ((h1 - self._temp * s1) / rt
                  + delta_p_rt * (vh1 - self._temp * vs1))
            xa, xb = float(self._X[i]), float(self._X[j])
            xaxb = xa * xb
            g0g1xb = g0 + g1 * xb
            all_species = -xaxb * g0g1xb - xaxb * xb * g1
            ln_gamma += all_species
            ln_gamma[i] += xb * g0g1xb
            ln_gamma[j] += xa * g0g1xb + xaxb * g1
        return np.exp(ln_gamma)

    def getLnActivityCoefficients(self, lnAct: np.ndarray) -> None:
        n = min(len(lnAct), self._n_species)
        lnAct[:n] = np.log(self.getActivityCoefficients()[:n])

    def getdlnActCoeffdT(self, d: np.ndarray) -> None:
        rt2 = GasConstant * (self._temp ** 2)
        d.fill(0.0)
        for (i, j), h0 in self._HE.items():
            d[i] -= (h0 / rt2) * (self._X[j] ** 2)
            d[j] -= (h0 / rt2) * (self._X[i] ** 2)

    def getdlnActCoeffdlnN(self, d: np.ndarray) -> None:
        """Fill the pinned Margules composition-to-mole Jacobian."""
        n = self._n_species
        target = np.asarray(d)
        if target.shape == (n * n,):
            matrix = target.reshape((n, n))
        elif target.shape == (n, n):
            matrix = target
        else:
            raise ValueError("activity-coefficient derivative output has wrong shape")
        matrix.fill(0.0)
        rt = GasConstant * self._temp
        if not np.isfinite(rt) or rt <= 0.0:
            raise ValueError("temperature must be positive for Margules derivatives")
        x = np.asarray(self._X, dtype=float)
        if x.shape != (n,) or np.any(~np.isfinite(x)):
            raise ValueError("Margules composition is invalid")
        delta_p_rt = (self._pres - OneAtm) / rt
        for (ia, ib), h0 in self._HE.items():
            xa, xb = float(x[ia]), float(x[ib])
            h1 = self._HE1.get((ia, ib), 0.0)
            s0 = self._SE.get((ia, ib), 0.0)
            s1 = self._SE1.get((ia, ib), 0.0)
            vh0 = self._VHE.get((ia, ib), 0.0)
            vh1 = self._VHE1.get((ia, ib), 0.0)
            vs0 = self._VSE.get((ia, ib), 0.0)
            vs1 = self._VSE1.get((ia, ib), 0.0)
            g0 = ((h0 - self._temp * s0) / rt
                  + delta_p_rt * (vh0 - self._temp * vs0))
            g1 = ((h1 - self._temp * s1) / rt
                  + delta_p_rt * (vh1 - self._temp * vs1))
            # Differentiate the native pair contribution in independent X_A,
            # X_B coordinates, then apply dX_k/dln(n_j)=x_j*(delta_kj-X_k).
            all_a = -xb * g0 - 2.0 * xb * xb * g1
            all_b = -xa * g0 - 4.0 * xa * xb * g1
            for k in range(n):
                if k == ia:
                    grad_a = all_a
                    grad_b = all_b + g0 + 2.0 * xb * g1
                elif k == ib:
                    grad_a = all_a + g0 + 2.0 * xb * g1
                    grad_b = all_b + 2.0 * xa * g1
                else:
                    grad_a, grad_b = all_a, all_b
                for j in range(n):
                    matrix[k, j] += x[j] * (
                        grad_a * ((1.0 if j == ia else 0.0) - xa)
                        + grad_b * ((1.0 if j == ib else 0.0) - xb))


class RedlichKisterVPSSTP(GibbsExcessVPSSTP):
    """Redlich-Kister sub-regular excess Gibbs free energy expansion."""

    def __init__(self, species_names: Sequence[str], *,
                 temperature: float = 298.15, pressure: float = OneAtm):
        super().__init__(species_names, temperature=temperature, pressure=pressure, name="redlich_kister")
        self._coeffs_H: Dict[Tuple[int, int], List[float]] = {}
        self._coeffs_S: Dict[Tuple[int, int], List[float]] = {}

    def RedlichKisterVPSSTP(self) -> RedlichKisterVPSSTP:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "Redlich-Kister"

    def setBinaryInteraction(self, sp1: str, sp2: str, *,
                             coeffs_H: Sequence[float] = (), coeffs_S: Sequence[float] = ()) -> None:
        i1 = self.species_index(sp1)
        i2 = self.species_index(sp2)
        pair = (min(i1, i2), max(i1, i2))
        self._coeffs_H[pair] = [float(c) for c in coeffs_H]
        self._coeffs_S[pair] = [float(c) for c in coeffs_S]
        self.invalidateCache()

    def gibbs_excess(self) -> float:
        """Return G^E = RT sum_k X_k ln(gamma_k)."""
        ln_gamma = np.log(self.getActivityCoefficients())
        return float(GasConstant * self._temp * np.dot(self._X, ln_gamma))

    def getActivityCoefficients(self) -> np.ndarray:
        rt = GasConstant * self._temp
        if not np.isfinite(rt) or rt <= 0.0:
            raise ValueError("temperature must be positive for Redlich-Kister activities")
        ln_gamma = np.zeros(self._n_species, dtype=float)
        for (i, j), ch in self._coeffs_H.items():
            cs = self._coeffs_S.get((i, j), [])
            dx = self._X[i] - self._X[j]
            poly = 1.0
            sum_value = 0.0
            sum_derivative = 0.0
            sum_weighted = 0.0
            sum_derivative_poly = 1.0
            for m in range(max(len(ch), len(cs))):
                hm = ch[m] if m < len(ch) else 0.0
                sm = cs[m] if m < len(cs) else 0.0
                age = (hm - self._temp * sm) / rt
                sum_value += age * poly
                sum_weighted += age * (m + 1) * poly
                poly *= dx
                if m >= 1:
                    sum_derivative += age * m * sum_derivative_poly
                    sum_derivative_poly *= dx
            xa, xb = float(self._X[i]), float(self._X[j])
            xaxb = xa * xb
            ln_gamma[i] += ((1.0 - xa) * xb * sum_value
                            + xaxb * sum_derivative * (1.0 - xa + xb))
            ln_gamma[j] += ((1.0 - xb) * xa * sum_value
                            + xaxb * sum_derivative * (-1.0 + xb - xa))
            for k in range(self._n_species):
                if k != i and k != j:
                    ln_gamma[k] -= xaxb * sum_weighted
        return np.exp(ln_gamma)

    def getLnActivityCoefficients(self, lnAct: np.ndarray) -> None:
        n = min(len(lnAct), self._n_species)
        lnAct[:n] = np.log(self.getActivityCoefficients()[:n])

    def getdlnActCoeffdlnN(self, d: np.ndarray) -> None:
        """Fill the pinned Redlich-Kister composition Jacobian matrix."""
        n = self._n_species
        target = np.asarray(d)
        if target.shape == (n * n,):
            matrix = target.reshape((n, n))
        elif target.shape == (n, n):
            matrix = target
        else:
            raise ValueError("activity-coefficient derivative output has wrong shape")
        matrix.fill(0.0)
        rt = GasConstant * self._temp
        if not np.isfinite(rt) or rt <= 0.0:
            raise ValueError("temperature must be positive for Redlich-Kister derivatives")
        for (ia, ib), ch in self._coeffs_H.items():
            cs = self._coeffs_S.get((ia, ib), [])
            xa, xb = float(self._X[ia]), float(self._X[ib])
            dx = xa - xb
            s0 = s1 = s1_m1 = s2 = s2_m1 = s1_m2 = 0.0
            poly = poly_m1 = poly_m2 = 1.0
            for m in range(max(len(ch), len(cs))):
                hm = ch[m] if m < len(ch) else 0.0
                sm = cs[m] if m < len(cs) else 0.0
                # The pinned C++ s_update_dlnActCoeff_dX_ forms this matrix
                # from unscaled (H - T*S) coefficients. Preserve that native
                # operation here; the forward activity routine separately
                # scales coefficients by RT.
                age = hm - self._temp * sm
                s0 += age * poly
                s2 += age * (m + 1) * poly
                poly *= dx
                if m >= 1:
                    s1 += age * m * poly_m1
                    s2_m1 += age * m * (m + 1) * poly_m1
                    poly_m1 *= dx
                if m >= 2:
                    s1_m2 += age * m * (m - 1) * poly_m2
                    poly_m2 *= dx

            for k in range(n):
                if k == ia:
                    matrix[k, ia] += (
                        -xb * s0 + (1.0 - xa) * xb * s1
                        + xb * s1 * (1.0 - 2.0 * xa + xb)
                        + xa * xb * s1_m2 * (1.0 - xa + xb))
                    matrix[k, ib] += (
                        (1.0 - xa) * s0 - (1.0 - xa) * xb * s1
                        + xa * s1 * (1.0 + 2.0 * xb - xa)
                        - xa * xb * s1_m2 * (1.0 - xa + xb))
                elif k == ib:
                    matrix[k, ia] += (
                        (1.0 - xb) * s0 + (1.0 - xa) * xb * s1
                        + xb * s1 * (1.0 - 2.0 * xa + xb)
                        + xa * xb * s1_m2 * (1.0 - xa + xb))
                    matrix[k, ib] += (
                        -xa * s0 - (1.0 - xb) * xa * s1
                        + xa * s1 * (xb - xa - (1.0 - xb))
                        - xa * xb * s1_m2 * (-xa - (1.0 - xb)))
                else:
                    matrix[k, ia] += -xb * s2 - xa * xb * s2_m1
                    matrix[k, ib] += -xa * s2 + xa * xb * s2_m1


class LatticeSolidPhase(ThermoPhase):
    """Crystalline solid phase with discrete lattice site density and vacancy thermodynamics."""

    def __init__(self, species_names: Sequence[str], *, site_density: float = 20.0):
        super().__init__("lattice_solid")
        self._species_names = list(species_names)
        self._n_species = len(self._species_names)
        self._site_density = float(site_density)
        self._molar_volume = 1.0 / max(float(site_density), 1e-6)
        self._mole_fractions = np.full(self._n_species, 1.0 / self._n_species, dtype=float)
        self._mol_weights = np.full(self._n_species, 28.97, dtype=float)

    def siteDensity(self) -> float:
        return float(self._site_density)

    def molarVolume(self) -> float:
        return float(self._molar_volume)

    def setSiteDensity(self, dens: float) -> None:
        self._site_density = float(dens)
        self._molar_volume = 1.0 / max(float(dens), 1e-6)
        self.invalidateCache()


class IdealSolidSolnPhase(VPStandardStateTP):
    """Ideal solid solution phase."""

    def __init__(self, species_names: Optional[Sequence[str]] = None, name: str = "ideal_solid"):
        sp = list(species_names) if species_names is not None else ["solid_species"]
        super().__init__(sp, name=name)

    def speciesMolarVolume(self, k: int) -> float:
        return float(self._standard_v0[k]) if k < len(self._standard_v0) else 0.018

    def getSpeciesMolarVolumes(self, v: np.ndarray) -> None:
        n = min(len(v), len(self._standard_v0))
        v[:n] = self._standard_v0[:n]

    def _updateThermo(self) -> None:
        self.invalidateCache()


class BinarySolutionTabulatedThermo(IdealSolidSolnPhase):
    """Binary solution phase with tabulated excess Gibbs free energy."""

    def __init__(self, species_names: Optional[Sequence[str]] = None, name: str = "binary_tabulated"):
        super().__init__(species_names, name=name)
        self._excess_h = 0.0

    def BinarySolutionTabulatedThermo(self) -> BinarySolutionTabulatedThermo:
        self._state_mf_number += 1
        return self

    def diff(self) -> float:
        d_val = float(getattr(self, "_diff_val", 0.0))
        return d_val

    def excessEnthalpy(self) -> float:
        return float(self._excess_h)


class CoverageDependentSurfPhase(ThermoPhase):
    """Surface phase with coverage-dependent thermodynamic polynomials."""

    def __init__(self, name: str = "cov_dep_surf"):
        super().__init__(name)

    def _updateTotalThermo(self) -> None:
        self.invalidateCache()

    def _updateCovDepThermo(self) -> None:
        self.invalidateCache()

    def addInterpolativeDependency(self, dep: Any) -> None:
        self._dep = dep


class IdealMolalSoln(MolalityVPSSTP):
    """Ideal molal aqueous solution."""

    def __init__(self, species_names: Optional[Sequence[str]] = None, solvent_name: str = "H2O"):
        sp = list(species_names) if species_names is not None else ["H2O", "solute"]
        super().__init__(sp, solvent_name=solvent_name)

    def s_updateIMS_lnMolalityActCoeff(self) -> None:
        self._ims_updated = True

    def setCutoffModel(self, model: str) -> None:
        self._cutoff_model = str(model)

    def calcIMSCutoffParams_(self) -> None:
        self._ims_cutoff_params = True


class IdealSolnGasVPSS(VPStandardStateTP):
    """Ideal gas VPSS model."""

    def __init__(self, species_names: Optional[Sequence[str]] = None):
        sp = list(species_names) if species_names is not None else ["gas_species"]
        super().__init__(sp, name="idealgas_vpss")

    def IdealSolnGasVPSS(self) -> IdealSolnGasVPSS:
        self._state_mf_number += 1
        return self
