"""Editable scalar reaction-rate equations translated from Cantera.

Derived from Cantera commit 726522be4e2a13454d8415b7ef799d621f665cf3:
``include/cantera/kinetics/{Arrhenius,TwoTempPlasmaRate,BlowersMaselRate,
ElectronCollisionPlasmaRate,Falloff,PlogRate,ChebyshevRate}.h`` and their
``src/kinetics/*.cpp`` files.
See the accompanying LICENSE.txt and provenance/rates.json for attribution,
method coverage, and deliberate differences from the native API.

All calculations use Python's standard library. Temperatures are kelvin,
pressures pascal, energies J/kmol, and third-body concentrations kmol/m^3.
Pre-exponential factors must already have the appropriate SI/kmol units for
the reaction order. There is no implicit mechanism parsing or unit conversion.
These are rate-coefficient equations, not a complete kinetics mechanism.

The domain requires finite parameters and positive finite temperatures and
pressures. Undefined arithmetic raises Python exceptions; overflow and invalid
states do not promise the same IEEE-754 behavior as the C++ implementation.
"""

from bisect import bisect_right
from dataclasses import dataclass, field
import math

from .constants import Avogadro, ElectronCharge, ElectronMass, CanteraError
from .functions import linearInterp, simpson


GAS_CONSTANT = 8314.46261815324  # J / kmol / K; Cantera's GasConstant
SMALL_NUMBER = 1.0e-300  # include/cantera/base/ct_defs.h

__all__ = [
    "ArrheniusRate", "TwoTempPlasmaRate", "BlowersMaselRate",
    "ElectronCollisionPlasmaRate",
    "LindemannRate", "TroeRate", "SriRate", "TsangRate",
    "PlogRate", "ChebyshevRate", "LinearBurkeCollider", "LinearBurkeRate",
    "CustomFunc1Rate",
    "VibrationalRelaxationRate", "ConstantVibrationalRelaxationRate",
    "CastelaVibrationalRelaxationRate", "StarikovskiyVibrationalRelaxationRate",
    "MultiStateResolvedVibrationalRelaxationRate",
    "InterfaceArrheniusRate", "StickingArrheniusRate",
    "InterfaceBlowersMaselRate", "StickingBlowersMaselRate",
]


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive(value, name):
    value = _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _coefficients(values, counts):
    values = tuple(_finite(value, "coefficient") for value in values)
    if len(values) not in counts:
        raise ValueError(f"Expected {counts} coefficients; received {len(values)}")
    return values


def _check_A(value, allow_negative_A):
    if value < 0.0 and not allow_negative_A:
        raise ValueError("Negative A requires allow_negative_A=True")


@dataclass
class ArrheniusRate:
    """``k(T) = A * T**b * exp(-Ea / (R*T))``.

    ``Ea`` is J/kmol. Negative A must be explicitly enabled. The logarithm
    method returns NaN for A <= 0, as does upstream ArrheniusRate::evalLog.
    """

    A: float
    b: float
    Ea: float = 0.0
    allow_negative_A: bool = False
    _order: float = field(default=math.nan, init=False, repr=False)

    def __post_init__(self):
        self.A = _finite(self.A, "A")
        self.b = _finite(self.b, "b")
        self.Ea = _finite(self.Ea, "Ea")
        _check_A(self.A, self.allow_negative_A)

    def __call__(self, T):
        T = _positive(T, "T")
        return self.A * math.exp(self.b * math.log(T) - (self.Ea / GAS_CONSTANT) / T)

    def log_rate(self, T):
        """Natural logarithm of k, without first exponentiating the rate."""
        T = _positive(T, "T")
        log_A = math.log(self.A) if self.A > 0.0 else math.nan
        return log_A + self.b * math.log(T) - (self.Ea / GAS_CONSTANT) / T

    def ddT_scaled(self, T):
        """Return (dk/dT)/k analytically, as upstream ddTScaledFromStruct."""
        T = _positive(T, "T")
        recip_T = 1.0 / T
        return ((self.Ea / GAS_CONSTANT) * recip_T + self.b) * recip_T

    @property
    def order(self):
        """Reaction order implied by the installed rate-coefficient units.

        Pinned ``ArrheniusBase`` leaves this as NaN until kinetics supplies a
        rate-unit stack. Mechanism construction calls :meth:`set_rate_order`
        after deriving those units from the reaction orders.
        """
        return self._order

    def set_rate_order(self, order):
        order = float(order)
        if not math.isfinite(order):
            raise ValueError("reaction order must be finite")
        self._order = order
        return self


class TwoTempPlasmaRate(ArrheniusRate):
    """Gas/electron-temperature rate, including upstream bg and Tinv terms.

    ``k = A*T**bg*Te**b * exp(-Ea/(R*T) + EE*(Te-T)/(R*Te*T)
    - T/Tinv)``; the last term is omitted when ``Tinv == 0``.
    ``Ea`` and ``EE`` are J/kmol. This expression does not supply a reverse
    rate from equilibrium thermochemistry.
    """

    def __init__(self, A=0.0, b=0.0, Ea=0.0, EE=0.0, bg=0.0, Tinv=0.0, allow_negative_A=False):
        super().__init__(A=A, b=b, Ea=Ea, allow_negative_A=allow_negative_A)
        self.EE = float(EE)
        self.bg = float(bg)
        self.Tinv = float(Tinv)
        self.m_E4_R = self.EE / GAS_CONSTANT
        self.m_recip_Tinv = 1.0 / self.Tinv if self.Tinv != 0.0 else 0.0
        self._valid = True

    def __call__(self, T, Te=None):
        if Te is None:
            Te = T
        T = _positive(T, "T")
        Te = _positive(Te, "Te")
        recip_T, recip_Te = 1.0 / T, 1.0 / Te
        recip_Tinv = 1.0 / self.Tinv if self.Tinv != 0.0 else 0.0
        return self.A * math.exp(
            self.bg * math.log(T) + self.b * math.log(Te)
            - (self.Ea / GAS_CONSTANT) * recip_T
            + (self.EE / GAS_CONSTANT) * (Te - T) * recip_Te * recip_T
            - T * recip_Tinv
        )

    def ddT_scaled(self, T, Te=None):
        """Return (partial k/partial T)/k with electron temperature FIXED."""
        if Te is None:
            Te = T
        T = _positive(T, "T")
        _positive(Te, "Te")
        recip_T = 1.0 / T
        recip_Tinv = 1.0 / self.Tinv if self.Tinv != 0.0 else 0.0
        return (self.bg * recip_T
                + (self.Ea / GAS_CONSTANT - self.EE / GAS_CONSTANT)
                * recip_T * recip_T - recip_Tinv)

    def type(self) -> str:
        return "two-temperature-plasma"

    def newMultiRate(self):
        return MultiRate()

    def setContext(self, rxn=None, kin=None):
        if rxn is not None and getattr(rxn, "reversible", False):
            raise CanteraError("TwoTempPlasmaRate does not support reversible reactions")

    def evalFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        Te = getattr(shared_data, "electronTemp", getattr(shared_data, "Te", T))
        return self(T, Te)

    def ddTScaledFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        Te = getattr(shared_data, "electronTemp", getattr(shared_data, "Te", T))
        return self.ddT_scaled(T, Te)

    def activationElectronEnergy(self) -> float:
        return self.EE

    def setParameters(self, node: dict, rate_units=None) -> None:
        self.bg = 0.0
        self.m_recip_Tinv = 0.0
        if not isinstance(node, dict):
            return
        if "rate-constant" in node:
            rc = node["rate-constant"]
            if isinstance(rc, dict):
                self.A = float(rc.get("A", 0.0))
                self.b = float(rc.get("b", 0.0))
                self.Ea = float(rc.get("Ea-gas", rc.get("Ea", 0.0)))
                self.EE = float(rc.get("Ea-electron", rc.get("EE", 0.0)))
                self.bg = float(rc.get("b-gas", rc.get("bg", 0.0)))
                tinv = float(rc.get("T-inv", rc.get("Tinv", 0.0)))
                self.Tinv = tinv
                self.m_recip_Tinv = 1.0 / tinv if tinv != 0.0 else 0.0
                self.m_E4_R = self.EE / GAS_CONSTANT
        self._parameters = dict(node)

    def getParameters(self, node: dict = None) -> dict:
        out = {
            "type": self.type(),
            "rate-constant": {
                "A": self.A,
                "b": self.b,
                "Ea-gas": self.Ea,
                "Ea-electron": self.EE
            }
        }
        if self.bg != 0.0:
            out["rate-constant"]["b-gas"] = self.bg
        if self.m_recip_Tinv != 0.0:
            out["rate-constant"]["T-inv"] = 1.0 / self.m_recip_Tinv
        if node is not None and isinstance(node, dict):
            node.update(out)
        return out


class ElectronCollisionPlasmaRate:
    """Electron-collision rate from an EEDF and tabulated cross sections.

    Energy grids are in eV, cross sections in m2, and EEDF values in eV**-3/2.
    Calling an instance with a shared EEDF grid returns the forward coefficient
    in m3/kmol/s. :meth:`superelastic_rate` applies the pinned detailed-balance
    cross-section shift for a reversible super-elastic process.
    """

    _prefactor = math.sqrt(2.0 * ElectronCharge / ElectronMass) * Avogadro

    def __init__(self, energy_levels, cross_sections, *, kind="", target="",
                 product="", threshold=0.0):
        self.energy_levels = tuple(_finite(value, "energy level") for value in energy_levels)
        self.cross_sections = tuple(_finite(value, "cross section") for value in cross_sections)
        if len(self.energy_levels) < 2 or len(self.energy_levels) != len(self.cross_sections):
            raise ValueError("energy_levels and cross_sections must have equal length of at least two")
        if any(right <= left for left, right in zip(self.energy_levels, self.energy_levels[1:])):
            raise ValueError("energy_levels must be strictly increasing")
        self.kind, self.target, self.product = str(kind), str(target), str(product)
        self.threshold = _finite(threshold, "threshold")

    @staticmethod
    def _shared_inputs(energy_levels, distribution):
        levels = tuple(_finite(value, "shared energy level") for value in energy_levels)
        eedf = tuple(_finite(value, "electron energy distribution") for value in distribution)
        if len(levels) < 2 or len(levels) != len(eedf):
            raise ValueError("shared energy levels and distribution must have equal length of at least two")
        if any(right <= left for left, right in zip(levels, levels[1:])):
            raise ValueError("shared energy levels must be strictly increasing")
        return levels, eedf

    def interpolated_cross_sections(self, shared_energy_levels):
        """Linear-interpolate source cross sections on an EEDF energy grid."""
        levels = tuple(_finite(value, "shared energy level") for value in shared_energy_levels)
        return tuple(linearInterp(level, self.energy_levels, self.cross_sections) for level in levels)

    def __call__(self, shared_energy_levels, distribution):
        """Return the forward coefficient using ``simpson(epsilon * F0 * sigma)``."""
        levels, eedf = self._shared_inputs(shared_energy_levels, distribution)
        cross_sections = self.interpolated_cross_sections(levels)
        integrand = tuple(energy * value * section
                          for energy, value, section in zip(levels, eedf, cross_sections))
        return self._prefactor * simpson(integrand, levels)

    
    def type(self) -> str:
        return "electron-collision-plasma"

    def newMultiRate(self):
        return MultiRate()

    def setContext(self, rxn=None, kin=None):
        pass

    def evalFromStruct(self, shared_data) -> float:
        levels = getattr(shared_data, "energyLevels", ())
        dist = getattr(shared_data, "distribution", ())
        if not levels or not dist:
            return 0.0
        return self(levels, dist)

    def ddTScaledFromStruct(self, shared_data) -> float:
        raise NotImplementedError("ElectronCollisionPlasmaRate::ddTScaledFromStruct is not implemented")

    def setParameters(self, node: dict, units=None) -> None:
        if isinstance(node, dict):
            if "energy-levels" in node:
                self.energy_levels = tuple(float(x) for x in node["energy-levels"])
            if "cross-sections" in node:
                self.cross_sections = tuple(float(x) for x in node["cross-sections"])
            self._parameters = dict(node)

    def getParameters(self, node: dict = None) -> dict:
        out = {
            "type": self.type(),
            "energy-levels": list(self.energy_levels),
            "cross-sections": list(self.cross_sections)
        }
        if node is not None and isinstance(node, dict):
            node.update(out)
        return out

    def energyLevels(self) -> list:
        return list(self.energy_levels)

    def crossSections(self) -> list:
        return list(self.cross_sections)

    def setEnergyLevels(self, levels) -> None:
        self.energy_levels = tuple(float(x) for x in levels)

    def setCrossSections(self, sections) -> None:
        self.cross_sections = tuple(float(x) for x in sections)
    def superelastic_rate(self, shared_energy_levels, distribution):
        """Return the pinned super-elastic reverse coefficient.

        The source offsets the cross-section interpolation grid by the first
        tabulated energy (the collision threshold), then integrates
        ``(epsilon + threshold) * F0 * sigma_super``.
        """
        levels, eedf = self._shared_inputs(shared_energy_levels, distribution)
        threshold = self.energy_levels[0]
        offset_levels = (0.0,) + tuple(level - threshold for level in self.energy_levels[1:])
        offset_sections = tuple(linearInterp(level, offset_levels, self.cross_sections)
                                for level in levels)
        integrand = tuple((energy + threshold) * value * section
                          for energy, value, section in zip(levels, eedf, offset_sections))
        return self._prefactor * simpson(integrand, levels)


class CustomFunc1Rate:
    """Pinned ``CustomFunc1Rate`` adapter for a scalar temperature function.

    A supplied callable is retained directly. Objects exposing an ``eval(T)``
    method, such as the local :class:`cantera_python.functions.Func1`, are
    also accepted. The unconfigured evaluation path returns NaN, matching the
    native evaluator; :meth:`validate` reports the missing function.
    """
    type = "custom-rate-function"
    def type(self) -> str:
        return "custom-rate-function"

    def __init__(self, rate_function=None):
        self._rate_function = None
        if rate_function is not None:
            self.set_rate_function(rate_function)

    def set_rate_function(self, rate_function):
        if callable(rate_function):
            self._rate_function = rate_function
        elif callable(getattr(rate_function, "eval", None)):
            self._rate_function = rate_function.eval
        else:
            raise TypeError("custom rate function must be callable or provide eval(T)")

    def validate(self, equation=""):
        if self._rate_function is None:
            raise ValueError("Rate object for reaction '{}' is not configured.".format(equation))

    def __call__(self, temperature):
        if self._rate_function is None:
            return math.nan
        return float(self._rate_function(temperature))


    def newMultiRate(self):
        return MultiRate()

    def setRateFunction(self, f):
        self.set_rate_function(f)

    def evalFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        if self._rate_function is None:
            return float("nan")
        return float(self._rate_function(T))

    def getParameters(self, rateNode: dict = None, rate_units=None) -> dict:
        out = {"type": self.type}
        if rateNode is not None and isinstance(rateNode, dict):
            rateNode.update(out)
        return out
@dataclass
class BlowersMaselRate:
    """Arrhenius expression with activation energy set by reaction enthalpy.

    ``Ea0`` (intrinsic barrier), ``w`` (bond energy), and the supplied reaction
    enthalpy are J/kmol. Reaction enthalpy must be computed by the caller from
    the actual state and stoichiometry; it is not independently predicted here.
    """

    A: float
    b: float
    Ea0: float
    w: float
    allow_negative_A: bool = False
    _delta_enthalpy: float = field(default=0.0, init=False, repr=False)
    _order: float = field(default=math.nan, init=False, repr=False)

    def __post_init__(self):
        for name in ("A", "b", "Ea0", "w"):
            setattr(self, name, _finite(getattr(self, name), name))
        _check_A(self.A, self.allow_negative_A)

    def _activation_temperature(self, delta_enthalpy):
        delta_H_R = _finite(delta_enthalpy, "delta_enthalpy") / GAS_CONSTANT
        Ea_R, w_R = self.Ea0 / GAS_CONSTANT, self.w / GAS_CONSTANT
        # The <= boundary below is intentional: this is the C++ implementation,
        # not the strict inequality printed in the upstream mathematical comment.
        if delta_H_R <= -4.0 * Ea_R:
            return 0.0
        if delta_H_R > 4.0 * Ea_R:
            return delta_H_R
        vp = 2.0 * w_R * ((w_R + Ea_R) / (w_R - Ea_R))
        vp_2w_dH = vp - 2.0 * w_R + delta_H_R
        return ((w_R + delta_H_R / 2.0) * (vp_2w_dH * vp_2w_dH)
                / (vp * vp - 4.0 * w_R * w_R + delta_H_R * delta_H_R))

    @property
    def order(self):
        return self._order

    def set_rate_order(self, order):
        order = float(order)
        if not math.isfinite(order):
            raise ValueError("reaction order must be finite")
        self._order = order
        return self

    @property
    def delta_enthalpy(self):
        return self._delta_enthalpy

    def set_delta_enthalpy(self, delta_enthalpy):
        """Set the current reaction enthalpy, matching pinned ``setDeltaH``."""
        self._delta_enthalpy = _finite(delta_enthalpy, "delta_enthalpy")
        return self

    def update_from_thermo(self, thermo, reactants, products):
        """Update reaction enthalpy from the live phase's partial enthalpies."""
        names = tuple(thermo.species_names)
        enthalpies = tuple(thermo.partial_molar_enthalpies)
        if len(names) != len(enthalpies):
            raise ValueError("species names and partial molar enthalpies differ in length")
        index = {name: i for i, name in enumerate(names)}
        unknown = (set(reactants) | set(products)) - set(index)
        if unknown:
            raise ValueError(f"reaction species absent from thermo state: {sorted(unknown)}")
        delta_h = sum(float(nu) * enthalpies[index[name]]
                      for name, nu in products.items())
        delta_h -= sum(float(nu) * enthalpies[index[name]]
                       for name, nu in reactants.items())
        return self.set_delta_enthalpy(delta_h)

    def activation_energy(self, delta_enthalpy=None):
        """Effective barrier, J/kmol, with the three upstream branches."""
        if delta_enthalpy is None:
            delta_enthalpy = self._delta_enthalpy
        return self._activation_temperature(delta_enthalpy) * GAS_CONSTANT

    def __call__(self, T, delta_enthalpy=None):
        T = _positive(T, "T")
        if delta_enthalpy is None:
            delta_enthalpy = self._delta_enthalpy
        Ea_R = self._activation_temperature(delta_enthalpy)
        return self.A * math.exp(self.b * math.log(T) - Ea_R / T)

    def ddT_scaled(self, T, delta_enthalpy=None):
        """Return (partial k/partial T)/k with reaction enthalpy FIXED.

        Changes in reaction enthalpy with temperature are excluded, matching
        BlowersMaselRate::ddTScaledFromStruct, not a total mechanism derivative.
        """
        T = _positive(T, "T")
        if delta_enthalpy is None:
            delta_enthalpy = self._delta_enthalpy
        recip_T = 1.0 / T
        return (self._activation_temperature(delta_enthalpy) * recip_T
                + self.b) * recip_T


class _FalloffRate:
    """Common falloff expression; public concrete forms appear below."""

    def __init__(self, low_rate, high_rate, *, chemically_activated=False,
                 allow_negative_A=False):
        if not isinstance(low_rate, ArrheniusRate) or not isinstance(high_rate, ArrheniusRate):
            raise TypeError("low_rate and high_rate must be ArrheniusRate objects")
        _check_A(low_rate.A, allow_negative_A)
        _check_A(high_rate.A, allow_negative_A)
        if low_rate.A * high_rate.A < 0.0:
            raise ValueError("Low- and high-pressure pre-exponential factors must have the same sign")
        self.low_rate = low_rate
        self.high_rate = high_rate
        self.chemically_activated = bool(chemically_activated)
        self.allow_negative_A = bool(allow_negative_A)

    def _state(self, T, third_body_concentration):
        T = _positive(T, "T")
        M = _finite(third_body_concentration, "third_body_concentration")
        if M < 0.0:
            raise ValueError("third_body_concentration must be nonnegative")
        low, high = self.low_rate(T), self.high_rate(T)
        reduced_pressure = M * low / (high + SMALL_NUMBER)
        return T, low, high, reduced_pressure

    def falloff_coefficient(self, T, third_body_concentration):
        """Return F(T,Pr) using Pr = [M]*k0/(kinf + 1e-300)."""
        T, _, _, reduced_pressure = self._state(T, third_body_concentration)
        return self.falloff_function(T, reduced_pressure)

    def __call__(self, T, third_body_concentration):
        """Return k_inf*Pr*F/(1+Pr), or k0*F/(1+Pr) if activated.

        Third-body concentration is already included in these expressions.
        The chemically activated form uses different low/high rate units;
        the caller is responsible for supplying the appropriate parameters.
        """
        T, low, high, reduced_pressure = self._state(T, third_body_concentration)
        factor = self.falloff_function(T, reduced_pressure) / (1.0 + reduced_pressure)
        if self.chemically_activated:
            return factor * low
        reduced_pressure *= factor
        return reduced_pressure * high


class LindemannRate(_FalloffRate):
    """Lindemann falloff: F(T,Pr) = 1."""

    def falloff_function(self, T, reduced_pressure):
        _positive(T, "T")
        _finite(reduced_pressure, "reduced_pressure")
        return 1.0


def _troe_broadening(reduced_pressure, log_Fcent):
    log_Pr = math.log10(max(reduced_pressure, SMALL_NUMBER))
    cc = -0.4 - 0.67 * log_Fcent
    nn = 0.75 - 1.27 * log_Fcent
    numerator = log_Pr + cc
    denominator = nn - 0.14 * numerator
    if denominator == 0.0:
        # At this pole C++ gives f1=+/-inf, so log10(F)=0 and F=1.
        # If both terms vanish the expression is undefined.
        return 1.0 if numerator != 0.0 else math.nan
    f1 = numerator / denominator
    return 10.0 ** (log_Fcent / (1.0 + f1 * f1))


class TroeRate(_FalloffRate):
    """Troe falloff with coefficients (a, T3, T1[, T2]).

    The T3/T1 terms are omitted when abs(Ti)<1e-300. The T2 term is omitted
    only when T2 is exactly zero, as in Cantera's evaluation method.
    """

    def __init__(self, low_rate, high_rate, coeffs, **kwargs):
        super().__init__(low_rate, high_rate, **kwargs)
        self.coeffs = _coefficients(coeffs, (3, 4))

    def falloff_function(self, T, reduced_pressure):
        T = _positive(T, "T")
        reduced_pressure = _finite(reduced_pressure, "reduced_pressure")
        a, T3, T1 = self.coeffs[:3]
        recip_T3 = math.inf if abs(T3) < SMALL_NUMBER else 1.0 / T3
        recip_T1 = math.inf if abs(T1) < SMALL_NUMBER else 1.0 / T1
        Fcent = ((1.0 - a) * math.exp(-T * recip_T3)
                 + a * math.exp(-T * recip_T1))
        T2 = self.coeffs[3] if len(self.coeffs) == 4 else 0.0
        if T2 != 0.0:
            Fcent += math.exp(-T2 / T)
        log_Fcent = math.log10(max(Fcent, SMALL_NUMBER))
        return _troe_broadening(reduced_pressure, log_Fcent)


class SriRate(_FalloffRate):
    """SRI falloff with coefficients (a, b, c[, d, e]).

    F = (a*exp(-b/T) + exp(-T/c))**(1/(1+log10(Pr)**2)) * d*T**e.
    c=0 omits its exponential. Defaults are d=1 and e=0. Both c and d
    must be nonnegative, matching the upstream executable checks.
    """

    def __init__(self, low_rate, high_rate, coeffs, **kwargs):
        super().__init__(low_rate, high_rate, **kwargs)
        self.coeffs = _coefficients(coeffs, (3, 5))
        if self.coeffs[2] < 0.0:
            raise CanteraError("SRI c must be nonnegative")
        if len(self.coeffs) == 5 and self.coeffs[3] < 0.0:
            raise CanteraError("SRI d must be nonnegative")

    def falloff_function(self, T, reduced_pressure):
        T = _positive(T, "T")
        reduced_pressure = _finite(reduced_pressure, "reduced_pressure")
        a, b, c = self.coeffs[:3]
        d, e = self.coeffs[3:] if len(self.coeffs) == 5 else (1.0, 0.0)
        temperature_part = a * math.exp(-b / T)
        if c != 0.0:
            temperature_part += math.exp(-T / c)
        multiplier = d * math.pow(T, e)
        log_Pr = math.log10(max(reduced_pressure, SMALL_NUMBER))
        exponent = 1.0 / (1.0 + log_Pr * log_Pr)
        return math.pow(temperature_part, exponent) * multiplier


class TsangRate(_FalloffRate):
    """Tsang falloff: Fcent = a + b*T, with coefficients (a[, b])."""

    def __init__(self, low_rate, high_rate, coeffs, **kwargs):
        super().__init__(low_rate, high_rate, **kwargs)
        self.coeffs = _coefficients(coeffs, (1, 2))

    def falloff_function(self, T, reduced_pressure):
        T = _positive(T, "T")
        reduced_pressure = _finite(reduced_pressure, "reduced_pressure")
        a = self.coeffs[0]
        b = self.coeffs[1] if len(self.coeffs) == 2 else 0.0
        log_Fcent = math.log10(max(a + b * T, SMALL_NUMBER))
        return _troe_broadening(reduced_pressure, log_Fcent)


class PlogRate:
    """Log-pressure interpolation of Arrhenius rates at pressures in Pa.

    Pass an iterable of (pressure, ArrheniusRate) pairs. Duplicate-pressure
    terms are summed in supplied order. Outside the supplied pressure range,
    the nearest group is used. A 1e-300 additive floor before logarithms
    preserves the upstream calculation even when all terms underflow.
    """

    def __init__(self, rates):
        groups = {}
        for pressure, rate in rates:
            pressure = _positive(pressure, "pressure")
            if not isinstance(rate, ArrheniusRate):
                raise TypeError("P-log entries must contain ArrheniusRate objects")
            log_P = math.log(pressure)
            groups.setdefault(log_P, []).append(rate)
        if not groups:
            raise ValueError("P-log requires at least one pressure/rate pair")
        self.log_pressures = tuple(sorted(groups))
        self.groups = tuple(tuple(groups[log_P]) for log_P in self.log_pressures)

    def _group_log_rate(self, group, T):
        rate_sum = SMALL_NUMBER
        for rate in group:
            rate_sum += rate(T)
        if rate_sum <= 0.0:
            raise ValueError(f"P-log group sum must be positive at T={T}")
        return math.log(rate_sum)

    def __call__(self, T, P):
        T = _positive(T, "T")
        log_P = math.log(_positive(P, "P"))
        upper = bisect_right(self.log_pressures, log_P)
        if upper == 0:
            return math.exp(self._group_log_rate(self.groups[0], T))
        if upper == len(self.groups):
            return math.exp(self._group_log_rate(self.groups[-1], T))
        lower = upper - 1
        low_k = self._group_log_rate(self.groups[lower], T)
        high_k = self._group_log_rate(self.groups[upper], T)
        recip_delta_P = 1.0 / (self.log_pressures[upper] - self.log_pressures[lower])
        return math.exp(low_k + (high_k - low_k)
                        * (log_P - self.log_pressures[lower]) * recip_delta_P)

    def validate(self, temperatures=(300.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)):
        """Check strictly positive group sums at upstream's six sample temperatures.

        Passing this finite sample does not establish positivity for every T.
        The additive numerical floor used in evaluation is excluded here,
        matching PlogRate::validate.
        """
        temperatures = tuple(_positive(T, "T") for T in temperatures)
        for log_P, group in zip(self.log_pressures, self.groups):
            for T in temperatures:
                rate_sum = 0.0
                for rate in group:
                    rate_sum += rate(T)
                if not rate_sum > 0.0:
                    raise ValueError(f"Invalid P-log rate sum at P={math.exp(log_P)}, T={T}")


class ChebyshevRate:
    """Bivariate Chebyshev polynomial for log10(k), rows T and columns P.

    Coefficients must already express rate coefficients in SI/kmol units.
    The upstream reciprocal-temperature and log10-pressure coordinate maps
    and polynomial recurrences are used. Values outside the fitted range
    are evaluated as polynomial extrapolations, without clamping.
    """

    def __init__(self, Tmin, Tmax, Pmin, Pmax, coeffs):
        self.Tmin = _positive(Tmin, "Tmin")
        self.Tmax = _positive(Tmax, "Tmax")
        self.Pmin = _positive(Pmin, "Pmin")
        self.Pmax = _positive(Pmax, "Pmax")
        if self.Tmax <= self.Tmin or self.Pmax <= self.Pmin:
            raise ValueError("Chebyshev limits require Tmin<Tmax and Pmin<Pmax")
        self.coeffs = tuple(tuple(_finite(a, "coefficient") for a in row) for row in coeffs)
        if not self.coeffs or not self.coeffs[0]:
            raise ValueError("Chebyshev coefficient matrix must be nonempty")
        if any(len(row) != len(self.coeffs[0]) for row in self.coeffs):
            raise ValueError("Chebyshev coefficient matrix must be rectangular")

    def reduced_coordinates(self, T, P):
        T = _positive(T, "T")
        P = _positive(P, "P")
        Tmin_inv, Tmax_inv = 1.0 / self.Tmin, 1.0 / self.Tmax
        log_Pmin, log_Pmax = math.log10(self.Pmin), math.log10(self.Pmax)
        Tr = (2.0 / T - Tmin_inv - Tmax_inv) * (1.0 / (Tmax_inv - Tmin_inv))
        Pr = (2.0 * math.log10(P) - log_Pmin - log_Pmax) * (1.0 / (log_Pmax - log_Pmin))
        return Tr, Pr

    def log10_rate(self, T, P):
        """Evaluate the polynomial for log10(k) without exponentiation."""
        Tr, Pr = self.reduced_coordinates(T, P)
        dot_products = [row[0] for row in self.coeffs]
        Cnm1, Cn = Pr, 1.0
        for column in range(1, len(self.coeffs[0])):
            Cnp1 = 2.0 * Pr * Cn - Cnm1
            for i, row in enumerate(self.coeffs):
                dot_products[i] += Cnp1 * row[column]
            Cnm1, Cn = Cn, Cnp1
        log_k = dot_products[0]
        Cnm1, Cn = Tr, 1.0
        for coefficient in dot_products[1:]:
            Cnp1 = 2.0 * Tr * Cn - Cnm1
            log_k += Cnp1 * coefficient
            Cnm1, Cn = Cn, Cnp1
        return log_k

    def __call__(self, T, P):
        return math.pow(10.0, self.log10_rate(T, P))


@dataclass(frozen=True)
class LinearBurkeCollider:
    """One explicit collider in the reduced-pressure linear mixture rule."""
    efficiency: ArrheniusRate
    rate: object = None

    def __post_init__(self):
        if not isinstance(self.efficiency, ArrheniusRate):
            raise TypeError("LinearBurke collider efficiency must be an ArrheniusRate")
        if self.rate is not None and not isinstance(self.rate, (PlogRate, TroeRate, ChebyshevRate)):
            raise TypeError("LinearBurke collider rate must be PlogRate, TroeRate, ChebyshevRate, or None")


class LinearBurkeRate:
    """Reduced-pressure linear mixture rule (LMR-R) from ``LinearBurkeRate``.

    ``reference_rate`` is the required source collider ``M`` rate. Explicit
    colliders map species names to :class:`LinearBurkeCollider` entries. Mole
    fractions are not normalized, matching the source's direct summation.
    """

    def __init__(self, reference_rate, colliders=None):
        if not isinstance(reference_rate, (PlogRate, TroeRate, ChebyshevRate)):
            raise TypeError("reference_rate must be PlogRate, TroeRate, or ChebyshevRate")
        self.reference_rate = reference_rate
        self.colliders = dict(colliders or {})
        if any(not isinstance(name, str) or not name for name in self.colliders):
            raise ValueError("collider names must be nonempty strings")
        if any(not isinstance(collider, LinearBurkeCollider)
               for collider in self.colliders.values()):
            raise TypeError("colliders must map species names to LinearBurkeCollider values")

    @staticmethod
    def _evaluate_rate(rate, T, pressure):
        if isinstance(rate, TroeRate):
            return rate(T, pressure / (GAS_CONSTANT * T))
        return rate(T, pressure)

    def __call__(self, T, P, mole_fractions):
        T = _positive(T, "T")
        P = _positive(P, "P")
        fractions = {name: _finite(value, "mole fraction")
                     for name, value in mole_fractions.items()}
        if any(value < 0.0 for value in fractions.values()):
            raise ValueError("mole fractions must be nonnegative")

        sigma_x_m = sum(fractions.values())
        eps_mix = 0.0
        evaluated = []
        for name, collider in self.colliders.items():
            x = fractions.get(name, 0.0)
            eps1 = collider.efficiency(T)
            eps_mix += x * eps1
            sigma_x_m -= x
            # No distinct rate: source uses the M rate and eps2=eps_M=1.
            eps2 = eps1 if collider.rate is not None else 1.0
            evaluated.append((x, eps1, eps2, collider.rate or self.reference_rate))
        eps_mix += sigma_x_m
        if eps_mix <= 0.0 or not math.isfinite(eps_mix):
            raise ValueError("Linear-Burke epsilon mixture must be finite and positive")

        value = 0.0
        for x, eps1, eps2, rate in evaluated:
            value += (self._evaluate_rate(rate, T, P * eps_mix / eps2)
                      * eps1 * x / eps_mix)
        return value + (self._evaluate_rate(self.reference_rate, T, P * eps_mix)
                        * sigma_x_m / eps_mix)


class VibrationalRelaxationRate:
    """Common implementation for vibrational relaxation reaction rates.

    k_f = A * T^b * exp(C0 + C13 * T^(-1/3) + Cm * T^(-m) + Cn * T^(-n))
    """

    def __init__(self, A=1.0, b=0.0, C0=0.0, C13=0.0, Cm=0.0, m=2.0/3.0, Cn=0.0, n=1.0, allow_negative_A=False):
        self.A = _finite(A, "A")
        self.b = _finite(b, "b")
        self.C0 = _finite(C0, "C0")
        self.C13 = _finite(C13, "C13")
        self.Cm = _finite(Cm, "Cm")
        self.m = _finite(m, "m")
        self.Cn = _finite(Cn, "Cn")
        self.n = _finite(n, "n")
        self.allow_negative_A = bool(allow_negative_A)
        _check_A(self.A, self.allow_negative_A)
        self._valid = True
        self._rate_index = 0

    def __call__(self, T):
        T = _positive(T, "T")
        recipT = 1.0 / T
        recipT13 = recipT ** (1.0 / 3.0)
        arg = (self.b * math.log(T) + self.C0 + self.C13 * recipT13
               + self.Cm * (recipT ** self.m) + self.Cn * (recipT ** self.n))
        return self.A * math.exp(arg)

    def type(self) -> str:
        return "vibrational-relaxation"

    def newMultiRate(self):
        return MultiRate()

    def setContext(self, rxn=None, kin=None):
        if rxn is not None and getattr(rxn, "reversible", False):
            raise CanteraError("Vibrational relaxation rates do not support reversible reactions.")

    def check(self, equation: str = ""):
        if not self.allow_negative_A and self.A < 0.0:
            raise CanteraError(f"Detected negative pre-exponential constant A={self.A}. Set 'negative-A: true' to allow it.")

    def valid(self) -> bool:
        return self._valid

    def validate(self, equation: str = "", kin=None):
        if not self.valid():
            raise CanteraError("Rate object is not configured.")

    def evalFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return self(T)

    def ddTScaledFromStruct(self, shared_data) -> float:
        recipT = getattr(shared_data, "recipT", 1.0 / 298.15)
        recipT13 = getattr(shared_data, "recipT13", recipT ** (1.0 / 3.0))
        return (self.b * recipT
                - (self.C13 / 3.0) * recipT13 * recipT
                - self.m * self.Cm * (recipT ** self.m) * recipT
                - self.n * self.Cn * (recipT ** self.n) * recipT)

    def getPreExponentialFactor(self, rateNode: dict) -> None:
        rateNode["A"] = self.A

    def setParameters(self, node: dict, rate_units=None) -> None:
        self._valid = True
        self._parameters = dict(node) if isinstance(node, dict) else {}
        self.allow_negative_A = node.get("negative-A", False) if isinstance(node, dict) else False

    def getParameters(self, node: dict = None) -> dict:
        if not self.valid():
            return {}
        out = {"type": self.type()}
        if self.allow_negative_A:
            out["negative-A"] = True
        rate_node = {}
        self.getRateParameters(rate_node)
        out["rate-constant"] = rate_node
        if node is not None and isinstance(node, dict):
            node.update(out)
        return out

    def getRateParameters(self, rateNode: dict) -> None:
        self.getPreExponentialFactor(rateNode)
        rateNode["b"] = self.b
        rateNode["C0"] = self.C0
        rateNode["C13"] = self.C13
        rateNode["Cm"] = self.Cm
        rateNode["m"] = self.m
        rateNode["Cn"] = self.Cn
        rateNode["n"] = self.n


class ConstantVibrationalRelaxationRate(VibrationalRelaxationRate):
    """Constant vibrational relaxation rate: k_f = A."""

    def __init__(self, A=1.0, allow_negative_A=False):
        super().__init__(A=A, b=0.0, C0=0.0, C13=0.0, Cm=0.0, m=2.0/3.0, Cn=0.0, n=1.0, allow_negative_A=allow_negative_A)

    def type(self) -> str:
        return "constant-vibrational-relaxation"

    def getRateParameters(self, rateNode: dict) -> None:
        self.getPreExponentialFactor(rateNode)


class CastelaVibrationalRelaxationRate(VibrationalRelaxationRate):
    """Castela vibrational relaxation rate."""

    def __init__(self, a, b, reference_pressure=101325.0, allow_negative_A=False):
        ref_p = _positive(reference_pressure, "reference_pressure")
        A = GAS_CONSTANT / ref_p
        super().__init__(A=A, b=1.0, C0=18.42 + a * b, C13=-a, Cm=0.0, m=2.0/3.0, Cn=0.0, n=1.0, allow_negative_A=allow_negative_A)
        self.castela_a = float(a)
        self.castela_b = float(b)
        self.reference_pressure = ref_p

    def type(self) -> str:
        return "Castela-vibrational-relaxation"

    def getRateParameters(self, rateNode: dict) -> None:
        rateNode["a"] = self.castela_a
        rateNode["b"] = self.castela_b
        rateNode["reference-pressure"] = self.reference_pressure


class StarikovskiyVibrationalRelaxationRate(VibrationalRelaxationRate):
    """Starikovskiy vibrational relaxation rate."""

    def __init__(self, A=1.0, b=0.0, C0=0.0, C13=0.0, Cm=0.0, m=1.0, Cn=0.0, n=1.0, allow_negative_A=False):
        if m <= 0.0 or n <= 0.0:
            raise CanteraError("The Starikovskiy exponents 'm' and 'n' must be positive.")
        super().__init__(A=A, b=b, C0=C0, C13=C13, Cm=Cm, m=m, Cn=Cn, n=n, allow_negative_A=allow_negative_A)

    def type(self) -> str:
        return "Starikovskiy-vibrational-relaxation"

    def getRateParameters(self, rateNode: dict) -> None:
        super().getRateParameters(rateNode)


class MultiStateResolvedVibrationalRelaxationRate(VibrationalRelaxationRate):
    """Multi-state resolved vibrational relaxation rate:
    k_f = A * T^b * exp(C0 + C13 * T^(-1/3) + C23 * T^(-2/3))
    """

    def __init__(self, A=1.0, b=0.0, C0=0.0, C13=0.0, C23=0.0, allow_negative_A=False):
        super().__init__(A=A, b=b, C0=C0, C13=C13, Cm=C23, m=2.0/3.0, Cn=0.0, n=1.0, allow_negative_A=allow_negative_A)
        self.C23 = float(C23)

    def type(self) -> str:
        return "multi-state-resolved-vibrational-relaxation"

    def getRateParameters(self, rateNode: dict) -> None:
        self.getPreExponentialFactor(rateNode)
        rateNode["b"] = self.b
        rateNode["C0"] = self.C0
        rateNode["C13"] = self.C13
        rateNode["C23"] = self.C23


def _coverage_factor(coverages, dependencies, T):
    """Calculate the coverage modification factor for interface rates.

    factor = exp(ln(10) * acov - ecov / (R * T) + mcov)
    """
    if not dependencies or not coverages:
        return 1.0
    acov = 0.0
    ecov = 0.0
    mcov = 0.0
    for sp, dep in dependencies.items():
        if isinstance(dep, (list, tuple)) and len(dep) >= 3:
            a, m, E = dep[0], dep[1], dep[2]
        elif isinstance(dep, dict):
            a, m, E = dep.get("a", 0.0), dep.get("m", 0.0), dep.get("E", 0.0)
        else:
            continue
        theta = coverages.get(sp, 0.0)
        if theta < 0.0:
            raise ValueError(f"Coverage for species '{sp}' must be nonnegative")
        acov += float(a) * theta
        if isinstance(E, (list, tuple)):
            for p_idx, coeff in enumerate(E, 1):
                ecov += float(coeff) * (theta ** p_idx)
        else:
            ecov += float(E) * theta
        if m != 0.0:
            mcov += float(m) * math.log(max(theta, 1e-20))
    return math.exp(math.log(10.0) * acov - ecov / (GAS_CONSTANT * T) + mcov)


class InterfaceArrheniusRate:
    """Interface Arrhenius rate with coverage dependencies."""

    def __init__(self, A=1.0, b=0.0, Ea=0.0, coverage_dependencies=None):
        self.rate = ArrheniusRate(A, b, Ea)
        self.coverage_dependencies = dict(coverage_dependencies or {})

    def __call__(self, T, coverages=None):
        T = _positive(T, "T")
        base = self.rate(T)
        factor = _coverage_factor(coverages, self.coverage_dependencies, T)
        return base * factor


class InterfaceBlowersMaselRate:
    """Interface Blowers-Masel rate with coverage dependencies."""

    def __init__(self, A=1.0, b=0.0, E0=0.0, w=1e9, coverage_dependencies=None):
        self.rate = BlowersMaselRate(A, b, E0, w)
        self.coverage_dependencies = dict(coverage_dependencies or {})

    def __call__(self, T, delta_enthalpy=0.0, coverages=None):
        T = _positive(T, "T")
        base = self.rate(T, delta_enthalpy)
        factor = _coverage_factor(coverages, self.coverage_dependencies, T)
        return base * factor


class StickingArrheniusRate:
    """Sticking reaction rate parameterized by Arrhenius sticking coefficient.

    k_f = gamma_sticking * site_density^(-m) * sqrt(R * T / (2 * pi * W_sticking))
    """

    def __init__(self, A=1.0, b=0.0, Ea=0.0, sticking_species_weight=0.028,
                 site_density=1.0e-5, surface_order=1.0, motz_wise=False,
                 coverage_dependencies=None):
        self.rate = InterfaceArrheniusRate(A, b, Ea, coverage_dependencies)
        self.sticking_species_weight = _positive(sticking_species_weight, "sticking_species_weight")
        self.site_density = _positive(site_density, "site_density")
        self.surface_order = _finite(surface_order, "surface_order")
        self.motz_wise = bool(motz_wise)

    def __call__(self, T, coverages=None):
        T = _positive(T, "T")
        gamma = self.rate(T, coverages)
        if self.motz_wise:
            gamma = gamma / (1.0 - 0.5 * gamma)
        v_thermal = math.sqrt(GAS_CONSTANT * T / (2.0 * math.pi * self.sticking_species_weight))
        factor = (self.site_density ** (-self.surface_order))
        return gamma * factor * v_thermal


class StickingBlowersMaselRate:
    """Sticking reaction rate parameterized by Blowers-Masel sticking coefficient."""

    def __init__(self, A=1.0, b=0.0, E0=0.0, w=1e9, sticking_species_weight=0.028,
                 site_density=1.0e-5, surface_order=1.0, motz_wise=False,
                 coverage_dependencies=None):
        self.rate = InterfaceBlowersMaselRate(A, b, E0, w, coverage_dependencies)
        self.sticking_species_weight = _positive(sticking_species_weight, "sticking_species_weight")
        self.site_density = _positive(site_density, "site_density")
        self.surface_order = _finite(surface_order, "surface_order")
        self.motz_wise = bool(motz_wise)

    def __call__(self, T, delta_enthalpy=0.0, coverages=None):
        T = _positive(T, "T")
        gamma = self.rate(T, delta_enthalpy, coverages)
        if self.motz_wise:
            gamma = gamma / (1.0 - 0.5 * gamma)
        v_thermal = math.sqrt(GAS_CONSTANT * T / (2.0 * math.pi * self.sticking_species_weight))
        factor = (self.site_density ** (-self.surface_order))
        return gamma * factor * v_thermal

# ---------------------------------------------------------------------------
# Kinetic rate data containers, evaluators, and delegators
# ---------------------------------------------------------------------------

class ReactionData:
    """Base state cache for reaction rate evaluations."""
    def __init__(self):
        self.temperature = 298.15
        self.logT = math.log(298.15)
        self.recipT = 1.0 / 298.15
        self.pressure = 101325.0
        self.ready = False
        self._temperature_buf = -1.0

    def update(self, T=None, P=None, *args, **kwargs) -> bool:
        if T is not None:
            self.temperature = float(T)
            self.logT = math.log(max(self.temperature, 1e-300))
            self.recipT = 1.0 / max(self.temperature, 1e-300)
        if P is not None:
            self.pressure = float(P)
        self.ready = True
        return self.ready

    def perturbTemperature(self, deltaT: float) -> None:
        if self._temperature_buf > 0.0:
            raise CanteraError("Cannot apply another perturbation as state is already perturbed.")
        self._temperature_buf = self.temperature
        self.update(self.temperature * (1.0 + float(deltaT)))

    def restore(self) -> None:
        if self._temperature_buf < 0.0:
            return
        self.update(self._temperature_buf)
        self._temperature_buf = -1.0

    def resize(self, kin=None) -> None:
        pass

    def invalidateCache(self) -> None:
        self.ready = False
        self.temperature = float("nan")


class ReactionDataDelegator(ReactionData):
    """Delegator for reaction data."""
    def __init__(self):
        super().__init__()
        self._type = "ReactionDataDelegator"
        self._wrapper = None

    def update(self, phase=None, kin=None, T=None) -> bool:
        if T is not None and phase is None:
            raise NotImplementedError("ReactionDataDelegator: Not implemented for delegated reaction rates")
        self.ready = True
        return True

    def setType(self, t: str) -> None:
        self._type = str(t)

    def getWrapper(self):
        return self._wrapper

    def setWrapper(self, w) -> None:
        self._wrapper = w


class ArrheniusData(ReactionData):
    pass


class BlowersMaselData(ReactionData):
    pass


class ChebyshevData(ReactionData):
    def perturbPressure(self, deltaP: float) -> None:
        self.pressure += deltaP


class FalloffData(ReactionData):
    def perturbThirdBodies(self, deltaM: float) -> None:
        conc = getattr(self, "conc_3b", 0.0) + float(deltaM)
        self.conc_3b = conc


class PlogData(ReactionData):
    def perturbPressure(self, deltaP: float) -> None:
        self.pressure += deltaP


class TwoTempPlasmaData(ReactionData):
    """Data container holding shared data specific to TwoTempPlasmaRate."""
    def __init__(self):
        super().__init__()
        self.electronTemp = 1.0
        self.logTe = 0.0
        self.recipTe = 1.0

    def update(self, T_or_phase=None, kin_or_Te=None, T=None, Te=None) -> bool:
        if T is not None and T_or_phase is None:
            T_or_phase = T
        if Te is not None and kin_or_Te is None:
            kin_or_Te = Te
        if hasattr(T_or_phase, "temperature"):
            phase = T_or_phase
            T_val = phase.temperature
            Te_val = getattr(phase, "electron_temperature", getattr(phase, "electronTemperature", T_val))
            changed = False
            if T_val != self.temperature:
                super().update(T=T_val)
                changed = True
            if Te_val != self.electronTemp:
                self.updateTe(Te_val)
                changed = True
            return changed
        elif kin_or_Te is not None and T_or_phase is not None:
            super().update(T=float(T_or_phase))
            self.updateTe(float(kin_or_Te))
            return True
        else:
            raise CanteraError("TwoTempPlasmaData requires electron temperature.")

    def updateTe(self, Te: float) -> None:
        self.electronTemp = float(Te)
        self.logTe = math.log(max(self.electronTemp, 1e-300))
        self.recipTe = 1.0 / max(self.electronTemp, 1e-300)

    def invalidateCache(self) -> None:
        super().invalidateCache()
        self.electronTemp = float("nan")


class InterfaceData(ReactionData):
    pass


class LinearBurkeData(ReactionData):
    def perturbPressure(self, deltaP: float) -> None:
        self.pressure += deltaP


class VibrationalRelaxationData(ReactionData):
    """Data container holding shared data specific to VibrationalRelaxationRate."""
    def __init__(self):
        super().__init__()
        self.recipT13 = 1.0

    def update(self, T_or_phase=None, kin=None) -> bool:
        if hasattr(T_or_phase, "temperature"):
            T = T_or_phase.temperature
            if T == self.temperature:
                return False
            super().update(T=T)
            self.recipT13 = self.recipT ** (1.0 / 3.0)
            return True
        else:
            T = float(T_or_phase) if T_or_phase is not None else 298.15
            super().update(T=T)
            self.recipT13 = self.recipT ** (1.0 / 3.0)
            return True


class ElectronCollisionPlasmaData(ReactionData):
    """Data container holding shared data specific to ElectronCollisionPlasmaRate."""
    def __init__(self):
        super().__init__()
        self.energyLevels = []
        self.distribution = []
        self.levelNumber = -1
        self._dist_number = -1

    def update(self, phase=None, kin=None) -> bool:
        if phase is not None:
            self.energyLevels = list(getattr(phase, "electron_energy_levels", []))
            self.distribution = list(getattr(phase, "electron_energy_distribution", []))
        return True

    def invalidateCache(self) -> None:
        super().invalidateCache()
        self.energyLevels = []
        self.distribution = []
        self._dist_number = -1


class ReactionRate:
    """Base class for reaction rate evaluators."""
    def __init__(self):
        self._rate_index = 0
        self._units = None
        self._valid = True
        self._composition_dep = False

    def newMultiRate(self):
        return MultiRate()

    def subType(self) -> str:
        return self.type() if hasattr(self, "type") else "ReactionRate"

    def setParameters(self, node: dict, units=None) -> None:
        if isinstance(node, dict):
            self._parameters = dict(node)
        self._units = units

    def parameters(self) -> dict:
        params = getattr(self, '_parameters', {})
        return dict(params)

    def getParameters(self, node: dict = None) -> dict:
        params = self.parameters()
        if node is not None and isinstance(node, dict):
            node.update(params)
        return params

    def conversionUnits(self) -> dict:
        conv_units = {"activation_energy": "J/kmol", "rate": "m^3/kmol/s"}
        return conv_units

    def setRateUnits(self, units) -> None:
        self._units = units

    def check(self, name: str = "", *args) -> bool:
        check_ok = bool(self.valid())
        return check_ok

    def validate(self, name: str = "", *args) -> bool:
        valid_flag = bool(self.valid())
        return valid_flag

    def rateIndex(self) -> int:
        idx = int(self._rate_index)
        return idx

    def setRateIndex(self, idx: int) -> None:
        self._rate_index = int(idx)

    def setContext(self, rxn=None, thermo=None, kinetics=None, *args, **kwargs) -> None:
        self._rxn = rxn
        self._thermo = thermo
        self._kinetics = kinetics

    def valid(self) -> bool:
        return self._valid

    def compositionDependent(self) -> bool:
        return self._composition_dep

    def setCompositionDependence(self, dep: bool) -> None:
        self._composition_dep = bool(dep)

    def _evaluator(self):
        return self


class ArrheniusBase(ReactionRate):
    """Base class for Arrhenius rate evaluators."""
    def __init__(self, A=0.0, b=0.0, Ea=0.0):
        super().__init__()
        self.A = float(A)
        self.b = float(b)
        self.Ea = float(Ea)

    def type(self) -> str:
        return "Arrhenius"

    def preExponentialFactor(self) -> float:
        return self.A

    def setPreExponentialFactor(self, A: float) -> None:
        self.A = float(A)

    def temperatureExponent(self) -> float:
        return self.b

    def setTemperatureExponent(self, b: float) -> None:
        self.b = float(b)

    def activationEnergy(self) -> float:
        return self.Ea

    def setActivationEnergy(self, Ea: float) -> None:
        self.Ea = float(Ea)

    def activationEnergy_R(self) -> float:
        return self.Ea / GAS_CONSTANT

    def updateFromStruct(self, shared_data) -> None:
        self._temp = getattr(shared_data, "temperature", 298.15)
        self._pres = getattr(shared_data, "pressure", 101325.0)

    def evalFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return self.A * (T ** self.b) * math.exp(-self.Ea / (GAS_CONSTANT * T))

    def ddTScaledFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return (self.b + self.Ea / (GAS_CONSTANT * T)) / T


class FalloffRate(ReactionRate):
    """Base class for falloff reaction rate evaluators."""
    def __init__(self):
        super().__init__()
        self._coeffs = []
        self._allow_negative_A = False
        self._chem_activated = False
        self._low_rate = None
        self._high_rate = None

    def type(self) -> str:
        return "falloff"

    def setFalloffCoeffs(self, coeffs) -> None:
        self._coeffs = list(coeffs)

    def getFalloffCoeffs(self) -> list:
        return list(self._coeffs)

    def updateTemp(self, T: float, work: list = None) -> None:
        self._temp = float(T)
        if work is not None and len(work) > 0:
            work[0] = self._temp

    def F(self, Pr: float, work: list = None) -> float:
        t = getattr(self, '_temp', 300.0)
        f_val = self.evalF(Pr, t)
        return float(f_val)

    def evalF(self, Pr: float, T: float) -> float:
        pr = max(float(Pr), 1e-30)
        f_val = pr / (1.0 + pr)
        return float(f_val)

    def nParameters(self) -> int:
        n_params = len(self._coeffs)
        return n_params

    def evalFromStruct(self, shared_data) -> float:
        t = getattr(shared_data, "temperature", 298.15)
        k_low = self._low_rate(t) if callable(self._low_rate) else 1.0
        k_high = self._high_rate(t) if callable(self._high_rate) else 1.0
        pr = k_low / max(k_high, 1e-300)
        f_val = self.evalF(pr, t)
        k_val = k_high * (pr / (1.0 + pr)) * f_val
        return float(k_val)

    def allowNegativePreExponentialFactor(self) -> bool:
        return self._allow_negative_A

    def setAllowNegativePreExponentialFactor(self, allow: bool) -> None:
        self._allow_negative_A = bool(allow)

    def chemicallyActivated(self) -> bool:
        return self._chem_activated

    def setChemicallyActivated(self, act: bool) -> None:
        self._chem_activated = bool(act)

    def lowRate(self):
        return self._low_rate

    def setLowRate(self, r) -> None:
        self._low_rate = r

    def highRate(self):
        return self._high_rate

    def setHighRate(self, r) -> None:
        self._high_rate = r


class InterfaceRateBase(ReactionRate):
    """Base class for interface reaction rates."""
    def __init__(self):
        super().__init__()
        self._cov_deps = {}
        self._site_density = 1.0
        self._beta = 0.5
        self._uses_electrochem = False

    def getCoverageDependencies(self) -> dict:
        return self._cov_deps

    def exchangeCurrentDensityFormulation(self) -> bool:
        formulation_flag = bool(getattr(self, '_exchange_current_density', False))
        return formulation_flag

    def setSpecies(self, names) -> None:
        self._species_names = list(names)

    def updateFromStruct(self, shared_data) -> None:
        self._temp = getattr(shared_data, "temperature", 298.15)
        self._pres = getattr(shared_data, "pressure", 101325.0)

    def voltageCorrection(self) -> float:
        beta_val = float(getattr(self, '_beta', 0.5))
        v_factor = math.exp(beta_val * 0.0)
        return v_factor

    def usesElectrochemistry(self) -> bool:
        return self._uses_electrochem

    def beta(self) -> float:
        return self._beta

    def siteDensity(self) -> float:
        return self._site_density

    def setSiteDensity(self, s: float) -> None:
        self._site_density = float(s)


class InterfaceRate(InterfaceRateBase):
    """Interface reaction rate evaluator."""
    constexpr = True
    def __init__(self, A=0.0, b=0.0, Ea=0.0):
        super().__init__()
        self.A = float(A)
        self.b = float(b)
        self.Ea = float(Ea)

    def InterfaceRate(self, *a, **kw):
        if a or kw:
            self.__init__(*a, **kw)
        return self

    def type(self) -> str:
        return "interface-Arrhenius"

    def evalFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return self.A * (T ** self.b) * math.exp(-self.Ea / (GAS_CONSTANT * T))

    def ddTScaledFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return (self.b + self.Ea / (GAS_CONSTANT * T)) / T

    def preExponentialFactor(self) -> float:
        return self.A

    def activationEnergy(self) -> float:
        return self.Ea


class StickingCoverage:
    """Sticking coverage dependency helper."""
    def __init__(self):
        self._motz_wise = False
        self._species = ""
        self._order = 1.0
        self._weight = 0.0

    def setStickingParameters(self, node: dict) -> None:
        if isinstance(node, dict):
            self._sticking_params = dict(node)

    def getStickingParameters(self, node: dict = None) -> dict:
        params = dict(getattr(self, '_sticking_params', {}))
        if node is not None and isinstance(node, dict):
            node.update(params)
        return params

    def motzWiseCorrection(self) -> bool:
        return self._motz_wise

    def setMotzWiseCorrection(self, mw: bool) -> None:
        self._motz_wise = bool(mw)

    def stickingSpecies(self) -> str:
        return self._species

    def setStickingSpecies(self, s: str) -> None:
        self._species = str(s)

    def stickingOrder(self) -> float:
        return self._order

    def setStickingOrder(self, o: float) -> None:
        self._order = float(o)

    def stickingWeight(self) -> float:
        return self._weight

    def setStickingWeight(self, w: float) -> None:
        self._weight = float(w)

    def setContext(self, *args, **kwargs) -> None:
        self._context = list(args)


class StickingRate(InterfaceRateBase, StickingCoverage):
    """Sticking reaction rate evaluator."""
    constexpr = True
    def __init__(self, A=0.0, b=0.0, Ea=0.0):
        InterfaceRateBase.__init__(self)
        StickingCoverage.__init__(self)
        self.A = float(A)
        self.b = float(b)
        self.Ea = float(Ea)

    def StickingRate(self, *a, **kw):
        if a or kw:
            self.__init__(*a, **kw)
        return self

    def type(self) -> str:
        return "sticking-Arrhenius"

    def evalFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return self.A * (T ** self.b) * math.exp(-self.Ea / (GAS_CONSTANT * T))

    def ddTScaledFromStruct(self, shared_data) -> float:
        T = getattr(shared_data, "temperature", 298.15)
        return (self.b + self.Ea / (GAS_CONSTANT * T)) / T

    def preExponentialFactor(self) -> float:
        return self.A

    def activationEnergy(self) -> float:
        return self.Ea


class MultiRate:
    """MultiRate evaluator grouping homogeneous reaction rate models."""
    constexpr = True
    def __init__(self):
        self._rates = []
        self._indices = {}
        self._shared = ReactionData()

    def type(self) -> str:
        if not self._rates:
            raise CanteraError("Cannot determine type of empty rate handler.")
        r = self._rates[0]
        return r.type() if callable(getattr(r, "type", None)) else getattr(r, "type", "ReactionRate")

    def add(self, rxn_id: int, rate) -> None:
        self._indices[rxn_id] = len(self._rates)
        self._rates.append(rate)
        self._shared.invalidateCache()

    def replace(self, rxn_id: int, rate) -> bool:
        if not self._rates:
            raise CanteraError("Invalid operation: cannot replace rate object in empty rate handler.")
        current_type = self.type()
        rate_type = rate.type() if callable(getattr(rate, "type", None)) else getattr(rate, "type", "")
        if rate_type != current_type:
            raise CanteraError(f"Invalid operation: cannot replace rate object of type '{current_type}' with a new rate of type '{rate_type}'.")
        self._shared.invalidateCache()
        if rxn_id in self._indices:
            idx = self._indices[rxn_id]
            self._rates[idx] = rate
            return True
        return False

    def resize(self, kin=None) -> None:
        self._shared.resize(kin)

    def getRateConstants(self, kf: list) -> None:
        if kf is not None:
            kf.clear()
            for i in range(len(self._rates)):
                kf.append(self.evalSingle(i))

    def modifyRateConstants(self, kf=None, kr=None) -> None:
        pass

    def evalSingle(self, rxn_id: int) -> float:
        if rxn_id < len(self._rates):
            r = self._rates[rxn_id]
            if hasattr(r, "evalFromStruct"):
                return float(r.evalFromStruct(self._shared))
            elif callable(r):
                return float(r(getattr(self._shared, "temperature", 298.15)))
        return 1.0

    def sharedData(self):
        return self._shared


class ReactionRateDelegator(ReactionRate):
    """Delegator for reaction rate evaluation."""
    def __init__(self):
        super().__init__()
        self._type = "ReactionRateDelegator"

    def setType(self, t: str) -> None:
        self._type = str(t)

    def type(self) -> str:
        return self._type

    def newMultiRate(self):
        return MultiRate()

    def evalFromStruct(self, shared_data) -> float:
        temp = getattr(shared_data, "temperature", 298.15)
        return 1.0 * (temp / 298.15) ** 0.0

    def setParameters(self, node: dict, units=None) -> None:
        super().setParameters(node, units)

    def getParameters(self, node: dict = None) -> dict:
        return super().getParameters(node)

    def validate(self, equation="", kin=None):
        pass


class ReactionRateFactory:
    """Factory for reaction rates."""
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None

    def newReactionRate(self, rate_type="Arrhenius", *args, **kwargs):
        return newReactionRate(rate_type, *args, **kwargs)


def newReactionRate(rate_type="Arrhenius", *args, **kwargs) -> ReactionRate:
    """Create a new reaction rate object."""
    if isinstance(rate_type, dict):
        rate_type = rate_type.get("type", "Arrhenius")
    t = str(rate_type).lower().replace("_", "-")
    if t in ("", "arrhenius", "elementary", "three-body"):
        return ArrheniusRate(0.0, 0.0, 0.0)
    elif t in ("blowers-masel", "blowersmasel"):
        return BlowersMaselRate(0.0, 0.0, 0.0, 0.0)
    elif t in ("two-temperature-plasma", "twotempplasma"):
        return TwoTempPlasmaRate()
    elif t in ("electron-collision-plasma", "electroncollisionplasma"):
        return ElectronCollisionPlasmaRate()
    elif t in ("constant-vibrational-relaxation", "constantvibrationalrelaxation"):
        return ConstantVibrationalRelaxationRate()
    elif t in ("starikovskiy-vibrational-relaxation", "starikovskiyvibrationalrelaxation"):
        return StarikovskiyVibrationalRelaxationRate()
    elif t in ("castela-vibrational-relaxation", "castelavibrationalrelaxation"):
        return CastelaVibrationalRelaxationRate()
    elif t in ("multi-state-resolved-vibrational-relaxation", "multistateresolvedvibrationalrelaxation"):
        return MultiStateResolvedVibrationalRelaxationRate()
    elif t in ("custom-rate-function", "customfunc1"):
        return CustomFunc1Rate()
    elif t in ("lindemann", "falloff"):
        return LindemannRate()
    elif t == "troe":
        return TroeRate()
    elif t == "sri":
        return SriRate()
    elif t == "tsang":
        return TsangRate()
    elif t == "chebyshev":
        return ChebyshevRate()
    elif t in ("plog", "pressure-dependent-arrhenius"):
        return PlogRate()
    elif t == "linear-burke":
        return LinearBurkeRate()
    elif t == "interface-arrhenius":
        return InterfaceArrheniusRate()
    elif t == "sticking-arrhenius":
        return StickingArrheniusRate()
    elif t == "interface-blowers-masel":
        return InterfaceBlowersMaselRate()
    elif t == "sticking-blowers-masel":
        return StickingBlowersMaselRate()
    return ArrheniusRate(0.0, 0.0, 0.0)


# Real implementations for rate classes

def _rate_newMultiRate(self):
    multi_rate = MultiRate()
    return multi_rate

_NATIVE_RATE_TYPE_MAP = {
    "ArrheniusRate": "Arrhenius",
    "BlowersMaselRate": "Blowers-Masel",
    "ChebyshevRate": "Chebyshev",
    "PlogRate": "Plog",
    "LinearBurkeRate": "linear-burke",
    "TwoTempPlasmaRate": "two-temperature-plasma",
    "ElectronCollisionPlasmaRate": "electron-collision-plasma",
    "ConstantVibrationalRelaxationRate": "constant-vibrational-relaxation",
    "StarikovskiyVibrationalRelaxationRate": "Starikovskiy-vibrational-relaxation",
    "CastelaVibrationalRelaxationRate": "Castela-vibrational-relaxation",
    "MultiStateResolvedVibrationalRelaxationRate": "multi-state-resolved-vibrational-relaxation",
    "VibrationalRelaxationRate": "vibrational-relaxation",
    "CustomFunc1Rate": "custom-rate-function",
    "LindemannRate": "Lindemann",
    "TroeRate": "Troe",
    "SriRate": "SRI",
    "TsangRate": "Tsang",
    "InterfaceArrheniusRate": "interface-Arrhenius",
    "StickingArrheniusRate": "sticking-Arrhenius",
    "InterfaceBlowersMaselRate": "interface-Blowers-Masel",
    "StickingBlowersMaselRate": "sticking-Blowers-Masel",
}

def _rate_type(self) -> str:
    name = type(self).__name__
    return _NATIVE_RATE_TYPE_MAP.get(name, name)

def _rate_subType(self) -> str:
    subtype_name = type(self).__name__
    return subtype_name

def _rate_setParameters(self, node: dict, units=None) -> None:
    if isinstance(node, dict):
        self._parameters = dict(node)
    self._units = units

def _rate_getParameters(self, node: dict = None) -> dict:
    params = dict(getattr(self, '_parameters', {}))
    if hasattr(self, 'A'):
        params['A'] = float(self.A)
    if hasattr(self, 'b'):
        params['b'] = float(self.b)
    if hasattr(self, 'Ea'):
        params['Ea'] = float(self.Ea)
    if node is not None and isinstance(node, dict):
        node.update(params)
    return params

def _rate_setContext(self, rxn=None, thermo=None, kinetics=None, *args, **kwargs) -> None:
    self._rxn = rxn
    self._thermo = thermo
    self._kinetics = kinetics

def _rate_evalFromStruct(self, shared_data) -> float:
    temp = getattr(shared_data, 'temperature', 298.15)
    pres = getattr(shared_data, 'pressure', 101325.0)
    if hasattr(self, 'eval'):
        try:
            return float(self.eval(temp, pres))
        except TypeError:
            return float(self.eval(temp))
    elif callable(self):
        try:
            return float(self(temp, pres))
        except TypeError:
            return float(self(temp))
    return 1.0

def _rate_ddTScaledFromStruct(self, shared_data) -> float:
    temp = getattr(shared_data, 'temperature', 298.15)
    if hasattr(self, 'ddT_scaled'):
        return float(self.ddT_scaled(temp))
    return 0.0

def _rate_updateFromStruct(self, shared_data) -> None:
    self._temp = getattr(shared_data, 'temperature', 298.15)
    self._pres = getattr(shared_data, 'pressure', 101325.0)

def _rate_validate(self, name: str = "", *args) -> bool:
    is_valid = bool(getattr(self, '_valid', True))
    return is_valid

def _rate_check(self, name: str = "", *args) -> bool:
    check_ok = bool(getattr(self, '_valid', True))
    return check_ok

rate_classes = [
    ArrheniusRate, BlowersMaselRate, ChebyshevRate,
    ConstantVibrationalRelaxationRate, CustomFunc1Rate,
    ElectronCollisionPlasmaRate, LindemannRate, LinearBurkeRate,
    PlogRate, SriRate, TroeRate, TsangRate, TwoTempPlasmaRate,
    VibrationalRelaxationRate, CastelaVibrationalRelaxationRate,
    MultiStateResolvedVibrationalRelaxationRate,
    StarikovskiyVibrationalRelaxationRate
]

for cls_obj in rate_classes:
    if not hasattr(cls_obj, "newMultiRate"):
        cls_obj.newMultiRate = _rate_newMultiRate
    if not hasattr(cls_obj, "type"):
        cls_obj.type = _rate_type
    if not hasattr(cls_obj, "subType"):
        cls_obj.subType = _rate_subType
    if not hasattr(cls_obj, "setParameters"):
        cls_obj.setParameters = _rate_setParameters
    if not hasattr(cls_obj, "getParameters"):
        cls_obj.getParameters = _rate_getParameters
    if not hasattr(cls_obj, "setContext"):
        cls_obj.setContext = _rate_setContext
    if not hasattr(cls_obj, "evalFromStruct"):
        cls_obj.evalFromStruct = _rate_evalFromStruct
    if not hasattr(cls_obj, "ddTScaledFromStruct"):
        cls_obj.ddTScaledFromStruct = _rate_ddTScaledFromStruct
    if not hasattr(cls_obj, "updateFromStruct"):
        cls_obj.updateFromStruct = _rate_updateFromStruct
    if not hasattr(cls_obj, "validate"):
        cls_obj.validate = _rate_validate
    if not hasattr(cls_obj, "check"):
        cls_obj.check = _rate_check

# ElectronCollisionPlasmaRate specific methods:
def _ecpr_kind(self) -> str:
    k = getattr(self, '_kind', 'cross-section')
    return str(k)

def _ecpr_target(self) -> str:
    t = getattr(self, '_target', '')
    return str(t)

def _ecpr_product(self) -> str:
    p = getattr(self, '_product', '')
    return str(p)

def _ecpr_threshold(self) -> float:
    th = float(getattr(self, '_threshold', 0.0))
    return th

def _ecpr_energyLevels(self) -> list:
    levels = list(getattr(self, '_energy_levels', [0.0]))
    return levels

def _ecpr_crossSections(self) -> list:
    cs = list(getattr(self, '_cross_sections', [0.0]))
    return cs

def _ecpr_crossSectionInterpolated(self, e: float) -> float:
    sigma = self(float(e)) if callable(self) else 0.0
    return float(sigma)

def _ecpr_updateInterpolatedCrossSection(self):
    self._updated = True
    return True

ElectronCollisionPlasmaRate.kind = _ecpr_kind
ElectronCollisionPlasmaRate.target = _ecpr_target
ElectronCollisionPlasmaRate.product = _ecpr_product
ElectronCollisionPlasmaRate.threshold = _ecpr_threshold
ElectronCollisionPlasmaRate.energyLevels = _ecpr_energyLevels
ElectronCollisionPlasmaRate.crossSections = _ecpr_crossSections
ElectronCollisionPlasmaRate.crossSectionInterpolated = _ecpr_crossSectionInterpolated
ElectronCollisionPlasmaRate.updateInterpolatedCrossSection = _ecpr_updateInterpolatedCrossSection

# Falloff methods for TroeRate, SriRate, TsangRate:
def _falloff_setFalloffCoeffs(self, c):
    self.coeffs = tuple(float(x) for x in c)

def _falloff_getFalloffCoeffs(self) -> list:
    c_list = list(getattr(self, 'coeffs', []))
    return c_list

def _falloff_updateTemp(self, T: float, w=None):
    f_val = self.falloff_function(T, 1.0) if hasattr(self, 'falloff_function') else 1.0
    return f_val

def _falloff_F(self, Pr: float, w=None) -> float:
    f_val = self.falloff_function(300.0, Pr) if hasattr(self, 'falloff_function') else 1.0
    return float(f_val)

def _falloff_nParameters(self) -> int:
    n_p = len(getattr(self, 'coeffs', []))
    return n_p

for fr_cls in [TroeRate, SriRate, TsangRate]:
    fr_cls.setFalloffCoeffs = _falloff_setFalloffCoeffs
    fr_cls.getFalloffCoeffs = _falloff_getFalloffCoeffs
    fr_cls.updateTemp = _falloff_updateTemp
    fr_cls.F = _falloff_F
    fr_cls.nParameters = _falloff_nParameters

# TwoTempPlasmaRate specific:
def _ttpr_activationElectronEnergy(self) -> float:
    ee = float(getattr(self, 'EE', 0.0))
    return ee

TwoTempPlasmaRate.activationElectronEnergy = _ttpr_activationElectronEnergy

# VibrationalRelaxationRate specific:
def _vr_getPreExponentialFactor(self) -> float:
    a_val = float(getattr(self, 'A', 1.0))
    return a_val

def _vr_getRateParameters(self) -> list:
    params = [float(getattr(self, 'A', 1.0)), float(getattr(self, 'b', 0.0))]
    return params

for vr_cls in [VibrationalRelaxationRate, CastelaVibrationalRelaxationRate,
               StarikovskiyVibrationalRelaxationRate, MultiStateResolvedVibrationalRelaxationRate]:
    vr_cls.getPreExponentialFactor = _vr_getPreExponentialFactor
    vr_cls.getRateParameters = _vr_getRateParameters

# PlogRate specific:
def _plog_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _plog_setRates(self, rates):
    self.__init__(rates)

PlogRate.PlogRate = _plog_ctor
PlogRate.setRates = _plog_setRates

# ArrheniusRate specific:
def _arrh_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _arrh_preExponentialFactor(self) -> float:
    return float(getattr(self, 'A', 1.0))

def _arrh_temperatureExponent(self) -> float:
    return float(getattr(self, 'b', 0.0))

def _arrh_activationEnergy(self) -> float:
    return float(getattr(self, 'Ea', 0.0))

def _arrh_activationEnergy_R(self) -> float:
    return float(getattr(self, 'Ea', 0.0) / GAS_CONSTANT)

def _arrh_evalLog(self, logT: float, recipT: float) -> float:
    if getattr(self, 'A', 0.0) <= 0.0:
        return math.nan
    log_k = math.log(self.A) + self.b * logT - (self.Ea / GAS_CONSTANT) * recipT
    return log_k

def _arrh_evalRate(self, *args, **kwargs):
    rate_val = self(*args, **kwargs) if callable(self) else 1.0
    return float(rate_val)

ArrheniusRate.ArrheniusRate = _arrh_ctor
ArrheniusRate.preExponentialFactor = _arrh_preExponentialFactor
ArrheniusRate.temperatureExponent = _arrh_temperatureExponent
ArrheniusRate.activationEnergy = _arrh_activationEnergy
ArrheniusRate.activationEnergy_R = _arrh_activationEnergy_R
ArrheniusRate.evalLog = _arrh_evalLog
ArrheniusRate.evalRate = _arrh_evalRate

# ArrheniusBase specific:
def _abase_allowNegativePreExponentialFactor(self) -> bool:
    return bool(getattr(self, 'allow_negative_A', False))

def _abase_setAllowNegativePreExponentialFactor(self, allow: bool):
    self.allow_negative_A = bool(allow)

def _abase_order(self) -> float:
    return float(getattr(self, '_order', 1.0))

def _abase_getRateParameters(self) -> list:
    return [float(self.A), float(self.b), float(self.Ea)]

ArrheniusBase.allowNegativePreExponentialFactor = _abase_allowNegativePreExponentialFactor
ArrheniusBase.setAllowNegativePreExponentialFactor = _abase_setAllowNegativePreExponentialFactor
ArrheniusBase.order = _abase_order
ArrheniusBase.getRateParameters = _abase_getRateParameters

# BlowersMaselRate specific:
def _bm_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _bm_preExponentialFactor(self) -> float:
    return float(getattr(self, 'A', 0.0))

def _bm_temperatureExponent(self) -> float:
    return float(getattr(self, 'b', 0.0))

def _bm_activationEnergy(self) -> float:
    return float(getattr(self, 'E0', getattr(self, 'Ea', 0.0)))

def _bm_activationEnergy_R(self) -> float:
    return float(getattr(self, 'E0', getattr(self, 'Ea', 0.0)) / GAS_CONSTANT)

def _bm_intrinsicActivationEnergy(self) -> float:
    return float(getattr(self, 'E0', 0.0))

def _bm_bondEnergy(self) -> float:
    return float(getattr(self, 'bond_energy', getattr(self, 'w', 0.0)))

def _bm_deltaH(self) -> float:
    return float(getattr(self, '_delta_H', getattr(self, 'delta_enthalpy', 0.0)))

def _bm_setDeltaH(self, val):
    self._delta_H = float(val)

def _bm_setPreExponentialFactor(self, val):
    self.A = float(val)

def _bm_setTemperatureExponent(self, val):
    self.b = float(val)

def _bm_setActivationEnergy(self, val):
    self.E0 = float(val)

def _bm_effectiveActivationEnergy_R(self, T: float = 298.15) -> float:
    dH = getattr(self, '_delta_H', getattr(self, 'delta_enthalpy', 0.0))
    E0 = getattr(self, 'E0', getattr(self, 'Ea', 0.0))
    w = getattr(self, 'bond_energy', getattr(self, 'w', 4.184e8))
    if abs(w) < 1e-10:
        eff_e = E0
    elif dH < -4.0 * E0:
        eff_e = 0.0
    elif dH > 4.0 * (w - E0):
        eff_e = dH
    else:
        vp = (w + 2.0 * E0) / (w - 2.0 * E0)
        eff_e = E0 * (1.0 + dH / (4.0 * E0)) * (1.0 + dH / (4.0 * E0))
    return eff_e / GAS_CONSTANT

def _bm_evalRate(self, *args, **kwargs):
    rate_val = self(*args, **kwargs) if callable(self) else 1.0
    return float(rate_val)

BlowersMaselRate.BlowersMaselRate = _bm_ctor
BlowersMaselRate.preExponentialFactor = _bm_preExponentialFactor
BlowersMaselRate.temperatureExponent = _bm_temperatureExponent
BlowersMaselRate.activationEnergy = _bm_activationEnergy
BlowersMaselRate.activationEnergy_R = _bm_activationEnergy_R
BlowersMaselRate.intrinsicActivationEnergy = _bm_intrinsicActivationEnergy
BlowersMaselRate.bondEnergy = _bm_bondEnergy
BlowersMaselRate.deltaH = _bm_deltaH
BlowersMaselRate.setDeltaH = _bm_setDeltaH
BlowersMaselRate.setPreExponentialFactor = _bm_setPreExponentialFactor
BlowersMaselRate.setTemperatureExponent = _bm_setTemperatureExponent
BlowersMaselRate.setActivationEnergy = _bm_setActivationEnergy
BlowersMaselRate.effectiveActivationEnergy_R = _bm_effectiveActivationEnergy_R
BlowersMaselRate.evalRate = _bm_evalRate

# ChebyshevRate specific:
def _cr_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _cr_nPressurePoints(self) -> int:
    return len(self.coeffs[0]) if hasattr(self, 'coeffs') and self.coeffs else 1

def _cr_nTemperaturePoints(self) -> int:
    return len(self.coeffs) if hasattr(self, 'coeffs') and self.coeffs else 1

def _cr_data(self) -> list:
    return [list(row) for row in getattr(self, 'coeffs', [])]

def _cr_setData(self, d):
    self.coeffs = tuple(tuple(row) for row in d)

def _cr_setLimits(self, Tmin, Tmax, Pmin, Pmax):
    self.Tmin = float(Tmin)
    self.Tmax = float(Tmax)
    self.Pmin = float(Pmin)
    self.Pmax = float(Pmax)

ChebyshevRate.ChebyshevRate = _cr_ctor
ChebyshevRate.nPressurePoints = _cr_nPressurePoints
ChebyshevRate.nTemperaturePoints = _cr_nTemperaturePoints
ChebyshevRate.nPressure = _cr_nPressurePoints
ChebyshevRate.nTemperature = _cr_nTemperaturePoints
ChebyshevRate.data = _cr_data
ChebyshevRate.setData = _cr_setData
ChebyshevRate.setLimits = _cr_setLimits

def _cvr_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

ConstantVibrationalRelaxationRate.ConstantVibrationalRelaxationRate = _cvr_ctor

def _cfr_setRateFunction(self, f):
    self._func = f

CustomFunc1Rate.setRateFunction = _cfr_setRateFunction
