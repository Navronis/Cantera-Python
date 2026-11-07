"""Update rates.py and kinetics.py for Family 5 completeness."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RATES_FILE = ROOT / "cantera_python" / "rates.py"
KINETICS_FILE = ROOT / "cantera_python" / "kinetics.py"


def update_rates():
    content = RATES_FILE.read_text(encoding="utf-8")

    # 1. Update TwoTempPlasmaRate
    old_twotemp = re.search(r"@dataclass\s+class TwoTempPlasmaRate:.*?def ddT_scaled\(self, T, Te\):.*?\* recip_T \* recip_T - recip_Tinv\)", content, re.DOTALL)
    assert old_twotemp is not None, "Could not find TwoTempPlasmaRate in rates.py"

    new_twotemp = '''class TwoTempPlasmaRate(ArrheniusBase):
    """Gas/electron-temperature rate, including upstream bg and Tinv terms.

    ``k = A*T**bg*Te**b * exp(-Ea/(R*T) + EE*(Te-T)/(R*Te*T)
    - T/Tinv)``; the last term is omitted when ``Tinv == 0``.
    ``Ea`` and ``EE`` are J/kmol. This expression does not supply a reverse
    rate from equilibrium thermochemistry.
    """

    def __init__(self, A=0.0, b=0.0, Ea=0.0, EE=0.0, bg=0.0, Tinv=0.0, allow_negative_A=False):
        super().__init__(A=A, b=b, Ea=Ea)
        self.EE = float(EE)
        self.bg = float(bg)
        self.Tinv = float(Tinv)
        self.allow_negative_A = bool(allow_negative_A)
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
        return out'''

    content = content[:old_twotemp.start()] + new_twotemp + content[old_twotemp.end():]

    # 2. Update ElectronCollisionPlasmaRate methods
    # Add type, newMultiRate, setContext, evalFromStruct, ddTScaledFromStruct, setParameters, getParameters, etc.
    old_ecp = re.search(r"class ElectronCollisionPlasmaRate:.*?return self\._prefactor \* simpson\(integrand, levels\)", content, re.DOTALL)
    assert old_ecp is not None, "Could not find ElectronCollisionPlasmaRate in rates.py"

    ecp_extra = '''

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
        self.cross_sections = tuple(float(x) for x in sections)'''

    # Insert extra methods right before CustomFunc1Rate
    idx_custom = content.find("class CustomFunc1Rate:")
    assert idx_custom != -1, "Could not find CustomFunc1Rate"
    content = content[:idx_custom] + ecp_extra + "\n\n\n" + content[idx_custom:]

    # 3. Update CustomFunc1Rate methods
    old_custom = re.search(r"class CustomFunc1Rate:.*?def validate\(self, equation=\"\"\):.*?is not configured\.\"\.format\(equation\)\)", content, re.DOTALL)
    assert old_custom is not None, "Could not find CustomFunc1Rate block"

    custom_extra = '''
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
        return out'''

    idx_custom_val_end = old_custom.end()
    content = content[:idx_custom_val_end] + custom_extra + content[idx_custom_val_end:]

    # 4. Update VibrationalRelaxationRate and subclasses
    old_vib = re.search(r"class VibrationalRelaxationRate:.*?class MultiStateResolvedVibrationalRelaxationRate\(VibrationalRelaxationRate\):.*?self\.C23 = float\(C23\)", content, re.DOTALL)
    assert old_vib is not None, "Could not find VibrationalRelaxationRate block"

    new_vib = '''class VibrationalRelaxationRate(ReactionRate):
    """Common implementation for vibrational relaxation reaction rates.

    k_f = A * T^b * exp(C0 + C13 * T^(-1/3) + Cm * T^(-m) + Cn * T^(-n))
    """

    def __init__(self, A=1.0, b=0.0, C0=0.0, C13=0.0, Cm=0.0, m=2.0/3.0, Cn=0.0, n=1.0, allow_negative_A=False):
        super().__init__()
        self.A = _finite(A, "A")
        self.b = _finite(b, "b")
        self.C0 = _finite(C0, "C0")
        self.C13 = _finite(C13, "C13")
        self.Cm = _finite(Cm, "Cm")
        self.m = _finite(m, "m")
        self.Cn = _finite(Cn, "Cn")
        self.n = _finite(n, "n")
        self.allow_negative_A = bool(allow_negative_A)
        self._valid = True

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
        super().setParameters(node, rate_units)
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
        rateNode["C23"] = self.C23'''

    content = content[:old_vib.start()] + new_vib + content[old_vib.end():]

    # 5. Update ReactionData and ReactionDataDelegator and Plasma/Vibrational data classes
    old_rd = re.search(r"class ReactionData:.*?class ElectronCollisionPlasmaData\(ReactionData\):\s+pass", content, re.DOTALL)
    assert old_rd is not None, "Could not find ReactionData block"

    new_rd = '''class ReactionData:
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

    def update(self, T_or_phase=None, kin_or_Te=None) -> bool:
        if hasattr(T_or_phase, "temperature"):
            phase = T_or_phase
            T = phase.temperature
            Te = getattr(phase, "electron_temperature", getattr(phase, "electronTemperature", T))
            changed = False
            if T != self.temperature:
                super().update(T=T)
                changed = True
            if Te != self.electronTemp:
                self.updateTe(Te)
                changed = True
            return changed
        elif kin_or_Te is not None:
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
        self._dist_number = -1'''

    content = content[:old_rd.start()] + new_rd + content[old_rd.end():]

    # 6. Update MultiRate
    old_mr = re.search(r"class MultiRate:.*?def sharedData\(self\):.*?return self\._data", content, re.DOTALL)
    assert old_mr is not None, "Could not find MultiRate block"

    new_mr = '''class MultiRate:
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
        return self._shared'''

    content = content[:old_mr.start()] + new_mr + content[old_mr.end():]

    RATES_FILE.write_text(content, encoding="utf-8")
    print(f"Updated {RATES_FILE}")


def update_kinetics():
    content = KINETICS_FILE.read_text(encoding="utf-8")

    # 1. Update supported_types in kinetics.py
    old_sup = re.search(r"supported_types = \{.*?'linear-Burke',\s*\}", content, re.DOTALL)
    assert old_sup is not None, "Could not find supported_types in kinetics.py"

    new_sup = '''supported_types = {
        'elementary', 'three-body', 'falloff', 'chemically-activated', 'Blowers-Masel',
        'pressure-dependent-Arrhenius', 'Plog', 'Chebyshev', 'linear-Burke',
        'two-temperature-plasma', 'electron-collision-plasma',
        'vibrational-relaxation', 'constant-vibrational-relaxation',
        'Starikovskiy-vibrational-relaxation', 'Castela-vibrational-relaxation',
        'multi-state-resolved-vibrational-relaxation',
        'custom-rate-function',
    }'''

    content = content[:old_sup.start()] + new_sup + content[old_sup.end():]

    # 2. Update reaction parsing branches in parse_reaction (lines 600+)
    old_branch = re.search(r"elif rxn_type == 'linear-Burke':.*?rate_obj = LinearBurkeRate\(reference_rate, colliders\)", content, re.DOTALL)
    assert old_branch is not None, "Could not find linear-Burke branch in kinetics.py"

    extra_branches = '''
    elif rxn_type == 'two-temperature-plasma':
        from .rates import TwoTempPlasmaRate
        a_si, b, ea_si = parse_arrhenius(rate_node, eff_order)
        raw_ee = rate_node.get('Ea-electron', rate_node.get('EE', 0.0))
        if isinstance(raw_ee, str):
            match = re.fullmatch(r'\\s*([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*(.*)', raw_ee)
            if match:
                amt_str, unit_str = match.groups()
                ee_val = float(amt_str)
                unit_str = unit_str.strip()
                if unit_str:
                    ee_si = ee_val * parse_units(unit_str).factor
                else:
                    ea_unit = rate_node.get('Ea-unit', default_ea_unit)
                    ee_si = convert_activation_energy(ee_val, ea_unit, 'J/kmol')
            else:
                ea_unit = rate_node.get('Ea-unit', default_ea_unit)
                ee_si = convert_activation_energy(float(raw_ee), ea_unit, 'J/kmol')
        else:
            ea_unit = rate_node.get('Ea-unit', default_ea_unit)
            ee_si = convert_activation_energy(float(raw_ee), ea_unit, 'J/kmol')
        bg = float(rate_node.get('b-gas', rate_node.get('bg', 0.0)))
        tinv_raw = rate_node.get('T-inv', rate_node.get('Tinv', 0.0))
        tinv = parse_quantity(tinv_raw, 'K') if isinstance(tinv_raw, str) else float(tinv_raw)
        rate_obj = TwoTempPlasmaRate(A=a_si, b=b, Ea=ea_si, EE=ee_si, bg=bg, Tinv=tinv, allow_negative_A=allow_negative_a)
    elif rxn_type == 'electron-collision-plasma':
        from .rates import ElectronCollisionPlasmaRate
        energy_levels = rate_data.get('energy-levels', [])
        cross_sections = rate_data.get('cross-sections', [])
        rate_obj = ElectronCollisionPlasmaRate(energy_levels, cross_sections)
    elif rxn_type in ('vibrational-relaxation', 'constant-vibrational-relaxation'):
        from .rates import ConstantVibrationalRelaxationRate
        raw_a = rate_node.get('A', 1.0)
        a_si = float(raw_a) * (conc_inv_factor ** (eff_order - 1)) / time_factor
        rate_obj = ConstantVibrationalRelaxationRate(A=a_si, allow_negative_A=allow_negative_a)
    elif rxn_type == 'Starikovskiy-vibrational-relaxation':
        from .rates import StarikovskiyVibrationalRelaxationRate
        raw_a = rate_node.get('A', 1.0)
        a_si = float(raw_a) * (conc_inv_factor ** (eff_order - 1)) / time_factor
        b = float(rate_node.get('b', 0.0))
        c0 = float(rate_node.get('C0', 0.0))
        c13 = float(rate_node.get('C13', 0.0))
        cm = float(rate_node.get('Cm', 0.0))
        m = float(rate_node.get('m', 1.0))
        cn = float(rate_node.get('Cn', 0.0))
        n = float(rate_node.get('n', 1.0))
        rate_obj = StarikovskiyVibrationalRelaxationRate(A=a_si, b=b, C0=c0, C13=c13, Cm=cm, m=m, Cn=cn, n=n, allow_negative_A=allow_negative_a)
    elif rxn_type == 'Castela-vibrational-relaxation':
        from .rates import CastelaVibrationalRelaxationRate
        a_param = float(rate_node.get('a', 0.0))
        b_param = float(rate_node.get('b', 0.0))
        ref_p_raw = rate_node.get('reference-pressure', 101325.0)
        ref_p = parse_quantity(ref_p_raw, 'Pa') if isinstance(ref_p_raw, str) else float(ref_p_raw)
        rate_obj = CastelaVibrationalRelaxationRate(a=a_param, b=b_param, reference_pressure=ref_p, allow_negative_A=allow_negative_a)
    elif rxn_type == 'multi-state-resolved-vibrational-relaxation':
        from .rates import MultiStateResolvedVibrationalRelaxationRate
        raw_a = rate_node.get('A', 1.0)
        a_si = float(raw_a) * (conc_inv_factor ** (eff_order - 1)) / time_factor
        b = float(rate_node.get('b', 0.0))
        c0 = float(rate_node.get('C0', 0.0))
        c13 = float(rate_node.get('C13', 0.0))
        c23 = float(rate_node.get('C23', 0.0))
        rate_obj = MultiStateResolvedVibrationalRelaxationRate(A=a_si, b=b, C0=c0, C13=c13, C23=c23, allow_negative_A=allow_negative_a)
    elif rxn_type == 'custom-rate-function':
        from .rates import CustomFunc1Rate
        rate_obj = CustomFunc1Rate()'''

    idx_lb_end = old_branch.end()
    content = content[:idx_lb_end] + extra_branches + content[idx_lb_end:]

    # 3. Update _rate_value in ElementaryReaction
    old_rv = re.search(r"def _rate_value\(self, rate, temperature\):.*?return rate\(temperature\)", content, re.DOTALL)
    assert old_rv is not None, "Could not find _rate_value block"

    new_rv = '''def _rate_value(self, rate, temperature):
        from .rates import TwoTempPlasmaRate, ElectronCollisionPlasmaRate
        if isinstance(rate, BlowersMaselRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            if state is None:
                raise ValueError('Blowers-Masel rates require a live thermo state')
            rate.update_from_thermo(state, self.reactants, self.products)
        if isinstance(rate, (PlogRate, ChebyshevRate, LinearBurkeRate)):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            if state is None or not hasattr(state, 'pressure'):
                raise ValueError('pressure-dependent rates require a thermo state with pressure')
            if isinstance(rate, LinearBurkeRate):
                fractions = dict(zip(state.species_names, state.X))
                return rate(temperature, float(state.pressure), fractions)
            return rate(temperature, float(state.pressure))
        if isinstance(rate, TwoTempPlasmaRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            Te = getattr(state, 'electron_temperature', getattr(state, 'electronTemperature', temperature)) if state else temperature
            return rate(temperature, float(Te))
        if isinstance(rate, ElectronCollisionPlasmaRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            levels = getattr(state, 'electron_energy_levels', ()) if state else ()
            dist = getattr(state, 'electron_energy_distribution', ()) if state else ()
            return rate(levels, dist)
        return rate(temperature)'''

    content = content[:old_rv.start()] + new_rv + content[old_rv.end():]

    # 4. Update _forward_rate_constants in _VectorizedKinetics
    old_fwd = re.search(r"elif isinstance\(r\.rate, BlowersMaselRate\):.*?else:\s+out\[i\] = r\.rate\(T\)", content, re.DOTALL)
    assert old_fwd is not None, "Could not find _forward_rate_constants loop in kinetics.py"

    new_fwd = '''elif isinstance(r.rate, BlowersMaselRate):
                    r.rate.update_from_thermo(thermo, r.reactants, r.products)
                    out[i] = r.rate(T)
                elif type(r.rate).__name__ == "TwoTempPlasmaRate":
                    Te = getattr(thermo, 'electron_temperature', getattr(thermo, 'electronTemperature', T))
                    out[i] = r.rate(T, float(Te))
                elif type(r.rate).__name__ == "ElectronCollisionPlasmaRate":
                    levels = getattr(thermo, 'electron_energy_levels', ())
                    dist = getattr(thermo, 'electron_energy_distribution', ())
                    out[i] = r.rate(levels, dist)
                else:
                    out[i] = r.rate(T)'''

    content = content[:old_fwd.start()] + new_fwd + content[old_fwd.end():]

    KINETICS_FILE.write_text(content, encoding="utf-8")
    print(f"Updated {KINETICS_FILE}")


if __name__ == "__main__":
    update_rates()
    update_kinetics()
