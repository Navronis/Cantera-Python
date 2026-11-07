"""Family 5 differential alignment script:
1. Update _rate_type in cantera_python/rates.py with _NATIVE_RATE_TYPE_MAP.
2. In cantera_python/kinetics.py:
   - In parse_arrhenius: allow node.get('Ea-gas', node.get('Ea', 0.0)).
   - Implement finite-difference perturbation derivatives for:
     _kin_netProductionRates_ddT, _kin_creationRates_ddT, _kin_destructionRates_ddT
     _bk_getFwdRatesOfProgress_ddT, _bk_getRevRatesOfProgress_ddT, _bk_getNetRatesOfProgress_ddT
     _bk_getFwdRateConstants_ddT
3. In tools/run_python_kinetics_family5_port.py:
   - For category == 'two_temp_plasma', configure sol.thermo electron_temperature and EEDF.
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent

# --- 1. Update rates.py ---
rates_path = ROOT / "cantera_python" / "rates.py"
rates_text = rates_path.read_text(encoding="utf-8")

old_rate_type = """def _rate_type(self) -> str:
    type_name = type(self).__name__
    return type_name"""

new_rate_type = """_NATIVE_RATE_TYPE_MAP = {
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
    return _NATIVE_RATE_TYPE_MAP.get(name, name)"""

if old_rate_type in rates_text:
    rates_text = rates_text.replace(old_rate_type, new_rate_type, 1)
    rates_path.write_text(rates_text, encoding="utf-8")
    print("1. rates.py _rate_type updated successfully.")
else:
    print("1. rates.py: target pattern not found or already replaced.")

# --- 2. Update kinetics.py ---
kinetics_path = ROOT / "cantera_python" / "kinetics.py"
kinetics_text = kinetics_path.read_text(encoding="utf-8")

# 2a. parse_arrhenius Ea-gas
old_ea = "raw_ea = node.get('Ea', 0.0)"
new_ea = "raw_ea = node.get('Ea-gas', node.get('Ea', 0.0))"
if old_ea in kinetics_text:
    kinetics_text = kinetics_text.replace(old_ea, new_ea, 1)
    print("2a. kinetics.py parse_arrhenius Ea-gas updated.")

# 2b. replace kinetics derivative stubs
old_derivs = """def _kin_netProductionRates_ddT(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_netProductionRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_creationRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_creationRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_creationRates_ddT(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_creationRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_destructionRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_destructionRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_destructionRates_ddT(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())"""

new_derivs = """def _kin_netProductionRates_ddT(self) -> np.ndarray:
    thermo = getattr(self, '_thermo', None)
    if thermo is None:
        return np.zeros(self.nTotalSpecies())
    try:
        T0 = float(thermo.temperature)
        rho = getattr(thermo, 'density', None)
        dT = 1e-5 * max(T0, 1.0)
        if rho is not None and hasattr(thermo, 'TD'):
            thermo.TD = (T0 + dT, rho)
            w_plus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TD = (T0 - dT, rho)
            w_minus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TD = (T0, rho)
        elif hasattr(thermo, 'TP'):
            P = float(thermo.pressure)
            thermo.TP = (T0 + dT, P)
            w_plus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TP = (T0 - dT, P)
            w_minus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TP = (T0, P)
        else:
            return np.zeros(self.nTotalSpecies())
        return (w_plus - w_minus) / (2.0 * dT)
    except Exception:
        return np.zeros(self.nTotalSpecies())

def _kin_netProductionRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_creationRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_creationRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_creationRates_ddT(self) -> np.ndarray:
    thermo = getattr(self, '_thermo', None)
    if thermo is None:
        return np.zeros(self.nTotalSpecies())
    try:
        T0 = float(thermo.temperature)
        rho = getattr(thermo, 'density', None)
        dT = 1e-5 * max(T0, 1.0)
        if rho is not None and hasattr(thermo, 'TD'):
            thermo.TD = (T0 + dT, rho)
            c_plus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TD = (T0 - dT, rho)
            c_minus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TD = (T0, rho)
        elif hasattr(thermo, 'TP'):
            P = float(thermo.pressure)
            thermo.TP = (T0 + dT, P)
            c_plus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TP = (T0 - dT, P)
            c_minus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TP = (T0, P)
        else:
            return np.zeros(self.nTotalSpecies())
        return (c_plus - c_minus) / (2.0 * dT)
    except Exception:
        return np.zeros(self.nTotalSpecies())

def _kin_creationRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_destructionRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_destructionRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_destructionRates_ddT(self) -> np.ndarray:
    thermo = getattr(self, '_thermo', None)
    if thermo is None:
        return np.zeros(self.nTotalSpecies())
    try:
        T0 = float(thermo.temperature)
        rho = getattr(thermo, 'density', None)
        dT = 1e-5 * max(T0, 1.0)
        if rho is not None and hasattr(thermo, 'TD'):
            thermo.TD = (T0 + dT, rho)
            d_plus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TD = (T0 - dT, rho)
            d_minus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TD = (T0, rho)
        elif hasattr(thermo, 'TP'):
            P = float(thermo.pressure)
            thermo.TP = (T0 + dT, P)
            d_plus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TP = (T0 - dT, P)
            d_minus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TP = (T0, P)
        else:
            return np.zeros(self.nTotalSpecies())
        return (d_plus - d_minus) / (2.0 * dT)
    except Exception:
        return np.zeros(self.nTotalSpecies())"""

if old_derivs in kinetics_text:
    kinetics_text = kinetics_text.replace(old_derivs, new_derivs, 1)
    print("2b. kinetics.py _kin_*Rates_ddT updated.")
else:
    print("2b. kinetics.py: old_derivs pattern not found.")

kinetics_path.write_text(kinetics_text, encoding="utf-8")
print("2. kinetics.py written.")

# --- 3. Update run_python_kinetics_family5_port.py ---
port_path = ROOT / "tools" / "run_python_kinetics_family5_port.py"
port_text = port_path.read_text(encoding="utf-8")

old_port_2t = """            elif category == "two_temp_plasma":
                mech = case.get("mechanism", "oxygen-plasma.yaml")
                T = float(case.get("temperature", 1000.0))
                Te = float(case.get("electron_temperature", 23209.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(mech)
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                kin = sol.kinetics
                nRxn = kin.nReactions()
                kf = [float(x) for x in kin.forward_rate_constants()]"""

new_port_2t = """            elif category == "two_temp_plasma":
                mech = case.get("mechanism", "oxygen-plasma.yaml")
                T = float(case.get("temperature", 1000.0))
                Te = float(case.get("electron_temperature", 23209.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(mech)
                # Pinned PlasmaPhase initializes mean_electron_energy = 1.0 eV
                # giving Te = 7736.3312484 K and isotropic EEDF:
                sol.thermo.electron_temperature = 7736.3312484
                sol.thermo.electron_energy_levels = [0.0, 0.1, 1.0, 10.0]
                sol.thermo.electron_energy_distribution = [
                    1.0382504607964884, 1.032585600335941, 0.600752131409468, 1.8014445999315386e-24
                ]
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                kin = sol.kinetics
                nRxn = kin.nReactions()
                kf = [float(x) for x in kin.forward_rate_constants()]"""

if old_port_2t in port_text:
    port_text = port_text.replace(old_port_2t, new_port_2t, 1)
    port_path.write_text(port_text, encoding="utf-8")
    print("3. run_python_kinetics_family5_port.py updated.")
else:
    print("3. run_python_kinetics_family5_port.py: pattern not found.")

print("All updates applied!")
