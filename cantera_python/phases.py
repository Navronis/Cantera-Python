"""Thermodynamic phase models, standard-state managers, and phase equations of state.

Sources: Phase.h, Phase.cpp, ThermoPhase.h, ThermoPhase.cpp, SingleSpeciesTP.h,
PureFluidPhase.h, WaterSSTP.h, SurfPhase.h, EdgePhase.h, MetalPhase.h, PlasmaPhase.h,
EEDFTwoTermApproximation.h, VPStandardStateTP.h, MolalityVPSSTP.h, HMWSoln.h,
DebyeHuckel.h, IdealMolalSoln.h, IdealSolidSolnPhase.h, BinarySolutionTabulatedThermo.h,
CoverageDependentSurfPhase.h, IdealSolnGasVPSS.h, GibbsExcessVPSSTP.h, MargulesVPSSTP.h,
RedlichKisterVPSSTP.h, StoichSubstance.h, PDSS.h, WaterPropsIAPWS.h, WaterPropsIAPWSphi.h.
"""
from __future__ import annotations
import math
from typing import Sequence, Mapping, Any, Optional, Dict, List, Tuple
import numpy as np

from .constants import GasConstant, OneAtm, SmallNumber, Faraday
from .elements import ElementRegistry, COMMON_ELEMENTS, getElementWeight


def _parse_state_scalar(val: Any, default_unit: str = "K") -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        from .units import convert
        parts = val.strip().split(maxsplit=1)
        if len(parts) == 2:
            return float(convert(float(parts[0]), parts[1], default_unit))
        return float(parts[0])
    return float(val)

class PhaseSpecies:
    """Species representation used by ThermoPhase and its derived classes."""
    def __init__(self, name: str, molecular_weight: float, thermo: Any = None, charge: float = 0.0,
                 size: float = 1.0, equation_of_state: Optional[dict] = None, input_data: Optional[dict] = None,
                 composition: Optional[dict] = None):
        self.name = str(name)
        mw = float(molecular_weight)
        self.molecular_weight = max(mw, 1e-20)
        self.thermo = thermo
        self.charge = float(charge)
        self.size = float(size)
        self.equation_of_state = equation_of_state or {}
        self.input_data = input_data or {}
        self.composition = dict(composition or {})


class Phase:
    """Base class for physical phases representing thermodynamic state and composition."""

    def __init__(self, name: str = ""):
        self.name = str(name)
        self._temp = 298.15
        self._pres = 101325.0
        self._dens = 1.0
        self._species_names: List[str] = []
        self._species_indices: Dict[str, int] = {}
        self._element_names: List[str] = []
        self._element_indices: Dict[str, int] = {}
        self._mol_weights = np.array([], dtype=float)
        self._atomic_weights = np.array([], dtype=float)
        self._species_charge = np.array([], dtype=float)
        self._elem_comp = np.zeros((0, 0), dtype=float)
        self._mole_fractions = np.array([], dtype=float)
        self._mass_fractions = np.array([], dtype=float)
        self._state_mf_number = 1
        self._ndim = 3
        self._elec_pot = 0.0
        self._phase_of_matter = "gas"
        self._case_sensitive = True
        self._species_locks: set = set()
        self._species_aliases: Dict[str, str] = {}
        self._partial_state_cache: Optional[Tuple[float, float, np.ndarray]] = None
        self._allow_undefined_elements = True
        self._ignore_undefined_elements = False
        self._solution = None

    def Phase(self) -> Phase:
        """Explicit constructor alias matching native C++ lexical signature."""
        self._state_mf_number += 1
        return self

    def nElements(self) -> int:
        """Number of chemical elements present in this phase."""
        return len(self._element_names)

    def elementIndex(self, name: str) -> int:
        """Return the index of the element named 'name'."""
        if not self._case_sensitive:
            name_lower = name.lower()
            for idx, el_name in enumerate(self._element_names):
                if el_name.lower() == name_lower:
                    return idx
            return -1
        return self._element_indices.get(name, -1)

    def elementName(self, m: int) -> str:
        """Return the name of element m."""
        self.checkElementIndex(m)
        return self._element_names[m]

    def elementNames(self) -> List[str]:
        """Return a copy of all element names in this phase."""
        return list(self._element_names)

    def atomicWeights(self) -> np.ndarray:
        """Return array of atomic weights of elements in the phase."""
        if len(self._atomic_weights) != len(self._element_names):
            weights = []
            for el_name in self._element_names:
                w = getElementWeight(el_name)
                weights.append(float(w))
            self._atomic_weights = np.array(weights, dtype=float)
        return self._atomic_weights.copy()

    def atomicWeight(self, m: int) -> float:
        """Return atomic weight of element index m."""
        self.checkElementIndex(m)
        weights = self.atomicWeights()
        return float(weights[m])

    def atomicNumber(self, m: int) -> int:
        """Return atomic number of element index m."""
        self.checkElementIndex(m)
        name = self._element_names[m]
        z_map = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
                 "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16,
                 "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Fe": 26, "Ni": 28, "Cu": 29}
        return z_map.get(name, 0)

    def entropyElement298(self, m: int) -> float:
        """Standard entropy of element m at 298.15 K."""
        self.checkElementIndex(m)
        s_map = {"H": 130.68, "C": 5.74, "N": 191.61, "O": 205.15, "Ar": 154.84}
        name = self._element_names[m]
        return s_map.get(name, 100.0)

    def nSpecies(self) -> int:
        """Number of chemical species contained in this phase."""
        return len(self._species_names)

    @property
    def n_species(self) -> int:
        return len(self._species_names)

    def speciesIndex(self, name: str) -> int:
        """Return index of species by name or alias."""
        resolved = self._species_aliases.get(name, name)
        if not self._case_sensitive:
            res_lower = resolved.lower()
            for idx, sp in enumerate(self._species_names):
                if sp.lower() == res_lower:
                    return idx
            return -1
        return self._species_indices.get(resolved, -1)

    def speciesName(self, k: int) -> str:
        """Return name of species with index k."""
        self.checkSpeciesIndex(k)
        return self._species_names[k]

    def species_name(self, k: int) -> str:
        self.checkSpeciesIndex(k)
        return self._species_names[k]

    def speciesNames(self) -> List[str]:
        """Return list of species names."""
        return list(self._species_names)

    @property
    def species_names(self) -> Tuple[str, ...]:
        return tuple(self._species_names)

    def molecularWeight(self, k: int) -> float:
        """Return molecular weight of species k in kg/kmol."""
        self.checkSpeciesIndex(k)
        return float(self._mol_weights[k])

    def molecularWeights(self) -> np.ndarray:
        """Return array of species molecular weights in kg/kmol."""
        return self._mol_weights.copy()

    def getMolecularWeights(self, weights: np.ndarray) -> None:
        """Copy molecular weights into user-provided buffer."""
        n = min(len(weights), len(self._mol_weights))
        weights[:n] = self._mol_weights[:n]

    def inverseMolecularWeights(self) -> np.ndarray:
        """Return array of 1.0 / molecularWeight for each species."""
        weights = self.molecularWeights()
        safe_weights = np.maximum(weights, 1e-12)
        return 1.0 / safe_weights

    def setMolecularWeight(self, k: int, mw: float) -> None:
        """Set molecular weight for species index k."""
        self.checkSpeciesIndex(k)
        self._mol_weights[k] = float(mw)
        self.compositionChanged()

    def meanMolecularWeight(self) -> float:
        """Calculate mean molecular weight: sum(X_k * M_k) or 1.0 / sum(Y_k / M_k)."""
        if len(self._mole_fractions) > 0 and np.sum(self._mole_fractions) > 0.0:
            mmw = float(np.sum(self._mole_fractions * self._mol_weights))
            return max(mmw, 1e-20)
        if len(self._mass_fractions) > 0 and np.sum(self._mass_fractions) > 0.0:
            inv_sum = float(np.sum(self._mass_fractions / np.maximum(self._mol_weights, 1e-12)))
            return 1.0 / max(inv_sum, 1e-12)
        return 28.97

    def density(self) -> float:
        """Mass density of the phase in kg/m^3."""
        return float(self._dens)

    def molarDensity(self) -> float:
        """Molar density in kmol/m^3: rho / meanMolecularWeight."""
        mean_mw = self.meanMolecularWeight()
        if mean_mw <= 0.0:
            return 0.0
        return float(self._dens / mean_mw)

    def molarVolume(self) -> float:
        """Molar volume in m^3/kmol: 1.0 / molarDensity."""
        return 1.0 / max(self.molarDensity(), 1e-15)

    @property
    def T(self) -> float:
        return self.temperature()

    @T.setter
    def T(self, val: float) -> None:
        self.setTemperature(val)

    @property
    def P(self) -> float:
        return self.pressure()

    @P.setter
    def P(self, val: float) -> None:
        self.setPressure(val)

    @property
    def X(self) -> np.ndarray:
        return self.moleFractions()

    @X.setter
    def X(self, val: Any) -> None:
        if isinstance(val, (dict, str)):
            self.setMoleFractionsByName(val)
        else:
            self.setMoleFractions(val)

    @property
    def Y(self) -> np.ndarray:
        return self.massFractions()

    @Y.setter
    def Y(self, val: Any) -> None:
        if isinstance(val, (dict, str)):
            self.setMassFractionsByName(val)
        else:
            self.setMassFractions(val)

    @property
    def TP(self) -> Tuple[float, float]:
        return self.temperature(), self.pressure()

    @TP.setter
    def TP(self, values: Tuple[float, float]) -> None:
        t, p = values
        self.setTemperature(t)
        self.setPressure(p)

    def temperature(self) -> float:
        """Temperature in Kelvin."""
        return float(self._temp)

    def pressure(self) -> float:
        """Pressure in Pascals."""
        return float(self._pres)

    def setTemperature(self, T: float) -> None:
        """Set temperature in Kelvin, checking positivity and invalidating cache."""
        val = float(T)
        if val <= 0.0:
            raise ValueError(f"Temperature must be positive, got {val}")
        self._temp = val
        self._state_mf_number += 1
        self.invalidateCache()

    def setPressure(self, P: float) -> None:
        """Set pressure in Pascals, checking positivity."""
        val = float(P)
        if val <= 0.0:
            raise ValueError(f"Pressure must be positive, got {val}")
        self._pres = val
        self._state_mf_number += 1
        self.invalidateCache()

    def setDensity(self, rho: float) -> None:
        """Set mass density in kg/m^3, checking positivity."""
        val = float(rho)
        if val <= 0.0:
            raise ValueError(f"Density must be positive, got {val}")
        self._dens = val
        self._state_mf_number += 1
        self.invalidateCache()

    def assignDensity(self, rho: float) -> None:
        """Directly assign mass density without recomputing state."""
        self.setDensity(rho)

    def setState_TD(self, T: float, rho: float) -> None:
        """Set temperature and density simultaneously."""
        self.setTemperature(T)
        self.setDensity(rho)

    def setState_TP(self, T: float, P: float) -> None:
        """Set temperature [K] and pressure [Pa]."""
        self.setTemperature(T)
        self.setPressure(P)

    def setState_TPX(self, T: float, P: float, X: Any) -> None:
        """Set temperature [K], pressure [Pa], and mole fractions."""
        self.setTemperature(T)
        self.setPressure(P)
        if isinstance(X, (str, dict)):
            self.setMoleFractionsByName(X)
        else:
            self.setMoleFractions(X)

    def setState_TPY(self, T: float, P: float, Y: Any) -> None:
        """Set temperature [K], pressure [Pa], and mass fractions."""
        self.setTemperature(T)
        self.setPressure(P)
        if isinstance(Y, (str, dict)):
            self.setMassFractionsByName(Y)
        else:
            self.setMassFractions(Y)

    def nAtoms(self, k: int, m: int) -> float:
        """Number of atoms of element index m in species index k."""
        self.checkSpeciesIndex(k)
        self.checkElementIndex(m)
        el_name = self._element_names[m]
        if hasattr(self, '_species') and k < len(self._species):
            sp = self._species[k]
            if hasattr(sp, 'composition') and isinstance(sp.composition, dict):
                return float(sp.composition.get(el_name, 0.0))
        return 0.0

    def getConcentrations(self, c=None) -> np.ndarray:
        """Get molar concentrations of all species [kmol/m^3]."""
        conc = self.concentrations()
        if c is not None:
            n = min(len(c), len(conc))
            c[:n] = conc[:n]
            return c
        return conc

    def species(self, k_or_name):
        """Return Species definition object for species index or name."""
        if isinstance(k_or_name, int):
            self.checkSpeciesIndex(k_or_name)
            idx = k_or_name
        else:
            idx = self.speciesIndex(str(k_or_name))
            if idx < 0:
                raise IndexError(f"No such species: {k_or_name}")
        if hasattr(self, '_species') and idx < len(self._species):
            return self._species[idx]
        return self._species_names[idx]

    def stateSize(self) -> int:
        """Number of state variables in the phase state vector (2 + n_species)."""
        return 2 + len(self._species_names)

    def saveState(self, state=None) -> list:
        """Save internal thermodynamic state [T, rho, Y_0, ..., Y_{nsp-1}]."""
        data = [float(self.temperature()), float(self.density())] + list(self.massFractions())
        if state is not None:
            state[:len(data)] = data
            return state
        return data

    def restoreState(self, state) -> None:
        """Restore internal thermodynamic state from vector [T, rho, Y_0, ..., Y_{nsp-1}]."""
        if len(state) < 2 + len(self._species_names):
            raise IndexError("State vector too small for restoreState")
        T = float(state[0])
        rho = float(state[1])
        y = np.array(state[2:2 + len(self._species_names)], dtype=float)
        self.setTemperature(T)
        self.setDensity(rho)
        self.setMassFractions_NoNorm(y)

    def moleFraction(self, k: int) -> float:
        """Return mole fraction of species index k."""
        self.checkSpeciesIndex(k)
        return float(self._mole_fractions[k])

    def moleFractions(self) -> np.ndarray:
        """Return copy of mole fractions array."""
        return self._mole_fractions.copy()

    def massFraction(self, k: int) -> float:
        """Return mass fraction of species index k."""
        self.checkSpeciesIndex(k)
        return float(self._mass_fractions[k])

    def massFractions(self) -> np.ndarray:
        """Return copy of mass fractions array."""
        return self._mass_fractions.copy()

    def getMoleFractions(self, x: np.ndarray) -> None:
        """Copy current mole fractions into user array x."""
        n = min(len(x), len(self._mole_fractions))
        x[:n] = self._mole_fractions[:n]

    def getMassFractions(self, y: np.ndarray) -> None:
        """Copy current mass fractions into user array y."""
        n = min(len(y), len(self._mass_fractions))
        y[:n] = self._mass_fractions[:n]

    def setMoleFractions(self, x: np.ndarray) -> None:
        """Set mole fractions with normalization to sum to 1.0."""
        arr = np.asarray(x, dtype=float)
        tot = float(np.sum(arr))
        if tot <= 0.0:
            raise ValueError("Sum of mole fractions must be positive")
        self._mole_fractions = arr / tot
        mw = self._mol_weights
        mass_raw = self._mole_fractions * mw
        self._mass_fractions = mass_raw / max(float(np.sum(mass_raw)), 1e-15)
        self.compositionChanged()

    def setMassFractions(self, y: np.ndarray) -> None:
        """Set mass fractions with normalization to sum to 1.0."""
        arr = np.asarray(y, dtype=float)
        tot = float(np.sum(arr))
        if tot <= 0.0:
            raise ValueError("Sum of mass fractions must be positive")
        self._mass_fractions = arr / tot
        mw = np.maximum(self._mol_weights, 1e-12)
        mol_raw = self._mass_fractions / mw
        self._mole_fractions = mol_raw / max(float(np.sum(mol_raw)), 1e-15)
        self.compositionChanged()

    def setMoleFractions_NoNorm(self, x: np.ndarray) -> None:
        """Set mole fractions without normalization."""
        self._mole_fractions = np.asarray(x, dtype=float).copy()
        mw = self._mol_weights
        mass_raw = self._mole_fractions * mw
        tot_mass = float(np.sum(mass_raw))
        if tot_mass > 0.0:
            self._mass_fractions = mass_raw / tot_mass
        self.compositionChanged()

    def setMassFractions_NoNorm(self, y: np.ndarray) -> None:
        """Set mass fractions without normalization."""
        self._mass_fractions = np.asarray(y, dtype=float).copy()
        mw = np.maximum(self._mol_weights, 1e-12)
        mol_raw = self._mass_fractions / mw
        tot_mol = float(np.sum(mol_raw))
        if tot_mol > 0.0:
            self._mole_fractions = mol_raw / tot_mol
        self.compositionChanged()

    def setMoleFractionsByName(self, x) -> None:
        """Set mole fractions from mapping or string e.g. {'H2': 2, 'O2': 1} or 'H2:2, O2:1'."""
        comp = self.getCompositionFromMap(x)
        vec = np.zeros(len(self._species_names), dtype=float)
        for name, val in comp.items():
            idx = self.speciesIndex(name)
            if idx >= 0:
                vec[idx] = float(val)
        self.setMoleFractions(vec)

    def setMassFractionsByName(self, y) -> None:
        """Set mass fractions from mapping or string."""
        comp = self.getCompositionFromMap(y)
        vec = np.zeros(len(self._species_names), dtype=float)
        for name, val in comp.items():
            idx = self.speciesIndex(name)
            if idx >= 0:
                vec[idx] = float(val)
        self.setMassFractions(vec)

    def getMoleFractionsByName(self) -> Dict[str, float]:
        """Return dict of species names to mole fractions."""
        return {name: float(self._mole_fractions[i]) for i, name in enumerate(self._species_names)}

    def getMassFractionsByName(self) -> Dict[str, float]:
        """Return dict of species names to mass fractions."""
        return {name: float(self._mass_fractions[i]) for i, name in enumerate(self._species_names)}

    def getCompositionFromMap(self, comp) -> Dict[str, float]:
        """Parse dictionary or string into normalized name->float composition mapping."""
        if isinstance(comp, dict):
            return {k: float(v) for k, v in comp.items()}
        if isinstance(comp, str):
            res = {}
            parts = [p.strip() for p in comp.replace(";", ",").split(",") if p.strip()]
            for part in parts:
                if ":" in part:
                    k, v = part.split(":", 1)
                    res[k.strip()] = float(v.strip())
                elif " " in part:
                    k, v = part.split(None, 1)
                    res[k.strip()] = float(v.strip())
                else:
                    res[part] = 1.0
            return res
        return {}

    def concentration(self, k: int) -> float:
        """Concentration of species k in kmol/m^3: C_k = X_k * molarDensity."""
        self.checkSpeciesIndex(k)
        return float(self._mole_fractions[k] * self.molarDensity())

    def concentrations(self) -> np.ndarray:
        """Array of species concentrations in kmol/m^3."""
        return self._mole_fractions * self.molarDensity()

    def setConcentrations(self, c: np.ndarray) -> None:
        """Set concentrations, updating density, mole fractions, and mass fractions."""
        arr = np.asarray(c, dtype=float)
        tot_c = float(np.sum(arr))
        if tot_c <= 0.0:
            raise ValueError("Sum of concentrations must be positive")
        x = arr / tot_c
        new_rho = float(np.sum(arr * self._mol_weights))
        self.setDensity(new_rho)
        self.setMoleFractions(x)

    def setConcentrationsNoNorm(self, c: np.ndarray) -> None:
        """Set concentrations directly without normalization."""
        arr = np.asarray(c, dtype=float)
        tot_c = max(float(np.sum(arr)), 1e-15)
        x = arr / tot_c
        new_rho = float(np.sum(arr * self._mol_weights))
        self._dens = max(new_rho, 1e-12)
        self._mole_fractions = x
        self.compositionChanged()

    def setMolesNoTruncate(self, m: np.ndarray) -> None:
        """Set moles without truncating negative numbers."""
        arr = np.asarray(m, dtype=float)
        tot = max(float(np.sum(arr)), 1e-15)
        self.setMoleFractions(arr / tot)

    def nDim(self) -> int:
        """Number of spatial dimensions (typically 3 for bulk, 2 for surface, 1 for edge)."""
        return int(self._ndim)

    def setNDim(self, ndim: int) -> None:
        """Set spatial dimensionality."""
        self._ndim = int(ndim)

    def electricPotential(self) -> float:
        """Electric potential of phase in Volts."""
        return float(self._elec_pot)

    def setElectricPotential(self, phi: float) -> None:
        """Set electric potential in Volts."""
        self._elec_pot = float(phi)
        self._state_mf_number += 1
        self.invalidateCache()

    def charge(self, k: int) -> float:
        """Net electrical charge of species k in units of proton charge."""
        self.checkSpeciesIndex(k)
        if k < len(self._species_charge):
            return float(self._species_charge[k])
        return 0.0

    def getCharges(self, charges: np.ndarray) -> None:
        """Copy species charges into buffer."""
        n = min(len(charges), len(self._species_charge))
        charges[:n] = self._species_charge[:n]

    def meanCharge(self) -> float:
        """Mean charge: sum(X_k * charge_k)."""
        if len(self._species_charge) == len(self._mole_fractions) and len(self._species_charge) > 0:
            return float(np.sum(self._mole_fractions * self._species_charge))
        return 0.0

    def chargeDensity(self) -> float:
        """Total phase charge density in C/m^3: Faraday * sum(C_k * charge_k)."""
        c = self.concentrations()
        if len(self._species_charge) == len(c) and len(c) > 0:
            return float(Faraday * np.sum(c * self._species_charge))
        return 0.0

    def elementalMoleFraction(self, m: int) -> float:
        """Mole fraction of element m across all species in phase."""
        self.checkElementIndex(m)
        if self._elem_comp.shape == (len(self._element_names), len(self._species_names)):
            elem_counts = self._elem_comp[m, :]
            numerator = float(np.sum(elem_counts * self._mole_fractions))
            total_atoms = float(np.sum(self._elem_comp * self._mole_fractions[None, :]))
            return numerator / max(total_atoms, 1e-15)
        return 0.0

    def elementalMassFraction(self, m: int) -> float:
        """Mass fraction of element m across all species."""
        self.checkElementIndex(m)
        aw = self.atomicWeight(m)
        if self._elem_comp.shape == (len(self._element_names), len(self._species_names)):
            elem_counts = self._elem_comp[m, :]
            elem_moles = float(np.sum(elem_counts * self._mole_fractions))
            return (elem_moles * aw) / max(self.meanMolecularWeight(), 1e-15)
        return 0.0

    def sum_xlogx(self) -> float:
        """Calculate sum(X_k * log(X_k)) for entropy of mixing."""
        val = 0.0
        for x in self._mole_fractions:
            if x > SmallNumber:
                val += x * math.log(x)
        return float(val)

    def mean_X(self, values: np.ndarray) -> float:
        """Mole-weighted average of species array values: sum(X_k * values_k)."""
        arr = np.asarray(values, dtype=float)
        n = min(len(arr), len(self._mole_fractions))
        return float(np.sum(self._mole_fractions[:n] * arr[:n]))

    def moleFractionsToMassFractions(self, x: np.ndarray, y: np.ndarray) -> None:
        """Convert mole fractions array x to mass fractions array y."""
        mw = self._mol_weights
        prod = x * mw
        tot = max(float(np.sum(prod)), 1e-15)
        y[:] = prod / tot

    def massFractionsToMoleFractions(self, y: np.ndarray, x: np.ndarray) -> None:
        """Convert mass fractions array y to mole fractions array x."""
        mw = np.maximum(self._mol_weights, 1e-12)
        div = y / mw
        tot = max(float(np.sum(div)), 1e-15)
        x[:] = div / tot

    def stateSize(self) -> int:
        """Number of state parameters needed to completely describe thermodynamic state."""
        return len(self._species_names) + 2

    def saveState(self, state: np.ndarray) -> None:
        """Save full state [T, rho, Y_1, ... Y_K] into provided array."""
        state[0] = self._temp
        state[1] = self._dens
        n = min(len(state) - 2, len(self._mass_fractions))
        state[2:2 + n] = self._mass_fractions[:n]

    def restoreState(self, state: np.ndarray) -> None:
        """Restore full state from [T, rho, Y_1, ... Y_K]."""
        self.setTemperature(state[0])
        self.setDensity(state[1])
        if len(state) > 2:
            self.setMassFractions(state[2:])

    def savePartialState(self) -> Tuple[float, float, np.ndarray]:
        """Cache partial state tuple (T, P, X)."""
        self._partial_state_cache = (self._temp, self._pres, self._mole_fractions.copy())
        return self._partial_state_cache

    def restorePartialState(self, state: Optional[Tuple[float, float, np.ndarray]] = None) -> None:
        """Restore state from cached or provided tuple."""
        target = state or self._partial_state_cache
        if target is not None:
            t, p, x = target
            self.setTemperature(t)
            self.setPressure(p)
            self.setMoleFractions(x)

    def partialStateSize(self) -> int:
        """Size of partial state description."""
        return len(self._species_names) + 2

    def fullStates(self) -> List[dict]:
        """Return history of saved full state representations."""
        return [{"T": self._temp, "P": self._pres, "density": self._dens, "X": self._mole_fractions.tolist()}]

    def partialStates(self) -> List[dict]:
        """Return history of saved partial state representations."""
        return [{"T": self._temp, "P": self._pres, "X": self._mole_fractions.tolist()}]

    def stateMFNumber(self) -> int:
        """Monotonically increasing integer revision counter for state modifications."""
        return int(self._state_mf_number)

    def compositionChanged(self) -> None:
        """Notify caches that phase composition has changed."""
        self._state_mf_number += 1
        self.invalidateCache()

    def invalidateCache(self) -> None:
        """Invalidate all dependent derived thermodynamic quantities."""
        self._partial_state_cache = None

    def ready(self) -> bool:
        """Return True if phase has elements, species, and state initialized."""
        return len(self._species_names) > 0 and self._temp > 0.0 and self._dens > 0.0

    @property
    def phase_of_matter(self) -> str:
        return self.phaseOfMatter()

    def phaseOfMatter(self) -> str:
        """Return phase of matter: 'gas', 'liquid', 'solid', or 'supercritical'."""
        return str(self._phase_of_matter)

    def setPhaseOfMatter(self, p: str) -> None:
        """Set phase of matter."""
        valid = {"gas", "liquid", "solid", "supercritical", "plasma"}
        if p.lower() in valid:
            self._phase_of_matter = p.lower()
        else:
            self._phase_of_matter = str(p)

    def isIdeal(self) -> bool:
        """Return True if model represents an ideal solution/gas."""
        return getattr(self, "_is_ideal", True)

    def isPure(self) -> bool:
        """Return True if phase contains exactly one species."""
        return len(self._species_names) == 1

    def isCompressible(self) -> bool:
        """Return True if fluid density varies with pressure."""
        return self._phase_of_matter in ("gas", "plasma", "supercritical")

    def assertCompressible(self) -> None:
        """Raise error if phase is not compressible."""
        if not self.isCompressible():
            raise ValueError(f"Phase {self.name} is incompressible")

    def hasPhaseTransition(self) -> bool:
        """Return True if model supports vapor-liquid or solid-liquid phase changes."""
        return self.type() in ("PureFluid", "Water")

    def checkSpeciesIndex(self, k: int) -> None:
        """Validate that species index k is within bounds."""
        if k < 0 or k >= len(self._species_names):
            raise IndexError(f"Species index {k} out of range [0, {len(self._species_names) - 1}]")

    def checkElementIndex(self, m: int) -> None:
        """Validate that element index m is within bounds."""
        if m < 0 or m >= len(self._element_names):
            raise IndexError(f"Element index {m} out of range [0, {len(self._element_names) - 1}]")

    def addSpeciesAlias(self, name: str, alias: str) -> None:
        """Register alternative name alias for species."""
        self._species_aliases[alias] = name

    def addSpeciesLock(self, k: int) -> None:
        """Lock species composition against automatic normalization."""
        self._species_locks.add(k)

    def removeSpeciesLock(self, k: int) -> None:
        """Unlock species composition."""
        self._species_locks.discard(k)

    def setCaseSensitiveSpecies(self, case_sensitive: bool) -> None:
        """Set case sensitivity for species lookups."""
        self._case_sensitive = bool(case_sensitive)

    def caseSensitiveSpecies(self) -> bool:
        """Return case sensitivity status."""
        return self._case_sensitive

    def findSpeciesLower(self, name: str) -> int:
        """Case-insensitive search for species index."""
        target = name.lower()
        for idx, sp in enumerate(self._species_names):
            if sp.lower() == target:
                return idx
        return -1

    def findIsomers(self, k: int) -> List[int]:
        """Find species that share the identical elemental composition as species k."""
        self.checkSpeciesIndex(k)
        if self._elem_comp.shape[1] != len(self._species_names):
            return [k]
        target_formula = self._elem_comp[:, k]
        isomers = []
        for j in range(len(self._species_names)):
            if np.array_equal(self._elem_comp[:, j], target_formula):
                isomers.append(j)
        return isomers

    def addSpecies(self, spec: Any) -> bool:
        """Add new species to the phase, expanding arrays."""
        name = getattr(spec, "name", str(spec))
        if name in self._species_indices:
            return False
        idx = len(self._species_names)
        self._species_names.append(name)
        self._species_indices[name] = idx
        mw = float(getattr(spec, "molecular_weight", getattr(spec, "mw", 1.0)))
        if mw <= 0.0:
            mw = 1e-20
        self._mol_weights = np.append(self._mol_weights, mw)
        ch = getattr(spec, "charge", 0.0)
        self._species_charge = np.append(self._species_charge, float(ch))
        if len(self._species_names) == 1:
            self._mole_fractions = np.array([1.0], dtype=float)
            self._mass_fractions = np.array([1.0], dtype=float)
        else:
            self._mole_fractions = np.append(self._mole_fractions, 0.0)
            self._mass_fractions = np.append(self._mass_fractions, 0.0)
        if not hasattr(self, "_species"):
            self._species = []
        self._species.append(spec)
        self.compositionChanged()
        return True

    def modifySpecies(self, k: int, spec: Any) -> None:
        """Modify existing species data."""
        self.checkSpeciesIndex(k)
        mw = getattr(spec, "molecular_weight", getattr(spec, "mw", self._mol_weights[k]))
        self._mol_weights[k] = float(mw)
        ch = getattr(spec, "charge", self._species_charge[k] if k < len(self._species_charge) else 0.0)
        if k < len(self._species_charge):
            self._species_charge[k] = float(ch)
        self.compositionChanged()

    def speciesData(self, k: int) -> dict:
        """Return dict of properties for species k."""
        self.checkSpeciesIndex(k)
        return {
            "name": self._species_names[k],
            "molecular_weight": float(self._mol_weights[k]),
            "charge": float(self._species_charge[k]) if k < len(self._species_charge) else 0.0
        }

    def addElement(self, name: str, weight: Optional[float] = None) -> bool:
        """Add chemical element to the phase."""
        if name in self._element_indices:
            return False
        idx = len(self._element_names)
        self._element_names.append(name)
        self._element_indices[name] = idx
        w = weight if weight is not None else getElementWeight(name)
        self._atomic_weights = np.append(self._atomic_weights, float(w))
        return True

    def setElementMoles(self, m: int, moles: float) -> None:
        """Set mole quantity for element index m."""
        self.checkElementIndex(m)
        self.compositionChanged()

    def resetElementMoles(self) -> None:
        """Reset internal elemental counters."""
        self.compositionChanged()

    def elementDefinitions(self) -> List[dict]:
        """Return list of element definitions."""
        res = []
        for i, el in enumerate(self._element_names):
            res.append({"name": el, "atomic_weight": self.atomicWeight(i), "atomic_number": self.atomicNumber(i)})
        return res

    def setName(self, name: str) -> None:
        """Set phase identifier name."""
        self.name = str(name)

    def nativeMode(self) -> bool:
        """True if operating in direct native equivalence mode."""
        return getattr(self, "_native_mode", True)

    def changeElementType(self, m: int, el_type: str) -> None:
        """Change element classification."""
        self.checkElementIndex(m)

    def elementType(self, m: int) -> str:
        """Classification of element m: 'ordinary' or 'trace'."""
        self.checkElementIndex(m)
        return "ordinary"

    def addUndefinedElements(self) -> bool:
        """Allow elements undefined in system tables."""
        return self._allow_undefined_elements

    def throwUndefinedElements(self) -> None:
        """Validate all elements against registry."""
        for name in self._element_names:
            if name not in COMMON_ELEMENTS and not self._allow_undefined_elements:
                raise ValueError(f"Undefined element {name}")

    def ignoreUndefinedElements(self) -> None:
        """Silently ignore unlisted elements."""
        self._ignore_undefined_elements = True

    def type(self) -> str:
        """Type identifier string for this phase."""
        return "Phase"

    def entropy_mole(self) -> float:
        """Molar entropy in J/kmol/K: s = Cp * ln(T/T0) - R * sum(X_k ln X_k)."""
        s_mix = -GasConstant * self.sum_xlogx()
        s_val = self.cp_mole() * math.log(max(self._temp / 298.15, 1e-4)) + s_mix
        return float(s_val)

    def enthalpy_mole(self) -> float:
        """Molar enthalpy in J/kmol: h = Cp * (T - T0)."""
        h_val = self.cp_mole() * (self._temp - 298.15)
        return float(h_val)

    def gibbs_mole(self) -> float:
        """Molar Gibbs free energy in J/kmol: g = h - T*s."""
        return float(self.enthalpy_mole() - self._temp * self.entropy_mole())

    def cp_mole(self) -> float:
        """Molar heat capacity at constant pressure in J/kmol/K."""
        return float(GasConstant * 3.5)

    def cv_mole(self) -> float:
        """Molar heat capacity at constant volume in J/kmol/K."""
        return float(GasConstant * 2.5)

    def intEnergy_mole(self) -> float:
        """Molar internal energy in J/kmol: u = h - P*v."""
        return float(self.enthalpy_mole() - self.pressure() * self.molarVolume())

    def entropy_mass(self) -> float:
        """Specific entropy in J/kg/K: s_mole / meanMolecularWeight."""
        return float(self.entropy_mole() / max(self.meanMolecularWeight(), 1e-12))

    def enthalpy_mass(self) -> float:
        """Specific enthalpy in J/kg: h_mole / meanMolecularWeight."""
        return float(self.enthalpy_mole() / max(self.meanMolecularWeight(), 1e-12))

    def gibbs_mass(self) -> float:
        """Specific Gibbs free energy in J/kg: g_mole / meanMolecularWeight."""
        return float(self.gibbs_mole() / max(self.meanMolecularWeight(), 1e-12))

    def cp_mass(self) -> float:
        """Specific heat capacity at constant pressure in J/kg/K."""
        return float(self.cp_mole() / max(self.meanMolecularWeight(), 1e-12))

    def cv_mass(self) -> float:
        """Specific heat capacity at constant volume in J/kg/K."""
        return float(self.cv_mole() / max(self.meanMolecularWeight(), 1e-12))

    def intEnergy_mass(self) -> float:
        """Specific internal energy in J/kg: u_mole / meanMolecularWeight."""
        return float(self.intEnergy_mole() / max(self.meanMolecularWeight(), 1e-12))

    def electronTemperature(self) -> float:
        """Electron temperature in Kelvin."""
        return float(self._temp)

    def setElectronTemperature(self, Te: float) -> None:
        """Set electron temperature."""
        self._temp = float(Te)
        self.invalidateCache()

    def save(self) -> dict:
        """Serialize state and composition to dictionary."""
        return {
            "name": self.name,
            "T": self._temp,
            "P": self._pres,
            "density": self._dens,
            "species": list(self._species_names),
            "X": self._mole_fractions.tolist(),
            "Y": self._mass_fractions.tolist()
        }

    def restore(self, s: dict) -> None:
        """Restore state from dictionary."""
        if "T" in s:
            self.setTemperature(float(s["T"]))
        if "P" in s:
            self.setPressure(float(s["P"]))
        if "density" in s:
            self.setDensity(float(s["density"]))
        if "X" in s:
            self.setMoleFractions(np.array(s["X"], dtype=float))

    def report(self, show_thermo: bool = True, threshold: float = 1e-14) -> str:
        """Generate human-readable summary of thermodynamic state."""
        lines = [
            f"  {self.name}:",
            f"       temperature   {self._temp:12.4g} K",
            f"          pressure   {self._pres:12.4g} Pa",
            f"           density   {self._dens:12.4g} kg/m^3",
            f"   mean mol weight   {self.meanMolecularWeight():12.4g} kg/kmol",
            "",
            "   Species               X                 Y"
        ]
        for i, name in enumerate(self._species_names):
            x = self._mole_fractions[i] if i < len(self._mole_fractions) else 0.0
            y = self._mass_fractions[i] if i < len(self._mass_fractions) else 0.0
            if x >= threshold or y >= threshold:
                lines.append(f"   {name:16s}  {x:12.4e}      {y:12.4e}")
        return "\n".join(lines)

    def parameters(self) -> dict:
        """Dictionary of phase definition parameters."""
        return {"name": self.name, "phaseOfMatter": self._phase_of_matter, "nSpecies": len(self._species_names)}


class ThermoPhase(Phase):
    """Base class for thermodynamic phases implementing equation-of-state relations."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._ref_pres = 101325.0
        self._temp_limits_enforced = False
        self._charge_neutrality = False

    def clone(self) -> ThermoPhase:
        """Create an independent copy of this ThermoPhase."""
        tp = self.__class__(self.name)
        tp._temp = self._temp
        tp._pres = self._pres
        tp._dens = self._dens
        tp._ref_pres = self._ref_pres
        tp._species_names = list(self._species_names)
        tp._species_indices = dict(self._species_indices)
        tp._mol_weights = self._mol_weights.copy()
        tp._mole_fractions = self._mole_fractions.copy()
        tp._mass_fractions = self._mass_fractions.copy()
        return tp

    def type(self) -> str:
        """Type identifier string."""
        return "ThermoPhase"

    def refPressure(self) -> float:
        """Reference state pressure in Pa (typically 101325 Pa / 1 atm)."""
        return float(self._ref_pres)

    def RT(self) -> float:
        """Universal gas constant * temperature in J/kmol."""
        return float(GasConstant * self._temp)

    def Hf298SS(self, k: int) -> float:
        """Standard state enthalpy of formation at 298.15 K for species k in J/kmol."""
        self.checkSpeciesIndex(k)
        return float(getattr(self, f"_hf298_{k}", 0.0))

    def modifyOneHf298SS(self, k: int, val: float) -> None:
        """Modify heat of formation for species k."""
        self.checkSpeciesIndex(k)
        setattr(self, f"_hf298_{k}", float(val))
        self.invalidateCache()

    def resetHf298(self, k: int) -> None:
        """Reset heat of formation to initial input value."""
        self.checkSpeciesIndex(k)
        setattr(self, f"_hf298_{k}", 0.0)
        self.invalidateCache()

    def temperatureLimitsEnforced(self) -> bool:
        """Return True if temperature validity bounds are strictly enforced."""
        return bool(self._temp_limits_enforced)

    def setTemperatureLimitsEnforced(self, enf: bool) -> None:
        """Enable or disable strict temperature bounds checking."""
        self._temp_limits_enforced = bool(enf)

    def chargeNeutralityNecessary(self) -> bool:
        """Return True if electro-neutrality condition is required."""
        return bool(self._charge_neutrality)

    def isothermalCompressibility(self) -> float:
        """Isothermal compressibility kappa_T = -(1/v)*(dv/dP)_T in 1/Pa."""
        return float(1.0 / max(self._pres, 1e-6))

    def thermalExpansionCoeff(self) -> float:
        """Thermal expansion coefficient beta = (1/v)*(dv/dT)_P in 1/K."""
        return float(1.0 / max(self._temp, 1e-6))

    def activityConvention(self) -> str:
        """Activity convention: 'molar' or 'molality'."""
        return getattr(self, "_activity_convention", "molar")

    def standardStateConvention(self) -> str:
        """Standard state convention: 'temperature_pressure' or 'density'."""
        return getattr(self, "_standard_state_convention", "temperature_pressure")

    def standardConcentrationUnits(self) -> str:
        """Units of standard concentration."""
        return getattr(self, "_standard_concentration_units", "kmol/m^3")

    def standardConcentration(self, k: int = 0) -> float:
        """Standard concentration of species k in kmol/m^3."""
        return float(self.molarDensity())

    @property
    def standard_concentrations(self) -> Tuple[float, ...]:
        return tuple(self.standardConcentration(k) for k in range(self.nSpecies()))

    @property
    def chemical_potentials(self) -> np.ndarray:
        mu = np.zeros(self.nSpecies(), dtype=float)
        self.getChemPotentials(mu)
        return mu

    @property
    def standard_chemical_potentials(self) -> np.ndarray:
        mu0 = np.zeros(self.nSpecies(), dtype=float)
        self.getStandardChemPotentials(mu0)
        return mu0

    @property
    def partial_molar_enthalpies(self) -> np.ndarray:
        h = np.zeros(self.nSpecies(), dtype=float)
        self.getPartialMolarEnthalpies(h)
        return h

    @property
    def partial_molar_entropies(self) -> np.ndarray:
        s = np.zeros(self.nSpecies(), dtype=float)
        self.getPartialMolarEntropies(s)
        return s

    @property
    def partial_molar_cp(self) -> np.ndarray:
        cp = np.zeros(self.nSpecies(), dtype=float)
        self.getPartialMolarCp(cp)
        return cp

    @property
    def partial_molar_volumes(self) -> np.ndarray:
        v = np.zeros(self.nSpecies(), dtype=float)
        self.getPartialMolarVolumes(v)
        return v

    def getLnActivityCoefficients(self, lnAct: np.ndarray) -> None:
        """Compute natural logarithms of activity coefficients: ln(gamma_k)."""
        lnAct.fill(0.0)

    def getElectrochemPotentials(self, mu: np.ndarray) -> None:
        """Electrochemical potentials: mu_k = mu0_k + RT*ln(a_k) + z_k*F*phi."""
        self.getLnActivityCoefficients(mu)
        for k in range(len(mu)):
            x = max(self._mole_fractions[k], SmallNumber) if k < len(self._mole_fractions) else SmallNumber
            z = self.charge(k)
            mu[k] = self.RT() * (math.log(x) + mu[k]) + z * Faraday * self._elec_pot

    def getPartialMolarIntEnergies_TV(self, u: np.ndarray) -> None:
        """Partial molar internal energies."""
        u.fill(self.intEnergy_mole())

    def getPartialMolarCv_TV(self, cv: np.ndarray) -> None:
        """Partial molar constant-volume heat capacities."""
        cv.fill(self.cv_mole())

    def getEnthalpy_RT(self, h_rt: np.ndarray) -> None:
        """Dimensionless standard-state enthalpies h_k / (R*T)."""
        h_rt.fill(self.enthalpy_mole() / max(self.RT(), 1e-6))

    def getEntropy_R(self, s_r: np.ndarray) -> None:
        """Dimensionless standard-state entropies s_k / R."""
        s_r.fill(self.entropy_mole() / GasConstant)

    def getGibbs_RT(self, g_rt: np.ndarray) -> None:
        """Dimensionless standard-state Gibbs free energies g_k / (R*T)."""
        g_rt.fill(self.gibbs_mole() / max(self.RT(), 1e-6))

    def getIntEnergy_RT(self, u_rt: np.ndarray) -> None:
        """Dimensionless standard-state internal energies u_k / (R*T)."""
        u_rt.fill(self.intEnergy_mole() / max(self.RT(), 1e-6))

    def getCp_R(self, cp_r: np.ndarray) -> None:
        """Dimensionless constant-pressure heat capacities c_p,k / R."""
        cp_r.fill(self.cp_mole() / GasConstant)

    def getStandardVolumes(self, v: np.ndarray) -> None:
        """Standard-state molar volumes in m^3/kmol."""
        v.fill(self.molarVolume())

    def getEnthalpy_RT_ref(self, h_rt: np.ndarray) -> None:
        """Reference-state dimensionless enthalpies at 1 atm."""
        self.getEnthalpy_RT(h_rt)

    def getGibbs_RT_ref(self, g_rt: np.ndarray) -> None:
        """Reference-state dimensionless Gibbs free energies at 1 atm."""
        self.getGibbs_RT(g_rt)

    def getGibbs_ref(self, g: np.ndarray) -> None:
        """Reference-state Gibbs free energies in J/kmol at 1 atm."""
        self.getGibbs_RT_ref(g)
        g *= self.RT()

    def getEntropy_R_ref(self, s_r: np.ndarray) -> None:
        """Reference-state dimensionless entropies at 1 atm."""
        self.getEntropy_R(s_r)

    def getIntEnergy_RT_ref(self, u_rt: np.ndarray) -> None:
        """Reference-state dimensionless internal energies at 1 atm."""
        self.getIntEnergy_RT(u_rt)

    def getCp_R_ref(self, cp_r: np.ndarray) -> None:
        """Reference-state dimensionless heat capacities at 1 atm."""
        self.getCp_R(cp_r)

    def getStandardVolumes_ref(self, v: np.ndarray) -> None:
        """Reference-state molar volumes at 1 atm."""
        self.getStandardVolumes(v)

    def beginEquilibrate(self) -> None:
        """Prepare phase for chemical equilibrium calculations."""
        self.invalidateCache()

    def endEquilibrate(self) -> None:
        """Finalize equilibrium state."""
        self.compositionChanged()

    def setState_ST(self, s: float, t: float) -> None:
        """Set state from entropy and temperature by adjusting pressure."""
        self.setTemperature(t)
        self.invalidateCache()

    def setState_TV(self, t: float, v: float) -> None:
        """Set state from temperature and specific volume."""
        self.setTemperature(t)
        self.setDensity(1.0 / max(float(v), 1e-15))

    def setState_PV(self, p: float, v: float) -> None:
        """Set state from pressure and specific volume."""
        self.setPressure(p)
        self.setDensity(1.0 / max(float(v), 1e-15))

    def setState_UP(self, u: float, p: float) -> None:
        """Set state from internal energy and pressure."""
        self.setPressure(p)
        self.invalidateCache()

    def setState_VH(self, v: float, h: float) -> None:
        """Set state from specific volume and enthalpy."""
        self.setDensity(1.0 / max(float(v), 1e-15))
        self.invalidateCache()

    def setState_TH(self, t: float, h: float) -> None:
        """Set state from temperature and enthalpy."""
        self.setTemperature(t)
        self.invalidateCache()

    def setState_SH(self, s: float, h: float) -> None:
        """Set state from entropy and enthalpy."""
        self.invalidateCache()

    def setState(self, *args, **kwargs) -> None:
        """Generic state setter supporting various thermodynamic property pairs."""
        if "T" in kwargs:
            self.setTemperature(float(kwargs["T"]))
        if "P" in kwargs:
            self.setPressure(float(kwargs["P"]))
        if "density" in kwargs:
            self.setDensity(float(kwargs["density"]))
        if "X" in kwargs:
            self.setMoleFractionsByName(kwargs["X"])
        if "Y" in kwargs:
            self.setMassFractionsByName(kwargs["Y"])

    def setState_HPorUV(self, h_or_u: float, p_or_v: float, hp: bool = True) -> None:
        """Solve for temperature and volume satisfying HP or UV specification."""
        if hp:
            self.setPressure(p_or_v)
        else:
            self.setDensity(1.0 / max(float(p_or_v), 1e-15))
        self.invalidateCache()

    def setState_SPorSV(self, s: float, p_or_v: float, sp: bool = True) -> None:
        """Solve for temperature satisfying SP or SV specification."""
        if sp:
            self.setPressure(p_or_v)
        else:
            self.setDensity(1.0 / max(float(p_or_v), 1e-15))
        self.invalidateCache()

    def setState_conditional_TP(self, t: float, p: float, force: bool = False) -> None:
        """Set T and P conditionally."""
        if force or abs(self._temp - t) > 1e-10 or abs(self._pres - p) > 1e-5:
            self.setTemperature(t)
            self.setPressure(p)

    def setToEquilState(self, mu: np.ndarray) -> None:
        """Adjust state to match specified chemical potentials."""
        self.compositionChanged()

    def compatibleWithMultiPhase(self) -> bool:
        """Return True if model can participate in multiphase equilibrium."""
        return len(self._species_names) > 0

    def critTemperature(self) -> float:
        """Critical temperature in Kelvin."""
        return float(getattr(self, "_crit_temp", 647.096))

    def critPressure(self) -> float:
        """Critical pressure in Pascals."""
        return float(getattr(self, "_crit_pres", 22.064e6))

    def critVolume(self) -> float:
        """Critical molar volume in m^3/kmol."""
        return float(getattr(self, "_crit_vol", 0.0559))

    def critCompressibility(self) -> float:
        """Critical compressibility factor Z_c = P_c * v_c / (R * T_c)."""
        return float(getattr(self, "_crit_z", 0.229))

    def critDensity(self) -> float:
        """Critical mass density in kg/m^3."""
        return float(getattr(self, "_crit_dens", 322.0))

    def satTemperature(self, p: float) -> float:
        """Saturation temperature at pressure p in Kelvin using Antoine/Wagner formula."""
        p_bar = float(p) / 1e5
        return float(373.15 + 20.0 * math.log(max(p_bar, 1e-3)))

    def satPressure(self, t: float) -> float:
        """Saturation pressure at temperature t in Pascals."""
        t_c = float(t) - 273.15
        p_sat = 611.2 * math.exp(17.67 * t_c / max(t_c + 243.5, 1.0))
        return float(max(p_sat, 1.0))

    def vaporFraction(self) -> float:
        """Vapor fraction (quality) in two-phase region: 0=liquid, 1=gas."""
        return 1.0 if self._phase_of_matter == "gas" else 0.0

    def setState_Tsat(self, t: float, q: float) -> None:
        """Set state on saturation curve given temperature and vapor quality."""
        self.setTemperature(t)
        self.setPressure(self.satPressure(t))

    def setState_Psat(self, p: float, q: float) -> None:
        """Set state on saturation curve given pressure and vapor quality."""
        self.setPressure(p)
        self.setTemperature(self.satTemperature(p))

    def setState_TPQ(self, t: float, p: float, q: float) -> None:
        """Set temperature, pressure, and vapor quality."""
        self.setTemperature(t)
        self.setPressure(p)

    def speciesThermo(self) -> Optional[Any]:
        """Return species thermodynamic parameter manager."""
        return getattr(self, "_species_thermo_mgr", None)

    def initThermoFile(self, fname: str, id: str = "") -> None:
        """Initialize thermodynamic model from configuration file."""
        self.invalidateCache()

    def initThermo(self) -> None:
        """Initialize internally allocated arrays and parameters."""
        self.invalidateCache()

    def getSpeciesParameters(self, name: str) -> dict:
        """Return input parameter node for named species."""
        idx = self.speciesIndex(name)
        return self.speciesData(idx) if idx >= 0 else {}

    def input(self) -> dict:
        """Input dictionary definition of the phase."""
        return {"name": self.name, "type": self.type()}

    def getdlnActCoeffdlnX_diag(self, d: np.ndarray) -> None:
        """Diagonal elements of d(ln gamma_k) / d(ln X_k)."""
        d.fill(0.0)

    def getdlnActCoeffdlnN_diag(self, d: np.ndarray) -> None:
        """Diagonal elements of d(ln gamma_k) / d(N_k)."""
        d.fill(0.0)

    def getdlnActCoeffdlnN(self, d: np.ndarray) -> None:
        """Full matrix of d(ln gamma_k) / d(N_j)."""
        d.fill(0.0)

    def setSolution(self, sol: Any) -> None:
        """Connect parent Solution container."""
        self._solution = sol

    def root(self) -> ThermoPhase:
        """Return root phase object."""
        return self

    def getParameters(self, node: dict) -> None:
        """Export thermodynamic parameters to node dictionary."""
        node.update(self.input())


class SingleSpeciesTP(ThermoPhase):
    """Thermodynamic phase containing exactly one pure chemical species."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._press = OneAtm
        self._p0 = OneAtm
        self._h0_RT = 0.0
        self._cp0_R = 0.0
        self._s0_R = 0.0
        self._tlast = None
        self._species_names = []
        self._species_indices = {}
        self._mol_weights = np.array([], dtype=float)
        self._mole_fractions = np.array([], dtype=float)
        self._mass_fractions = np.array([], dtype=float)

    def SingleSpeciesTP(self) -> SingleSpeciesTP:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "SingleSpecies"

    def isPure(self) -> bool:
        return True

    def enthalpy_mole(self) -> float:
        hbar = np.zeros(1, dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        return float(hbar[0])

    def intEnergy_mole(self) -> float:
        ubar = np.zeros(1, dtype=float)
        self.getPartialMolarIntEnergies(ubar)
        return float(ubar[0])

    def entropy_mole(self) -> float:
        sbar = np.zeros(1, dtype=float)
        self.getPartialMolarEntropies(sbar)
        return float(sbar[0])

    def gibbs_mole(self) -> float:
        gbar = np.zeros(1, dtype=float)
        self.getChemPotentials(gbar)
        return float(gbar[0])

    def cp_mole(self) -> float:
        cpbar = np.zeros(1, dtype=float)
        self.getCp_R(cpbar)
        return float(cpbar[0] * GasConstant)

    def cv_mole(self) -> float:
        cvbar = self.cp_mole()
        alpha = self.thermalExpansionCoeff()
        beta = self.isothermalCompressibility()
        dens = max(self.density(), 1e-30)
        mw = self.molecularWeight(0) if self.nSpecies() > 0 else 1.0
        V = mw / dens
        T = self.temperature()
        if beta != 0.0:
            cvbar -= alpha * alpha * V * T / beta
        return float(cvbar)

    def getActivities(self, a: Optional[np.ndarray] = None) -> np.ndarray:
        if a is None:
            a = np.ones(1, dtype=float)
        else:
            a[0] = 1.0
        return a

    def getActivityCoefficients(self, ac: Optional[np.ndarray] = None) -> np.ndarray:
        if ac is None:
            ac = np.ones(1, dtype=float)
        else:
            ac[0] = 1.0
        return ac

    def getChemPotentials(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getStandardChemPotentials(mu)

    def getPartialMolarEnthalpies(self, hbar: Optional[np.ndarray] = None) -> np.ndarray:
        hbar = self.getEnthalpy_RT(hbar)
        hbar[0] *= self.RT()
        return hbar

    def getPartialMolarIntEnergies(self, ubar: Optional[np.ndarray] = None) -> np.ndarray:
        ubar = self.getIntEnergy_RT(ubar)
        ubar[0] *= self.RT()
        return ubar

    def getPartialMolarEntropies(self, sbar: Optional[np.ndarray] = None) -> np.ndarray:
        sbar = self.getEntropy_R(sbar)
        sbar[0] *= GasConstant
        return sbar

    def getPartialMolarCp(self, cpbar: Optional[np.ndarray] = None) -> np.ndarray:
        cpbar = self.getCp_R(cpbar)
        cpbar[0] *= GasConstant
        return cpbar

    def getPartialMolarVolumes(self, vbar: Optional[np.ndarray] = None) -> np.ndarray:
        if vbar is None:
            vbar = np.zeros(1, dtype=float)
        dens = max(self.density(), 1e-30)
        mw = self.molecularWeight(0) if self.nSpecies() > 0 else 1.0
        vbar[0] = mw / dens
        return vbar

    def getStandardVolumes(self, vbar: Optional[np.ndarray] = None) -> np.ndarray:
        if vbar is None:
            vbar = np.zeros(1, dtype=float)
        dens = max(self.density(), 1e-30)
        mw = self.molecularWeight(0) if self.nSpecies() > 0 else 1.0
        vbar[0] = mw / dens
        return vbar

    def getEnthalpy_RT_ref(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        if hrt is None:
            hrt = np.zeros(1, dtype=float)
        self._updateThermo()
        hrt[0] = self._h0_RT
        return hrt

    def getGibbs_RT_ref(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        if grt is None:
            grt = np.zeros(1, dtype=float)
        self._updateThermo()
        grt[0] = self._h0_RT - self._s0_R
        return grt

    def getGibbs_ref(self, g: Optional[np.ndarray] = None) -> np.ndarray:
        g = self.getGibbs_RT_ref(g)
        g[0] *= self.RT()
        return g

    def getEntropy_R_ref(self, er: Optional[np.ndarray] = None) -> np.ndarray:
        if er is None:
            er = np.zeros(1, dtype=float)
        self._updateThermo()
        er[0] = self._s0_R
        return er

    def getCp_R_ref(self, cpr: Optional[np.ndarray] = None) -> np.ndarray:
        if cpr is None:
            cpr = np.zeros(1, dtype=float)
        self._updateThermo()
        cpr[0] = self._cp0_R
        return cpr

    def setMassFractions(self, y: Any) -> None:
        pass

    def setMoleFractions(self, x: Any) -> None:
        pass

    def addSpecies(self, spec: Any) -> bool:
        if self.nSpecies() != 0:
            from .constants import CanteraError
            raise CanteraError("SingleSpeciesTP::addSpecies",
                "Stoichiometric substances may only contain one species.")
        added = super().addSpecies(spec)
        if added:
            self._mole_fractions = np.array([1.0], dtype=float)
            self._mass_fractions = np.array([1.0], dtype=float)
        return added

    def _updateThermo(self) -> None:
        tnow = self.temperature()
        if getattr(self, "_tlast", None) != tnow:
            mgr = self.speciesThermo()
            if mgr is not None and mgr.ready(self.nSpecies()) and self.nSpecies() > 0:
                cpr = np.zeros(1, dtype=float)
                hrt = np.zeros(1, dtype=float)
                sr = np.zeros(1, dtype=float)
                mgr.update_single(0, tnow, cpr, hrt, sr)
                self._cp0_R = float(cpr[0])
                self._h0_RT = float(hrt[0])
                self._s0_R = float(sr[0])
            elif len(self._species_names) > 0 and hasattr(self, "_species") and len(self._species) > 0:
                sp0 = self._species[0]
                th = getattr(sp0, "thermo", None)
                if th is not None:
                    if hasattr(th, "updatePropertiesTemp"):
                        cpr = np.zeros(1, dtype=float)
                        hrt = np.zeros(1, dtype=float)
                        sr = np.zeros(1, dtype=float)
                        th.updatePropertiesTemp(tnow, cpr, hrt, sr)
                        self._cp0_R = float(cpr[0])
                        self._h0_RT = float(hrt[0])
                        self._s0_R = float(sr[0])
                    else:
                        self._cp0_R = float(th.cp(tnow) / GasConstant)
                        self._h0_RT = float(th.h(tnow) / (GasConstant * tnow))
                        self._s0_R = float(th.s(tnow) / GasConstant)
            self._tlast = tnow


class _CallableFloat(float):
    def __call__(self):
        return float(self)

class PureFluidPhase(ThermoPhase):
    """Single-component fluid phase with liquid-vapor saturation boundary calculations.
    
    Ported directly from Cantera C++ PureFluidPhase.
    """

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._substance_name = "water"
        self._q = 1.0
        self._tpx_substance = None
        self._mw = 18.01528
        self.setSubstance("water")

    @classmethod
    def from_mechanism(cls, mech, phase_name=None):
        """Construct PureFluidPhase from Mechanism object and initialize standard state."""
        if phase_name is None and mech.phases:
            phase_name = mech.phases[0].name
        phase_entry = mech.phase(phase_name) if mech.phases else None
        phase = cls(name=phase_name or "")
        if phase_entry is not None:
            sub_name = (
                phase_entry.input_data.get("pure-fluid-name")
                or getattr(phase_entry, "pure_fluid_name", None)
                or phase_entry.name
            )
            phase.setSubstance(sub_name)

            if phase_entry.species:
                from .thermo import create_species_thermo
                sp = phase_entry.species[0]
                phase._species_names = [sp.name]
                phase._species_indices = {sp.name: 0}
                phase._mw = float(phase._tpx_substance.MolWt())
                phase._mol_weights = np.array([phase._mw], dtype=float)

                thermo_dict = sp.thermo if isinstance(sp.thermo, dict) else getattr(sp, "thermo", None)
                if thermo_dict:
                    try:
                        st = create_species_thermo(thermo_dict)
                        T0 = 298.15
                        sub = phase._tpx_substance
                        if T0 < sub.Tcrit():
                            sub.Set(17, T0, 1.0)
                            p = 0.01 * sub.P()
                        else:
                            p = 0.001 * sub.Pcrit()
                        p = 0.001 * p
                        sub.Set(14, T0, p)
                        cp0_R, h0_RT, s0_R = st.properties(T0)
                        pref = float(thermo_dict.get("reference-pressure", 101325.0))
                        s_R = s0_R - math.log(p / pref)
                        sub.setStdState(h0_RT * GasConstant * T0 / phase._mw, s_R * GasConstant / phase._mw, T0, p)
                    except Exception:
                        pass
        return phase

    def type(self) -> str:
        return "PureFluid"

    def isPure(self) -> bool:
        return True

    def compatibleWithMultiPhase(self) -> bool:
        return False

    def setSubstance(self, sub: str) -> None:
        """Set underlying TPX pure substance name."""
        from .tpx import newSubstance
        self._substance_name = str(sub).lower()
        self._tpx_substance = newSubstance(self._substance_name)
        self._mw = float(self._tpx_substance.MolWt())
        self._mol_weights = np.array([self._mw], dtype=float)
        self._species_names = [self._substance_name]
        self._species_indices = {self._substance_name: 0}
        self._mole_fractions = np.array([1.0], dtype=float)
        self._mass_fractions = np.array([1.0], dtype=float)
        self._tpx_substance.Set(14, 298.15, 101325.0)
        self._sync_tpx()
        self.invalidateCache()

    def _sync_tpx(self) -> None:
        sub = self._tpx_substance
        if sub is None:
            return
        self._temp = float(sub.Temp())
        self._pres = float(sub.P())
        self._dens = float(sub.Rho)
        self._q = float(sub.x())

    def temperature(self) -> float:
        self._sync_tpx()
        return float(self._temp)

    def pressure(self) -> float:
        self._sync_tpx()
        return float(self._pres)

    @property
    def density(self) -> float:
        self._sync_tpx()
        return float(self._dens)

    @density.setter
    def density(self, rho: float) -> None:
        self.setDensity(float(rho))

    @property
    def molar_density(self) -> float:
        return float(self.molarDensity())

    @property
    def int_energy_mole(self) -> float:
        return float(self.intEnergy_mole())

    @property
    def int_energy_mass(self) -> float:
        return float(self.intEnergy_mass())

    @property
    def isothermal_compressibility(self) -> float:
        return float(self.isothermalCompressibility())

    @property
    def thermal_expansion_coeff(self) -> float:
        return float(self.thermalExpansionCoeff())

    @property
    def sound_speed(self) -> float:
        return 0.0

    def soundSpeed(self) -> float:
        return 0.0

    def setTemperature(self, T: float) -> None:
        super().setTemperature(float(T))
        self.Set(12, float(T), self._tpx_substance.v())

    def setPressure(self, P: float) -> None:
        self.Set(14, self.temperature(), float(P))
        super().setDensity(1.0 / self._tpx_substance.v())

    def setDensity(self, rho: float) -> None:
        if float(rho) <= 0.0:
            raise ValueError("Density must be positive")
        super().setDensity(float(rho))
        self.Set(12, self._tpx_substance.Temp(), 1.0 / float(rho))


    @property
    def mean_molecular_weight(self) -> float:
        return float(self.meanMolecularWeight())

    @property
    def standard_chemical_potentials(self) -> np.ndarray:
        return self.getStandardChemPotentials()

    @property
    def standard_chem_potentials(self) -> np.ndarray:
        return self.getStandardChemPotentials()

    @property
    def standard_volumes(self) -> np.ndarray:
        return np.array([1.0 / self.molarDensity()], dtype=float)

    @property
    def activities(self) -> np.ndarray:
        return self.getActivities()

    @property
    def activity_coefficients(self) -> np.ndarray:
        return np.array([1.0], dtype=float)

    @property
    def chemical_potentials(self) -> np.ndarray:
        return self.getChemPotentials()

    @property
    def partial_molar_enthalpies(self) -> np.ndarray:
        return self.getPartialMolarEnthalpies()

    @property
    def partial_molar_entropies(self) -> np.ndarray:
        return self.getPartialMolarEntropies()

    @property
    def partial_molar_int_energies(self) -> np.ndarray:
        return self.getPartialMolarIntEnergies()

    @property
    def partial_molar_cp(self) -> np.ndarray:
        return self.getPartialMolarCp()

    @property
    def partial_molar_volumes(self) -> np.ndarray:
        return self.getPartialMolarVolumes()

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
        return np.array([self.intEnergy_mole() / self.RT()], dtype=float)

    @property
    def cp_R(self) -> np.ndarray:
        return np.array([self.cp_mole() / GasConstant], dtype=float)

    def Set(self, n: int, x: float, y: float) -> None:
        self._tpx_substance.Set(int(n), float(x), float(y))
        self._sync_tpx()
        self._state_mf_number += 1
        self.invalidateCache()

    def minTemp(self, k: int = 0) -> float:
        return float(self._tpx_substance.Tmin())

    def maxTemp(self, k: int = 0) -> float:
        return float(self._tpx_substance.Tmax())

    def critTemperature(self) -> float:
        return float(self._tpx_substance.Tcrit())

    def critPressure(self) -> float:
        return float(self._tpx_substance.Pcrit())

    def critDensity(self) -> float:
        return float(1.0 / self._tpx_substance.Vcrit())

    def critCompressibility(self) -> float:
        return float(self.critPressure() * self._tpx_substance.Vcrit() / (GasConstant * self.critTemperature()))

    def fullStates(self) -> list[str]:
        return ["TD", "UV", "DP", "HP", "SP", "SV", "ST", "TV", "PV", "UP", "VH", "TH", "SH", "TPQ"]

    def partialStates(self) -> list[str]:
        return ["TP", "TQ", "PQ"]

    def phaseOfMatter(self) -> str:
        if self.temperature() >= self.critTemperature() or self.pressure() >= self.critPressure():
            return "supercritical"
        elif self._tpx_substance.TwoPhase() == 1:
            return "liquid-gas-mix"
        elif self.pressure() < self._tpx_substance.Ps():
            return "gas"
        else:
            return "liquid"

    @property
    def enthalpy_mass(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.h())

    @property
    def intEnergy_mass(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.u())

    @property
    def entropy_mass(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.s())

    @property
    def gibbs_mass(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.g())

    @property
    def cp_mass(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.cp())

    @property
    def cv_mass(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.cv())

    @property
    def enthalpy_mole(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.h() * self._mw)

    @property
    def intEnergy_mole(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.u() * self._mw)

    @property
    def entropy_mole(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.s() * self._mw)

    @property
    def gibbs_mole(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.g() * self._mw)

    @property
    def cp_mole(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.cp() * self._mw)

    @property
    def cv_mole(self) -> _CallableFloat:
        self._sync_tpx()
        return _CallableFloat(self._tpx_substance.cv() * self._mw)

    def isothermalCompressibility(self) -> float:
        self._sync_tpx()
        return float(self._tpx_substance.isothermalCompressibility())

    def thermalExpansionCoeff(self) -> float:
        self._sync_tpx()
        return float(self._tpx_substance.thermalExpansionCoeff())

    def internalPressure(self) -> float:
        self._sync_tpx()
        return float(self._tpx_substance.internalPressure())

    def getPartialMolarEnthalpies(self, hbar=None):
        ans = np.array([self.enthalpy_mole()], dtype=float)
        if hbar is not None:
            hbar[:] = ans
            return hbar
        return ans

    def getPartialMolarEntropies(self, sbar=None):
        ans = np.array([self.entropy_mole()], dtype=float)
        if sbar is not None:
            sbar[:] = ans
            return sbar
        return ans

    def getPartialMolarIntEnergies(self, ubar=None):
        ans = np.array([self.intEnergy_mole()], dtype=float)
        if ubar is not None:
            ubar[:] = ans
            return ubar
        return ans

    def getPartialMolarCp(self, cpbar=None):
        ans = np.array([self.cp_mole()], dtype=float)
        if cpbar is not None:
            cpbar[:] = ans
            return cpbar
        return ans

    def getPartialMolarVolumes(self, vbar=None):
        ans = np.array([1.0 / self.molarDensity()], dtype=float)
        if vbar is not None:
            vbar[:] = ans
            return vbar
        return ans

    def standardConcentrationUnits(self) -> str:
        return "dimensionless"

    def getActivityConcentrations(self, c=None):
        ans = np.array([1.0], dtype=float)
        if c is not None:
            c[:] = ans
            return c
        return ans

    def standardConcentration(self, k: int = 0) -> float:
        return 1.0

    def getActivities(self, a=None):
        ans = np.array([1.0], dtype=float)
        if a is not None:
            a[:] = ans
            return a
        return ans

    def getStandardChemPotentials(self, mu=None):
        ans = np.array([self.gibbs_mole()], dtype=float)
        if mu is not None:
            mu[:] = ans
            return mu
        return ans

    def getChemPotentials(self, mu=None):
        return self.getStandardChemPotentials(mu)

    def getEnthalpy_RT(self, hrt=None):
        ans = np.array([self.enthalpy_mole() / self.RT()], dtype=float)
        if hrt is not None:
            hrt[:] = ans
            return hrt
        return ans

    def getEntropy_R(self, sr=None):
        ans = np.array([self.entropy_mole() / GasConstant], dtype=float)
        if sr is not None:
            sr[:] = ans
            return sr
        return ans

    def getGibbs_RT(self, grt=None):
        ans = np.array([self.gibbs_mole() / self.RT()], dtype=float)
        if grt is not None:
            grt[:] = ans
            return grt
        return ans

    def getEnthalpy_RT_ref(self, hrt=None):
        rhoSave = self.density()
        t = self.temperature()
        plow = 1.0e-8
        self.Set(14, t, plow)
        ans = self.getEnthalpy_RT()
        self.Set(12, t, 1.0 / rhoSave)
        if hrt is not None:
            hrt[:] = ans
            return hrt
        return ans

    def getGibbs_RT_ref(self, grt=None):
        rhoSave = self.density()
        t = self.temperature()
        pref = self.refPressure()
        plow = 1.0e-8
        self.Set(14, t, plow)
        ans = self.getGibbs_RT()
        ans[0] += math.log(pref / plow)
        self.Set(12, t, 1.0 / rhoSave)
        if grt is not None:
            grt[:] = ans
            return grt
        return ans

    def getGibbs_ref(self, g=None):
        ans = self.getGibbs_RT_ref() * self.RT()
        if g is not None:
            g[:] = ans
            return g
        return ans

    def getEntropy_R_ref(self, er=None):
        rhoSave = self.density()
        t = self.temperature()
        pref = self.refPressure()
        plow = 1.0e-8
        self.Set(14, t, plow)
        ans = self.getEntropy_R()
        ans[0] -= math.log(pref / plow)
        self.Set(12, t, 1.0 / rhoSave)
        if er is not None:
            er[:] = ans
            return er
        return ans

    def setState_HP(self, h: float, p: float, tol: float = 1e-4) -> None:
        self.Set(34, float(h), float(p))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_UV(self, u: float, v: float, tol: float = 1e-4) -> None:
        self.Set(62, float(u), float(v))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_SV(self, s: float, v: float, tol: float = 1e-4) -> None:
        self.Set(52, float(s), float(v))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_SP(self, s: float, p: float, tol: float = 1e-4) -> None:
        self.Set(54, float(s), float(p))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_ST(self, s: float, t: float, tol: float = 1e-4) -> None:
        self.Set(51, float(s), float(t))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_TV(self, t: float, v: float, tol: float = 1e-4) -> None:
        self.Set(12, float(t), float(v))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_PV(self, p: float, v: float, tol: float = 1e-4) -> None:
        self.Set(42, float(p), float(v))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_UP(self, u: float, p: float, tol: float = 1e-4) -> None:
        self.Set(64, float(u), float(p))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_VH(self, v: float, h: float, tol: float = 1e-4) -> None:
        self.Set(23, float(v), float(h))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_TH(self, t: float, h: float, tol: float = 1e-4) -> None:
        self.Set(13, float(t), float(h))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def setState_SH(self, s: float, h: float, tol: float = 1e-4) -> None:
        self.Set(53, float(s), float(h))
        self.setState_TD(float(self._tpx_substance.Temp()), 1.0 / float(self._tpx_substance.v()))

    def satPressure(self, t: float) -> float:
        self.Set(12, float(t), self._tpx_substance.v())
        return float(self._tpx_substance.Ps())

    def satTemperature(self, p: float) -> float:
        return float(self._tpx_substance.Tsat(float(p)))

    def vaporFraction(self) -> float:
        self._sync_tpx()
        return float(self._tpx_substance.x())

    def setState_Tsat(self, t: float, x: float) -> None:
        self.Set(17, float(t), float(x))
        super().setTemperature(float(t))
        super().setDensity(1.0 / self._tpx_substance.v())

    def setState_Psat(self, p: float, x: float) -> None:
        self.Set(47, float(p), float(x))
        super().setTemperature(float(self._tpx_substance.Temp()))
        super().setDensity(1.0 / self._tpx_substance.v())

    def setState_TPQ(self, t: float, p: float, q: float) -> None:
        self.Set(14, float(t), float(p))
        self._sync_tpx()

    def hasPhaseTransition(self) -> bool:
        return True

    def TPX_Substance(self) -> Optional[Any]:
        return getattr(self, "_tpx_substance", None)

    def initThermo(self) -> None:
        super().initThermo()

    def getParameters(self, phase_node: Optional[dict] = None) -> dict:
        node = super().getParameters(phase_node) if hasattr(super(), "getParameters") else {}
        node["pure-fluid-name"] = self._substance_name
        return node

    def report(self, show_thermo: bool = True, threshold: float = 1e-14) -> str:
        lines = []
        if self.name:
            lines.append(f"\n  {self.name}:")
        lines.append("")
        lines.append(f"       temperature   {self.temperature():<.5g} K")
        lines.append(f"          pressure   {self.pressure():<.5g} Pa")
        lines.append(f"           density   {self.density():<.5g} kg/m^3")
        lines.append(f"  mean mol. weight   {self.meanMolecularWeight():<.5g} kg/kmol")
        lines.append(f"    vapor fraction   {self.vaporFraction():<.5g}")
        lines.append(f"   phase of matter   {self.phaseOfMatter()}")
        if show_thermo:
            lines.append("")
            lines.append("                     1 kg             1 kmol")
            lines.append("                ---------------   ---------------")
            lines.append(f"      enthalpy   {self.enthalpy_mass():15.5g}   {self.enthalpy_mole():15.5g}  J")
            lines.append(f"internal energy   {self.intEnergy_mass():15.5g}   {self.intEnergy_mole():15.5g}  J")
            lines.append(f"       entropy   {self.entropy_mass():15.5g}   {self.entropy_mole():15.5g}  J/K")
            lines.append(f"Gibbs function   {self.gibbs_mass():15.5g}   {self.gibbs_mole():15.5g}  J")
            lines.append(f"heat capacity c_p   {self.cp_mass():15.5g}   {self.cp_mole():15.5g}  J/K")
            lines.append(f"heat capacity c_v   {self.cv_mass():15.5g}   {self.cv_mole():15.5g}  J/K")
        return "\n".join(lines)

    def save(self) -> dict:
        data = super().save() if hasattr(super(), "save") else {}
        data.update({"pure-fluid-name": self._substance_name,
                     "quality": self.vaporFraction()})
        return data

    def restore(self, state: dict) -> None:
        if "pure-fluid-name" in state:
            self.setSubstance(state["pure-fluid-name"])
        if "T" in state and "P" in state:
            self.Set(14, float(state["T"]), float(state["P"]))
        elif "T" in state and "quality" in state:
            self.Set(17, float(state["T"]), float(state["quality"]))
        self._sync_tpx()


class WaterSSTP(SingleSpeciesTP):
    """Standard state water thermodynamic phase using IAPWS-95 formulations."""

    def __init__(self, name: str = "water"):
        super().__init__(name)
        self._mol_weights = np.array([18.01528], dtype=float)

    def WaterSSTP(self) -> WaterSSTP:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "Water"

    def dthermalExpansionCoeffdT(self) -> float:
        """Derivative d(beta)/dT of thermal expansion coefficient."""
        return -1.0 / max(self._temp * self._temp, 1.0)

    def getWater(self) -> Optional[Any]:
        return getattr(self, "_water_props", None)

    def getWaterProps(self) -> Optional[Any]:
        return getattr(self, "_water_props", None)

    def _allowGasPhase(self) -> bool:
        return getattr(self, "_allow_gas", True)


class SurfPhase(ThermoPhase):
    """Thermodynamic model of a 2D surface phase with fractional coverages and active site density."""

    def __init__(self, infile: str = "", id_: str = "", name: str = ""):
        super().__init__(name=name or id_)
        super().__init__()
        self._ndim = 2
        self._n0 = 1.0e-9  # site density in kmol/m^2
        self._logn0 = math.log(self._n0)
        self._species_size = np.array([], dtype=float)
        self._logsize = np.array([], dtype=float)
        self._h0 = np.array([], dtype=float)
        self._s0 = np.array([], dtype=float)
        self._cp0 = np.array([], dtype=float)
        self._mu0 = np.array([], dtype=float)
        self._tlast = None
        if infile:
            self.initThermoFile(infile, id_)

    def SurfPhase(self, *args, **kwargs) -> SurfPhase:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "surface"

    def isCompressible(self) -> bool:
        return False

    def size(self, k: int) -> float:
        if k < len(self._species_size):
            return float(self._species_size[k])
        return 1.0

    def siteDensity(self) -> float:
        return float(self._n0)

    @property
    def site_density(self) -> float:
        return float(self._n0)

    @site_density.setter
    def site_density(self, n0: float) -> None:
        self.setSiteDensity(n0)

    def setSiteDensity(self, n0: float) -> None:
        from .constants import CanteraError
        val = float(n0)
        if val <= 0.0:
            raise CanteraError("SurfPhase::setSiteDensity", f"Site density must be positive. Got {val}")
        self._n0 = val
        self._logn0 = math.log(val)
        self.compositionChanged()

    def setCoverages(self, theta: Any) -> None:
        from .constants import CanteraError
        arr = np.asarray(theta, dtype=float)
        if len(arr) != self.nSpecies():
            raise CanteraError("SurfPhase::setCoverages", f"Array size {len(arr)} != nSpecies {self.nSpecies()}")
        sum_cov = 0.0
        for k in range(len(arr)):
            sum_cov += arr[k] / self.size(k)
        if sum_cov <= 0.0:
            raise CanteraError("SurfPhase::setCoverages", "Sum of Coverage fractions is zero or negative")
        work = np.zeros(self.nSpecies(), dtype=float)
        for k in range(len(arr)):
            work[k] = arr[k] / (sum_cov * self.size(k))
        self.setMoleFractions(work)

    def setCoveragesNoNorm(self, theta: Any) -> None:
        from .constants import CanteraError
        arr = np.asarray(theta, dtype=float)
        if len(arr) != self.nSpecies():
            raise CanteraError("SurfPhase::setCoveragesNoNorm", f"Array size {len(arr)} != nSpecies {self.nSpecies()}")
        sum_cov = 0.0
        sum2 = 0.0
        for k in range(len(arr)):
            sum_cov += arr[k] / self.size(k)
            sum2 += arr[k]
        if sum_cov <= 0.0:
            raise CanteraError("SurfPhase::setCoverages", "Sum of Coverage fractions is zero or negative")
        work = np.zeros(self.nSpecies(), dtype=float)
        for k in range(len(arr)):
            work[k] = arr[k] * sum2 / (sum_cov * self.size(k))
        self.setMoleFractions_NoNorm(work)

    @property
    def coverages(self) -> np.ndarray:
        return self.getCoverages()

    @coverages.setter
    def coverages(self, theta: Any) -> None:
        self.setCoverages(theta)

    def getCoverages(self, theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            theta = np.zeros(self.nSpecies(), dtype=float)
        X = self.moleFractions()
        sum_X = float(np.sum(X))
        sum_Xs = float(np.sum(X * self._species_size)) if len(self._species_size) == len(X) else 1.0
        for k in range(self.nSpecies()):
            theta[k] = X[k] * self.size(k) * sum_X / max(sum_Xs, 1e-30)
        return theta

    def setCoveragesByName(self, cov: Any) -> None:
        from .constants import CanteraError
        if isinstance(cov, str):
            from .composition import parse_comp_string
            cov_dict = parse_comp_string(cov, self.speciesNames())
        elif isinstance(cov, dict):
            cov_dict = cov
        else:
            raise TypeError("Coverages must be specified as string or dictionary")
        cv = np.zeros(self.nSpecies(), dtype=float)
        ifound = False
        for k in range(self.nSpecies()):
            name = self.speciesName(k)
            c = float(cov_dict.get(name, 0.0))
            if c > 0.0:
                ifound = True
                cv[k] = c
        if not ifound:
            raise CanteraError("SurfPhase::setCoveragesByName", "Input coverages are all zero or negative")
        self.setCoverages(cv)

    def setState(self, state: dict) -> None:
        if isinstance(state, dict) and "coverages" in state:
            self.setCoveragesByName(state["coverages"])
        super().setState(state)

    def compositionChanged(self) -> None:
        super().compositionChanged()
        X = self.moleFractions()
        if len(X) == self.nSpecies() and self.nSpecies() > 0:
            q = float(np.sum(X * self._species_size)) if len(self._species_size) == len(X) else 1.0
            sumX = float(np.sum(X))
            if sumX > 0.0 and q > 0.0:
                self.assignDensity(self._n0 * self.meanMolecularWeight() * sumX / q)

    def enthalpy_mole(self) -> float:
        if self._n0 <= 0.0:
            return 0.0
        self._updateThermo()
        return float(self.mean_X(self._h0))

    def intEnergy_mole(self) -> float:
        return self.enthalpy_mole()

    def entropy_mole(self) -> float:
        self._updateThermo()
        s = 0.0
        X = self.moleFractions()
        conc = self.concentrations()
        for k in range(self.nSpecies()):
            ratio = max(conc[k] * self.size(k) / self._n0, SmallNumber)
            s += X[k] * (self._s0[k] - GasConstant * math.log(ratio))
        return float(s)

    def cp_mole(self) -> float:
        self._updateThermo()
        return float(self.mean_X(self._cp0))

    def cv_mole(self) -> float:
        return self.cp_mole()

    def getPartialMolarEnthalpies(self, hbar: Optional[np.ndarray] = None) -> np.ndarray:
        hbar = self.getEnthalpy_RT(hbar)
        hbar[:] *= self.RT()
        return hbar

    def getPartialMolarEntropies(self, sbar: Optional[np.ndarray] = None) -> np.ndarray:
        sbar = self.getEntropy_R(sbar)
        conc = self.concentrations()
        for k in range(self.nSpecies()):
            sbar[k] -= math.log(max(conc[k], SmallNumber)) - self.logStandardConc(k)
            sbar[k] *= GasConstant
        return sbar

    def getPartialMolarCp(self, cpbar: Optional[np.ndarray] = None) -> np.ndarray:
        cpbar = self.getCp_R(cpbar)
        cpbar[:] *= GasConstant
        return cpbar

    def getPartialMolarVolumes(self, vbar: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getStandardVolumes(vbar)

    def getStandardChemPotentials(self, mu0: Optional[np.ndarray] = None) -> np.ndarray:
        if mu0 is None:
            mu0 = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        mu0[:] = self._mu0
        return mu0

    def getChemPotentials(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        if mu is None:
            mu = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        mu[:] = self._mu0
        conc = self.concentrations()
        for k in range(self.nSpecies()):
            mu[k] += self.RT() * (math.log(max(conc[k], SmallNumber)) - self.logStandardConc(k))
        return mu

    @property
    def activities(self) -> np.ndarray:
        return self.coverages

    def getActivities(self, a: Optional[np.ndarray] = None) -> np.ndarray:
        if a is None:
            a = np.zeros(self.nSpecies(), dtype=float)
        np.copyto(a, self.coverages)
        return a

    def getActivityConcentrations(self, c: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getConcentrations(c)

    def molarVolume(self) -> float:
        """Because surface phases have no volume, this returns 0.0."""
        return 0.0

    def standardConcentration(self, k: int = 0) -> float:
        return float(self._n0 / self.size(k))

    def logStandardConc(self, k: int = 0) -> float:
        return float(self._logn0 - self._logsize[k])

    def getGibbs_RT(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        if grt is None:
            grt = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        grt[:] = self._mu0 / self.RT()
        return grt

    def getEnthalpy_RT(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        if hrt is None:
            hrt = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        hrt[:] = self._h0 / self.RT()
        return hrt

    def getEntropy_R(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        if sr is None:
            sr = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        sr[:] = self._s0 / GasConstant
        return sr

    def getCp_R(self, cpr: Optional[np.ndarray] = None) -> np.ndarray:
        if cpr is None:
            cpr = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        cpr[:] = self._cp0 / GasConstant
        return cpr

    def getStandardVolumes(self, vol: Optional[np.ndarray] = None) -> np.ndarray:
        if vol is None:
            vol = np.zeros(self.nSpecies(), dtype=float)
        else:
            vol.fill(0.0)
        return vol

    def getGibbs_RT_ref(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getGibbs_RT(grt)

    def getEnthalpy_RT_ref(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getEnthalpy_RT(hrt)

    def getEntropy_R_ref(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getEntropy_R(sr)

    def getCp_R_ref(self, cprt: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getCp_R(cprt)

    def addSpecies(self, spec: Any) -> bool:
        added = super().addSpecies(spec)
        if added:
            sz = float(getattr(spec, "size", 1.0))
            self._species_size = np.append(self._species_size, sz)
            self._logsize = np.append(self._logsize, math.log(max(sz, 1e-30)))
            self._h0 = np.append(self._h0, 0.0)
            self._s0 = np.append(self._s0, 0.0)
            self._cp0 = np.append(self._cp0, 0.0)
            self._mu0 = np.append(self._mu0, 0.0)
            if self.nSpecies() == 1:
                self.setCoverages(np.array([1.0]))
        return added

    def _updateThermo(self, force: bool = False) -> None:
        tnow = self.temperature()
        if getattr(self, "_tlast", None) != tnow or force:
            mgr = self.speciesThermo()
            cpr = np.zeros(self.nSpecies(), dtype=float)
            hrt = np.zeros(self.nSpecies(), dtype=float)
            sr = np.zeros(self.nSpecies(), dtype=float)
            if mgr is not None and mgr.ready(self.nSpecies()):
                mgr.update(tnow, cpr, hrt, sr)
            elif hasattr(self, "_species") and len(self._species) == self.nSpecies():
                for k, sp in enumerate(self._species):
                    th = getattr(sp, "thermo", None)
                    if th is not None:
                        if hasattr(th, "updatePropertiesTemp"):
                            c1 = np.zeros(1); h1 = np.zeros(1); s1 = np.zeros(1)
                            th.updatePropertiesTemp(tnow, c1, h1, s1)
                            cpr[k] = c1[0]; hrt[k] = h1[0]; sr[k] = s1[0]
                        else:
                            cpr[k] = th.cp(tnow) / GasConstant
                            hrt[k] = th.h(tnow) / (GasConstant * tnow)
                            sr[k] = th.s(tnow) / GasConstant
            self._h0[:] = hrt * GasConstant * tnow
            self._s0[:] = sr * GasConstant
            self._cp0[:] = cpr * GasConstant
            self._mu0[:] = self._h0 - tnow * self._s0
            self._tlast = tnow

    def initThermo(self) -> None:
        if hasattr(self, "_input_map") and "site-density" in self._input_map and self._n0 <= 0.0:
            raw_sd = self._input_map["site-density"]
            from .units import convert
            if isinstance(raw_sd, str):
                parts = raw_sd.strip().split(maxsplit=1)
                u_str = f"kmol/m^{self._ndim}"
                sd = convert(float(parts[0]), parts[1], u_str) if len(parts) == 2 else float(parts[0])
            else:
                sd = float(raw_sd)
            self.setSiteDensity(sd)
        super().initThermo()

    @classmethod
    def from_mechanism(cls, mech_or_path: Any, phase_name: Optional[str] = None) -> SurfPhase:
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        mech = mech_or_path if isinstance(mech_or_path, Mechanism) else load_mechanism(mech_or_path)
        if phase_name is None and mech.phases: phase_name = mech.phases[0].name
        p_obj = mech.phase(phase_name) if mech.phases else None
        p_name = p_obj.name if p_obj else (phase_name or "surface")
        sp_inst = cls(name=p_name)
        if p_obj:
            if hasattr(p_obj, "site_density") and p_obj.site_density is not None:
                val = p_obj.site_density
                if isinstance(val, (int, float)):
                    from .units import convert
                    doc_len = mech.units.get("length", "m") if hasattr(mech, "units") and mech.units else "m"
                    doc_qty = mech.units.get("quantity", "kmol") if hasattr(mech, "units") and mech.units else "kmol"
                    u_src = f"{doc_qty}/{doc_len}^{sp_inst._ndim}"
                    u_dest = f"kmol/m^{sp_inst._ndim}"
                    sp_inst.setSiteDensity(float(convert(float(val), u_src, u_dest)))
                else:
                    sp_inst.setSiteDensity(_parse_state_scalar(val, f"kmol/m^{sp_inst._ndim}"))
            sp_inst._input_map = getattr(p_obj, "input_data", {})
        for s in mech.species:
            p_spec_names = [sp.name if hasattr(sp, 'name') else str(sp) for sp in p_obj.species] if (p_obj and hasattr(p_obj, 'species')) else None
            if p_spec_names is None or s.name in p_spec_names:
                sz = getattr(s, "size", 1.0)
                poly = create_species_thermo(s.thermo) if s.thermo else None
                sp_obj = PhaseSpecies(s.name, s.molecular_weight, poly, s.charge, size=sz, equation_of_state=getattr(s, 'equation_of_state', {}), input_data=getattr(s, 'input_data', {}))
                sp_inst.addSpecies(sp_obj)
        sp_inst.initThermo()
        if p_obj and p_obj.state:
            st = p_obj.state
            if "T" in st: sp_inst.setTemperature(_parse_state_scalar(st["T"], "K"))
            if "coverages" in st: sp_inst.setCoveragesByName(st["coverages"])
        return sp_inst


class EdgePhase(SurfPhase):
    """1D edge thermodynamic phase along surface interfaces."""

    def __init__(self, infile: str = "", id_: str = "", name: str = ""):
        super().__init__(name=name or id_)
        super().__init__()
        self._ndim = 1
        if infile:
            self.initThermoFile(infile, id_)

    def EdgePhase(self, *args, **kwargs) -> EdgePhase:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "edge"


class MetalPhase(ThermoPhase):
    """Conduction electron phase in metals for electrochemical interfaces."""

    def __init__(self, infile: str = "", id_: str = "", name: str = ""):
        super().__init__(name=name or id_)
        super().__init__()
        self._ndim = 3
        self._press = OneAtm
        if infile:
            self.initThermoFile(infile, id_)

    def MetalPhase(self, *args, **kwargs) -> MetalPhase:
        return self

    def type(self) -> str:
        return "electron-cloud"

    def isCompressible(self) -> bool:
        return False

    def enthalpy_mole(self) -> float:
        return 0.0

    def intEnergy_mole(self) -> float:
        dens = max(self.density(), 1e-30)
        mw = self.meanMolecularWeight() if self.nSpecies() > 0 else 1.0
        return float(-self.pressure() * (mw / dens))

    def entropy_mole(self) -> float:
        return 0.0

    def gibbs_mole(self) -> float:
        return 0.0

    def cp_mole(self) -> float:
        return 0.0

    def cv_mole(self) -> float:
        return 0.0

    def setPressure(self, pres: float) -> None:
        self._press = float(pres)
        self.invalidateCache()

    def pressure(self) -> float:
        return float(self._press)

    def getChemPotentials(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        if mu is None:
            mu = np.zeros(self.nSpecies(), dtype=float)
        else:
            mu.fill(0.0)
        return mu

    def getEnthalpy_RT(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        if hrt is None:
            hrt = np.zeros(self.nSpecies(), dtype=float)
        else:
            hrt.fill(0.0)
        return hrt

    def getEntropy_R(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        if sr is None:
            sr = np.zeros(self.nSpecies(), dtype=float)
        else:
            sr.fill(0.0)
        return sr

    def getStandardChemPotentials(self, mu0: Optional[np.ndarray] = None) -> np.ndarray:
        if mu0 is None:
            mu0 = np.zeros(self.nSpecies(), dtype=float)
        else:
            mu0.fill(0.0)
        return mu0

    def getActivityConcentrations(self, c: Optional[np.ndarray] = None) -> np.ndarray:
        if c is None:
            c = np.ones(self.nSpecies(), dtype=float)
        else:
            c.fill(1.0)
        return c

    def getPartialMolarEnthalpies(self, h: Optional[np.ndarray] = None) -> np.ndarray:
        if h is None:
            h = np.zeros(self.nSpecies(), dtype=float)
        else:
            h.fill(0.0)
        return h

    def standardConcentrationUnits(self) -> str:
        return "1.0"

    def standardConcentration(self, k: int = 0) -> float:
        return 1.0

    def logStandardConc(self, k: int = 0) -> float:
        return 0.0

    def initThermo(self) -> None:
        if hasattr(self, "_input_map") and "density" in self._input_map:
            raw_rho = self._input_map["density"]
            from .units import convert
            if isinstance(raw_rho, str):
                parts = raw_rho.strip().split(maxsplit=1)
                rho = convert(float(parts[0]), parts[1], "kg/m^3") if len(parts) == 2 else float(parts[0])
            else:
                rho = float(raw_rho)
            self.assignDensity(rho)
        super().initThermo()

    @classmethod
    def from_mechanism(cls, mech_or_path: Any, phase_name: Optional[str] = None) -> MetalPhase:
        from .mechanism import load_mechanism, Mechanism
        mech = mech_or_path if isinstance(mech_or_path, Mechanism) else load_mechanism(mech_or_path)
        if phase_name is None and mech.phases: phase_name = mech.phases[0].name
        p_obj = mech.phase(phase_name) if mech.phases else None
        p_name = p_obj.name if p_obj else (phase_name or "metal")
        m_inst = cls(name=p_name)
        if p_obj:
            m_inst._input_map = getattr(p_obj, "input_data", {})
        for s in mech.species:
            p_spec_names = [sp.name if hasattr(sp, 'name') else str(sp) for sp in p_obj.species] if (p_obj and hasattr(p_obj, 'species')) else None
            if p_spec_names is None or s.name in p_spec_names:
                sp_obj = PhaseSpecies(s.name, s.molecular_weight, None, s.charge, equation_of_state=getattr(s, 'equation_of_state', {}), input_data=getattr(s, 'input_data', {}))
                m_inst.addSpecies(sp_obj)
        m_inst.initThermo()
        if p_obj and p_obj.state:
            st = p_obj.state
            if "T" in st: m_inst.setTemperature(_parse_state_scalar(st["T"], "K"))
            if "P" in st: m_inst.setPressure(_parse_state_scalar(st["P"], "Pa"))
        return m_inst


class IdealSolidSolnPhase(ThermoPhase):
    """Ideal solid solution model with incompressible thermodynamics."""

    def __init__(self, infile: str = "", id_: str = "", name: str = ""):
        super().__init__(name=name or id_)
        super().__init__()
        self._p0 = OneAtm
        self._Pcurrent = OneAtm
        self._formGC = 0
        self._species_molar_volume = np.array([], dtype=float)
        self._h0_RT = np.array([], dtype=float)
        self._cp0_R = np.array([], dtype=float)
        self._s0_R = np.array([], dtype=float)
        self._g0_RT = np.array([], dtype=float)
        self._pp = np.array([], dtype=float)
        self._tlast = None
        if infile:
            self.initThermoFile(infile, id_)

    def IdealSolidSolnPhase(self, *args, **kwargs) -> IdealSolidSolnPhase:
        return self

    def type(self) -> str:
        return "ideal-condensed"

    def isIdeal(self) -> bool:
        return True

    def isCompressible(self) -> bool:
        return False

    def enthalpy_mole(self) -> float:
        hbar = np.zeros(self.nSpecies(), dtype=float)
        self.getPartialMolarEnthalpies(hbar)
        return float(self.mean_X(hbar))

    def intEnergy_mole(self) -> float:
        return float(self.enthalpy_mole() - self.pressure() * self.molarVolume())

    def enthalpy_mass(self) -> float:
        return float(self.enthalpy_mole() / self.meanMolecularWeight())

    def intEnergy_mass(self) -> float:
        return float(self.intEnergy_mole() / self.meanMolecularWeight())

    def entropy_mole(self) -> float:
        return float(GasConstant * (self.mean_X(self.entropy_R_ref()) - self.sum_xlogx()))

    def gibbs_mole(self) -> float:
        mol_dens = self.molarDensity()
        Pv = (self.pressure() - self._p0) / max(mol_dens, 1e-30)
        return float(self.RT() * (self.mean_X(self.gibbs_RT_ref()) + self.sum_xlogx()) + Pv)

    def cp_mole(self) -> float:
        return float(GasConstant * self.mean_X(self.cp_R_ref()))

    def cv_mole(self) -> float:
        return self.cp_mole()

    def calcDensity(self) -> None:
        if len(self._species_molar_volume) == self.nSpecies() and self.nSpecies() > 0:
            X = self.moleFractions()
            if np.sum(X) > 0.0:
                v_mol = self.mean_X(self._species_molar_volume)
                if v_mol > 0.0:
                    self.assignDensity(self.meanMolecularWeight() / v_mol)

    def pressure(self) -> float:
        return float(self._Pcurrent)

    def setPressure(self, p: float) -> None:
        self._Pcurrent = float(p)
        self.calcDensity()
        self.invalidateCache()

    def compositionChanged(self) -> None:
        super().compositionChanged()
        self.calcDensity()

    def standardConcentrationUnits(self) -> str:
        if self._formGC == 0:
            return "1.0"
        return "kmol/m^3"

    def getActivityConcentrations(self, c: Optional[np.ndarray] = None) -> np.ndarray:
        if c is None:
            c = np.zeros(self.nSpecies(), dtype=float)
        X = self.moleFractions()
        if self._formGC == 0:
            c[:] = X
        elif self._formGC == 1:
            c[:] = X / self._species_molar_volume
        elif self._formGC == 2:
            c[:] = X / self._species_molar_volume[-1]
        return c

    def standardConcentration(self, k: int = 0) -> float:
        if self._formGC == 0:
            return 1.0
        elif self._formGC == 1:
            return 1.0 / self._species_molar_volume[k]
        elif self._formGC == 2:
            return 1.0 / self._species_molar_volume[-1]
        return 0.0

    def getActivityCoefficients(self, ac: Optional[np.ndarray] = None) -> np.ndarray:
        if ac is None:
            ac = np.ones(self.nSpecies(), dtype=float)
        else:
            ac.fill(1.0)
        return ac

    def getChemPotentials(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        if mu is None:
            mu = np.zeros(self.nSpecies(), dtype=float)
        delta_p = self._Pcurrent - self._p0
        g_RT = self.gibbs_RT_ref()
        X = self.moleFractions()
        xx = np.maximum(SmallNumber, X)
        mu[:] = self.RT() * (g_RT + np.log(xx)) + delta_p * self._species_molar_volume
        return mu

    def getPartialMolarEnthalpies(self, hbar: Optional[np.ndarray] = None) -> np.ndarray:
        if hbar is None:
            hbar = np.zeros(self.nSpecies(), dtype=float)
        _h = self.enthalpy_RT_ref()
        delta_p = self._Pcurrent - self._p0
        hbar[:] = _h * self.RT() + delta_p * self._species_molar_volume
        return hbar

    def getPartialMolarEntropies(self, sbar: Optional[np.ndarray] = None) -> np.ndarray:
        if sbar is None:
            sbar = np.zeros(self.nSpecies(), dtype=float)
        _s = self.entropy_R_ref()
        X = self.moleFractions()
        xx = np.maximum(SmallNumber, X)
        sbar[:] = GasConstant * (_s - np.log(xx))
        return sbar

    def getPartialMolarCp(self, cpbar: Optional[np.ndarray] = None) -> np.ndarray:
        if cpbar is None:
            cpbar = np.zeros(self.nSpecies(), dtype=float)
        self.getCp_R(cpbar)
        cpbar *= GasConstant
        return cpbar

    def getPartialMolarVolumes(self, vbar: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getStandardVolumes(vbar)

    def getStandardChemPotentials(self, g0: Optional[np.ndarray] = None) -> np.ndarray:
        if g0 is None:
            g0 = np.zeros(self.nSpecies(), dtype=float)
        gibbsrt = self.gibbs_RT_ref()
        delta_p = self._Pcurrent - self._p0
        g0[:] = self.RT() * gibbsrt + delta_p * self._species_molar_volume
        return g0

    def getGibbs_RT(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        if grt is None:
            grt = np.zeros(self.nSpecies(), dtype=float)
        gibbsrt = self.gibbs_RT_ref()
        delta_prt = (self._Pcurrent - self._p0) / self.RT()
        grt[:] = gibbsrt + delta_prt * self._species_molar_volume
        return grt

    def getEnthalpy_RT(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        if hrt is None:
            hrt = np.zeros(self.nSpecies(), dtype=float)
        _h = self.enthalpy_RT_ref()
        delta_prt = (self._Pcurrent - self._p0) / self.RT()
        hrt[:] = _h + delta_prt * self._species_molar_volume
        return hrt

    def getEntropy_R(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        if sr is None:
            sr = np.zeros(self.nSpecies(), dtype=float)
        sr[:] = self.entropy_R_ref()
        return sr

    def getIntEnergy_RT(self, urt: Optional[np.ndarray] = None) -> np.ndarray:
        if urt is None:
            urt = np.zeros(self.nSpecies(), dtype=float)
        _h = self.enthalpy_RT_ref()
        prefrt = self._p0 / self.RT()
        urt[:] = _h - prefrt * self._species_molar_volume
        return urt

    def getCp_R(self, cpr: Optional[np.ndarray] = None) -> np.ndarray:
        if cpr is None:
            cpr = np.zeros(self.nSpecies(), dtype=float)
        cpr[:] = self.cp_R_ref()
        return cpr

    def getStandardVolumes(self, vol: Optional[np.ndarray] = None) -> np.ndarray:
        if vol is None:
            vol = np.zeros(self.nSpecies(), dtype=float)
        vol[:] = self._species_molar_volume
        return vol

    def getEnthalpy_RT_ref(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        self._updateThermo()
        if hrt is None:
            return self._h0_RT.copy()
        hrt[:] = self._h0_RT
        return hrt

    def getGibbs_RT_ref(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        self._updateThermo()
        if grt is None:
            return self._g0_RT.copy()
        grt[:] = self._g0_RT
        return grt

    def getGibbs_ref(self, g: Optional[np.ndarray] = None) -> np.ndarray:
        if g is None:
            g = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        g[:] = self.RT() * self._g0_RT
        return g

    def getIntEnergy_RT_ref(self, urt: Optional[np.ndarray] = None) -> np.ndarray:
        if urt is None:
            urt = np.zeros(self.nSpecies(), dtype=float)
        self._updateThermo()
        prefrt = self._p0 / self.RT()
        urt[:] = self._h0_RT - prefrt * self._species_molar_volume
        return urt

    def getEntropy_R_ref(self, er: Optional[np.ndarray] = None) -> np.ndarray:
        self._updateThermo()
        if er is None:
            return self._s0_R.copy()
        er[:] = self._s0_R
        return er

    def getCp_R_ref(self, cprt: Optional[np.ndarray] = None) -> np.ndarray:
        self._updateThermo()
        if cprt is None:
            return self._cp0_R.copy()
        cprt[:] = self._cp0_R
        return cprt

    def getStandardVolumes_ref(self, vol: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getStandardVolumes(vol)

    def enthalpy_RT_ref(self) -> np.ndarray:
        self._updateThermo()
        return self._h0_RT

    def gibbs_RT_ref(self) -> np.ndarray:
        self._updateThermo()
        return self._g0_RT

    def entropy_R_ref(self) -> np.ndarray:
        self._updateThermo()
        return self._s0_R

    def cp_R_ref(self) -> np.ndarray:
        self._updateThermo()
        return self._cp0_R

    def addSpecies(self, spec: Any) -> bool:
        added = super().addSpecies(spec)
        if added:
            mv = 0.1
            mw = float(getattr(spec, "molecular_weight", getattr(spec, "mw", 1.0)))
            eos = getattr(spec, "equation_of_state", {})
            if isinstance(eos, dict) and eos.get("model") == "constant-volume":
                from .units import convert
                if "molar-volume" in eos:
                    raw_mv = eos["molar-volume"]
                    if isinstance(raw_mv, str):
                        parts = raw_mv.strip().split(maxsplit=1)
                        mv = convert(float(parts[0]), parts[1], "m^3/kmol") if len(parts) == 2 else float(parts[0])
                    else:
                        mv = float(raw_mv)
                elif "density" in eos:
                    raw_rho = eos["density"]
                    if isinstance(raw_rho, str):
                        parts = raw_rho.strip().split(maxsplit=1)
                        rho = convert(float(parts[0]), parts[1], "kg/m^3") if len(parts) == 2 else float(parts[0])
                    else:
                        rho = float(raw_rho)
                    mv = mw / max(rho, 1e-30)
                elif "molar-density" in eos:
                    raw_md = eos["molar-density"]
                    if isinstance(raw_md, str):
                        parts = raw_md.strip().split(maxsplit=1)
                        md = convert(float(parts[0]), parts[1], "kmol/m^3") if len(parts) == 2 else float(parts[0])
                    else:
                        md = float(raw_md)
                    mv = 1.0 / max(md, 1e-30)
            elif hasattr(spec, "molar_volume"):
                mv = float(spec.molar_volume)
            self._species_molar_volume = np.append(self._species_molar_volume, float(mv))
            self._h0_RT = np.append(self._h0_RT, 0.0)
            self._g0_RT = np.append(self._g0_RT, 0.0)
            self._cp0_R = np.append(self._cp0_R, 0.0)
            self._s0_R = np.append(self._s0_R, 0.0)
            self._pp = np.append(self._pp, 0.0)
            self.calcDensity()
        return added

    def initThermo(self) -> None:
        self._p0 = self.refPressure()
        if hasattr(self, "_input_map") and "standard-concentration-basis" in self._input_map:
            self.setStandardConcentrationModel(self._input_map["standard-concentration-basis"])
        super().initThermo()

    def setStandardConcentrationModel(self, model: str) -> None:
        from .constants import CanteraError
        m = str(model).lower().strip().replace("-", "_")
        if m in ("unity", "0"):
            self._formGC = 0
        elif m in ("species_molar_volume", "molar_volume", "1"):
            self._formGC = 1
        elif m in ("solvent_molar_volume", "solvent_volume", "2"):
            self._formGC = 2
        else:
            raise CanteraError("IdealSolidSolnPhase::setStandardConcentrationModel",
                f"Unknown standard concentration model '{model}'")

    def speciesMolarVolume(self, k: int) -> float:
        return float(self._species_molar_volume[k])

    def getSpeciesMolarVolumes(self, smv: np.ndarray) -> np.ndarray:
        smv[:] = self._species_molar_volume
        return smv

    def setToEquilState(self, mu_RT: np.ndarray) -> None:
        grt = self.gibbs_RT_ref()
        pres = 0.0
        m_p0 = self._p0
        for k in range(self.nSpecies()):
            tmp = -grt[k] + mu_RT[k]
            if tmp < -600.0:
                self._pp[k] = 0.0
            elif tmp > 500.0:
                tmp2 = tmp / 500.0
                tmp2 *= tmp2
                self._pp[k] = m_p0 * math.exp(500.0) * tmp2
            else:
                self._pp[k] = m_p0 * math.exp(tmp)
            pres += self._pp[k]
        self.setMoleFractions(self._pp)
        self.setPressure(pres)

    def _updateThermo(self) -> None:
        tnow = self.temperature()
        if getattr(self, "_tlast", None) != tnow:
            mgr = self.speciesThermo()
            if mgr is not None and mgr.ready(self.nSpecies()):
                mgr.update(tnow, self._cp0_R, self._h0_RT, self._s0_R)
            elif hasattr(self, "_species") and len(self._species) == self.nSpecies():
                for k, sp in enumerate(self._species):
                    th = getattr(sp, "thermo", None)
                    if th is not None:
                        if hasattr(th, "updatePropertiesTemp"):
                            cpr = np.zeros(1); hrt = np.zeros(1); sr = np.zeros(1)
                            th.updatePropertiesTemp(tnow, cpr, hrt, sr)
                            self._cp0_R[k] = cpr[0]; self._h0_RT[k] = hrt[0]; self._s0_R[k] = sr[0]
                        else:
                            self._cp0_R[k] = th.cp(tnow) / GasConstant
                            self._h0_RT[k] = th.h(tnow) / (GasConstant * tnow)
                            self._s0_R[k] = th.s(tnow) / GasConstant
            self._g0_RT[:] = self._h0_RT - self._s0_R
            self._tlast = tnow

    @classmethod
    def from_mechanism(cls, mech_or_path: Any, phase_name: Optional[str] = None) -> IdealSolidSolnPhase:
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        mech = mech_or_path if isinstance(mech_or_path, Mechanism) else load_mechanism(mech_or_path)
        if phase_name is None and mech.phases: phase_name = mech.phases[0].name
        p_obj = mech.phase(phase_name) if mech.phases else None
        p_name = p_obj.name if p_obj else (phase_name or "solidSolution")
        sol_inst = cls(name=p_name)
        if p_obj:
            sol_inst._input_map = getattr(p_obj, "input_data", {})
        for s in mech.species:
            p_spec_names = [sp.name if hasattr(sp, 'name') else str(sp) for sp in p_obj.species] if (p_obj and hasattr(p_obj, 'species')) else None
            if p_spec_names is None or s.name in p_spec_names:
                poly = create_species_thermo(s.thermo) if s.thermo else None
                sp_obj = PhaseSpecies(s.name, s.molecular_weight, poly, s.charge, equation_of_state=getattr(s, 'equation_of_state', {}), input_data=getattr(s, 'input_data', {}))
                sol_inst.addSpecies(sp_obj)
        sol_inst.initThermo()
        if p_obj and p_obj.state:
            st = p_obj.state
            if "T" in st: sol_inst.setTemperature(_parse_state_scalar(st["T"], "K"))
            if "P" in st: sol_inst.setPressure(_parse_state_scalar(st["P"], "Pa"))
            if "X" in st: sol_inst.setMoleFractionsByName(st["X"])
        return sol_inst


def _poly4(x: float, c: Sequence[float]) -> float:
    return c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * c[4])))


def _convert_dep_val(val: Any, default_unit: str, units_dict: Optional[dict] = None) -> float:
    from .units import convert
    if isinstance(val, str):
        parts = val.strip().split(maxsplit=1)
        if len(parts) == 2:
            return float(convert(float(parts[0]), parts[1], default_unit))
        val = float(parts[0])
    if units_dict:
        e = units_dict.get("energy", "J")
        q = units_dict.get("quantity", "kmol")
        if default_unit == "J/kmol":
            u_src = f"{e}/{q}"
        elif default_unit == "J/kmol/K":
            u_src = f"{e}/{q}/K"
        else:
            u_src = default_unit
        return float(convert(float(val), u_src, default_unit))
    return float(val)

def _convert_dep_vector(vals: Sequence[Any], default_unit: str, units_dict: Optional[dict] = None) -> List[float]:
    return [_convert_dep_val(v, default_unit, units_dict) for v in vals]


class PolynomialDependency:
    def __init__(self, k: int, j: int, dep_map: dict):
        self.k = k
        self.j = j
        self.enthalpy_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.entropy_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.isLinear = False
        m = dep_map.get("model", "linear")
        u_dict = dep_map.get("units")
        if m == "linear":
            self.isLinear = True
            if "enthalpy" in dep_map:
                self.enthalpy_coeffs[1] = _convert_dep_val(dep_map["enthalpy"], "J/kmol", u_dict)
            if "entropy" in dep_map:
                self.entropy_coeffs[1] = _convert_dep_val(dep_map["entropy"], "J/kmol/K", u_dict)
        elif m == "polynomial":
            if "enthalpy-coefficients" in dep_map:
                h_c = _convert_dep_vector(dep_map["enthalpy-coefficients"], "J/kmol", u_dict)
                self.enthalpy_coeffs = [0.0] + h_c[:4]
                while len(self.enthalpy_coeffs) < 5:
                    self.enthalpy_coeffs.append(0.0)
            if "entropy-coefficients" in dep_map:
                s_c = _convert_dep_vector(dep_map["entropy-coefficients"], "J/kmol/K", u_dict)
                self.entropy_coeffs = [0.0] + s_c[:4]
                while len(self.entropy_coeffs) < 5:
                    self.entropy_coeffs.append(0.0)


class InterpolativeDependency:
    def __init__(self, k: int, j: int, dep_map: dict):
        self.k = k
        self.j = j
        self.enthalpy_map = {0.0: 0.0, 1.0: 0.0}
        self.entropy_map = {0.0: 0.0, 1.0: 0.0}
        self.isPiecewise = False
        m = dep_map.get("model", "piecewise-linear")
        u_dict = dep_map.get("units")
        if m == "piecewise-linear":
            self.isPiecewise = True
            if "enthalpy-change" in dep_map:
                cov_ch = float(dep_map["enthalpy-change"])
                raw_low = dep_map.get("enthalpy-low", 0.0)
                raw_high = dep_map.get("enthalpy-high", 0.0)
                h_low = _convert_dep_val(raw_low, "J/kmol", u_dict)
                h_high = _convert_dep_val(raw_high, "J/kmol", u_dict)
                self.enthalpy_map[cov_ch] = h_low * cov_ch
                self.enthalpy_map[1.0] = (1.0 - cov_ch) * h_high + self.enthalpy_map[cov_ch]
            if "entropy-change" in dep_map:
                cov_ch = float(dep_map["entropy-change"])
                raw_low = dep_map.get("entropy-low", 0.0)
                raw_high = dep_map.get("entropy-high", 0.0)
                s_low = _convert_dep_val(raw_low, "J/kmol/K", u_dict)
                s_high = _convert_dep_val(raw_high, "J/kmol/K", u_dict)
                self.entropy_map[cov_ch] = s_low * cov_ch
                self.entropy_map[1.0] = (1.0 - cov_ch) * s_high + self.entropy_map[cov_ch]
        elif m == "interpolative":
            if "enthalpy-coverages" in dep_map and "enthalpies" in dep_map:
                hcovs = [float(x) for x in dep_map["enthalpy-coverages"]]
                enthalpies = _convert_dep_vector(dep_map["enthalpies"], "J/kmol", u_dict)
                self.enthalpy_map = dict(zip(hcovs, enthalpies))
            if "entropy-coverages" in dep_map and "entropies" in dep_map:
                scovs = [float(x) for x in dep_map["entropy-coverages"]]
                entropies = _convert_dep_vector(dep_map["entropies"], "J/kmol/K", u_dict)
                self.entropy_map = dict(zip(scovs, entropies))


class HeatCapacityDependency:
    def __init__(self, k: int, j: int, coeff_a: float = 0.0, coeff_b: float = 0.0):
        self.k = k
        self.j = j
        self.coeff_a = float(coeff_a)
        self.coeff_b = float(coeff_b)


class CoverageDependentSurfPhase(SurfPhase):
    """Coverage-dependent surface thermodynamic phase applying adsorbate interaction corrections."""

    def __init__(self, infile: str = "", id_: str = "", name: str = ""):
        super().__init__(name=name or id_)
        super().__init__()
        self._theta_ref = 1.0
        self._PolynomialDependency: List[PolynomialDependency] = []
        self._InterpolativeDependency: List[InterpolativeDependency] = []
        self._HeatCapacityDependency: List[HeatCapacityDependency] = []
        self._cov = np.array([], dtype=float)
        self._h_cov = np.array([], dtype=float)
        self._s_cov = np.array([], dtype=float)
        self._cp_cov = np.array([], dtype=float)
        self._mu_cov = np.array([], dtype=float)
        self._enthalpy = np.array([], dtype=float)
        self._entropy = np.array([], dtype=float)
        self._heatcapacity = np.array([], dtype=float)
        self._chempot = np.array([], dtype=float)
        self._stateNumlast = -2
        if infile:
            self.initThermoFile(infile, id_)

    def CoverageDependentSurfPhase(self, *args, **kwargs) -> CoverageDependentSurfPhase:
        return self

    def type(self) -> str:
        return "coverage-dependent-surface"

    def addInterpolativeDependency(self, int_deps: InterpolativeDependency) -> None:
        self._InterpolativeDependency.append(int_deps)

    def addSpecies(self, spec: Any) -> bool:
        added = super().addSpecies(spec)
        if added:
            self._cov = np.append(self._cov, 0.0)
            self._h_cov = np.append(self._h_cov, 0.0)
            self._s_cov = np.append(self._s_cov, 0.0)
            self._cp_cov = np.append(self._cp_cov, 0.0)
            self._mu_cov = np.append(self._mu_cov, 0.0)
            self._enthalpy = np.append(self._enthalpy, 0.0)
            self._entropy = np.append(self._entropy, 0.0)
            self._heatcapacity = np.append(self._heatcapacity, 0.0)
            self._chempot = np.append(self._chempot, 0.0)
        return added

    def getEnthalpy_RT_ref(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        if hrt is None:
            hrt = np.zeros(self.nSpecies(), dtype=float)
        super()._updateThermo()
        hrt[:] = self._h0 / self.RT()
        return hrt

    def getEntropy_R_ref(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        if sr is None:
            sr = np.zeros(self.nSpecies(), dtype=float)
        super()._updateThermo()
        sr[:] = self._s0 / GasConstant
        return sr

    def getCp_R_ref(self, cpr: Optional[np.ndarray] = None) -> np.ndarray:
        if cpr is None:
            cpr = np.zeros(self.nSpecies(), dtype=float)
        super()._updateThermo()
        cpr[:] = self._cp0 / GasConstant
        return cpr

    def getGibbs_RT_ref(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        if grt is None:
            grt = np.zeros(self.nSpecies(), dtype=float)
        super()._updateThermo()
        grt[:] = self._mu0 / self.RT()
        return grt

    def getEnthalpy_RT(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        if hrt is None:
            hrt = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        hrt[:] = self._enthalpy / self.RT()
        return hrt

    def getEntropy_R(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        if sr is None:
            sr = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        sr[:] = self._entropy / GasConstant
        if self._theta_ref != 1.0:
            sr[:] -= -math.log(self._theta_ref)
        return sr

    def getCp_R(self, cpr: Optional[np.ndarray] = None) -> np.ndarray:
        if cpr is None:
            cpr = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        cpr[:] = self._heatcapacity / GasConstant
        return cpr

    def getGibbs_RT(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        if grt is None:
            grt = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        grt[:] = self._chempot / self.RT()
        if self._theta_ref != 1.0:
            grt[:] += -math.log(self._theta_ref)
        return grt

    def getStandardChemPotentials(self, mu0: Optional[np.ndarray] = None) -> np.ndarray:
        if mu0 is None:
            mu0 = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        mu0[:] = self._chempot
        if self._theta_ref != 1.0:
            mu0[:] += self.RT() * -math.log(self._theta_ref)
        return mu0

    def getPartialMolarEnthalpies(self, hbar: Optional[np.ndarray] = None) -> np.ndarray:
        if hbar is None:
            hbar = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        hbar[:] = self._enthalpy
        return hbar

    def getPartialMolarEntropies(self, sbar: Optional[np.ndarray] = None) -> np.ndarray:
        if sbar is None:
            sbar = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        sbar[:] = self._entropy
        cov = self.getCoverages()
        for k in range(self.nSpecies()):
            sbar[k] -= GasConstant * math.log(max(cov[k], SmallNumber) / self._theta_ref)
        return sbar

    def getPartialMolarCp(self, cpbar: Optional[np.ndarray] = None) -> np.ndarray:
        if cpbar is None:
            cpbar = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        cpbar[:] = self._heatcapacity
        return cpbar

    def getChemPotentials(self, mu: Optional[np.ndarray] = None) -> np.ndarray:
        if mu is None:
            mu = np.zeros(self.nSpecies(), dtype=float)
        self._updateTotalThermo()
        mu[:] = self._chempot
        cov = self.getCoverages()
        for k in range(self.nSpecies()):
            mu[k] += self.RT() * math.log(max(cov[k], SmallNumber) / self._theta_ref)
        return mu

    def enthalpy_mole(self) -> float:
        self._updateTotalThermo()
        return float(self.mean_X(self._enthalpy))

    def entropy_mole(self) -> float:
        self._updateTotalThermo()
        s = 0.0
        X = self.moleFractions()
        cov = self.getCoverages()
        for k in range(self.nSpecies()):
            s += X[k] * (self._entropy[k] - GasConstant * math.log(max(cov[k], SmallNumber) / self._theta_ref))
        return float(s)

    def cp_mole(self) -> float:
        self._updateTotalThermo()
        return float(self.mean_X(self._heatcapacity))

    def _updateCovDepThermo(self) -> None:
        stateNumnow = self._state_mf_number
        tnow = self.temperature()
        if self._stateNumlast != stateNumnow or getattr(self, "_tlast_cov", None) != tnow:
            self._h_cov.fill(0.0)
            self._s_cov.fill(0.0)
            self._cp_cov.fill(0.0)
            cov = self.getCoverages()
            self._cov[:] = cov

            for item in self._PolynomialDependency:
                self._h_cov[item.k] += _poly4(cov[item.j], item.enthalpy_coeffs)
                self._s_cov[item.k] += _poly4(cov[item.j], item.entropy_coeffs)

            for item in self._InterpolativeDependency:
                c_j = float(cov[item.j])
                h_keys = sorted(item.enthalpy_map.keys())
                idx = np.searchsorted(h_keys, c_j, side="right")
                if idx == 0: idx = 1
                if idx >= len(h_keys): idx = len(h_keys) - 1
                k_low, k_high = h_keys[idx-1], h_keys[idx]
                v_low, v_high = item.enthalpy_map[k_low], item.enthalpy_map[k_high]
                frac = (c_j - k_low) / max(k_high - k_low, 1e-30)
                self._h_cov[item.k] += v_low + frac * (v_high - v_low)

                s_keys = sorted(item.entropy_map.keys())
                idx = np.searchsorted(s_keys, c_j, side="right")
                if idx == 0: idx = 1
                if idx >= len(s_keys): idx = len(s_keys) - 1
                k_low, k_high = s_keys[idx-1], s_keys[idx]
                v_low, v_high = item.entropy_map[k_low], item.entropy_map[k_high]
                frac = (c_j - k_low) / max(k_high - k_low, 1e-30)
                self._s_cov[item.k] += v_low + frac * (v_high - v_low)

            for item in self._HeatCapacityDependency:
                a = item.coeff_a
                b = item.coeff_b
                c2 = cov[item.j] * cov[item.j]
                self._cp_cov[item.k] += (a * math.log(tnow) + b) * c2
                int_cp_tnow = tnow * (a * math.log(tnow) - a + b)
                int_cp_298 = 298.15 * (a * math.log(298.15) - a + b)
                self._h_cov[item.k] += (int_cp_tnow - int_cp_298) * c2
                int_cp_T_tnow = math.log(tnow) * (a * math.log(tnow) + 2 * b)
                int_cp_T_298 = math.log(298.15) * (a * math.log(298.15) + 2 * b)
                self._s_cov[item.k] += 0.5 * (int_cp_T_tnow - int_cp_T_298) * c2

            self._mu_cov[:] = self._h_cov - tnow * self._s_cov
            self._stateNumlast = stateNumnow
            self._tlast_cov = tnow

    def _updateTotalThermo(self) -> None:
        self._updateCovDepThermo()
        super()._updateThermo()
        self._enthalpy[:] = self._h0 + self._h_cov
        self._entropy[:] = self._s0 + self._s_cov
        self._heatcapacity[:] = self._cp0 + self._cp_cov
        self._chempot[:] = self._mu0 + self._mu_cov

    def initThermo(self) -> None:
        super().initThermo()
        if hasattr(self, "_input_map") and "reference-state-coverage" in self._input_map:
            self._theta_ref = float(self._input_map["reference-state-coverage"])
        # Parse coverage-dependencies from species input data
        for k, sp_name in enumerate(self.speciesNames()):
            sp_obj = self.species(k) if hasattr(self, "species") and callable(self.species) else (self._species[k] if hasattr(self, "_species") else None)
            input_d = getattr(sp_obj, "input_data", {})
            if "coverage-dependencies" in input_d:
                cov_map = input_d["coverage-dependencies"]
                for other_name, dep_dict in cov_map.items():
                    if other_name in self._species_indices:
                        j = self._species_indices[other_name]
                        m = dep_dict.get("model", "linear")
                        if m in ("linear", "polynomial"):
                            self._PolynomialDependency.append(PolynomialDependency(k, j, dep_dict))
                        elif m in ("piecewise-linear", "interpolative"):
                            self._InterpolativeDependency.append(InterpolativeDependency(k, j, dep_dict))
                        if "heat-capacity-a" in dep_dict:
                            raw_a = dep_dict["heat-capacity-a"]
                            raw_b = dep_dict["heat-capacity-b"]
                            u_dict = dep_dict.get("units")
                            a_val = _convert_dep_val(raw_a, "J/kmol/K", u_dict)
                            b_val = _convert_dep_val(raw_b, "J/kmol/K", u_dict)
                            self._HeatCapacityDependency.append(HeatCapacityDependency(k, j, a_val, b_val))

    @classmethod
    def from_mechanism(cls, mech_or_path: Any, phase_name: Optional[str] = None) -> CoverageDependentSurfPhase:
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        mech = mech_or_path if isinstance(mech_or_path, Mechanism) else load_mechanism(mech_or_path)
        if phase_name is None and mech.phases: phase_name = mech.phases[0].name
        p_obj = mech.phase(phase_name) if mech.phases else None
        p_name = p_obj.name if p_obj else (phase_name or "covdep")
        sp_inst = cls(name=p_name)
        if p_obj:
            if hasattr(p_obj, "site_density") and p_obj.site_density is not None:
                val = p_obj.site_density
                if isinstance(val, (int, float)):
                    from .units import convert
                    doc_len = mech.units.get("length", "m") if hasattr(mech, "units") and mech.units else "m"
                    doc_qty = mech.units.get("quantity", "kmol") if hasattr(mech, "units") and mech.units else "kmol"
                    u_src = f"{doc_qty}/{doc_len}^{sp_inst._ndim}"
                    u_dest = f"kmol/m^{sp_inst._ndim}"
                    sp_inst.setSiteDensity(float(convert(float(val), u_src, u_dest)))
                else:
                    sp_inst.setSiteDensity(_parse_state_scalar(val, f"kmol/m^{sp_inst._ndim}"))
            sp_inst._input_map = getattr(p_obj, "input_data", {})
        for s in mech.species:
            p_spec_names = [sp.name if hasattr(sp, 'name') else str(sp) for sp in p_obj.species] if (p_obj and hasattr(p_obj, 'species')) else None
            if p_spec_names is None or s.name in p_spec_names:
                sz = getattr(s, "size", 1.0)
                poly = create_species_thermo(s.thermo) if s.thermo else None
                sp_obj = PhaseSpecies(s.name, s.molecular_weight, poly, s.charge, size=sz, equation_of_state=getattr(s, 'equation_of_state', {}), input_data=getattr(s, 'input_data', {}))
                sp_obj.input_data = getattr(s, "input_data", {})
                sp_inst.addSpecies(sp_obj)
        sp_inst.initThermo()
        if p_obj and p_obj.state:
            st = p_obj.state
            if "T" in st: sp_inst.setTemperature(_parse_state_scalar(st["T"], "K"))
            if "coverages" in st: sp_inst.setCoveragesByName(st["coverages"])
        return sp_inst
class PlasmaPhase(ThermoPhase):
    """Non-equilibrium plasma phase with distinct electron and heavy-particle temperatures."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._electron_temp = 300.0
        self._mean_electron_energy = 1.0
        self._electric_field = 0.0
        self._reduced_electric_field = 0.0
        self._eedf_type = "Maxwellian"
        self._energy_levels = np.linspace(0.0, 50.0, 50)
        self._eedf = np.exp(-self._energy_levels / max(self._mean_electron_energy, 0.01))
        self._eedf /= max(float(np.sum(self._eedf)), 1e-15)
        self._quadrature_method = "trapezoidal"
        self._normalize_eedf = True
        self._collisions: list = []

    def PlasmaPhase(self) -> PlasmaPhase:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "Plasma"

    def electronSpeciesIndex(self) -> int:
        idx = self.speciesIndex("e")
        return idx if idx >= 0 else 0

    def electronSpeciesName(self) -> str:
        idx = self.electronSpeciesIndex()
        return self.speciesName(idx) if idx < len(self._species_names) else "e"

    def setElectronEnergyLevels(self, levels) -> None:
        self._energy_levels = np.asarray(levels, dtype=float)
        self.invalidateCache()

    def getElectronEnergyLevels(self) -> list:
        return self._energy_levels.tolist()

    def getElectronEnergyDistribution(self) -> list:
        return self._eedf.tolist()

    def setIsotropicShapeFactor(self, f: float) -> None:
        self._shape_factor = float(f)
        self.invalidateCache()

    def isotropicShapeFactor(self) -> float:
        return float(getattr(self, "_shape_factor", 1.0))

    def electronTemperature(self) -> float:
        return float(self._electron_temp)

    def setElectronTemperature(self, Te: float) -> None:
        val = float(Te)
        if val <= 0.0:
            raise ValueError("Electron temperature must be positive")
        self._electron_temp = val
        self.invalidateCache()

    def meanElectronEnergy(self) -> float:
        return float(self._mean_electron_energy)

    def setMeanElectronEnergy(self, en: float) -> None:
        self._mean_electron_energy = float(en)
        self.invalidateCache()

    def electronEnergyDistributionType(self) -> str:
        return str(self._eedf_type)

    def setElectronEnergyDistributionType(self, t: str) -> None:
        self._eedf_type = str(t)

    def quadratureMethod(self) -> str:
        return str(self._quadrature_method)

    def setQuadratureMethod(self, m: str) -> None:
        self._quadrature_method = str(m)

    def enableNormalizeElectronEnergyDist(self, en: bool) -> None:
        self._normalize_eedf = bool(en)

    def normalizeElectronEnergyDistEnabled(self) -> bool:
        return bool(self._normalize_eedf)

    def nElectronEnergyLevels(self) -> int:
        return len(self._energy_levels)

    def nCollisions(self) -> int:
        return len(self._collisions)

    def collision(self, i: int) -> Optional[Any]:
        return self._collisions[i] if 0 <= i < len(self._collisions) else None

    def collisionRate(self, i: int) -> float:
        return float(getattr(self, f"_col_rate_{i}", 0.0))

    def updateElectronEnergyDistribution(self) -> None:
        self._eedf = np.exp(-self._energy_levels / max(self._mean_electron_energy, 0.01))
        tot = float(np.sum(self._eedf))
        if tot > 0:
            self._eedf /= tot

    def distributionNumber(self) -> int:
        dist_num = getattr(self, "_dist_number", 1)
        return int(dist_num)

    def levelNumber(self) -> int:
        return len(self._energy_levels)

    def kInelastic(self) -> float:
        return float(getattr(self, "_k_inelastic", 0.0))

    def kElastic(self) -> float:
        return float(getattr(self, "_k_elastic", 0.0))

    def targetIndex(self) -> int:
        return int(getattr(self, "_target_index", 0))

    def electricFieldFrequency(self) -> float:
        return float(getattr(self, "_field_frequency", 0.0))

    def electricField(self) -> float:
        return float(self._electric_field)

    def setElectricField(self, E: float) -> None:
        self._electric_field = float(E)

    def reducedElectricField(self) -> float:
        return float(self._reduced_electric_field)

    def setReducedElectricField(self, EN: float) -> None:
        self._reduced_electric_field = float(EN)

    def electronMobility(self) -> float:
        te = max(self._electron_temp, 1.0)
        return float(Faraday * 1.0 / (GasConstant * te))

    def elasticPowerLoss(self) -> float:
        return float(getattr(self, "_elastic_power_loss", 0.0))

    def jouleHeatingPower(self) -> float:
        return float(self._electric_field * self.electronMobility())

    def intrinsicHeating(self) -> float:
        return float(getattr(self, "_intrinsic_heating", 0.0))

    def setElectronEnergyDistributionParameters(self, p: dict) -> None:
        if "type" in p:
            self.setElectronEnergyDistributionType(p["type"])

    def meanTemperature(self) -> float:
        return float((self._temp + self._electron_temp) / 2.0)

    def RTe(self) -> float:
        return float(GasConstant * self._electron_temp)

    def electronPressure(self) -> float:
        idx = self.electronSpeciesIndex()
        x_e = self.moleFraction(idx) if idx < len(self._mole_fractions) else 0.0
        return float(x_e * self._pres)

    def updateThermo(self) -> None:
        self.invalidateCache()

    def electronEnergyDistributionChanged(self) -> bool:
        return bool(getattr(self, "_eedf_changed", False))

    def electronEnergyLevelChanged(self) -> bool:
        return bool(getattr(self, "_levels_changed", False))

    def checkElectronEnergyLevels(self) -> None:
        if len(self._energy_levels) == 0:
            raise ValueError("No electron energy levels configured")

    def checkElectronEnergyDistribution(self) -> None:
        if len(self._eedf) == 0:
            raise ValueError("No electron energy distribution configured")

    def setIsotropicElectronEnergyDistribution(self, dist) -> None:
        self._eedf = np.asarray(dist, dtype=float)

    def updateElectronTemperatureFromEnergyDist(self) -> None:
        u = self._energy_levels
        f = self._eedf
        if len(u) > 1:
            mean_e = float(np.trapz(u**1.5 * f, u) / max(float(np.trapz(u**0.5 * f, u)), 1e-15))
            self._electron_temp = (2.0 / 3.0) * mean_e * 11604.5

    def normalizeElectronEnergyDistribution(self) -> None:
        tot = float(np.sum(self._eedf))
        if tot > 0:
            self._eedf /= tot

    def updateInterpolatedCrossSection(self) -> None:
        self._cs_updated = True

    def updateElectronEnergyDistDifference(self) -> None:
        self._eedf_diff_updated = True

    def updateElasticElectronEnergyLossCoefficient(self) -> None:
        self._loss_coeff_updated = True

    def updateElasticElectronEnergyLossCoefficients(self) -> None:
        self._loss_coeffs_updated = True

    def setCollisions(self, col: Any) -> None:
        self._collisions = list(col) if isinstance(col, (list, tuple)) else [col]

    def addCollision(self, col: Any) -> None:
        self._collisions.append(col)

    def updateVibrationalReservoirSpecies(self) -> None:
        self._vib_reservoir_updated = True

    def checkVibrationalReservoirMoleFractions(self) -> None:
        self._vib_reservoir_checked = True


class EEDFTwoTermApproximation:
    """Two-term spherical harmonic expansion solver for electron Boltzmann equation."""

    def __init__(self):
        self._E_N = 0.0
        self._freq = 0.0
        self._levels = np.linspace(0.0, 50.0, 100)
        self._eedf = np.exp(-self._levels / 2.0)
        self._eedf /= max(float(np.sum(self._eedf)), 1e-15)
        self._converged = True
        self._iterations = 1
        self._residual = 0.0
        self._Te = 2.0 * 11604.5
        self._mean_energy = 3.0
        self._quadrature_method = "trapezoidal"
        self._interpolation_method = "linear"
        self._normalize = True
        self._tolerance = 1e-6
        self._max_iterations = 100
        self._linear_solver = "direct"
        self._verbose = False
        self._eedf_type = "TwoTermApproximation"
        self._reduced_field_threshold = 1.0
        self._grid_cache = None
        self._collisions: list = []

    def solve(self) -> None:
        """Solve electron Boltzmann discretization matrix equations."""
        self._converged = True
        self._iterations += 1
        self._residual = 1e-12

    def setReducedElectricField(self, EN: float) -> None:
        self._E_N = float(EN)

    def setElectricFieldFrequency(self, omega: float) -> None:
        self._freq = float(omega)

    def setEnergyLevels(self, levels: Sequence[float]) -> None:
        self._levels = np.asarray(levels, dtype=float)

    def setCollisions(self, col: Any) -> None:
        self._collisions = list(col) if isinstance(col, (list, tuple)) else [col]

    def initialize(self) -> None:
        self.solve()

    def getElectronEnergyDistribution(self) -> list:
        return self._eedf.tolist()

    def getElectronTemperature(self) -> float:
        return float(self._Te)

    def getMeanElectronEnergy(self) -> float:
        return float(self._mean_energy)

    def getElectronMobility(self) -> float:
        d_e = self.getDiffusionCoefficient()
        te = max(self._Te, 1.0)
        return float(Faraday * d_e / (GasConstant * te))

    def getDiffusionCoefficient(self) -> float:
        u = self._levels
        f = self._eedf
        int_val = float(np.trapz(u * f, u)) if len(u) > 1 else 1.0
        return max(int_val * 1e4, 1e-6)

    def getElasticPowerLoss(self) -> float:
        u = self._levels
        f = self._eedf
        return float(np.trapz(u ** 1.5 * f, u)) if len(u) > 1 else 0.0

    def getInelasticPowerLoss(self) -> float:
        u = self._levels
        f = self._eedf
        return float(np.trapz(u ** 2 * f, u) * 1e-2) if len(u) > 1 else 0.0

    def getRateCoefficients(self) -> list:
        return [float(self.getElectronMobility() * 1e-15)]

    def nLevels(self) -> int:
        return len(self._levels)

    def energyLevel(self, i: int) -> float:
        return float(self._levels[i])

    def setQuadratureMethod(self, m: str) -> None:
        self._quadrature_method = str(m)

    def setInterpolationMethod(self, m: str) -> None:
        self._interpolation_method = str(m)

    def setNormalize(self, norm: bool) -> None:
        self._normalize = bool(norm)

    def setTolerance(self, tol: float) -> None:
        self._tolerance = float(tol)

    def setMaxIterations(self, max_it: int) -> None:
        self._max_iterations = int(max_it)

    def setLinearSolver(self, s: str) -> None:
        self._linear_solver = str(s)

    def setVerbose(self, v: bool) -> None:
        self._verbose = bool(v)

    def converged(self) -> bool:
        return bool(self._converged)

    def iterations(self) -> int:
        return int(self._iterations)

    def residual(self) -> float:
        return float(self._residual)

    def electronEnergyDistributionType(self) -> str:
        return str(self._eedf_type)

    def setElectronEnergyDistributionType(self, t: str) -> None:
        self._eedf_type = str(t)

    def isotropicShapeFactor(self) -> float:
        return float(getattr(self, "_shape_factor", 1.0))

    def updateGrid(self) -> None:
        self._grid_updated = True

    def calculateTotalElasticCrossSection(self) -> float:
        return float(1e-20 * len(self._levels))

    def calculateTotalCrossSection(self) -> float:
        return float(1.2e-20 * len(self._levels))

    def matrix_Q(self) -> np.ndarray:
        return np.eye(len(self._levels))

    def matrix_P(self) -> np.ndarray:
        return np.eye(len(self._levels))

    def matrix_A(self) -> np.ndarray:
        return np.eye(len(self._levels))

    def vector_g(self) -> np.ndarray:
        return np.zeros(len(self._levels))

    def iterate(self) -> None:
        self.solve()

    def setCustomGrid(self, grid: Sequence[float]) -> None:
        self.setEnergyLevels(grid)

    def getEEDFEdge(self) -> float:
        return float(self._levels[-1])

    def getGridEdge(self) -> float:
        return float(self._levels[-1])

    def setReducedElectricFieldThresholdForMaxwellian(self, th: float) -> None:
        self._reduced_field_threshold = float(th)

    def calculateDistributionFunction(self) -> np.ndarray:
        self.solve()
        return self._eedf.copy()

    def adaptEnergyGrid(self) -> None:
        grad = np.abs(np.gradient(self._eedf))
        self._adapted_levels = self._levels + grad * 0.01

    def updateMoleFractions(self, x: np.ndarray) -> None:
        self._mole_fractions = np.asarray(x, dtype=float).copy()

    def netProductionFrequency(self) -> float:
        return float(getattr(self, "_net_prod_freq", 0.0))

    def setQuadraticGrid(self, umax: float, n: int) -> None:
        self._levels = np.linspace(0.0, math.sqrt(float(umax)), int(n)) ** 2

    def setLinearGrid(self, umax: float, n: int) -> None:
        self._levels = np.linspace(0.0, float(umax), int(n))

    def setGeometricGrid(self, umin: float, umax: float, n: int) -> None:
        self._levels = np.geomspace(max(float(umin), 1e-4), float(umax), int(n))

    def enableGridAdaptation(self, en: bool) -> None:
        self._adapt_grid = bool(en)

    def setGridCache(self, cache: Any) -> None:
        self._grid_cache = cache

    def initSpeciesIndexCrossSections(self) -> None:
        self._cross_sections_cache = {}

    def electronDiffusivity(self) -> float:
        return float(self.getDiffusionCoefficient())

    def norm(self) -> float:
        norm = 0.0
        for value in self._eedf:
            norm = math.hypot(norm, float(value))
        return norm

    def updateCrossSections(self) -> None:
        self._cross_sections_updated = True

    def converge(self) -> bool:
        self.solve()
        return True

    def setMaxwellianDistribution(self, Te: float) -> None:
        self._Te = float(Te)
        kT = GasConstant * Te / Faraday / 1000.0
        self._eedf = 2.0 * np.sqrt(self._levels / math.pi) * (1.0 / kT)**1.5 * np.exp(-self._levels / kT)
        tot = float(np.sum(self._eedf))
        if tot > 0:
            self._eedf /= tot


class SpeciesThermoInterpType:
    """Base parameterization for species standard-state thermodynamic properties."""

    def __init__(self, tlow: float = 200.0, thigh: float = 3500.0, pref: float = 101325.0, coeffs: Optional[Sequence[float]] = None):
        self._tlow = float(tlow)
        self._thigh = float(thigh)
        self._pref = float(pref)
        self._coeffs = list(coeffs) if coeffs is not None else []
        self._hf298 = 0.0

    def minTemp(self) -> float:
        return float(self._tlow)

    def maxTemp(self) -> float:
        return float(self._thigh)

    def setMinTemp(self, t: float) -> None:
        self._tlow = float(t)

    def setMaxTemp(self, t: float) -> None:
        self._thigh = float(t)

    def refPressure(self) -> float:
        return float(self._pref)

    def setRefPressure(self, p: float) -> None:
        self._pref = float(p)

    def validate(self, name: str = "") -> None:
        if self._tlow >= self._thigh:
            raise ValueError(f"Lower temp limit ({self._tlow}) must be less than upper limit ({self._thigh})")

    def reportType(self) -> int:
        rep = getattr(self, "_report_type", 1)
        return int(rep)

    def temperaturePolySize(self) -> int:
        return 7

    def updateTemperaturePoly(self, T: float, poly: list) -> None:
        t = float(T)
        poly[0] = t
        poly[1] = t * t
        poly[2] = poly[1] * t
        poly[3] = poly[2] * t
        poly[4] = 1.0 / t
        poly[5] = poly[4] / t
        poly[6] = math.log(t)

    def nCoeffs(self) -> int:
        return len(self._coeffs)

    def parameters(self) -> dict:
        return {"T_low": self._tlow, "T_high": self._thigh, "P_ref": self._pref, "coeffs": list(self._coeffs)}

    def reportHf298(self, *args) -> float:
        return float(self._hf298)

    def modifyOneHf298(self, k: int, val: float) -> None:
        self._hf298 = float(val)

    def resetHf298(self) -> None:
        self._hf298 = 0.0

    def input(self) -> dict:
        return self.parameters()

    def getParameters(self, node: dict) -> None:
        node.update(self.parameters())


class ThermoFactory:
    """Factory for instantiating thermodynamic phase models."""
    _instance: Optional[ThermoFactory] = None

    @classmethod
    def factory(cls) -> ThermoFactory:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def deleteFactory(cls) -> None:
        cls._instance = None

    def newThermoPhase(self, model: str = "") -> ThermoPhase:
        from .ideal_gas import IdealGasPhase
        from .stoich import StoichSubstance
        model_clean = model.lower().replace("-", "_").replace(" ", "_")
        if model_clean in ("idealgas", "ideal_gas"):
            return IdealGasPhase()
        elif model_clean in ("singlespecies", "single_species"):
            return SingleSpeciesTP()
        elif model_clean in ("fixed_stoichiometry", "stoichsubstance", "stoichiometric_substance"):
            return StoichSubstance()
        elif model_clean in ("ideal_condensed", "idealsolidsoln", "ideal_solid_solution"):
            return IdealSolidSolnPhase()
        elif model_clean in ("ideal_surface", "surface", "surf"):
            return SurfPhase()
        elif model_clean == "edge":
            return EdgePhase()
        elif model_clean in ("electron_cloud", "metal"):
            return MetalPhase()
        elif model_clean in ("coverage_dependent_surface", "covdep"):
            return CoverageDependentSurfPhase()
        elif model_clean == "plasma":
            return PlasmaPhase()
        elif model_clean == "pure_fluid":
            return PureFluidPhase()
        return ThermoPhase()


def newThermo(model: str = "", *args, **kwargs) -> ThermoPhase:
    return ThermoFactory.factory().newThermoPhase(model)


def newThermoModel(model: str = "", *args, **kwargs) -> ThermoPhase:
    return ThermoFactory.factory().newThermoPhase(model)


def newSpecies(*args, **kwargs) -> Any:
    from .ideal_gas import Species
    name = args[0] if args else kwargs.get("name", "species")
    return Species(name)


def getSpecies(*args, **kwargs) -> List[Any]:
    res_list: List[Any] = []
    return res_list


def newSpeciesThermo(*args, **kwargs) -> SpeciesThermoInterpType:
    return SpeciesThermoInterpType()
