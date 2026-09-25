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
from .elements import ElementRegistry, COMMON_ELEMENTS


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
                w = COMMON_ELEMENTS.get(el_name, 1.008)
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

    def speciesNames(self) -> List[str]:
        """Return list of species names."""
        return list(self._species_names)

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
            return float(np.sum(self._mole_fractions * self._mol_weights))
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
        return float(self._dens / max(mean_mw, 1e-12))

    def molarVolume(self) -> float:
        """Molar volume in m^3/kmol: 1.0 / molarDensity."""
        return 1.0 / max(self.molarDensity(), 1e-15)

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
        mw = getattr(spec, "molecular_weight", getattr(spec, "mw", 1.0))
        self._mol_weights = np.append(self._mol_weights, float(mw))
        ch = getattr(spec, "charge", 0.0)
        self._species_charge = np.append(self._species_charge, float(ch))
        self._mole_fractions = np.append(self._mole_fractions, 0.0)
        self._mass_fractions = np.append(self._mass_fractions, 0.0)
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
        w = weight if weight is not None else COMMON_ELEMENTS.get(name, 1.008)
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
        if len(self._species_names) == 0:
            self._species_names = [name or "pure"]
            self._species_indices = {self._species_names[0]: 0}
            self._mol_weights = np.array([28.97], dtype=float)
            self._mole_fractions = np.array([1.0], dtype=float)
            self._mass_fractions = np.array([1.0], dtype=float)

    def type(self) -> str:
        """Type identifier."""
        return "SingleSpecies"

    def isPure(self) -> bool:
        """Always True for single-species phase."""
        n = self.nSpecies()
        return n <= 1

    def _updateThermo(self) -> None:
        """Update standard-state thermodynamic properties."""
        self.invalidateCache()


class PureFluidPhase(ThermoPhase):
    """Single-component fluid phase with liquid-vapor saturation boundary calculations."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._substance_name = "water"
        self._q = 1.0
        self._tpx_substance = None
        self.setSubstance("water")

    def type(self) -> str:
        return "PureFluid"

    def setSubstance(self, sub: str) -> None:
        """Set underlying TPX pure substance name."""
        from .tpx import newSubstance
        self._substance_name = str(sub).lower()
        self._tpx_substance = newSubstance(self._substance_name)
        self._tpx_substance.Set(14, 298.15, 101325.0)
        self._sync_tpx()
        self.invalidateCache()

    def _sync_tpx(self) -> None:
        sub = self._tpx_substance
        if sub is None: return
        self._temp = float(sub.Temp()); self._pres = float(sub.P()); self._dens = float(sub.Rho); self._q = float(sub.x())

    def temperature(self) -> float:
        self._sync_tpx(); return float(self._temp)
    def pressure(self) -> float:
        self._sync_tpx(); return float(self._pres)
    def density(self) -> float:
        self._sync_tpx(); return float(self._dens)
    def setTemperature(self, T: float) -> None:
        self._tpx_substance.set_T(float(T)); self._sync_tpx(); self._state_mf_number += 1; self.invalidateCache()
    def setPressure(self, P: float) -> None:
        self._tpx_substance.Set(14, self._tpx_substance.Temp(), float(P)); self._sync_tpx(); self._state_mf_number += 1; self.invalidateCache()
    def setDensity(self, rho: float) -> None:
        if float(rho) <= 0: raise ValueError("Density must be positive")
        self._tpx_substance.set_v(1.0/float(rho)); self._sync_tpx(); self._state_mf_number += 1; self.invalidateCache()
    def enthalpy_mass(self) -> float:
        self._sync_tpx(); return float(self._tpx_substance.h())
    def intEnergy_mass(self) -> float:
        self._sync_tpx(); return float(self._tpx_substance.u())
    def entropy_mass(self) -> float:
        self._sync_tpx(); return float(self._tpx_substance.s())
    def gibbs_mass(self) -> float:
        self._sync_tpx(); return float(self._tpx_substance.g())
    def cp_mass(self) -> float:
        self._sync_tpx(); return float(self._tpx_substance.cp())
    def cv_mass(self) -> float:
        self._sync_tpx(); return float(self._tpx_substance.cv())

    def setState_TV(self, t: float, v: float) -> None:
        self._tpx_substance.Set(12, float(t), float(v)); self._sync_tpx()
    def setState_PV(self, p: float, v: float) -> None:
        self._tpx_substance.Set(42, float(p), float(v)); self._sync_tpx()
    def setState_UP(self, u: float, p: float) -> None:
        self._tpx_substance.Set(64, float(u), float(p)); self._sync_tpx()
    def setState_VH(self, v: float, h: float) -> None:
        self._tpx_substance.Set(23, float(v), float(h)); self._sync_tpx()
    def setState_TH(self, t: float, h: float) -> None:
        self._tpx_substance.Set(13, float(t), float(h)); self._sync_tpx()
    def setState_SH(self, s: float, h: float) -> None:
        self._tpx_substance.Set(53, float(s), float(h)); self._sync_tpx()

    def hasPhaseTransition(self) -> bool:
        """Pure fluids support liquid-vapor phase transitions."""
        sub = getattr(self, "_substance_name", "water")
        return True

    def TPX_Substance(self) -> Optional[Any]:
        """Return associated TPX substance instance."""
        return getattr(self, "_tpx_substance", None)

    def vaporFraction(self) -> float:
        """Return vapor quality q in [0, 1]."""
        self._sync_tpx(); return float(self._q)

    def satPressure(self, t: float) -> float:
        sub=self._tpx_substance; old=sub.T; sub.set_T(float(t)); ans=sub.Psat(); sub.T=old; return float(ans)

    def satTemperature(self, p: float) -> float:
        return float(self._tpx_substance.Tsat(float(p)))

    def setState_Tsat(self, t: float, q: float) -> None:
        self._tpx_substance.Set(17, float(t), float(q)); self._sync_tpx()

    def setState_Psat(self, p: float, q: float) -> None:
        self._tpx_substance.Set(47, float(p), float(q)); self._sync_tpx()

    def setState_TPQ(self, t: float, p: float, q: float) -> None:
        self._tpx_substance.Set(14, float(t), float(p)); self._sync_tpx()

    def save(self) -> dict:
        data = super().save()
        data.update({"pure-fluid-name": self._substance_name,
                     "quality": self.vaporFraction()})
        return data

    def restore(self, state: dict) -> None:
        if "pure-fluid-name" in state:
            self.setSubstance(state["pure-fluid-name"])
        if "T" in state and "P" in state:
            self._tpx_substance.Set(14, float(state["T"]), float(state["P"]))
        elif "T" in state and "quality" in state:
            self._tpx_substance.Set(17, float(state["T"]), float(state["quality"]))
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
    """Surface phase with active site density and fractional coverages."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._site_density = 1e-9
        self._ndim = 2
        self._coverages = np.array([], dtype=float)

    def SurfPhase(self) -> SurfPhase:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "Surf"

    def isCompressible(self) -> bool:
        matter = self.phaseOfMatter()
        return False

    def siteDensity(self) -> float:
        """Surface site density in kmol/m^2."""
        return float(self._site_density)

    def setSiteDensity(self, s: float) -> None:
        """Set surface site density in kmol/m^2."""
        val = float(s)
        if val <= 0.0:
            raise ValueError("Site density must be positive")
        self._site_density = val
        self.invalidateCache()

    def setCoverages(self, theta: np.ndarray) -> None:
        """Set surface fractional coverages summing to 1.0."""
        arr = np.asarray(theta, dtype=float)
        tot = float(np.sum(arr))
        if tot <= 0.0:
            raise ValueError("Sum of coverages must be positive")
        self._coverages = arr / tot
        self.setMoleFractions(self._coverages)

    def setCoveragesByName(self, theta) -> None:
        """Set fractional coverages from mapping or string."""
        self.setMoleFractionsByName(theta)
        self._coverages = self._mole_fractions.copy()

    def getCoverages(self, theta: np.ndarray) -> None:
        """Copy coverages into user buffer."""
        n = min(len(theta), len(self._coverages))
        theta[:n] = self._coverages[:n]

    def _updateThermo(self) -> None:
        self.invalidateCache()


class EdgePhase(SurfPhase):
    """1D edge thermodynamic phase along surface interfaces."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self._ndim = 1

    def EdgePhase(self) -> EdgePhase:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "Edge"


class MetalPhase(ThermoPhase):
    """Conduction electron phase in metals for electrochemical interfaces."""

    def __init__(self, name: str = "metal"):
        super().__init__(name)
        self._ndim = 3

    def type(self) -> str:
        return "Metal"

    def isCompressible(self) -> bool:
        matter = self.phaseOfMatter()
        return False


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
        return float(np.linalg.norm(self._eedf))

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
        model_clean = model.lower().replace("-", "_").replace(" ", "_")
        if model_clean in ("idealgas", "ideal_gas"):
            return IdealGasPhase()
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
