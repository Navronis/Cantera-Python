"""Multiphase state container and incomplete VCS API scaffolding.

Cantera's VCS active-set equilibrium algorithm is not ported. The module keeps
the multiphase container and a single-phase convenience path, while rejecting
true multiphase equilibrium instead of returning heuristic phase amounts.
"""
from __future__ import annotations

import numpy as np
import math
import operator

from .constants import Faraday, GasConstant, OneBar
from .equilibrium import EquilibriumError
from .numerics import DenseMatrix, _python_l2_norm

class VCSError(EquilibriumError):
    """Exception raised by the VCS multiphase solver."""
    pass


VCS_SPECIES_MINOR = 0
VCS_SPECIES_MAJOR = 1
VCS_SPECIES_COMPONENT = 2
VCS_SPECIES_ZEROEDMS = -2
VCS_SPECIES_ZEROEDSS = -3
VCS_SPECIES_DELETED = -4
VCS_SPECIES_INTERFACIALVOLTAGE = -5
VCS_SPECIES_ZEROEDPHASE = -6
VCS_RELDELETE_SPECIES_CUTOFF = 1.0e-64
VCS_SPECIES_ACTIVEBUTZERO = -7
VCS_SPECIES_STOICHZERO = -8
VCS_DELETE_MINORSPECIES_CUTOFF = 1.0e-140
VCS_SMALL_MULTIPHASE_SPECIES = 1.0e-25
VCS_DELETE_PHASE_CUTOFF = 1.0e-13
VCS_SUCCESS = 0


class MultiPhase:
    def IntEnergy(self) -> float:
        self.updatePhases()
        return sum(p.intEnergy_mole * m for p, m in zip(self._phases, self._phase_moles)) if self._phases else 0.0

    def addPhase(self, phase, moles: float = 1.0):
        self.add_phase(phase, moles)

    def calcElemAbundances(self) -> np.ndarray:
        abundances = np.zeros(self.nElements(), dtype=float)
        for ip, phase in enumerate(self._phases):
            x = self._phase_mole_fractions(ip)
            for k, name in enumerate(phase.species_names):
                amount = self._phase_moles[ip] * x[k]
                for e, element in enumerate(self.element_names):
                    abundances[e] += amount * self._n_atoms(phase, k, name, element)
        return abundances

    @staticmethod
    def _n_atoms(phase, species_index: int, species_name: str, element: str) -> float:
        method = getattr(phase, "n_atoms", None)
        if callable(method):
            return float(method(species_name, element))
        mechanism = getattr(phase, "_mechanism", None)
        if mechanism is not None:
            for species in mechanism.species:
                if species.name == species_name:
                    return float(species.composition.get(element, 0.0))
        if callable(getattr(phase, "species", None)):
            species = phase.species(species_index)
        else:
            species = phase.species[species_index]
        composition = getattr(species, "composition", {})
        return float(composition.get(element, 0.0))

    def checkElementIndex(self, m: int) -> None:
        if m < 0 or m >= self.nElements():
            raise IndexError("Element index out of range")

    def _check_phase_index(self, p: int) -> int:
        index = int(p)
        if index < 0 or index >= self.n_phases:
            raise IndexError(f"Phase index {index} is out of range")
        return index

    def elementMoles(self, m: int) -> float:
        self.checkElementIndex(m)
        return float(self.calcElemAbundances()[m])

    def enthalpy(self) -> float:
        self.updatePhases()
        return sum(getattr(p, 'enthalpy_mole', 0.0) * m for p, m in zip(self._phases, self._phase_moles))

    def entropy(self) -> float:
        self.updatePhases()
        return sum(getattr(p, 'entropy_mole', 0.0) * m for p, m in zip(self._phases, self._phase_moles))

    def getElemAbundances(self, elem: np.ndarray) -> None:
        values = np.asarray(elem)
        abundances = self.calcElemAbundances()
        if values.ndim != 1 or values.size < abundances.size:
            raise ValueError("element abundance output array is too small")
        values[:abundances.size] = abundances

    def getMoles(self, moles: np.ndarray) -> None:
        values = np.asarray(moles)
        species_moles = self._species_moles()
        if values.ndim != 1 or values.size < len(species_moles):
            raise ValueError("species mole output array is too small")
        values[:len(species_moles)] = species_moles

    def gibbs(self) -> float:
        return sum(getattr(p, 'gibbs_mole', 0.0) * m for p, m in zip(self._phases, self._phase_moles))

    def init(self) -> None:
        if self._initialized:
            return
        if any(not getattr(phase, "species_names", None) for phase in self._phases):
            raise VCSError("Every phase must contain at least one species")
        self._mole_fractions = [np.asarray(phase.X, dtype=float).copy()
                                for phase in self._phases]
        self._initialized = True
        self.uploadMoleFractionsFromPhases()
        self.updatePhases()

    @property
    def element_names(self) -> tuple[str, ...]:
        """Element names in first-seen phase order, matching MultiPhase."""
        names = []
        for phase in self._phases:
            # MultiPhase takes element order from each ThermoPhase's indexed
            # element API. YAML declaration order can differ from the phase's
            # canonical order (for example, after automatic element discovery).
            for element in phase.element_names:
                if element not in names:
                    names.append(element)
        return tuple(names)

    @property
    def n_species(self) -> int:
        return sum(len(phase.species_names) for phase in self._phases)

    def nSpecies(self) -> int:
        return self.n_species

    def _species_moles(self) -> np.ndarray:
        values = []
        for phase_index, phase_moles in enumerate(self._phase_moles):
            values.extend(phase_moles * self._phase_mole_fractions(phase_index))
        return np.asarray(values, dtype=float)

    def _phase_mole_fractions(self, phase_index: int) -> np.ndarray:
        if self._initialized:
            return self._mole_fractions[phase_index]
        return np.asarray(self._phases[phase_index].X, dtype=float)

    def _species_location(self, k: int) -> tuple[int, int]:
        index = int(k)
        if index < 0 or index >= self.n_species:
            raise IndexError(f"Species index {index} is out of range")
        for ip, phase in enumerate(self._phases):
            count = len(phase.species_names)
            if index < count:
                return ip, index
            index -= count
        raise IndexError(f"Species index {k} is out of range")

    def moleFraction(self, k: int) -> float:
        ip, local = self._species_location(k)
        return float(self._phase_mole_fractions(ip)[local])

    def nElements(self) -> int:
        return len(self.element_names)

    def nPhases(self) -> int:
        return len(self._phases)

    def phaseCharge(self, p: int) -> float:
        p = self._check_phase_index(p)
        phase = self._phases[p]
        x = self._phase_mole_fractions(p)
        nsp = phase.n_species if hasattr(phase, "n_species") else phase.nSpecies()
        if hasattr(phase, "charge") and callable(phase.charge):
            charge = sum(float(phase.charge(k)) * x[k] for k in range(nsp))
        else:
            species_get = phase.species if callable(getattr(phase, "species", None)) else None
            species_list = phase.species if not callable(getattr(phase, "species", None)) else getattr(phase, "_species", None)
            charge = sum(float(getattr(species_get(k) if species_get else species_list[k], "charge", 0.0)) * x[k]
                         for k in range(nsp))
        return float(Faraday * charge * self._phase_moles[p])

    def phaseIndex(self, name: str) -> int:
        for i, p in enumerate(self._phases):
            if getattr(p, 'name', '') == name:
                return i
        raise KeyError(f"Phase {name!r} not found")

    def elementIndex(self, name: str) -> int:
        try:
            return self.element_names.index(name)
        except ValueError as exc:
            raise KeyError(f"Element {name!r} not found") from exc

    def phaseMoles(self, p: int) -> float:
        return self._phase_moles[self._check_phase_index(p)]

    def phaseName(self, p: int) -> str:
        phase = self._phases[self._check_phase_index(p)]
        return getattr(phase, 'name', f'phase_{p}')

    @property
    def phase_names(self) -> tuple[str, ...]:
        return tuple(self.phaseName(i) for i in range(self.n_phases))

    def setMoles(self, moles: np.ndarray) -> None:
        self.init()
        values = np.asarray(moles, dtype=float)
        if values.ndim != 1 or values.size < self.n_species:
            raise ValueError("species mole input array is too small")
        if not np.all(np.isfinite(values[:self.n_species])):
            raise ValueError("species mole amounts must be finite")
        loc = 0
        for ip, phase in enumerate(self._phases):
            count = len(phase.species_names)
            amounts = values[loc:loc + count]
            total = float(np.sum(amounts))
            self._phase_moles[ip] = total
            if count > 1 and total > 1e-300:
                self._mole_fractions[ip] = amounts / total
            elif count == 1:
                self._mole_fractions[ip] = np.ones(1, dtype=float)
            loc += count
        self.updatePhases()

    def setMolesByName(self, name: str, moles: float) -> None:
        idx = self.phaseIndex(name)
        self._phase_moles[idx] = float(moles)

    def setPhaseMoleFractions(self, p: int, x: np.ndarray) -> None:
        self.init()
        p = self._check_phase_index(p)
        phase = self._phases[p]
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.size < len(phase.species_names):
            raise ValueError("phase mole-fraction array is too small")
        phase.TPX = (self._temperature, self._pressure,
                     values[:len(phase.species_names)])
        self._mole_fractions[p] = np.asarray(phase.X, dtype=float).copy()

    def setPhaseMoles(self, p: int, moles: float) -> None:
        self.set_phase_moles(p, moles)

    def setTemperature(self, T: float) -> None:
        self.set_TP(T, self.P)

    def solutionSpecies(self, k: int) -> bool:
        ip, _ = self._species_location(k)
        return len(self._phases[ip].species_names) > 1

    def speciesMoles(self, k: int) -> float:
        self._species_location(k)
        return float(self._species_moles()[int(k)])

    def speciesName(self, k: int) -> str:
        ip, local = self._species_location(k)
        return self._phases[ip].species_names[local]

    def speciesPhaseIndex(self, k: int) -> int:
        return self._species_location(k)[0]

    def tempOK(self, p: int | None = None) -> bool:
        indices = range(len(self._phases)) if p is None else (int(p),)
        for ip in indices:
            phase = self._phases[ip]
            if len(phase.species_names) > 1:
                if not (phase.min_temp <= self.T <= phase.max_temp):
                    return False
        return True

    def updatePhases(self) -> None:
        if self._phases:
            self.init()
            for ip, phase in enumerate(self._phases):
                phase.TPX = (self._temperature, self._pressure,
                             self._mole_fractions[ip])

    def uploadMoleFractionsFromPhases(self) -> None:
        if not self._initialized:
            self.init()
            return
        self._mole_fractions = [np.asarray(phase.X, dtype=float).copy()
                                for phase in self._phases]
        self.calcElemAbundances()

    @property
    def charge(self) -> float:
        return float(sum(self.phaseCharge(i) for i in range(self.n_phases)))

    """A collection of phases in thermodynamic contact."""
    def __init__(self):
        self._phases = []
        self._phase_moles = []
        self._initialized = False
        self._mole_fractions = []
        self._temperature = 298.15
        self._pressure = OneBar

    def add_phase(self, phase, moles: float = 1.0):
        """Add a phase to the multiphase mixture with an initial mole number."""
        if self._initialized:
            raise VCSError("phases cannot be added after init() has been called")
        if self._temperature == 298.15 and phase.T > 2.0e-3:
            self._temperature = float(phase.T)
            self._pressure = float(phase.P)
        self._phases.append(phase)
        self._phase_moles.append(float(moles))

    @property
    def n_phases(self) -> int:
        return len(self._phases)

    def phase(self, index: int):
        self.init()
        index = self._check_phase_index(index)
        phase = self._phases[index]
        phase.TPX = (self._temperature, self._pressure,
                     self._mole_fractions[index])
        return phase

    @property
    def phases(self):
        return tuple(self._phases)

    @property
    def phase_moles(self) -> list[float]:
        return list(self._phase_moles)

    def set_phase_moles(self, index: int, moles: float):
        self._phase_moles[self._check_phase_index(index)] = float(moles)

    @property
    def T(self) -> float:
        return self._temperature

    @property
    def P(self) -> float:
        return self._pressure

    def set_TP(self, T: float, P: float):
        self._temperature = float(T)
        self._pressure = float(P)
        self.updatePhases()

    def species_names(self) -> list[str]:
        return [name for phase in self._phases for name in phase.species_names]

    def total_enthalpy(self) -> float:
        self.updatePhases()
        h = 0.0
        for i, p in enumerate(self._phases):
            m = self._phase_moles[i]
            if m > 0.0:
                h += m * p.enthalpy_mole
        return h

    def total_entropy(self) -> float:
        self.updatePhases()
        s = 0.0
        for i, p in enumerate(self._phases):
            m = self._phase_moles[i]
            if m > 0.0:
                s += m * p.entropy_mole
        return s

    def total_gibbs(self) -> float:
        self.updatePhases()
        g = 0.0
        for i, p in enumerate(self._phases):
            m = self._phase_moles[i]
            if m > 0.0:
                g += m * p.gibbs_mole
        return g

    def equilibrate(self, XY: str = "TP", solver: str = "auto",
                    rtol: float = 1e-9, max_steps: int = 500,
                    max_iter: int = 100, estimate_equil: int = 0,
                    log_level: int = 0):
        """Equilibrate the multiphase mixture using VCS or stoichiometric solvers."""
        XY = XY.upper()
        if solver in ("auto", "vcs"):
            if XY == "TP":
                solver_inst = VCS_SOLVE()
                solver_inst.vcs_solve_TP(self, print_lvl=log_level, max_steps=max_steps)
                return self
            elif XY in ("HP", "SP", "UV", "SV"):
                return self._equilibrate_multiphase_with_T(XY, rtol=rtol, max_steps=max_steps)
            else:
                raise ValueError(f"Unsupported property pair {XY} for MultiPhase.equilibrate")
        elif solver == "gibbs":
            raise NotImplementedError("Gibbs multiphase solver is not active")
        else:
            raise ValueError(f"Unknown solver {solver}")

    def _equilibrate_multiphase_with_T(self, XY: str, rtol: float = 1e-9, max_steps: int = 500):
        if self.n_phases == 1:
            phase = self._phases[0]
            conserved_mass = self._phase_moles[0] * phase.mean_molecular_weight
            phase.equilibrate(XY, rtol=rtol, max_steps=max_steps)
            self.uploadMoleFractionsFromPhases()
            self.set_TP(phase.T, phase.P)
            self.set_phase_moles(0, conserved_mass / phase.mean_molecular_weight)
            return self
        Htarget = self.total_enthalpy() if XY in ("HP", "UV") else self.total_entropy()
        Tlow = 0.5 * min(p.min_temp for p in self._phases)
        Thigh = 2.0 * max(p.max_temp for p in self._phases)
        for _ in range(100):
            solver_inst = VCS_SOLVE()
            solver_inst.vcs_solve_TP(self, max_steps=max_steps)
            val = self.total_enthalpy() if XY in ("HP", "UV") else self.total_entropy()
            if abs(val - Htarget) <= rtol * max(abs(Htarget), 1.0):
                break
            cp = max(sum(m * p.cp_mole for m, p in zip(self._phase_moles, self._phases) if m > 0), 1.0)
            dT = (Htarget - val) / cp
            dT = max(min(dT, 100.0), -100.0)
            Tnew = max(min(self.T + dT, Thigh), Tlow)
            if abs(Tnew - self.T) < 1e-3:
                break
            self.set_TP(Tnew, self.P)
        return self


def vcs_equilibrate(multiphase: MultiPhase, mode: str = "TP", *,
                    rtol: float = 1e-9, max_steps: int = 500):
    """Equilibrate multiphase mixture using the VCS active-set solver."""
    return multiphase.equilibrate(mode, solver="vcs", rtol=rtol, max_steps=max_steps)


class vcs_VolPhase:
    """Python translation of the VCS phase-local thermodynamic workspace.

    The active-set VCS driver is still open. This class stores the phase-local
    quantities used by that driver and avoids returning fabricated constants.
    """

    _EXIST_NO = 0
    _EXIST_YES = 2
    _EXIST_ALWAYS = 3
    _EXIST_ZEROED = -6
    _UNKNOWN_MOLES = 0
    _UNKNOWN_VOLTAGE = -5
    _STATE_OLD = 0
    _STATE_NEW = 1
    _STATE_PHASE_STABILITY = 2
    _STATE_TMP = 3
    _ELEM_TYPE_ABSPOS = 0
    _ELEM_TYPE_ELECTRON_CHARGE = 1
    _ELEM_TYPE_CHARGE_NEUTRALITY = 2

    def __init__(self, thermo_phase: Any, phase_num: int = 0):
        self._tp = thermo_phase
        self._phase_num = int(phase_num)
        self._species_names = tuple(thermo_phase.species_names)
        sp = getattr(thermo_phase, "species", ())
        if callable(sp):
            sp = getattr(thermo_phase, "_species", [thermo_phase.species(k) for k in range(len(self._species_names))])
        self._species = tuple(sp)
        self._base_element_names = self._get_element_names(thermo_phase)
        self._element_names = self._base_element_names
        self._formula_matrix = np.zeros(
            (len(self._species_names), len(self._element_names)), dtype=float)
        self._element_global_index = list(range(len(self._element_names)))
        self._species_global_index = list(range(len(self._species_names)))
        self._element_active = [True] * len(self._element_names)
        self._element_type = [self._ELEM_TYPE_ABSPOS] * len(self._element_names)
        self._species_unknown_type = [self._UNKNOWN_MOLES] * len(self._species_names)
        self._phi_index = None
        self._x = np.asarray(thermo_phase.X, dtype=float).copy()
        self._creation_mole_numbers = self._x.copy()
        self._creation_global_rxn_numbers = tuple(-1 for _ in self._species_names)
        self._total_moles = 0.0
        self._phi = float(getattr(thermo_phase, "electric_potential", 0.0))
        self._existence = self._EXIST_NO
        self._state_status = self._STATE_OLD
        self._moles_current = False
        self._temperature = float(thermo_phase.T)
        self._pressure = float(thermo_phase.P)
        self._activity_coefficients = np.ones(len(self._species_names), dtype=float)
        self._activity_coefficients_current = False
        self._standard_chemical_potentials = np.zeros(len(self._species_names), dtype=float)
        self._partial_molar_volumes = np.zeros(len(self._species_names), dtype=float)
        self._partial_molar_volumes_current = False
        self._standard_chemical_potentials_current = False
        thermo_api = getattr(thermo_phase, "thermo", thermo_phase)
        ideal_flag = getattr(thermo_api, "is_ideal", None)
        if ideal_flag is None:
            ideal_flag = getattr(thermo_api, "isIdeal", False)
        if callable(ideal_flag):
            ideal_flag = ideal_flag()
        self._ideal = (len(self._species_names) == 1 or bool(ideal_flag))
        self._configure_element_formula_matrix()
        if len(self._species_names) == 1 and self._species[0].charge != 0.0:
            self.setPhiVarIndex(0)
        self._updateMoleFractionDependencies()

    @staticmethod
    def _get_element_names(phase) -> tuple[str, ...]:
        # This mirrors vcs_VolPhase::transferElementsFM, which iterates
        # ThermoPhase::elementName(eT) rather than mechanism-file declarations.
        return tuple(phase.element_names)

    def _n_atoms(self, species_index: int, element: str) -> float:
        method = getattr(self._tp, "n_atoms", None)
        if callable(method):
            return float(method(species_index, element))
        name = self._species_names[species_index]
        mechanism = getattr(self._tp, "_mechanism", None)
        if mechanism is not None:
            for species in mechanism.species:
                if species.name == name:
                    return float(species.composition.get(element, 0.0))
        species = self._species[species_index]
        return float(getattr(species, "composition", {}).get(element, 0.0))

    def _species_charges(self) -> np.ndarray:
        charges = []
        phase_charge = getattr(self._tp, "charge", None)
        for k in range(len(self._species_names)):
            if k < len(self._species):
                charges.append(float(getattr(self._species[k], "charge", 0.0)))
            elif callable(phase_charge):
                charges.append(float(phase_charge(k)))
            else:
                charges.append(0.0)
        return np.asarray(charges, dtype=float)

    def _configure_element_formula_matrix(self) -> None:
        """Translate the pinned phase-local elemental and charge constraints."""
        names = list(self._base_element_names)
        types = [self._ELEM_TYPE_ABSPOS] * len(names)
        active = [True] * len(names)
        charges = self._species_charges()
        has_charge = bool(np.any(charges != 0.0))
        required = getattr(self._tp, "chargeNeutralityNecessary", None)
        if callable(required):
            requires_neutrality = bool(required())
        else:
            requires_neutrality = bool(getattr(
                self._tp, "charge_neutrality_necessary", False))

        charge_neutrality_element = None
        if has_charge and requires_neutrality:
            charge_neutrality_element = len(names)
            names.append("cn_" + (getattr(self._tp, "name", "")
                                  or f"phase{self._phase_num}"))
            types.append(self._ELEM_TYPE_CHARGE_NEUTRALITY)
            active.append(True)

        electron_element = None
        if has_charge:
            if "E" in self._base_element_names:
                electron_element = self._base_element_names.index("E")
                types[electron_element] = self._ELEM_TYPE_ELECTRON_CHARGE
                if requires_neutrality:
                    active[electron_element] = False
            else:
                electron_element = len(names)
                names.append("E")
                types.append(self._ELEM_TYPE_ELECTRON_CHARGE)
                active.append(False)

        formula = np.zeros((len(self._species_names), len(names)), dtype=float)
        for e, element in enumerate(self._base_element_names):
            formula[:, e] = [self._n_atoms(k, element)
                             for k in range(len(self._species_names))]
        if electron_element is not None:
            formula[:, electron_element] = -charges
        if charge_neutrality_element is not None:
            formula[:, charge_neutrality_element] = charges

        self._element_names = tuple(names)
        self._element_type = types
        self._element_active = active
        self._element_global_index = [
            index if index < len(self._base_element_names) else -1
            for index in range(len(names))]
        self._formula_matrix = formula
        self.ChargeNeutralityElement = charge_neutrality_element
        self._electron_element = electron_element

    def _check_species_index(self, k: int) -> int:
        index = int(k)
        if index < 0 or index >= len(self._species_names):
            raise IndexError(f"Species index {index} is out of range")
        return index

    def _updateMoleFractionDependencies(self) -> None:
        # The pinned stability iteration sends intermediate compositions to
        # the thermo phase before applying its damping step. Those trial X
        # vectors can contain negative entries. Ideal-phase activity
        # coefficients are composition-independent, so keep this VCS scratch
        # state locally rather than asking the stricter public TPX setter to
        # accept a nonphysical trial composition.
        if np.any(~np.isfinite(self._x)) or np.any(self._x < 0.0):
            if self._ideal:
                return
            raise VCSError(
                "nonideal VCS phase trial composition is outside the Python "
                "thermo model's admissible domain")
        if len(self._species_names) == 1:
            self._tp.TP = self._temperature, self._pressure
        else:
            self._tp.TPX = (self._temperature, self._pressure, self._x)
        if not self._ideal:
            self._activity_coefficients_current = False
            self._partial_molar_volumes_current = False

    def _updateActCoeff(self) -> None:
        if not self._activity_coefficients_current:
            if self._ideal:
                values = np.ones(len(self._species_names), dtype=float)
            else:
                values = np.asarray(self._tp.activity_coefficients, dtype=float)
            if values.shape != (len(self._species_names),):
                raise VCSError("phase activity coefficients have an invalid shape")
            self._activity_coefficients = values.copy()
            self._activity_coefficients_current = True

    def _updateG0(self) -> None:
        from .constants import GasConstant
        values = []
        for species in self._species:
            thermo = species.thermo
            cp_r, h_rt, s_r = thermo.properties(self._temperature)
            values.append(GasConstant * self._temperature * (h_rt - s_r))
        self._g0 = np.asarray(values, dtype=float)

    def _updateGStar(self) -> None:
        self._standard_chemical_potentials = np.asarray(
            self._tp.standard_chemical_potentials, dtype=float).copy()
        self._standard_chemical_potentials_current = True

    def _updateVolPM(self) -> None:
        self._partial_molar_volumes = np.asarray(
            self._tp.partial_molar_volumes, dtype=float).copy()
        self._partial_molar_volumes_current = True

    def _updateVolStar(self) -> None:
        values = getattr(self._tp, "standard_volumes", None)
        if values is None:
            raise NotImplementedError(
                "phase standard molar volumes are unavailable in the Python phase API")
        self._standard_molar_volumes = np.asarray(values, dtype=float).copy()

    def G0_calc_one(self, k: int) -> float:
        self._check_species_index(k)
        if not hasattr(self, "_g0"):
            self._updateG0()
        return float(self._g0[k])

    def GStar_calc_one(self, k: int) -> float:
        self._check_species_index(k)
        self._updateGStar()
        return float(self._standard_chemical_potentials[k])

    def creationMoleNumbers(self, k: int) -> float:
        return float(self._creation_mole_numbers[self._check_species_index(k)])

    def setCreationMoleNumbers(self, mole_fractions, creation_global_rxn_numbers) -> None:
        values = np.asarray(mole_fractions, dtype=float)
        if values.shape != (len(self._species_names),):
            raise ValueError("creation mole-fraction vector has the wrong size")
        if len(creation_global_rxn_numbers) != len(self._species_names):
            raise ValueError("creation reaction-index vector has the wrong size")
        self._creation_mole_numbers = values.copy()
        self._creation_global_rxn_numbers = tuple(
            int(index) for index in creation_global_rxn_numbers)

    def d(self) -> float:
        return float(getattr(self._tp, "size", 1.0))

    def electricPotential(self) -> float:
        return self._phi

    def elemGlobalIndex(self, e: int) -> int:
        return self._element_global_index[int(e)]

    def elemResize(self, n: int) -> None:
        count = int(n)
        if count < 0:
            raise ValueError("element count must be nonnegative")
        old = len(self._element_names)
        if count < old:
            self._element_names = self._element_names[:count]
            self._element_global_index = self._element_global_index[:count]
            self._element_active = self._element_active[:count]
            self._element_type = self._element_type[:count]
            self._formula_matrix = self._formula_matrix[:, :count]
        elif count > old:
            extra = count - old
            self._element_names += tuple("" for _ in range(extra))
            self._element_global_index.extend([-1] * extra)
            self._element_active.extend([True] * extra)
            self._element_type.extend([0] * extra)
            self._formula_matrix = np.pad(self._formula_matrix, ((0, 0), (0, extra)))

    def elementActive(self, e: int) -> bool:
        return bool(self._element_active[int(e)])

    def elementType(self, e: int) -> int:
        return int(self._element_type[int(e)])

    def elementName(self, e: int) -> str:
        return self._element_names[int(e)]

    def eos_name(self) -> str:
        return str(getattr(self._tp, "thermo_model", type(self._tp.thermo).__name__))

    def exists(self) -> int:
        return self._existence

    def getFormulaMatrix(self, fm: np.ndarray) -> None:
        output = np.asarray(fm)
        if output.shape[0] < self._formula_matrix.shape[0] or output.shape[1] < self._formula_matrix.shape[1]:
            raise ValueError("formula matrix output is too small")
        output[:self._formula_matrix.shape[0], :self._formula_matrix.shape[1]] = self._formula_matrix

    def isIdealSoln(self) -> bool:
        return self._ideal

    def moleFraction(self, k: int) -> float:
        return float(self._x[self._check_species_index(k)])

    def moleFractions(self) -> np.ndarray:
        return self._x.copy()

    def nElemConstraints(self) -> int:
        return len(self._element_names)

    def np_dLnActCoeffdMolNumber(self) -> np.ndarray:
        if self._ideal:
            return np.zeros((len(self._species_names), len(self._species_names)))
        model = getattr(self._tp, "_thermo", self._tp)
        if type(model).__name__ not in ("MargulesVPSSTP", "RedlichKisterVPSSTP"):
            raise NotImplementedError(
                "VCS activity Jacobians are only implemented for the current "
                "Margules and Redlich-Kister models; other nonideal phase "
                "derivatives are not ported")
        derivative = getattr(model, "getdlnActCoeffdlnN", None)
        if not callable(derivative):
            raise NotImplementedError(
                "nonideal d(ln activity coefficient)/d(moles) is unavailable")

        nsp = len(self._species_names)
        logarithmic = np.zeros((nsp, nsp), dtype=float)
        derivative(logarithmic)
        if np.any(~np.isfinite(logarithmic)):
            raise VCSError("nonideal activity Jacobian contains non-finite values")

        # Match vcs_VolPhase::_updateLnActCoeffJac: first obtain
        # d(ln gamma_k)/d(ln n_j), then scale each column by the phase total
        # divided by the regularized base amount of species j.
        phase_total = float(self._total_moles)
        if phase_total < 1.0e-14:
            phase_total = 1.0
        values = logarithmic.copy()
        for j in range(nsp):
            moles_j_base = phase_total * float(self._x[j])
            if moles_j_base < 1.0e-200:
                moles_j_base = (1.0e-7 * moles_j_base
                                + 1.0e-13 * phase_total + 1.0e-150)
            values[:, j] *= phase_total / moles_j_base
        return values

    def phiVarIndex(self) -> int:
        return -1 if self._phi_index is None else self._phi_index

    def sendToVCS_ActCoeff(self, state_calc, act=None, *, moles=None) -> None:
        # Two-argument form follows the C++ API. A one-argument call remains
        # supported for the earlier Python helper API.
        if act is None:
            output = np.asarray(state_calc)
            state = self._state_status
        else:
            output = np.asarray(act)
            state = int(state_calc)
        if output.ndim != 1 or output.size <= max(self._species_global_index, default=-1):
            raise ValueError("activity-coefficient output array is too small")
        self.updateFromVCS_MoleNumbers(moles, state)
        self._updateActCoeff()
        for local, global_index in enumerate(self._species_global_index):
            output[global_index] = self._activity_coefficients[local]

    def sendToVCS_GStar(self, gs: np.ndarray) -> None:
        self._updateGStar()
        output = np.asarray(gs)
        if output.ndim != 1 or output.size <= max(self._species_global_index, default=-1):
            raise ValueError("standard chemical-potential output array is too small")
        for local, global_index in enumerate(self._species_global_index):
            output[global_index] = self._standard_chemical_potentials[local]

    def sendToVCS_LnActCoeffJac(self, jac: np.ndarray) -> None:
        values = self.np_dLnActCoeffdMolNumber()
        output = np.asarray(jac)
        for i, gi in enumerate(self._species_global_index):
            for j, gj in enumerate(self._species_global_index):
                output[gi, gj] = values[i, j]

    def sendToVCS_VolPM(self, vpm: np.ndarray) -> None:
        self._updateVolPM()
        output = np.asarray(vpm)
        for local, global_index in enumerate(self._species_global_index):
            output[global_index] = self._partial_molar_volumes[local]

    def setElemGlobalIndex(self, e: int, idx: int) -> None:
        self._element_global_index[int(e)] = int(idx)

    def setExistence(self, exists: bool) -> None:
        value = int(exists)
        if value in (self._EXIST_NO, self._EXIST_ZEROED) and self._total_moles != 0.0:
            raise ValueError("cannot mark a phase absent while it contains moles")
        if value in (self._EXIST_YES, self._EXIST_ALWAYS) and self._total_moles == 0.0 and not (
                len(self._species_names) == 1 and self._phi_index == 0):
            raise ValueError("cannot mark a zero-mole phase as present")
        self._existence = value

    def setMoleFractions(self, x: np.ndarray) -> None:
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.size < len(self._species_names):
            raise ValueError("mole-fraction input array is too small")
        values = values[:len(self._species_names)].copy()
        total = float(np.sum(values))
        if abs(total) > 1e-13:
            values /= total
        self._x = values
        self._updateMoleFractionDependencies()

    def setMoleFractionsState(self, mol_num: float, x: np.ndarray,
                              vcs_state_status: int = _STATE_TMP) -> None:
        values = np.asarray(x, dtype=float)
        if values.ndim != 1 or values.size < len(self._species_names):
            raise ValueError("mole-fraction input array is too small")
        if mol_num != 0.0 and vcs_state_status != self._STATE_TMP:
            raise ValueError("nonzero phase amount requires temporary state status")
        if mol_num != 0.0 and self._existence == self._EXIST_ZEROED:
            raise ValueError("a zeroed phase cannot be assigned nonzero moles")
        values = values[:len(self._species_names)].copy()
        total = float(np.sum(values))
        if total == 0.0:
            raise ValueError("mole fractions must have a nonzero sum")
        values /= total
        self._x = values
        self._total_moles = float(mol_num)
        self._existence = (self._EXIST_YES if mol_num > 0.0
                           else min(self._existence, self._EXIST_NO))
        self._state_status = int(vcs_state_status)
        self._moles_current = mol_num == 0.0
        self._updateMoleFractionDependencies()

    def setMolesCurrent(self, state_calc: int = _STATE_OLD) -> None:
        self._moles_current = True
        self._state_status = int(state_calc)

    def setMolesOutOfDate(self, state_calc: int | None = None) -> None:
        self._moles_current = False
        if state_calc is not None and state_calc != -1:
            self._state_status = int(state_calc)

    def setPhiVarIndex(self, sp: int) -> None:
        index = self._check_species_index(sp)
        self._phi_index = index
        self._species_unknown_type[index] = self._UNKNOWN_VOLTAGE
        if len(self._species_names) == 1 and index == 0:
            self._existence = self._EXIST_ALWAYS

    def setSpGlobalIndexVCS(self, sp: int, idx: int) -> None:
        self._species_global_index[self._check_species_index(sp)] = int(idx)

    def setTotalMoles(self, tm: float) -> None:
        self._total_moles = float(tm)
        if len(self._species_names) == 1 and self._phi_index == 0:
            self._existence = self._EXIST_ALWAYS
        else:
            self._existence = self._EXIST_YES if tm > 0.0 else self._EXIST_NO

    def spGlobalIndexVCS(self, sp: int) -> int:
        return self._species_global_index[self._check_species_index(sp)]

    def speciesUnknownType(self, sp: int) -> int:
        return self._species_unknown_type[self._check_species_index(sp)]

    def totalMoles(self) -> float:
        return self._total_moles

    def transferElementsFM(self) -> None:
        self._configure_element_formula_matrix()

    def setElectricPotential(self, potential: float) -> None:
        """Set phase potential and invalidate native-equivalent state caches."""
        self._phi = float(potential)
        if hasattr(self._tp, "electric_potential"):
            self._tp.electric_potential = self._phi
        self._activity_coefficients_current = False
        self._partial_molar_volumes_current = False
        self._standard_chemical_potentials_current = False

    def setMolesFromVCS(self, state_calc: int, moles) -> None:
        """Translate a global VCS species vector into this phase's state."""
        state = int(state_calc)
        if state not in (self._STATE_OLD, self._STATE_NEW):
            raise VCSError("VCS mole upload requires an OLD or NEW state")
        values = np.asarray(moles, dtype=float)
        if values.ndim != 1:
            raise ValueError("VCS mole vector must be one-dimensional")

        amounts = np.zeros(len(self._species_names), dtype=float)
        for k, global_index in enumerate(self._species_global_index):
            if global_index < 0 or global_index >= values.size:
                raise IndexError(
                    f"VCS species index {global_index} is outside the mole vector")
            if self._species_unknown_type[k] != self._UNKNOWN_VOLTAGE:
                amounts[k] = max(0.0, float(values[global_index]))

        total = 0.0
        for amount in amounts:
            total += float(amount)
        self._total_moles = total
        if total > 0.0:
            self._x = amounts / total
            self._existence = self._EXIST_YES
        else:
            # The native method retains the prior mole fractions when a phase
            # disappears; phase stability handling chooses a replacement later.
            self._existence = self._EXIST_NO

        if self._phi_index is not None:
            self._x[self._phi_index] = (
                1.0 if len(self._species_names) == 1 else 0.0)
            phi_global = self._species_global_index[self._phi_index]
            if phi_global < 0 or phi_global >= values.size:
                raise IndexError(
                    f"VCS voltage index {phi_global} is outside the mole vector")
            self.setElectricPotential(float(values[phi_global]))
            if len(self._species_names) == 1:
                self._existence = self._EXIST_ALWAYS

        self._updateMoleFractionDependencies()
        if state == self._STATE_OLD and total > 0.0:
            self._creation_mole_numbers = self._x.copy()
        self._moles_current = True
        self._state_status = state

    def setMolesFromVCSCheck(self, state_calc: int, moles,
                             phase_moles) -> None:
        """Update phase composition and verify its cached phase total."""
        self.setMolesFromVCS(state_calc, moles)
        totals = np.asarray(phase_moles, dtype=float)
        if totals.ndim != 1 or self._phase_num >= totals.size:
            raise ValueError("VCS phase-total vector has the wrong size")
        expected = float(totals[self._phase_num])
        if expected != self._total_moles and not vcs_doubleEqual(
                expected, self._total_moles):
            raise VCSError(
                "VCS phase mole total is inconsistent with the species vector")

    def updateFromVCS_MoleNumbers(self, moles=None,
                                  state_calc: int = _STATE_OLD) -> None:
        """Refresh this phase only when the requested VCS state is stale."""
        state = int(state_calc)
        if state not in (self._STATE_OLD, self._STATE_NEW):
            return
        if not self._moles_current or state != self._state_status:
            if moles is None:
                raise ValueError(
                    "VCS mole-number update requires a solver state vector")
            self.setMolesFromVCS(state, moles)

    def updateActCoeffCache(self, state_calc: int, moles,
                            output: np.ndarray | None = None) -> np.ndarray:
        """Refresh phase state/activity coefficients and optionally export them."""
        self.updateFromVCS_MoleNumbers(moles, state_calc)
        self._updateActCoeff()
        if output is None:
            return self._activity_coefficients.copy()
        target = np.asarray(output)
        if target.ndim != 1 or target.size <= max(self._species_global_index, default=-1):
            raise ValueError("activity-coefficient output array is too small")
        for local, global_index in enumerate(self._species_global_index):
            target[global_index] = self._activity_coefficients[local]
        return target

    def thermoPhase(self):
        return self._tp

    def phaseNum(self) -> int:
        return self._phase_num

    def nSpecies(self) -> int:
        return len(self._species_names)

    def speciesName(self, k: int) -> str:
        return self._species_names[self._check_species_index(k)]

    def activityCoefficients(self) -> np.ndarray:
        self._updateActCoeff()
        return self._activity_coefficients.copy()

    def temperature(self) -> float:
        return self._temperature

    def pressure(self) -> float:
        return self._pressure

    def setState_TP(self, temperature: float, pressure: float) -> None:
        temperature = float(temperature)
        pressure = float(pressure)
        if temperature == self._temperature and pressure == self._pressure:
            return
        if hasattr(self._tp, "electric_potential"):
            self._tp.electric_potential = self._phi
        self._temperature = temperature
        self._pressure = pressure
        self._updateMoleFractionDependencies()
        self._activity_coefficients_current = False
        self._partial_molar_volumes_current = False
        self._standard_chemical_potentials_current = False
        for name in ("_g0",):
            if hasattr(self, name):
                delattr(self, name)

class VCS_SOLVE:
    """VCS problem preparation with the equilibrium iteration still open."""

    def __init__(self):
        self._prepared_problem = None
        self._temperature = None
        self._pressure = None
        self._species_status = None
        self._num_rxn_minor_zeroed = 0
        self._num_species_reduced = 0
        self._num_reactions_reduced = 0
        self._tolmaj = 0.010
        self._tolmin = 0.010
        self._tolmin2 = 0.001
        self._tolmaj2 = 0.001
        self._delta_mol_num_species = None
        self._delta_phase_moles = None
        self._iterations = 0

    def prepare_problem(self, multiphase: MultiPhase, *, test: float = -1.0e-10):
        """Translate the VCS phase/species map and choose its component basis."""
        if not isinstance(multiphase, MultiPhase):
            raise TypeError("multiphase must be a MultiPhase instance")
        if not multiphase._phases:
            raise VCSError("VCS problem must contain at least one phase")

        self._temperature = float(multiphase.T)
        self._pressure = float(multiphase.P)
        multiphase.set_TP(self._temperature, self._pressure)

        phase_objects = []
        species_phase_ids = []
        species_unknown_types = []
        species_single_phase = []
        element_names = list(multiphase.element_names)
        element_types = [vcs_VolPhase._ELEM_TYPE_ABSPOS] * len(element_names)
        element_active = [True] * len(element_names)
        phase_local_to_global = []
        phase_species_offsets = []
        species_offset = 0

        for iph, phase in enumerate(multiphase._phases):
            vphase = vcs_VolPhase(phase, phase_num=iph)
            phase_objects.append(vphase)
            mapping = []
            for local, name in enumerate(vphase._element_names):
                if name not in element_names:
                    element_names.append(name)
                    element_types.append(vphase._element_type[local])
                    element_active.append(vphase._element_active[local])
                else:
                    global_index = element_names.index(name)
                    if vphase._element_type[local] != vcs_VolPhase._ELEM_TYPE_ABSPOS:
                        element_types[global_index] = vphase._element_type[local]
                    element_active[global_index] = (
                        element_active[global_index] and vphase._element_active[local])
                mapping.append(element_names.index(name))
            phase_local_to_global.append(tuple(mapping))
            count = len(phase.species_names)
            phase_species_offsets.append(species_offset)
            species_phase_ids.extend([iph] * count)
            species_unknown_types.extend(vphase._species_unknown_type)
            species_single_phase.extend([count <= 1] * count)
            species_offset += count

        species_names = tuple(
            name for phase in multiphase._phases for name in phase.species_names)
        formula = np.zeros((len(species_names), len(element_names)), dtype=float)
        species_standard_gibbs_rt = np.zeros(len(species_names), dtype=float)
        row = 0
        rt = GasConstant * multiphase.T
        if not np.isfinite(rt) or rt <= 0.0:
            raise VCSError("VCS temperature must be finite and positive")
        for iph, vphase in enumerate(phase_objects):
            for k in range(vphase.nSpecies()):
                for local, global_index in enumerate(phase_local_to_global[iph]):
                    formula[row, global_index] = vphase._formula_matrix[k, local]
                species_standard_gibbs_rt[row] = (
                    vphase.GStar_calc_one(k) / rt)
                row += 1

        mole_numbers = multiphase._species_moles()
        unknown_types = np.asarray(species_unknown_types, dtype=int)
        abundance_goals = vcs_elab(formula, mole_numbers, unknown_types)
        species_sizes = np.empty(len(species_names), dtype=float)
        for kspec in range(len(species_names)):
            size = 0.0
            for jelem in range(len(element_names)):
                size += abs(float(formula[kspec, jelem]))
            species_sizes[kspec] = size if size > 0.0 else 1.0
        delta_g_rxn = np.zeros(max(len(species_names) - min(
            len(element_names), len(species_names)), 0), dtype=float)
        basis = vcs_basopt_components(
            formula, mole_numbers, abundance_goals, element_active,
            element_types, species_sizes=species_sizes,
            single_species_phases=species_single_phase,
            species_unknown_types=unknown_types,
            standard_gibbs_rt=species_standard_gibbs_rt,
            delta_g_rxn=delta_g_rxn, test=test)
        inverse_permutation = np.empty(len(species_names), dtype=int)
        inverse_permutation[np.asarray(basis["species_permutation"], dtype=int)] = (
            np.arange(len(species_names), dtype=int))
        for iph, vphase in enumerate(phase_objects):
            offset = phase_species_offsets[iph]
            for local_species in range(vphase.nSpecies()):
                vphase.setSpGlobalIndexVCS(
                    local_species, int(inverse_permutation[offset + local_species]))
        if np.any(unknown_types == vcs_VolPhase._UNKNOWN_VOLTAGE):
            raise VCSError(
                "VCS interfacial-voltage stoichiometry is not implemented")
        permuted_formula = formula[list(basis["species_permutation"]), :].copy()
        permuted_moles = np.asarray(mole_numbers, dtype=float)[
            list(basis["species_permutation"])].copy()
        permuted_unknown = unknown_types[list(basis["species_permutation"])].copy()
        permuted_gibbs_rt = species_standard_gibbs_rt[
            list(basis["species_permutation"])].copy()
        permuted_single_phase = np.asarray(species_single_phase, dtype=bool)[
            list(basis["species_permutation"])].copy()
        species_charges = np.concatenate([
            vphase._species_charges() for vphase in phase_objects])
        permuted_charges = species_charges[
            list(basis["species_permutation"])].copy()
        (permuted_formula, element_order, reordered_goals,
         reordered_active, reordered_types) = vcs_elem_rearrange(
            permuted_formula, abundance_goals, element_active, element_types,
            basis["num_components"])
        stoichiometry = vcs_formation_stoichiometry(
            permuted_formula, basis["num_components"])
        basis_phase_ids = np.asarray(species_phase_ids, dtype=int)[
            list(basis["species_permutation"])].copy()
        phase_delta_moles, phase_participation = vcs_phase_reaction_changes(
            stoichiometry, basis_phase_ids, basis["num_components"],
            len(phase_objects))
        basis["formation_reaction_species"] = tuple(
            range(int(basis["num_components"]), len(species_names)))
        basis["formation_reaction_stoichiometry"] = stoichiometry
        basis["formation_reaction_size"] = np.sum(np.abs(stoichiometry), axis=0)
        self._prepared_problem = {
            "temperature": multiphase.T,
            "pressure": multiphase.P,
            "num_reactions_reduced": (
                len(species_names) - int(basis["num_components"])),
            "element_names": tuple(element_names),
            "element_types": np.asarray(element_types, dtype=int),
            "element_active": np.asarray(element_active, dtype=bool),
            "species_names": species_names,
            "species_phase_ids": np.asarray(species_phase_ids, dtype=int),
            "species_unknown_types": unknown_types,
            "basis_ordered_species_phase_ids": basis_phase_ids,
            "basis_ordered_species_unknown_types": permuted_unknown,
            "basis_ordered_species_single_phase": permuted_single_phase,
            "basis_ordered_species_charges": permuted_charges,
            "basis_ordered_standard_gibbs_rt": permuted_gibbs_rt,
            "formula_matrix": formula,
            "basis_ordered_formula_matrix": permuted_formula,
            "basis_ordered_mole_numbers": permuted_moles,
            "basis_ordered_species_unknown_types": permuted_unknown,
            "basis_ordered_species_names": tuple(
                species_names[i] for i in basis["species_permutation"]),
            "element_order_after_basis_rearrangement": tuple(
                element_names[i] for i in element_order),
            "reordered_element_abundance_goals": reordered_goals,
            "reordered_element_active": reordered_active,
            "reordered_element_types": reordered_types,
            "formation_reaction_species": basis["formation_reaction_species"],
            "formation_reaction_stoichiometry": stoichiometry,
            "formation_reaction_size": basis.get("formation_reaction_size", np.sum(np.abs(stoichiometry), axis=0)),
            "delta_mol_num_phase": phase_delta_moles,
            "phase_participation": phase_participation,
            "mole_numbers": np.asarray(mole_numbers, dtype=float),
            "element_abundance_goals": abundance_goals,
            "species_standard_gibbs_rt": species_standard_gibbs_rt,
            "phase_objects": tuple(phase_objects),
            "component_basis": basis,
        }
        self._num_species_reduced = len(species_names)
        self._num_reactions_reduced = len(species_names) - int(
            basis["num_components"])
        self._delta_mol_num_species = np.zeros(len(species_names), dtype=float)
        self._delta_phase_moles = np.zeros(len(phase_objects), dtype=float)
        self.vcs_evalSS_TP(0, 0, self._temperature, self._pressure)
        return self._prepared_problem

    def vcs_basopt(self, do_just_components: bool = False,
                   test: float = -1.0e-10) -> int:
        """Re-select the VCS component basis and rebuild its reaction system.

        This ports the pinned ``VCS_SOLVE::vcs_basopt`` re-optimization path:
        the modified Gram-Schmidt selector chooses species from the current
        OLD amounts, species slots are switched using ``vcs_switch_pos``, then
        the independent element coordinates, formation reactions, and phase
        mole-change tables are rebuilt. The voltage-specific source branch is
        rejected by preparation, so this method handles material-species
        systems only.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before basis optimization")
        if not np.isfinite(test):
            raise ValueError("VCS basis marker must be finite")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            self.initialize_thermodynamic_state()
            state = self._old_thermodynamic_state
        nsp = len(p["species_names"])
        old_basis = p["component_basis"]
        old_ncomp = int(old_basis["num_components"])
        old_nspecies_reduced = (nsp if self._num_species_reduced is None
                                 else int(self._num_species_reduced))
        if not 0 <= old_nspecies_reduced <= nsp:
            raise VCSError("VCS reduced species count is out of range")
        predeleted = nsp - old_nspecies_reduced
        permutation = tuple(int(i) for i in old_basis["species_permutation"])
        if (len(permutation) != nsp or sorted(permutation) != list(range(nsp))
                or np.any(p["basis_ordered_species_unknown_types"] ==
                          vcs_VolPhase._UNKNOWN_VOLTAGE)):
            raise VCSError("VCS basis optimization received an invalid species map")

        nc_trial = min(p["basis_ordered_formula_matrix"].shape)
        current_moles = np.asarray(state["mole_numbers_old"], dtype=float)
        current_gibbs = np.asarray(p["basis_ordered_standard_gibbs_rt"], dtype=float)
        if (current_moles.shape != (nsp,) or current_gibbs.shape != (nsp,)
                or not np.all(np.isfinite(current_moles))
                or np.any(current_moles < 0.0)):
            raise VCSError("VCS basis optimization has invalid OLD species state")

        # vcs_basopt uses reaction Gibbs values indexed by the candidate
        # product slot. The initial candidate range begins at nc_trial; map
        # those slots through the current component count explicitly.
        delta_for_candidates = np.zeros(max(nsp - nc_trial, 0), dtype=float)
        old_delta = np.asarray(state["delta_g_rxn_old"], dtype=float)
        for slot in range(nc_trial, nsp):
            old_reaction = slot - old_ncomp
            candidate_reaction = slot - nc_trial
            if 0 <= old_reaction < old_delta.size:
                delta_for_candidates[candidate_reaction] = old_delta[old_reaction]

        candidate = vcs_basopt_components(
            p["basis_ordered_formula_matrix"], current_moles,
            p["reordered_element_abundance_goals"],
            p["reordered_element_active"], p["reordered_element_types"],
            single_species_phases=p["basis_ordered_species_single_phase"],
            species_unknown_types=p["basis_ordered_species_unknown_types"],
            standard_gibbs_rt=current_gibbs,
            delta_g_rxn=delta_for_candidates, test=test)
        new_ncomp = int(candidate["num_components"])
        desired_slots = tuple(int(i) for i in candidate["species_permutation"])
        if (new_ncomp < 0 or new_ncomp > nc_trial
                or len(desired_slots) != nsp
                or sorted(desired_slots) != list(range(nsp))):
            raise VCSError("VCS basis selector returned an invalid permutation")

        # Apply the source's pairwise slot switches so phase-local VCS indices
        # and every species-indexed state vector stay synchronized.
        current_slots = list(range(nsp))
        for target, wanted in enumerate(desired_slots):
            if current_slots[target] == wanted:
                continue
            source = current_slots.index(wanted, target + 1)
            self.vcs_switch_pos(False, target, source)
            current_slots[target], current_slots[source] = (
                current_slots[source], current_slots[target])
        new_permutation = tuple(permutation[i] for i in desired_slots)

        if do_just_components:
            old_basis["num_components"] = new_ncomp
            old_basis["species_permutation"] = new_permutation
            old_basis["component_species"] = new_permutation[:new_ncomp]
            old_basis["remaining_species"] = new_permutation[new_ncomp:]
            old_basis["working_mole_numbers"] = np.asarray(
                candidate["working_mole_numbers"], dtype=float).copy()
            old_basis["orthogonal_basis_squared_norms"] = np.asarray(
                candidate["orthogonal_basis_squared_norms"], dtype=float).copy()
            p["basis_ordered_species_names"] = tuple(
                p["species_names"][i] for i in new_permutation)
            p["basis_ordered_mole_numbers"] = state["mole_numbers_old"].copy()
            reduced_reactions = old_nspecies_reduced - new_ncomp
            if reduced_reactions < 0:
                raise VCSError(
                    "VCS component basis exceeds the reduced species count")
            p["num_reactions_reduced"] = reduced_reactions
            self._num_species_reduced = old_nspecies_reduced
            self._num_reactions_reduced = reduced_reactions
            return new_ncomp

        # Rebuild the element coordinate order for the newly chosen basis.
        # This is the array counterpart of the pinned formula-matrix basis
        # setup; source element order remains available in formula_matrix.
        raw_formula = np.asarray(p["formula_matrix"], dtype=float)[
            list(new_permutation), :]
        (ordered_formula, element_order, ordered_goals, ordered_active,
         ordered_types) = vcs_elem_rearrange(
            raw_formula, p["element_abundance_goals"], p["element_active"],
            p["element_types"], new_ncomp)

        old_basis["num_components"] = new_ncomp
        old_basis["species_permutation"] = new_permutation
        old_basis["component_species"] = new_permutation[:new_ncomp]
        old_basis["remaining_species"] = new_permutation[new_ncomp:]
        old_basis["working_mole_numbers"] = np.asarray(
            candidate["working_mole_numbers"], dtype=float).copy()
        old_basis["orthogonal_basis_squared_norms"] = np.asarray(
            candidate["orthogonal_basis_squared_norms"], dtype=float).copy()
        p["basis_ordered_formula_matrix"] = ordered_formula
        p["basis_ordered_species_names"] = tuple(
            p["species_names"][i] for i in new_permutation)
        p["basis_ordered_mole_numbers"] = state["mole_numbers_old"].copy()
        p["element_order_after_basis_rearrangement"] = tuple(
            p["element_names"][i] for i in element_order)
        p["reordered_element_abundance_goals"] = ordered_goals
        p["reordered_element_active"] = ordered_active
        p["reordered_element_types"] = ordered_types

        stoichiometry = vcs_formation_stoichiometry(ordered_formula, new_ncomp)
        p["formation_reaction_species"] = tuple(range(new_ncomp, nsp))
        p["formation_reaction_stoichiometry"] = stoichiometry
        p["formation_reaction_size"] = np.sum(np.abs(stoichiometry), axis=0)
        phase_changes, participation = vcs_phase_reaction_changes(
            stoichiometry, p["basis_ordered_species_phase_ids"], new_ncomp,
            len(p["phase_objects"]))
        p["delta_mol_num_phase"] = phase_changes
        p["phase_participation"] = participation
        total_reactions = nsp - new_ncomp
        reduced_reactions = total_reactions - predeleted
        if reduced_reactions < 0:
            raise VCSError("VCS component basis exceeds the reduced species count")
        p["num_reactions_reduced"] = reduced_reactions
        self._num_species_reduced = old_nspecies_reduced
        self._num_reactions_reduced = reduced_reactions
        self._delta_mol_num_species = np.zeros(nsp, dtype=float)
        self._delta_phase_moles = np.zeros(len(p["phase_objects"]), dtype=float)

        # The component basis may have changed, so reaction-indexed Gibbs
        # values must be recalculated against the rebuilt stoichiometry.
        reactions = np.arange(new_ncomp, nsp, dtype=int)
        for suffix in ("old", "new"):
            state[f"delta_g_rxn_{suffix}"] = vcs_deltag(
                np.zeros(nsp - new_ncomp, dtype=float),
                state[f"species_free_energies_{suffix}"], stoichiometry,
                reactions, state["species_status"], new_ncomp,
                state[f"mole_numbers_{suffix}"])
        if do_just_components:
            return new_ncomp
        return new_ncomp

    def vcs_setFlagsVolPhases(self, up_to_date: bool,
                              state_calc: int = vcs_VolPhase._STATE_OLD) -> None:
        """Set phase mole-cache flags using the pinned VCS state convention."""
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before updating phase flags")
        if state_calc not in (vcs_VolPhase._STATE_OLD,
                              vcs_VolPhase._STATE_NEW):
            raise ValueError("VCS phase state must be OLD or NEW")
        for phase in self._prepared_problem["phase_objects"]:
            if up_to_date:
                phase.setMolesCurrent(state_calc)
            else:
                phase.setMolesOutOfDate(state_calc)

    def vcs_updateVP(self, state_calc: int = vcs_VolPhase._STATE_OLD) -> None:
        """Upload one solver mole state to every phase, matching ``vcs_updateVP``."""
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before phase upload")
        if state_calc == vcs_VolPhase._STATE_OLD:
            state_key, totals_key = "mole_numbers_old", "phase_moles_old"
        elif state_calc == vcs_VolPhase._STATE_NEW:
            state_key, totals_key = "mole_numbers_new", "phase_moles_new"
        else:
            raise ValueError("VCS phase state must be OLD or NEW")
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            raise VCSError("VCS thermodynamic state must be initialized before phase upload")
        moles = np.asarray(state[state_key], dtype=float)
        totals = np.asarray(state[totals_key], dtype=float)
        expected_species = len(self._prepared_problem["species_names"])
        expected_phases = len(self._prepared_problem["phase_objects"])
        if (moles.shape != (expected_species,) or totals.shape != (expected_phases,)
                or np.any(~np.isfinite(moles)) or np.any(moles < 0.0)
                or np.any(~np.isfinite(totals)) or np.any(totals < 0.0)):
            raise VCSError("VCS phase upload vectors have invalid dimensions or values")
        for phase in self._prepared_problem["phase_objects"]:
            phase.setMolesFromVCSCheck(state_calc, moles, totals)

    def _refresh_old_thermodynamics(self) -> None:
        """Recompute OLD activities, chemical potentials, and reaction Gibbs."""
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before thermodynamic refresh")
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        nsp = len(p["species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        self.vcs_updateVP(vcs_VolPhase._STATE_OLD)
        activity = np.ones(nsp, dtype=float)
        potentials = np.zeros(len(p["phase_objects"]), dtype=float)
        for iph, phase in enumerate(p["phase_objects"]):
            if phase.nSpecies() > 1:
                phase.sendToVCS_ActCoeff(
                    vcs_VolPhase._STATE_OLD, activity,
                    moles=state["mole_numbers_old"])
            potentials[iph] = phase.electricPotential()
        state["activity_coefficients_old"] = activity
        state["phase_potentials_old"] = potentials
        reaction_species = np.arange(ncomp, nsp, dtype=int)
        state["species_free_energies_old"] = vcs_dfe_species_free_energies(
            p["basis_ordered_standard_gibbs_rt"],
            state["ln_mnaught_species"], state["mole_numbers_old"],
            state["phase_moles_old"], p["basis_ordered_species_phase_ids"],
            activity, p["basis_ordered_species_charges"], potentials,
            state["species_status"], p["basis_ordered_species_single_phase"],
            p["basis_ordered_species_unknown_types"], reaction_species,
            ncomp, faraday_dim=Faraday / (GasConstant * p["temperature"]),
            initial_free_energies=state["species_free_energies_old"])
        state["delta_g_rxn_old"] = vcs_deltag(
            np.zeros(nsp - ncomp, dtype=float),
            state["species_free_energies_old"],
            p["formation_reaction_stoichiometry"], reaction_species,
            state["species_status"], ncomp, state["mole_numbers_old"])

    def solve_tp_component_calc(self) -> bool:
        """Run the pinned VCS initial component-state calculation stage."""
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before component calculation")
        if getattr(self, "_old_thermodynamic_state", None) is None:
            self.initialize_thermodynamic_state()
        self.vcs_basopt(False, -1.0e-10)
        self._refresh_old_thermodynamics()
        all_minor_zeroed = self.vcs_evaluate_speciesType()
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        balanced = vcs_elabcheck(
            vcs_elab(p["basis_ordered_formula_matrix"],
                     state["mole_numbers_old"],
                     p["basis_ordered_species_unknown_types"]),
            p["reordered_element_abundance_goals"],
            p["component_basis"]["num_components"], p["reordered_element_active"],
            p["reordered_element_types"], p["basis_ordered_formula_matrix"],
            state["mole_numbers_old"])
        if not balanced:
            self.vcs_elcorr()
            self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
            self._refresh_old_thermodynamics()
        return bool(all_minor_zeroed)

    def vcs_evalSS_TP(self, ipr: int, ip1: int, temperature: float,
                      pressure: float) -> int:
        """Update phase states and dimensionless standard Gibbs energies.

        This follows the pinned VCS_SOLVE::vcs_evalSS_TP body: each volume
        phase is set to the solver's current T/P, its standard-state chemical
        potentials are copied into the global species vector, and that vector
        is divided by R*T. The native routine receives ipr, ip1, temperature,
        and pressure but uses the solver's stored T/P values; the Python
        method preserves that behavior.
        """
        if self._prepared_problem is None:
            raise VCSError(
                "VCS problem must be prepared before standard-state evaluation")
        del ipr, ip1, temperature, pressure
        prepared = self._prepared_problem
        temp = float(self._temperature)
        pres = float(self._pressure)
        rt = GasConstant * temp
        if not np.isfinite(rt) or rt <= 0.0:
            raise VCSError("VCS temperature must be finite and positive")

        gstar_rt = np.zeros(len(prepared["species_names"]), dtype=float)
        for phase in prepared["phase_objects"]:
            phase.setState_TP(temp, pres)
            phase.sendToVCS_GStar(gstar_rt)
        gstar_rt /= rt

        permutation = np.asarray(
            prepared["component_basis"]["species_permutation"], dtype=int)
        prepared["basis_ordered_standard_gibbs_rt"] = gstar_rt
        standard_source_order = np.empty_like(gstar_rt)
        standard_source_order[permutation] = gstar_rt
        prepared["species_standard_gibbs_rt"] = standard_source_order
        return VCS_SUCCESS

    def update_phase_state(
        self, mole_numbers, state_calc: int = vcs_VolPhase._STATE_OLD
    ):
        """Synchronize prepared phase objects from one basis-ordered VCS vector.

        This ports the phase-mole aggregation and VCS phase-state upload used
        before the native chemical-potential/activity-coefficient evaluation.
        It does not perform the VCS equilibrium iteration.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before updating phase state")
        state = int(state_calc)
        moles = np.asarray(mole_numbers, dtype=float)
        prepared = self._prepared_problem
        if moles.shape != (len(prepared["species_names"]),):
            raise ValueError("VCS species mole vector has the wrong size")
        totals, total = vcs_tmoles(
            moles, prepared["basis_ordered_species_phase_ids"],
            prepared["basis_ordered_species_unknown_types"],
            len(prepared["phase_objects"]))
        for iph, phase in enumerate(prepared["phase_objects"]):
            phase.setTotalMoles(float(totals[iph]))
            phase.setMolesOutOfDate(state)
            phase.setMolesFromVCSCheck(state, moles, totals)
        return {"phase_moles": totals, "total_moles": total}

    def initialize_thermodynamic_state(self, mole_numbers=None):
        """Build the OLD VCS thermodynamic vectors from prepared phase state.

        This ports the VCS standard-state, activity-coefficient, and
        species/reaction free-energy setup needed before phase-stability tests.
        It does not estimate equilibrium composition or mutate element totals.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before state initialization")
        p = self._prepared_problem
        nsp = len(p["species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        moles = (np.asarray(p["basis_ordered_mole_numbers"], dtype=float).copy()
                 if mole_numbers is None else np.asarray(mole_numbers, dtype=float).copy())
        if moles.shape != (nsp,) or np.any(~np.isfinite(moles)):
            raise ValueError("VCS OLD mole vector has invalid dimensions or values")
        if np.any(moles < 0.0):
            raise ValueError("VCS OLD mole numbers must be nonnegative")
        totals, total = vcs_tmoles(
            moles, p["basis_ordered_species_phase_ids"],
            p["basis_ordered_species_unknown_types"],
            len(p["phase_objects"]))
        activity = np.ones(nsp, dtype=float)
        phase_potentials = np.zeros(len(p["phase_objects"]), dtype=float)
        ln_mnaught = np.zeros(nsp, dtype=float)
        for iph, phase in enumerate(p["phase_objects"]):
            phase.setTotalMoles(float(totals[iph]))
            phase.setMolesOutOfDate(vcs_VolPhase._STATE_OLD)
            phase.setMolesFromVCSCheck(vcs_VolPhase._STATE_OLD, moles, totals)
            phase_potentials[iph] = phase.electricPotential()
            if phase.nSpecies() > 1:
                phase.sendToVCS_ActCoeff(
                    vcs_VolPhase._STATE_OLD, activity, moles=moles)
            convention = getattr(phase._tp, "activityConvention", None)
            convention = convention() if callable(convention) else convention
            if convention is None or str(convention).lower() in ("0", "molar"):
                convention_val = 0
            else:
                try:
                    convention_val = int(convention)
                except ValueError:
                    convention_val = 1
            if convention_val != 0 and phase.nSpecies() > 1:
                molecular_weights = np.asarray(phase._tp.molecular_weights,
                                               dtype=float)
                solvent_mnaught = float(molecular_weights[0]) / 1000.0
                if not np.isfinite(solvent_mnaught) or solvent_mnaught <= 0.0:
                    raise VCSError("invalid VCS solvent standard concentration")
                for local in range(1, phase.nSpecies()):
                    global_index = phase.spGlobalIndexVCS(local)
                    ln_mnaught[global_index] = math.log(solvent_mnaught)

        statuses = np.full(nsp, VCS_SPECIES_MAJOR, dtype=int)
        standard = p["basis_ordered_standard_gibbs_rt"]
        reaction_species = np.arange(ncomp, nsp, dtype=int)
        free_energies = vcs_dfe_species_free_energies(
            standard, ln_mnaught, moles, totals,
            p["basis_ordered_species_phase_ids"], activity,
            p["basis_ordered_species_charges"], phase_potentials, statuses,
            p["basis_ordered_species_single_phase"],
            p["basis_ordered_species_unknown_types"], reaction_species,
            ncomp, faraday_dim=Faraday / (GasConstant * p["temperature"]))
        delta_g = vcs_deltag(
            np.zeros(nsp - ncomp, dtype=float), free_energies,
            p["formation_reaction_stoichiometry"], reaction_species,
            statuses, ncomp, moles)
        state = {
            "mole_numbers_old": moles,
            "mole_numbers_new": moles.copy(),
            "phase_moles_old": totals,
            "phase_moles_new": totals.copy(),
            "total_moles_old": float(total),
            "standard_gibbs_rt": np.asarray(standard).copy(),
            "ln_mnaught_species": ln_mnaught,
            "activity_coefficients_old": activity,
            "activity_coefficients_new": activity.copy(),
            "phase_potentials_old": phase_potentials,
            "species_status": statuses,
            "species_free_energies_old": free_energies,
            "species_free_energies_new": free_energies.copy(),
            "delta_g_rxn_old": delta_g,
            "delta_g_rxn_new": delta_g.copy(),
        }
        self._old_thermodynamic_state = state
        return {key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in state.items()}

    def vcs_CalcLnActCoeffJac(self, mole_numbers_old=None):
        """Assemble the VCS activity-coefficient mole-number Jacobian.

        This follows the pinned ``vcs_CalcLnActCoeffJac`` scatter order. Ideal
        phases need no calculation; nonideal phase derivatives are requested
        from their Python thermodynamic model and fail explicitly when absent.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before Jacobian evaluation")
        p = self._prepared_problem
        nsp = len(p["species_names"])
        moles = (np.asarray(p["basis_ordered_mole_numbers"], dtype=float).copy()
                 if mole_numbers_old is None else
                 np.asarray(mole_numbers_old, dtype=float))
        if moles.shape != (nsp,) or np.any(~np.isfinite(moles)) or np.any(moles < 0.0):
            raise ValueError("VCS Jacobian mole vector has invalid dimensions or values")
        jacobian = np.zeros((nsp, nsp), dtype=float)
        for phase in p["phase_objects"]:
            if phase.nSpecies() <= 1 or phase.isIdealSoln():
                continue
            phase.setMolesFromVCS(vcs_VolPhase._STATE_OLD, moles)
            phase.sendToVCS_LnActCoeffJac(jacobian)
        return jacobian

    def vcs_species_type(self, species_index, mole_numbers, phase_moles,
                         delta_g_rxn_old):
        """Classify a species as major, minor, or excluded from VCS iteration.

        This is a direct translation of the pinned
        ``VCS_SOLVE::vcs_species_type`` decision order. The vectors are in the
        current component-basis ordering, and the method does not modify them.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before species classification")
        p = self._prepared_problem
        kspec = int(species_index)
        nsp = len(p["species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        moles = np.asarray(mole_numbers, dtype=float)
        phase_totals = np.asarray(phase_moles, dtype=float)
        delta_g = np.asarray(delta_g_rxn_old, dtype=float)
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        unknown = np.asarray(p.get("basis_ordered_species_unknown_types", np.zeros(nsp, dtype=int)), dtype=int)
        single_phase = np.asarray(p["basis_ordered_species_single_phase"], dtype=bool)
        formula = np.asarray(p["basis_ordered_formula_matrix"], dtype=float)
        elem_types = np.asarray(p["reordered_element_types"], dtype=int)
        elem_goals = np.asarray(p["reordered_element_abundance_goals"], dtype=float)
        stoich = np.asarray(p["formation_reaction_stoichiometry"], dtype=float)
        reaction_sizes = np.asarray(p["formation_reaction_size"], dtype=float)
        nphase = len(p["phase_objects"])
        nrxn = nsp - ncomp
        if (kspec < 0 or kspec >= nsp or moles.shape != (nsp,)
                or phase_totals.shape != (nphase,) or delta_g.shape != (nrxn,)
                or phase_ids.shape != (nsp,) or unknown.shape != (nsp,)
                or single_phase.shape != (nsp,) or formula.ndim != 2
                or formula.shape[0] != nsp or elem_types.shape != (formula.shape[1],)
                or elem_goals.shape != elem_types.shape or stoich.shape != (ncomp, nrxn)
                or reaction_sizes.shape != (nrxn,)):
            raise ValueError("VCS species-classification arrays have inconsistent dimensions")

        if unknown[kspec] == vcs_VolPhase._UNKNOWN_VOLTAGE:
            return VCS_SPECIES_INTERFACIALVOLTAGE

        iph = int(phase_ids[kspec])
        if iph < 0 or iph >= nphase:
            raise ValueError("VCS species has an invalid phase index")
        irxn = kspec - ncomp
        phase_exists = int(p["phase_objects"][iph].exists())

        if moles[kspec] <= 0.0:
            if phase_totals[iph] <= 0.0 and not single_phase[kspec]:
                return VCS_SPECIES_ZEROEDMS

            for jelem in range(formula.shape[1]):
                if elem_types[jelem] == vcs_VolPhase._ELEM_TYPE_ABSPOS:
                    atom_count = formula[kspec, jelem]
                    if atom_count > 0.0 and elem_goals[jelem] / atom_count < VCS_DELETE_MINORSPECIES_CUTOFF:
                        return (VCS_SPECIES_ZEROEDSS if single_phase[kspec]
                                else VCS_SPECIES_STOICHZERO)

            if irxn >= 0:
                for jcomp in range(ncomp):
                    stoic_component = stoich[jcomp, irxn]
                    if stoic_component == 0.0:
                        continue
                    component_change = -stoic_component
                    if component_change > 0.0:
                        if moles[jcomp] < 1.0e-60:
                            return (VCS_SPECIES_ZEROEDSS if single_phase[kspec]
                                    else VCS_SPECIES_STOICHZERO)
                    elif component_change < 0.0:
                        component_phase = int(phase_ids[jcomp])
                        if int(p["phase_objects"][component_phase].exists()) <= 0:
                            return (VCS_SPECIES_ZEROEDSS if single_phase[kspec]
                                    else VCS_SPECIES_STOICHZERO)

            if irxn >= 0 and delta_g[irxn] >= 0.0:
                if single_phase[kspec]:
                    return VCS_SPECIES_ZEROEDSS
                if phase_exists >= vcs_VolPhase._EXIST_YES:
                    return VCS_SPECIES_ACTIVEBUTZERO
                if phase_exists == vcs_VolPhase._EXIST_ZEROED:
                    return VCS_SPECIES_ZEROEDPHASE
                return VCS_SPECIES_ZEROEDMS

            if phase_totals[iph] > 0.0:
                return (VCS_SPECIES_MAJOR if single_phase[kspec]
                        else VCS_SPECIES_ACTIVEBUTZERO)
            return (VCS_SPECIES_MAJOR if single_phase[kspec]
                    else VCS_SPECIES_ZEROEDMS)

        if single_phase[kspec]:
            return VCS_SPECIES_MAJOR
        if moles[kspec] > phase_totals[iph] * 0.001:
            return VCS_SPECIES_MAJOR
        if irxn < 0:
            return VCS_SPECIES_MAJOR

        size_adjustment = reaction_sizes[irxn] * math.sqrt(nrxn)
        for jcomp in range(ncomp):
            if (not single_phase[jcomp] and stoich[jcomp, irxn] != 0.0
                    and moles[kspec] * size_adjustment >= moles[jcomp] * 0.01):
                return VCS_SPECIES_MAJOR
        return VCS_SPECIES_MINOR

    def vcs_evaluate_speciesType(self, mole_numbers=None, phase_moles=None,
                                 delta_g_rxn_old=None):
        """Reclassify every species and report whether all minor reactions vanish.

        Mirrors the pinned ``VCS_SOLVE::vcs_evaluate_speciesType`` loop. When
        explicit vectors are omitted, the most recently initialized OLD state
        is used; if no OLD state exists, it is constructed from the prepared
        problem first. The status vector and reduced-reaction counter are kept
        on the solver, matching the native object's owned state.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before species evaluation")
        supplied = (mole_numbers, phase_moles, delta_g_rxn_old)
        if all(value is None for value in supplied):
            state = getattr(self, "_old_thermodynamic_state", None)
            if state is None:
                state = self.initialize_thermodynamic_state()
            mole_numbers = state["mole_numbers_old"]
            phase_moles = state["phase_moles_old"]
            delta_g_rxn_old = state["delta_g_rxn_old"]
        elif any(value is None for value in supplied):
            raise ValueError(
                "VCS species evaluation requires all three state vectors together")

        p = self._prepared_problem
        nsp = len(p["species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        reduced_reaction_count = int(p.get("num_reactions_reduced", nrxn))
        moles = np.asarray(mole_numbers, dtype=float)
        phase_totals = np.asarray(phase_moles, dtype=float)
        delta_g = np.asarray(delta_g_rxn_old, dtype=float)
        if (ncomp < 0 or ncomp > nsp
                or reduced_reaction_count < 0 or reduced_reaction_count > nrxn
                or moles.shape != (nsp,)
                or phase_totals.shape != (len(p["phase_objects"]),)
                or delta_g.shape != (nrxn,)):
            raise ValueError("VCS species-evaluation vectors have inconsistent dimensions")

        status = np.empty(nsp, dtype=int)
        minor_zeroed = 0
        for kspec in range(nsp):
            status[kspec] = self.vcs_species_type(
                kspec, moles, phase_totals, delta_g)
            if kspec >= ncomp and status[kspec] != VCS_SPECIES_MAJOR:
                minor_zeroed += 1
        self._species_status = status
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is not None and np.shape(state.get("species_status")) == (nsp,):
            state["species_status"][:] = status
        self._num_rxn_minor_zeroed = minor_zeroed
        return minor_zeroed >= reduced_reaction_count

    def vcs_switch_pos(self, ifunc: bool, k1: int, k2: int) -> None:
        """Swap VCS species slots, and optionally their reaction slots.

        This translates the state permutation in pinned
        ``VCS_SOLVE::vcs_switch_pos`` (``src/equil/vcs_solve_TP.cpp``). It
        keeps the prepared Python state and each phase's local-to-global VCS
        map synchronized. The outer VCS iteration is still incomplete; this
        method only performs the source routine's permutation operation.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before switching species")
        if isinstance(k1, bool) or isinstance(k2, bool):
            raise TypeError("VCS species positions must be integer indices")
        try:
            k1, k2 = operator.index(k1), operator.index(k2)
        except TypeError as exc:
            raise TypeError("VCS species positions must be integer indices") from exc
        p = self._prepared_problem
        nsp = len(p["basis_ordered_species_names"])
        if k1 == k2:
            if not 0 <= k1 < nsp:
                raise IndexError("VCS species position is out of range")
            return
        if not (0 <= k1 < nsp and 0 <= k2 < nsp):
            raise IndexError("VCS species position is out of range")

        basis = p["component_basis"]
        ncomp = int(basis["num_components"])
        stoich = np.asarray(p["formation_reaction_stoichiometry"])
        nrxn = nsp - ncomp
        if stoich.shape != (ncomp, nrxn):
            raise VCSError("VCS formation stoichiometry has inconsistent dimensions")

        # Resolve both phase-local species before changing any state, so a
        # broken map cannot leave a half-permuted solver.
        phase_locations = {}
        for target in (k1, k2):
            matches = []
            for iph, phase in enumerate(p["phase_objects"]):
                for local in range(phase.nSpecies()):
                    if phase.spGlobalIndexVCS(local) == target:
                        matches.append((phase, local, iph))
            if len(matches) != 1:
                raise VCSError(
                    f"VCS global species slot {target} has {len(matches)} phase mappings")
            phase_locations[target] = matches[0]

        # These arrays are the Python counterparts of the species-indexed
        # members swapped by the native routine. Validate all before mutation.
        row_fields = (
            "basis_ordered_species_phase_ids",
            "basis_ordered_species_unknown_types",
            "basis_ordered_species_single_phase",
            "basis_ordered_species_charges",
            "basis_ordered_standard_gibbs_rt",
            "basis_ordered_formula_matrix",
            "basis_ordered_mole_numbers",
            "basis_ordered_species_names",
        )
        row_values = {}
        for key in row_fields:
            values = p[key]
            if len(values) != nsp:
                raise VCSError(f"VCS species field {key} has inconsistent dimensions")
            row_values[key] = values

        permutation = basis["species_permutation"]
        if len(permutation) != nsp:
            raise VCSError("VCS species permutation has inconsistent dimensions")
        if self._species_status is not None and self._species_status.shape != (nsp,):
            raise VCSError("VCS species status vector has inconsistent dimensions")

        r1, r2 = k1 - ncomp, k2 - ncomp
        if ifunc and not (0 <= r1 < nrxn and 0 <= r2 < nrxn):
            raise IndexError("VCS reaction swap requires two non-component species")
        reaction_fields = (
            p["formation_reaction_size"],
            p["formation_reaction_species"],
        )
        if ifunc and any(len(values) != nrxn for values in reaction_fields):
            raise VCSError("VCS reaction fields have inconsistent dimensions")
        phase_reaction_fields = (
            p.get("delta_mol_num_phase"), p.get("phase_participation"))
        if ifunc:
            for key, values in zip(("delta_mol_num_phase", "phase_participation"),
                                   phase_reaction_fields):
                if values is None or np.asarray(values).shape != (
                        len(p["phase_objects"]), nrxn):
                    raise VCSError(f"VCS phase-reaction field {key} has inconsistent dimensions")
        working_moles = basis.get("working_mole_numbers")
        if working_moles is not None and np.shape(working_moles) != (nsp,):
            raise VCSError("VCS basis mole workspace has inconsistent dimensions")
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is not None:
            for key in ("mole_numbers_old", "mole_numbers_new",
                        "standard_gibbs_rt", "ln_mnaught_species",
                        "activity_coefficients_old", "activity_coefficients_new",
                        "species_status", "species_free_energies_old",
                        "species_free_energies_new"):
                values = state.get(key)
                if values is not None and np.shape(values) != (nsp,):
                    raise VCSError(
                        f"VCS species field {key} has inconsistent dimensions")
            if ifunc:
                for key in ("delta_g_rxn_old", "delta_g_rxn_new",
                            "delta_g_rxn_tmp"):
                    values = state.get(key)
                    if values is not None and np.shape(values) != (nrxn,):
                        raise VCSError(
                            f"VCS reaction field {key} has inconsistent dimensions")
        delta_moles = getattr(self, "_delta_mol_num_species", None)
        if delta_moles is not None and np.shape(delta_moles) != (nsp,):
            raise VCSError("VCS species-step vector has inconsistent dimensions")

        for key, values in row_values.items():
            if isinstance(values, tuple):
                swapped = list(values)
                swapped[k1], swapped[k2] = swapped[k2], swapped[k1]
                p[key] = tuple(swapped)
            elif isinstance(values, np.ndarray) and values.ndim > 1:
                values[[k1, k2], ...] = values[[k2, k1], ...]
            else:
                values[k1], values[k2] = values[k2], values[k1]
        swapped_permutation = list(permutation)
        swapped_permutation[k1], swapped_permutation[k2] = (
            swapped_permutation[k2], swapped_permutation[k1])
        basis["species_permutation"] = tuple(swapped_permutation)
        basis["component_species"] = tuple(swapped_permutation[:ncomp])
        basis["remaining_species"] = tuple(swapped_permutation[ncomp:])
        if working_moles is not None:
            working_moles[k1], working_moles[k2] = (
                working_moles[k2], working_moles[k1])
        if self._species_status is not None:
            self._species_status[k1], self._species_status[k2] = (
                self._species_status[k2], self._species_status[k1])
        phase_locations[k1][0].setSpGlobalIndexVCS(phase_locations[k1][1], k2)
        phase_locations[k2][0].setSpGlobalIndexVCS(phase_locations[k2][1], k1)

        if ifunc:
            stoich[:, [r1, r2]] = stoich[:, [r2, r1]]
            for values in phase_reaction_fields:
                values[:, [r1, r2]] = values[:, [r2, r1]]
            p["formation_reaction_size"][r1], p["formation_reaction_size"][r2] = (
                p["formation_reaction_size"][r2], p["formation_reaction_size"][r1])
            reaction_species = list(p["formation_reaction_species"])
            reaction_species[r1], reaction_species[r2] = (
                reaction_species[r2], reaction_species[r1])
            p["formation_reaction_species"] = tuple(reaction_species)

        if state is not None:
            for key in ("mole_numbers_old", "mole_numbers_new",
                        "standard_gibbs_rt", "ln_mnaught_species",
                        "activity_coefficients_old", "activity_coefficients_new",
                        "species_status", "species_free_energies_old",
                        "species_free_energies_new"):
                values = state.get(key)
                if values is not None:
                    values[k1], values[k2] = values[k2], values[k1]
        if delta_moles is not None:
            delta_moles[k1], delta_moles[k2] = delta_moles[k2], delta_moles[k1]
        if ifunc and state is not None:
            for key in ("delta_g_rxn_old", "delta_g_rxn_new", "delta_g_rxn_tmp"):
                values = state.get(key)
                if values is not None:
                    values[r1], values[r2] = values[r2], values[r1]

    def vcs_RxnStepSizes(self, delta_g_rxn_new, *, species_status=None,
                         delta_mol_num_species=None, phase_exists=None,
                         use_activity_jacobian=False,
                         ln_act_coeff_jacobian=None):
        """Compute a VCS reaction step from the initialized OLD state.

        This is the source routine's numerical step calculation, not the full
        outer equilibrium iteration. Callers provide the current NEW reaction
        Gibbs vector; the OLD mole/activity state is taken from
        :meth:`initialize_thermodynamic_state`.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before reaction steps")
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        p = self._prepared_problem
        ncomp = int(p["component_basis"]["num_components"])
        status = (self._species_status if species_status is None
                  and self._species_status is not None else
                  state["species_status"] if species_status is None else
                  np.asarray(species_status, dtype=int))
        phase_exists_values = (np.asarray([phase.exists() for phase in p["phase_objects"]], dtype=int)
                               if phase_exists is None else
                               np.asarray(phase_exists, dtype=int))
        if use_activity_jacobian and ln_act_coeff_jacobian is None:
            ln_act_coeff_jacobian = self.vcs_CalcLnActCoeffJac(
                state["mole_numbers_old"])
        result = vcs_reaction_step_sizes(
            p["formation_reaction_stoichiometry"], delta_g_rxn_new,
            state["mole_numbers_old"], status,
            p["basis_ordered_species_unknown_types"],
            p["basis_ordered_species_single_phase"],
            p["basis_ordered_species_phase_ids"], state["phase_moles_old"],
            np.asarray([phase.nSpecies() == 1 for phase in p["phase_objects"]]),
            p["delta_mol_num_phase"], state["total_moles_old"],
            phase_exists=phase_exists_values,
            reaction_species_indices=np.arange(ncomp, len(p["species_names"])),
            delta_mol_num_species=(self._delta_mol_num_species
                                   if delta_mol_num_species is None
                                   and hasattr(self, "_delta_mol_num_species")
                                   else delta_mol_num_species),
            num_reactions_reduced=p.get("num_reactions_reduced"),
            use_activity_jacobian=use_activity_jacobian,
            ln_act_coeff_jacobian=ln_act_coeff_jacobian,
            force_component_calc=getattr(self, "_force_component_calc", 0),
            special_species=getattr(self, "_special_species", 0))
        self._delta_mol_num_species = result["delta_mol_num_species"].copy()
        self._force_component_calc = result["force_component_calc"]
        self._special_species = result["special_species"]
        self._deleted_phase = result["deleted_phase"]
        return result

    def delta_species(self, species_index, delta):
        """Apply the pinned VCS formation-reaction extent for one species.

        This ports ``VCS_SOLVE::delta_species``. A limiting component can
        shorten the requested extent; if a required reactant component is
        already absent, the operation returns failure without changing state.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before changing species")
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        p = self._prepared_problem
        result = vcs_delta_species_step(
            p["formation_reaction_stoichiometry"], state["mole_numbers_old"],
            state["phase_moles_old"],
            p["basis_ordered_species_unknown_types"],
            p["basis_ordered_species_phase_ids"], species_index, delta)
        state["mole_numbers_old"][:] = result["mole_numbers"]
        state["phase_moles_old"][:] = result["phase_moles"]
        state["total_moles_old"] = result["total_moles"]
        p["basis_ordered_mole_numbers"] = result["mole_numbers"].copy()
        for phase_index in result["invalidated_phase_ids"]:
            p["phase_objects"][phase_index].setMolesOutOfDate(
                vcs_VolPhase._STATE_OLD)
        return result

    def vcs_zero_species(self, species_index):
        """Zero one VCS species using Cantera's ``vcs_zero_species`` path.

        Voltage unknowns and species already at zero are no-ops. Otherwise,
        the native method requests an extent of ``-moles[species_index]`` and
        returns the status from ``delta_species``; this method preserves that
        sequence and return convention.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before zeroing species")
        if isinstance(species_index, bool):
            raise TypeError("VCS species index must be an integer")
        try:
            kspec = operator.index(species_index)
        except TypeError as exc:
            raise TypeError("VCS species index must be an integer") from exc
        unknown = self._prepared_problem[
            "basis_ordered_species_unknown_types"]
        if kspec < 0 or kspec >= len(unknown):
            raise IndexError("VCS species index is out of range")
        if unknown[kspec] == vcs_VolPhase._UNKNOWN_VOLTAGE:
            return 1
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        amount = float(state["mole_numbers_old"][kspec])
        if amount == 0.0:
            return 1
        return int(self.delta_species(kspec, -amount)["success"])

    def vcs_delete_species(self, species_index):
        """Remove an active noncomponent species from the VCS reduced set.

        This follows ``VCS_SOLVE::vcs_delete_species`` through the native
        species/reaction swap, active-count updates, and phase-annihilation
        transition. Phase annihilation requiring a nonzero-component permanent
        reaction rebalance is rejected before mutation because that native
        branch is not yet translated.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before deleting species")
        if isinstance(species_index, bool):
            raise TypeError("VCS species index must be an integer")
        try:
            kspec = operator.index(species_index)
        except TypeError as exc:
            raise TypeError("VCS species index must be an integer") from exc

        p = self._prepared_problem
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        active_species = int(self._num_species_reduced or nsp)
        active_reactions = int(self._num_reactions_reduced
                               if self._num_species_reduced else
                               p.get("num_reactions_reduced", nsp - ncomp))
        if kspec < ncomp or kspec >= active_species:
            raise IndexError("VCS deletion requires an active noncomponent species")
        if active_reactions != active_species - ncomp:
            raise VCSError("VCS active species and reaction counts are inconsistent")

        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        old_moles = np.asarray(state["mole_numbers_old"], dtype=float)
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        phase_index = int(phase_ids[kspec])
        phase = p["phase_objects"][phase_index]
        single_phase = np.asarray(
            p["basis_ordered_species_single_phase"], dtype=bool)
        annihilates_phase = False

        # Detect the unsupported native phase-annihilation branch without
        # modifying solver or phase state. Use the same extent logic as
        # delta_species so component-limited deletion is diagnosed correctly.
        phase_exists_always = getattr(vcs_VolPhase, "_EXIST_ALWAYS", 2)
        if (not single_phase[kspec]
                and phase.exists() != phase_exists_always):
            requested = -float(old_moles[kspec])
            preview = vcs_delta_species_step(
                p["formation_reaction_stoichiometry"], old_moles,
                state["phase_moles_old"],
                p["basis_ordered_species_unknown_types"], phase_ids,
                kspec, requested)
            if preview["success"]:
                post_moles = preview["mole_numbers"]
                still_exists = any(
                    k != kspec
                    and phase_ids[k] == phase_index
                    and p["basis_ordered_species_unknown_types"][k]
                        != vcs_VolPhase._UNKNOWN_VOLTAGE
                    and post_moles[k] > 0.0
                    for k in range(active_species))
                if not still_exists:
                    component_rebalance_needed = any(
                        k < ncomp and phase_ids[k] == phase_index
                        and post_moles[k] != 0.0
                        for k in range(ncomp))
                    if component_rebalance_needed:
                        raise NotImplementedError(
                            "VCS phase annihilation with nonzero component "
                            "moles requires the pinned permanent-reaction "
                            "rebalance")
                    annihilates_phase = True

        klast = active_species - 1
        irxn = kspec - ncomp
        result = self.vcs_zero_species(kspec)
        if not result:
            raise VCSError("VCS species could not be zeroed for deletion")

        status = (self._species_status if self._species_status is not None
                  else np.asarray(state["species_status"], dtype=int).copy())
        if status[kspec] != VCS_SPECIES_MAJOR:
            self._num_rxn_minor_zeroed -= 1
        status[kspec] = VCS_SPECIES_DELETED
        self._species_status = status
        if "species_status" in state:
            state["species_status"][kspec] = VCS_SPECIES_DELETED
        state["delta_g_rxn_new"][irxn] = 0.0
        state["delta_g_rxn_old"][irxn] = 0.0
        state["species_free_energies_new"][kspec] = 0.0
        state["species_free_energies_old"][kspec] = 0.0
        state["mole_numbers_new"][kspec] = state["mole_numbers_old"][kspec]

        if kspec != klast:
            self.vcs_switch_pos(True, klast, kspec)

        # The phase object is identified before the swap, matching the native
        # pointer retained by vcs_delete_species.
        phase.setMolesFromVCSCheck(
            vcs_VolPhase._STATE_OLD,
            state["mole_numbers_old"], state["phase_moles_old"])
        self._num_reactions_reduced -= 1
        self._num_species_reduced -= 1
        p["num_reactions_reduced"] = self._num_reactions_reduced
        p["num_species_reduced"] = self._num_species_reduced
        if annihilates_phase:
            self.vcs_delete_multiphase(phase_index)
        return int(self._num_reactions_reduced == 0)

    def vcs_delete_multiphase(self, phase_index):
        """Delete an extinct multiphase and reactivate its zeroed species.

        This ports the pinned phase-deletion loops for phases with no nonzero
        component amounts. The native permanent-reaction rebalance for a
        residual nonzero component is explicitly unsupported and rejected
        before mutation.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before deleting a phase")
        p = self._prepared_problem
        phase_index = operator.index(phase_index)
        if phase_index < 0 or phase_index >= len(p["phase_objects"]):
            raise IndexError("VCS phase index is out of range")
        phase = p["phase_objects"][phase_index]
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        ncomp = int(p["component_basis"]["num_components"])
        nsp = len(p["basis_ordered_species_names"])
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        unknown = np.asarray(p.get("basis_ordered_species_unknown_types", np.zeros(nsp, dtype=int)), dtype=int)
        old_moles = state["mole_numbers_old"]
        if any(phase_ids[k] == phase_index and old_moles[k] != 0.0
               for k in range(ncomp)):
            raise NotImplementedError(
                "VCS phase deletion with nonzero component moles requires "
                "the pinned permanent-reaction rebalance")

        active = int(self._num_species_reduced)
        phase.setTotalMoles(0.0)
        if self._delta_mol_num_species is None:
            self._delta_mol_num_species = np.zeros(nsp, dtype=float)
        if self._delta_phase_moles is None:
            self._delta_phase_moles = np.zeros(len(p["phase_objects"]), dtype=float)
        successful = True

        # The pinned routine visits every active noncomponent in the extinct
        # phase, including zero-mole species, and applies its native extent.
        for kspec in range(ncomp, active):
            if phase_ids[kspec] == phase_index and unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                extent = -float(old_moles[kspec])
                result = self.delta_species(kspec, extent)
                if result["success"] != 1:
                    successful = False
                    state["mole_numbers_old"][kspec] = 0.0
                    state["mole_numbers_new"][kspec] = 0.0
                    self._delta_mol_num_species[kspec] = 0.0
                    totals, total = vcs_tmoles(
                        state["mole_numbers_old"], phase_ids, unknown,
                        len(p["phase_objects"]))
                    state["phase_moles_old"][:] = totals
                    state["total_moles_old"] = total
                else:
                    state["mole_numbers_old"][kspec] = 0.0
                    state["mole_numbers_new"][kspec] = 0.0
                    self._delta_mol_num_species[kspec] = 0.0
                if self._species_status is not None:
                    self._species_status[kspec] = VCS_SPECIES_ZEROEDMS
                if "species_status" in state:
                    state["species_status"][kspec] = VCS_SPECIES_ZEROEDMS

        for kcomp in range(ncomp):
            if phase_ids[kcomp] == phase_index:
                state["mole_numbers_old"][kcomp] = 0.0
                if self._species_status is not None:
                    self._species_status[kcomp] = VCS_SPECIES_ZEROEDMS
                if "species_status" in state:
                    state["species_status"][kcomp] = VCS_SPECIES_ZEROEDMS

        kspec = active
        while kspec < nsp:
            if phase_ids[kspec] == phase_index:
                state["mole_numbers_old"][kspec] = 0.0
                state["mole_numbers_new"][kspec] = 0.0
                self._delta_mol_num_species[kspec] = 0.0
                if self._species_status is not None:
                    self._species_status[kspec] = VCS_SPECIES_ZEROEDMS
                if "species_status" in state:
                    state["species_status"][kspec] = VCS_SPECIES_ZEROEDMS
                self._num_reactions_reduced += 1
                self._num_species_reduced += 1
                active = self._num_species_reduced
                p["num_reactions_reduced"] = self._num_reactions_reduced
                p["num_species_reduced"] = self._num_species_reduced
                if kspec != active - 1:
                    self.vcs_switch_pos(True, active - 1, kspec)
                # The native for-loop advances kspec after the slot swap.
            kspec += 1

        state["phase_moles_old"][phase_index] = 0.0
        state["phase_moles_new"][phase_index] = 0.0
        self._delta_phase_moles[phase_index] = 0.0
        phase.setTotalMoles(0.0)
        return successful

    def vcs_reinsert_deleted(self, species_index):
        """Reinsert one inactive noncomponent species into the reduced set.

        This follows ``VCS_SOLVE::vcs_reinsert_deleted``: seed the species by
        the phase-relative cutoff, classify it as minor (or major for a
        single-species phase), refresh its phase cache, restore phase
        existence when possible, update active counters, and swap it into the
        last active slot. The outer phase-birth stability scan remains a
        caller responsibility.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before reinserting species")
        if isinstance(species_index, bool):
            raise TypeError("VCS species index must be an integer")
        try:
            kspec = operator.index(species_index)
        except TypeError as exc:
            raise TypeError("VCS species index must be an integer") from exc

        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        active = int(self._num_species_reduced)
        if kspec < active or kspec >= nsp or kspec < ncomp:
            raise IndexError("VCS reinsertion requires an inactive noncomponent species")
        if int(self._num_reactions_reduced) != active - ncomp:
            raise VCSError("VCS active species and reaction counts are inconsistent")

        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        phase_index = int(phase_ids[kspec])
        phase = p["phase_objects"][phase_index]
        phase_moles = np.asarray(state["phase_moles_old"], dtype=float)
        # The native extent is a tiny positive phase-relative seed. For an
        # absent phase this evaluates to zero, matching the source expression.
        delta = (float(phase_moles[phase_index])
                 * VCS_RELDELETE_SPECIES_CUTOFF * 10.0)
        self.delta_species(kspec, delta)

        status = (self._species_status if self._species_status is not None
                  else np.asarray(state["species_status"], dtype=int).copy())
        status[kspec] = VCS_SPECIES_MINOR
        single_phase = np.asarray(
            p["basis_ordered_species_single_phase"], dtype=bool)
        if single_phase[kspec]:
            status[kspec] = VCS_SPECIES_MAJOR
            self._num_rxn_minor_zeroed -= 1
        self._species_status = status
        if "species_status" in state:
            state["species_status"][kspec] = status[kspec]

        phase.setMolesFromVCSCheck(
            vcs_VolPhase._STATE_OLD,
            state["mole_numbers_old"], state["phase_moles_old"])
        if not single_phase[kspec] and phase.exists() == vcs_VolPhase._EXIST_NO:
            phase.setExistence(vcs_VolPhase._EXIST_YES)
            for k in range(nsp):
                if (phase_ids[k] == phase_index
                        and status[k] != VCS_SPECIES_DELETED):
                    status[k] = VCS_SPECIES_MINOR
                    if "species_status" in state:
                        state["species_status"][k] = VCS_SPECIES_MINOR
        elif single_phase[kspec]:
            phase.setExistence(vcs_VolPhase._EXIST_YES)

        self._num_reactions_reduced += 1
        self._num_species_reduced += 1
        p["num_reactions_reduced"] = self._num_reactions_reduced
        p["num_species_reduced"] = self._num_species_reduced
        self._num_rxn_minor_zeroed += 1
        if kspec != self._num_species_reduced - 1:
            self.vcs_switch_pos(True, self._num_species_reduced - 1, kspec)

    def vcs_recheck_deleted(self):
        """Recheck inactive species for phase or species reappearance.

        This translates ``VCS_SOLVE::vcs_recheck_deleted`` through its
        free-energy reconstruction, formation-reaction Gibbs evaluation, and
        insertion decision. The calling TP iteration and its thermodynamic
        refresh remain outside this method.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before rechecking species")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        active_species = int(self._num_species_reduced)
        active_reactions = int(self._num_reactions_reduced)
        if (not ncomp <= active_species <= nsp
                or active_reactions != active_species - ncomp):
            raise VCSError("VCS active species and reaction counts are inconsistent")
        if active_species == nsp:
            return 0

        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        reaction_species = np.asarray(p["formation_reaction_species"], dtype=int)
        if reaction_species.shape != (nrxn,):
            raise VCSError("VCS reaction-to-species map has inconsistent dimensions")
        phase_moles_old = np.asarray(state["phase_moles_old"], dtype=float)
        activities_old = np.asarray(state["activity_coefficients_old"], dtype=float)
        free_new = np.asarray(state["species_free_energies_new"], dtype=float)
        standard = np.asarray(state["standard_gibbs_rt"], dtype=float)
        ln_mnaught = np.asarray(state["ln_mnaught_species"], dtype=float)
        charges = np.asarray(p["basis_ordered_species_charges"], dtype=float)
        potentials = np.asarray(state["phase_potentials_old"], dtype=float)
        if (phase_ids.shape != (nsp,) or activities_old.shape != (nsp,)
                or free_new.shape != (nsp,) or standard.shape != (nsp,)
                or ln_mnaught.shape != (nsp,) or charges.shape != (nsp,)
                or phase_moles_old.shape != (len(p["phase_objects"]),)
                or potentials.shape != phase_moles_old.shape):
            raise VCSError("VCS deleted-species state vectors have inconsistent dimensions")

        # The native expression is evaluated in IEEE floating-point arithmetic;
        # numpy.log preserves -inf for a zero activity as C++ log does.
        for kspec in range(active_species, nsp):
            iph = int(phase_ids[kspec])
            with np.errstate(divide="ignore", invalid="ignore"):
                free_new[kspec] = (
                    standard[kspec] + np.log(activities_old[kspec])
                    - ln_mnaught[kspec]
                    + charges[kspec] * Faraday / (GasConstant * p["temperature"])
                    * potentials[iph])

        delta_g_new = vcs_deltag(
            state["delta_g_rxn_new"], free_new,
            p["formation_reaction_stoichiometry"], reaction_species,
            state["species_status"], ncomp, state["mole_numbers_new"],
            L=0, do_deleted=True, num_reactions_reduced=active_reactions)
        state["delta_g_rxn_new"][:] = delta_g_new

        phase_cutoffs = np.zeros(len(p["phase_objects"]), dtype=float)
        live_phase_cutoff = math.log(1.0 / VCS_RELDELETE_SPECIES_CUTOFF)
        for iph, total in enumerate(phase_moles_old):
            if total > 0.0:
                phase_cutoffs[iph] = live_phase_cutoff

        added = 0
        # Match the native loop's moving active boundary and reaction-slot
        # mapping: each successful reinsertion changes the next loop index.
        irxn = active_reactions
        while irxn < nrxn:
            kspec = int(reaction_species[irxn])
            iph = int(phase_ids[kspec])
            amount = float(phase_moles_old[iph])
            dg = float(delta_g_new[irxn])
            if amount == 0.0:
                if dg < 0.0:
                    self.vcs_reinsert_deleted(kspec)
                    added += 1
                    active_reactions = int(self._num_reactions_reduced)
                    reaction_species = np.asarray(
                        p["formation_reaction_species"], dtype=int)
                    delta_g_new = state["delta_g_rxn_new"]
                    phase_ids = np.asarray(
                        p["basis_ordered_species_phase_ids"], dtype=int)
                else:
                    state["mole_numbers_old"][kspec] = 0.0
            elif amount > 0.0 and dg < phase_cutoffs[iph]:
                self.vcs_reinsert_deleted(kspec)
                added += 1
                active_reactions = int(self._num_reactions_reduced)
                reaction_species = np.asarray(
                    p["formation_reaction_species"], dtype=int)
                delta_g_new = state["delta_g_rxn_new"]
                phase_ids = np.asarray(
                    p["basis_ordered_species_phase_ids"], dtype=int)
            irxn += 1
        return added

    def vcs_add_all_deleted(self):
        """Estimate and apply additions for all inactive species.

        This ports the pinned ``VCS_SOLVE::vcs_add_all_deleted`` sequence:
        three trial activity/free-energy updates, bounded reaction extents,
        a refresh of the OLD thermodynamic state, and the native minor-species
        convergence count. It does not run the surrounding TP iteration.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before adding species")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        active_species = int(self._num_species_reduced)
        active_reactions = int(self._num_reactions_reduced)
        if (not ncomp <= active_species <= nsp
                or active_reactions != active_species - ncomp):
            raise VCSError("VCS active species and reaction counts are inconsistent")
        if active_species == nsp:
            return 0

        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        unknown = np.asarray(p["basis_ordered_species_unknown_types"], dtype=int)
        single = np.asarray(p["basis_ordered_species_single_phase"], dtype=bool)
        reaction_species = np.asarray(p["formation_reaction_species"], dtype=int)
        stoich = np.asarray(p["formation_reaction_stoichiometry"], dtype=float)
        charges = np.asarray(p["basis_ordered_species_charges"], dtype=float)
        standard = np.asarray(p["basis_ordered_standard_gibbs_rt"], dtype=float)
        ln_mnaught = np.asarray(state["ln_mnaught_species"], dtype=float)
        phase_potentials = np.asarray(state["phase_potentials_old"], dtype=float)
        phase_moles_old = np.asarray(state["phase_moles_old"], dtype=float)
        mole_old = np.asarray(state["mole_numbers_old"], dtype=float)
        mole_new = np.asarray(state["mole_numbers_new"], dtype=float)
        activity_old = np.asarray(state["activity_coefficients_old"], dtype=float)
        activity_new = np.asarray(state["activity_coefficients_new"], dtype=float)
        free_old = np.asarray(state["species_free_energies_old"], dtype=float)
        free_new = np.asarray(state["species_free_energies_new"], dtype=float)
        status = np.asarray(state["species_status"], dtype=int)
        delta_old = np.asarray(state["delta_g_rxn_old"], dtype=float)
        delta_new = np.asarray(state["delta_g_rxn_new"], dtype=float)
        nphase = len(p["phase_objects"])
        if (phase_ids.shape != (nsp,) or unknown.shape != (nsp,)
                or single.shape != (nsp,) or reaction_species.shape != (nrxn,)
                or stoich.shape != (ncomp, nrxn) or charges.shape != (nsp,)
                or standard.shape != (nsp,) or ln_mnaught.shape != (nsp,)
                or phase_potentials.shape != (nphase,)
                or phase_moles_old.shape != (nphase,)
                or mole_old.shape != (nsp,) or mole_new.shape != (nsp,)
                or activity_old.shape != (nsp,) or activity_new.shape != (nsp,)
                or free_old.shape != (nsp,) or free_new.shape != (nsp,)
                or status.shape != (nsp,) or delta_old.shape != (nrxn,)
                or delta_new.shape != (nrxn,)):
            raise VCSError("VCS deleted-species vectors have inconsistent dimensions")

        # Native begins from the OLD composition and assigns a tiny nonzero
        # trial amount to inactive species that are exactly zero.
        mole_new[:] = mole_old
        for kspec in range(active_species, nsp):
            if mole_new[kspec] == 0.0:
                mole_new[kspec] = VCS_DELETE_MINORSPECIES_CUTOFF * 1.0e-10

        faraday_dim = Faraday / (GasConstant * float(p["temperature"]))
        for _ in range(3):
            totals_new, _ = vcs_tmoles(mole_new, phase_ids, unknown, nphase)
            for iph, phase in enumerate(p["phase_objects"]):
                phase.setTotalMoles(float(totals_new[iph]))
                phase.setMolesOutOfDate(vcs_VolPhase._STATE_NEW)
                phase.setMolesFromVCSCheck(
                    vcs_VolPhase._STATE_NEW, mole_new, totals_new)
                if phase.nSpecies() > 1:
                    phase.sendToVCS_ActCoeff(
                        vcs_VolPhase._STATE_NEW, activity_new, moles=mole_new)
            for kspec in range(active_species, nsp):
                iph = int(phase_ids[kspec])
                with np.errstate(divide="ignore", invalid="ignore"):
                    free_new[kspec] = (
                        standard[kspec] + np.log(activity_new[kspec])
                        - ln_mnaught[kspec]
                        + charges[kspec] * faraday_dim * phase_potentials[iph])
            delta_new[:] = vcs_deltag(
                delta_new, free_new, stoich, reaction_species, status,
                ncomp, mole_new, L=0, do_deleted=True,
                num_reactions_reduced=active_reactions)
            for irxn in range(active_reactions, nrxn):
                kspec = int(reaction_species[irxn])
                iph = int(phase_ids[kspec])
                if phase_moles_old[iph] > 0.0:
                    max_dg = min(float(delta_new[irxn]), 690.0)
                    with np.errstate(over="ignore", invalid="ignore"):
                        mole_new[kspec] = phase_moles_old[iph] * np.exp(-max_dg)

        # Apply each trial amount as an extent in native reaction-slot order.
        for irxn in range(active_reactions, nrxn):
            kspec = int(reaction_species[irxn])
            iph = int(phase_ids[kspec])
            if phase_moles_old[iph] > 0.0:
                extent = float(mole_new[kspec])
                result = self.delta_species(kspec, extent)
                if not result["success"] and result["delta"] > 1.0e-50:
                    result = self.delta_species(kspec, 1.0e-50)

        # Re-evaluate old-state activity coefficients and chemical potentials
        # after the extents, following vcs_setFlagsVolPhases(false, OLD) and
        # vcs_dfe(OLD, 0, 0, nsp).
        totals_old, total_old = vcs_tmoles(mole_old, phase_ids, unknown, nphase)
        for iph, phase in enumerate(p["phase_objects"]):
            phase.setTotalMoles(float(totals_old[iph]))
            phase.setMolesOutOfDate(vcs_VolPhase._STATE_OLD)
            phase.setMolesFromVCSCheck(
                vcs_VolPhase._STATE_OLD, mole_old, totals_old)
            if phase.nSpecies() > 1:
                phase.sendToVCS_ActCoeff(
                    vcs_VolPhase._STATE_OLD, activity_old, moles=mole_old)
            phase_potentials[iph] = phase.electricPotential()
        state["phase_moles_old"][:] = totals_old
        state["total_moles_old"] = float(total_old)
        free_old[:] = vcs_dfe_species_free_energies(
            standard, ln_mnaught, mole_old, totals_old, phase_ids,
            activity_old, charges, phase_potentials, status, single, unknown,
            reaction_species, ncomp, faraday_dim=faraday_dim, L=0,
            initial_free_energies=free_old)
        delta_old[:] = vcs_deltag(
            delta_old, free_old, stoich, reaction_species, status, ncomp,
            mole_old, L=0, do_deleted=True,
            num_reactions_reduced=active_reactions)

        remaining = 0
        for irxn in range(active_reactions, nrxn):
            kspec = int(reaction_species[irxn])
            iph = int(phase_ids[kspec])
            if (totals_old[iph] > 0.0
                    and abs(float(delta_old[irxn])) > self._tolmin):
                with np.errstate(over="ignore", invalid="ignore"):
                    predicted = mole_old[kspec] * np.exp(-delta_old[irxn])
                if (predicted > VCS_DELETE_MINORSPECIES_CUTOFF
                        or mole_old[kspec] > VCS_DELETE_MINORSPECIES_CUTOFF):
                    remaining += 1
        return remaining

    def vcs_glob_step_damp(self):
        """Apply Cantera's parabolic global-step damping when slopes turn up.

        The native routine compares OLD and NEW Gibbs slopes along the current
        species-step vector. It leaves the trial untouched when the step is
        already descending, the fitted factor is outside the accepted range,
        or the fit is singular. Otherwise it interpolates NEW moles and
        recomputes trial chemical potentials and reaction Gibbs changes.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before step damping")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        active_species = int(self._num_species_reduced)
        active_reactions = int(self._num_reactions_reduced)
        nrxn = nsp - ncomp
        if (not ncomp <= active_species <= nsp
                or active_reactions != active_species - ncomp):
            raise VCSError("VCS active species and reaction counts are inconsistent")
        delta_moles = self._delta_mol_num_species
        if delta_moles is None:
            raise VCSError("VCS species-step vector has not been calculated")
        delta_moles = np.asarray(delta_moles, dtype=float)
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        unknown = np.asarray(p["basis_ordered_species_unknown_types"], dtype=int)
        if (delta_moles.shape != (nsp,) or phase_ids.shape != (nsp,)
                or unknown.shape != (nsp,)):
            raise VCSError("VCS step-damping vectors have inconsistent dimensions")

        dg_old = np.asarray(state["delta_g_rxn_old"], dtype=float)
        dg_new = np.asarray(state["delta_g_rxn_new"], dtype=float)
        if dg_old.shape != (nrxn,) or dg_new.shape != (nrxn,):
            raise VCSError("VCS reaction Gibbs vectors have inconsistent dimensions")
        slope_new = 0.0
        slope_old = 0.0
        for irxn in range(active_reactions):
            kspec = irxn + ncomp
            if unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                slope_new += float(dg_new[irxn]) * float(delta_moles[kspec])
                slope_old += float(dg_old[irxn]) * float(delta_moles[kspec])
        if slope_old > 0.0 or slope_new <= 0.0:
            return False

        damping = 1.0
        difference = slope_old - slope_new
        if abs(difference) > 1.0e-200:
            damping = slope_old / difference
        if damping >= 0.95 or damping < 0.0:
            return False

        mole_old = np.asarray(state["mole_numbers_old"], dtype=float)
        mole_new = np.asarray(state["mole_numbers_new"], dtype=float)
        phase_old = np.asarray(state["phase_moles_old"], dtype=float)
        phase_new = np.asarray(state["phase_moles_new"], dtype=float)
        nphase = len(p["phase_objects"])
        if (mole_old.shape != (nsp,) or mole_new.shape != (nsp,)
                or phase_old.shape != (nphase,) or phase_new.shape != (nphase,)):
            raise VCSError("VCS damped-state vectors have inconsistent dimensions")
        delta_phase = self._delta_phase_moles
        if delta_phase is None:
            delta_phase = np.zeros(nphase, dtype=float)
            for k in range(nsp):
                if unknown[k] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                    delta_phase[phase_ids[k]] += delta_moles[k]
            self._delta_phase_moles = delta_phase
        delta_phase = np.asarray(delta_phase, dtype=float)
        if delta_phase.shape != (nphase,):
            raise VCSError("VCS phase-step vector has inconsistent dimensions")

        mole_new[:active_species] = (
            mole_old[:active_species] + damping * delta_moles[:active_species])
        phase_new[:] = phase_old + damping * delta_phase
        totals, _ = vcs_tmoles(mole_new, phase_ids, unknown, nphase)
        if any(not vcs_doubleEqual(float(a), float(b))
               for a, b in zip(totals, phase_new)):
            raise VCSError("VCS damped phase totals disagree with species moles")
        state["total_moles_new"] = sum(float(x) for x in phase_new)

        # Refresh NEW phase compositions and activity coefficients exactly for
        # the interpolated trial, then evaluate the active species potentials.
        activity_new = np.asarray(state["activity_coefficients_new"], dtype=float)
        for iph, phase in enumerate(p["phase_objects"]):
            phase.setTotalMoles(float(phase_new[iph]))
            phase.setMolesOutOfDate(vcs_VolPhase._STATE_NEW)
            phase.setMolesFromVCSCheck(
                vcs_VolPhase._STATE_NEW, mole_new, phase_new)
            if phase.nSpecies() > 1:
                phase.sendToVCS_ActCoeff(
                    vcs_VolPhase._STATE_NEW, activity_new, moles=mole_new)
        phase_potentials = np.asarray(state["phase_potentials_old"], dtype=float).copy()
        for iph, phase in enumerate(p["phase_objects"]):
            phase_potentials[iph] = phase.electricPotential()
        state["phase_potentials_new"] = phase_potentials
        standard = np.asarray(p["basis_ordered_standard_gibbs_rt"], dtype=float)
        free_new = np.asarray(state["species_free_energies_new"], dtype=float)
        free_new[:] = vcs_dfe_species_free_energies(
            standard, state["ln_mnaught_species"], mole_new, phase_new,
            phase_ids, activity_new, p["basis_ordered_species_charges"],
            phase_potentials, state["species_status"],
            p["basis_ordered_species_single_phase"], unknown,
            p["formation_reaction_species"], ncomp,
            faraday_dim=Faraday / (GasConstant * float(p["temperature"])),
            L=0, lbot=0, ltop=active_species,
            initial_free_energies=free_new)
        dg_new[:] = vcs_deltag(
            dg_new, free_new, p["formation_reaction_stoichiometry"],
            p["formation_reaction_species"], state["species_status"],
            ncomp, mole_new, L=0, do_deleted=False,
            num_reactions_reduced=active_reactions)

        # The native return value indicates that the interpolated trial was
        # applied; it does not reject the trial based on its adjusted slope.
        return True

    def vcs_minor_alt_calc(self, species_index, reaction_index,
                           ln_act_coeff_jacobian=None):
        """Calculate the pinned alternative minor-species mole-number step.

        Returns ``(delta_moles, delete_species)`` corresponding to the C++
        output-reference pair. The routine fits a local power law to the
        activity coefficient, clips the inferred exponent, limits extreme
        updates, and applies Cantera's small-species deletion cutoff.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before minor steps")
        if isinstance(species_index, bool) or isinstance(reaction_index, bool):
            raise TypeError("VCS species and reaction indices must be integers")
        try:
            kspec = operator.index(species_index)
            irxn = operator.index(reaction_index)
        except TypeError as exc:
            raise TypeError("VCS species and reaction indices must be integers") from exc
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        if kspec < ncomp or kspec >= nsp:
            raise IndexError("VCS minor step requires a noncomponent species")
        if irxn < 0 or irxn >= nrxn:
            raise IndexError("VCS minor-step reaction index is out of range")
        moles = np.asarray(state["mole_numbers_old"], dtype=float)
        phase_totals = np.asarray(state["phase_moles_old"], dtype=float)
        delta_g = np.asarray(state["delta_g_rxn_old"], dtype=float)
        unknown = np.asarray(p["basis_ordered_species_unknown_types"], dtype=int)
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        if (moles.shape != (nsp,) or unknown.shape != (nsp,)
                or phase_ids.shape != (nsp,) or phase_totals.shape !=
                (len(p["phase_objects"]),) or delta_g.shape != (nrxn,)):
            raise VCSError("VCS minor-step vectors have inconsistent dimensions")

        do_delete = False
        if unknown[kspec] == vcs_VolPhase._UNKNOWN_VOLTAGE:
            faraday_dim = Faraday / (GasConstant * float(p["temperature"]))
            return float(delta_g[irxn] / faraday_dim), False

        amount = float(moles[kspec])
        if amount <= 0.0:
            amount = VCS_DELETE_MINORSPECIES_CUTOFF
        dg = max(float(delta_g[irxn]), -200.0)
        if dg >= 23.0:
            trial = amount * 1.0e-10
            if amount < VCS_DELETE_MINORSPECIES_CUTOFF:
                return -amount, True
            return trial - amount, False
        if abs(dg) <= self._tolmin2:
            return 0.0, False

        if ln_act_coeff_jacobian is None:
            ln_act_coeff_jacobian = self.vcs_CalcLnActCoeffJac(moles)
        jacobian = np.asarray(ln_act_coeff_jacobian, dtype=float)
        if jacobian.shape != (nsp, nsp):
            raise ValueError("VCS activity-coefficient Jacobian has the wrong shape")
        iph = int(phase_ids[kspec])
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = float(np.divide(jacobian[kspec, kspec], phase_totals[iph]))
        exponent = float(np.clip(amount * slope, -1.0 + 1.0e-8, 100.0))
        trial_exponent = float(np.clip(-dg / (1.0 + exponent), -200.0, 200.0))
        trial = amount * math.exp(trial_exponent)

        if trial > 100.0 * amount:
            maximum = 0.0001 * float(phase_totals[iph])
            if maximum < 100.0 * amount:
                maximum = 100.0 * amount
            trial = min(trial, maximum)
        elif 1.0e10 * trial < amount:
            trial = 1.0e-10 * amount

        if trial < VCS_DELETE_MINORSPECIES_CUTOFF:
            do_delete = True
            return -amount, do_delete
        return trial - amount, do_delete

    def vcs_apply_minor_trial(self, species_index, reaction_index, *,
                              iteration=0, ln_act_coeff_jacobian=None):
        """Apply the minor-species branch of pinned ``solve_tp_inner``.

        This applies the source's iteration-zero restriction, alternative
        minor-species estimate, deletion/recheck transition, and component and
        phase extent accumulation. It covers this branch only; the enclosing
        TP active-set state machine remains unported.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before applying a minor trial")
        if isinstance(iteration, bool):
            raise TypeError("VCS iteration index must be an integer")
        try:
            iteration = operator.index(iteration)
        except TypeError as exc:
            raise TypeError("VCS iteration index must be an integer") from exc
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        kspec, irxn = operator.index(species_index), operator.index(reaction_index)
        if kspec < ncomp or kspec >= nsp or irxn < 0 or irxn >= nrxn:
            raise IndexError("VCS minor trial species or reaction index is out of range")
        reaction_species = np.asarray(p["formation_reaction_species"], dtype=int)
        if reaction_species.shape != (nrxn,) or reaction_species[irxn] != kspec:
            raise VCSError("VCS minor trial reaction does not form the requested species")
        status = (self._species_status if self._species_status is not None else
                  np.asarray(state["species_status"], dtype=int))
        self._species_status = status
        if status[kspec] != VCS_SPECIES_MINOR:
            raise VCSError("VCS minor trial requires a minor-status species")

        mole_old = np.asarray(state["mole_numbers_old"], dtype=float)
        mole_new = np.asarray(state["mole_numbers_new"], dtype=float)
        if mole_old.shape != (nsp,) or mole_new.shape != (nsp,):
            raise VCSError("VCS minor trial mole vectors have inconsistent dimensions")
        if self._delta_mol_num_species is None:
            self._delta_mol_num_species = np.zeros(nsp, dtype=float)
        delta_moles = np.asarray(self._delta_mol_num_species, dtype=float)
        if delta_moles.shape != (nsp,):
            raise VCSError("VCS species-step vector has inconsistent dimensions")

        if iteration != 0:
            mole_new[kspec] = mole_old[kspec]
            delta_moles[kspec] = 0.0
            return {"delta": 0.0, "deleted": False,
                    "recheck_required": False, "skipped_minor": True}

        delta, do_delete = self.vcs_minor_alt_calc(
            kspec, irxn, ln_act_coeff_jacobian)
        delta_moles[kspec] = delta
        mole_new[kspec] = mole_old[kspec] + delta
        if do_delete:
            delta_moles[kspec] = 0.0
            recheck = self.vcs_delete_species(kspec)
            return {"delta": delta, "deleted": True,
                    "recheck_required": bool(recheck), "skipped_minor": False}

        stoich = np.asarray(p["formation_reaction_stoichiometry"], dtype=float)
        phase_changes = np.asarray(p["delta_mol_num_phase"], dtype=float)
        nphase = len(p["phase_objects"])
        if stoich.shape != (ncomp, nrxn) or phase_changes.shape != (nphase, nrxn):
            raise VCSError("VCS minor trial stoichiometry has inconsistent dimensions")
        delta_moles[:ncomp] += stoich[:, irxn] * delta
        if self._delta_phase_moles is None:
            self._delta_phase_moles = np.zeros(nphase, dtype=float)
        delta_phase = np.asarray(self._delta_phase_moles, dtype=float)
        if delta_phase.shape != (nphase,):
            raise VCSError("VCS phase-step vector has inconsistent dimensions")
        delta_phase[:] += phase_changes[:, irxn] * delta
        return {"delta": float(delta), "deleted": False,
                "recheck_required": False, "skipped_minor": False}

    def vcs_apply_major_trial(self, species_index, reaction_index):
        """Apply the pinned major-species branch of ``solve_tp_inner``.

        This covers superconvergence, reaction-step sign consistency,
        nonnegative mole limits for multispecies phases, and the native
        single-species phase-zero transition. The outer TP stage machine must
        still process a returned phase-deletion request.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before applying a major trial")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        kspec, irxn = operator.index(species_index), operator.index(reaction_index)
        if kspec < ncomp or kspec >= nsp or irxn < 0 or irxn >= nrxn:
            raise IndexError("VCS major trial species or reaction index is out of range")
        reaction_species = np.asarray(p["formation_reaction_species"], dtype=int)
        if reaction_species.shape != (nrxn,) or reaction_species[irxn] != kspec:
            raise VCSError("VCS major trial reaction does not form the requested species")
        status = (self._species_status if self._species_status is not None else
                  np.asarray(state["species_status"], dtype=int))
        if status[kspec] != VCS_SPECIES_MAJOR:
            raise VCSError("VCS major trial requires a major-status species")
        unknown = np.asarray(p["basis_ordered_species_unknown_types"], dtype=int)
        single = np.asarray(p["basis_ordered_species_single_phase"], dtype=bool)
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        stoich = np.asarray(p["formation_reaction_stoichiometry"], dtype=float)
        phase_changes = np.asarray(p["delta_mol_num_phase"], dtype=float)
        phase_objects = p["phase_objects"]
        nphase = len(phase_objects)
        if (unknown.shape != (nsp,) or single.shape != (nsp,)
                or phase_ids.shape != (nsp,) or stoich.shape != (ncomp, nrxn)
                or phase_changes.shape != (nphase, nrxn)):
            raise VCSError("VCS major trial metadata has inconsistent dimensions")
        mole_old = np.asarray(state["mole_numbers_old"], dtype=float)
        mole_new = np.asarray(state["mole_numbers_new"], dtype=float)
        delta_g_new = np.asarray(state["delta_g_rxn_new"], dtype=float)
        if (mole_old.shape != (nsp,) or mole_new.shape != (nsp,)
                or delta_g_new.shape != (nrxn,)):
            raise VCSError("VCS major trial state vectors have inconsistent dimensions")
        if self._delta_mol_num_species is None:
            self._delta_mol_num_species = np.zeros(nsp, dtype=float)
        delta_moles = np.asarray(self._delta_mol_num_species, dtype=float)
        if delta_moles.shape != (nsp,):
            raise VCSError("VCS species-step vector has inconsistent dimensions")
        if self._delta_phase_moles is None:
            self._delta_phase_moles = np.zeros(nphase, dtype=float)
        delta_phase = np.asarray(self._delta_phase_moles, dtype=float)
        if delta_phase.shape != (nphase,):
            raise VCSError("VCS phase-step vector has inconsistent dimensions")

        if abs(float(delta_g_new[irxn])) <= self._tolmaj2:
            mole_new[kspec] = mole_old[kspec]
            delta_moles[kspec] = 0.0
            return {"delta": 0.0, "phase_delete": None,
                    "all_minor_zeroed": False, "superconverged": True}

        delta = float(delta_moles[kspec])
        if float(delta_g_new[irxn]) * delta > 0.0:
            delta = 0.0
            delta_moles[kspec] = 0.0
        mole_new[kspec] = mole_old[kspec] + delta
        phase_delete = None
        all_minor_zeroed = False
        if mole_new[kspec] <= 0.0:
            if not single[kspec]:
                delta = -0.9 * float(mole_old[kspec])
                delta_moles[kspec] = delta
                mole_new[kspec] = mole_old[kspec] + delta
            else:
                delta = -float(mole_old[kspec])
                for kcomp in range(ncomp):
                    coefficient = float(stoich[kcomp, irxn])
                    if coefficient != 0.0:
                        trial_component = (float(mole_old[kcomp])
                                           + coefficient * delta)
                        if trial_component <= float(mole_old[kcomp]) * 0.01 - 1.0e-150:
                            delta = max(
                                delta,
                                float(mole_old[kcomp]) * -0.99 / coefficient)
                mole_new[kspec] = mole_old[kspec] + delta
                if mole_new[kspec] > 0.0:
                    delta_moles[kspec] = delta
                else:
                    mole_new[kspec] = 0.0
                    phase_delete = int(phase_ids[kspec])
                    status[kspec] = VCS_SPECIES_ZEROEDSS
                    self._num_rxn_minor_zeroed += 1
                    all_minor_zeroed = (
                        self._num_rxn_minor_zeroed == self._num_reactions_reduced)
                    delta_moles[:] = 0.0
                    mole_new[:] = mole_old
                    delta_moles[kspec] = delta
                    mole_new[kspec] = 0.0
                    delta_moles[:ncomp] = 0.0
                    delta_phase[:] = 0.0
                    self._species_status = status
                    if "species_status" in state:
                        state["species_status"][kspec] = VCS_SPECIES_ZEROEDSS

        if delta != 0.0 and unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
            if abs(float(delta_moles[kspec]) - delta) >= 1.0e-14 * (
                    abs(float(delta_moles[kspec])) + abs(delta) + 1.0e-32):
                raise VCSError("VCS major trial species extent is inconsistent")
            delta_moles[:ncomp] += stoich[:, irxn] * delta
            delta_phase[:] += phase_changes[:, irxn] * delta

        # Match solve_tp_inner::checkDelta1: phase-total increments must equal
        # the sum of species increments in each phase, normalized by total moles.
        phase_delta_from_species = np.zeros(nphase, dtype=float)
        for k in range(kspec + 1):
            if unknown[k] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                phase_delta_from_species[phase_ids[k]] += delta_moles[k]
        total_scale = max(float(state.get("total_moles_old", 0.0)), 1.0e-4)
        if any(not vcs_doubleEqual(float(a) / total_scale, float(b) / total_scale)
               for a, b in zip(phase_delta_from_species, delta_phase)):
            raise VCSError("VCS major trial species and phase mole changes disagree")

        return {"delta": float(delta), "phase_delete": phase_delete,
                "all_minor_zeroed": all_minor_zeroed,
                "superconverged": False}

    def vcs_apply_zeroed_trial(self, species_index, reaction_index):
        """Apply the zeroed-species reappearance branch of ``solve_tp_inner``.

        A zeroed species is reconsidered only when its current reaction step
        proposes a positive amount and its NEW formation Gibbs energy is
        negative. Element-abundance bounds, phase existence, and the source's
        1% step seeding are preserved.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before zeroed-species trials")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        nsp = len(p["basis_ordered_species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        kspec, irxn = operator.index(species_index), operator.index(reaction_index)
        if kspec < ncomp or kspec >= nsp or irxn < 0 or irxn >= nrxn:
            raise IndexError("VCS zeroed trial species or reaction index is out of range")
        reaction_species = np.asarray(p["formation_reaction_species"], dtype=int)
        if reaction_species.shape != (nrxn,) or reaction_species[irxn] != kspec:
            raise VCSError("VCS zeroed trial reaction does not form the requested species")
        status = (self._species_status if self._species_status is not None else
                  np.asarray(state["species_status"], dtype=int))
        if (status[kspec] >= VCS_SPECIES_MINOR
                or status[kspec] == VCS_SPECIES_INTERFACIALVOLTAGE):
            raise VCSError("VCS zeroed trial requires a zeroed species status")
        phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
        phase_id = int(phase_ids[kspec])
        mole_old = np.asarray(state["mole_numbers_old"], dtype=float)
        mole_new = np.asarray(state["mole_numbers_new"], dtype=float)
        delta_g_new = np.asarray(state["delta_g_rxn_new"], dtype=float)
        if (mole_old.shape != (nsp,) or mole_new.shape != (nsp,)
                or delta_g_new.shape != (nrxn,)):
            raise VCSError("VCS zeroed trial state vectors have inconsistent dimensions")
        if self._delta_mol_num_species is None:
            self._delta_mol_num_species = np.zeros(nsp, dtype=float)
        delta_moles = np.asarray(self._delta_mol_num_species, dtype=float)
        if delta_moles.shape != (nsp,):
            raise VCSError("VCS species-step vector has inconsistent dimensions")

        resurrect = delta_moles[kspec] > 0.0
        if float(delta_g_new[irxn]) >= 0.0 or not resurrect:
            mole_new[kspec] = mole_old[kspec]
            delta_moles[kspec] = 0.0
            return {"resurrected": False, "delta": 0.0,
                    "species_status": int(status[kspec]),
                    "blocked_by_element_bound": False}

        formula = np.asarray(p["basis_ordered_formula_matrix"], dtype=float)
        element_types = np.asarray(p["reordered_element_types"], dtype=int)
        element_goals = np.asarray(
            p["reordered_element_abundance_goals"], dtype=float)
        if (formula.ndim != 2 or formula.shape[0] != nsp
                or element_types.shape != (formula.shape[1],)
                or element_goals.shape != (formula.shape[1],)):
            raise VCSError("VCS zeroed trial element metadata has inconsistent dimensions")
        for jelem, element_type in enumerate(element_types):
            atom_comp = float(formula[kspec, jelem])
            if (element_type == vcs_VolPhase._ELEM_TYPE_ABSPOS
                    and atom_comp > 0.0
                    and element_goals[jelem] / atom_comp
                        < VCS_DELETE_MINORSPECIES_CUTOFF):
                mole_new[kspec] = mole_old[kspec]
                delta_moles[kspec] = 0.0
                return {"resurrected": False, "delta": 0.0,
                        "species_status": int(status[kspec]),
                        "blocked_by_element_bound": True}

        phase = p["phase_objects"][phase_id]
        if phase.exists() == vcs_VolPhase._EXIST_NO:
            status[kspec] = VCS_SPECIES_MAJOR
        else:
            status[kspec] = VCS_SPECIES_MINOR
        if "species_status" in state:
            state["species_status"][kspec] = status[kspec]

        if delta_moles[kspec] > 0.0:
            delta = float(delta_moles[kspec]) * 0.01
            mole_new[kspec] = mole_old[kspec] + delta
        else:
            delta = (float(state["total_moles_old"])
                     * VCS_DELETE_PHASE_CUTOFF * 10.0
                     - float(mole_old[kspec]))
            mole_new[kspec] = float(state["total_moles_old"])
            mole_new[kspec] *= VCS_DELETE_PHASE_CUTOFF * 10.0
        delta_moles[kspec] = delta

        stoich = np.asarray(p["formation_reaction_stoichiometry"], dtype=float)
        phase_changes = np.asarray(p["delta_mol_num_phase"], dtype=float)
        nphase = len(p["phase_objects"])
        if (stoich.shape != (ncomp, nrxn)
                or phase_changes.shape != (nphase, nrxn)):
            raise VCSError("VCS zeroed trial stoichiometry has inconsistent dimensions")
        if self._delta_phase_moles is None:
            self._delta_phase_moles = np.zeros(nphase, dtype=float)
        delta_phase = np.asarray(self._delta_phase_moles, dtype=float)
        if delta_phase.shape != (nphase,):
            raise VCSError("VCS phase-step vector has inconsistent dimensions")
        delta_moles[:ncomp] += stoich[:, irxn] * delta
        delta_phase[:] += phase_changes[:, irxn] * delta

        phase_delta_from_species = np.zeros(nphase, dtype=float)
        for k in range(kspec + 1):
            if p["basis_ordered_species_unknown_types"][k] != (
                    vcs_VolPhase._UNKNOWN_VOLTAGE):
                phase_delta_from_species[phase_ids[k]] += delta_moles[k]
        scale = max(float(state.get("total_moles_old", 0.0)), 1.0e-4)
        if any(not vcs_doubleEqual(float(a) / scale, float(b) / scale)
               for a, b in zip(phase_delta_from_species, delta_phase)):
            raise VCSError("VCS zeroed trial species and phase mole changes disagree")
        self._species_status = status
        return {"resurrected": True, "delta": float(delta),
                "species_status": int(status[kspec]),
                "blocked_by_element_bound": False}

    def vcs_total_gibbs(self, mole_numbers=None, chemical_potentials=None,
                        *, active_species_count=None):
        """Sum VCS species Gibbs contributions in native active-index order.

        This translates ``VCS_SOLVE::vcs_Total_Gibbs``. Interfacial-voltage
        unknowns are solver variables rather than material species and are
        excluded; entries beyond the current reduced active-species count are
        also excluded.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before Gibbs evaluation")
        state = getattr(self, "_old_thermodynamic_state", None)
        if (mole_numbers is None or chemical_potentials is None) and state is None:
            state = self.initialize_thermodynamic_state()
        if mole_numbers is None:
            mole_numbers = state["mole_numbers_old"]
        if chemical_potentials is None:
            chemical_potentials = state["species_free_energies_old"]
        moles = np.asarray(mole_numbers, dtype=float)
        potentials = np.asarray(chemical_potentials, dtype=float)
        unknown = np.asarray(
            self._prepared_problem["basis_ordered_species_unknown_types"],
            dtype=int)
        if moles.ndim != 1 or potentials.ndim != 1 or unknown.ndim != 1:
            raise ValueError("VCS Gibbs inputs must be one-dimensional")
        if moles.shape != unknown.shape or potentials.shape != unknown.shape:
            raise ValueError("VCS Gibbs vectors must match the prepared species count")
        count = (getattr(self, "_num_species_reduced", unknown.size)
                 if active_species_count is None else active_species_count)
        if isinstance(count, bool):
            raise TypeError("active VCS species count must be an integer")
        try:
            count = operator.index(count)
        except TypeError as exc:
            raise TypeError("active VCS species count must be an integer") from exc
        if count < 0 or count > unknown.size:
            raise ValueError("active VCS species count is out of range")

        gibbs = 0.0
        for kspec in range(count):
            if unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                gibbs += float(moles[kspec]) * float(potentials[kspec])
        return gibbs

    def vcs_elcorr(self, aa=None, x=None):
        """Correct OLD species moles to restore elemental abundances.

        This ports ``VCS_SOLVE::vcs_elcorr`` from the pinned source, including
        its single-species and positive-element bounds passes, component-basis
        LU correction, damped nonnegative update, and charge-specific repairs.
        ``aa`` is a caller-owned scratch buffer retained for native signature
        compatibility; the native implementation checks its size but does not
        use it in this routine. The matrix solve uses this project's Python
        ``DenseMatrix`` LU implementation.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before element correction")
        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()
        formula = np.asarray(p["basis_ordered_formula_matrix"], dtype=float)
        unknown = np.asarray(p["basis_ordered_species_unknown_types"], dtype=int)
        goals = np.asarray(p["reordered_element_abundance_goals"], dtype=float)
        element_active = np.asarray(p["reordered_element_active"], dtype=bool)
        element_types = np.asarray(p["reordered_element_types"], dtype=int)
        moles = np.asarray(state["mole_numbers_old"], dtype=float)
        status = (self._species_status if self._species_status is not None else
                  np.asarray(state["species_status"], dtype=int))
        nsp, nelem = formula.shape
        ncomp = int(p["component_basis"]["num_components"])
        if (nsp == 0 or nelem == 0 or not 0 <= ncomp <= min(nsp, nelem)
                or unknown.shape != (nsp,) or goals.shape != (nelem,)
                or element_active.shape != (nelem,)
                or element_types.shape != (nelem,) or moles.shape != (nsp,)
                or status.shape != (nsp,)):
            raise VCSError("VCS element-correction arrays have inconsistent dimensions")
        if (not np.all(np.isfinite(formula)) or not np.all(np.isfinite(goals))
                or not np.all(np.isfinite(moles)) or np.any(moles < 0.0)):
            raise VCSError("VCS element-correction arrays contain invalid values")
        if aa is None:
            aa = np.zeros(nelem * nelem, dtype=float)
        if x is None:
            x = np.zeros(nelem, dtype=float)
        aa = np.asarray(aa)
        x = np.asarray(x)
        if aa.ndim != 1 or aa.size < nelem * nelem or x.ndim != 1 or x.size < nelem:
            raise ValueError("VCS element-correction scratch arrays are too small")

        active_species = int(getattr(self, "_num_species_reduced", nsp))
        if not 0 <= active_species <= nsp:
            raise VCSError("VCS active species count is out of range")

        def refresh_abundances():
            abundance = vcs_elab(formula, moles, unknown)
            state["species_status"][:] = status
            state["element_abundances_old"] = abundance.copy()
            state["element_abundance_residual_old"] = abundance - goals
            return abundance

        def refresh_phase_totals():
            phase_totals, total = vcs_tmoles(
                moles, p["basis_ordered_species_phase_ids"], unknown,
                len(p["phase_objects"]))
            state["phase_moles_old"][:] = phase_totals
            state["total_moles_old"] = total
            return phase_totals

        def finalize():
            refresh_phase_totals()

        abundance = refresh_abundances()
        original_abundance = abundance.copy()
        for i in range(nelem):
            x[i] = abundance[i] - goals[i]
        retn = 0

        # Correct isolated positive elements directly where the formula row
        # makes the responsible species unambiguous.
        changed = False
        for i in range(nelem):
            nonzero = 0
            has_negative = False
            for kspec in range(nsp):
                if unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                    coefficient = float(formula[kspec, i])
                    if coefficient < 0.0:
                        has_negative = True
                    if coefficient != 0.0:
                        nonzero += 1
            if has_negative:
                continue
            if nonzero < 2:
                for kspec in range(nsp):
                    if (unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE
                            and formula[kspec, i] > 0.0):
                        moles[kspec] = goals[i] / formula[kspec, i]
                        changed = True
            else:
                component_ids = [kspec for kspec in range(ncomp)
                                 if unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE
                                 and formula[kspec, i] > 0.0]
                if len(component_ids) == 1:
                    component = component_ids[0]
                    difference = float(goals[i])
                    for kspec in range(ncomp, nsp):
                        if unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                            difference -= formula[kspec, i] * moles[kspec]
                        # Preserve the source loop's assignment placement: it
                        # updates the component at every noncomponent index.
                        moles[component] = max(
                            0.0, difference / formula[component, i])
                        changed = True
        if changed:
            abundance = refresh_abundances()

        # Bound every species by each absolute-positive elemental inventory.
        changed = False
        for i in range(nelem):
            if element_types[i] != vcs_VolPhase._ELEM_TYPE_ABSPOS:
                continue
            for kspec in range(nsp):
                if unknown[kspec] == vcs_VolPhase._UNKNOWN_VOLTAGE:
                    continue
                coefficient = float(formula[kspec, i])
                if coefficient <= 0.0:
                    continue
                maximum = float(goals[i]) / coefficient
                if moles[kspec] > maximum:
                    moles[kspec] = maximum
                    changed = True
                    if moles[kspec] < VCS_DELETE_MINORSPECIES_CUTOFF:
                        moles[kspec] = 0.0
                        status[kspec] = (
                            VCS_SPECIES_ZEROEDSS
                            if p["basis_ordered_species_single_phase"][kspec]
                            else VCS_SPECIES_ACTIVEBUTZERO)
        if changed:
            abundance = refresh_abundances()

        # Solve the component correction system A*dx = (abundance-goal),
        # where A(j,i) = -formula(i,j), matching DenseMatrix's native layout.
        matrix = DenseMatrix(ncomp, ncomp)
        for i in range(ncomp):
            x[i] = abundance[i] - goals[i]
            if abs(float(x[i])) > 1.0e-13:
                retn = 1
            for j in range(ncomp):
                matrix[j, i] = -formula[i, j]
        if ncomp:
            try:
                x[:ncomp] = matrix.solve(np.asarray(x[:ncomp], dtype=float))
            except Exception as exc:
                raise VCSError("VCS element-correction component matrix is singular") from exc

        parameter = 0.5
        for i in range(ncomp):
            if moles[i] > 0.0:
                parameter = max(parameter, -float(x[i]) / float(moles[i]))
        parameter = min(parameter, 100.0)
        parameter = 1.0 / parameter
        if parameter < 1.0 and parameter > 0.0:
            retn = 2
            parameter *= 0.9999
        else:
            parameter = 1.0
        for i in range(ncomp):
            trial = moles[i] + parameter * x[i]
            if trial > 0.0:
                moles[i] = trial
            elif p["basis_ordered_species_single_phase"][i]:
                moles[i] = 0.0
            else:
                moles[i] *= 0.0001
        abundance = refresh_abundances()
        refresh_phase_totals()

        def balanced(check_all=False):
            return vcs_elabcheck(
                abundance, goals, ncomp, element_active, element_types,
                formula, moles, check_all_elements=check_all)

        # When the component correction had to be damped, try a species whose
        # elemental direction improves all component residuals at once.
        if retn >= 2:
            phase_ids = np.asarray(p["basis_ordered_species_phase_ids"], dtype=int)
            for kspec in range(nsp):
                if unknown[kspec] == vcs_VolPhase._UNKNOWN_VOLTAGE:
                    continue
                direction = 0.0
                good = True
                for i in range(ncomp):
                    value = formula[kspec, i] * (goals[i] - abundance[i])
                    if abs(value) > 1.0e-10:
                        if direction != 0.0 and direction * value < 0.0:
                            good = False
                            break
                        direction = value
                    elif formula[kspec, i] != 0.0:
                        good = False
                        break
                if not good:
                    continue
                adjustment = 0.0
                count = 0
                for i in range(ncomp):
                    if formula[kspec, i] != 0.0:
                        adjustment += ((goals[i] - abundance[i])
                                       / formula[kspec, i])
                        count += 1
                if count:
                    adjustment /= count
                moles[kspec] = max(moles[kspec] + adjustment, 1.0e-10)
                if kspec >= active_species:
                    self.vcs_reinsert_deleted(kspec)
                    active_species = int(self._num_species_reduced)
                    moles[active_species - 1] = adjustment
                    abundance = refresh_abundances()
                    finalize()
                    return retn
                abundance = refresh_abundances()
            if balanced():
                finalize()
                return 1

        # Repair charge-neutrality and zero-goal absolute-positive rows by
        # removing a species contribution with the wrong sign.
        for i in range(nelem):
            if (element_types[i] not in (
                    vcs_VolPhase._ELEM_TYPE_CHARGE_NEUTRALITY,
                    vcs_VolPhase._ELEM_TYPE_ABSPOS)
                    or (element_types[i] == vcs_VolPhase._ELEM_TYPE_ABSPOS
                        and goals[i] != 0.0)):
                continue
            for kspec in range(active_species):
                coefficient = formula[kspec, i]
                if abundance[i] > 0.0 and coefficient < 0.0:
                    moles[kspec] = max(
                        moles[kspec] - abundance[i] / coefficient, 0.0)
                    abundance = refresh_abundances()
                    break
                if abundance[i] < 0.0 and coefficient > 0.0:
                    moles[kspec] = max(
                        moles[kspec] - abundance[i] / coefficient, 0.0)
                    abundance = refresh_abundances()
                    break
        if balanced(check_all=True):
            finalize()
            return 1

        # Last source fallback: match electron-charge residual with a species
        # whose charge sign can move the abundance toward its requested value.
        for i in range(nelem):
            deviation = goals[i] - abundance[i]
            if (element_types[i] != vcs_VolPhase._ELEM_TYPE_ELECTRON_CHARGE
                    or abs(deviation) <= 1.0e-300):
                continue
            use_zeroed = True
            for kspec in range(active_species):
                coefficient = formula[kspec, i]
                if deviation < 0.0 and coefficient < 0.0 and moles[kspec] > 0.0:
                    use_zeroed = False
                elif deviation > 0.0 and coefficient > 0.0 and moles[kspec] > 0.0:
                    use_zeroed = False
            for kspec in range(active_species):
                if moles[kspec] > 0.0 or use_zeroed:
                    coefficient = formula[kspec, i]
                    if deviation < 0.0 and coefficient < 0.0:
                        moles[kspec] = max(
                            moles[kspec] + deviation / coefficient, 0.0)
                        abundance = refresh_abundances()
                        break
                    if deviation > 0.0 and coefficient > 0.0:
                        moles[kspec] = max(
                            moles[kspec] + deviation / coefficient, 0.0)
                        abundance = refresh_abundances()
                        break
        if balanced(check_all=True):
            finalize()
            return 1

        state["element_abundances_old"] = abundance.copy()
        state["element_abundance_residual_old"] = abundance - goals
        state["element_abundance_residual_before_correction"] = (
            original_abundance - goals)
        self._species_status = status
        return retn

    def vcs_phase_stability_test(
            self, phase_index, delta_g_rxn_old=None, fe_species_old=None,
            mole_numbers_old=None, activity_coefficients=None, *, max_iterations=200):
        """Run the source-mapped stability iteration on a prepared VCS phase.

        If OLD-state vectors are omitted, initialize them from the prepared
        phases. This method does not run the outer equilibrium solve or phase
        birth/death bookkeeping.
        """
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before phase stability")
        prepared = self._prepared_problem
        iph = int(phase_index)
        if iph < 0 or iph >= len(prepared["phase_objects"]):
            raise IndexError("VCS phase index is out of range")
        nsp = len(prepared["species_names"])
        if (delta_g_rxn_old is None or fe_species_old is None
                or mole_numbers_old is None or activity_coefficients is None):
            state = self.initialize_thermodynamic_state(mole_numbers_old)
            if delta_g_rxn_old is None:
                delta_g_rxn_old = state["delta_g_rxn_old"]
            if fe_species_old is None:
                fe_species_old = state["species_free_energies_old"]
            if mole_numbers_old is None:
                mole_numbers_old = state["mole_numbers_old"]
            if activity_coefficients is None:
                activity_coefficients = state["activity_coefficients_old"]
        moles = np.asarray(mole_numbers_old, dtype=float)
        if moles.shape != (nsp,):
            raise ValueError("OLD VCS mole vector has the wrong size")
        if activity_coefficients is None:
            activity = np.ones(nsp, dtype=float)
        else:
            activity = np.asarray(activity_coefficients, dtype=float).copy()
            if activity.shape != (nsp,):
                raise ValueError("VCS activity-coefficient vector has the wrong size")
        return vcs_phase_stability_successive_substitution(
            prepared["phase_objects"][iph],
            prepared["component_basis"]["num_components"],
            prepared["formation_reaction_stoichiometry"],
            delta_g_rxn_old, fe_species_old, moles, activity,
            max_iterations=max_iterations)

    def _refresh_new_thermodynamics(self) -> None:
        """Recompute NEW activities, chemical potentials, and reaction Gibbs."""
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before thermodynamic refresh")
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        nsp = len(p["species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        self.vcs_updateVP(vcs_VolPhase._STATE_NEW)
        activity = np.ones(nsp, dtype=float)
        potentials = np.zeros(len(p["phase_objects"]), dtype=float)
        for iph, phase in enumerate(p["phase_objects"]):
            if phase.nSpecies() > 1:
                phase.sendToVCS_ActCoeff(
                    vcs_VolPhase._STATE_NEW, activity,
                    moles=state["mole_numbers_new"])
            potentials[iph] = phase.electricPotential()
        state["activity_coefficients_new"] = activity
        state["phase_potentials_new"] = potentials
        reaction_species = np.arange(ncomp, nsp, dtype=int)
        unknown = p.get("basis_ordered_species_unknown_types", np.zeros(nsp, dtype=int))
        state["species_free_energies_new"] = vcs_dfe_species_free_energies(
            p["basis_ordered_standard_gibbs_rt"],
            state["ln_mnaught_species"], state["mole_numbers_new"],
            state["phase_moles_new"], p["basis_ordered_species_phase_ids"],
            activity, p["basis_ordered_species_charges"], potentials,
            state["species_status"], p["basis_ordered_species_single_phase"],
            unknown, reaction_species,
            ncomp, faraday_dim=Faraday / (GasConstant * p["temperature"]),
            initial_free_energies=state.get("species_free_energies_new"))
        state["delta_g_rxn_new"] = vcs_deltag(
            np.zeros(nsp - ncomp, dtype=float),
            state["species_free_energies_new"],
            p["formation_reaction_stoichiometry"], reaction_species,
            state["species_status"], ncomp, state["mole_numbers_new"])

    def vcs_updateMolNumVolPhases(self, state_calc: int) -> None:
        """Update VolPhase objects with current VCS mole numbers."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        moles_key = "mole_numbers_old" if state_calc == vcs_VolPhase._STATE_OLD else "mole_numbers_new"
        moles = state[moles_key]
        for phase in p["phase_objects"]:
            phase.setMolesFromVCS(state_calc, moles)

    def vcs_tmoles(self) -> float:
        """Update total phase moles and system mole total in state."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        nsp = len(p["species_names"])
        unknown = p.get("basis_ordered_species_unknown_types", np.zeros(nsp, dtype=int))
        phase_moles, total_moles = vcs_tmoles(
            state["mole_numbers_old"],
            p["basis_ordered_species_phase_ids"],
            unknown,
            len(p["phase_objects"]))
        state["phase_moles_old"][:] = phase_moles
        state["total_moles_old"] = float(total_moles)
        for iph, phase in enumerate(p["phase_objects"]):
            phase.setTotalMoles(phase_moles[iph])
        return float(total_moles)

    def vcs_elabcheck(self, ibound: int = 0) -> bool:
        """Check elemental abundance tolerances."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        nsp = len(p["species_names"])
        unknown = p.get("basis_ordered_species_unknown_types", np.zeros(nsp, dtype=int))
        abundances = vcs_elab(
            p["basis_ordered_formula_matrix"],
            state["mole_numbers_old"],
            unknown)
        return vcs_elabcheck(
            abundances,
            p["reordered_element_abundance_goals"],
            p["component_basis"]["num_components"],
            p["reordered_element_active"],
            p["reordered_element_types"],
            p["basis_ordered_formula_matrix"],
            state["mole_numbers_old"],
            check_all_elements=(ibound != 0))

    def vcs_popPhasePossible(self, iphase_pop: int) -> bool:
        """Determine if a dead phase can physically be popped by available components."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        phase = p["phase_objects"][iphase_pop]
        if phase.exists() > 0:
            return False
        ncomp = int(p["component_basis"]["num_components"])
        stoich = p["formation_reaction_stoichiometry"]
        nrxn = len(p["species_names"]) - ncomp
        old_moles = state["mole_numbers_old"]
        el_types = p["reordered_element_types"]

        for k in range(phase.nSpecies()):
            kspec = phase.spGlobalIndexVCS(k)
            if kspec >= ncomp:
                irxn = kspec - ncomp
                pop_possible = True
                for j in range(ncomp):
                    if el_types[j] == vcs_VolPhase._ELEM_TYPE_ABSPOS:
                        sc = stoich[j, irxn]
                        if sc < 0.0 and old_moles[j] <= 1.0e-13 * 0.5:
                            pop_possible = False
                            break
                if pop_possible:
                    return True
            else:
                for jrxn in range(nrxn):
                    if stoich[kspec, jrxn] > 0.0:
                        possible = True
                        for kcomp in range(ncomp):
                            if stoich[kcomp, jrxn] < 0.0 and old_moles[kcomp] <= 1.0e-13 * 0.5:
                                possible = False
                                break
                        if possible:
                            return True
                    elif stoich[kspec, jrxn] < 0.0:
                        jspec = jrxn + ncomp
                        if old_moles[jspec] <= 1.0e-13 * 0.5:
                            continue
                        possible = True
                        for kcomp in range(ncomp):
                            if stoich[kcomp, jrxn] > 0.0 and old_moles[kcomp] <= 1.0e-13 * 0.5:
                                possible = False
                                break
                        if possible:
                            return True
        return False

    def vcs_popPhaseID(self):
        """Identify which dead phase, if any, is most favorable to pop into existence."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        ncomp = int(p["component_basis"]["num_components"])
        dg_old = state["delta_g_rxn_old"]
        iphase_pop = None
        fe_phase_max = -1.0e30

        for iph, phase in enumerate(p["phase_objects"]):
            existence = phase.exists()
            if existence > 0:
                continue
            if phase.nSpecies() == 1:
                kspec = phase.spGlobalIndexVCS(0)
                if kspec < ncomp:
                    continue
                irxn = kspec - ncomp
                dg = float(dg_old[irxn])
                fe_phase = np.exp(-dg) - 1.0
                if fe_phase > 0.0 and fe_phase > fe_phase_max:
                    iphase_pop = iph
                    fe_phase_max = fe_phase
            elif self.vcs_popPhasePossible(iph):
                fe_phase = self.vcs_phase_stability_test(iph)
                if fe_phase > 0.0 and fe_phase > fe_phase_max:
                    iphase_pop = iph
                    fe_phase_max = fe_phase
        return iphase_pop

    def vcs_popPhaseRxnStepSizes(self, iphase_pop: int) -> int:
        """Compute reaction step sizes for a popping phase."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        phase = p["phase_objects"][iphase_pop]
        kspec = phase.spGlobalIndexVCS(0)
        ncomp = int(p["component_basis"]["num_components"])
        if kspec < ncomp:
            raise VCSError("vcs_popPhaseRxnStepSizes called for component species phase")
        irxn = kspec - ncomp
        total_moles = state.get("total_moles_old", 1.0)
        t_phase_moles = 10.0 * total_moles * 1.0e-14

        if self._delta_mol_num_species is None:
            self._delta_mol_num_species = np.zeros(len(p["species_names"]), dtype=float)

        if phase.nSpecies() == 1:
            s = 0.0
            stoich = p["formation_reaction_stoichiometry"]
            old_moles = state["mole_numbers_old"]
            single_phase = p["basis_ordered_species_single_phase"]
            for j in range(ncomp):
                if not single_phase[j] and old_moles[j] > 0.0:
                    s += (stoich[j, irxn] ** 2) / old_moles[j]
            phase_changes = p["delta_mol_num_phase"]
            old_phase_moles = state["phase_moles_old"]
            for j, ph in enumerate(p["phase_objects"]):
                if ph.nSpecies() > 1 and old_phase_moles[j] > 0.0:
                    s -= (phase_changes[j, irxn] ** 2) / old_phase_moles[j]
            if s > 0.0:
                use_act = p.get("use_activity_jacobian", False)
                if use_act and p.get("ln_act_coeff_jacobian") is not None:
                    s = vcs_hessian_diag_adj(
                        irxn, s, stoich, p.get("formation_reaction_species"),
                        p["basis_ordered_phase_ids"], old_phase_moles,
                        single_phase, p["ln_act_coeff_jacobian"]
                    )
                dg_new = state.get("delta_g_rxn_new", state["delta_g_rxn_old"])
                self._delta_mol_num_species[kspec] = -float(dg_new[irxn]) / s
            else:
                self._delta_mol_num_species[kspec] = t_phase_moles

            el_types = p["reordered_element_types"]
            for j in range(ncomp):
                sc = stoich[j, irxn]
                if sc != 0.0 and el_types[j] == vcs_VolPhase._ELEM_TYPE_ABSPOS:
                    neg_change = -sc * self._delta_mol_num_species[kspec]
                    if neg_change > old_moles[j]:
                        if old_moles[j] > 0.0:
                            self._delta_mol_num_species[kspec] = -0.5 * old_moles[j] / sc
                        else:
                            self._delta_mol_num_species[kspec] = 0.0
            if -self._delta_mol_num_species[kspec] > old_moles[kspec]:
                self._delta_mol_num_species[kspec] = -old_moles[kspec]
        else:
            frac_delta = phase.creationMoleNumbers()
            sum_frac = np.sum(frac_delta)
            if sum_frac <= 0.0:
                sum_frac = 1.0
            x_est = frac_delta / sum_frac
            delta_mol_phase = t_phase_moles
            damp = 1.0
            mol_tmp = state["mole_numbers_old"].copy()
            stoich = p["formation_reaction_stoichiometry"]
            el_types = p["reordered_element_types"]

            for k in range(phase.nSpecies()):
                sp_k = phase.spGlobalIndexVCS(k)
                delmol = delta_mol_phase * x_est[k]
                if sp_k >= ncomp:
                    rxn_k = sp_k - ncomp
                    for j in range(ncomp):
                        sc = stoich[j, rxn_k]
                        if sc != 0.0 and el_types[j] == vcs_VolPhase._ELEM_TYPE_ABSPOS:
                            mol_tmp[j] += sc * delmol

            for j in range(ncomp):
                del_j = state["mole_numbers_old"][j] - mol_tmp[j]
                if mol_tmp[j] < 0.0 and del_j > 0.0:
                    damp = min(damp, (state["mole_numbers_old"][j] / del_j) * 0.9)
            if damp <= 1.0e-6:
                return 3
            status = state["species_status"]
            for k in range(phase.nSpecies()):
                sp_k = phase.spGlobalIndexVCS(k)
                if sp_k < ncomp:
                    status[sp_k] = VCS_SPECIES_COMPONENT
                else:
                    self._delta_mol_num_species[sp_k] = delta_mol_phase * x_est[k] * damp
                    status[sp_k] = VCS_SPECIES_MAJOR if x_est[k] > 1.0e-3 else VCS_SPECIES_MINOR
        return 0

    def solve_tp_inner(self, iti: int, it1: int, uptodate_minors: bool,
                       all_minor_zeroed: bool, force_component_calc: int,
                       stage: int, basis_opt_count: int) -> tuple[int, int, bool, bool, int, int, int]:
        """Inner iteration of VCS_SOLVE::solve_TP."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        nsp = len(p["species_names"])
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = nsp - ncomp
        MAIN = 0
        EQUILIB_CHECK = 1
        RECHECK_DELETED = 3

        if iti == 0:
            if not uptodate_minors:
                self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
                self._refresh_old_thermodynamics()
            uptodate_minors = True
        else:
            uptodate_minors = False

        self.vcs_tmoles()
        state["mole_numbers_new"] = state["mole_numbers_old"].copy()
        state["phase_moles_new"] = state["phase_moles_old"].copy()
        state["delta_g_rxn_new"] = state["delta_g_rxn_old"].copy()
        state["species_free_energies_new"] = state["species_free_energies_old"].copy()
        state["activity_coefficients_new"] = state["activity_coefficients_old"].copy()
        self._delta_mol_num_species = np.zeros(nsp, dtype=float)
        self._delta_phase_moles = np.zeros(len(p["phase_objects"]), dtype=float)

        iphase_pop = self.vcs_popPhaseID()
        if iphase_pop is not None:
            soldel = self.vcs_popPhaseRxnStepSizes(iphase_pop)
            if soldel == 3:
                iphase_pop = None

        if iphase_pop is None:
            result = self.vcs_RxnStepSizes(state["delta_g_rxn_new"])
            iphase_delete = result.get("deleted_phase")
            force_component_calc = result.get("force_component_calc", 0)
            special_species = result.get("special_species")
        else:
            iphase_delete = None
            special_species = None

        if iphase_delete is not None:
            delta_moles = self._delta_mol_num_species
            state["mole_numbers_new"] = state["mole_numbers_old"] + delta_moles
            for k in range(nsp):
                iph = p["basis_ordered_species_phase_ids"][k]
                state["phase_moles_new"][iph] += delta_moles[k]
            if special_species is not None and special_species >= ncomp:
                if p["basis_ordered_species_single_phase"][special_species]:
                    state["species_status"][special_species] = VCS_SPECIES_ZEROEDSS
                    self._num_rxn_minor_zeroed += 1
                    all_minor_zeroed = (self._num_rxn_minor_zeroed == nrxn)
            self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_NEW)
            self._refresh_new_thermodynamics()
        else:
            for irxn in range(nrxn):
                kspec = int(p["formation_reaction_species"][irxn])
                status = state["species_status"][kspec]
                iph = p["basis_ordered_species_phase_ids"][kspec]
                if iphase_pop is not None:
                    if iph == iphase_pop:
                        dx = self._delta_mol_num_species[kspec]
                        state["mole_numbers_new"][kspec] = state["mole_numbers_old"][kspec] + dx
                    else:
                        dx = 0.0
                        self._delta_mol_num_species[kspec] = 0.0
                        state["mole_numbers_new"][kspec] = state["mole_numbers_old"][kspec]
                    stoich = p["formation_reaction_stoichiometry"]
                    for k in range(ncomp):
                        self._delta_mol_num_species[k] += dx * stoich[k, irxn]
                    phase_changes = p["delta_mol_num_phase"]
                    for iph_idx in range(len(p["phase_objects"])):
                        self._delta_phase_moles[iph_idx] += dx * phase_changes[iph_idx, irxn]
                elif status == VCS_SPECIES_INTERFACIALVOLTAGE:
                    dx, _ = self.vcs_minor_alt_calc(kspec, irxn)
                    self._delta_mol_num_species[kspec] = dx
                    state["mole_numbers_new"][kspec] = state["mole_numbers_old"][kspec] + dx
                elif status < VCS_SPECIES_MINOR:
                    self.vcs_apply_zeroed_trial(kspec, irxn)
                elif status == VCS_SPECIES_MINOR:
                    if iti != 0:
                        state["mole_numbers_new"][kspec] = state["mole_numbers_old"][kspec]
                        self._delta_mol_num_species[kspec] = 0.0
                        continue
                    res = self.vcs_apply_minor_trial(kspec, irxn, iteration=iti)
                    if res.get("deleted"):
                        if res.get("recheck_required"):
                            stage = RECHECK_DELETED
                            return iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count
                        continue
                else:
                    res = self.vcs_apply_major_trial(kspec, irxn)
                    if res.get("phase_delete") is not None:
                        break

            # Limit reduction of component species to 99%
            par = 0.5
            for k in range(ncomp):
                if state["mole_numbers_old"][k] > 0.0:
                    xx = -self._delta_mol_num_species[k] / state["mole_numbers_old"][k]
                    if par < xx:
                        par = xx
                elif self._delta_mol_num_species[k] < 0.0:
                    iph = p["basis_ordered_species_phase_ids"][k]
                    self._delta_phase_moles[iph] -= self._delta_mol_num_species[k]
                    self._delta_mol_num_species[k] = 0.0
            par = 1.0 / par
            if 0.0 < par <= 1.01:
                par *= 0.99
                self._delta_mol_num_species *= par
                self._delta_phase_moles *= par

            state["mole_numbers_new"] = np.maximum(
                state["mole_numbers_old"] + self._delta_mol_num_species, 0.0)
            state["phase_moles_new"] = np.maximum(
                state["phase_moles_old"] + self._delta_phase_moles, 0.0)

            self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_NEW)
            self._refresh_new_thermodynamics()

        # Global step damping / forcer
        self.vcs_glob_step_damp()

        # Accept step
        self.vcs_updateMolNumVolPhases(vcs_VolPhase._STATE_NEW)
        state["mole_numbers_old"][:] = state["mole_numbers_new"]
        state["phase_moles_old"][:] = state["phase_moles_new"]
        state["delta_g_rxn_old"][:] = state["delta_g_rxn_new"]
        state["species_free_energies_old"][:] = state["species_free_energies_new"]
        state["activity_coefficients_old"][:] = state["activity_coefficients_new"]
        self.vcs_setFlagsVolPhases(True, vcs_VolPhase._STATE_OLD)
        self._iterations += 1
        it1 += 1

        # Check microscopic multispecies phase deletion
        just_deleted_multiphase = False
        total_moles = state.get("total_moles_old", 1.0)
        if total_moles <= 0.0:
            total_moles = max(np.sum(state["mole_numbers_old"]), 1.0e-14)
        for iph, phase in enumerate(p["phase_objects"]):
            if phase.nSpecies() > 1 and state["phase_moles_old"][iph] > 0.0:
                if state["phase_moles_old"][iph] / total_moles <= 1.0e-14:
                    just_deleted_multiphase = True
                    self.vcs_delete_multiphase(iph)
        if just_deleted_multiphase:
            self.vcs_basopt(False, -1.0e-10)
            self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
            self._refresh_old_thermodynamics()
            iti = 0
            return iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count

        # Element abundance check
        if not self.vcs_elabcheck(0):
            self.vcs_elcorr()
            self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
            self._refresh_old_thermodynamics()
            uptodate_minors = True

        # Check for optimum basis
        max_basis_opt = 200
        if basis_opt_count <= max_basis_opt:
            stoich = p["formation_reaction_stoichiometry"]
            sp_sizes = p.get("basis_ordered_species_sizes")
            if sp_sizes is None or len(sp_sizes) != nsp:
                sp_sizes = np.ones(nsp, dtype=float)
            single_phase = p["basis_ordered_species_single_phase"]
            swapped = False
            for irxn in range(nrxn):
                k = int(p["formation_reaction_species"][irxn])
                for j in range(ncomp):
                    do_swap = False
                    if single_phase[j]:
                        do_swap = (state["mole_numbers_old"][k] * sp_sizes[k]) > (state["mole_numbers_old"][j] * sp_sizes[j] * 1.01)
                        if not single_phase[k] and do_swap:
                            do_swap = state["mole_numbers_old"][k] > (state["mole_numbers_old"][j] * 1.01)
                    else:
                        if single_phase[k]:
                            do_swap = (state["mole_numbers_old"][k] * sp_sizes[k]) > (state["mole_numbers_old"][j] * sp_sizes[j] * 1.01)
                            if not do_swap:
                                do_swap = state["mole_numbers_old"][k] > (state["mole_numbers_old"][j] * 1.01)
                        else:
                            do_swap = (state["mole_numbers_old"][k] * sp_sizes[k]) > (state["mole_numbers_old"][j] * sp_sizes[j] * 1.01)
                    if do_swap and stoich[j, irxn] != 0.0:
                        force_component_calc = 1
                        basis_opt_count += 1
                        swapped = True
                        return iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count
            if not swapped:
                basis_opt_count = 0

        stage = EQUILIB_CHECK
        if iti != 0:
            return iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count
        all_minor_zeroed = self.vcs_evaluate_speciesType()
        return iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count

    def solve_tp_equilib_check(self, all_minor_zeroed: bool, uptodate_minors: bool,
                               give_up_on_elem_abund: bool, iconv: int,
                               iti: int, it1: int, maxit: int,
                               stage: int, lec: bool) -> tuple[bool, bool, bool, int, int, int, bool]:
        """Check convergence tolerances on major and minor species."""
        p = self._prepared_problem
        state = self._old_thermodynamic_state
        ncomp = int(p["component_basis"]["num_components"])
        nrxn = len(p["species_names"]) - ncomp
        dg_new = state.get("delta_g_rxn_new", state["delta_g_rxn_old"])
        status = state["species_status"]
        MAIN = 0
        ELEM_ABUND_CHECK = 2
        RECHECK_DELETED = 3
        RETURN_A = 4

        if not all_minor_zeroed:
            for irxn in range(nrxn):
                kspec = int(p["formation_reaction_species"][irxn])
                if status[kspec] == VCS_SPECIES_MAJOR and abs(float(dg_new[irxn])) > self._tolmaj:
                    if self._iterations >= maxit:
                        iconv = -1
                        stage = RETURN_A
                    else:
                        iti = ((it1 // 4) * 4) - it1
                        stage = MAIN
                    return all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec

        if self._num_rxn_minor_zeroed != 0:
            if iti != 0:
                self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
                self._refresh_old_thermodynamics()
                uptodate_minors = True
            for irxn in range(nrxn):
                kspec = int(p["formation_reaction_species"][irxn])
                if status[kspec] == VCS_SPECIES_MINOR and abs(float(dg_new[irxn])) > self._tolmin:
                    if self._iterations >= maxit:
                        iconv = -1
                        stage = RETURN_A
                    else:
                        iti = 0
                        stage = MAIN
                    return all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec

        self.vcs_updateVP(vcs_VolPhase._STATE_OLD)
        if lec:
            if not give_up_on_elem_abund:
                if not self.vcs_elabcheck(1):
                    stage = ELEM_ABUND_CHECK
                    return all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec
            if self._num_species_reduced != len(p["species_names"]):
                stage = RECHECK_DELETED
                return all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec
            stage = RETURN_A
            return all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec
        lec = True
        return all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec

    def solve_tp_elem_abund_check(self, iti: int, stage: int, lec: bool,
                                  give_up_on_elem_abund: bool,
                                  final_elem_abund_attempts: int,
                                  iconv: int) -> tuple[int, int, bool, bool, int, int]:
        """Perform end-game elemental abundance checks and corrections."""
        MAIN = 0
        EQUILIB_CHECK = 1

        if not self.vcs_elabcheck(1):
            nc_before = self.vcs_elabcheck(0)
            self.vcs_elcorr()
            nc_after = self.vcs_elabcheck(0)
            ne_after = self.vcs_elabcheck(1)
            self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
            self._refresh_old_thermodynamics()
            if not nc_before:
                if nc_after:
                    lec = False
                    iti = 0
                    stage = MAIN
                else:
                    if final_elem_abund_attempts >= 3:
                        give_up_on_elem_abund = True
                        stage = EQUILIB_CHECK
                    else:
                        final_elem_abund_attempts += 1
                        lec = False
                        iti = 0
                        stage = MAIN
                return iti, stage, lec, give_up_on_elem_abund, final_elem_abund_attempts, iconv
            elif nc_after:
                if not ne_after:
                    give_up_on_elem_abund = True
                stage = EQUILIB_CHECK
                return iti, stage, lec, give_up_on_elem_abund, final_elem_abund_attempts, iconv

        self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
        self._refresh_old_thermodynamics()
        stage = EQUILIB_CHECK
        return iti, stage, lec, give_up_on_elem_abund, final_elem_abund_attempts, iconv

    def solve_TP(self, print_lvl: int = 0, print_details: int = 0,
                 maxit: int = 500) -> int:
        """Execute the pinned VCS active-set TP state machine to convergence."""
        if self._prepared_problem is None:
            raise VCSError("VCS problem must be prepared before solve_TP")

        MAIN = 0
        EQUILIB_CHECK = 1
        ELEM_ABUND_CHECK = 2
        RECHECK_DELETED = 3
        RETURN_A = 4
        RETURN_B = 5

        self._iterations = 0
        stage = MAIN
        all_minor_zeroed = False
        it1 = 0
        iti = 0
        give_up_on_elem_abund = False
        final_elem_abund_attempts = 0
        uptodate_minors = True
        force_component_calc = 1
        basis_opt_count = 0
        iconv = 0

        p = self._prepared_problem
        state = getattr(self, "_old_thermodynamic_state", None)
        if state is None:
            state = self.initialize_thermodynamic_state()

        self.vcs_tmoles()
        self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
        self._refresh_old_thermodynamics()

        lec = False
        while True:
            if stage == MAIN:
                if force_component_calc:
                    self.solve_tp_component_calc()
                    it1 = 1
                    force_component_calc = 0
                    iti = 0
                if self._iterations >= maxit:
                    iconv = -1
                    stage = RETURN_A
                    continue
                iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count = self.solve_tp_inner(
                    iti, it1, uptodate_minors, all_minor_zeroed, force_component_calc, stage, basis_opt_count)
                lec = False
            elif stage == EQUILIB_CHECK:
                all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, stage, lec = self.solve_tp_equilib_check(
                    all_minor_zeroed, uptodate_minors, give_up_on_elem_abund, iconv, iti, it1, maxit, stage, lec)
            elif stage == ELEM_ABUND_CHECK:
                iti, stage, lec, give_up_on_elem_abund, final_elem_abund_attempts, iconv = self.solve_tp_elem_abund_check(
                    iti, stage, lec, give_up_on_elem_abund, final_elem_abund_attempts, iconv)
            elif stage == RECHECK_DELETED:
                npb = self.vcs_recheck_deleted()
                if npb <= 0:
                    stage = RETURN_B
                else:
                    self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
                    self._refresh_old_thermodynamics()
                    iti = 0
                    stage = MAIN
            elif stage == RETURN_A:
                npb = self.vcs_recheck_deleted()
                if npb > 0:
                    self.vcs_setFlagsVolPhases(False, vcs_VolPhase._STATE_OLD)
                    self._refresh_old_thermodynamics()
                    iti = 0
                    stage = MAIN
                else:
                    stage = RETURN_B
            elif stage == RETURN_B:
                npb = self.vcs_add_all_deleted()
                if npb > 0:
                    iti = 0
                    stage = MAIN
                else:
                    break

        self.vcs_updateVP(vcs_VolPhase._STATE_OLD)
        return iconv

    def vcs_solve_TP(self, multiphase: MultiPhase, print_lvl: int = 0,
                     vcs_mode: int = 0, max_steps: int = 500) -> int:
        """Solve multiphase chemical equilibrium at fixed T and P."""
        self.prepare_problem(multiphase)
        ret = self.solve_TP(print_lvl=print_lvl, maxit=max_steps)
        if ret != 0:
            raise VCSError(f"VCS solver failed with status code {ret}")
        multiphase.uploadMoleFractionsFromPhases()
        for ip, phase in enumerate(self._prepared_problem["phase_objects"]):
            multiphase.set_phase_moles(ip, phase.totalMoles())
        multiphase.calcElemAbundances()
        return ret

    def vcs(self, multiphase: MultiPhase, print_lvl: int = 0) -> int:
        """Alias for vcs_solve_TP."""
        return self.vcs_solve_TP(multiphase, print_lvl=print_lvl)


class MultiPhaseEquil:
    """Public placeholder for Cantera's stoichiometric multiphase solver."""

    def __init__(self, multiphase: MultiPhase, print_lvl: int = 0):
        self._mp = multiphase
        self._print_lvl = int(print_lvl)

    def equilibrate(self, XY: str = "TP", rtol: float = 1e-9,
                    max_steps: int = 500) -> int:
        raise NotImplementedError(
            "MultiPhaseEquil is not implemented; no equilibrium state was computed"
        )

def vcs_l2norm(vec: Sequence[float]) -> float:
    return _python_l2_norm(vec)

def vcs_speciesType_string(stype: int) -> str:
    return "VCS_SPECIES_TYPE_MOMENT" if stype == 0 else "VCS_SPECIES_TYPE_INTERMEDIATE"

def vcs_doubleEqual(d1: float, d2: float) -> bool:
    denom = abs(d1) + abs(d2) + 1.0
    return (abs(d1 - d2) / denom) <= 1.0e-10


def vcs_elab(formula_matrix, mole_numbers, species_unknown_types=None):
    """Recompute VCS element abundances using the pinned ``vcs_elab`` loop.

    ``formula_matrix`` uses species rows and element columns. Interfacial
    voltage unknowns are excluded because their solver variable is a voltage,
    not a species amount.
    """
    formula = np.asarray(formula_matrix, dtype=float)
    moles = np.asarray(mole_numbers, dtype=float)
    if formula.ndim != 2 or moles.ndim != 1 or formula.shape[0] != moles.size:
        raise ValueError("formula matrix rows must match the mole-number vector")
    unknown_types = (np.zeros(moles.size, dtype=int)
                     if species_unknown_types is None
                     else np.asarray(species_unknown_types, dtype=int))
    if unknown_types.ndim != 1 or unknown_types.size != moles.size:
        raise ValueError("species unknown types must match the species count")

    abundances = np.zeros(formula.shape[1], dtype=float)
    for j in range(formula.shape[1]):
        abundances[j] = 0.0
        for i in range(formula.shape[0]):
            if unknown_types[i] != vcs_VolPhase._UNKNOWN_VOLTAGE:
                abundances[j] += formula[i, j] * moles[i]
    return abundances


def vcs_tmoles(species_moles, species_phase_ids, species_unknown_types,
               phase_count: int):
    """Recompute native VCS phase and system mole totals in source order."""
    moles = np.asarray(species_moles, dtype=float)
    phase_ids = np.asarray(species_phase_ids, dtype=int)
    unknown = np.asarray(species_unknown_types, dtype=int)
    nphase = int(phase_count)
    if (moles.ndim != 1 or phase_ids.shape != moles.shape
            or unknown.shape != moles.shape or nphase < 0
            or np.any(phase_ids < 0) or np.any(phase_ids >= nphase)):
        if moles.size == 0 and nphase == 0 and phase_ids.shape == moles.shape:
            return np.zeros(0, dtype=float), 0.0
        raise ValueError("VCS phase-total inputs have inconsistent dimensions")
    totals = np.zeros(nphase, dtype=float)
    for k in range(moles.size):
        if unknown[k] == vcs_VolPhase._UNKNOWN_MOLES:
            totals[phase_ids[k]] += float(moles[k])
    system_total = 0.0
    for phase_total in totals:
        system_total += float(phase_total)
    return totals, system_total


def vcs_delta_species_step(formation_stoichiometry, species_moles_old,
                           phase_moles_old, species_unknown_types,
                           species_phase_ids, species_index, delta):
    """Apply one native VCS formation-reaction extent to a copied state."""
    stoich = np.asarray(formation_stoichiometry, dtype=float)
    moles = np.asarray(species_moles_old, dtype=float)
    phase_moles = np.asarray(phase_moles_old, dtype=float)
    unknown = np.asarray(species_unknown_types, dtype=int)
    phase_ids = np.asarray(species_phase_ids, dtype=int)
    if stoich.ndim != 2 or moles.ndim != 1:
        raise ValueError("VCS reaction stoichiometry and moles have invalid rank")
    ncomp, nrxn = stoich.shape
    nsp = moles.size
    if (ncomp > nsp or unknown.shape != (nsp,) or phase_ids.shape != (nsp,)
            or phase_moles.ndim != 1 or np.any(phase_ids < 0)
            or np.any(phase_ids >= phase_moles.size)
            or not np.all(np.isfinite(stoich))
            or not np.all(np.isfinite(moles))
            or not np.all(np.isfinite(phase_moles))
            or np.any(moles < 0.0) or np.any(phase_moles < 0.0)):
        raise ValueError("VCS delta-species arrays have inconsistent values or dimensions")
    if isinstance(species_index, bool):
        raise TypeError("VCS species index must be an integer")
    try:
        kspec = operator.index(species_index)
    except TypeError as exc:
        raise TypeError("VCS species index must be an integer") from exc
    if kspec < ncomp or kspec >= nsp:
        raise IndexError("VCS delta_species requires a non-component species")
    irxn = kspec - ncomp
    if irxn >= nrxn:
        raise IndexError("VCS species has no formation reaction")
    extent = float(delta)
    if not math.isfinite(extent):
        raise ValueError("VCS species extent must be finite")

    updated_moles = moles.copy()
    updated_phase_moles = phase_moles.copy()
    invalidated_phase_ids = []
    success = 1
    # Interfacial voltage variables are not species amounts; native VCS leaves
    # both their extent and the mole vectors untouched.
    if unknown[kspec] != vcs_VolPhase._UNKNOWN_VOLTAGE:
        reaction = stoich[:, irxn]
        for jcomp in range(ncomp):
            coefficient = float(reaction[jcomp])
            if moles[jcomp] > 0.0:
                change = coefficient * extent
                if -change > moles[jcomp]:
                    success = 0
                    extent = min(extent, -float(moles[jcomp]) / coefficient)
            if moles[jcomp] <= 0.0 and coefficient < 0.0:
                extent = 0.0
                return {
                    "success": 0,
                    "delta": extent,
                    "mole_numbers": updated_moles,
                    "phase_moles": updated_phase_moles,
                    "total_moles": float(sum(float(x) for x in updated_phase_moles)),
                    "invalidated_phase_ids": (),
                }

        updated_moles[kspec] += extent
        product_phase = int(phase_ids[kspec])
        updated_phase_moles[product_phase] += extent
        # The native method invalidates the product phase unconditionally,
        # then invalidates only component phases with nonzero mole changes.
        invalidated_phase_ids.append(product_phase)
        for jcomp in range(ncomp):
            change = float(reaction[jcomp]) * extent
            if change != 0.0:
                iphase = int(phase_ids[jcomp])
                invalidated_phase_ids.append(iphase)
                updated_phase_moles[iphase] += change
                updated_moles[jcomp] = max(updated_moles[jcomp] + change, 0.0)

    system_total = 0.0
    for phase_total in updated_phase_moles:
        system_total += float(phase_total)
    return {
        "success": success,
        "delta": extent,
        "mole_numbers": updated_moles,
        "phase_moles": updated_phase_moles,
        "total_moles": system_total,
        "invalidated_phase_ids": tuple(invalidated_phase_ids),
    }


def vcs_phase_stability_successive_substitution(
        phase, num_components, stoich_coefficients, delta_g_rxn_old,
        fe_species_old, mole_numbers_old, activity_coefficients,
        *, max_iterations=200):
    """Port ``VCS_SOLVE::vcs_phaseStabilityTest`` successive substitution.

    Arrays use global VCS ordering; stoichiometry is component-by-formation-
    reaction, and the phase carries local-to-global species indices. The
    routine intentionally mirrors the pinned scalar loop and damping order.
    """
    ncomp = int(num_components)
    stoich = np.asarray(stoich_coefficients, dtype=float)
    delta_g_old = np.asarray(delta_g_rxn_old, dtype=float)
    fe_old = np.asarray(fe_species_old, dtype=float)
    moles_old = np.asarray(mole_numbers_old, dtype=float)
    activity = np.asarray(activity_coefficients, dtype=float).copy()
    nsp = phase.nSpecies()
    global_species = [phase.spGlobalIndexVCS(k) for k in range(nsp)]
    if (ncomp < 0 or stoich.ndim != 2 or stoich.shape[0] != ncomp
            or delta_g_old.shape != (stoich.shape[1],)
            or fe_old.ndim != 1 or moles_old.shape != fe_old.shape
            or activity.shape != fe_old.shape
            or any(k < 0 or k >= fe_old.size for k in global_species)):
        raise ValueError("VCS phase-stability arrays have inconsistent dimensions")
    if any(k - ncomp >= stoich.shape[1] for k in global_species if k >= ncomp):
        raise ValueError("VCS phase species exceed the formation-reaction array")

    min_iterations = 1 if nsp <= 1 else 3
    frac_new = np.asarray(phase._creation_mole_numbers, dtype=float).copy()
    creation_rxn_numbers = list(
        getattr(phase, "_creation_global_rxn_numbers", (-1,) * nsp))
    if len(creation_rxn_numbers) != nsp:
        creation_rxn_numbers = [-1] * nsp
    for k in range(nsp):
        if frac_new[k] < 1.0e-13:
            frac_new[k] = 1.0e-13
    components = [k for k, global_k in enumerate(global_species)
                  if global_k < ncomp]
    def l2norm_source_order(values):
        total = 0.0
        for value in values:
            total += float(value) * float(value)
        return math.sqrt(total / len(values)) if len(values) else 0.0

    norm_update = 0.1 * l2norm_source_order(frac_new)
    damp = 1.0e-2
    dir_prod = 0.0
    delta_g_deficient = delta_g_old.copy()
    fe_deficient = fe_old.copy()
    act_new = activity.copy()
    phase.sendToVCS_ActCoeff(vcs_VolPhase._STATE_OLD, act_new,
                             moles=moles_old)
    x_est = np.zeros(nsp, dtype=float)
    frac_raw = np.zeros(nsp, dtype=float)
    func_stability = float("nan")
    converged = False
    iterations = 0

    for its in range(max_iterations):
        iterations = its + 1
        damp_old = damp
        norm_old = norm_update
        frac_old = frac_new.copy()
        dir_prod_old = dir_prod

        for local_c in components:
            global_c = global_species[local_c]
            frac_old[local_c] = 0.0
            for k, global_k in enumerate(global_species):
                if global_k >= ncomp:
                    frac_old[local_c] += (
                        stoich[global_c, global_k - ncomp] * frac_old[k])

        sum_frac = 0.0
        for value in frac_old:
            sum_frac += float(value)
        if sum_frac <= 0.0:
            sum_frac = 1.0
        sum_xcomp = 0.0
        for k, global_k in enumerate(global_species):
            x_est[k] = frac_old[k] / sum_frac
            if global_k < ncomp:
                sum_xcomp += float(x_est[k])

        phase.setMoleFractionsState(0.0, x_est, vcs_VolPhase._STATE_PHASE_STABILITY)
        phase.sendToVCS_ActCoeff(vcs_VolPhase._STATE_OLD, act_new,
                                 moles=moles_old)

        for local_c in components:
            global_c = global_species[local_c]
            mole_fraction = max(float(x_est[local_c]), VCS_DELETE_MINORSPECIES_CUTOFF)
            fe_deficient[global_c] = (
                fe_old[global_c] + math.log(act_new[global_c] * mole_fraction))

        for i, local_c in enumerate(components):
            global_c = global_species[local_c]
            for global_k in global_species:
                if global_k >= ncomp:
                    irxn = global_k - ncomp
                    if i == 0:
                        delta_g_deficient[irxn] = delta_g_old[irxn]
                    coefficient = stoich[global_c, irxn]
                    if coefficient != 0.0:
                        delta_g_deficient[irxn] += coefficient * (
                            fe_deficient[global_c] - fe_old[global_c])

        e_phi = np.zeros(nsp, dtype=float)
        sum_e_phi = 0.0
        func_stability = sum_xcomp - 1.0
        for k, global_k in enumerate(global_species):
            if global_k >= ncomp:
                irxn = global_k - ncomp
                dg = min(50.0, max(-50.0, float(delta_g_deficient[irxn])))
                e_phi[k] = math.exp(-dg) / act_new[global_k]
                sum_e_phi += float(e_phi[k])
                func_stability += float(e_phi[k])

        for k, global_k in enumerate(global_species):
            b = e_phi[k] / sum_e_phi * (1.0 - sum_xcomp)
            if global_k >= ncomp:
                frac_raw[k] = b

        for local_c in components:
            global_c = global_species[local_c]
            frac_raw[local_c] = 0.0
            for k, global_k in enumerate(global_species):
                if global_k >= ncomp:
                    frac_raw[local_c] += (
                        stoich[global_c, global_k - ncomp] * frac_raw[k])

        del_frac = frac_raw - frac_old
        norm_update = l2norm_source_order(del_frac)
        dir_prod = 0.0
        for k in range(nsp):
            dir_prod += float(frac_old[k] * del_frac[k])
        crossed_sign = dir_prod * dir_prod_old < 0.0

        damp = 0.5
        if damp_old < 0.25:
            damp = 2.0 * damp_old
        if crossed_sign:
            if norm_update * 1.5 > norm_old:
                damp = 0.5 * damp_old
            elif norm_update * 2.0 > norm_old:
                damp = 0.8 * damp_old
        else:
            if norm_update > norm_old * 2.0:
                damp = 0.6 * damp_old
            elif norm_update > norm_old * 1.2:
                damp = 0.9 * damp_old

        for k in range(nsp):
            change = float(del_frac[k])
            old_value = float(frac_old[k])
            if abs(damp * change) > 0.3 * abs(old_value):
                damp = max(0.3 * abs(old_value) / abs(change),
                           1.0e-8 / abs(change))
            if change < 0.0 and 2.0 * damp * (-change) > old_value:
                damp = old_value / (2.0 * -change)
            if change > 0.0 and 2.0 * damp * change > old_value:
                damp = old_value / (2.0 * change)
        damp = max(damp, 1.0e-6)
        for k in range(nsp):
            frac_new[k] = frac_old[k] + damp * del_frac[k]

        if norm_update < 1.0e-5 * damp and its >= min_iterations:
            converged = True
            break

    if converged:
        phase.setMoleFractionsState(
            0.0, x_est, vcs_VolPhase._STATE_PHASE_STABILITY)
        phase.setCreationMoleNumbers(frac_new, creation_rxn_numbers)
    return {
        "func_phase_stability": float(func_stability),
        "converged": converged,
        "iterations": iterations,
        "mole_fractions": x_est.copy(),
        "creation_mole_numbers": frac_new.copy(),
        "activity_coefficients": act_new,
    }


def vcs_elabcheck(abundances, abundance_goals, num_components,
                  element_active, element_types, formula_matrix,
                  mole_numbers, *, check_all_elements=False):
    """Match pinned ``VCS_SOLVE::vcs_elabcheck`` abundance tolerances."""
    actual = np.asarray(abundances, dtype=float)
    goals = np.asarray(abundance_goals, dtype=float)
    active = np.asarray(element_active, dtype=bool)
    types = np.asarray(element_types, dtype=int)
    formula = np.asarray(formula_matrix, dtype=float)
    moles = np.asarray(mole_numbers, dtype=float)
    nelem = actual.size
    if (actual.ndim != 1 or goals.shape != (nelem,) or active.shape != (nelem,)
            or types.shape != (nelem,) or formula.ndim != 2
            or formula.shape[1] != nelem or moles.shape != (formula.shape[0],)):
        raise ValueError("VCS elemental balance arrays have inconsistent dimensions")

    top = nelem if check_all_elements else min(max(int(num_components), 0), nelem)
    for i in range(top):
        error = abs(actual[i] - goals[i])
        if active[i] and error > abs(goals[i]) * 1.0e-12:
            if (types[i] == vcs_VolPhase._ELEM_TYPE_CHARGE_NEUTRALITY
                    and goals[i] != 0.0):
                raise VCSError("Problem with charge neutrality condition")
            if goals[i] == 0.0 or types[i] == vcs_VolPhase._ELEM_TYPE_ELECTRON_CHARGE:
                scale = 1.0e-140
                multisign = False
                for kspec in range(formula.shape[0]):
                    coefficient = formula[kspec, i]
                    if coefficient < 0.0:
                        multisign = True
                    if coefficient != 0.0:
                        scale = max(scale, abs(coefficient * moles[kspec]))
                if multisign:
                    if error > 1.0e-11 * scale:
                        return False
                elif error > 1.0e-140:
                    return False
            else:
                return False
    return True


def vcs_elem_rearrange(formula_matrix, element_abundance_goals, element_active,
                       element_types, num_components):
    """Reorder element rows using pinned ``vcs_elem_rearrange`` MGS logic.

    The formula matrix is species-by-element. The result places the first
    ``num_components`` linearly independent active element columns first.
    """
    formula = np.asarray(formula_matrix, dtype=float).copy()
    goals = np.asarray(element_abundance_goals, dtype=float).copy()
    active = np.asarray(element_active, dtype=bool).copy()
    types = np.asarray(element_types, dtype=int).copy()
    if formula.ndim != 2 or goals.shape != (formula.shape[1],):
        raise ValueError("VCS formula and element-goal dimensions do not match")
    if active.shape != goals.shape or types.shape != goals.shape:
        raise ValueError("VCS element metadata dimensions do not match")
    ncomp = int(num_components)
    if ncomp < 0 or ncomp > min(formula.shape):
        raise ValueError("invalid VCS component count")
    order = list(range(formula.shape[1]))
    q_columns = []
    q_norms = []
    work = goals.copy()
    marker = -1.0e10
    while np.any(work == marker):
        marker -= 1.0
    for jr in range(ncomp):
        while True:
            candidate = next((k for k in range(jr, formula.shape[1])
                              if active[k] and work[k] != marker), None)
            if candidate is None:
                raise VCSError("vcs_elem_rearrange could not find an active independent element")
            work[candidate] = marker
            q = [float(formula[i, candidate]) for i in range(ncomp)]
            for prev, norm_sq in zip(q_columns, q_norms):
                dot = 0.0
                for i in range(ncomp):
                    dot += q[i] * prev[i]
                coefficient = dot / norm_sq
                for i in range(ncomp):
                    q[i] -= coefficient * prev[i]
            norm_sq = 0.0
            for value in q:
                norm_sq += value * value
            if norm_sq > 1.0e-6:
                q_columns.append(q)
                q_norms.append(norm_sq)
                break
        if jr != candidate:
            formula[:, [jr, candidate]] = formula[:, [candidate, jr]]
            goals[[jr, candidate]] = goals[[candidate, jr]]
            active[[jr, candidate]] = active[[candidate, jr]]
            types[[jr, candidate]] = types[[candidate, jr]]
            work[[jr, candidate]] = work[[candidate, jr]]
            order[jr], order[candidate] = order[candidate], order[jr]
    return formula, tuple(order), goals, active, types


def vcs_formation_stoichiometry(formula_matrix, num_components):
    """Solve Cantera's ``C X + B = 0`` component formation system in Python."""
    formula = np.asarray(formula_matrix, dtype=float)
    ncomp = int(num_components)
    if formula.ndim != 2 or ncomp < 0 or ncomp > min(formula.shape):
        raise ValueError("invalid VCS formula matrix or component count")
    n_reactions = formula.shape[0] - ncomp
    if n_reactions == 0:
        return np.empty((ncomp, 0), dtype=float)
    coefficient = DenseMatrix(ncomp, ncomp)
    coefficient._data[:, :] = formula[:ncomp, :ncomp].T
    rhs = -formula[ncomp:, :ncomp].T.copy()
    try:
        return coefficient.solve(rhs)
    except Exception as err:
        raise VCSError("VCS component formula matrix is singular") from err


def vcs_phase_reaction_changes(formation_stoichiometry, species_phase_ids,
                               num_components, num_phases, *, zero_tolerance=1.0e-6):
    """Build phase mole changes and participation counts for VCS reactions.

    Translates the ``EVALUATE DELTA N VALUES`` block in pinned
    ``VCS_SOLVE::vcs_solve_TP``. A formation reaction creates one mole of its
    noncomponent product; component coefficients are accumulated by phase,
    with the native ``1e-6`` coefficient cutoff applied in-place.
    """
    stoich = np.asarray(formation_stoichiometry, dtype=float)
    phase_ids = np.asarray(species_phase_ids, dtype=int)
    ncomp = int(num_components)
    nphase = int(num_phases)
    cutoff = float(zero_tolerance)
    if (stoich.ndim != 2 or phase_ids.ndim != 1 or ncomp < 0
            or nphase < 0 or ncomp > phase_ids.size
            or stoich.shape != (ncomp, phase_ids.size - ncomp)
            or not np.isfinite(cutoff) or cutoff < 0.0
            or np.any(phase_ids < 0) or np.any(phase_ids >= nphase)):
        raise ValueError("VCS phase-reaction arrays have inconsistent dimensions")
    nreactions = stoich.shape[1]
    delta_phase = np.zeros((nphase, nreactions), dtype=float)
    participation = np.zeros((nphase, nreactions), dtype=int)
    for irxn in range(nreactions):
        product_phase = int(phase_ids[ncomp + irxn])
        delta_phase[product_phase, irxn] = 1.0
        participation[product_phase, irxn] += 1
        for jcomp in range(ncomp):
            coefficient = float(stoich[jcomp, irxn])
            if abs(coefficient) <= cutoff:
                stoich[jcomp, irxn] = 0.0
                continue
            component_phase = int(phase_ids[jcomp])
            delta_phase[component_phase, irxn] += coefficient
            participation[component_phase, irxn] += 1
    return delta_phase, participation


def vcs_hessian_act_coeff_diag(reaction_index, stoichiometric_coefficients,
                               reaction_species_indices, species_phase_ids,
                               phase_moles_old, species_single_phase,
                               ln_act_coeff_jacobian):
    """Evaluate pinned ``vcs_Hessian_actCoeff_diag`` from its Jacobian."""
    stoich = np.asarray(stoichiometric_coefficients, dtype=float)
    reaction_species = np.asarray(reaction_species_indices, dtype=int)
    phase_ids = np.asarray(species_phase_ids, dtype=int)
    phase_moles = np.asarray(phase_moles_old, dtype=float)
    single = np.asarray(species_single_phase, dtype=bool)
    jacobian = np.asarray(ln_act_coeff_jacobian, dtype=float)
    if stoich.ndim != 2:
        raise ValueError("VCS Hessian stoichiometry must be a matrix")
    ncomp, nrxn = stoich.shape
    nsp = phase_ids.size
    irxn = int(reaction_index)
    if (reaction_species.shape != (nrxn,) or single.shape != (nsp,)
            or phase_moles.ndim != 1 or jacobian.shape != (nsp, nsp)
            or not 0 <= irxn < nrxn or np.any(reaction_species < ncomp)
            or np.any(reaction_species >= nsp)
            or np.any(phase_ids < 0) or np.any(phase_ids >= phase_moles.size)
            or np.any(phase_moles < 0.0) or np.any(~np.isfinite(jacobian))):
        raise ValueError("VCS Hessian arrays have inconsistent dimensions or values")
    kspec = int(reaction_species[irxn])
    phase = int(phase_ids[kspec])
    phase_moles_product = max(float(phase_moles[phase]), 1.0e-13)
    nu = stoich[:, irxn]
    value = float(jacobian[kspec, kspec]) / phase_moles_product
    for jcomp in range(ncomp):
        if single[jcomp]:
            continue
        for kcomp in range(ncomp):
            if phase_ids[kcomp] == phase_ids[jcomp]:
                moles_in_phase = float(phase_moles[phase_ids[kcomp]])
                if moles_in_phase > 0.0:
                    value += (nu[kcomp] * nu[jcomp]
                              * jacobian[jcomp, kcomp] / moles_in_phase)
        if phase == phase_ids[jcomp]:
            value += (nu[jcomp]
                      * (jacobian[jcomp, kspec] + jacobian[kspec, jcomp])
                      / phase_moles_product)
    return float(value)


def vcs_hessian_diag_adj(reaction_index, ideal_diagonal,
                         stoichiometric_coefficients,
                         reaction_species_indices, species_phase_ids,
                         phase_moles_old, species_single_phase,
                         ln_act_coeff_jacobian):
    """Apply the pinned activity-Hessian safeguard to the ideal diagonal."""
    ideal = float(ideal_diagonal)
    if ideal <= 0.0:
        raise VCSError("VCS Hessian ideal diagonal must be positive")
    activity = vcs_hessian_act_coeff_diag(
        reaction_index, stoichiometric_coefficients,
        reaction_species_indices, species_phase_ids, phase_moles_old,
        species_single_phase, ln_act_coeff_jacobian)
    if activity >= 0.0:
        return ideal + activity
    if abs(activity) < 0.6666 * ideal:
        return ideal + activity
    return ideal - 0.6666 * ideal


def vcs_reaction_step_sizes(
        stoichiometric_coefficients, delta_g_rxn_new, mole_numbers_old,
        species_status, species_unknown_types, species_single_phase,
        species_phase_ids, phase_moles_old, phase_single_species,
        delta_mol_num_phase, total_moles, *, phase_exists=None,
        reaction_species_indices=None, delta_mol_num_species=None,
        num_reactions_reduced=None, tol_major2=1.0e-10,
        use_activity_jacobian=False, ln_act_coeff_jacobian=None,
        force_component_calc=0, special_species=0):
    """Compute VCS formation-reaction steps from the pinned algorithm.

    This ports ``VCS_SOLVE::vcs_RxnStepSizes`` from
    ``src/equil/vcs_solve.cpp`` for the represented OLD/NEW state. A caller
    using the activity-coefficient Hessian path must provide its Jacobian; the
    pinned Hessian adjustment is evaluated directly in Python.
    The returned step vector preserves incoming entries on native ``continue``
    paths and is zeroed only by the native single-species phase deletion path.
    """
    stoich = np.asarray(stoichiometric_coefficients, dtype=float)
    delta_g = np.asarray(delta_g_rxn_new, dtype=float)
    moles = np.asarray(mole_numbers_old, dtype=float)
    status = np.asarray(species_status, dtype=int)
    unknown = np.asarray(species_unknown_types, dtype=int)
    single_species = np.asarray(species_single_phase, dtype=bool)
    phase_ids = np.asarray(species_phase_ids, dtype=int)
    phase_moles = np.asarray(phase_moles_old, dtype=float)
    phase_single = np.asarray(phase_single_species, dtype=bool)
    delta_phase = np.asarray(delta_mol_num_phase, dtype=float)
    if stoich.ndim != 2:
        raise ValueError("VCS stoichiometry must be a matrix")
    ncomp, nreactions = stoich.shape
    nsp = moles.size
    nphase = phase_moles.size
    reduced = (nreactions if num_reactions_reduced is None
               else int(num_reactions_reduced))
    rxn_species = (np.arange(ncomp, ncomp + nreactions, dtype=int)
                   if reaction_species_indices is None else
                   np.asarray(reaction_species_indices, dtype=int))
    steps = (np.zeros(nsp, dtype=float) if delta_mol_num_species is None
             else np.asarray(delta_mol_num_species, dtype=float).copy())
    exists = (np.zeros(nphase, dtype=int) if phase_exists is None
              else np.asarray(phase_exists, dtype=int))
    total = float(total_moles)
    tol = float(tol_major2)
    if (moles.ndim != 1 or ncomp > nsp or delta_g.shape != (nreactions,)
            or status.shape != (nsp,) or unknown.shape != (nsp,)
            or single_species.shape != (nsp,) or phase_ids.shape != (nsp,)
            or phase_single.shape != (nphase,)
            or delta_phase.shape != (nphase, nreactions)
            or rxn_species.shape != (nreactions,) or steps.shape != (nsp,)
            or exists.shape != (nphase,) or reduced < 0 or reduced > nreactions
            or np.any(rxn_species < 0) or np.any(rxn_species >= nsp)
            or np.any(phase_ids < 0) or np.any(phase_ids >= nphase)
            or not np.isfinite(total) or total <= 0.0
            or not np.isfinite(tol) or tol < 0.0
            or np.any(~np.isfinite(stoich)) or np.any(~np.isfinite(delta_g))
            or np.any(~np.isfinite(moles)) or np.any(~np.isfinite(phase_moles))
            or np.any(moles < 0.0) or np.any(phase_moles < 0.0)):
        raise ValueError("VCS reaction-step arrays have inconsistent dimensions or values")
    if use_activity_jacobian and ln_act_coeff_jacobian is None:
        raise NotImplementedError(
            "VCS activity-Jacobian reaction steps require the phase Jacobian")
    if use_activity_jacobian:
        ln_act_coeff_jacobian = np.asarray(ln_act_coeff_jacobian, dtype=float)
        if (ln_act_coeff_jacobian.shape != (nsp, nsp)
                or np.any(~np.isfinite(ln_act_coeff_jacobian))):
            raise ValueError("VCS activity Jacobian has invalid dimensions or values")

    force_component = int(force_component_calc)
    special = int(special_species)
    deleted_phase = None
    for irxn in range(reduced):
        kspec = int(rxn_species[irxn])
        if status[kspec] == VCS_SPECIES_ZEROEDPHASE:
            steps[kspec] = 0.0
            continue
        if unknown[kspec] == VCS_SPECIES_INTERFACIALVOLTAGE:
            continue

        if moles[kspec] == 0.0 and not single_species[kspec]:
            if delta_g[irxn] < -1.0e-4:
                iph = int(phase_ids[kspec])
                phase_fraction = phase_moles[iph] / total
                if exists[iph] != 0 and phase_fraction > VCS_DELETE_PHASE_CUTOFF:
                    steps[kspec] = total * VCS_SMALL_MULTIPHASE_SPECIES
                    if status[kspec] != VCS_SPECIES_STOICHZERO:
                        steps[kspec] *= 10.0
                else:
                    steps[kspec] = 0.0
                    if exists[iph] > 0 and phase_fraction > 0.0:
                        steps[kspec] = total * VCS_SMALL_MULTIPHASE_SPECIES * 10.0
            else:
                steps[kspec] = 0.0
            continue

        if abs(delta_g[irxn]) <= tol:
            continue
        if status[kspec] != VCS_SPECIES_MAJOR and delta_g[irxn] >= 0.0:
            continue

        s = 0.0 if single_species[kspec] else 1.0 / moles[kspec]
        for jcomp in range(ncomp):
            if (not single_species[jcomp] and moles[jcomp] > 0.0):
                s += stoich[jcomp, irxn] ** 2 / moles[jcomp]
        for iph in range(nphase):
            if not phase_single[iph] and phase_moles[iph] > 0.0:
                s -= delta_phase[iph, irxn] ** 2 / phase_moles[iph]

        if s != 0.0:
            if use_activity_jacobian:
                s = vcs_hessian_diag_adj(
                    irxn, s, stoich, rxn_species, phase_ids, phase_moles,
                    single_species, ln_act_coeff_jacobian)
            steps[kspec] = -delta_g[irxn] / s
            for jcomp in range(ncomp):
                coefficient = stoich[jcomp, irxn]
                if coefficient != 0.0:
                    negative_change = -coefficient * steps[kspec]
                    if negative_change > moles[jcomp]:
                        if moles[jcomp] > 0.0:
                            steps[kspec] = -moles[jcomp] / coefficient
                        else:
                            steps[kspec] = 0.0
            if -steps[kspec] > moles[kspec]:
                steps[kspec] = -moles[kspec]
            continue

        # The ideal Hessian is zero: choose the first single-species phase
        # exhausted by the reaction and return immediately, as native does.
        if delta_g[irxn] > 0.0:
            dss = float(moles[kspec])
            limiting_species = kspec
            for jcomp in range(ncomp):
                if stoich[jcomp, irxn] > 0.0:
                    amount = moles[jcomp] / stoich[jcomp, irxn]
                    if amount < dss:
                        dss = amount
                        limiting_species = jcomp
            dss = -dss
        else:
            dss = 1.0e10
            # Native initializes the local limiting-species index to zero for
            # each call; the incoming kSpecial reference is output-only until
            # this branch chooses a phase to delete.
            limiting_species = 0
            for jcomp in range(ncomp):
                if stoich[jcomp, irxn] < 0.0:
                    amount = -moles[jcomp] / stoich[jcomp, irxn]
                    if amount < dss:
                        dss = amount
                        limiting_species = jcomp
        if dss != 0.0:
            if limiting_species == kspec and not single_species[kspec]:
                steps[kspec] = -moles[kspec]
                continue
            steps[:] = 0.0
            steps[kspec] = dss
            steps[:ncomp] = dss * stoich[:, irxn]
            deleted_phase = int(phase_ids[limiting_species])
            special = int(limiting_species)
            if limiting_species != kspec:
                force_component = 1
            return {
                "delta_mol_num_species": steps,
                "deleted_phase": deleted_phase,
                "force_component_calc": force_component,
                "special_species": special,
            }

    return {
        "delta_mol_num_species": steps,
        "deleted_phase": deleted_phase,
        "force_component_calc": force_component,
        "special_species": special,
    }


def vcs_deltag(delta_g_reactions, species_free_energies,
               formation_stoichiometry, reaction_species_indices,
               species_status, component_count, species_moles, *,
               L=0, do_deleted=False, num_reactions_reduced=None,
               old_species_moles=None):
    """Update VCS formation-reaction Gibbs changes using pinned ``vcs_deltag``.

    All values are dimensionless (divided by RT). Reaction ``i`` forms one
    noncomponent species from the component vector in column ``i`` of the
    component-by-reaction stoichiometric matrix. Inputs are copied; the updated
    reaction vector is returned.
    """
    delta_g = np.asarray(delta_g_reactions, dtype=float).copy()
    free_energy = np.asarray(species_free_energies, dtype=float)
    stoich = np.asarray(formation_stoichiometry, dtype=float)
    reaction_species = np.asarray(reaction_species_indices, dtype=int)
    status = np.asarray(species_status, dtype=int)
    moles = np.asarray(species_moles, dtype=float)
    nc = int(component_count)
    if (delta_g.ndim != 1 or free_energy.ndim != 1 or stoich.ndim != 2
            or reaction_species.ndim != 1 or status.ndim != 1
            or moles.ndim != 1 or stoich.shape[0] != nc
            or stoich.shape[1] != reaction_species.size
            or reaction_species.size > delta_g.size
            or free_energy.size != status.size or moles.size != status.size
            or nc > status.size
            or np.any(reaction_species < 0)
            or np.any(reaction_species >= status.size)):
        raise ValueError("VCS deltag arrays have inconsistent dimensions")
    reduced = (reaction_species.size if num_reactions_reduced is None
               else int(num_reactions_reduced))
    if reduced < 0 or reduced > reaction_species.size:
        raise ValueError("reduced reaction count is out of range")
    limit = reaction_species.size if do_deleted else reduced
    old_moles = (moles if old_species_moles is None else
                 np.asarray(old_species_moles, dtype=float))
    if old_moles.shape != moles.shape:
        raise ValueError("old species mole vector has the wrong size")

    if L < 0:
        selected = [i for i in range(reduced)
                    if status[nc + i] != VCS_SPECIES_MINOR]
    elif L == 0:
        selected = range(limit)
    else:
        selected = [i for i in range(reduced)
                    if status[nc + i] <= VCS_SPECIES_MINOR]

    for irxn in selected:
        kspec = int(reaction_species[irxn])
        value = float(free_energy[kspec])
        constrained_by_zero_component = False
        for kcomp in range(nc):
            coefficient = float(stoich[kcomp, irxn])
            value += coefficient * float(free_energy[kcomp])
            component_moles = (float(old_moles[kcomp]) if L > 0
                               else float(moles[kcomp]))
            if (component_moles < VCS_DELETE_MINORSPECIES_CUTOFF
                    and coefficient < 0.0):
                constrained_by_zero_component = True
        if constrained_by_zero_component:
            value = max(0.0, value)
        delta_g[irxn] = value
    return delta_g


def vcs_dfe_species_free_energies(species_free_energies,
                                  ln_mnaught_species, species_moles,
                                  phase_moles, species_phase_ids,
                                  activity_coefficients, species_charges,
                                  phase_potentials, species_status,
                                  single_species_phases,
                                  species_unknown_types,
                                  reaction_species_indices,
                                  component_count, *, faraday_dim,
                                  L=0, lbot=0, ltop=None,
                                  initial_free_energies=None):
    """Evaluate dimensionless VCS chemical potentials using pinned ``vcs_dfe``.

    The phase activity-coefficient update is performed by the caller; these
    arrays must contain values for the selected old or trial state. This
    function translates Cantera's scalar species update and major/minor loops,
    including small-mole, zeroed-phase, solid, and interfacial-voltage cases.
    Inputs are copied and a new free-energy vector is returned.
    """
    standard = np.asarray(species_free_energies, dtype=float)
    ln_mnaught = np.asarray(ln_mnaught_species, dtype=float)
    moles = np.asarray(species_moles, dtype=float)
    phase_totals = np.asarray(phase_moles, dtype=float)
    phase_ids = np.asarray(species_phase_ids, dtype=int)
    activity = np.asarray(activity_coefficients, dtype=float)
    charges = np.asarray(species_charges, dtype=float)
    potentials = np.asarray(phase_potentials, dtype=float)
    status = np.asarray(species_status, dtype=int)
    single = np.asarray(single_species_phases, dtype=bool)
    unknown = np.asarray(species_unknown_types, dtype=int)
    rxn_species = np.asarray(reaction_species_indices, dtype=int)
    nsp = standard.size
    nc = int(component_count)
    vectors = (ln_mnaught, moles, phase_ids, activity, charges, status, single, unknown)
    if (standard.ndim != 1 or any(value.shape != (nsp,) for value in vectors)
            or phase_totals.ndim != 1 or potentials.shape != phase_totals.shape
            or rxn_species.ndim != 1 or nc < 0 or nc > nsp
            or np.any(phase_ids < 0) or np.any(phase_ids >= phase_totals.size)
            or np.any(rxn_species < 0) or np.any(rxn_species >= nsp)
            or not np.isfinite(faraday_dim)):
        raise ValueError("VCS dfe arrays have inconsistent dimensions")
    if np.any(phase_totals < 0.0):
        raise ValueError("VCS phase mole totals must be nonnegative")

    # The native routine checks the cached phase totals before evaluating any
    # chemical potentials. Voltage variables are not species mole numbers.
    calculated_totals = np.zeros(phase_totals.size, dtype=float)
    for k in range(nsp):
        if unknown[k] != VCS_SPECIES_INTERFACIALVOLTAGE:
            calculated_totals[phase_ids[k]] += moles[k]
    if any(not vcs_doubleEqual(float(a), float(b))
           for a, b in zip(calculated_totals, phase_totals)):
        raise VCSError("VCS phase mole totals do not match species mole numbers")

    initial = (standard if initial_free_energies is None else
               np.asarray(initial_free_energies, dtype=float))
    if initial.shape != (nsp,):
        raise ValueError("initial VCS free-energy vector has the wrong size")
    result = initial.copy()

    def update_species(k):
        iph = phase_ids[k]
        phi_term = charges[k] * float(faraday_dim) * potentials[iph]
        if unknown[k] == VCS_SPECIES_INTERFACIALVOLTAGE:
            if moles[k] != potentials[iph] or charges[k] != -1.0:
                raise VCSError("inconsistent VCS interfacial-voltage species")
            return standard[k] + phi_term
        if single[k]:
            return standard[k] + phi_term
        if status[k] in (VCS_SPECIES_ZEROEDMS, VCS_SPECIES_ZEROEDPHASE):
            return standard[k] - ln_mnaught[k] + phi_term

        total = phase_totals[iph]
        if moles[k] <= VCS_DELETE_MINORSPECIES_CUTOFF:
            if total > 0.0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_term = (np.log(activity[k] * VCS_DELETE_MINORSPECIES_CUTOFF)
                                - np.log(total))
            else:
                return standard[k] - ln_mnaught[k] + phi_term
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                log_term = np.log(activity[k] * moles[k]) - np.log(total)
        return standard[k] + log_term - ln_mnaught[k] + phi_term

    if L == 0:
        top = nsp if ltop is None else int(ltop)
        bottom = int(lbot)
        if bottom < 0 or top < bottom or top > nsp:
            raise ValueError("VCS dfe species range is out of bounds")
        selected = range(bottom, top)
    else:
        bottom = int(lbot)
        if bottom < 0 or bottom > nc:
            raise ValueError("VCS dfe component range is out of bounds")
        selected = list(range(bottom, nc))
        for irxn, kspec in enumerate(rxn_species):
            if (L < 0 and status[kspec] != VCS_SPECIES_MINOR
                    or L > 0 and status[kspec] == VCS_SPECIES_MINOR):
                selected.append(int(kspec))
    for k in selected:
        result[k] = update_species(k)
    return result


def vcs_basopt_components(formula_matrix, mole_numbers, element_abundance_goals,
                          element_active, element_types, *, species_sizes=None,
                          single_species_phases=None, species_unknown_types=None,
                          standard_gibbs_rt=None, delta_g_rxn=None,
                          test=-1.0e-10):
    """Select an independent VCS component basis (pinned ``vcs_basopt``).

    This ports the basis-selection half of ``VCS_SOLVE::vcs_basopt``. The
    returned permutation puts selected component species first. Formation
    stoichiometry construction and its voltage-specific correction are a
    separate step and are not claimed by this function.
    """
    formula = np.asarray(formula_matrix, dtype=float)
    moles = np.asarray(mole_numbers, dtype=float)
    goals = np.asarray(element_abundance_goals, dtype=float)
    active = np.asarray(element_active, dtype=bool)
    kinds = np.asarray(element_types, dtype=int)
    if formula.ndim != 2:
        raise ValueError("formula matrix must be two-dimensional")
    n_species, n_elements = formula.shape
    if (moles.shape != (n_species,) or goals.shape != (n_elements,)
            or active.shape != (n_elements,) or kinds.shape != (n_elements,)):
        raise ValueError("VCS basis-selection arrays have inconsistent dimensions")
    if species_sizes is None:
        sizes = np.sum(np.abs(formula), axis=1)
        sizes[sizes == 0.0] = 1.0
    else:
        sizes = np.asarray(species_sizes, dtype=float)
    single_species = (np.zeros(n_species, dtype=bool)
                      if single_species_phases is None
                      else np.asarray(single_species_phases, dtype=bool))
    unknown = (np.zeros(n_species, dtype=int)
               if species_unknown_types is None
               else np.asarray(species_unknown_types, dtype=int))
    gibbs = (np.zeros(n_species, dtype=float) if standard_gibbs_rt is None
             else np.asarray(standard_gibbs_rt, dtype=float))
    nc_trial = min(n_elements, n_species)
    rxn_delta = (np.zeros(n_species - nc_trial, dtype=float)
                 if delta_g_rxn is None else np.asarray(delta_g_rxn, dtype=float))
    if (sizes.shape != (n_species,) or single_species.shape != (n_species,)
            or unknown.shape != (n_species,) or gibbs.shape != (n_species,)
            or rxn_delta.ndim != 1):
        raise ValueError("VCS species metadata has inconsistent dimensions")

    order = list(range(n_species))
    aw = moles.copy()
    aw[unknown == vcs_VolPhase._UNKNOWN_VOLTAGE] = test
    q_basis = np.zeros((n_elements, nc_trial), dtype=float)
    q_norm_squared = np.zeros(nc_trial, dtype=float)
    n_components = nc_trial
    jr = 0

    while jr < nc_trial:
        while True:
            species = order[jr]
            if sizes[species] <= 0.0:
                raise VCSError("vcs_basopt: species size is nonpositive")
            k = jr
            big = aw[jr] * sizes[species] * 1.01
            for i in range(jr + 1, n_species):
                species_i = order[i]
                if sizes[species_i] <= 0.0:
                    raise VCSError("vcs_basopt: species size is nonpositive")
                do_swap = False
                if single_species[species]:
                    do_swap = aw[i] * sizes[species_i] > big
                    if not single_species[species_i] and do_swap:
                        do_swap = aw[i] > aw[k] * 1.001
                elif single_species[species_i]:
                    do_swap = aw[i] * sizes[species_i] > big
                    if not do_swap:
                        do_swap = aw[i] > aw[k] * 1.001
                else:
                    do_swap = aw[i] * sizes[species_i] > big
                if do_swap:
                    k = i
                    big = aw[i] * sizes[species_i] * 1.01

            if (aw[k] != test and aw[k] < 1.0e-140):
                max_conc_poss = 0.0
                k_found = None
                min_nonzeroes = 100000
                for kspec in range(nc_trial, n_species):
                    species = order[kspec]
                    if (aw[kspec] < 0.0
                            or unknown[species] == vcs_VolPhase._UNKNOWN_VOLTAGE):
                        continue
                    max_conc_species = 1.0e10
                    nonzeroes = 0
                    for j in range(n_elements):
                        if active[j] and kinds[j] == 0:
                            nu = formula[species, j]
                            if nu != 0.0:
                                nonzeroes += 1
                                max_conc_species = min(
                                    goals[j] / nu, max_conc_species)
                    if (max_conc_species >= max_conc_poss
                            or max_conc_species > 1.0e-5):
                        if nonzeroes <= min_nonzeroes:
                            if k_found is None or nonzeroes < min_nonzeroes:
                                k_found = kspec
                            elif gibbs[species] <= gibbs[order[k_found]]:
                                k_found = kspec
                        min_nonzeroes = min(min_nonzeroes, nonzeroes)
                        max_conc_poss = max(max_conc_poss, max_conc_species)
                if k_found is None:
                    g_min = 0.0
                    k_found = k
                    for kspec in range(nc_trial, n_species):
                        if aw[kspec] >= 0.0:
                            irxn = kspec - nc_trial
                            if irxn < rxn_delta.size and rxn_delta[irxn] < g_min:
                                g_min = rxn_delta[irxn]
                                k_found = kspec
                k = k_found

            if aw[k] == test:
                n_components = jr
                break

            aw[k] = test
            q = [float(formula[order[k], j]) for j in range(n_elements)]
            for j in range(jr):
                dot = 0.0
                for i in range(n_elements):
                    dot += q[i] * float(q_basis[i, j])
                coefficient = dot / q_norm_squared[j]
                for i in range(n_elements):
                    q[i] -= coefficient * float(q_basis[i, j])
            norm_squared = 0.0
            for value in q:
                norm_squared += value * value
            q_norm_squared[jr] = norm_squared
            if q_norm_squared[jr] > 1.0e-6:
                for i in range(n_elements):
                    q_basis[i, jr] = q[i]
                break

        if n_components != nc_trial and jr >= n_components:
            break
        if aw[k] == test and q_norm_squared[jr] <= 1.0e-6:
            # A dependent candidate was marked unavailable; retry this basis
            # slot until a new independent species is selected or exhausted.
            continue
        if jr != k:
            order[jr], order[k] = order[k], order[jr]
            aw[jr], aw[k] = aw[k], aw[jr]
        jr += 1

    return {
        "num_components": int(n_components),
        "component_species": tuple(order[:n_components]),
        "species_permutation": tuple(order),
        "remaining_species": tuple(order[n_components:]),
        "working_mole_numbers": aw,
        "orthogonal_basis_squared_norms": q_norm_squared[:n_components].copy(),
    }
