"""Cantera-Python: Unofficial Experimental Pure-Python Cantera Solver Reconstruction.

This release publishes Solver Wave 1: Thermodynamic Inverse-State Solvers
(setState_HP, setState_UV, setState_SP, setState_SV, and their unified
bracketed Newton-bisection contraction solvers).

Pinned upstream Cantera commit: 726522be4e2a13454d8415b7ef799d621f665cf3.
Full Cantera library port is NOT complete.
"""
from __future__ import annotations
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

from .constants import GasConstant, OneAtm, CanteraError
from .composition import CompositionError, SpeciesRegistry, parse_comp_string
from .elements import COMMON_ELEMENTS, Element, ElementError, ElementRegistry
from .units import UnitError, Units, UnitStack, UnitSystem, convert, convert_activation_energy, parse_units
from .anymap import AnyBase, AnyValue, AnyMap, Application, AnyMapError
from .mechanism import Mechanism, MechanismError, MechanismPhase, MechanismSpecies, load_mechanism
from .multi_species_thermo import MultiSpeciesThermo
from .thermo import (
    ThermoPhase,
    NasaPoly1,
    NasaPoly2,
    Nasa9Poly1,
    Nasa9PolyMultiTempRegion,
    ShomatePoly,
    ShomatePoly2,
    ConstCpPoly,
    Mu0Poly,
    create_species_thermo,
)
from .state import (
    set_state_td,
    set_state_tp,
    set_state_hp,
    set_state_sp,
    set_state_uv,
    set_state_sv,
)
from .ideal_gas import IdealGasMixture, IdealGasPhase, Species
from .equilibrium import EquilibriumError, equilibrate_ideal_gas
from .solution import Solution

__version__ = "0.1.0"
UPSTREAM_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"
SOLVER_WAVE = 1
PORT_COMPLETE = False
FULL_CANTERA_PORT_COMPLETE = False

__all__ = [
    "GasConstant",
    "OneAtm",
    "CanteraError",
    "Units",
    "convert",
    "parse_units",
    "Element",
    "parse_comp_string",
    "AnyMap",
    "load_mechanism",
    "MultiSpeciesThermo",
    "ThermoPhase",
    "IdealGasPhase",
    "IdealGasMixture",
    "Species",
    "Solution",
    "set_state_td",
    "set_state_tp",
    "set_state_hp",
    "set_state_sp",
    "set_state_uv",
    "set_state_sv",
    "equilibrate_ideal_gas",
    "DATA_DIR",
    "__version__",
    "UPSTREAM_COMMIT",
    "SOLVER_WAVE",
    "PORT_COMPLETE",
    "FULL_CANTERA_PORT_COMPLETE",
]
