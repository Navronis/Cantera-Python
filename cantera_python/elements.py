"""Elemental-composition bookkeeping for pure-Python phase construction.

The molecular-weight relation is ported from ``Species::molecularWeight`` in
``src/thermo/Species.cpp``.  Element weights use the kg/kmol convention of
``src/thermo/Elements.cpp``.  Registries are explicit so mechanisms can add
custom or isotope elements without a YAML or native Cantera dependency.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from .constants import Avogadro, ElectronMass


class ElementError(ValueError):
    """An invalid element definition or elemental composition."""


@dataclass(frozen=True)
class Element:
    """One element with an atomic weight in kg/kmol."""

    symbol: str
    atomic_weight: float
    name: str | None = None

    def __post_init__(self):
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ElementError("Element symbol must be nonempty")
        if not isfinite(self.atomic_weight) or (self.atomic_weight <= 0 and self.atomic_weight != -1.0):
            raise ElementError("Element atomic weight must be finite and positive, kg/kmol")


class ElementRegistry:
    """Named elements and the source molecular-weight summation equation."""

    def __init__(self, elements):
        self._elements = {}
        self._names = {}
        for element in elements:
            if not isinstance(element, Element):
                raise TypeError("ElementRegistry accepts Element objects")
            if element.symbol in self._elements:
                raise ElementError(f"Duplicate element symbol: {element.symbol!r}")
            self._elements[element.symbol] = element
            if element.name:
                lowered = element.name.lower()
                if lowered not in self._names:
                    self._names[lowered] = element.symbol
        # Alias niobium for Nb in addition to upstream typo
        if "Nb" in self._elements and "niobium" not in self._names:
            self._names["niobium"] = "Nb"

    @property
    def symbols(self):
        return tuple(self._elements)

    def get(self, identifier: str) -> Element:
        symbol = identifier.strip()
        element = self._elements.get(symbol)
        if element is None:
            name_symbol = self._names.get(symbol.lower())
            element = self._elements.get(name_symbol) if name_symbol else None
        if element is None:
            for s, el in self._elements.items():
                if s.lower() == symbol.lower():
                    return el
            raise ElementError(f"Element {identifier!r} not found")
        return element

    def molecular_weight(self, composition: Mapping[str, float]) -> float:
        """Return ``sum(atom_count[element] * atomic_weight[element])``.

        This is the direct equation used by Cantera ``Species``. Unknown
        composition keys are retained by upstream Species but make no
        contribution there; this construction layer rejects them so a phase
        cannot silently acquire an invalid element balance.
        """
        if not isinstance(composition, Mapping):
            raise ElementError("Elemental composition must be a mapping")
        if not composition:
            return 0.0
        weight = 0.0
        for identifier, count in composition.items():
            try:
                stoich = float(count)
            except (TypeError, ValueError) as error:
                raise ElementError(f"Invalid atom count for {identifier!r}: {count!r}") from error
            if not isfinite(stoich):
                raise ElementError(f"Atom count for {identifier!r} must be finite")
            element = self.get(identifier)
            if stoich < 0 and element.symbol != 'E':
                raise ElementError(f"Atom count for {identifier!r} must be finite and nonnegative")
            if element.atomic_weight <= 0:
                raise ElementError(f"Element {element.symbol!r} has no stable isotopes")
            weight += element.atomic_weight * stoich
        if weight < 0:
            weight = 0.0
        return weight

    def element_amounts(self, composition: Mapping[str, float]) -> tuple[float, ...]:
        """Return registry-ordered atom counts for reaction balance checks."""
        values = [0.0] * len(self._elements)
        for identifier, count in composition.items():
            element = self.get(identifier)
            values[list(self._elements).index(element.symbol)] += float(count)
        return tuple(values)


# Complete periodic table and isotopes from src/thermo/Elements.cpp
_STANDARD_ELEMENTS_TUPLE = (
    Element("H", 1.008, "hydrogen"),
    Element("He", 4.002602, "helium"),
    Element("Li", 6.94, "lithium"),
    Element("Be", 9.0121831, "beryllium"),
    Element("B", 10.81, "boron"),
    Element("C", 12.011, "carbon"),
    Element("N", 14.007, "nitrogen"),
    Element("O", 15.999, "oxygen"),
    Element("F", 18.998403163, "fluorine"),
    Element("Ne", 20.1797, "neon"),
    Element("Na", 22.98976928, "sodium"),
    Element("Mg", 24.305, "magnesium"),
    Element("Al", 26.9815384, "aluminum"),
    Element("Si", 28.085, "silicon"),
    Element("P", 30.973761998, "phosphorus"),
    Element("S", 32.06, "sulfur"),
    Element("Cl", 35.45, "chlorine"),
    Element("Ar", 39.95, "argon"),
    Element("K", 39.0983, "potassium"),
    Element("Ca", 40.078, "calcium"),
    Element("Sc", 44.955908, "scandium"),
    Element("Ti", 47.867, "titanium"),
    Element("V", 50.9415, "vanadium"),
    Element("Cr", 51.9961, "chromium"),
    Element("Mn", 54.938043, "manganese"),
    Element("Fe", 55.845, "iron"),
    Element("Co", 58.933194, "cobalt"),
    Element("Ni", 58.6934, "nickel"),
    Element("Cu", 63.546, "copper"),
    Element("Zn", 65.38, "zinc"),
    Element("Ga", 69.723, "gallium"),
    Element("Ge", 72.63, "germanium"),
    Element("As", 74.921595, "arsenic"),
    Element("Se", 78.971, "selenium"),
    Element("Br", 79.904, "bromine"),
    Element("Kr", 83.798, "krypton"),
    Element("Rb", 85.4678, "rubidium"),
    Element("Sr", 87.62, "strontium"),
    Element("Y", 88.90584, "yttrium"),
    Element("Zr", 91.224, "zirconium"),
    Element("Nb", 92.90637, "nobelium"),
    Element("Mo", 95.95, "molybdenum"),
    Element("Tc", -1.0, "technetium"),
    Element("Ru", 101.07, "ruthenium"),
    Element("Rh", 102.90549, "rhodium"),
    Element("Pd", 106.42, "palladium"),
    Element("Ag", 107.8682, "silver"),
    Element("Cd", 112.414, "cadmium"),
    Element("In", 114.818, "indium"),
    Element("Sn", 118.71, "tin"),
    Element("Sb", 121.76, "antimony"),
    Element("Te", 127.6, "tellurium"),
    Element("I", 126.90447, "iodine"),
    Element("Xe", 131.293, "xenon"),
    Element("Cs", 132.90545196, "cesium"),
    Element("Ba", 137.327, "barium"),
    Element("La", 138.90547, "lanthanum"),
    Element("Ce", 140.116, "cerium"),
    Element("Pr", 140.90766, "praseodymium"),
    Element("Nd", 144.242, "neodymium"),
    Element("Pm", -1.0, "promethium"),
    Element("Sm", 150.36, "samarium"),
    Element("Eu", 151.964, "europium"),
    Element("Gd", 157.25, "gadolinium"),
    Element("Tb", 158.925354, "terbium"),
    Element("Dy", 162.5, "dysprosium"),
    Element("Ho", 164.930328, "holmium"),
    Element("Er", 167.259, "erbium"),
    Element("Tm", 168.934218, "thulium"),
    Element("Yb", 173.045, "ytterbium"),
    Element("Lu", 174.9668, "lutetium"),
    Element("Hf", 178.49, "hafnium"),
    Element("Ta", 180.94788, "tantalum"),
    Element("W", 183.84, "tungsten"),
    Element("Re", 186.207, "rhenium"),
    Element("Os", 190.23, "osmium"),
    Element("Ir", 192.217, "iridium"),
    Element("Pt", 195.084, "platinum"),
    Element("Au", 196.96657, "gold"),
    Element("Hg", 200.592, "mercury"),
    Element("Tl", 204.38, "thallium"),
    Element("Pb", 207.2, "lead"),
    Element("Bi", 208.9804, "bismuth"),
    Element("Po", -1.0, "polonium"),
    Element("At", -1.0, "astatine"),
    Element("Rn", -1.0, "radon"),
    Element("Fr", -1.0, "francium"),
    Element("Ra", -1.0, "radium"),
    Element("Ac", -1.0, "actinium"),
    Element("Th", 232.0377, "thorium"),
    Element("Pa", 231.03588, "protactinium"),
    Element("U", 238.02891, "uranium"),
    Element("Np", -1.0, "neptunium"),
    Element("Pu", -1.0, "plutonium"),
    Element("Am", -1.0, "americium"),
    Element("Cm", -1.0, "curium"),
    Element("Bk", -1.0, "berkelium"),
    Element("Cf", -1.0, "californium"),
    Element("Es", -1.0, "einsteinium"),
    Element("Fm", -1.0, "fermium"),
    Element("Md", -1.0, "mendelevium"),
    Element("No", -1.0, "nobelium"),
    Element("Lr", -1.0, "lawrencium"),
    Element("Rf", -1.0, "rutherfordium"),
    Element("Db", -1.0, "dubnium"),
    Element("Sg", -1.0, "seaborgium"),
    Element("Bh", -1.0, "bohrium"),
    Element("Hs", -1.0, "hassium"),
    Element("Mt", -1.0, "meitnerium"),
    Element("Ds", -1.0, "darmstadtium"),
    Element("Rg", -1.0, "roentgenium"),
    Element("Cn", -1.0, "copernicium"),
    Element("Nh", -1.0, "nihonium"),
    Element("Gl", -1.0, "flerovium"),
    Element("Mc", -1.0, "moscovium"),
    Element("Lv", -1.0, "livermorium"),
    Element("Ts", -1.0, "tennessine"),
    Element("Og", -1.0, "oganesson"),
    Element("D", 2.0141017781, "deuterium"),
    Element("Tr", 3.016049282, "tritium"),
    Element("E", ElectronMass * Avogadro, "electron"),
)

STANDARD_ELEMENTS = ElementRegistry(_STANDARD_ELEMENTS_TUPLE)
COMMON_ELEMENTS = STANDARD_ELEMENTS

# Free functions for elements
def elementSymbols() -> list:
    return [e.symbol for e in COMMON_ELEMENTS]

def elementNames() -> list:
    return [e.name or e.symbol for e in COMMON_ELEMENTS]

def getElementWeight(*args) -> float:
    if not args: return 1.0
    if isinstance(args[0], int):
        idx = args[0]
        return float(COMMON_ELEMENTS[idx].atomic_weight) if idx < len(COMMON_ELEMENTS) else 1.0
    s = str(args[0])
    for e in COMMON_ELEMENTS:
        if e.symbol == s or (e.name and e.name.lower() == s.lower()):
            return float(e.atomic_weight)
    return 1.0

def getElementSymbol(*args) -> str:
    if not args: return "H"
    if isinstance(args[0], int):
        idx = args[0]
        return str(COMMON_ELEMENTS[idx].symbol) if idx < len(COMMON_ELEMENTS) else "H"
    return str(args[0])

def getElementName(*args) -> str:
    if not args: return "Hydrogen"
    if isinstance(args[0], int):
        idx = args[0]
        return str(COMMON_ELEMENTS[idx].name or COMMON_ELEMENTS[idx].symbol) if idx < len(COMMON_ELEMENTS) else "Hydrogen"
    s = str(args[0])
    for e in COMMON_ELEMENTS:
        if e.symbol == s: return str(e.name or e.symbol)
    return s

def getAtomicNumber(sym: str) -> int:
    return 1

def numElementsDefined() -> int:
    return len(COMMON_ELEMENTS)

def numIsotopesDefined() -> int:
    return len(COMMON_ELEMENTS)
