"""Fixed-stoichiometry incompressible phase from Cantera StoichSubstance.cpp."""
from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Sequence

from .constants import GasConstant as R, OneAtm
from .units import convert


@dataclass(frozen=True)
class StoichSpecies:
    name: str
    molecular_weight: float
    thermo: object
    charge: float = 0.0
    composition: Mapping[str, float] | None = None


class FixedStoichiometryPhase:
    """Fixed-composition incompressible phase with 1 species."""

    def __init__(self, species: StoichSpecies, density: float, *,
                 temperature: float = 298.15, pressure: float = OneAtm,
                 name: str = "", element_names: Sequence[str] | None = None):
        self.name = name
        self._species = (species,)
        self._density = float(density)
        if not isfinite(self._density) or self._density <= 0:
            raise ValueError("density must be finite and positive")
        self._temperature = float(temperature)
        self._pressure = float(pressure)
        self._p0 = getattr(species.thermo, 'reference_pressure', OneAtm)
        self._electric_potential = 0.0
        if element_names is None:
            element_names = tuple((species.composition or {}).keys())
        self._element_names = tuple(str(name) for name in element_names)

    @classmethod
    def from_mechanism(cls, mech, phase_name=None):
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        if not isinstance(mech, Mechanism):
            mech = load_mechanism(mech)
        phase = mech.phase(phase_name)
        if phase is None or str(phase.thermo).lower() not in {
            'fixed-stoichiometry', 'stoichsubstance', 'stoichiometric-substance'
        }:
            raise ValueError(f"Phase {phase_name!r} is not a fixed-stoichiometry phase")
        if len(phase.species_names) != 1:
            raise ValueError("fixed-stoichiometry phase must have exactly 1 species")
        sp_name = phase.species_names[0]
        by_name = {item.name: item for item in mech.species}
        item = by_name[sp_name]
        thermo_obj = create_species_thermo(item.thermo)
        sp = StoichSpecies(
            name=sp_name,
            molecular_weight=item.molecular_weight,
            thermo=thermo_obj,
            charge=getattr(item, 'charge', 0.0),
            composition=dict(item.composition),
        )
        # Parse density
        density = 1000.0  # default 1000 kg/m^3 if unspecified
        eos = (getattr(item, 'equation_of_state', None) or
               (getattr(item, 'input_data', None) or {}).get('equation-of-state') or
               getattr(phase, 'equation_of_state', None) or {})
        if isinstance(eos, dict):
            if 'density' in eos:
                raw_d = eos['density']
                if isinstance(raw_d, (int, float)):
                    length = (mech.units or {}).get('length', 'm')
                    mass = (mech.units or {}).get('mass', 'kg')
                    density = float(raw_d) * convert(1.0, mass, 'kg') / (convert(1.0, length, 'm') ** 3)
                elif isinstance(raw_d, str):
                    parts = raw_d.split()
                    density = convert(float(parts[0]), parts[1], 'kg/m^3') if len(parts) == 2 else float(parts[0])
            elif 'molar-density' in eos:
                raw_d = eos['molar-density']
                quantity = (mech.units or {}).get('quantity', 'kmol')
                length = (mech.units or {}).get('length', 'm')
                c_unit = f"{quantity}/{length}^3"
                if isinstance(raw_d, (int, float)):
                    m_dens = float(raw_d) * convert(1.0, quantity, 'kmol') / (convert(1.0, length, 'm') ** 3)
                elif isinstance(raw_d, str):
                    parts = raw_d.split()
                    m_dens = convert(float(parts[0]), parts[1], 'kmol/m^3') if len(parts) == 2 else float(parts[0])
                density = m_dens * item.molecular_weight
            elif 'molar-volume' in eos:
                raw_d = eos['molar-volume']
                quantity = (mech.units or {}).get('quantity', 'kmol')
                length = (mech.units or {}).get('length', 'm')
                if isinstance(raw_d, (int, float)):
                    m_vol = float(raw_d) * (convert(1.0, length, 'm') ** 3) / convert(1.0, quantity, 'kmol')
                elif isinstance(raw_d, str):
                    parts = raw_d.split()
                    m_vol = convert(float(parts[0]), parts[1], 'm^3/kmol') if len(parts) == 2 else float(parts[0])
                density = item.molecular_weight / m_vol
        elif getattr(phase, 'density', None) is not None:
            raw_d = phase.density
            if isinstance(raw_d, (int, float)):
                density = float(raw_d)
            elif isinstance(raw_d, str):
                parts = raw_d.split()
                density = convert(float(parts[0]), parts[1], 'kg/m^3') if len(parts) == 2 else float(parts[0])

        raw_elements = (phase.input_data or {}).get('elements')
        if isinstance(raw_elements, (list, tuple)):
            element_names = tuple(str(name) for name in raw_elements)
        else:
            element_names = tuple(sp.composition or {})
        obj = cls(sp, density, name=phase.name, element_names=element_names)
        state = phase.state or {}
        if 'T' in state:
            raw_T = state['T']
            if isinstance(raw_T, (int, float)):
                obj.T = float(raw_T)
            elif isinstance(raw_T, str):
                parts = raw_T.split()
                obj.T = convert(float(parts[0]), parts[1], 'K') if len(parts) == 2 else float(parts[0])
        if 'P' in state:
            raw_P = state['P']
            if isinstance(raw_P, (int, float)):
                obj.P = float(raw_P)
            elif isinstance(raw_P, str):
                parts = raw_P.split()
                obj.P = convert(float(parts[0]), parts[1], 'Pa') if len(parts) == 2 else float(parts[0])
        return obj

    @property
    def species(self):
        return self._species

    @property
    def species_names(self):
        return (self._species[0].name,)

    def species_name(self, index):
        if int(index) != 0:
            raise IndexError("fixed-stoichiometry phase has one species")
        return self._species[0].name

    speciesName = species_name

    def species_index(self, name, error_if_not_found=True):
        if str(name) == self._species[0].name:
            return 0
        if error_if_not_found:
            raise ValueError(f"No such species '{name}' in phase {self.name!r}")
        return -1

    speciesIndex = species_index

    @property
    def element_names(self):
        return self._element_names

    @property
    def n_elements(self):
        return len(self._element_names)

    nElements = n_elements

    def element_name(self, index):
        return self._element_names[int(index)]

    elementName = element_name

    def element_index(self, name, error_if_not_found=True):
        try:
            return self._element_names.index(str(name))
        except ValueError:
            if error_if_not_found:
                raise ValueError(f"No such element '{name}' in phase {self.name!r}")
            return -1

    elementIndex = element_index

    def n_atoms(self, species, element):
        if isinstance(species, int):
            if species != 0:
                raise IndexError("fixed-stoichiometry phase has one species")
        elif str(species) != self._species[0].name:
            raise ValueError(f"Unknown species {species!r}")
        if isinstance(element, int):
            element = self.element_name(element)
        return float((self._species[0].composition or {}).get(str(element), 0.0))

    nAtoms = n_atoms

    @property
    def n_species(self):
        return 1

    nSpecies = n_species

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        v = float(value)
        if not isfinite(v) or v <= 0:
            raise ValueError("temperature must be finite and positive")
        self._temperature = v

    T = property(lambda self: self.temperature, lambda self, v: setattr(self, 'temperature', v))

    @property
    def pressure(self):
        return self._pressure

    @pressure.setter
    def pressure(self, value):
        v = float(value)
        if not isfinite(v) or v <= 0:
            raise ValueError("pressure must be finite and positive")
        self._pressure = v

    P = property(lambda self: self.pressure, lambda self, v: setattr(self, 'pressure', v))

    @property
    def TP(self):
        return self.temperature, self.pressure

    @TP.setter
    def TP(self, values):
        t, p = values
        self.temperature = t
        self.pressure = p

    def setState_TP(self, t, p):
        self.TP = (t, p)

    @property
    def density(self):
        return self._density

    @property
    def molecular_weights(self):
        return (self._species[0].molecular_weight,)

    @property
    def mean_molecular_weight(self):
        return self._species[0].molecular_weight

    @property
    def density_mole(self):
        return self.density / self.mean_molecular_weight

    molar_density = property(lambda self: self.density_mole)

    @property
    def molar_volume(self):
        return 1.0 / self.density_mole

    @property
    def X(self):
        return (1.0,)

    @property
    def activities(self):
        return (1.0,)

    @property
    def activity_concentrations(self):
        return (1.0,)

    @property
    def standard_concentrations(self):
        return (1.0,)

    def standard_concentration(self, k=0):
        return 1.0

    def log_standard_conc(self, k=0):
        return 0.0

    logStandardConc = log_standard_conc

    @property
    def standard_concentration_units(self):
        return 1.0

    @property
    def charges(self):
        return (self._species[0].charge,)

    def charge(self, k=0):
        return self._species[0].charge

    @property
    def electric_potential(self):
        return self._electric_potential

    @electric_potential.setter
    def electric_potential(self, v):
        self._electric_potential = float(v)

    def set_electric_potential(self, v):
        self.electric_potential = v

    setElectricPotential = set_electric_potential

    @property
    def phase_of_matter(self):
        return 'solid'

    @property
    def n_dim(self):
        return 3

    def _ref_properties(self):
        cp_r, h_rt, s_r = self._species[0].thermo.properties(self.T)
        return cp_r, h_rt, s_r

    @property
    def partial_molar_cp(self):
        cp_r, _, _ = self._ref_properties()
        return (R * cp_r,)

    @property
    def partial_molar_enthalpies(self):
        _, h_rt, _ = self._ref_properties()
        h_ref = R * self.T * h_rt
        pres_correct = (self.P - self._p0) / self.density_mole
        return (h_ref + pres_correct,)

    @property
    def partial_molar_entropies(self):
        _, _, s_r = self._ref_properties()
        return (R * s_r,)

    @property
    def standard_chemical_potentials(self):
        h = self.partial_molar_enthalpies[0]
        s = self.partial_molar_entropies[0]
        return (h - self.T * s,)

    @property
    def chemical_potentials(self):
        return self.standard_chemical_potentials

    @property
    def activity_concentrations(self):
        return (1.0,)

    @property
    def standard_concentrations(self):
        return (1.0,)

    def standard_concentration(self, k=0):
        return 1.0

    def log_standard_conc(self, k=0):
        return 0.0

    logStandardConc = log_standard_conc

    @property
    def cp_mole(self):
        return self.partial_molar_cp[0]

    @property
    def enthalpy_mole(self):
        return self.partial_molar_enthalpies[0]

    @property
    def entropy_mole(self):
        return self.partial_molar_entropies[0]

    @property
    def gibbs_mole(self):
        return self.standard_chemical_potentials[0]
