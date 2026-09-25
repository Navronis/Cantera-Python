"""Scalar ideal-gas mixture property equations from Cantera, in Python.

Sources: IdealGasPhase.{h,cpp}, Phase.cpp, ThermoPhase.h. This is an explicit
immutable-state interface to selected property equations, not a port of the
full ThermoPhase/Solution API. No kinetics, equilibrium, or time integration
is implied. Species use kg/kmol; molar properties use J/kmol and J/kmol/K.
"""
from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from .constants import CanteraError, GasConstant as R, OneAtm, SmallNumber
from .composition import parse_comp_string


@dataclass(frozen=True)
class Species:
    name: str
    molecular_weight: float
    thermo: object
    charge: float = 0.0

    def __post_init__(self):
        if not self.name:
            raise ValueError('Species must have a name')
        if not isfinite(self.molecular_weight) or self.molecular_weight <= 0:
            raise ValueError('Molecular weight must be finite and positive, kg/kmol')
        if not callable(getattr(self.thermo, 'properties', None)):
            raise TypeError('Species thermo must provide properties(T)')
        if not isfinite(self.charge):
            raise ValueError('Species charge must be finite')


class IdealGasMixture:
    """Evaluate ideal-gas properties for one fixed T, P, and composition.

    Use .at(temperature=..., pressure=..., mole_fractions=...) for a new state.
    Positive input mole amounts are normalized. Unknown species and negative
    amounts raise rather than being silently discarded. All species must use
    the same reference pressure, as required by upstream MultiSpeciesThermo.
    """

    def __init__(self, species, *, temperature=298.15, pressure=OneAtm,
                 mole_fractions=None):
        self._species = tuple(species)
        if not self._species or len({s.name for s in self._species}) != len(self._species):
            raise ValueError('Provide one or more species with distinct names')
        self._temperature = float(temperature)
        self._pressure = float(pressure)
        if not isfinite(self._temperature) or self._temperature <= 0:
            raise ValueError('Temperature must be finite and positive, K')
        if not isfinite(self._pressure) or self._pressure <= 0:
            raise ValueError('Pressure must be finite and positive, Pa')
        self._reference_pressure = float(self._species[0].thermo.reference_pressure)
        if any(s.thermo.reference_pressure != self._reference_pressure for s in self._species):
            raise ValueError('All species must have the same reference pressure')
        if mole_fractions is None:
            amounts = (1.0,) + (0.0,) * (len(self._species) - 1)
        elif isinstance(mole_fractions, str):
            parsed = parse_comp_string(mole_fractions, self.species_names)
            amounts = tuple(parsed[s.name] for s in self._species)
        elif isinstance(mole_fractions, Mapping):
            unknown = set(mole_fractions) - set(self.species_names)
            if unknown:
                raise ValueError(f'Unknown species: {sorted(unknown)}')
            amounts = tuple(float(mole_fractions.get(s.name, 0.0)) for s in self._species)
        else:
            amounts = tuple(float(x) for x in mole_fractions)
        if len(amounts) != len(self._species):
            raise ValueError('Composition length must equal the species count')
        if any(not isfinite(x) or x < -1e-7 for x in amounts):
            raise ValueError('Mole amounts must be finite and nonnegative')
        amounts = tuple(max(0.0, x) for x in amounts)
        total = sum(amounts)
        if not isfinite(total) or total <= 0:
            raise ValueError('Mole amounts must have a finite positive sum')
        self._x = tuple(x / total for x in amounts)
        # Additive species enthalpy perturbations in J/kmol. These mirror
        # MultiSpeciesThermo::modifyOneHf298SS: h, u, g, and chemical
        # potentials shift while cp and entropy remain unchanged.
        self._enthalpy_offsets = (0.0,) * len(self._species)

    @property
    def species(self):
        return self._species

    @property
    def species_names(self):
        return tuple(s.name for s in self._species)

    @property
    def temperature(self):
        return self._temperature

    @property
    def pressure(self):
        if getattr(self, '_density_override', None) is not None:
            # Preserve Phase's independent density state and the operation
            # ordering of IdealGasPhase::pressure(): R * (rho / Wbar) * T.
            return R * (self._density_override / self.mean_molecular_weight) * self.temperature
        return self._pressure

    @property
    def reference_pressure(self):
        return self._reference_pressure

    @property
    def X(self):
        return self._x

    @property
    def Y(self):
        return tuple(x * s.molecular_weight / self.mean_molecular_weight
                     for x, s in zip(self.X, self.species))

    def at(self, *, temperature=None, pressure=None, mole_fractions=None):
        state = type(self)(
            self.species,
            temperature=self.temperature if temperature is None else temperature,
            pressure=self.pressure if pressure is None else pressure,
            mole_fractions=self.X if mole_fractions is None else mole_fractions)
        state._enthalpy_offsets = self._enthalpy_offsets
        return state

    def with_enthalpy_offset(self, species_index, offset):
        """Clone the state with one additive standard enthalpy perturbation."""
        species_index = int(species_index)
        if species_index < 0 or species_index >= len(self._species):
            raise IndexError(f'species index {species_index} is out of range')
        offset = float(offset)
        if not isfinite(offset):
            raise ValueError('species enthalpy offset must be finite, J/kmol')
        state = self._at_density_unchecked(
            self.temperature, self.density) if getattr(
                self, '_density_override', None) is not None else self.at()
        values = list(self._enthalpy_offsets)
        values[species_index] = offset
        state._enthalpy_offsets = tuple(values)
        return state

    def _at_density_unchecked(self, temperature, density):
        """Clone with Phase-style independent density, without revalidation.

        This internal constructor is needed for faithful mutating setter
        semantics. The pinned ``Phase`` stores temperature and density as its
        independent state and accepts positive infinity; pressure is derived.
        Public construction continues to validate ordinary T/P inputs.
        """
        state = object.__new__(type(self))
        state._species = self._species
        state._temperature = float(temperature)
        state._pressure = None
        state._density_override = float(density)
        state._reference_pressure = self._reference_pressure
        state._x = self._x
        state._enthalpy_offsets = self._enthalpy_offsets
        return state

    def _reference(self):
        values = []
        for species, offset in zip(self.species, self._enthalpy_offsets):
            cp_R, h_RT, s_R = species.thermo.properties(self.temperature)
            values.append((cp_R, h_RT + offset / (R * self.temperature), s_R))
        return tuple(values)

    def _mean(self, values):
        return sum(x * value for x, value in zip(self.X, values))

    @property
    def mean_molecular_weight(self):
        return self._mean(s.molecular_weight for s in self.species)

    @property
    def molar_density(self):
        if getattr(self, '_density_override', None) is not None:
            return self._density_override / self.mean_molecular_weight
        return self.pressure / (R * self.temperature)

    @property
    def density(self):
        if getattr(self, '_density_override', None) is not None:
            return self._density_override
        return self.molar_density * self.mean_molecular_weight

    @property
    def concentrations(self):
        return tuple(x * self.molar_density for x in self.X)

    @property
    def cp_mole(self):
        return R * self._mean(p[0] for p in self._reference())

    @property
    def cv_mole(self):
        return self.cp_mole - R

    @property
    def enthalpy_mole(self):
        return R * self.temperature * self._mean(p[1] for p in self._reference())

    @property
    def int_energy_mole(self):
        return self.enthalpy_mole - R * self.temperature

    def _sum_xlogx(self):
        # Preserve Phase::sum_xlogx's mole-per-unit-mass floor and expression.
        mw = self.mean_molecular_weight
        ym = tuple(x / mw for x in self.X)
        return mw * sum(v * log(max(v, SmallNumber)) for v in ym) + log(mw)

    @property
    def entropy_mole(self):
        return R * (self._mean(p[2] for p in self._reference()) - self._sum_xlogx()
                    - log(self.pressure / self.reference_pressure))

    @property
    def gibbs_mole(self):
        return self.enthalpy_mole - self.temperature * self.entropy_mole

    @property
    def cp_mass(self):
        return self.cp_mole / self.mean_molecular_weight

    @property
    def cv_mass(self):
        return self.cv_mole / self.mean_molecular_weight

    @property
    def enthalpy_mass(self):
        return self.enthalpy_mole / self.mean_molecular_weight

    @property
    def int_energy_mass(self):
        return self.int_energy_mole / self.mean_molecular_weight

    @property
    def entropy_mass(self):
        return self.entropy_mole / self.mean_molecular_weight

    @property
    def gibbs_mass(self):
        return self.gibbs_mole / self.mean_molecular_weight

    @property
    def sound_speed(self):
        return sqrt(self.cp_mole / self.cv_mole * R
                    / self.mean_molecular_weight * self.temperature)

    @property
    def isothermal_compressibility(self):
        return 1.0 / self.pressure

    @property
    def thermal_expansion_coeff(self):
        return 1.0 / self.temperature

    @property
    def internal_pressure(self):
        """Internal pressure, ``T*(dP/dT)_V - P``, in Pa.

        For an ideal gas ``(dP/dT)_V = P/T``, so this is identically zero.
        """
        return 0.0

    @property
    def activities(self):
        return self.X

    @property
    def activity_coefficients(self):
        return (1.0,) * len(self.species)

    @property
    def activity_concentrations(self):
        return self.concentrations

    @property
    def standard_concentration(self):
        return self.molar_density

    @property
    def partial_molar_enthalpies(self):
        return tuple(R * self.temperature * p[1] for p in self._reference())

    @property
    def partial_molar_int_energies(self):
        return tuple(R * self.temperature * (p[1] - 1.0) for p in self._reference())

    @property
    def partial_molar_cp(self):
        return tuple(R * p[0] for p in self._reference())

    @property
    def partial_molar_volumes(self):
        return (1.0 / self.molar_density,) * len(self.species)

    @property
    def partial_molar_entropies(self):
        logp = log(self.pressure / self.reference_pressure)
        return tuple(R * (p[2] - logp - log(max(x, SmallNumber)))
                     for p, x in zip(self._reference(), self.X))

    @property
    def standard_chemical_potentials(self):
        logp = log(self.pressure / self.reference_pressure)
        return tuple(R * self.temperature * (p[1] - p[2] + logp)
                     for p in self._reference())

    @property
    def chemical_potentials(self):
        return tuple(mu + R * self.temperature * log(max(x, SmallNumber))
                     for mu, x in zip(self.standard_chemical_potentials, self.X))

    def _mass_to_mole(self, mass_fractions):
        if isinstance(mass_fractions, str):
            mass_fractions = parse_comp_string(mass_fractions, self.species_names)
        if isinstance(mass_fractions, Mapping):
            y = [float(mass_fractions.get(s.name, 0.0)) for s in self._species]
        else:
            y = [float(v) for v in mass_fractions]
        if len(y) != len(self._species):
            raise ValueError("Mass fractions length must equal species count")
        moles = [max(0.0, yk) / s.molecular_weight for yk, s in zip(y, self._species)]
        tot = sum(moles)
        if tot <= 0:
            moles = [1.0] + [0.0] * (len(self._species) - 1)
            tot = 1.0
        return tuple(m / tot for m in moles)

    @staticmethod
    def _fill_or_return(buffer, values):
        vals = tuple(values)
        if buffer is not None:
            if len(buffer) != len(vals):
                raise ValueError(f"Array size mismatch: expected {len(vals)}, got {len(buffer)}")
            for i, v in enumerate(vals):
                buffer[i] = v
        return vals

    def setState_TP(self, temperature, pressure):
        from .state import set_state_tp
        return set_state_tp(self, temperature, pressure)

    def setState_TD(self, temperature, density):
        from .state import set_state_td
        return set_state_td(self, temperature, density)

    def setState_DP(self, density, pressure):
        from .state import set_state_dp
        return set_state_dp(self, density, pressure)

    def setState_HP(self, target_enthalpy_mass, pressure, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        from .state import set_state_hp
        return set_state_hp(self, target_enthalpy_mass, pressure, rtol=effective_tol, max_iterations=max_iterations)

    def setState_UV(self, target_int_energy_mass, specific_volume, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        from .state import set_state_uv
        return set_state_uv(self, target_int_energy_mass, specific_volume, rtol=effective_tol, max_iterations=max_iterations)

    def setState_SP(self, target_entropy_mass, pressure, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        from .state import set_state_sp
        return set_state_sp(self, target_entropy_mass, pressure, rtol=effective_tol, max_iterations=max_iterations)

    def setState_SV(self, target_entropy_mass, specific_volume, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        from .state import set_state_sv
        return set_state_sv(self, target_entropy_mass, specific_volume, rtol=effective_tol, max_iterations=max_iterations)

    def setState_TPX(self, temperature, pressure, mole_fractions):
        return self.at(temperature=temperature, pressure=pressure, mole_fractions=mole_fractions)

    def setState_TPY(self, temperature, pressure, mass_fractions):
        x = self._mass_to_mole(mass_fractions)
        return self.at(temperature=temperature, pressure=pressure, mole_fractions=x)

    def getChemPotentials(self, mu=None):
        return self._fill_or_return(mu, self.chemical_potentials)

    def getStandardChemPotentials(self, mu0=None):
        return self._fill_or_return(mu0, self.standard_chemical_potentials)

    def getPartialMolarEnthalpies(self, hbar=None):
        return self._fill_or_return(hbar, self.partial_molar_enthalpies)

    def getPartialMolarEntropies(self, sbar=None):
        return self._fill_or_return(sbar, self.partial_molar_entropies)

    def getPartialMolarIntEnergies(self, ubar=None):
        return self._fill_or_return(ubar, self.partial_molar_int_energies)

    def getPartialMolarCp(self, cpbar=None):
        return self._fill_or_return(cpbar, self.partial_molar_cp)

    def getPartialMolarVolumes(self, vbar=None):
        return self._fill_or_return(vbar, self.partial_molar_volumes)

    def getActivityCoefficients(self, ac=None):
        return self._fill_or_return(ac, self.activity_coefficients)

    def getActivities(self, a=None):
        return self._fill_or_return(a, self.activities)

    def standardConcentration(self, k=0):
        return self.standard_concentration

    def getMassFractions(self, y=None):
        return self._fill_or_return(y, self.Y)

    def getMoleFractions(self, x=None):
        return self._fill_or_return(x, self.X)

    def getConcentrations(self, c=None):
        return self._fill_or_return(c, self.concentrations)

    def getActivityConcentrations(self, c=None):
        return self._fill_or_return(c, self.concentrations)

    @property
    def n_species(self):
        return len(self._species)

    @property
    def nSpecies(self):
        return len(self._species)

    @property
    def molecular_weights(self):
        return tuple(s.molecular_weight for s in self._species)

    @property
    def molecularWeights(self):
        return self.molecular_weights

    def species_index(self, name):
        for idx, s in enumerate(self._species):
            if s.name == name:
                return idx
        raise KeyError(f"Unknown species {name!r}")

    def species_name(self, k):
        return self._species[k].name

    def molecular_weight(self, k):
        if isinstance(k, str):
            k = self.species_index(k)
        return self._species[k].molecular_weight

    def mole_fraction(self, k):
        if isinstance(k, str):
            k = self.species_index(k)
        return self.X[k]

    def mass_fraction(self, k):
        if isinstance(k, str):
            k = self.species_index(k)
        return self.Y[k]


class IdealGasPhase:
    """Stateful, mutable Cantera-compatible IdealGasPhase."""

    def __init__(self, species, *, temperature=298.15, pressure=OneAtm, mole_fractions=None, species_compositions=None, name=""):
        self.name = name
        self._state = IdealGasMixture(species, temperature=temperature, pressure=pressure, mole_fractions=mole_fractions)
        self._species_compositions = dict(species_compositions or {})

    @classmethod
    def from_mechanism(cls, mech_or_path, phase_name=None, *, temperature=298.15, pressure=OneAtm):
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        if not isinstance(mech_or_path, Mechanism):
            mech = load_mechanism(mech_or_path)
        else:
            mech = mech_or_path
        initial_state = None
        if mech.phases:
            p_obj = mech.phase(phase_name)
            if p_obj is not None:
                initial_state = p_obj.state
            mech = mech.for_phase(phase_name)

        gas_species = []
        comp = {}
        for s in mech.species:
            if s.thermo:
                poly = create_species_thermo(s.thermo)
                gas_species.append(Species(s.name, s.molecular_weight, poly, s.charge))
                comp[s.name] = dict(s.composition)
        if not gas_species:
            raise ValueError("No species with thermo data found in mechanism")

        T = temperature
        P = pressure
        mole_fractions = None
        mass_fractions = None
        if initial_state:
            from .units import convert
            if 'T' in initial_state:
                raw_t = initial_state['T']
                if isinstance(raw_t, str):
                    parts = raw_t.strip().split(maxsplit=1)
                    T = convert(float(parts[0]), parts[1], 'K') if len(parts) == 2 else float(parts[0])
                else:
                    T = float(raw_t)
            if 'P' in initial_state:
                raw_p = initial_state['P']
                if isinstance(raw_p, str):
                    parts = raw_p.strip().split(maxsplit=1)
                    P = convert(float(parts[0]), parts[1], 'Pa') if len(parts) == 2 else float(parts[0])
                else:
                    P = float(raw_p)
            if 'X' in initial_state:
                mole_fractions = initial_state['X']
            elif 'Y' in initial_state:
                mass_fractions = initial_state['Y']
        p_name = p_obj.name if (mech.phases and p_obj is not None) else (phase_name or "")
        phase = cls(gas_species, temperature=T, pressure=P,
                    mole_fractions=mole_fractions,
                    species_compositions=comp, name=p_name)
        if mass_fractions is not None:
            phase.setState_TPY(T, P, mass_fractions)
        return phase

    def equilibrate(self, mode='TP', *, rtol=1e-9, max_steps=500, max_temperature_steps=100):
        """Equilibrate this ideal-gas phase at fixed TP, HP, SP, UV, or SV."""
        from .equilibrium import equilibrate_ideal_gas
        if not self._species_compositions:
            raise ValueError("Species elemental compositions required for equilibrium calculation")
        return equilibrate_ideal_gas(self, self._species_compositions, mode=mode,
                                     rtol=rtol, max_steps=max_steps,
                                     max_temperature_steps=max_temperature_steps)

    def _set_species_enthalpy_offset(self, species, offset):
        """Set an additive standard-state species enthalpy offset (J/kmol)."""
        index = self.species_index(species) if isinstance(species, str) else int(species)
        self._state = self._state.with_enthalpy_offset(index, offset)

    @property
    def species(self):
        return self._state.species

    @property
    def species_names(self):
        return self._state.species_names

    @property
    def n_species(self):
        return self._state.n_species

    @property
    def nSpecies(self):
        return self._state.nSpecies

    @property
    def min_temp(self):
        return max(species.thermo.min_temp for species in self._state.species)

    @property
    def max_temp(self):
        return min(species.thermo.max_temp for species in self._state.species)

    def minTemp(self, k=None):
        if k is None:
            return self.min_temp
        return self._state.species[int(k)].thermo.min_temp

    def maxTemp(self, k=None):
        if k is None:
            return self.max_temp
        return self._state.species[int(k)].thermo.max_temp

    @property
    def temperature(self):
        return self._state.temperature

    @temperature.setter
    def temperature(self, val):
        self.setState_TP(val, self.pressure)

    @property
    def T(self):
        return self.temperature

    @T.setter
    def T(self, val):
        self.temperature = val

    @property
    def pressure(self):
        return self._state.pressure

    @pressure.setter
    def pressure(self, val):
        self.setState_TP(self.temperature, val)

    @property
    def P(self):
        return self.pressure

    @P.setter
    def P(self, val):
        self.pressure = val

    @property
    def density(self):
        return self._state.density

    @density.setter
    def density(self, val):
        self.setState_TD(self.temperature, val)

    @property
    def molar_density(self):
        return self._state.molar_density

    @property
    def volume(self):
        return 1.0 / self.density

    @property
    def specific_volume(self):
        return 1.0 / self.density

    @property
    def mean_molecular_weight(self):
        return self._state.mean_molecular_weight

    def meanMolecularWeight(self):
        return self._state.mean_molecular_weight

    @property
    def X(self):
        return self._state.X

    @X.setter
    def X(self, val):
        self._state = self._state.at(mole_fractions=val)

    @property
    def Y(self):
        return self._state.Y

    @Y.setter
    def Y(self, val):
        self.setState_TPY(self.temperature, self.pressure, val)

    @property
    def concentrations(self):
        return self._state.concentrations

    @property
    def cp_mole(self):
        return self._state.cp_mole

    @property
    def cv_mole(self):
        return self._state.cv_mole

    @property
    def enthalpy_mole(self):
        return self._state.enthalpy_mole

    @property
    def int_energy_mole(self):
        return self._state.int_energy_mole

    @property
    def entropy_mole(self):
        return self._state.entropy_mole

    @property
    def gibbs_mole(self):
        return self._state.gibbs_mole

    @property
    def cp_mass(self):
        return self._state.cp_mass

    @property
    def cv_mass(self):
        return self._state.cv_mass

    @property
    def enthalpy_mass(self):
        return self._state.enthalpy_mass

    @property
    def int_energy_mass(self):
        return self._state.int_energy_mass

    @property
    def entropy_mass(self):
        return self._state.entropy_mass

    @property
    def gibbs_mass(self):
        return self._state.gibbs_mass

    @property
    def sound_speed(self):
        return self._state.sound_speed

    @property
    def internal_pressure(self):
        return self._state.internal_pressure

    def internalPressure(self):
        return self._state.internal_pressure

    def soundSpeed(self):
        return self._state.sound_speed

    @property
    def chemical_potentials(self):
        return self._state.chemical_potentials

    @property
    def standard_chemical_potentials(self):
        return self._state.standard_chemical_potentials

    @property
    def partial_molar_enthalpies(self):
        return self._state.partial_molar_enthalpies

    @property
    def partial_molar_entropies(self):
        return self._state.partial_molar_entropies

    @property
    def partial_molar_int_energies(self):
        return self._state.partial_molar_int_energies

    @property
    def partial_molar_cp(self):
        return self._state.partial_molar_cp

    @property
    def partial_molar_volumes(self):
        return self._state.partial_molar_volumes

    @property
    def activities(self):
        return self._state.activities

    @property
    def activity_coefficients(self):
        return self._state.activity_coefficients

    @property
    def activity_concentrations(self):
        return self._state.concentrations

    @property
    def standard_concentration(self):
        return self._state.standard_concentration

    # State tuples
    @property
    def TP(self):
        return self.temperature, self.pressure

    @TP.setter
    def TP(self, val):
        t, p = val
        self.setState_TP(t, p)

    @property
    def TPX(self):
        return self.temperature, self.pressure, self.X

    @TPX.setter
    def TPX(self, val):
        t, p, x = val
        self.setState_TPX(t, p, x)

    @property
    def TPY(self):
        return self.temperature, self.pressure, self.Y

    @TPY.setter
    def TPY(self, val):
        t, p, y = val
        self.setState_TPY(t, p, y)

    @property
    def TD(self):
        return self.temperature, self.density

    @TD.setter
    def TD(self, val):
        t, rho = val
        self.setState_TD(t, rho)

    @property
    def TDX(self):
        return self.temperature, self.density, self.X

    @TDX.setter
    def TDX(self, val):
        t, rho, x = val
        self.X = x
        self.setState_TD(t, rho)

    @property
    def TDY(self):
        return self.temperature, self.density, self.Y

    @TDY.setter
    def TDY(self, val):
        t, rho, y = val
        self.Y = y
        self.setState_TD(t, rho)

    @property
    def DP(self):
        return self.density, self.pressure

    @DP.setter
    def DP(self, val):
        rho, p = val
        self.setState_DP(rho, p)

    @property
    def HP(self):
        return self.enthalpy_mass, self.pressure

    @HP.setter
    def HP(self, val):
        h, p = val
        self.setState_HP(h, p)

    @property
    def UV(self):
        return self.int_energy_mass, self.specific_volume

    @UV.setter
    def UV(self, val):
        u, v = val
        self.setState_UV(u, v)

    @property
    def SP(self):
        return self.entropy_mass, self.pressure

    @SP.setter
    def SP(self, val):
        s, p = val
        self.setState_SP(s, p)

    @property
    def SV(self):
        return self.entropy_mass, self.specific_volume

    @SV.setter
    def SV(self, val):
        s, v = val
        self.setState_SV(s, v)

    # Mutator methods
    def setState_TP(self, t, p):
        self._state = self._state.setState_TP(t, p)

    set_state_tp = setState_TP

    def setState_TPX(self, t, p, x):
        self._state = self._state.setState_TPX(t, p, x)

    set_state_tpx = setState_TPX

    def setState_TPY(self, t, p, y):
        self._state = self._state.setState_TPY(t, p, y)

    set_state_tpy = setState_TPY

    def setState_TD(self, t, rho):
        self._state = self._state.setState_TD(t, rho)

    def setState_DP(self, rho, p):
        # Pinned IdealGasPhase.h:369-376 validates pressure before any state
        # change, then Phase::setDensity mutates density, and only afterwards
        # computes and applies temperature. A failed temperature update thus
        # deliberately leaves the new density installed.
        rho = float(rho)
        p = float(p)
        if p <= 0.0:
            raise ValueError('pressure must be positive')
        if not rho > 0.0:
            raise ValueError(f'density must be positive. density = {rho}')

        old_temperature = self.temperature
        mean_molecular_weight = self.mean_molecular_weight
        self._state = self._state._at_density_unchecked(old_temperature, rho)

        denominator = R * rho
        numerator = p * mean_molecular_weight
        # C++ floating-point division produces +inf for positive/zero. Python
        # raises ZeroDivisionError, so spell out this IEEE-754 case.
        temperature = float('inf') if denominator == 0.0 and numerator > 0.0 \
            else numerator / denominator
        if not temperature > 0.0:
            raise ValueError(f'temperature must be positive. T = {temperature}')
        self._state = self._state._at_density_unchecked(temperature, rho)

    def setState_HP(self, h, p, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._state = self._state.setState_HP(h, p, rtol=effective_tol, max_iterations=max_iterations)

    def setState_UV(self, u, v, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._state = self._state.setState_UV(u, v, rtol=effective_tol, max_iterations=max_iterations)

    def setState_SP(self, s, p, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._state = self._state.setState_SP(s, p, rtol=effective_tol, max_iterations=max_iterations)

    def setState_SV(self, s, v, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._state = self._state.setState_SV(s, v, rtol=effective_tol, max_iterations=max_iterations)

    def setMassFractions_NoNorm(self, y):
        self.Y = y

    # Forward array getters
    def getChemPotentials(self, mu=None):
        return self._state.getChemPotentials(mu)

    def getStandardChemPotentials(self, mu0=None):
        return self._state.getStandardChemPotentials(mu0)

    def getPartialMolarEnthalpies(self, hbar=None):
        return self._state.getPartialMolarEnthalpies(hbar)

    def getPartialMolarEntropies(self, sbar=None):
        return self._state.getPartialMolarEntropies(sbar)

    def getPartialMolarIntEnergies(self, ubar=None):
        return self._state.getPartialMolarIntEnergies(ubar)

    def getPartialMolarCp(self, cpbar=None):
        return self._state.getPartialMolarCp(cpbar)

    def getPartialMolarVolumes(self, vbar=None):
        return self._state.getPartialMolarVolumes(vbar)

    def getActivityCoefficients(self, ac=None):
        return self._state.getActivityCoefficients(ac)

    def getActivities(self, a=None):
        return self._state.getActivities(a)

    def standardConcentration(self, k=0):
        return self._state.standardConcentration(k)

    def getMassFractions(self, y=None):
        return self._state.getMassFractions(y)

    def getMoleFractions(self, x=None):
        return self._state.getMoleFractions(x)

    def getConcentrations(self, c=None):
        return self._state.getConcentrations(c)

    def getActivityConcentrations(self, c=None):
        return self._state.getActivityConcentrations(c)

    def species_index(self, name):
        return self._state.species_index(name)

    def species_name(self, k):
        return self._state.species_name(k)

    def molecular_weight(self, k):
        return self._state.molecular_weight(k)

    @property
    def molecular_weights(self):
        return self._state.molecular_weights

    @property
    def molecularWeights(self):
        return self._state.molecular_weights

    # Elemental composition and equivalence ratio methods
    def n_atoms(self, species, element):
        """Number of atoms of an element in a species."""
        sp_name = self.species_name(species) if isinstance(species, (int, np.integer)) else str(species)
        elem_name = element if isinstance(element, str) else self.element_name(element)
        return float(self._species_compositions.get(sp_name, {}).get(elem_name, 0.0))

    nAtoms = n_atoms

    @property
    def element_names(self) -> list[str]:
        elements = set()
        for comp in self._species_compositions.values():
            elements.update(comp.keys())
        return sorted(elements)

    def element_name(self, m: int) -> str:
        names = self.element_names
        if 0 <= m < len(names):
            return names[m]
        raise CanteraError("Phase::elementName", f"No element with index {m}")

    elementName = element_name

    def element_index(self, element: str, error_if_not_found: bool = True) -> int:
        names = self.element_names
        if element in names:
            return names.index(element)
        if error_if_not_found:
            raise CanteraError("Phase::elementIndex", f"No such element '{element}'")
        return -1

    elementIndex = element_index

    def _composition_to_array(self, comp, basis: str = "mole") -> np.ndarray:
        if isinstance(comp, str):
            comp_str = comp if ":" in comp else f"{comp}:1.0"
            comp_dict = parse_comp_string(comp_str, self.species_names)
        elif isinstance(comp, dict):
            comp_dict = comp
        elif isinstance(comp, (list, tuple, np.ndarray)):
            arr = np.asarray(comp, dtype=float)
            if len(arr) != self.n_species:
                raise CanteraError("Phase::checkArraySize", f"Array size {len(arr)} != {self.n_species}")
            return arr
        else:
            raise TypeError(f"Invalid composition type: {type(comp)}")

        arr = np.zeros(self.n_species, dtype=float)
        for k, v in comp_dict.items():
            arr[self.species_index(k)] = float(v)
        return arr

    def o2_required(self, y=None) -> float:
        """O2 required per kg mixture for complete combustion."""
        y_arr = np.asarray(self.Y if y is None else y, dtype=float)
        total_mass = float(np.sum(y_arr))
        if total_mass <= 0.0:
            raise CanteraError("ThermoPhase::o2Required", "No composition specified")
        mw = self.molecular_weights
        o2_req = 0.0
        for k in range(self.n_species):
            mol_per_kg = y_arr[k] / mw[k]
            n_c = self.n_atoms(k, "C")
            n_s = self.n_atoms(k, "S")
            n_h = self.n_atoms(k, "H")
            o2_req += mol_per_kg * (n_c + n_s + 0.25 * n_h)
        return o2_req / total_mass

    o2Required = o2_required

    def o2_present(self, y=None) -> float:
        """O2 present per kg mixture."""
        y_arr = np.asarray(self.Y if y is None else y, dtype=float)
        total_mass = float(np.sum(y_arr))
        if total_mass <= 0.0:
            raise CanteraError("ThermoPhase::o2Present", "No composition specified")
        mw = self.molecular_weights
        o2_pres = 0.0
        for k in range(self.n_species):
            mol_per_kg = y_arr[k] / mw[k]
            n_o = self.n_atoms(k, "O")
            o2_pres += mol_per_kg * n_o
        return 0.5 * o2_pres / total_mass

    o2Present = o2_present

    @property
    def equivalence_ratio(self) -> float:
        """Equivalence ratio of the current mixture."""
        y = self.Y
        o2_req = self.o2_required(y)
        o2_pres = self.o2_present(y)
        if o2_pres == 0.0:
            return float("inf")
        if o2_req == 0.0:
            return 0.0
        return o2_req / o2_pres

    equivalenceRatio = equivalence_ratio

    def stoich_air_fuel_ratio(self, fuel, oxidizer, basis: str = "mole") -> float:
        """Stoichiometric air to fuel ratio."""
        fuel_comp = self._composition_to_array(fuel, basis)
        ox_comp = self._composition_to_array(oxidizer, basis)
        mw = self.molecular_weights
        if basis == "mole":
            fuel_y = fuel_comp * mw / np.sum(fuel_comp * mw)
            ox_y = ox_comp * mw / np.sum(ox_comp * mw)
        else:
            fuel_y = fuel_comp / np.sum(fuel_comp)
            ox_y = ox_comp / np.sum(ox_comp)

        o2_req_fuel = self.o2_required(fuel_y) - self.o2_present(fuel_y)
        o2_req_ox = self.o2_required(ox_y) - self.o2_present(ox_y)

        if o2_req_fuel < 0.0 or o2_req_ox > 0.0:
            raise CanteraError(
                "ThermoPhase::stoichAirFuelRatio",
                "Fuel composition contains too much oxygen or oxidizer contains "
                "not enough oxygen. Fuel and oxidizer composition mixed up?",
            )
        if o2_req_ox == 0.0:
            return float("inf")
        return o2_req_fuel / (-o2_req_ox)

    stoichAirFuelRatio = stoich_air_fuel_ratio

    def set_equivalence_ratio(
        self,
        phi: float,
        fuel: Any,
        oxidizer: Any,
        basis: str = "mole",
        *,
        diluent: Any = None,
        fraction: Any = None,
    ) -> None:
        """Set the composition to a mixture at equivalence ratio phi, holding T and P constant."""
        phi = float(phi)
        if phi < 0.0:
            raise CanteraError("ThermoPhase::setEquivalenceRatio", "Equivalence ratio phi must be >= 0")
        p = self.pressure
        t = self.temperature

        fuel_comp = self._composition_to_array(fuel, basis)
        ox_comp = self._composition_to_array(oxidizer, basis)
        mw = self.molecular_weights

        if basis == "mole":
            fuel_y = fuel_comp * mw / np.sum(fuel_comp * mw)
            ox_y = ox_comp * mw / np.sum(ox_comp * mw)
        else:
            fuel_y = fuel_comp / np.sum(fuel_comp)
            ox_y = ox_comp / np.sum(ox_comp)

        afr_st = self.stoich_air_fuel_ratio(fuel_y, ox_y, basis="mass")
        sum_f = float(np.sum(fuel_y))
        sum_o = float(np.sum(ox_y))

        y = phi * fuel_y / sum_f + afr_st * ox_y / sum_o
        y /= np.sum(y)

        self.setState_TPY(t, p, y)

        if (fraction is None) != (diluent is None):
            raise ValueError("If dilution is used, both 'fraction' and 'diluent' parameters are required.")

        if fraction is not None:
            if isinstance(fraction, str):
                parts = fraction.split(":")
                fraction_type = parts[0].strip()
                fraction_val = float(parts[1])
            elif isinstance(fraction, dict):
                fraction_type, fraction_val = next(iter(fraction.items()))
                fraction_val = float(fraction_val)
            else:
                raise ValueError("The fraction argument must be given as string or dictionary.")

            if fraction_type not in ("fuel", "oxidizer", "diluent"):
                raise ValueError("The fraction must specify 'fuel', 'oxidizer' or 'diluent'")
            if fraction_val < 0 or fraction_val > 1:
                raise ValueError("The fraction must be between 0 and 1")

            diluent_comp = self._composition_to_array(diluent, basis)
            if fraction_type == "diluent":
                if basis == "mole":
                    x_fuelox = self.X
                    x_dil = diluent_comp / np.sum(diluent_comp)
                    self.X = (1.0 - fraction_val) * x_fuelox + fraction_val * x_dil
                else:
                    y_fuelox = self.Y
                    y_dil = diluent_comp / np.sum(diluent_comp)
                    self.Y = (1.0 - fraction_val) * y_fuelox + fraction_val * y_dil
                self.setState_TP(t, p)

    setEquivalenceRatio = set_equivalence_ratio

    def mixture_fraction(self, fuel, oxidizer, basis: str = "mole", element: str = "Bilger") -> float:
        """Compute the mixture fraction between fuel and oxidizer."""
        fuel_comp = self._composition_to_array(fuel, basis)
        ox_comp = self._composition_to_array(oxidizer, basis)
        mw = self.molecular_weights
        if basis == "mole":
            fuel_y = fuel_comp * mw / np.sum(fuel_comp * mw)
            ox_y = ox_comp * mw / np.sum(ox_comp * mw)
        else:
            fuel_y = fuel_comp / np.sum(fuel_comp)
            ox_y = ox_comp / np.sum(ox_comp)

        if element == "Bilger":
            o2_req_fuel = self.o2_required(fuel_y) - self.o2_present(fuel_y)
            o2_req_ox = self.o2_required(ox_y) - self.o2_present(ox_y)
            y = self.Y
            o2_req_mix = self.o2_required(y) - self.o2_present(y)
            if o2_req_fuel < 0.0 or o2_req_ox > 0.0:
                raise CanteraError(
                    "ThermoPhase::mixtureFraction",
                    "Fuel composition contains too much oxygen or oxidizer contains not enough oxygen.",
                )
            denom = o2_req_fuel - o2_req_ox
            if denom == 0.0:
                raise CanteraError("ThermoPhase::mixtureFraction", "Fuel and oxidizer have the same composition")
            z = (o2_req_mix - o2_req_ox) / denom
            return float(np.clip(z, 0.0, 1.0))
        else:
            sum_yf = float(np.sum(fuel_y))
            sum_yo = float(np.sum(ox_y))
            if sum_yf == 0.0 or sum_yo == 0.0:
                raise CanteraError("ThermoPhase::mixtureFraction", "No fuel and/or oxidizer composition specified")
            m = self.element_index(element)
            z_fuel = sum(fuel_y[k] / mw[k] * self.n_atoms(k, m) for k in range(self.n_species)) / sum_yf
            z_ox = sum(ox_y[k] / mw[k] * self.n_atoms(k, m) for k in range(self.n_species)) / sum_yo
            z_mix = sum(self.Y[k] / mw[k] * self.n_atoms(k, m) for k in range(self.n_species))
            if z_fuel == z_ox:
                raise CanteraError("ThermoPhase::mixtureFraction", f"Fuel and oxidizer have the same composition for element {element}")
            z = (z_mix - z_ox) / (z_fuel - z_ox)
            return float(np.clip(z, 0.0, 1.0))

    mixtureFraction = mixture_fraction

    def set_mixture_fraction(self, mixture_fraction: float, fuel, oxidizer, basis: str = "mole") -> None:
        """Set composition to a specified mixture fraction, holding T and P constant."""
        t, p = self.temperature, self.pressure
        fuel_comp = self._composition_to_array(fuel, basis)
        ox_comp = self._composition_to_array(oxidizer, basis)
        mw = self.molecular_weights
        if basis == "mole":
            fuel_y = fuel_comp * mw / np.sum(fuel_comp * mw)
            ox_y = ox_comp * mw / np.sum(ox_comp * mw)
        else:
            fuel_y = fuel_comp / np.sum(fuel_comp)
            ox_y = ox_comp / np.sum(ox_comp)
        z = float(mixture_fraction)
        y = z * fuel_y + (1.0 - z) * ox_y
        y /= np.sum(y)
        self.setState_TPY(t, p, y)

    setMixtureFraction = set_mixture_fraction

# Species methods
def _species_parameters(self):
    node = {"name": self.name, "weight": float(self.molecular_weight)}
    return node

def _species_molecular_weight(self):
    mw = float(self.molecular_weight)
    return mw

def _species_set_molecular_weight(self, mw):
    val = float(mw)
    self.molecular_weight = val

Species.parameters = _species_parameters
Species.molecularWeight = _species_molecular_weight
Species.setMolecularWeight = _species_set_molecular_weight

# Methods on IdealGasPhase
def _igp_ctor(self, *args, **kwargs):
    if args or kwargs:
        self.__init__(*args, **kwargs)
    return self

def _igp_type(self) -> str:
    phase_type = "IdealGas"
    return phase_type

def _igp_is_ideal(self) -> bool:
    is_id = True
    return is_id

def _igp_is_compressible(self) -> bool:
    is_comp = True
    return is_comp

def _igp_std_conc_units(self) -> str:
    units_str = "kmol/m^3"
    return units_str

def _igp_isothermal_compressibility(self) -> float:
    p = float(self.pressure)
    kappa_t = 1.0 / max(p, 1e-30)
    return kappa_t

def _igp_thermal_expansion_coeff(self) -> float:
    t = float(self.temperature)
    alpha_p = 1.0 / max(t, 1e-30)
    return alpha_p

def _igp_enthalpy_RT_ref(self):
    t = self.temperature
    return [s.thermo.properties(t)[1] for s in self._state.species]

def _igp_cp_R_ref(self):
    t = self.temperature
    return [s.thermo.properties(t)[0] for s in self._state.species]

def _igp_entropy_R_ref(self):
    t = self.temperature
    return [s.thermo.properties(t)[2] for s in self._state.species]

def _igp_gibbs_RT_ref(self):
    h = _igp_enthalpy_RT_ref(self)
    s = _igp_entropy_R_ref(self)
    return [hk - sk for hk, sk in zip(h, s)]

def _igp_getEnthalpy_RT(self, hrt=None):
    res = _igp_enthalpy_RT_ref(self)
    if hrt is not None:
        for i, v in enumerate(res):
            hrt[i] = v
        return hrt
    return res

def _igp_getEntropy_R(self, sr=None):
    import math
    p_ref = getattr(self._state, 'reference_pressure', OneAtm)
    p_ratio = math.log(max(self.pressure / p_ref, 1e-30))
    s_ref = _igp_entropy_R_ref(self)
    res = [s - p_ratio for s in s_ref]
    if sr is not None:
        for i, v in enumerate(res):
            sr[i] = v
        return sr
    return res

def _igp_getGibbs_RT(self, grt=None):
    import math
    p_ref = getattr(self._state, 'reference_pressure', OneAtm)
    p_ratio = math.log(max(self.pressure / p_ref, 1e-30))
    g_ref = _igp_gibbs_RT_ref(self)
    res = [g + p_ratio for g in g_ref]
    if grt is not None:
        for i, v in enumerate(res):
            grt[i] = v
        return grt
    return res

def _igp_getIntEnergy_RT(self, urt=None):
    h_ref = _igp_enthalpy_RT_ref(self)
    res = [h - 1.0 for h in h_ref]
    if urt is not None:
        for i, v in enumerate(res):
            urt[i] = v
        return urt
    return res

def _igp_getCp_R(self, cpr=None):
    res = _igp_cp_R_ref(self)
    if cpr is not None:
        for i, v in enumerate(res):
            cpr[i] = v
        return cpr
    return res

def _igp_getStandardVolumes(self, vol=None):
    dens = self.molar_density
    v_std = 1.0 / max(dens, 1e-30)
    res = [v_std] * self.n_species
    if vol is not None:
        for i, v in enumerate(res):
            vol[i] = v
        return vol
    return res

def _igp_getEnthalpy_RT_ref(self, hrt=None):
    return _igp_getEnthalpy_RT(self, hrt)

def _igp_getGibbs_RT_ref(self, grt=None):
    res = _igp_gibbs_RT_ref(self)
    if grt is not None:
        for i, v in enumerate(res):
            grt[i] = v
        return grt
    return res

def _igp_getGibbs_ref(self, g=None):
    rt_val = R * self.temperature
    g_ref = _igp_gibbs_RT_ref(self)
    res = [grt * rt_val for grt in g_ref]
    if g is not None:
        for i, v in enumerate(res):
            g[i] = v
        return g
    return res

def _igp_getEntropy_R_ref(self, er=None):
    res = _igp_entropy_R_ref(self)
    if er is not None:
        for i, v in enumerate(res):
            er[i] = v
        return er
    return res

def _igp_getCp_R_ref(self, cprt=None):
    return _igp_getCp_R(self, cprt)

def _igp_getStandardVolumes_ref(self, vol=None):
    p_ref = getattr(self._state, 'reference_pressure', OneAtm)
    v_ref = (R * self.temperature) / p_ref
    res = [v_ref] * self.n_species
    if vol is not None:
        for i, v in enumerate(res):
            vol[i] = v
        return vol
    return res

def _igp_initThermo(self):
    self._p0 = getattr(self._state, 'reference_pressure', OneAtm)
    self.updateThermo()

def _igp_getParameters(self):
    params = {"type": "IdealGas", "temperature": float(self.temperature), "pressure": float(self.pressure)}
    return params

def _igp_getSpeciesParameters(self, name, speciesNode=None):
    idx = self.species_index(name) if isinstance(name, str) else int(name)
    s = self._state.species[idx]
    data = {"name": s.name, "molecular-weight": float(s.molecular_weight)}
    if speciesNode is not None and isinstance(speciesNode, dict):
        speciesNode.update(data)
    return data

def _igp_addSpecies(self, spec):
    sp_list = list(self._state.species)
    sp_list.append(spec)
    self._state = IdealGasMixture(sp_list, temperature=self.temperature, pressure=self.pressure)
    return True

def _igp_modifySpecies(self, k, spec):
    idx = self.species_index(k) if isinstance(k, str) else int(k)
    sp_list = list(self._state.species)
    sp_list[idx] = spec
    self._state = IdealGasMixture(sp_list, temperature=self.temperature, pressure=self.pressure, mole_fractions=self.X)
    return True

def _igp_pressureDerivatives(self):
    rho = self.density
    wm = self.mean_molecular_weight
    dp_drho = R * self.temperature / wm
    dp_dt = rho * R / wm
    return dp_drho, dp_dt

def _igp_getLnActivityCoefficients(self, lnact=None):
    res = [0.0] * self.n_species
    if lnact is not None:
        for i, v in enumerate(res):
            lnact[i] = v
        return lnact
    return res

def _igp_calcDensity(self):
    wm = self.mean_molecular_weight
    dens = self.pressure * wm / (R * self.temperature)
    return dens

def _igp_standardConcentration(self, k=0):
    c_std = self.pressure / (R * self.temperature)
    return c_std

def _igp_getIntEnergy_RT_ref(self, urt=None):
    return _igp_getIntEnergy_RT(self, urt)

def _igp_setToEquilState(self, mu_RT):
    import math
    grt = _igp_gibbs_RT_ref(self)
    pp = []
    p_tot = 0.0
    p0 = getattr(self._state, 'reference_pressure', OneAtm)
    for k in range(self.n_species):
        tmp = -grt[k] + mu_RT[k]
        if tmp < -600.0:
            pk = 0.0
        elif tmp > 300.0:
            tmp2 = (tmp / 300.0) ** 2
            pk = p0 * math.exp(300.0) * tmp2
        else:
            pk = p0 * math.exp(tmp)
        pp.append(pk)
        p_tot += pk
    if p_tot > 0.0:
        self.X = [p / p_tot for p in pp]
        self.pressure = p_tot

def _igp_phaseOfMatter(self) -> str:
    matter = "gas"
    return matter

def _igp_getPartialMolarCv_TV(self, cvtilde=None):
    cp_ref = _igp_cp_R_ref(self)
    res = [R * (cp - 1.0) for cp in cp_ref]
    if cvtilde is not None:
        for i, v in enumerate(res):
            cvtilde[i] = v
        return cvtilde
    return res

def _igp_getPartialMolarIntEnergies_TV(self, utilde=None):
    return self.getPartialMolarIntEnergies(utilde)

def _igp_updateThermo(self):
    _ = _igp_enthalpy_RT_ref(self)
    return True

IdealGasPhase.IdealGasPhase = _igp_ctor
IdealGasPhase.type = _igp_type
IdealGasPhase.isIdeal = _igp_is_ideal
IdealGasPhase.isCompressible = _igp_is_compressible
IdealGasPhase.standardConcentrationUnits = _igp_std_conc_units
IdealGasPhase.isothermalCompressibility = _igp_isothermal_compressibility
IdealGasPhase.thermalExpansionCoeff = _igp_thermal_expansion_coeff
IdealGasPhase.getEnthalpy_RT = _igp_getEnthalpy_RT
IdealGasPhase.getEntropy_R = _igp_getEntropy_R
IdealGasPhase.getGibbs_RT = _igp_getGibbs_RT
IdealGasPhase.getIntEnergy_RT = _igp_getIntEnergy_RT
IdealGasPhase.getCp_R = _igp_getCp_R
IdealGasPhase.getStandardVolumes = _igp_getStandardVolumes
IdealGasPhase.getEnthalpy_RT_ref = _igp_getEnthalpy_RT_ref
IdealGasPhase.getGibbs_RT_ref = _igp_getGibbs_RT_ref
IdealGasPhase.getGibbs_ref = _igp_getGibbs_ref
IdealGasPhase.getEntropy_R_ref = _igp_getEntropy_R_ref
IdealGasPhase.getCp_R_ref = _igp_getCp_R_ref
IdealGasPhase.getStandardVolumes_ref = _igp_getStandardVolumes_ref
IdealGasPhase.initThermo = _igp_initThermo
IdealGasPhase.getParameters = _igp_getParameters
IdealGasPhase.getSpeciesParameters = _igp_getSpeciesParameters
IdealGasPhase.addSpecies = _igp_addSpecies
IdealGasPhase.modifySpecies = _igp_modifySpecies
IdealGasPhase.pressureDerivatives = _igp_pressureDerivatives
IdealGasPhase.getLnActivityCoefficients = _igp_getLnActivityCoefficients
IdealGasPhase.calcDensity = _igp_calcDensity
IdealGasPhase.standardConcentration = _igp_standardConcentration
IdealGasPhase.getIntEnergy_RT_ref = _igp_getIntEnergy_RT_ref
IdealGasPhase.setToEquilState = _igp_setToEquilState
IdealGasPhase.phaseOfMatter = _igp_phaseOfMatter
IdealGasPhase.enthalpy_RT_ref = _igp_enthalpy_RT_ref
IdealGasPhase.cp_R_ref = _igp_cp_R_ref
IdealGasPhase.getPartialMolarCv_TV = _igp_getPartialMolarCv_TV
IdealGasPhase.gibbs_RT_ref = _igp_gibbs_RT_ref
IdealGasPhase.updateThermo = _igp_updateThermo
IdealGasPhase.getPartialMolarIntEnergies_TV = _igp_getPartialMolarIntEnergies_TV
IdealGasPhase.entropy_R_ref = _igp_entropy_R_ref
