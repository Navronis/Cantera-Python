"""Incompressible ideal-solid-solution equations from Cantera.

This scalar, immutable-state layer translates the property equations in
``IdealSolidSolnPhase.cpp``. It deliberately omits the parent ``ThermoPhase``
state engine, YAML construction, equilibrium solvers, and mutable APIs.
Temperatures are K, pressures Pa, molecular weights kg/kmol, molar volumes
m3/kmol, and molar thermodynamic properties J/kmol (or J/kmol/K).
"""
from math import isfinite, log
from typing import Mapping

from .constants import GasConstant as R, OneAtm, SmallNumber
from .ideal_gas import Species


class IdealSolidSolution:
    """Evaluate selected ``IdealSolidSolnPhase`` equations at one state.

    ``molar_volumes`` is supplied in species order. ``standard_concentration``
    accepts upstream's ``unity``, ``species-molar-volume`` / ``molar_volume``,
    and ``solvent-molar-volume`` / ``solvent_volume`` forms. The final species
    is the solvent for the latter form, matching the source implementation.
    """

    _CONCENTRATION_MODELS = {
        'unity': 0,
        'species-molar-volume': 1,
        'molar_volume': 1,
        'solvent-molar-volume': 2,
        'solvent_volume': 2,
    }

    def __init__(self, species, molar_volumes, *, temperature=298.15,
                 pressure=OneAtm, mole_fractions=None,
                 standard_concentration='unity'):
        self._species = tuple(species)
        if not self._species or len({s.name for s in self._species}) != len(self._species):
            raise ValueError('Provide one or more species with distinct names')
        if not all(isinstance(s, Species) for s in self._species):
            raise TypeError('species must contain cantera_python Species objects')
        self._volumes = tuple(float(v) for v in molar_volumes)
        if len(self._volumes) != len(self._species):
            raise ValueError('Molar-volume count must equal species count')
        if any(not isfinite(v) or v <= 0 for v in self._volumes):
            raise ValueError('Molar volumes must be finite and positive, m3/kmol')
        self._temperature = float(temperature)
        self._pressure = float(pressure)
        if not isfinite(self._temperature) or self._temperature <= 0:
            raise ValueError('Temperature must be finite and positive, K')
        if not isfinite(self._pressure) or self._pressure <= 0:
            raise ValueError('Pressure must be finite and positive, Pa')
        self._reference_pressure = float(self._species[0].thermo.reference_pressure)
        if any(s.thermo.reference_pressure != self._reference_pressure for s in self._species):
            raise ValueError('All species must have the same reference pressure')
        try:
            self._concentration_model = self._CONCENTRATION_MODELS[
                str(standard_concentration).lower()]
        except KeyError as exc:
            raise ValueError('Unknown standard-concentration model') from exc
        if mole_fractions is None:
            amounts = (1.0,) + (0.0,) * (len(self._species) - 1)
        elif isinstance(mole_fractions, Mapping):
            unknown = set(mole_fractions) - set(self.species_names)
            if unknown:
                raise ValueError(f'Unknown species: {sorted(unknown)}')
            amounts = tuple(float(mole_fractions.get(s.name, 0.0)) for s in self._species)
        else:
            amounts = tuple(float(x) for x in mole_fractions)
        if len(amounts) != len(self._species):
            raise ValueError('Composition length must equal species count')
        if any(not isfinite(x) or x < 0 for x in amounts):
            raise ValueError('Mole amounts must be finite and nonnegative')
        total = sum(amounts)
        if not isfinite(total) or total <= 0:
            raise ValueError('Mole amounts must have a finite positive sum')
        self._x = tuple(x / total for x in amounts)

    @property
    def species(self):
        return self._species

    @property
    def n_species(self):
        return len(self._species)

    @property
    def nSpecies(self):
        return len(self._species)

    @property
    def species_names(self):
        return tuple(s.name for s in self.species)

    @property
    def molar_volumes(self):
        return self._volumes

    @property
    def temperature(self):
        return self._temperature

    @property
    def pressure(self):
        return self._pressure

    @property
    def reference_pressure(self):
        return self._reference_pressure

    @property
    def X(self):
        return self._x

    def at(self, *, temperature=None, pressure=None, mole_fractions=None,
           standard_concentration=None):
        return type(self)(
            self.species, self.molar_volumes,
            temperature=self.temperature if temperature is None else temperature,
            pressure=self.pressure if pressure is None else pressure,
            mole_fractions=self.X if mole_fractions is None else mole_fractions,
            standard_concentration=(self.standard_concentration_model
                                    if standard_concentration is None
                                    else standard_concentration),
        )

    @property
    def standard_concentration_model(self):
        return ('unity', 'species-molar-volume', 'solvent-molar-volume')[
            self._concentration_model]

    def _reference(self):
        return tuple(s.thermo.properties(self.temperature) for s in self.species)

    def _mean(self, values):
        return sum(x * value for x, value in zip(self.X, values))

    @property
    def mean_molecular_weight(self):
        return self._mean(s.molecular_weight for s in self.species)

    @property
    def molar_volume(self):
        return self._mean(self.molar_volumes)

    @property
    def density(self):
        # IdealSolidSolnPhase::calcDensity
        return self.mean_molecular_weight / self.molar_volume

    @property
    def molar_density(self):
        return 1.0 / self.molar_volume

    @property
    def isothermal_compressibility(self):
        return 0.0

    @property
    def thermal_expansion_coeff(self):
        return 0.0

    def _sum_xlogx(self):
        # Phase::sum_xlogx, written in its direct mole-fraction equivalent.
        return sum(x * log(max(x, SmallNumber)) for x in self.X)

    @property
    def cp_mole(self):
        return R * self._mean(p[0] for p in self._reference())

    @property
    def cv_mole(self):
        return self.cp_mole

    @property
    def entropy_mole(self):
        return R * (self._mean(p[2] for p in self._reference()) - self._sum_xlogx())

    @property
    def enthalpy_mole(self):
        return self._mean(self.partial_molar_enthalpies)

    @property
    def int_energy_mole(self):
        return self.enthalpy_mole - self.pressure * self.molar_volume

    @property
    def gibbs_mole(self):
        # IdealSolidSolnPhase::gibbs_mole; Pv = (P - Pref) / molarDensity.
        return (R * self.temperature *
                (self._mean(p[1] - p[2] for p in self._reference()) + self._sum_xlogx())
                + (self.pressure - self.reference_pressure) * self.molar_volume)

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
    def activities(self):
        return self.X

    @property
    def activity_coefficients(self):
        return (1.0,) * len(self.species)

    @property
    def activity_concentrations(self):
        if self._concentration_model == 0:
            return self.X
        denominator = (self.molar_volumes if self._concentration_model == 1
                       else (self.molar_volumes[-1],) * len(self.species))
        return tuple(x / volume for x, volume in zip(self.X, denominator))

    @property
    def standard_concentrations(self):
        if self._concentration_model == 0:
            return (1.0,) * len(self.species)
        denominator = (self.molar_volumes if self._concentration_model == 1
                       else (self.molar_volumes[-1],) * len(self.species))
        return tuple(1.0 / volume for volume in denominator)

    @property
    def partial_molar_enthalpies(self):
        delta_p = self.pressure - self.reference_pressure
        return tuple(R * self.temperature * p[1] + delta_p * volume
                     for p, volume in zip(self._reference(), self.molar_volumes))

    @property
    def partial_molar_entropies(self):
        return tuple(R * (p[2] - log(max(x, SmallNumber)))
                     for p, x in zip(self._reference(), self.X))

    @property
    def partial_molar_cp(self):
        return tuple(R * p[0] for p in self._reference())

    @property
    def partial_molar_volumes(self):
        return self.molar_volumes

    @property
    def standard_chemical_potentials(self):
        delta_p = self.pressure - self.reference_pressure
        return tuple(R * self.temperature * (p[1] - p[2]) + delta_p * volume
                     for p, volume in zip(self._reference(), self.molar_volumes))

    @property
    def chemical_potentials(self):
        return tuple(mu + R * self.temperature * log(max(x, SmallNumber))
                     for mu, x in zip(self.standard_chemical_potentials, self.X))
