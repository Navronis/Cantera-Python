"""Fixed-stoichiometry incompressible phase equations from Cantera.

This is an immutable scalar-state translation of selected ``StoichSubstance``
and ``SingleSpeciesTP`` property methods. It is intentionally smaller than
Cantera's mutable ``ThermoPhase`` hierarchy and does not parse input files.
"""
from math import isfinite

from .constants import GasConstant as R, OneAtm
from .ideal_gas import Species


class StoichiometricSubstance:
    """One-species constant-volume phase at a specified temperature and pressure.

    Give exactly one ``Species`` and either a mass ``density`` (kg/m3) or a
    ``molar_volume`` (m3/kmol). ``at(pressure=...)`` is the immutable analogue
    of ``StoichSubstance::setPressure``: it changes pressure only and leaves
    density and volume fixed.
    """

    def __init__(self, species, *, density=None, molar_volume=None,
                 temperature=298.15, pressure=OneAtm):
        if not isinstance(species, Species):
            raise TypeError('species must be a cantera_python Species object')
        if (density is None) == (molar_volume is None):
            raise ValueError('Specify exactly one of density or molar_volume')
        self._species = species
        self._temperature = float(temperature)
        self._pressure = float(pressure)
        if not isfinite(self._temperature) or self._temperature <= 0:
            raise ValueError('Temperature must be finite and positive, K')
        if not isfinite(self._pressure) or self._pressure <= 0:
            raise ValueError('Pressure must be finite and positive, Pa')
        if density is not None:
            self._density = float(density)
            if not isfinite(self._density) or self._density <= 0:
                raise ValueError('Density must be finite and positive, kg/m3')
            self._molar_volume = species.molecular_weight / self._density
        else:
            self._molar_volume = float(molar_volume)
            if not isfinite(self._molar_volume) or self._molar_volume <= 0:
                raise ValueError('Molar volume must be finite and positive, m3/kmol')
            self._density = species.molecular_weight / self._molar_volume

    @property
    def species(self):
        return self._species

    @property
    def temperature(self):
        return self._temperature

    @property
    def pressure(self):
        return self._pressure

    @property
    def reference_pressure(self):
        return self.species.thermo.reference_pressure

    @property
    def density(self):
        return self._density

    @property
    def molar_volume(self):
        return self._molar_volume

    @property
    def molar_density(self):
        return 1.0 / self.molar_volume

    def at(self, *, temperature=None, pressure=None):
        """Return the same fixed-volume phase at a new T and/or P."""
        return type(self)(
            self.species, molar_volume=self.molar_volume,
            temperature=self.temperature if temperature is None else temperature,
            pressure=self.pressure if pressure is None else pressure,
        )

    def with_pressure(self, pressure):
        """Immutable ``StoichSubstance::setPressure`` state transformation."""
        return self.at(pressure=pressure)

    def _reference(self):
        return self.species.thermo.properties(self.temperature)

    @property
    def isothermal_compressibility(self):
        return 0.0

    @property
    def thermal_expansion_coeff(self):
        return 0.0

    @property
    def internal_pressure(self):
        return 0.0

    @property
    def activities(self):
        return (1.0,)

    @property
    def activity_coefficients(self):
        return (1.0,)

    @property
    def activity_concentrations(self):
        return (1.0,)

    @property
    def standard_concentration(self):
        return 1.0

    @property
    def log_standard_concentration(self):
        return 0.0

    @property
    def cp_mole(self):
        return R * self._reference()[0]

    @property
    def cv_mole(self):
        # SingleSpeciesTP::cv_mole, with alpha = beta = 0.
        return self.cp_mole

    @property
    def enthalpy_mole(self):
        cp_r, h_rt, _ = self._reference()
        del cp_r
        return R * self.temperature * h_rt + (self.pressure - self.reference_pressure) * self.molar_volume

    @property
    def entropy_mole(self):
        return R * self._reference()[2]

    @property
    def gibbs_mole(self):
        return self.enthalpy_mole - self.temperature * self.entropy_mole

    @property
    def int_energy_mole(self):
        # StoichSubstance::getIntEnergy_RT: pressure independent.
        _, h_rt, _ = self._reference()
        return R * self.temperature * h_rt - self.reference_pressure * self.molar_volume

    @property
    def cp_mass(self):
        return self.cp_mole / self.species.molecular_weight

    @property
    def cv_mass(self):
        return self.cv_mole / self.species.molecular_weight

    @property
    def enthalpy_mass(self):
        return self.enthalpy_mole / self.species.molecular_weight

    @property
    def int_energy_mass(self):
        return self.int_energy_mole / self.species.molecular_weight

    @property
    def entropy_mass(self):
        return self.entropy_mole / self.species.molecular_weight

    @property
    def gibbs_mass(self):
        return self.gibbs_mole / self.species.molecular_weight

    @property
    def partial_molar_enthalpies(self):
        return (self.enthalpy_mole,)

    @property
    def partial_molar_int_energies(self):
        return (self.int_energy_mole,)

    @property
    def partial_molar_entropies(self):
        return (self.entropy_mole,)

    @property
    def partial_molar_cp(self):
        return (self.cp_mole,)

    @property
    def partial_molar_volumes(self):
        return (self.molar_volume,)

    @property
    def standard_chemical_potentials(self):
        return (self.gibbs_mole,)

    @property
    def chemical_potentials(self):
        return self.standard_chemical_potentials

from .phases import SingleSpeciesTP

class StoichSubstance(SingleSpeciesTP):
    """Stoichiometric substance thermodynamic phase."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._press = OneAtm
        self._p0 = OneAtm

    def StoichSubstance(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        return self

    def type(self) -> str:
        phase_type = "StoichSubstance"
        return phase_type

    def isCompressible(self) -> bool:
        compressible_flag = False
        return compressible_flag

    def isothermalCompressibility(self) -> float:
        kappa_t = 0.0
        return kappa_t

    def thermalExpansionCoeff(self) -> float:
        alpha_p = 0.0
        return alpha_p

    def internalPressure(self) -> float:
        pi_t = 0.0
        return pi_t

    def standardConcentrationUnits(self) -> str:
        units_str = "kmol/m^3"
        return units_str

    def getActivityConcentrations(self, c=None):
        if c is not None:
            c[0] = 1.0
            return c
        return [1.0]

    def standardConcentration(self, k: int = 0) -> float:
        c_std = 1.0
        return c_std

    def logStandardConc(self, k: int = 0) -> float:
        log_c = 0.0
        return log_c

    def getStandardChemPotentials(self, mu0=None):
        grt = self.getGibbs_RT()
        rt_val = self.RT()
        chem_pot = [g * rt_val for g in grt]
        if mu0 is not None:
            for i, val in enumerate(chem_pot):
                mu0[i] = val
            return mu0
        return chem_pot

    def getEnthalpy_RT(self, hrt=None):
        hrt_ref = self.getEnthalpy_RT_ref()
        mol_dens = self.molarDensity()
        if mol_dens <= 0.0:
            mol_dens = 1.0
        pres_correct = (self.pressure() - self.refPressure()) / mol_dens
        rt_val = self.RT()
        res = [hrt_ref[0] + pres_correct / rt_val]
        if hrt is not None:
            hrt[0] = res[0]
            return hrt
        return res

    def getEntropy_R(self, sr=None):
        sr_ref = self.getEntropy_R_ref()
        res = [sr_ref[0]]
        if sr is not None:
            sr[0] = res[0]
            return sr
        return res

    def getGibbs_RT(self, grt=None):
        hrt = self.getEnthalpy_RT()
        sr = self.getEntropy_R()
        res = [hrt[0] - sr[0]]
        if grt is not None:
            grt[0] = res[0]
            return grt
        return res

    def getCp_R(self, cpr=None):
        self._updateThermo()
        val = getattr(self, '_cp0_R', 0.0)
        res = [val]
        if cpr is not None:
            cpr[0] = val
            return cpr
        return res

    def getIntEnergy_RT(self, urt=None):
        self._updateThermo()
        mol_dens = self.molarDensity()
        if mol_dens <= 0.0:
            mol_dens = 1.0
        h0 = getattr(self, '_h0_RT', 0.0)
        rt_val = self.RT()
        p0_val = self.refPressure()
        val = h0 - p0_val / mol_dens / rt_val
        res = [val]
        if urt is not None:
            urt[0] = val
            return urt
        return res

    def getIntEnergy_RT_ref(self, urt=None):
        return self.getIntEnergy_RT(urt)

    def initThermo(self):
        super().initThermo()
        self._p0 = self.refPressure()

    def getSpeciesParameters(self, name: str, speciesNode=None):
        super().getSpeciesParameters(name, speciesNode)
        res = {"density": self.density(), "molar-volume": self.meanMolecularWeight() / max(self.density(), 1e-30)}
        if speciesNode is not None and isinstance(speciesNode, dict):
            speciesNode.update(res)
        return res
