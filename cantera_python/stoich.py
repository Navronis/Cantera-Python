from __future__ import annotations
"""Fixed-stoichiometry incompressible phase equations from Cantera.

This is an immutable scalar-state translation of selected ``StoichSubstance``
and ``SingleSpeciesTP`` property methods. It is intentionally smaller than
Cantera's mutable ``ThermoPhase`` hierarchy and does not parse input files.
"""
from math import isfinite
import numpy as np

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
    """Fixed-composition incompressible stoichiometric substance."""

    def __init__(self, infile: str = "", id_: str = "", name: str = ""):
        super().__init__(name=name or id_)
        super().__init__()
        self._press = OneAtm
        self._p0 = OneAtm
        if infile:
            self.initThermoFile(infile, id_)

    def StoichSubstance(self, *args, **kwargs) -> StoichSubstance:
        self._state_mf_number += 1
        return self

    def type(self) -> str:
        return "stoichiometric-substance"

    def isCompressible(self) -> bool:
        return False

    def pressure(self) -> float:
        return float(self._press)

    def setPressure(self, p: float) -> None:
        self._press = float(p)
        self.invalidateCache()

    def isothermalCompressibility(self) -> float:
        return 0.0

    def thermalExpansionCoeff(self) -> float:
        return 0.0

    def internalPressure(self) -> float:
        return 0.0

    def standardConcentrationUnits(self) -> str:
        return "1.0"

    def getActivityConcentrations(self, c: Optional[np.ndarray] = None) -> np.ndarray:
        if c is not None:
            c[0] = 1.0
            return c
        return np.ones(1, dtype=float)

    def standardConcentration(self, k: int = 0) -> float:
        return 1.0

    def logStandardConc(self, k: int = 0) -> float:
        return 0.0

    def getStandardChemPotentials(self, mu0: Optional[np.ndarray] = None) -> np.ndarray:
        grt = self.getGibbs_RT()
        rt_val = self.RT()
        if mu0 is not None:
            mu0[0] = grt[0] * rt_val
            return mu0
        return np.array([grt[0] * rt_val], dtype=float)

    def getEnthalpy_RT(self, hrt: Optional[np.ndarray] = None) -> np.ndarray:
        hrt_ref = self.getEnthalpy_RT_ref()
        mol_dens = self.molarDensity()
        if mol_dens <= 0.0:
            dens = max(self.density(), 1e-30)
            mw = self.molecularWeight(0) if self.nSpecies() > 0 else 1.0
            mol_dens = dens / mw
        pres_correct = (self.pressure() - self.refPressure()) / max(mol_dens, 1e-30)
        rt_val = self.RT()
        val = hrt_ref[0] + pres_correct / rt_val
        if hrt is not None:
            hrt[0] = val
            return hrt
        return np.array([val], dtype=float)

    def getEntropy_R(self, sr: Optional[np.ndarray] = None) -> np.ndarray:
        sr_ref = self.getEntropy_R_ref()
        if sr is not None:
            sr[0] = sr_ref[0]
            return sr
        return np.array([sr_ref[0]], dtype=float)

    def getGibbs_RT(self, grt: Optional[np.ndarray] = None) -> np.ndarray:
        hrt = self.getEnthalpy_RT()
        self._updateThermo()
        val = hrt[0] - self._s0_R
        if grt is not None:
            grt[0] = val
            return grt
        return np.array([val], dtype=float)

    def getCp_R(self, cpr: Optional[np.ndarray] = None) -> np.ndarray:
        self._updateThermo()
        val = getattr(self, "_cp0_R", 0.0)
        if cpr is not None:
            cpr[0] = val
            return cpr
        return np.array([val], dtype=float)

    def getIntEnergy_RT(self, urt: Optional[np.ndarray] = None) -> np.ndarray:
        self._updateThermo()
        mol_dens = self.molarDensity()
        if mol_dens <= 0.0:
            dens = max(self.density(), 1e-30)
            mw = self.molecularWeight(0) if self.nSpecies() > 0 else 1.0
            mol_dens = dens / mw
        h0 = getattr(self, "_h0_RT", 0.0)
        rt_val = self.RT()
        p0_val = self.refPressure()
        val = h0 - (p0_val / max(mol_dens, 1e-30)) / rt_val
        if urt is not None:
            urt[0] = val
            return urt
        return np.array([val], dtype=float)

    def getIntEnergy_RT_ref(self, urt: Optional[np.ndarray] = None) -> np.ndarray:
        return self.getIntEnergy_RT(urt)

    def initThermo(self) -> None:
        from .constants import CanteraError
        if self.nSpecies() != 1:
            raise CanteraError("StoichSubstance::initThermo",
                "stoichiometric substances may only contain one species.")
        self._p0 = self.refPressure()
        super().initThermo()

    def getSpeciesParameters(self, name: str, speciesNode: Optional[dict] = None) -> dict:
        super().getSpeciesParameters(name, speciesNode)
        dens = max(self.density(), 1e-30)
        mw = self.molecularWeight(0) if self.nSpecies() > 0 else 1.0
        res = {"density": self.density(), "molar-volume": mw / dens}
        if speciesNode is not None and isinstance(speciesNode, dict):
            speciesNode.update(res)
        return res

    @classmethod
    def from_mechanism(cls, mech_or_path: Any, phase_name: Optional[str] = None) -> StoichSubstance:
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        mech = mech_or_path if isinstance(mech_or_path, Mechanism) else load_mechanism(mech_or_path)
        if phase_name is None and mech.phases: phase_name = mech.phases[0].name
        p_obj = mech.phase(phase_name) if mech.phases else None
        p_name = p_obj.name if p_obj else (phase_name or "stoich")
        sub_inst = cls()
        sub_inst.name = p_name
        if hasattr(mech, 'elements') and mech.elements:
            for el in mech.elements:
                el_name = el.name if hasattr(el, 'name') else str(el)
                sub_inst.addElement(el_name)
        elif hasattr(p_obj, 'elements') and p_obj.elements:
            for el in p_obj.elements:
                el_name = el.name if hasattr(el, 'name') else str(el)
                sub_inst.addElement(el_name)
        for s in mech.species:
            p_spec_names = [sp.name if hasattr(sp, 'name') else str(sp) for sp in p_obj.species] if (p_obj and hasattr(p_obj, 'species')) else None
            if p_spec_names is None or s.name in p_spec_names:
                if getattr(s, 'composition', None):
                    for el_name in s.composition.keys():
                        if el_name not in sub_inst.elementNames():
                            sub_inst.addElement(el_name)
                poly = create_species_thermo(s.thermo) if s.thermo else None
                from .phases import PhaseSpecies
                sp_obj = PhaseSpecies(s.name, s.molecular_weight, poly, s.charge, composition=getattr(s, 'composition', {}), equation_of_state=getattr(s, 'equation_of_state', {}), input_data=getattr(s, 'input_data', {}))
                sub_inst.addSpecies(sp_obj)
                # Assign density from species EOS
                eos = getattr(s, "equation_of_state", {})
                if isinstance(eos, dict) and eos.get("model") == "constant-volume":
                    from .units import convert
                    if "density" in eos:
                        raw_rho = eos["density"]
                        rho = convert(float(raw_rho.split()[0]), raw_rho.split()[1], "kg/m^3") if isinstance(raw_rho, str) and len(raw_rho.split()) == 2 else float(raw_rho)
                        sub_inst.assignDensity(rho)
                    elif "molar-volume" in eos:
                        raw_mv = eos["molar-volume"]
                        mv = convert(float(raw_mv.split()[0]), raw_mv.split()[1], "m^3/kmol") if isinstance(raw_mv, str) and len(raw_mv.split()) == 2 else float(raw_mv)
                        sub_inst.assignDensity(s.molecular_weight / max(mv, 1e-30))
                    elif "molar-density" in eos:
                        raw_md = eos["molar-density"]
                        md = convert(float(raw_md.split()[0]), raw_md.split()[1], "kmol/m^3") if isinstance(raw_md, str) and len(raw_md.split()) == 2 else float(raw_md)
                        sub_inst.assignDensity(s.molecular_weight * md)
                break
        sub_inst.initThermo()
        if p_obj and p_obj.state:
            st = p_obj.state
            if "T" in st: sub_inst.setTemperature(float(str(st["T"]).split()[0]))
            if "P" in st: sub_inst.setPressure(float(str(st["P"]).split()[0]))
        return sub_inst
