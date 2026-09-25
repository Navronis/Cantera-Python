"""Unified Solution class combining Phase, Kinetics, and Transport.

Matches Cantera's ct.Solution interface.
"""
from __future__ import annotations
from pathlib import Path
import re
from typing import Any

import numpy as np

from .ideal_gas import IdealGasPhase
try:
    from .cubic_eos import PengRobinson, RedlichKwongMFTP
except ImportError:
    PengRobinson = RedlichKwongMFTP = None

try:
    from .surface import IdealSurfacePhase
except ImportError:
    IdealSurfacePhase = None

try:
    from .fixed_stoichiometry import FixedStoichiometryPhase
except ImportError:
    FixedStoichiometryPhase = None

try:
    from .kinetics import GasKinetics
except ImportError:
    GasKinetics = None

from .constants import CanteraError

try:
    from .transport import (DustyGasTransport, IonGasTransport, MixTransport,
                            MultiTransport, UnityLewisTransport)
except ImportError:
    DustyGasTransport = IonGasTransport = MixTransport = MultiTransport = UnityLewisTransport = None

from .mechanism import load_mechanism, Mechanism, _as_document
from .mechanism import MechanismError
from .units import convert


_THERMO_ALIASES = {
    'ideal-gas': 'ideal-gas', 'idealgas': 'ideal-gas',
    'plasma': 'ideal-gas',
    'ideal-surface': 'ideal-surface', 'surface': 'ideal-surface',
    'surf': 'ideal-surface',
    'peng-robinson': 'Peng-Robinson', 'pengrobinson': 'Peng-Robinson',
    'redlich-kwong': 'Redlich-Kwong', 'redlichkwong': 'Redlich-Kwong',
    'fixed-stoichiometry': 'fixed-stoichiometry', 'stoichsubstance': 'fixed-stoichiometry',
    'stoichiometric-substance': 'fixed-stoichiometry',
}
_KINETICS_ALIASES = {
    'bulk': 'bulk', 'gas': 'bulk', 'gaskinetics': 'bulk',
    'none': 'none', 'kinetics': 'none', '': 'none',
    'surface': 'surface', 'interface': 'surface', 'surf': 'surface',
    'edge': 'edge',
}
_TRANSPORT_ALIASES = {
    'mixture-averaged': 'mixture-averaged', 'mix': 'mixture-averaged',
    'ionized-gas': 'ionized-gas', 'ion': 'ionized-gas',
    'none': 'none', 'transport': 'none', '': 'none',
    'unity-lewis-number': 'unity-lewis-number', 'unitylewis': 'unity-lewis-number',
    'mixture-averaged-ck': 'mixture-averaged-ck', 'ck_mix': 'mixture-averaged-ck',
    'multicomponent': 'multicomponent', 'multi': 'multicomponent',
    'multicomponent-ck': 'multicomponent-ck', 'ck_multi': 'multicomponent-ck',
    'dusty-gas': 'dusty-gas', 'dustygas': 'dusty-gas', 'dusty gas': 'dusty-gas',
    'water': 'water', 'high-pressure': 'high-pressure', 'highp': 'high-pressure',
}


def _model_key(value):
    return '' if value is None else str(value).strip().lower()


def _canonical_model(value, aliases):
    key = _model_key(value)
    return aliases.get(key, key)


def _state_quantity(value, destination, unit_system=None):
    """Convert a YAML state scalar to Cantera's SI base units."""
    if isinstance(value, (int, float)):
        defaults = {'K': 'temperature', 'Pa': 'pressure'}
        source = (unit_system or {}).get(defaults.get(destination), destination)
        return convert(float(value), source, destination)
    if isinstance(value, str):
        match = re.fullmatch(
            r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S+)\s*',
            value)
        if match:
            amount, unit = match.groups()
            return convert(float(amount), unit, destination)
    raise MechanismError(f'Invalid state quantity {value!r}; expected a number or value with units')


def _apply_initial_state(thermo, state, unit_system=None):
    if not state:
        return
    st = dict(state)
    if 'mole-fractions' in st and 'X' not in st:
        st['X'] = st.pop('mole-fractions')
    if 'mass-fractions' in st and 'Y' not in st:
        st['Y'] = st.pop('mass-fractions')
    if 'temperature' in st and 'T' not in st:
        st['T'] = st.pop('temperature')
    if 'pressure' in st and 'P' not in st:
        st['P'] = st.pop('pressure')
    if 'density' in st and 'D' not in st:
        st['D'] = st.pop('density')

    unknown = set(st) - {'T', 'P', 'X', 'Y', 'D', 'vapor-fraction'}
    if unknown:
        raise MechanismError(f'Unsupported phase state fields: {sorted(unknown)}')
    if 'X' in st and 'Y' in st:
        raise MechanismError("Phase state cannot specify both 'X' and 'Y'")
    temperature = _state_quantity(st.get('T', thermo.T), 'K', unit_system)
    pressure = _state_quantity(st.get('P', thermo.P), 'Pa', unit_system)
    if 'X' in st:
        thermo.TPX = temperature, pressure, st['X']
    elif 'Y' in st:
        thermo.TPY = temperature, pressure, st['Y']
    else:
        thermo.TP = temperature, pressure


class Solution:
    """Composite Cantera Solution providing unified Phase, Kinetics, and Transport access."""

    def __new__(cls, infile=None, phase_name=None, adjacent=None, *, thermo=None, kinetics=None, transport=None):
        if infile is not None:
            if isinstance(infile, (str, Path)) or hasattr(infile, 'read_text'):
                mech = load_mechanism(infile)
            elif isinstance(infile, Mechanism):
                mech = infile
            elif isinstance(infile, dict):
                mech = load_mechanism(infile)
            else:
                mech = None
            if mech is not None:
                p_name = phase_name
                if p_name is None and mech.phases:
                    p_name = mech.phases[0].name
                phase = mech.phase(p_name) if mech.phases else None
                thermo_model = _canonical_model(
                    phase.thermo if phase else 'ideal-gas', _THERMO_ALIASES)
                if thermo_model == 'ideal-surface':
                    from .surface import Interface
                    return Interface.from_mechanism(mech, phase_name=p_name, adjacent=adjacent)
        return super().__new__(cls)

    def __init__(self, infile=None, phase_name=None, adjacent=None, *, thermo=None, kinetics=None, transport=None):
        if infile is not None:
            if isinstance(infile, (str, Path)) or hasattr(infile, 'read_text'):
                mech = load_mechanism(infile)
            elif isinstance(infile, Mechanism):
                mech = infile
            elif isinstance(infile, dict):
                mech = load_mechanism(infile)
            else:
                raise TypeError("infile must be a mechanism file path, dictionary, or Mechanism instance")

            if phase_name is None and mech.phases:
                phase_name = mech.phases[0].name
            phase = mech.phase(phase_name) if mech.phases else None
            if mech.phases:
                mech = mech.for_phase(phase_name)
            self._mechanism = mech

            thermo_model = _canonical_model(
                phase.thermo if phase else 'ideal-gas', _THERMO_ALIASES)
            if thermo_model == 'ideal-gas':
                self._thermo = IdealGasPhase.from_mechanism(mech)
                _apply_initial_state(self._thermo, phase.state if phase else None, mech.units)
            elif thermo_model == 'ideal-surface':
                self._thermo = IdealSurfacePhase.from_mechanism(mech)
            elif thermo_model == 'Peng-Robinson':
                self._thermo = PengRobinson.from_mechanism(mech)
                _apply_initial_state(self._thermo, phase.state if phase else None, mech.units)
            elif thermo_model == 'Redlich-Kwong':
                self._thermo = RedlichKwongMFTP.from_mechanism(mech)
                _apply_initial_state(self._thermo, phase.state if phase else None, mech.units)
            elif thermo_model == 'fixed-stoichiometry':
                self._thermo = FixedStoichiometryPhase.from_mechanism(mech, phase_name=phase_name)
                _apply_initial_state(self._thermo, phase.state if phase else None, mech.units)
            else:
                raise NotImplementedError(
                    f"Thermo model {phase.thermo!r} is not yet implemented by the pure-Python port")

            kinetics_model = _canonical_model(
                phase.kinetics if phase else 'bulk', _KINETICS_ALIASES)
            if kinetics_model == 'none' or GasKinetics is None:
                self._kinetics = None
            elif kinetics_model == 'bulk':
                self._kinetics = GasKinetics.from_mechanism(mech, thermo=self._thermo)
            else:
                raise NotImplementedError(
                    f"Kinetics model {phase.kinetics!r} is not yet implemented by the pure-Python port")

            transport_model = transport if isinstance(transport, str) else (
                phase.transport if phase else 'mixture-averaged')
            transport_key = _canonical_model(transport_model, _TRANSPORT_ALIASES)
            if transport_key == 'none' or MixTransport is None:
                self._transport = None
            elif transport_key == 'mixture-averaged' and MixTransport is not None:
                self._transport = MixTransport.from_mechanism(mech, thermo=self._thermo)
            elif transport_key == 'ionized-gas' and IonGasTransport is not None:
                self._transport = IonGasTransport.from_mechanism(mech, thermo=self._thermo)
            elif transport_key == 'multicomponent' and MultiTransport is not None:
                self._transport = MultiTransport.from_mechanism(mech, thermo=self._thermo)
            elif transport_key == 'dusty-gas' and DustyGasTransport is not None:
                self._transport = DustyGasTransport.from_mechanism(
                    mech, thermo=self._thermo)
            else:
                self._transport = None
            self.name = phase.name if phase else ''
            self.thermo_model = phase.thermo if phase else 'ideal-gas'
            self.kinetics_model = phase.kinetics if phase else 'bulk'
            self._transport_model = transport_model
        else:
            self._mechanism = None
            self._thermo = thermo
            self._kinetics = kinetics
            self._transport = transport
            self._transport_model = getattr(transport, 'transport_model', 'none') if transport else 'none'

    def write_yaml(self, filename=None):
        if self._mechanism is None:
            raise ValueError('Solution was not constructed from a serializable mechanism')
        return self._mechanism.to_yaml(filename)

    def equilibrate(self, mode, *, rtol=1e-9, max_steps=500,
                    max_temperature_steps=100):
        """Equilibrate an ideal-gas phase at fixed TP or HP."""
        from .equilibrium import equilibrate_ideal_gas
        if not isinstance(self._thermo, IdealGasPhase):
            raise NotImplementedError('equilibrium is currently implemented for ideal gases')
        if self._mechanism is None:
            raise ValueError('equilibrium requires species elemental compositions')
        compositions = {species.name: species.composition
                        for species in self._mechanism.species
                        if species.name in self.species_names}
        self.equilibrium_diagnostics = equilibrate_ideal_gas(
            self._thermo, compositions, mode, rtol=rtol,
            max_steps=max_steps, max_temperature_steps=max_temperature_steps)
        return None

    @property
    def thermo(self):
        return self._thermo

    @property
    def kinetics(self):
        return self._kinetics

    @property
    def transport(self):
        return self._transport

    # Delegation to Phase
    @property
    def n_species(self):
        return self._thermo.n_species

    @property
    def nSpecies(self):
        return self._thermo.nSpecies

    @property
    def species_names(self):
        return self._thermo.species_names

    @property
    def species(self):
        return self._thermo.species

    def species_name(self, k):
        return self._thermo.species_name(k)

    def species_index(self, name):
        return self._thermo.species_index(name)

    def molecular_weight(self, k):
        return self._thermo.molecular_weight(k)

    @property
    def molecular_weights(self):
        return self._thermo.molecular_weights

    @property
    def molecularWeights(self):
        return self._thermo.molecularWeights

    @property
    def mean_molecular_weight(self):
        return self._thermo.mean_molecular_weight

    def meanMolecularWeight(self):
        return self._thermo.mean_molecular_weight

    @property
    def temperature(self):
        return self._thermo.temperature

    @temperature.setter
    def temperature(self, val):
        self._thermo.temperature = val

    @property
    def T(self):
        return self.temperature

    @T.setter
    def T(self, val):
        self.temperature = val

    @property
    def pressure(self):
        return self._thermo.pressure

    @pressure.setter
    def pressure(self, val):
        self._thermo.pressure = val

    @property
    def P(self):
        return self.pressure

    @P.setter
    def P(self, val):
        self.pressure = val

    @property
    def density(self):
        return self._thermo.density

    @density.setter
    def density(self, val):
        self._thermo.density = val

    @property
    def molar_density(self):
        return self._thermo.molar_density

    @property
    def volume(self):
        return self._thermo.volume

    @property
    def specific_volume(self):
        return self._thermo.specific_volume

    @property
    def X(self):
        return self._thermo.X

    @X.setter
    def X(self, val):
        self._thermo.X = val

    @property
    def Y(self):
        return self._thermo.Y

    @Y.setter
    def Y(self, val):
        self._thermo.Y = val

    @property
    def concentrations(self):
        return self._thermo.concentrations

    @property
    def cp_mole(self):
        return self._thermo.cp_mole

    @property
    def cv_mole(self):
        return self._thermo.cv_mole

    @property
    def cp_mass(self):
        return self._thermo.cp_mass

    @property
    def cv_mass(self):
        return self._thermo.cv_mass

    @property
    def enthalpy_mole(self):
        return self._thermo.enthalpy_mole

    @property
    def enthalpy_mass(self):
        return self._thermo.enthalpy_mass

    @property
    def int_energy_mole(self):
        return self._thermo.int_energy_mole

    @property
    def int_energy_mass(self):
        return self._thermo.int_energy_mass

    @property
    def entropy_mole(self):
        return self._thermo.entropy_mole

    @property
    def entropy_mass(self):
        return self._thermo.entropy_mass

    @property
    def gibbs_mole(self):
        return self._thermo.gibbs_mole

    @property
    def gibbs_mass(self):
        return self._thermo.gibbs_mass

    @property
    def sound_speed(self):
        return self._thermo.sound_speed

    @property
    def internal_pressure(self):
        return self._thermo.internal_pressure

    def internalPressure(self):
        return self._thermo.internal_pressure

    def soundSpeed(self):
        return self._thermo.sound_speed

    @property
    def chemical_potentials(self):
        return self._thermo.chemical_potentials

    @property
    def standard_chemical_potentials(self):
        return self._thermo.standard_chemical_potentials

    @property
    def standard_concentration(self):
        return self._thermo.standard_concentration

    @property
    def partial_molar_enthalpies(self):
        return self._thermo.partial_molar_enthalpies

    @property
    def partial_molar_entropies(self):
        return self._thermo.partial_molar_entropies

    @property
    def partial_molar_int_energies(self):
        return self._thermo.partial_molar_int_energies

    @property
    def partial_molar_cp(self):
        return self._thermo.partial_molar_cp

    @property
    def partial_molar_volumes(self):
        return self._thermo.partial_molar_volumes

    @property
    def activities(self):
        return self._thermo.activities

    @property
    def activity_coefficients(self):
        return self._thermo.activity_coefficients

    @property
    def concentrations(self):
        return getattr(self._thermo, 'concentrations', None)

    @property
    def activity_concentrations(self):
        return getattr(self._thermo, 'activity_concentrations', self.concentrations)

    def standard_concentration(self, k=0):
        sc = getattr(self._thermo, 'standard_concentration', 1.0)
        return sc(k) if callable(sc) else float(sc)

    def setState_TP(self, t, p):
        if hasattr(self._thermo, 'setState_TP'):
            self._thermo.setState_TP(t, p)
        else:
            self._thermo.TP = (t, p)

    def setState_TPX(self, t, p, x):
        if hasattr(self._thermo, 'setState_TPX'):
            self._thermo.setState_TPX(t, p, x)
        else:
            self._thermo.TPX = (t, p, x)

    def setState_TPY(self, t, p, y):
        if hasattr(self._thermo, 'setState_TPY'):
            self._thermo.setState_TPY(t, p, y)
        else:
            self._thermo.TPY = (t, p, y)

    # State tuples
    @property
    def TP(self):
        return self._thermo.TP

    @TP.setter
    def TP(self, val):
        self._thermo.TP = val

    @property
    def TPX(self):
        return self._thermo.TPX

    @TPX.setter
    def TPX(self, val):
        self._thermo.TPX = val

    @property
    def TPY(self):
        return self._thermo.TPY

    @TPY.setter
    def TPY(self, val):
        self._thermo.TPY = val

    @property
    def TD(self):
        return self._thermo.TD

    @TD.setter
    def TD(self, val):
        self._thermo.TD = val

    @property
    def DP(self):
        return self._thermo.DP

    @DP.setter
    def DP(self, val):
        self._thermo.DP = val

    @property
    def HP(self):
        return self._thermo.HP

    @HP.setter
    def HP(self, val):
        self._thermo.HP = val

    @property
    def UV(self):
        return self._thermo.UV

    @UV.setter
    def UV(self, val):
        self._thermo.UV = val

    @property
    def SP(self):
        return self._thermo.SP

    @SP.setter
    def SP(self, val):
        self._thermo.SP = val

    @property
    def SV(self):
        return self._thermo.SV

    @SV.setter
    def SV(self, val):
        self._thermo.SV = val

    # State mutators
    def setState_TP(self, t, p):
        self._thermo.setState_TP(t, p)

    def setState_TPX(self, t, p, x):
        self._thermo.setState_TPX(t, p, x)

    def setState_TPY(self, t, p, y):
        self._thermo.setState_TPY(t, p, y)

    def setState_TD(self, t, rho):
        self._thermo.setState_TD(t, rho)

    def setState_DP(self, rho, p):
        self._thermo.setState_DP(rho, p)

    def setState_HP(self, h, p, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._thermo.setState_HP(h, p, rtol=effective_tol, max_iterations=max_iterations)

    def setState_UV(self, u, v, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._thermo.setState_UV(u, v, rtol=effective_tol, max_iterations=max_iterations)

    def setState_SP(self, s, p, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._thermo.setState_SP(s, p, rtol=effective_tol, max_iterations=max_iterations)

    def setState_SV(self, s, v, tol=None, *, rtol=1e-9, max_iterations=500):
        effective_tol = tol if tol is not None else rtol
        self._thermo.setState_SV(s, v, rtol=effective_tol, max_iterations=max_iterations)

    # Delegation to Kinetics
    @property
    def n_reactions(self):
        return self._kinetics.n_reactions if self._kinetics else 0

    @property
    def nReactions(self):
        return self._kinetics.nReactions if self._kinetics else 0

    @property
    def reactions(self):
        return self._kinetics.reactions if self._kinetics else ()

    def reaction(self, i):
        return self._kinetics.reaction(i)

    def set_multiplier(self, value, reaction=None):
        if self._kinetics is None:
            raise ValueError('solution has no kinetics manager')
        self._kinetics.set_multiplier(value, reaction)

    def multiplier(self, reaction):
        if self._kinetics is None:
            raise ValueError('solution has no kinetics manager')
        return self._kinetics.multiplier(reaction)

    @property
    def forward_rate_constants(self):
        return self._kinetics.forward_rate_constants() if self._kinetics else ()

    @property
    def reverse_rate_constants(self):
        return self._kinetics.reverse_rate_constants() if self._kinetics else ()

    @property
    def equilibrium_constants(self):
        return self._kinetics.equilibrium_constants() if self._kinetics else ()

    @property
    def forward_rates_of_progress(self):
        return self._kinetics.forward_rates_of_progress() if self._kinetics else ()

    @property
    def reverse_rates_of_progress(self):
        return self._kinetics.reverse_rates_of_progress() if self._kinetics else ()

    @property
    def net_rates_of_progress(self):
        return self._kinetics.net_rates_of_progress() if self._kinetics else ()

    @property
    def net_production_rates(self):
        return self._kinetics.net_production_rates() if self._kinetics else ()

    @property
    def creation_rates(self):
        return self._kinetics.creation_rates() if self._kinetics else ()

    @property
    def destruction_rates(self):
        return self._kinetics.destruction_rates() if self._kinetics else ()

    @property
    def delta_enthalpy(self):
        return self._kinetics.delta_enthalpy() if self._kinetics else ()

    @property
    def delta_entropy(self):
        return self._kinetics.delta_entropy() if self._kinetics else ()

    @property
    def delta_gibbs(self):
        return self._kinetics.delta_gibbs() if self._kinetics else ()

    @property
    def heat_release_rate(self):
        return self._kinetics.heat_release_rate if self._kinetics else 0.0

    @property
    def heat_production_rates(self):
        return self._kinetics.heat_production_rates if self._kinetics else ()

    # Delegation to Transport
    @property
    def transport_model(self) -> str:
        if self._transport is None:
            return 'none'
        if hasattr(self._transport, 'transport_model'):
            return self._transport.transport_model
        return getattr(self, '_transport_model', 'none')

    @transport_model.setter
    def transport_model(self, model: str) -> None:
        self._set_transport_model(model)

    def _set_transport_model(self, transport_model: str) -> None:
        transport_key = _canonical_model(transport_model, _TRANSPORT_ALIASES)
        if transport_key == 'none':
            if getattr(self, '_in_flow', False):
                raise CanteraError("TransportFactory::newTransport", f"Invalid Transport model '{transport_model}'.")
            self._transport = None
            self._transport_model = 'none'
            return
        elif transport_key == 'mixture-averaged':
            if self._mechanism:
                self._transport = MixTransport.from_mechanism(self._mechanism, thermo=self._thermo)
            else:
                self._transport = MixTransport(self._thermo)
        elif transport_key == 'unity-lewis-number':
            if self._mechanism:
                self._transport = UnityLewisTransport.from_mechanism(self._mechanism, thermo=self._thermo)
            else:
                self._transport = UnityLewisTransport(self._thermo)
        elif transport_key == 'ionized-gas':
            if self._mechanism:
                self._transport = IonGasTransport.from_mechanism(self._mechanism, thermo=self._thermo)
            else:
                self._transport = IonGasTransport(self._thermo)
        elif transport_key == 'multicomponent':
            if self._mechanism:
                self._transport = MultiTransport.from_mechanism(self._mechanism, thermo=self._thermo)
            else:
                self._transport = MultiTransport(self._thermo)
        elif transport_key == 'dusty-gas':
            if self._mechanism:
                self._transport = DustyGasTransport.from_mechanism(
                    self._mechanism, thermo=self._thermo)
            else:
                self._transport = DustyGasTransport(self._thermo)
        else:
            raise CanteraError("TransportFactory::newTransport", f"Invalid Transport model '{transport_model}'.")
        self._transport_model = transport_model

    @property
    def CK_mode(self) -> bool:
        return self._transport.CK_mode if self._transport else False

    @property
    def viscosity(self) -> float:
        return self._transport.viscosity if self._transport else 0.0

    def species_viscosities(self):
        return self._transport.species_viscosities() if self._transport else ()

    @property
    def thermal_conductivity(self) -> float:
        return self._transport.thermal_conductivity if self._transport else 0.0

    @property
    def mix_diff_coeffs(self):
        return self._transport.mix_diff_coeffs() if self._transport else ()

    @property
    def mix_diff_coeffs_mole(self):
        return self._transport.mix_diff_coeffs_mole if self._transport else ()

    @property
    def mix_diff_coeffs_mass(self):
        return self._transport.mix_diff_coeffs_mass if self._transport else ()

    @property
    def binary_diff_coeffs(self) -> np.ndarray:
        if self._transport and hasattr(self._transport, 'binary_diff_coeffs'):
            return self._transport.binary_diff_coeffs
        if self._transport and hasattr(self._transport, 'binary_diffusion_coefficients'):
            return np.array(self._transport.binary_diffusion_coefficients())
        return np.empty((self.n_species, self.n_species))

    @property
    def multi_diff_coeffs(self) -> np.ndarray:
        if self._transport and hasattr(self._transport, 'multi_diff_coeffs'):
            return self._transport.multi_diff_coeffs
        raise NotImplementedError("Current transport model does not provide multi_diff_coeffs")

    @property
    def thermal_diff_coeffs(self) -> np.ndarray:
        if self._transport and hasattr(self._transport, 'thermal_diff_coeffs'):
            return self._transport.thermal_diff_coeffs
        raise NotImplementedError("Current transport model does not provide thermal_diff_coeffs")

    @property
    def mobilities(self):
        if self._transport and hasattr(self._transport, 'mobilities'):
            return self._transport.mobilities()
        return ()

    @property
    def electrical_conductivity(self) -> float:
        if self._transport and hasattr(self._transport, 'electrical_conductivity'):
            return self._transport.electrical_conductivity
        return 0.0

    def get_binary_diff_coeffs(self, ld, d):
        if self._transport and hasattr(self._transport, 'getBinaryDiffCoeffs'):
            self._transport.getBinaryDiffCoeffs(ld, d)

    def getBinaryDiffCoeffs(self, ld, d):
        return self.get_binary_diff_coeffs(ld, d)

    def get_multi_diff_coeffs(self, ld, d):
        if self._transport and hasattr(self._transport, 'getMultiDiffCoeffs'):
            self._transport.getMultiDiffCoeffs(ld, d)

    def getMultiDiffCoeffs(self, ld, d):
        return self.get_multi_diff_coeffs(ld, d)

    def get_thermal_diff_coeffs(self, dt):
        if self._transport and hasattr(self._transport, 'getThermalDiffCoeffs'):
            self._transport.getThermalDiffCoeffs(dt)

    def getThermalDiffCoeffs(self, dt):
        return self.get_thermal_diff_coeffs(dt)

    def get_mix_diff_coeffs(self, d):
        if self._transport and hasattr(self._transport, 'getMixDiffCoeffs'):
            self._transport.getMixDiffCoeffs(d)

    def getMixDiffCoeffs(self, d):
        return self.get_mix_diff_coeffs(d)

    def get_species_fluxes(self, ndim, grad_T, ldx, grad_X, ldf, fluxes):
        if self._transport and hasattr(self._transport, 'getSpeciesFluxes'):
            self._transport.getSpeciesFluxes(ndim, grad_T, ldx, grad_X, ldf, fluxes)

    def getSpeciesFluxes(self, ndim, grad_T, ldx, grad_X, ldf, fluxes):
        return self.get_species_fluxes(ndim, grad_T, ldx, grad_X, ldf, fluxes)

    def get_mass_fluxes(self, state1, state2, delta, fluxes):
        if self._transport and hasattr(self._transport, 'getMassFluxes'):
            self._transport.getMassFluxes(state1, state2, delta, fluxes)

    def getMassFluxes(self, state1, state2, delta, fluxes):
        return self.get_mass_fluxes(state1, state2, delta, fluxes)

    def get_molar_fluxes(self, state1, state2, delta, fluxes):
        if self._transport and hasattr(self._transport, 'getMolarFluxes'):
            self._transport.getMolarFluxes(state1, state2, delta, fluxes)

    def getMolarFluxes(self, state1, state2, delta, fluxes):
        return self.get_molar_fluxes(state1, state2, delta, fluxes)

    def __getattr__(self, name):
        if '_thermo' in self.__dict__ and self._thermo is not None and hasattr(self._thermo, name):
            return getattr(self._thermo, name)
        if '_kinetics' in self.__dict__ and self._kinetics is not None and hasattr(self._kinetics, name):
            return getattr(self._kinetics, name)
        if '_transport' in self.__dict__ and self._transport is not None and hasattr(self._transport, name):
            return getattr(self._transport, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")


class DustyGas(Solution):
    """Composite gas phase using the dusty-gas porous transport model."""

    def __init__(self, infile=None, phase_name=None, adjacent=None, *,
                 thermo=None, kinetics=None, transport=None):
        if infile is None:
            dusty_transport = (transport if isinstance(transport, DustyGasTransport)
                               else DustyGasTransport(thermo))
            super().__init__(thermo=thermo, kinetics=kinetics,
                             transport=dusty_transport)
        else:
            super().__init__(infile, phase_name, adjacent, thermo=thermo,
                             kinetics=kinetics, transport="DustyGas")

    def _set_dusty_parameter(self, method, value):
        getattr(self._transport, method)(value)

    @property
    def porosity(self):
        raise AttributeError("unreadable attribute 'porosity'")

    @porosity.setter
    def porosity(self, value):
        self._set_dusty_parameter("setPorosity", value)

    @property
    def tortuosity(self):
        raise AttributeError("unreadable attribute 'tortuosity'")

    @tortuosity.setter
    def tortuosity(self, value):
        self._set_dusty_parameter("setTortuosity", value)

    @property
    def mean_pore_radius(self):
        raise AttributeError("unreadable attribute 'mean_pore_radius'")

    @mean_pore_radius.setter
    def mean_pore_radius(self, value):
        self._set_dusty_parameter("setMeanPoreRadius", value)

    @property
    def mean_particle_diameter(self):
        raise AttributeError("unreadable attribute 'mean_particle_diameter'")

    @mean_particle_diameter.setter
    def mean_particle_diameter(self, value):
        self._set_dusty_parameter("setMeanParticleDiameter", value)

    @property
    def permeability(self):
        raise AttributeError("unreadable attribute 'permeability'")

    @permeability.setter
    def permeability(self, value):
        self._set_dusty_parameter("setPermeability", value)

    def molar_fluxes(self, T1, T2, rho1, rho2, Y1, Y2, delta):
        return self._transport.molar_fluxes(
            T1, T2, rho1, rho2, Y1, Y2, delta)

# Real methods on Solution
def _sol_create(*args, **kwargs):
    inst = Solution(*args, **kwargs)
    return inst

def _sol_setName(self, name: str) -> None:
    self._name = str(name)

def _sol_setThermo(self, thermo) -> None:
    self._thermo = thermo

def _sol_setKinetics(self, kinetics) -> None:
    self._kinetics = kinetics

def _sol_setTransport(self, transport) -> None:
    self._transport = transport

def _sol_transportModel(self) -> str:
    m = getattr(self, '_transport_model', 'Mix')
    return str(m)

def _sol_addAdjacent(self, adj) -> None:
    self._adjacent = getattr(self, '_adjacent', [])
    self._adjacent.append(adj)

def _sol_adjacent(self, i: int):
    adj_list = getattr(self, '_adjacent', [])
    return adj_list[i]

def _sol_nAdjacent(self) -> int:
    adj_list = getattr(self, '_adjacent', [])
    return len(adj_list)

def _sol_adjacentName(self, i: int) -> str:
    adj = self.adjacent(i)
    name = getattr(adj, 'name', f"adjacent_{i}")
    return str(name)

def _sol_parameters(self) -> dict:
    params = {"name": getattr(self, 'name', ''), "thermo": getattr(self._thermo, 'type', lambda: '')()}
    return params

def _sol_header(self) -> dict:
    h = {"generator": "Cantera Pure-Python"}
    return h

def _sol_source(self) -> str:
    s = getattr(self, '_source', '')
    return str(s)

def _sol_setSource(self, s: str) -> None:
    self._source = str(s)

def _sol_holdExternalHandle(self, name: str, handle) -> None:
    self._handles = getattr(self, '_handles', {})
    self._handles[name] = handle

def _sol_getExternalHandle(self, name: str):
    handles = getattr(self, '_handles', {})
    return handles.get(name)

def _sol_registerChangedCallback(self, cb) -> None:
    self._callbacks = getattr(self, '_callbacks', [])
    self._callbacks.append(cb)

def _sol_removeChangedCallback(self, cb) -> None:
    callbacks = getattr(self, '_callbacks', [])
    if cb in callbacks:
        callbacks.remove(cb)

Solution.create = _sol_create
Solution.setName = _sol_setName
Solution.setThermo = _sol_setThermo
Solution.setKinetics = _sol_setKinetics
Solution.setTransport = _sol_setTransport
Solution.transportModel = _sol_transportModel
Solution.addAdjacent = _sol_addAdjacent
Solution.adjacent = _sol_adjacent
Solution.nAdjacent = _sol_nAdjacent
Solution.adjacentName = _sol_adjacentName
Solution.parameters = _sol_parameters
Solution.header = _sol_header
Solution.source = _sol_source
Solution.setSource = _sol_setSource
Solution.holdExternalHandle = _sol_holdExternalHandle
Solution.getExternalHandle = _sol_getExternalHandle
Solution.registerChangedCallback = _sol_registerChangedCallback
Solution.removeChangedCallback = _sol_removeChangedCallback

try:
    from .surface import Interface
    Interface.create = _sol_create
    Interface.setThermo = _sol_setThermo
    Interface.setKinetics = _sol_setKinetics
except ImportError:
    pass

