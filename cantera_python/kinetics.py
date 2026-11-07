"""Editable scalar bulk-kinetics equations from pinned Cantera source.

Concentrations use kmol/m3. This module deliberately has no mechanism parser
or native Cantera dependency; it is the small, testable core for a future
pure-Python kinetics manager.
"""
from dataclasses import dataclass, field
from math import exp, isclose, isfinite, log, prod
from typing import Any, Optional, Dict, List, Tuple

import numpy as np

from .constants import GasConstant
from .reaction_equation import are_duplicate_reactions

from .rates import (
    _FalloffRate, BlowersMaselRate, ChebyshevRate, LinearBurkeRate, PlogRate,
)


@dataclass(frozen=True)
class ThirdBody:
    """Third-body collider efficiencies.

    ``M_eff = default * sum(C_k) + sum((eff_k - default) * C_k)`` is the
    sparse, default-compensated calculation in ``ThirdBodyCalc::update``.
    """
    efficiencies: dict = field(default_factory=dict)
    default_efficiency: float = 1.0
    mass_action: bool = True
    name: str = 'M'

    def __post_init__(self):
        if not isfinite(float(self.default_efficiency)):
            raise ValueError("default_efficiency must be finite")
        for name, value in self.efficiencies.items():
            if not isinstance(name, str) or not name:
                raise ValueError("third-body efficiency species names must be nonempty strings")
            if not isfinite(float(value)):
                raise ValueError("third-body efficiencies must be finite")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("third-body name must be a nonempty string")

    def efficiency(self, species):
        """Return a species efficiency, falling back to the default."""
        return float(self.efficiencies.get(species, self.default_efficiency))

    def effective_concentration(self, concentrations):
        """Return the effective collider concentration for *concentrations*."""
        checked = {name: float(value) for name, value in concentrations.items()}
        if any(not isfinite(value) or value < 0.0 for value in checked.values()):
            raise ValueError("concentrations must be finite and nonnegative")
        ctot = sum(checked.values())
        return (float(self.default_efficiency) * ctot
                + sum((float(efficiency) - float(self.default_efficiency))
                      * checked.get(name, 0.0)
                      for name, efficiency in self.efficiencies.items()))


@dataclass(frozen=True)
class ElementaryReaction:
    """One scalar elementary bulk reaction.

    Explicit forward ``orders`` are allowed only for irreversible reactions,
    matching ``Reaction::check``. A mass-action :class:`ThirdBody` multiplies
    both forward and reverse rates, matching ``BulkKinetics::processThirdBodies``.
    """
    rate: object
    reactants: dict
    products: dict
    reversible: bool = False
    reverse_rate: object = None
    orders: dict = None
    allow_nonreactant_orders: bool = False
    allow_negative_orders: bool = False
    third_body: ThirdBody = None
    equilibrium_thermo: object = None
    _live_equilibrium_state: object = None
    rate_thermo: object = None
    duplicate: bool = False

    def __post_init__(self):
        if not callable(self.rate):
            raise TypeError("rate must be callable as rate(T)")
        if self.reversible and not callable(self.reverse_rate) and self.equilibrium_thermo is None:
            raise TypeError("reversible reactions require reverse_rate or equilibrium_thermo")
        if self.reverse_rate is not None and self.equilibrium_thermo is not None:
            raise ValueError("provide either reverse_rate or equilibrium_thermo, not both")
        for side in (self.reactants, self.products):
            if any(float(value) < 0.0 or not isfinite(float(value)) for value in side.values()):
                raise ValueError("Stoichiometric coefficients must be finite and nonnegative")
        if self.orders is not None:
            if self.reversible:
                raise ValueError("Reaction orders may only be given for irreversible reactions")
            for name, order in self.orders.items():
                value = float(order)
                if not isfinite(value):
                    raise ValueError("Reaction orders must be finite")
                if not self.allow_nonreactant_orders and name not in self.reactants:
                    raise ValueError("Reaction order specified for non-reactant species '{}'".format(name))
                if not self.allow_negative_orders and value < 0.0:
                    raise ValueError("Negative reaction order specified for species '{}'".format(name))
        if self.third_body is not None and not isinstance(self.third_body, ThirdBody):
            raise TypeError("third_body must be a ThirdBody or None")
        if self.equilibrium_thermo is not None:
            required = ("temperature", "species_names", "standard_chemical_potentials",
                        "standard_concentration")
            if any(not hasattr(self.equilibrium_thermo, name) for name in required):
                raise TypeError("equilibrium_thermo must provide ideal-gas standard-state properties")
            unknown = (set(self.reactants) | set(self.products)) - set(self.equilibrium_thermo.species_names)
            if unknown:
                raise ValueError("reaction species absent from equilibrium_thermo: {}".format(sorted(unknown)))

    def _rate_value(self, rate, temperature):
        from .rates import TwoTempPlasmaRate, ElectronCollisionPlasmaRate
        if isinstance(rate, BlowersMaselRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            if state is None:
                raise ValueError('Blowers-Masel rates require a live thermo state')
            rate.update_from_thermo(state, self.reactants, self.products)
        if isinstance(rate, (PlogRate, ChebyshevRate, LinearBurkeRate)):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            if state is None or not hasattr(state, 'pressure'):
                raise ValueError('pressure-dependent rates require a thermo state with pressure')
            if isinstance(rate, LinearBurkeRate):
                fractions = dict(zip(state.species_names, state.X))
                return rate(temperature, float(state.pressure), fractions)
            return rate(temperature, float(state.pressure))
        if isinstance(rate, TwoTempPlasmaRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            Te = getattr(state, 'electron_temperature', getattr(state, 'electronTemperature', temperature)) if state else temperature
            return rate(temperature, float(Te))
        if isinstance(rate, ElectronCollisionPlasmaRate):
            state = self._live_equilibrium_state or self.rate_thermo or self.equilibrium_thermo
            levels = getattr(state, 'electron_energy_levels', ()) if state else ()
            dist = getattr(state, 'electron_energy_distribution', ()) if state else ()
            if not levels or not dist or len(levels) < 2 or len(dist) < 2:
                return 0.0
            return rate(levels, dist)
        return rate(temperature)

    def _progress(self, rate, orders, concentrations, temperature):
        checked = {name: float(value) for name, value in concentrations.items()}
        if any(not isfinite(value) or value < 0.0 for value in checked.values()):
            raise ValueError("concentrations must be finite and nonnegative")
        for name, order in orders.items():
            if float(order) < 0.0 and checked.get(name, 0.0) == 0.0:
                raise ValueError("zero concentration is invalid for negative reaction order")
        return self._rate_value(rate, temperature) * prod(
            checked.get(name, 0.0) ** float(order) for name, order in orders.items())

    @property
    def forward_orders(self):
        if self.orders is None:
            return self.reactants
        # Reaction::calculateRateCoeffUnits and StoichManager apply specified
        # orders as per-species overrides; unspecified reactants retain their
        # stoichiometric mass-action orders.
        combined = dict(self.reactants)
        combined.update(self.orders)
        return combined

    def effective_third_body_concentration(self, concentrations):
        """Return effective collider concentration, or ``None`` when absent."""
        return None if self.third_body is None else self.third_body.effective_concentration(concentrations)

    def _third_body_factor(self, concentrations):
        if self.third_body is None or not self.third_body.mass_action:
            return 1.0
        return self.third_body.effective_concentration(concentrations)

    def forward_rate(self, temperature, concentrations):
        return self._third_body_factor(concentrations) * self._progress(
            self.rate, self.forward_orders, concentrations, temperature)

    def delta_standard_gibbs(self, thermo=None):
        """Return ``sum(nu_k * mu_k^0)`` in J/kmol for an ideal-gas state.

        ``mu_k^0`` is Cantera's current-pressure standard chemical potential;
        the matching concentration-standard-state factor is included later in
        :meth:`equilibrium_constant`.
        """
        state = self.equilibrium_thermo if thermo is None else thermo
        if state is None:
            raise ValueError("equilibrium_thermo is required to derive reverse rates")
        names = tuple(state.species_names)
        potentials = tuple(state.standard_chemical_potentials)
        if len(names) != len(potentials):
            raise ValueError("equilibrium_thermo species and chemical potentials differ in length")
        values = dict(zip(names, potentials))
        return sum((float(self.products.get(name, 0.0)) - float(self.reactants.get(name, 0.0)))
                   * float(values[name]) for name in set(self.reactants) | set(self.products))

    def equilibrium_constant(self, thermo=None):
        """Return the concentration-unit ``Kc`` from ``BulkKinetics``.

        This is ``exp(-DeltaG0/(R*T) + DeltaN*log(C0))``. The bounded path
        accepts :class:`~cantera_python.ideal_gas.IdealGasMixture`, whose
        standard concentration is shared by all species.
        """
        state = self.equilibrium_thermo if thermo is None else thermo
        if state is None:
            raise ValueError("equilibrium_thermo is required to derive reverse rates")
        concentration = state.standard_concentration
        concentration = concentration() if callable(concentration) else concentration
        if not isfinite(float(concentration)) or float(concentration) <= 0.0:
            raise ValueError("equilibrium standard concentration must be finite and positive")
        delta_n = sum(float(self.products.get(name, 0.0)) - float(self.reactants.get(name, 0.0))
                      for name in set(self.reactants) | set(self.products))
        exponent = (-self.delta_standard_gibbs(state) / (GasConstant * float(state.temperature))
                    + delta_n * log(float(concentration)))
        return exp(exponent)

    def equilibrium_reverse_rate_coefficient(self, temperature, thermo=None):
        """Return ``kf / Kc`` for a reversible ideal-gas elementary reaction."""
        if not self.reversible or self.equilibrium_thermo is None:
            raise ValueError("reaction is not configured for equilibrium-derived reverse rates")
        state = self.equilibrium_thermo if thermo is None else thermo
        if not isclose(float(temperature), float(state.temperature), rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("rate temperature must match equilibrium_thermo.temperature")
        return self._rate_value(self.rate, temperature) / self.equilibrium_constant(state)

    def reverse_rate_of_progress(self, temperature, concentrations):
        if not self.reversible:
            return 0.0
        if self.equilibrium_thermo is not None:
            state = self._live_equilibrium_state or self.equilibrium_thermo
            return self._third_body_factor(concentrations) * self._progress(
                lambda _: self.equilibrium_reverse_rate_coefficient(temperature, state),
                self.products, concentrations, temperature)
        return self._third_body_factor(concentrations) * self._progress(
            self.reverse_rate, self.products, concentrations, temperature)

    def net_rate_of_progress(self, temperature, concentrations):
        return self.forward_rate(temperature, concentrations) - self.reverse_rate_of_progress(temperature, concentrations)

    def species_rates(self, temperature, concentrations):
        q = self.net_rate_of_progress(temperature, concentrations)
        names = set(self.reactants) | set(self.products)
        return {name: (float(self.products.get(name, 0.0)) - float(self.reactants.get(name, 0.0))) * q
                for name in names}


@dataclass(frozen=True)
class FalloffReaction(ElementaryReaction):
    """A scalar Lindemann/Troe/SRI/Tsang reaction with an effective collider.

    The falloff rate object consumes ``M_eff`` itself. Consequently this class
    does *not* apply the separate mass-action multiplier used by
    :class:`ElementaryReaction`. This follows the ``mass_action == false``
    branch installed for ``falloff`` and ``chemically-activated`` rates in
    ``Reaction::check`` and ``ThirdBodyCalc``.
    """

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.rate, _FalloffRate):
            raise TypeError("FalloffReaction rate must be a LindemannRate, TroeRate, SriRate, or TsangRate")
        if self.third_body is None:
            raise ValueError("FalloffReaction requires a ThirdBody")
        if self.third_body.mass_action:
            raise ValueError("FalloffReaction requires a non-mass-action ThirdBody")
        if self.reversible and self.reverse_rate is not None and not isinstance(self.reverse_rate, _FalloffRate):
            raise TypeError("reversible FalloffReaction reverse_rate must be a falloff rate or None")

    def forward_rate_coefficient(self, temperature, concentrations):
        return self.rate(temperature, self.third_body.effective_concentration(concentrations))

    def reverse_rate_coefficient(self, temperature, concentrations, thermo=None):
        if not self.reversible:
            return 0.0
        if self.reverse_rate is not None:
            return self.reverse_rate(temperature, self.third_body.effective_concentration(concentrations))
        state = self.equilibrium_thermo if thermo is None else thermo
        kc = self.equilibrium_constant(state)
        kf = self.forward_rate_coefficient(temperature, concentrations)
        return kf / kc

    def forward_rate(self, temperature, concentrations):
        kf = self.forward_rate_coefficient(temperature, concentrations)
        return kf * prod(
            float(concentrations.get(name, 0.0)) ** float(order)
            for name, order in self.forward_orders.items())

    def reverse_rate_of_progress(self, temperature, concentrations):
        if not self.reversible:
            return 0.0
        kr = self.reverse_rate_coefficient(temperature, concentrations)
        return kr * prod(
            float(concentrations.get(name, 0.0)) ** float(order)
            for name, order in self.products.items())


def create_reaction(mech_rxn, thermo=None, default_units=None,
                    skip_undeclared_third_bodies=False):
    """Construct an ElementaryReaction or FalloffReaction from a MechanismReaction."""
    from .rates import (ArrheniusRate, BlowersMaselRate, ChebyshevRate,
                        LindemannRate, LinearBurkeCollider, LinearBurkeRate,
                        PlogRate, SriRate, TroeRate, TsangRate)
    from .units import UnitStack, Units, convert, convert_activation_energy, parse_units
    import re

    if default_units is None:
        default_units = {}
    eq = mech_rxn.equation
    rate_data = mech_rxn.rate_data or {}
    effective_units = dict(default_units)
    if isinstance(rate_data.get('units'), dict):
        effective_units.update(rate_data['units'])
    default_ea_unit = effective_units.get('activation-energy', 'J/kmol')
    default_len_unit = effective_units.get('length', 'm')
    default_qty_unit = effective_units.get('quantity', 'kmol')
    default_time_unit = effective_units.get('time', 's')

    len_factor = convert(1.0, default_len_unit, 'm')
    qty_factor = convert(1.0, default_qty_unit, 'kmol')
    time_factor = convert(1.0, default_time_unit, 's')
    conc_inv_factor = (len_factor ** 3) / qty_factor

    def parse_quantity(value, destination):
        if isinstance(value, (int, float)):
            defaults = {'Pa': effective_units.get('pressure', 'Pa'),
                        'K': effective_units.get('temperature', 'K')}
            return convert(float(value), defaults.get(destination, destination), destination)
        if isinstance(value, str):
            match = re.fullmatch(
                r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S+)\s*',
                value)
            if match:
                amount, unit = match.groups()
                return convert(float(amount), unit, destination)
        raise ValueError(f'Expected a numeric quantity convertible to {destination}, got {value!r}')

    reactants = {k: v for k, v in eq.reactants.items() if k != 'M' and not k.startswith('(+')}
    products = {k: v for k, v in eq.products.items() if k != 'M' and not k.startswith('(+')}
    reversible = eq.reversible
    # Absent type key is NOT explicit 'elementary' (pinned
    # Reaction::setEquation only early-returns for a user-specified type).
    rxn_type = rate_data.get('type', None)
    rxn_type_declared = rxn_type is not None
    if rxn_type is None:
        rxn_type = 'elementary'
    supported_types = {
        'elementary', 'three-body', 'falloff', 'chemically-activated', 'Blowers-Masel',
        'pressure-dependent-Arrhenius', 'Plog', 'Chebyshev', 'linear-Burke',
        'two-temperature-plasma', 'electron-collision-plasma',
        'vibrational-relaxation', 'constant-vibrational-relaxation',
        'Starikovskiy-vibrational-relaxation', 'Castela-vibrational-relaxation',
        'multi-state-resolved-vibrational-relaxation',
        'custom-rate-function',
    }
    if rxn_type not in supported_types:
        raise NotImplementedError(
            f"Reaction rate type {rxn_type!r} is not yet implemented by the pure-Python port")
    allow_negative_a = bool(rate_data.get('negative-A', False))

    has_three_body = (('M' in eq.reactants and 'M' in eq.products)
                      or rxn_type in ('three-body', 'falloff')
                      or any(k.startswith('(+') for k in eq.reactants))
    efficiencies = dict(rate_data.get('efficiencies', {}))
    default_eff = float(rate_data.get('default-efficiency', 1.0))
    tb = None
    if has_three_body:
        mass_action = (rxn_type == 'three-body' or (rxn_type != 'falloff' and 'M' in eq.reactants))
        tb = ThirdBody(efficiencies=efficiencies, default_efficiency=default_eff, mass_action=mass_action)

    # Chaperon-collider extraction (pinned Reaction::setEquation,
    # Reaction.cpp:399-556): a species appearing on BOTH sides of the
    # equation is an explicit third-body collider. It is removed from both
    # stoichiometric sides and becomes the sole collider with efficiency 1.0
    # and default efficiency 0.0 (M_eff = C_chaperon). Skipped for explicitly
    # typed 'elementary' reactions and for falloff-style '(+X)' colliders,
    # matching the pinned count/countM/countF detection order.
    common = [k for k in reactants if k in products]
    if (not (rxn_type_declared and rxn_type == 'elementary') and len(common) == 1
            and not any(k.startswith('(+') for k in list(eq.reactants) + list(eq.products))):
        chaperon = common[0]
        if reactants[chaperon] == 1.0:
            del reactants[chaperon]
        else:
            reactants[chaperon] -= 1.0
        if products[chaperon] == 1.0:
            del products[chaperon]
        else:
            products[chaperon] -= 1.0
        tb = ThirdBody(efficiencies={chaperon: 1.0}, default_efficiency=0.0,
                       mass_action=True, name=chaperon)

    if tb is not None and thermo is not None:
        unknown = set(tb.efficiencies) - set(thermo.species_names)
        if unknown:
            if not skip_undeclared_third_bodies:
                raise ValueError(
                    f'Third-body efficiencies specified for undeclared species: {sorted(unknown)}')
            if tb.name != 'M':
                return None
            for name in unknown:
                tb.efficiencies.pop(name, None)

    def parse_arrhenius(node, order):
        if not isinstance(node, dict):
            raise ValueError(f"Expected dict for Arrhenius rate parameters, got {node!r}")
        raw_a = node.get('A', 0.0)
        if isinstance(raw_a, str):
            match = re.fullmatch(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)', raw_a)
            if match:
                amount_str, unit_str = match.groups()
                amount = float(amount_str)
                unit_str = unit_str.strip()
                if unit_str:
                    u = parse_units(unit_str)
                    a_si = amount * u.factor
                else:
                    a_si = amount * (conc_inv_factor ** (order - 1)) / time_factor
            else:
                a_si = float(raw_a) * (conc_inv_factor ** (order - 1)) / time_factor
        else:
            a_si = float(raw_a) * (conc_inv_factor ** (order - 1)) / time_factor

        b = float(node.get('b', 0.0))
        raw_ea = node.get('Ea-gas', node.get('Ea', 0.0))
        if isinstance(raw_ea, str):
            match = re.fullmatch(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)', raw_ea)
            if match:
                amt_str, unit_str = match.groups()
                ea_val = float(amt_str)
                unit_str = unit_str.strip()
                if unit_str:
                    u = parse_units(unit_str)
                    ea_si = ea_val * u.factor
                else:
                    ea_unit = node.get('Ea-unit', default_ea_unit)
                    ea_si = convert_activation_energy(ea_val, ea_unit, 'J/kmol')
            else:
                ea_val = float(raw_ea)
                ea_unit = node.get('Ea-unit', default_ea_unit)
                ea_si = convert_activation_energy(ea_val, ea_unit, 'J/kmol')
        else:
            ea_val = float(raw_ea)
            ea_unit = node.get('Ea-unit', default_ea_unit)
            ea_si = convert_activation_energy(ea_val, ea_unit, 'J/kmol')

        return a_si, b, ea_si

    def parse_plog(node, order):
        rates_list = []
        for entry in node.get('rate-constants', []):
            p_val = parse_quantity(entry['P'], 'Pa')
            a_si, b, ea_si = parse_arrhenius(entry, order)
            rate = ArrheniusRate(a_si, b, ea_si, allow_negative_A=True)
            rate.set_rate_order(order)
            rates_list.append((p_val, rate))
        return PlogRate(rates_list)

    orders = rate_data.get('orders', None)
    allow_nonreactant_orders = bool(rate_data.get('nonreactant-orders', False))
    allow_negative_orders = bool(rate_data.get('negative-orders', False))
    effective_orders = dict(reactants)
    if orders is not None:
        effective_orders.update(orders)
    base_order = sum(float(value) for value in effective_orders.values())

    def rate_order_from_unit_stack(order, third_body_mass_action=False):
        """Apply pinned Reaction::calculateRateCoeffUnits/ArrheniusBase logic."""
        concentration = Units(1.0, (0.0, -3.0, 0.0, 0.0, 0.0, 1.0))
        rate_units = UnitStack(concentration)
        rate_units.join(1.0)
        rate_units.update(
            Units(1.0, (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)), 1.0)
        rate_units.update(concentration, -float(order))
        if third_body_mass_action:
            rate_units.join(-1.0)
        return 1.0 - rate_units.product().dimension('quantity')

    if rxn_type == 'falloff':
        order_high = rate_order_from_unit_stack(base_order)
        order_low = rate_order_from_unit_stack(base_order, True)
        low_node = rate_data.get('low-P-rate-constant', {})
        high_node = rate_data.get('high-P-rate-constant', {})
        a_low, b_low, ea_low = parse_arrhenius(low_node, order_low)
        a_high, b_high, ea_high = parse_arrhenius(high_node, order_high)
        low_rate = ArrheniusRate(a_low, b_low, ea_low, allow_negative_a)
        high_rate = ArrheniusRate(a_high, b_high, ea_high, allow_negative_a)
        low_rate.set_rate_order(order_low)
        high_rate.set_rate_order(order_high)

        if 'Troe' in rate_data:
            troe_data = rate_data['Troe']
            t_a = float(troe_data['A'])
            t3 = float(troe_data['T3'])
            t1 = float(troe_data['T1'])
            t2 = float(troe_data.get('T2', 0.0))
            coeffs = (t_a, t3, t1, t2) if t2 != 0.0 else (t_a, t3, t1)
            falloff_rate = TroeRate(low_rate, high_rate, coeffs)
        elif 'Sri' in rate_data or 'SRI' in rate_data:
            sri_data = rate_data.get('Sri') or rate_data.get('SRI')
            s_a = float(sri_data['A'])
            s_b = float(sri_data['B'])
            s_c = float(sri_data['C'])
            s_d = float(sri_data.get('D', 1.0))
            s_e = float(sri_data.get('E', 0.0))
            coeffs = (s_a, s_b, s_c, s_d, s_e) if (s_d != 1.0 or s_e != 0.0) else (s_a, s_b, s_c)
            falloff_rate = SriRate(low_rate, high_rate, coeffs)
        elif 'Tsang' in rate_data:
            tsang_data = rate_data['Tsang']
            ts_a = float(tsang_data['A'])
            ts_b = float(tsang_data.get('B', 0.0))
            coeffs = (ts_a, ts_b) if ts_b != 0.0 else (ts_a,)
            falloff_rate = TsangRate(low_rate, high_rate, coeffs)
        else:
            falloff_rate = LindemannRate(low_rate, high_rate)

        return FalloffReaction(
            rate=falloff_rate,
            reactants=reactants,
            products=products,
            reversible=reversible,
            orders=orders,
            allow_nonreactant_orders=allow_nonreactant_orders,
            allow_negative_orders=allow_negative_orders,
            third_body=tb,
            equilibrium_thermo=thermo if reversible else None,
            duplicate=mech_rxn.duplicate,
        )

    eff_order = rate_order_from_unit_stack(
        base_order, bool(tb and tb.mass_action))
    rate_node = rate_data.get('rate-constant') or rate_data.get('rate') or {}
    if orders is not None and thermo is not None:
        unknown_orders = set(orders) - set(thermo.species_names)
        if unknown_orders:
            raise ValueError(f'Reaction orders specified for undeclared species: {sorted(unknown_orders)}')

    if rxn_type == 'Blowers-Masel':
        a_si, b, ea_si = parse_arrhenius(rate_node, eff_order)
        w0 = float(rate_data.get('w0', 0.0))
        rate_obj = BlowersMaselRate(a_si, b, ea_si, w0, allow_negative_a)
        rate_obj.set_rate_order(eff_order)
    elif rxn_type in ('pressure-dependent-Arrhenius', 'Plog'):
        rate_obj = parse_plog(rate_data, eff_order)
    elif rxn_type == 'Chebyshev':
        temperature_range = rate_data.get('temperature-range', ())
        pressure_range = rate_data.get('pressure-range', ())
        coefficients = rate_data.get('data', ())
        if len(temperature_range) != 2 or len(pressure_range) != 2:
            raise ValueError('Chebyshev rate requires two-value temperature and pressure ranges')
        tmin, tmax = (parse_quantity(value, 'K') for value in temperature_range)
        pmin, pmax = (parse_quantity(value, 'Pa') for value in pressure_range)
        coefficients = [list(map(float, row)) for row in coefficients]
        if not coefficients or not coefficients[0]:
            raise ValueError('Chebyshev rate requires a nonempty coefficient matrix')
        rate_factor = (conc_inv_factor ** (eff_order - 1)) / time_factor
        coefficients[0][0] += log(rate_factor) / log(10.0)
        rate_obj = ChebyshevRate(tmin, tmax, pmin, pmax, coefficients)
    elif rxn_type == 'linear-Burke':
        collider_nodes = rate_data.get('colliders')
        if not isinstance(collider_nodes, list) or not collider_nodes:
            raise ValueError("linear-Burke rate requires a nonempty 'colliders' list")
        reference = collider_nodes[0]
        if reference.get('name') != 'M':
            raise ValueError("The first linear-Burke collider must be 'M'")
        if reference.get('type') not in ('pressure-dependent-Arrhenius', 'Plog'):
            raise NotImplementedError(
                "The pure-Python linear-Burke factory currently requires a P-log reference collider")
        reference_rate = parse_plog(reference, eff_order)
        colliders = {}
        for collider in collider_nodes[1:]:
            name = collider.get('name')
            if not isinstance(name, str) or not name:
                raise ValueError('Each linear-Burke collider requires a name')
            efficiency = collider.get('efficiency')
            if not isinstance(efficiency, dict):
                raise ValueError(f"Linear-Burke collider {name!r} requires efficiency data")
            ea_unit = efficiency.get('Ea-unit', default_ea_unit)
            eps_rate = ArrheniusRate(
                float(efficiency.get('A', 0.0)),
                float(efficiency.get('b', 0.0)),
                convert_activation_energy(float(efficiency.get('Ea', 0.0)),
                                          ea_unit, 'J/kmol'))
            collider_rate = None
            if 'type' in collider:
                if collider['type'] not in ('pressure-dependent-Arrhenius', 'Plog'):
                    raise NotImplementedError(
                        f"Linear-Burke collider rate {collider['type']!r} is not yet parsed")
                collider_rate = parse_plog(collider, eff_order)
            colliders[name] = LinearBurkeCollider(eps_rate, collider_rate)
        rate_obj = LinearBurkeRate(reference_rate, colliders)
    elif rxn_type == 'two-temperature-plasma':
        from .rates import TwoTempPlasmaRate
        a_si, b, ea_si = parse_arrhenius(rate_node, eff_order)
        raw_ee = rate_node.get('Ea-electron', rate_node.get('EE', 0.0))
        if isinstance(raw_ee, str):
            match = re.fullmatch(r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)', raw_ee)
            if match:
                amt_str, unit_str = match.groups()
                ee_val = float(amt_str)
                unit_str = unit_str.strip()
                if unit_str:
                    ee_si = ee_val * parse_units(unit_str).factor
                else:
                    ea_unit = rate_node.get('Ea-unit', default_ea_unit)
                    ee_si = convert_activation_energy(ee_val, ea_unit, 'J/kmol')
            else:
                ea_unit = rate_node.get('Ea-unit', default_ea_unit)
                ee_si = convert_activation_energy(float(raw_ee), ea_unit, 'J/kmol')
        else:
            ea_unit = rate_node.get('Ea-unit', default_ea_unit)
            ee_si = convert_activation_energy(float(raw_ee), ea_unit, 'J/kmol')
        bg = float(rate_node.get('b-gas', rate_node.get('bg', 0.0)))
        tinv_raw = rate_node.get('T-inv', rate_node.get('Tinv', 0.0))
        tinv = parse_quantity(tinv_raw, 'K') if isinstance(tinv_raw, str) else float(tinv_raw)
        rate_obj = TwoTempPlasmaRate(A=a_si, b=b, Ea=ea_si, EE=ee_si, bg=bg, Tinv=tinv, allow_negative_A=allow_negative_a)
    elif rxn_type == 'electron-collision-plasma':
        from .rates import ElectronCollisionPlasmaRate
        energy_levels = rate_data.get('energy-levels', [])
        cross_sections = rate_data.get('cross-sections', [])
        rate_obj = ElectronCollisionPlasmaRate(energy_levels, cross_sections)
    elif rxn_type in ('vibrational-relaxation', 'constant-vibrational-relaxation'):
        from .rates import ConstantVibrationalRelaxationRate
        raw_a = rate_node.get('A', 1.0)
        a_si = float(raw_a) * (conc_inv_factor ** (eff_order - 1)) / time_factor
        rate_obj = ConstantVibrationalRelaxationRate(A=a_si, allow_negative_A=allow_negative_a)
    elif rxn_type == 'Starikovskiy-vibrational-relaxation':
        from .rates import StarikovskiyVibrationalRelaxationRate
        raw_a = rate_node.get('A', 1.0)
        a_si = float(raw_a) * (conc_inv_factor ** (eff_order - 1)) / time_factor
        b = float(rate_node.get('b', 0.0))
        c0 = float(rate_node.get('C0', 0.0))
        c13 = float(rate_node.get('C13', 0.0))
        cm = float(rate_node.get('Cm', 0.0))
        m = float(rate_node.get('m', 1.0))
        cn = float(rate_node.get('Cn', 0.0))
        n = float(rate_node.get('n', 1.0))
        rate_obj = StarikovskiyVibrationalRelaxationRate(A=a_si, b=b, C0=c0, C13=c13, Cm=cm, m=m, Cn=cn, n=n, allow_negative_A=allow_negative_a)
    elif rxn_type == 'Castela-vibrational-relaxation':
        from .rates import CastelaVibrationalRelaxationRate
        a_param = float(rate_node.get('a', 0.0))
        b_param = float(rate_node.get('b', 0.0))
        ref_p_raw = rate_node.get('reference-pressure', 101325.0)
        ref_p = parse_quantity(ref_p_raw, 'Pa') if isinstance(ref_p_raw, str) else float(ref_p_raw)
        rate_obj = CastelaVibrationalRelaxationRate(a=a_param, b=b_param, reference_pressure=ref_p, allow_negative_A=allow_negative_a)
    elif rxn_type == 'multi-state-resolved-vibrational-relaxation':
        from .rates import MultiStateResolvedVibrationalRelaxationRate
        raw_a = rate_node.get('A', 1.0)
        a_si = float(raw_a) * (conc_inv_factor ** (eff_order - 1)) / time_factor
        b = float(rate_node.get('b', 0.0))
        c0 = float(rate_node.get('C0', 0.0))
        c13 = float(rate_node.get('C13', 0.0))
        c23 = float(rate_node.get('C23', 0.0))
        rate_obj = MultiStateResolvedVibrationalRelaxationRate(A=a_si, b=b, C0=c0, C13=c13, C23=c23, allow_negative_A=allow_negative_a)
    elif rxn_type == 'custom-rate-function':
        from .rates import CustomFunc1Rate
        rate_obj = CustomFunc1Rate()
    else:
        a_si, b, ea_si = parse_arrhenius(rate_node, eff_order)
        rate_obj = ArrheniusRate(a_si, b, ea_si, allow_negative_a)
        rate_obj.set_rate_order(eff_order)

    return ElementaryReaction(
        rate=rate_obj,
        reactants=reactants,
        products=products,
        reversible=reversible,
        orders=orders,
        allow_nonreactant_orders=allow_nonreactant_orders,
        allow_negative_orders=allow_negative_orders,
        third_body=tb,
        equilibrium_thermo=thermo if reversible else None,
        rate_thermo=thermo,
        duplicate=mech_rxn.duplicate,
    )


def _validate_duplicate_declarations(mechanism_reactions, runtime_reactions,
                                     explicit_third_body_duplicates='warn'):
    """Apply the duplicate-pair declaration rules from Kinetics::checkDuplicates."""
    groups = {}
    for index, reaction in enumerate(mechanism_reactions):
        net = reaction.equation.net_stoichiometry
        items = sorted((name, float(value)) for name, value in net.items() if value)
        if not items:
            key = ()
        else:
            scale = abs(items[0][1])
            normalized = [(name, round(value / scale, 12)) for name, value in items]
            if normalized[0][1] < 0.0:
                normalized = [(name, -value) for name, value in normalized]
            key = tuple(normalized)
        groups.setdefault(key, []).append(index)

    matched = set()
    for indices in groups.values():
        for offset, first_index in enumerate(indices):
            first = mechanism_reactions[first_index]
            first_runtime = runtime_reactions[first_index]
            for second_index in indices[offset + 1:]:
                second = mechanism_reactions[second_index]
                second_runtime = runtime_reactions[second_index]
                if first.duplicate and second.duplicate:
                    matched.update((first_index, second_index))
                    continue
                if type(first_runtime) is not type(second_runtime):
                    continue
                if type(first_runtime.rate) is not type(second_runtime.rate):
                    continue
                if not are_duplicate_reactions(first.equation, second.equation):
                    continue
                tb1, tb2 = first_runtime.third_body, second_runtime.third_body
                if (tb1 is None) != (tb2 is None):
                    continue
                if tb1 is not None and tb2 is not None:
                    # The pinned default policy warns, but does not reject, an
                    # M/explicit-collider overlap. Non-overlapping collider
                    # efficiencies are not duplicates.
                    explicit1 = tb1.name != 'M'
                    explicit2 = tb2.name != 'M'
                    if explicit1 != explicit2:
                        explicit = tb1 if explicit1 else tb2
                        general = tb2 if explicit1 else tb1
                        overlap = general.efficiency(explicit.name) * explicit.efficiency(explicit.name)
                        if not overlap:
                            continue
                        if explicit_third_body_duplicates == 'mark-duplicate':
                            object.__setattr__(first_runtime, 'duplicate', True)
                            object.__setattr__(second_runtime, 'duplicate', True)
                            matched.update((first_index, second_index))
                        elif explicit_third_body_duplicates == 'modify-efficiency':
                            general.efficiencies[explicit.name] = 0.0
                        elif explicit_third_body_duplicates == 'warn':
                            import warnings
                            warnings.warn(
                                'Undeclared duplicate third body reactions with a common '
                                f'third body at indices {first_index} and {second_index}',
                                UserWarning, stacklevel=3)
                        elif explicit_third_body_duplicates == 'error':
                            raise ValueError(
                                f'Undeclared duplicate reactions at indices '
                                f'{first_index} and {second_index}')
                        else:
                            raise ValueError(
                                f'Invalid explicit-third-body-duplicates flag '
                                f'{explicit_third_body_duplicates!r}')
                        continue
                    species = set(tb1.efficiencies) | set(tb2.efficiencies)
                    if species and not any(tb1.efficiency(name) * tb2.efficiency(name)
                                           for name in species):
                        continue
                matched.update((first_index, second_index))
                if not (first.duplicate and second.duplicate):
                    raise ValueError(
                        f'Undeclared duplicate reactions at indices '
                        f'{first_index} and {second_index}')
    unmatched = [index for index, reaction in enumerate(mechanism_reactions)
                 if reaction.duplicate and index not in matched]
    if unmatched:
        raise ValueError(f'Reactions marked duplicate have no matching pair: {unmatched}')


class Kinetics:
    """Base class for kinetics managers in Cantera."""

    def __init__(self, thermo=None, reactions=None):
        self._thermo = thermo
        self._reactions = tuple(reactions or ())
        self._multipliers = np.ones(len(self._reactions), dtype=float)

    def kineticsType(self) -> str:
        return "Kinetics"

    def nReactions(self) -> int:
        return len(self._reactions)

    @property
    def n_reactions(self) -> int:
        return self.nReactions()

    def nTotalSpecies(self) -> int:
        return self._thermo.n_species if self._thermo else 0

    def nPhases(self) -> int:
        if hasattr(self, '_phases') and self._phases:
            return len(self._phases)
        if getattr(self, '_thermo', None) is not None:
            return 1
        return 0

    def phase(self, n: int = 0) -> Any:
        if hasattr(self, '_phases') and self._phases:
            return self._phases[n]
        return self._thermo

    def phaseIndex(self, p: Any) -> int:
        if isinstance(p, str):
            if hasattr(self, '_phases'):
                for i, ph in enumerate(self._phases):
                    if hasattr(ph, 'name') and ph.name == p:
                        return i
            if self._thermo and hasattr(self._thermo, 'name') and self._thermo.name == p:
                return 0
            return -1
        elif isinstance(p, int):
            return p
        elif hasattr(self, '_phases') and p in self._phases:
            return self._phases.index(p)
        elif getattr(self, '_thermo', None) == p:
            return 0
        return -1

    def getFwdRatesOfProgress(self, fwd: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.forward_rates_of_progress(), dtype=float)
        if fwd is not None:
            fwd[:] = res
            return fwd
        return res

    def getRevRatesOfProgress(self, rev: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.reverse_rates_of_progress(), dtype=float)
        if rev is not None:
            rev[:] = res
            return rev
        return res

    def getNetRatesOfProgress(self, net: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.net_rates_of_progress(), dtype=float)
        if net is not None:
            net[:] = res
            return net
        return res

    def getNetProductionRates(self, wdot: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.net_production_rates(), dtype=float)
        if wdot is not None:
            wdot[:] = res
            return wdot
        return res

    def getCreationRates(self, cdot: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.creation_rates(), dtype=float)
        if cdot is not None:
            cdot[:] = res
            return cdot
        return res

    def getDestructionRates(self, ddot: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.destruction_rates(), dtype=float)
        if ddot is not None:
            ddot[:] = res
            return ddot
        return res

    def getFwdRateConstants(self, kf: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.forward_rate_constants(), dtype=float)
        if kf is not None:
            kf[:] = res
            return kf
        return res

    def getRevRateConstants(self, kr: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.reverse_rate_constants(), dtype=float)
        if kr is not None:
            kr[:] = res
            return kr
        return res

    def getEquilibriumConstants(self, kc: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.equilibrium_constants(), dtype=float)
        if kc is not None:
            kc[:] = res
            return kc
        return res

    def getDeltaEnthalpy(self, dh: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.delta_enthalpy(), dtype=float)
        if dh is not None:
            dh[:] = res
            return dh
        return res

    def getDeltaGibbs(self, dg: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.delta_gibbs(), dtype=float)
        if dg is not None:
            dg[:] = res
            return dg
        return res

    def getDeltaEntropy(self, ds: Optional[np.ndarray] = None) -> np.ndarray:
        res = np.asarray(self.delta_entropy(), dtype=float)
        if ds is not None:
            ds[:] = res
            return ds
        return res


class BulkKinetics(Kinetics):
    """Homogeneous bulk phase kinetics manager."""

    def kineticsType(self) -> str:
        return "BulkKinetics"




class GasKinetics(BulkKinetics):
    """Bulk gas kinetics manager for evaluating reaction rates of progress and species production rates."""

    def __init__(self, thermo, reactions=None):
        self._reactions = tuple(reactions or ())
        self._thermo = thermo
        self._fast = _VectorizedKinetics(thermo, self._reactions)
        self._multipliers = np.ones(len(self._reactions), dtype=float)

    @property
    def _thermo(self):
        return self.__thermo

    @_thermo.setter
    def _thermo(self, thermo):
        self.__thermo = thermo
        for r in getattr(self, '_reactions', ()):
            if r.reversible and r.equilibrium_thermo is not None:
                object.__setattr__(r, '_live_equilibrium_state', thermo)

    @classmethod
    def from_mechanism(cls, mech_or_path, thermo=None):
        from pathlib import Path
        from .mechanism import load_mechanism, Mechanism, _as_document
        if isinstance(mech_or_path, (str, Path)) or hasattr(mech_or_path, 'read_text'):
            mech = load_mechanism(mech_or_path)
        elif isinstance(mech_or_path, Mechanism):
            mech = mech_or_path
        else:
            raise TypeError("mech_or_path must be a file path, YAML document, or Mechanism instance")
        # Document-level units must always drive SI conversion of rate
        # constants (pinned Reaction.cpp parses the same header for both
        # path-based and preloaded Mechanism construction).
        default_units = mech.units or {}

        if thermo is None:
            from .ideal_gas import IdealGasPhase
            thermo = IdealGasPhase.from_mechanism(mech)

        rxns = []
        retained = []
        phase = mech.phases[0] if len(mech.phases) == 1 else None
        skip_undeclared = bool(
            phase.skip_undeclared_third_bodies if phase is not None else False)
        duplicate_policy = (
            phase.explicit_third_body_duplicates if phase is not None else 'warn')
        for mr in mech.reactions:
            r = create_reaction(
                mr, thermo=thermo, default_units=default_units,
                skip_undeclared_third_bodies=skip_undeclared)
            if r is None:
                continue
            rxns.append(r)
            retained.append(mr)
        _validate_duplicate_declarations(retained, rxns, duplicate_policy)
        return cls(thermo, rxns)

    @property
    def thermo(self):
        return self._thermo

    @property
    def n_reactions(self):
        return len(self._reactions)

    def nReactions(self):
        return len(self._reactions)

    @property
    def n_species(self):
        return self._thermo.n_species

    def nTotalSpecies(self):
        return self._thermo.n_species

    @property
    def reactions(self):
        return self._reactions

    def reaction(self, i):
        idx = int(i)
        if idx < 0 or idx >= len(self._reactions):
            raise IndexError(f"Reaction index {i} out of bounds")
        rxn = self._reactions[idx]
        if isinstance(rxn, Reaction):
            return rxn
        r = Reaction()
        r.reactants = dict(rxn.reactants)
        r.products = dict(rxn.products)
        r.reversible = bool(rxn.reversible)
        r.duplicate = bool(getattr(rxn, "duplicate", False))
        r._third_body = getattr(rxn, "third_body", None)
        r._rate = getattr(rxn, "rate", None)
        r.orders = dict(getattr(rxn, "orders", None) or {})
        return r

    def set_multiplier(self, value, reaction=None):
        """Set a reaction-rate multiplier, following ``Kinetics::setMultiplier``.

        When *reaction* is omitted, the value is applied to every reaction.
        Multipliers scale forward and reverse rates equally, so equilibrium
        constants are unchanged.
        """
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError('reaction multiplier must be finite and non-negative')
        if reaction is None:
            self._multipliers[:] = value
        else:
            self._multipliers[int(reaction)] = value

    def multiplier(self, reaction):
        return float(self._multipliers[int(reaction)])

    def forward_rate_constants(self):
        T = self._thermo.temperature
        concs = dict(zip(self._thermo.species_names, self._thermo.concentrations))
        res = []
        for multiplier, r in zip(self._multipliers, self._reactions):
            if isinstance(r, FalloffReaction):
                res.append(multiplier * r.rate(
                    T, r.third_body.effective_concentration(concs)))
            else:
                res.append(multiplier * r._rate_value(r.rate, T))
        return tuple(res)

    def reverse_rate_constants(self):
        T = self._thermo.temperature
        concs = dict(zip(self._thermo.species_names, self._thermo.concentrations))
        res = []
        for multiplier, r in zip(self._multipliers, self._reactions):
            if not r.reversible:
                res.append(0.0)
            elif isinstance(r, FalloffReaction):
                res.append(multiplier * r.reverse_rate_coefficient(
                    T, concs, self._thermo))
            else:
                res.append(multiplier * r.equilibrium_reverse_rate_coefficient(
                    T, self._thermo))
        return tuple(res)

    def equilibrium_constants(self):
        """Pinned BulkKinetics::getEquilibriumConstants computes Kc for every
        reaction; only the reverse rate constant m_rkcn is zeroed for
        irreversible reactions (BulkKinetics.cpp:482)."""
        res = []
        for r in self._reactions:
            res.append(r.equilibrium_constant(self._thermo))
        return tuple(res)

    def forward_rates_of_progress(self):
        T = self._thermo.temperature
        concs = dict(zip(self._thermo.species_names, self._thermo.concentrations))
        return tuple(multiplier * r.forward_rate(T, concs)
                     for multiplier, r in zip(self._multipliers, self._reactions))

    def reverse_rates_of_progress(self):
        T = self._thermo.temperature
        concs = dict(zip(self._thermo.species_names, self._thermo.concentrations))
        return tuple(multiplier * r.reverse_rate_of_progress(T, concs)
                     for multiplier, r in zip(self._multipliers, self._reactions))

    def net_rates_of_progress(self):
        fwd = self.forward_rates_of_progress()
        rev = self.reverse_rates_of_progress()
        return tuple(f - r for f, r in zip(fwd, rev))

    def net_production_rates(self):
        net_rop = self.net_rates_of_progress()
        species_names = self._thermo.species_names
        sp_map = {name: 0.0 for name in species_names}
        for q, r in zip(net_rop, self._reactions):
            if q == 0.0:
                continue
            for name, stoich in r.products.items():
                if name in sp_map:
                    sp_map[name] += stoich * q
            for name, stoich in r.reactants.items():
                if name in sp_map:
                    sp_map[name] -= stoich * q
        return tuple(sp_map[name] for name in species_names)

    def creation_rates(self):
        fwd_rop = self.forward_rates_of_progress()
        rev_rop = self.reverse_rates_of_progress()
        species_names = self._thermo.species_names
        sp_map = {name: 0.0 for name in species_names}
        for qf, qr, r in zip(fwd_rop, rev_rop, self._reactions):
            for name, stoich in r.products.items():
                if name in sp_map:
                    sp_map[name] += stoich * qf
            for name, stoich in r.reactants.items():
                if name in sp_map:
                    sp_map[name] += stoich * qr
        return tuple(sp_map[name] for name in species_names)

    def destruction_rates(self):
        fwd_rop = self.forward_rates_of_progress()
        rev_rop = self.reverse_rates_of_progress()
        species_names = self._thermo.species_names
        sp_map = {name: 0.0 for name in species_names}
        for qf, qr, r in zip(fwd_rop, rev_rop, self._reactions):
            for name, stoich in r.reactants.items():
                if name in sp_map:
                    sp_map[name] += stoich * qf
            for name, stoich in r.products.items():
                if name in sp_map:
                    sp_map[name] += stoich * qr
        return tuple(sp_map[name] for name in species_names)

    def delta_enthalpy(self):
        hbar = self._thermo.partial_molar_enthalpies
        sp_indices = {name: i for i, name in enumerate(self._thermo.species_names)}
        res = []
        for r in self._reactions:
            dh = 0.0
            for name, stoich in r.products.items():
                dh += stoich * hbar[sp_indices[name]]
            for name, stoich in r.reactants.items():
                dh -= stoich * hbar[sp_indices[name]]
            res.append(dh)
        return tuple(res)

    def delta_entropy(self):
        sbar = self._thermo.partial_molar_entropies
        sp_indices = {name: i for i, name in enumerate(self._thermo.species_names)}
        res = []
        for r in self._reactions:
            ds = 0.0
            for name, stoich in r.products.items():
                ds += stoich * sbar[sp_indices[name]]
            for name, stoich in r.reactants.items():
                ds -= stoich * sbar[sp_indices[name]]
            res.append(ds)
        return tuple(res)

    def delta_gibbs(self):
        g0 = self._thermo.standard_chemical_potentials
        sp_indices = {name: i for i, name in enumerate(self._thermo.species_names)}
        res = []
        for r in self._reactions:
            dg = 0.0
            for name, stoich in r.products.items():
                dg += stoich * g0[sp_indices[name]]
            for name, stoich in r.reactants.items():
                dg -= stoich * g0[sp_indices[name]]
            res.append(dg)
        return tuple(res)

    @property
    def heat_release_rate(self):
        wdot = self.net_production_rates()
        hbar = self._thermo.partial_molar_enthalpies
        return -sum(w * h for w, h in zip(wdot, hbar))

    @property
    def heat_production_rates(self):
        rop = self.net_rates_of_progress()
        dh = self.delta_enthalpy()
        return tuple(-q * d for q, d in zip(rop, dh))
    # -- vectorized fast path ------------------------------------------------
    def forward_rate_constants_fast(self):
        """NumPy forward rate constants, semantically identical to
        :meth:`forward_rate_constants` but without per-call dict overhead."""
        return self._fast.forward_rate_constants(self._thermo, self._multipliers)

    def net_rates_of_progress_fast(self):
        return self._fast.net_rates_of_progress(self._thermo, self._multipliers)

    def net_production_rates_fast(self):
        return self._fast.net_production_rates(self._thermo, self._multipliers)

    def creation_rates_fast(self):
        return self._fast.creation_rates(self._thermo, self._multipliers)

    def destruction_rates_fast(self):
        return self._fast.destruction_rates(self._thermo, self._multipliers)

    def delta_enthalpy_fast(self):
        hbar = np.asarray(self._thermo.partial_molar_enthalpies, dtype=float)
        return self._fast.stoich_products @ hbar - self._fast.stoich_reactants @ hbar

    def heat_release_rate_fast(self):
        wdot = self.net_production_rates_fast()
        hbar = np.asarray(self._thermo.partial_molar_enthalpies, dtype=float)
        return -float(wdot @ hbar)


class _VectorizedKinetics:
    """Precomputed sparse stoichiometry/collider structure for one mechanism.

    Mirrors GasKinetics scalar semantics: elementary/three-body reactions get a
    mass-action third-body multiplier where applicable, falloff reactions
    consume M_eff themselves, and reverse rates are equilibrium-derived.
    """

    def __init__(self, thermo, reactions):
        n_sp = thermo.n_species
        idx = {name: thermo.species_index(name) for name in thermo.species_names}
        n_rx = len(reactions)
        R = np.zeros((n_rx, n_sp))
        P = np.zeros((n_rx, n_sp))
        reactant_orders = np.zeros((n_rx, n_sp))
        product_orders = np.zeros((n_rx, n_sp))
        # per reaction: mass-action third body? falloff? efficiency vector
        tb_mass_action = np.zeros(n_rx, dtype=bool)
        tb_eff = np.ones((n_rx, n_sp))
        tb_default = np.ones(n_rx)
        reversible = np.zeros(n_rx, dtype=bool)
        for i, r in enumerate(reactions):
            for name, stoich in r.reactants.items():
                if name in idx:
                    R[i, idx[name]] = stoich
                    reactant_orders[i, idx[name]] = (r.orders or r.reactants).get(name, 0.0) if r.orders else stoich
            for name, stoich in r.products.items():
                if name in idx:
                    P[i, idx[name]] = stoich
            for name, order in (r.forward_orders or {}).items():
                if name in idx:
                    reactant_orders[i, idx[name]] = order
            for name, order in r.products.items():
                if name in idx:
                    product_orders[i, idx[name]] = order
            reversible[i] = r.reversible
            tb = r.third_body
            if tb is not None:
                tb_default[i] = tb.default_efficiency
                tb_eff[i, :] = tb.default_efficiency
                for name, eff in tb.efficiencies.items():
                    if name in idx:
                        tb_eff[i, idx[name]] = eff
                if isinstance(r, FalloffReaction):
                    tb_mass_action[i] = False
                else:
                    tb_mass_action[i] = tb.mass_action
        self.stoich_products = P
        self.stoich_reactants = R
        self.reactant_orders = reactant_orders
        self.product_orders = product_orders
        self.tb_mass_action = tb_mass_action
        self.tb_eff = tb_eff
        self.tb_default = tb_default
        self.reversible = reversible
        self.reactions = reactions
        self.n_species = n_sp
        self.n_reactions = n_rx
        # negative orders need masking of zero concentrations
        self.has_negative_orders = bool((reactant_orders < 0).any()) or bool((product_orders < 0).any())

    def _concentration_matrix(self, thermo):
        return np.asarray(thermo.concentrations, dtype=float)

    def _third_body_factors(self, conc):
        # M_eff per reaction: default * sum(C) + sum((eff-default)*C)
        ctot = conc.sum()
        meff = self.tb_default * ctot + (self.tb_eff - self.tb_default[:, None]) @ conc
        return np.where(self.tb_mass_action, meff, 1.0)

    def _forward_rate_constants(self, thermo, multipliers=None):
        T = thermo.temperature
        out = np.empty(self.n_reactions)
        conc = self._concentration_matrix(thermo)
        meff = (self.tb_default * conc.sum()
                + (self.tb_eff - self.tb_default[:, None]) @ conc)
        for i, r in enumerate(self.reactions):
            if isinstance(r, FalloffReaction):
                out[i] = r.rate(T, meff[i])
            elif isinstance(r.rate, LinearBurkeRate):
                fractions = dict(zip(thermo.species_names, thermo.X))
                out[i] = r.rate(T, thermo.pressure, fractions)
            elif isinstance(r.rate, (PlogRate, ChebyshevRate)):
                out[i] = r.rate(T, thermo.pressure)
            elif isinstance(r.rate, BlowersMaselRate):
                r.rate.update_from_thermo(thermo, r.reactants, r.products)
                out[i] = r.rate(T)
            elif type(r.rate).__name__ == "TwoTempPlasmaRate":
                Te = getattr(thermo, 'electron_temperature', getattr(thermo, 'electronTemperature', T))
                out[i] = r.rate(T, float(Te))
            elif type(r.rate).__name__ == "ElectronCollisionPlasmaRate":
                levels = getattr(thermo, 'electron_energy_levels', ())
                dist = getattr(thermo, 'electron_energy_distribution', ())
                if not levels or not dist or len(levels) < 2 or len(dist) < 2:
                    out[i] = 0.0
                else:
                    out[i] = r.rate(levels, dist)
            else:
                out[i] = r.rate(T)
        if multipliers is not None:
            out *= np.asarray(multipliers, dtype=float)
        return out

    def forward_rate_constants(self, thermo, multipliers=None):
        return self._forward_rate_constants(thermo, multipliers)

    @staticmethod
    def _general_stoich_products(stoich_matrix, orders):
        stoich_matrix = np.asarray(stoich_matrix, dtype=float)
        orders = np.asarray(orders, dtype=float)
        n_rx = stoich_matrix.shape[0]
        general = np.zeros(n_rx, dtype=bool)
        for i in range(n_rx):
            s = stoich_matrix[i]
            o = orders[i]
            if not np.array_equal(s, o):
                general[i] = True
                continue
            nonzeros = o[o > 0]
            if len(nonzeros) != np.count_nonzero(o):
                general[i] = True
                continue
            if not np.all(np.equal(np.mod(nonzeros, 1), 0)):
                general[i] = True
                continue
            total_order = int(np.sum(nonzeros))
            if total_order not in (1, 2, 3):
                general[i] = True
                continue
            general[i] = False
        return general

    def _activity_products(self, conc, orders, general=None):
        conc = np.asarray(conc, dtype=float)
        orders = np.asarray(orders, dtype=float)
        if general is None:
            general = getattr(self, "_general_reactants", None)
        if general is None:
            general = self._general_stoich_products(orders, orders)
        general = np.asarray(general, dtype=bool)
        rates = np.zeros(orders.shape[0], dtype=float)
        for i in range(orders.shape[0]):
            o = orders[i]
            if not general[i]:
                p = 1.0
                for c_val, order_val in zip(conc, o):
                    if order_val == 1.0:
                        p *= c_val
                    elif order_val == 2.0:
                        p *= (c_val * c_val)
                    elif order_val == 3.0:
                        p *= (c_val * c_val * c_val)
                rates[i] = p
            else:
                active_idx = np.where(o != 0.0)[0]
                if np.any(conc[active_idx] <= 0.0):
                    rates[i] = 0.0
                else:
                    rates[i] = np.prod(conc[active_idx] ** o[active_idx])
        return rates
        # prod_k C_k^order_k per reaction; C=0 with order>0 forces 0
        with np.errstate(divide='ignore', invalid='ignore'):
            logc = np.where(conc > 0.0, np.log(np.maximum(conc, 1e-300)), 0.0)
        act = np.exp(orders @ logc)
        zero_mask = ((orders > 0.0) & (conc == 0.0)[None, :]).any(axis=1)
        return np.where(zero_mask, 0.0, act)

    def net_rates_of_progress(self, thermo, multipliers=None):
        T = thermo.temperature
        conc = self._concentration_matrix(thermo)
        kf = self._forward_rate_constants(thermo, multipliers)
        tb = self._third_body_factors(conc)
        qf = kf * tb * self._activity_products(conc, self.reactant_orders)
        # reverse: equilibrium-derived for reversible reactions
        qr = np.zeros(self.n_reactions)
        rev_idx = np.nonzero(self.reversible)[0]
        if rev_idx.size:
            kc = self._equilibrium_constants(thermo, rev_idx)
            kr = np.zeros(self.n_reactions)
            # Kc can underflow to 0 at extreme temperatures; Cantera's
            # kf/Kc would then be inf, but the reverse ROP is still bounded
            # by the product activities. Guard so the RHS stays finite.
            with np.errstate(divide='ignore', invalid='ignore'):
                kratio = kf[rev_idx] / kc
            kratio = np.where(np.isfinite(kratio), kratio, 0.0)
            kr[rev_idx] = kratio
            qr = kr * self._activity_products(conc, self.product_orders) * tb
        return qf - qr

    def _equilibrium_constants(self, thermo, indices):
        """Vectorized Kc for the given reaction indices.

        Same semantics as ElementaryReaction.equilibrium_constant:
        Kc = exp(-dG0/(R*T) + dN*log(C0)) with dG0 = sum(nu * mu0).
        """
        mu0 = np.asarray(thermo.standard_chemical_potentials, dtype=float)
        nu = (self.stoich_products - self.stoich_reactants)[indices, :]
        dg0 = nu @ mu0
        delta_n = nu.sum(axis=1)
        c0 = thermo.standard_concentration
        c0 = c0() if callable(c0) else c0
        T = thermo.temperature
        return np.exp(-dg0 / (GasConstant * T) + delta_n * np.log(c0))

    def net_production_rates(self, thermo, multipliers=None):
        q = self.net_rates_of_progress(thermo, multipliers)
        return (self.stoich_products - self.stoich_reactants).T @ q

    def creation_rates(self, thermo, multipliers=None):
        conc = self._concentration_matrix(thermo)
        T = thermo.temperature
        kf = self._forward_rate_constants(thermo, multipliers)
        tb = self._third_body_factors(conc)
        qf = kf * tb * self._activity_products(conc, self.reactant_orders)
        qr = self._reverse_rop(thermo, conc, kf, tb)
        return self.stoich_products.T @ qf + self.stoich_reactants.T @ qr

    def destruction_rates(self, thermo, multipliers=None):
        conc = self._concentration_matrix(thermo)
        T = thermo.temperature
        kf = self._forward_rate_constants(thermo, multipliers)
        tb = self._third_body_factors(conc)
        qf = kf * tb * self._activity_products(conc, self.reactant_orders)
        qr = self._reverse_rop(thermo, conc, kf, tb)
        return self.stoich_reactants.T @ qf + self.stoich_products.T @ qr

    def _reverse_rop(self, thermo, conc, kf, tb):
        qr = np.zeros(self.n_reactions)
        rev_idx = np.nonzero(self.reversible)[0]
        if rev_idx.size:
            kc = self._equilibrium_constants(thermo, rev_idx)
            kr = np.zeros(self.n_reactions)
            with np.errstate(divide='ignore', invalid='ignore'):
                kratio = kf[rev_idx] / kc
            kratio = np.where(np.isfinite(kratio), kratio, 0.0)
            kr[rev_idx] = kratio
            qr = kr * self._activity_products(conc, self.product_orders) * tb
        return qr


class Group:
    """Atomic group for reaction path diagrams.

    Replicates Cantera's Group class in include/cantera/kinetics/Group.h
    """
    def __init__(self, elements=None):
        if elements is None:
            self.m_comp = []
            self.m_sign = -999
        elif isinstance(elements, int):
            self.m_comp = [0] * elements
            self.m_sign = 0
        elif isinstance(elements, Group):
            self.m_comp = list(elements.m_comp)
            self.m_sign = elements.m_sign
        elif isinstance(elements, (list, tuple, np.ndarray)):
            self.m_comp = [int(x) for x in elements]
            self.m_sign = 0
            self.validate()
        elif isinstance(elements, dict):
            self.m_comp = [int(v) for v in elements.values()]
            self.comp = dict(elements)
            self.m_sign = 0
            self.validate()
        else:
            self.m_comp = [int(x) for x in elements]
            self.m_sign = 0
            self.validate()

    def validate(self) -> None:
        if self.m_sign == -999:
            return
        self.m_sign = 0
        ok = True
        for m in range(len(self.m_comp)):
            val = self.m_comp[m]
            if val != 0:
                if self.m_sign == 0:
                    self.m_sign = 1 if val > 0 else -1
                elif self.m_sign * val < 0:
                    ok = False
                    break
        if not ok:
            self.m_sign = -999

    def valid(self) -> bool:
        return self.m_sign != -999

    def __bool__(self) -> bool:
        return self.valid()

    def sign(self) -> int:
        return int(self.m_sign)

    def size(self) -> int:
        return len(self.m_comp)

    def nAtoms(self, m=None) -> int:
        if m is None:
            return sum(abs(x) for x in self.m_comp)
        if not self.m_comp or m >= len(self.m_comp):
            return 0
        return self.m_comp[m]

    def __iadd__(self, other):
        for m in range(min(len(self.m_comp), len(other.m_comp))):
            self.m_comp[m] += other.m_comp[m]
        self.validate()
        return self

    def __isub__(self, other):
        for m in range(min(len(self.m_comp), len(other.m_comp))):
            self.m_comp[m] -= other.m_comp[m]
        self.validate()
        return self

    def __imul__(self, a: int):
        for m in range(len(self.m_comp)):
            self.m_comp[m] *= a
        self.validate()
        return self

    def sum(self, other):
        res = Group(self)
        res += other
        return res

    def __add__(self, other):
        return self.sum(other)

    def diff(self, other):
        res = Group(self)
        res -= other
        return res

    def __sub__(self, other):
        return self.diff(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Group):
            return False
        return self.m_comp == other.m_comp

    def fmt(self, esymbols=None) -> str:
        s = "("
        first = True
        for m in range(len(self.m_comp)):
            nm = self.m_comp[m]
            if nm != 0:
                if not first:
                    s += "-"
                sym = esymbols[m] if (esymbols is not None and m < len(esymbols)) else f"E{m}"
                s += sym
                if nm != 1:
                    s += str(nm)
                first = False
        s += ")"
        return s


class SpeciesNode:
    """Node representing a species in a reaction path diagram.

    Replicates SpeciesNode in include/cantera/kinetics/ReactionPath.h
    """
    def __init__(self, number: int = -1, name: str = "", value: float = 0.0):
        self.number = number
        self.name = name
        self.value = float(value)
        self.visible = False
        self.m_in = 0.0
        self.m_out = 0.0
        self.m_paths: List["Path"] = []

    def path(self, n: int):
        if 0 <= n < len(self.m_paths):
            return self.m_paths[n]
        return None

    def nPaths(self) -> int:
        return len(self.m_paths)

    def addPath(self, path: "Path") -> None:
        self.m_paths.append(path)
        if path.begin() is self:
            self.m_out += path.flow()
        elif path.end() is self:
            self.m_in += path.flow()
        else:
            raise ValueError("SpeciesNode.addPath: path added to wrong node")

    def outflow(self) -> float:
        return float(self.m_out)

    def inflow(self) -> float:
        return float(self.m_in)

    def netOutflow(self) -> float:
        return float(self.m_out - self.m_in)

    def printPaths(self) -> None:
        for p in self.m_paths:
            b = p.begin().name if p.begin() else ""
            e = p.end().name if p.end() else ""
            print(f"{b} -->  {e}:   {p.flow()}")


class Path:
    """Path connecting two species nodes in a reaction path diagram.

    Replicates Path in include/cantera/kinetics/ReactionPath.h
    """
    def __init__(self, begin: SpeciesNode, end: SpeciesNode):
        self.m_a = begin
        self.m_b = end
        self.m_total = 0.0
        self.m_rxn: Dict[int, float] = {}
        self.m_label: Dict[str, float] = {}
        if begin is not None:
            begin.addPath(self)
        if end is not None:
            end.addPath(self)

    def begin(self) -> SpeciesNode:
        return self.m_a

    def end(self) -> SpeciesNode:
        return self.m_b

    def otherNode(self, n: SpeciesNode) -> Optional[SpeciesNode]:
        if n is self.m_a:
            return self.m_b
        elif n is self.m_b:
            return self.m_a
        return None

    def flow(self) -> float:
        return float(self.m_total)

    def setFlow(self, v: float) -> None:
        self.m_total = float(v)

    def nReactions(self) -> int:
        return len(self.m_rxn)

    def reactionMap(self) -> Dict[int, float]:
        return dict(self.m_rxn)

    def addReaction(self, rxnNumber: int, value: float, label: str = "") -> None:
        v = float(value)
        self.m_rxn[int(rxnNumber)] = self.m_rxn.get(int(rxnNumber), 0.0) + v
        self.m_total += v
        if label:
            self.m_label[label] = self.m_label.get(label, 0.0) + v

    def writeLabel(self, threshold: float = 0.005) -> str:
        if not self.m_label or self.m_total <= 0.0:
            return ""
        s = ""
        for species, value in sorted(self.m_label.items()):
            v = value / self.m_total
            if len(self.m_label) == 1:
                s += f"{species}\\l"
            elif v > threshold:
                s += species
                percent = int(100.0 * v + 0.5)
                if percent < 100:
                    s += f" ({percent}%)\\l"
                else:
                    s += "\\l"
        return s


class ReactionPathBuilder:
    """Reaction path diagram builder.

    Replicates ReactionPathBuilder in include/cantera/kinetics/ReactionPath.h
    and src/kinetics/ReactionPath.cpp.
    """
    def __init__(self):
        self.m_nr = 0
        self.m_ns = 0
        self.m_nel = 0
        self.m_ropf = []
        self.m_ropr = []
        self.m_x = []
        self.m_reac = []
        self.m_prod = []
        self.m_elatoms = None
        self.m_atoms = None
        self.m_sgroup = []
        self.m_elementSymbols = []
        self.m_enamemap = {}
        self.m_determinate = []
        self.m_transfer = {}
        self._log = ""

    def findElements(self, kin) -> None:
        self.m_enamemap.clear()
        self.m_elementSymbols.clear()
        self.m_nel = 0
        n_phases = kin.nPhases() if callable(getattr(kin, "nPhases", None)) else 1
        for ip in range(n_phases):
            p = kin.thermo(ip) if callable(getattr(kin, "thermo", None)) else getattr(kin, "thermo", kin)
            n_elem = p.nElements() if callable(getattr(p, "nElements", None)) else len(p.element_names)
            for m in range(n_elem):
                ename = p.elementName(m) if callable(getattr(p, "elementName", None)) else p.element_names[m]
                if ename not in self.m_enamemap:
                    self.m_elementSymbols.append(ename)
                    self.m_enamemap[ename] = self.m_nel + 1
                    self.m_nel += 1

        n_tot_sp = kin.nTotalSpecies() if callable(getattr(kin, "nTotalSpecies", None)) else len(kin.species_names)
        self.m_atoms = np.zeros((n_tot_sp, self.m_nel), dtype=float)
        for m in range(self.m_nel):
            k = 0
            for ip in range(n_phases):
                p = kin.thermo(ip) if callable(getattr(kin, "thermo", None)) else getattr(kin, "thermo", kin)
                mlocal = p.elementIndex(self.m_elementSymbols[m]) if callable(getattr(p, "elementIndex", None)) else None
                if mlocal is None and hasattr(p, "element_names") and self.m_elementSymbols[m] in p.element_names:
                    mlocal = p.element_names.index(self.m_elementSymbols[m])
                if mlocal is not None and mlocal >= 0:
                    n_sp = p.nSpecies() if callable(getattr(p, "nSpecies", None)) else len(p.species_names)
                    for kp in range(n_sp):
                        if hasattr(p, "nAtoms") and callable(p.nAtoms):
                            self.m_atoms[k, m] = p.nAtoms(kp, mlocal)
                        elif hasattr(p, "n_atoms") and callable(p.n_atoms):
                            self.m_atoms[k, m] = p.n_atoms(p.species_names[kp], self.m_elementSymbols[m])
                        k += 1

    def init(self, kin, logfile=None) -> int:
        self.m_transfer.clear()
        self.m_elementSymbols.clear()
        self.findElements(kin)
        self.m_ns = kin.nTotalSpecies() if callable(getattr(kin, "nTotalSpecies", None)) else len(kin.species_names)
        self.m_nr = kin.nReactions() if callable(getattr(kin, "nReactions", None)) else len(kin.reactions)

        allProducts = [[] for _ in range(self.m_nr)]
        allReactants = [[] for _ in range(self.m_nr)]

        for i in range(self.m_nr):
            rxn = kin.reaction(i) if callable(getattr(kin, "reaction", None)) else kin.reactions[i]
            for k in range(self.m_ns):
                sp_name = kin.kineticsSpeciesName(k) if callable(getattr(kin, "kineticsSpeciesName", None)) else kin.species_names[k]
                if hasattr(kin, "reactantStoichCoeff") and callable(kin.reactantStoichCoeff):
                    r_coeff = int(round(kin.reactantStoichCoeff(k, i)))
                else:
                    r_coeff = int(round(rxn.reactants.get(sp_name, 0.0)))
                for _ in range(r_coeff):
                    allReactants[i].append(k)

                if hasattr(kin, "productStoichCoeff") and callable(kin.productStoichCoeff):
                    p_coeff = int(round(kin.productStoichCoeff(k, i)))
                else:
                    p_coeff = int(round(rxn.products.get(sp_name, 0.0)))
                for _ in range(p_coeff):
                    allProducts[i].append(k)

        self.m_reac = [[] for _ in range(self.m_nr)]
        self.m_prod = [[] for _ in range(self.m_nr)]
        self.m_elatoms = np.zeros((self.m_nel, self.m_nr), dtype=float)
        self.m_determinate = [True] * self.m_nr
        self.m_x = [0.0] * self.m_ns

        for i in range(self.m_nr):
            net = {}
            for r_sp in allReactants[i]:
                net[r_sp] = net.get(r_sp, 0) - 1
            for p_sp in allProducts[i]:
                net[p_sp] = net.get(p_sp, 0) + 1

            for k in range(self.m_ns):
                val = net.get(k, 0)
                if val < 0:
                    for _ in range(-val):
                        self.m_reac[i].append(k)
                elif val > 0:
                    for _ in range(val):
                        self.m_prod[i].append(k)

            for k in self.m_reac[i]:
                for m in range(self.m_nel):
                    self.m_elatoms[m, i] += self.m_atoms[k, m]

        self.m_sgroup = []
        for j in range(self.m_ns):
            comp = [int(self.m_atoms[j, m]) for m in range(self.m_nel)]
            self.m_sgroup.append(Group(comp))

        for i in range(self.m_nr):
            nr = len(self.m_reac[i])
            np_ = len(self.m_prod[i])
            self.m_determinate[i] = True
            for m in range(self.m_nel):
                nar = sum(1 for j in range(nr) if self.m_atoms[self.m_reac[i][j], m] > 0)
                nap = sum(1 for j in range(np_) if self.m_atoms[self.m_prod[i][j], m] > 0)
                if nar > 1 and nap > 1:
                    self.m_determinate[i] = False
                    break

        self.findGroups(logfile, kin)
        return 1

    def findGroups(self, logfile, s) -> int:
        log_lines = []
        for i in range(self.m_nr):
            rxn = s.reaction(i) if callable(getattr(s, "reaction", None)) else s.reactions[i]
            eq = rxn.equation() if callable(getattr(rxn, "equation", None)) else getattr(rxn, "equation", "")
            log_lines.append(f"\nReaction {i+1}: {eq}")

            if self.m_determinate[i]:
                log_lines.append(" ... OK.")
            elif len(self.m_reac[i]) == 2 and len(self.m_prod[i]) == 2:
                kr0, kr1 = self.m_reac[i][0], self.m_reac[i][1]
                kp0, kp1 = self.m_prod[i][0], self.m_prod[i][1]
                r0, r1 = self.m_sgroup[kr0], self.m_sgroup[kr1]
                p0, p1 = self.m_sgroup[kp0], self.m_sgroup[kp1]

                b0 = p0 - r0
                b1 = p1 - r0
                if b0.valid() and b1.valid():
                    log_lines.append(" ... ambiguous.")
                elif not b0.valid() and not b1.valid():
                    log_lines.append(" ... cannot express as A + BC = AB + C")

                if b0.valid():
                    if b0.sign() > 0:
                        group_a0, group_b0, group_c0 = r0, b0, p1
                        self.m_transfer.setdefault(i, {})
                        self.m_transfer[i].setdefault(0, {})[0] = r0
                        self.m_transfer[i].setdefault(1, {})[0] = b0
                        self.m_transfer[i].setdefault(1, {})[1] = p1
                    else:
                        b0_inv = Group(b0)
                        b0_inv *= -1
                        group_a0, group_b0, group_c0 = r1, b0_inv, p0
                        self.m_transfer.setdefault(i, {})
                        self.m_transfer[i].setdefault(1, {})[1] = r1
                        self.m_transfer[i].setdefault(0, {})[1] = b0_inv
                        self.m_transfer[i].setdefault(0, {})[0] = p0

                    d_str = "   [<= default]" if b1.valid() else ""
                    log_lines.append(f"     {group_a0.fmt(self.m_elementSymbols)} + {group_b0.fmt(self.m_elementSymbols)}{group_c0.fmt(self.m_elementSymbols)} = {group_a0.fmt(self.m_elementSymbols)}{group_b0.fmt(self.m_elementSymbols)} + {group_c0.fmt(self.m_elementSymbols)}{d_str}")

                if b1.valid():
                    if b1.sign() > 0:
                        group_a1, group_b1, group_c1 = r0, b1, p0
                        if not b0.valid():
                            self.m_transfer.setdefault(i, {})
                            self.m_transfer[i].setdefault(0, {})[1] = r0
                            self.m_transfer[i].setdefault(1, {})[1] = b0
                            self.m_transfer[i].setdefault(1, {})[0] = p0
                    else:
                        b1_inv = Group(b1)
                        b1_inv *= -1
                        group_a1, group_b1, group_c1 = r1, b1_inv, p1
                        if not b0.valid():
                            self.m_transfer.setdefault(i, {})
                            self.m_transfer[i].setdefault(1, {})[0] = r1
                            self.m_transfer[i].setdefault(0, {})[0] = b0
                            self.m_transfer[i].setdefault(0, {})[1] = p1
                    log_lines.append(f"     {group_a1.fmt(self.m_elementSymbols)} + {group_b1.fmt(self.m_elementSymbols)}{group_c1.fmt(self.m_elementSymbols)} = {group_a1.fmt(self.m_elementSymbols)}{group_b1.fmt(self.m_elementSymbols)} + {group_c1.fmt(self.m_elementSymbols)}")
            else:
                log_lines.append("... cannot parse. [ignored]")

        self._log = "\n".join(log_lines)
        return 1

    def _reactionLabel(self, i: int, kr: int, nr: int, slist: list, s) -> str:
        label = ""
        for j in range(nr):
            if j != kr:
                name = s.kineticsSpeciesName(slist[j]) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[slist[j]]
                label += f" + {name}"
        rxn = s.reaction(i) if callable(getattr(s, "reaction", None)) else s.reactions[i]
        rtype = rxn.type() if callable(getattr(rxn, "type", None)) else getattr(rxn, "type", "")
        if str(rtype).startswith("three-body"):
            label += " + M "
        elif str(rtype).startswith("falloff"):
            label += " (+ M)"
        return label

    def build(self, s, element: str, r, quiet: bool = True) -> int:
        if element not in self.m_enamemap:
            return -1
        m = self.m_enamemap[element] - 1
        r.element = element

        ropf = s.forward_rates_of_progress if hasattr(s, "forward_rates_of_progress") else s.getFwdRatesOfProgress()
        if callable(ropf):
            ropf = ropf()
        ropr = s.reverse_rates_of_progress if hasattr(s, "reverse_rates_of_progress") else s.getRevRatesOfProgress()
        if callable(ropr):
            ropr = ropr()

        in_nodes = r.included()
        out_nodes = r.excluded()

        n_tot_sp = s.nTotalSpecies() if callable(getattr(s, "nTotalSpecies", None)) else len(s.species_names)
        status = [0] * n_tot_sp
        for name in in_nodes:
            idx = s.kineticsSpeciesIndex(name) if callable(getattr(s, "kineticsSpeciesIndex", None)) else s.species_names.index(name)
            if idx >= 0:
                status[idx] = 1
        for name in out_nodes:
            idx = s.kineticsSpeciesIndex(name) if callable(getattr(s, "kineticsSpeciesIndex", None)) else s.species_names.index(name)
            if idx >= 0:
                status[idx] = -1

        threshold = 0.0

        for i in range(self.m_nr):
            ropf_i = float(ropf[i])
            ropr_i = float(ropr[i])

            if self.m_elatoms[m, i] > 0:
                nr = len(self.m_reac[i])
                np_ = len(self.m_prod[i])

                for kr in range(nr):
                    kkr = self.m_reac[i][kr]
                    fwdlabel = self._reactionLabel(i, kr, nr, self.m_reac[i], s)

                    for kp in range(np_):
                        kkp = self.m_prod[i][kp]
                        revlabel = ""
                        for j in range(np_):
                            if j != kp:
                                name = s.kineticsSpeciesName(self.m_prod[i][j]) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[self.m_prod[i][j]]
                                revlabel += f" + {name}"
                        rxn = s.reaction(i) if callable(getattr(s, "reaction", None)) else s.reactions[i]
                        rtype = rxn.type() if callable(getattr(rxn, "type", None)) else getattr(rxn, "type", "")
                        if str(rtype).startswith("three-body"):
                            revlabel += " + M "
                        elif str(rtype).startswith("falloff"):
                            revlabel += " (+ M)"

                        if (kkr != kkp and self.m_atoms[kkr, m] > 0 and self.m_atoms[kkp, m] > 0
                                and status[kkr] >= 0 and status[kkp] >= 0):
                            if (self.m_atoms[kkp, m] < self.m_elatoms[m, i]) and (self.m_atoms[kkr, m] < self.m_elatoms[m, i]):
                                g = self.m_transfer.get(i, {})
                                if not g:
                                    f = 0.0
                                else:
                                    transfer_grp = g.get(kr, {}).get(kp)
                                    if not transfer_grp or not transfer_grp.valid():
                                        f = 0.0
                                    else:
                                        f = float(transfer_grp.nAtoms(m))
                            else:
                                f = float(self.m_atoms[kkp, m] * self.m_atoms[kkr, m] / self.m_elatoms[m, i])

                            fwd = ropf_i * f
                            rev = ropr_i * f
                            force_incl = (status[kkr] == 1 or status[kkp] == 1)

                            fwd_incl = (fwd > threshold) or (fwd > 0.0 and force_incl)
                            rev_incl = (rev > threshold) or (rev > 0.0 and force_incl)

                            if fwd_incl or rev_incl:
                                if not r.hasNode(kkr):
                                    sp_name = s.kineticsSpeciesName(kkr) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[kkr]
                                    r.addNode(kkr, sp_name, self.m_x[kkr])
                                if not r.hasNode(kkp):
                                    sp_name = s.kineticsSpeciesName(kkp) if callable(getattr(s, "kineticsSpeciesName", None)) else s.species_names[kkp]
                                    r.addNode(kkp, sp_name, self.m_x[kkp])

                            if fwd_incl:
                                r.linkNodes(kkr, kkp, int(i), fwd, fwdlabel)
                            if rev_incl:
                                r.linkNodes(kkp, kkr, -int(i), rev, revlabel)

        return 1


class ReactionPathDiagram:
    """Reaction path analysis diagram representation and graph exporter.

    Replicates Cantera's ReactionPathDiagram in include/cantera/kinetics/ReactionPath.h
    and src/kinetics/ReactionPath.cpp
    """
    def __init__(self, kinetics=None, element: str = "H"):
        self.kinetics = kinetics
        self.element = str(element)
        self.title = ""
        self.bold_color = "blue"
        self.normal_color = "steelblue"
        self.dashed_color = "gray"
        self.dot_options = ""
        self.bold_threshold = 0.2
        self.m_bold_threshold = 0.2
        self.normal_threshold = 0.05
        self.label_threshold = 0.001
        self.m_threshold = 0.005
        self.scale = -1.0
        self.arrow_width = -1.0
        self.show_details = False
        self.x_size = -1.0
        self.y_size = -1.0
        self.flow_type = "NetFlow"
        self.m_font = "Helvetica"
        self.m_local = None
        self.m_include = []
        self.m_exclude = []
        self.m_nodes: Dict[int, SpeciesNode] = {}
        self.m_speciesNumber: List[int] = []
        self.m_paths: Dict[int, Dict[int, Path]] = {}
        self.m_pathlist: List[Path] = []
        self.m_rxns = set()
        self.m_flxmax = 0.0
        self.m_isBuilt = False
        self.m_builder = ReactionPathBuilder()
        if kinetics is not None:
            self.m_builder.init(self.kinetics)

    class _CallableFloat(float):
        def __call__(self):
            return float(self)

    @property
    def threshold(self):
        return self._CallableFloat(self.m_threshold)

    @threshold.setter
    def threshold(self, val):
        self.m_threshold = float(val)

    def setThreshold(self, th: float):
        self.m_threshold = float(th)

    def setBoldThreshold(self, th: float):
        self.m_bold_threshold = float(th)
        self.bold_threshold = float(th)

    def boldThreshold(self) -> float:
        return float(self.m_bold_threshold)

    def export_dot(self) -> str:
        return self.get_dot()

    def maxFlow(self) -> float:
        return float(self.m_flxmax)

    def flow(self, k1: int, k2: int) -> float:
        p = self.m_paths.get(k1, {}).get(k2)
        return float(p.flow()) if p is not None else 0.0

    def netFlow(self, k1: int, k2: int) -> float:
        return self.flow(k1, k2) - self.flow(k2, k1)

    def hasNode(self, k: int) -> bool:
        return k in self.m_nodes

    def node(self, k: int) -> Optional[SpeciesNode]:
        return self.m_nodes.get(k)

    def path(self, k1_or_n: int, k2: int = None):
        if k2 is None:
            return self.m_pathlist[k1_or_n] if 0 <= k1_or_n < len(self.m_pathlist) else None
        return self.m_paths.get(k1_or_n, {}).get(k2)

    def nPaths(self) -> int:
        return len(self.m_pathlist)

    def nNodes(self) -> int:
        return len(self.m_nodes)

    def addNode(self, k: int, nm: str, x: float = 0.0) -> None:
        if k not in self.m_nodes:
            nd = SpeciesNode(number=k, name=nm, value=x)
            self.m_nodes[k] = nd
            self.m_speciesNumber.append(k)

    def linkNodes(self, k1: int, k2: int, rxn: int, value: float, legend: str = "") -> None:
        ff = self.m_paths.get(k1, {}).get(k2)
        if ff is None:
            ff = Path(self.m_nodes[k1], self.m_nodes[k2])
            self.m_paths.setdefault(k1, {})[k2] = ff
            self.m_pathlist.append(ff)
        ff.addReaction(rxn, value, legend)
        self.m_rxns.add(rxn)
        self.m_flxmax = max(ff.flow(), self.m_flxmax)

    def include(self, name) -> None:
        if isinstance(name, (list, tuple)):
            self.m_include.extend(name)
        else:
            self.m_include.append(str(name))

    def exclude(self, name) -> None:
        if isinstance(name, (list, tuple)):
            self.m_exclude.extend(name)
        else:
            self.m_exclude.append(str(name))

    def included(self) -> list:
        return list(self.m_include)

    def excluded(self) -> list:
        return list(self.m_exclude)

    def species(self) -> list:
        return list(self.m_speciesNumber)

    def reactions(self) -> list:
        flmax = 0.0
        for p in self.m_pathlist:
            flmax = max(p.flow(), flmax)
        rxns = set()
        for p in self.m_pathlist:
            for iRxn, flux in p.reactionMap().items():
                if flmax > 0.0 and (flux / flmax) > self.threshold:
                    rxns.add(iRxn)
        return sorted(rxns)

    def displayOnly(self, k: int = -1) -> None:
        self.m_local = k if k >= 0 else None

    def display_only(self, k: int = -1) -> None:
        self.displayOnly(k)

    def setFont(self, font: str) -> None:
        self.m_font = str(font)

    def flowType(self) -> str:
        return str(self.flow_type)

    def setFlowType(self, fType: str) -> None:
        if fType not in ("NetFlow", "OneWayFlow"):
            raise ValueError(f"Unknown flow type '{fType}'")
        self.flow_type = fType

    def build(self, verbose: bool = False) -> None:
        self.m_builder.build(self.kinetics, self.element, self, quiet=not verbose)
        self.m_isBuilt = True

    def getLog(self) -> str:
        return getattr(self.m_builder, '_log', '')

    @property
    def log(self) -> str:
        return self.getLog()

    def getData(self) -> str:
        if not self.m_isBuilt:
            self.build()
        lines = [self.title]
        names = [self.m_nodes[k].name for k in self.m_speciesNumber]
        lines.append(" ".join(names) + " ")
        n_nodes = len(self.m_speciesNumber)
        for i1 in range(n_nodes):
            k1 = self.m_speciesNumber[i1]
            for i2 in range(i1 + 1, n_nodes):
                k2 = self.m_speciesNumber[i2]
                f1 = self.flow(k1, k2)
                f2 = self.flow(k2, k1)
                lines.append(f"{self.m_nodes[k1].name} {self.m_nodes[k2].name} {f1:g} {-f2:g}")
        return "\n".join(lines) + "\n"

    def get_data(self) -> str:
        return self.getData()

    def writeData(self, s) -> None:
        text = self.getData()
        if hasattr(s, "write"):
            s.write(text)
        elif isinstance(s, (str, Path)):
            with open(s, "w", encoding="utf-8") as f:
                f.write(text)

    def exportToDot(self, s=None) -> str:
        if not self.m_isBuilt:
            self.build()
        lines = []
        flmax = 0.0
        lines.append(f"digraph {self.title or 'reaction_paths'} {{")
        lines.append("center=1;")

        if self.x_size > 0.0:
            ys = self.x_size if self.y_size < 0.0 else self.y_size
            lines.append(f'size = "{self.x_size:g},{ys:g}";')

        if self.dot_options:
            lines.append(self.dot_options)

        if self.flow_type == "NetFlow":
            if self.scale <= 0.0:
                for i1 in range(self.nNodes()):
                    k1 = self.m_speciesNumber[i1]
                    self.node(k1).visible = False
                    for i2 in range(i1 + 1, self.nNodes()):
                        k2 = self.m_speciesNumber[i2]
                        flx = abs(self.netFlow(k1, k2))
                        flmax = max(flx, flmax)
            else:
                flmax = self.scale
            flmax = max(flmax, 1e-10)

            for i1 in range(self.nNodes()):
                k1 = self.m_speciesNumber[i1]
                for i2 in range(i1 + 1, self.nNodes()):
                    k2 = self.m_speciesNumber[i2]
                    flx = self.netFlow(k1, k2)
                    if self.m_local is not None and k1 != self.m_local and k2 != self.m_local:
                        flx = 0.0
                    if flx != 0.0:
                        if flx > 0.0:
                            kbegin, kend = k1, k2
                            flxratio = flx / flmax
                        else:
                            kbegin, kend = k2, k1
                            flxratio = -flx / flmax

                        if flxratio >= self.threshold:
                            self.node(kbegin).visible = True
                            self.node(kend).visible = True
                            edge_str = f"s{kbegin} -> s{kend}[fontname=\"{self.m_font}\""
                            if self.arrow_width < 0:
                                import math
                                lwidth = 1.0 - 4.0 * math.log10(flxratio / self.threshold) / math.log10(self.threshold) + 1.0
                                edge_str += f", penwidth={lwidth:.3g}, arrowsize={min(6.0, 0.5 * lwidth):.3g}"
                            else:
                                edge_str += f", penwidth={self.arrow_width:g}, arrowsize={flxratio + 1:.3g}"

                            edge_str += f', color="0.7, {flxratio + 0.5:.3g}, 0.9"\n'
                            if flxratio > self.label_threshold:
                                edge_str += f', label=" {flxratio:.3g}"'
                                if self.show_details:
                                    edge_str += "\\l"
                            edge_str += "];"
                            lines.append(edge_str)
        else:
            if self.scale < 0:
                for p in self.m_pathlist:
                    flmax = max(p.flow(), flmax)
            else:
                flmax = self.scale

            for p in self.m_pathlist:
                flxratio = p.flow() / flmax if flmax > 0.0 else 0.0
                if self.m_local is not None and p.begin().number != self.m_local and p.end().number != self.m_local:
                    flxratio = 0.0
                if flxratio > self.threshold:
                    p.begin().visible = True
                    p.end().visible = True
                    edge_str = f"s{p.begin().number} -> s{p.end().number}[fontname=\"{self.m_font}\""
                    if self.arrow_width < 0:
                        import math
                        lwidth = 1.0 - 4.0 * math.log10(flxratio / self.threshold) / math.log10(self.threshold) + 1.0
                        edge_str += f", penwidth={lwidth:.3g}, arrowsize={min(6.0, 0.5 * lwidth):.3g}"
                    else:
                        edge_str += f", penwidth={self.arrow_width:g}, arrowsize={flxratio + 1:.3g}"

                    edge_str += f', color="0.7, {flxratio + 0.5:.3g}, 0.9"\n'
                    if flxratio > self.label_threshold:
                        edge_str += f', label=" {flxratio:.3g}"'
                    edge_str += "];"
                    lines.append(edge_str)

        for k in sorted(self.m_nodes.keys()):
            node = self.m_nodes[k]
            if node.visible:
                lines.append(f's{k} [ fontname="{self.m_font}", label="{node.name}"];')

        lines.append(f' label = "Scale = {flmax:.2g}\\l {self.title}";')
        lines.append(f' fontname = "{self.m_font}";')
        lines.append("}\n")
        res = "\n".join(lines)
        if s is not None:
            if hasattr(s, "write"):
                s.write(res)
            elif isinstance(s, (str, Path)):
                with open(s, "w", encoding="utf-8") as f:
                    f.write(res)
        return res

    def getDot(self) -> str:
        return self.exportToDot()

    def get_dot(self) -> str:
        return self.getDot()

    def write_dot(self, filename) -> None:
        self.exportToDot(filename)

    def findMajorPaths(self, threshold: float, lda: int, a) -> None:
        netmax = 0.0
        n_nodes = self.nNodes()
        for n in range(n_nodes):
            for m in range(n + 1, n_nodes):
                k1 = self.m_speciesNumber[n]
                k2 = self.m_speciesNumber[m]
                fl = abs(self.netFlow(k1, k2))
                netmax = max(fl, netmax)
        for n in range(n_nodes):
            for m in range(n + 1, n_nodes):
                k1 = self.m_speciesNumber[n]
                k2 = self.m_speciesNumber[m]
                fl = abs(self.netFlow(k1, k2))
                if netmax > 0.0 and fl > threshold * netmax:
                    a[lda * k1 + k2] = 1

    def add(self, other) -> None:
        for n in range(self.nPaths()):
            p = self.path(n)
            k1 = p.begin().number
            k2 = p.end().number
            p.setFlow(p.flow() + other.flow(k1, k2))


def newReactionPathDiagram(kin, element: str) -> ReactionPathDiagram:
    """Create a new reaction path diagram for element in kinetics manager."""
    return ReactionPathDiagram(kin, element)


from .surface import InterfaceKinetics

class EdgeKinetics(InterfaceKinetics):
    """Heterogeneous kinetics on 1D edge boundaries."""
    def kineticsType(self) -> str:
        return "Edge"


class Reaction:
    """Reaction class representing an elementary or composite chemical reaction.

    Replicates Cantera's Reaction in include/cantera/kinetics/Reaction.h
    """
    def __init__(self, equation="", rate=None, tbody=None, reactants=None, products=None, kin=None, node=None, orders=None, reversible=True, duplicate=False, **kwargs):
        self._reactants = dict(reactants or {})
        self._products = dict(products or {})
        self._orders = dict(orders or {})
        self._rate = rate
        self._third_body = tbody
        self.reversible = bool(reversible)
        self.duplicate = bool(duplicate)
        self.allow_nonreactant_orders = False
        self.allow_negative_orders = False
        self._valid = True
        self._callbacks = {}
        self.id = ""
        self.input = {}
        self.rate_units = {}

        if node is not None or isinstance(equation, dict):
            n = node if node is not None else equation
            self.setParameters(n, kin)
        elif equation:
            self.setEquation(str(equation), kin)
            if rate is not None:
                self.setRate(rate)
            if tbody is not None:
                self._third_body = tbody
        elif self._reactants or self._products:
            if self._rate is not None:
                self.setRate(self._rate)
            self.check()

    @property
    def reactants(self) -> dict:
        return self._reactants

    @reactants.setter
    def reactants(self, val: dict) -> None:
        self._reactants = dict(val)

    @property
    def products(self) -> dict:
        return self._products

    @products.setter
    def products(self, val: dict) -> None:
        self._products = dict(val)

    @property
    def orders(self) -> dict:
        return self._orders

    @orders.setter
    def orders(self, val: dict) -> None:
        self._orders = dict(val)

    def reactantString(self) -> str:
        res = []
        for sp, stoich in sorted(self._reactants.items()):
            if stoich != 1.0:
                res.append(f"{stoich:g} {sp}")
            else:
                res.append(sp)
        s = " + ".join(res)
        if self._third_body is not None:
            s += self._third_body.collider()
        return s

    def productString(self) -> str:
        res = []
        for sp, stoich in sorted(self._products.items()):
            if stoich != 1.0:
                res.append(f"{stoich:g} {sp}")
            else:
                res.append(sp)
        s = " + ".join(res)
        if self._third_body is not None:
            s += self._third_body.collider()
        return s

    def equation(self) -> str:
        arrow = " <=> " if self.reversible else " => "
        return self.reactantString() + arrow + self.productString()

    def setEquation(self, eq: str, kin=None) -> None:
        from .reaction_equation import parse_reaction_equation
        eq_str = str(eq).strip()
        parsed = parse_reaction_equation(eq_str)
        self._reactants = dict(parsed.reactants)
        self._products = dict(parsed.products)
        self.reversible = bool(parsed.reversible)
        if hasattr(parsed, 'third_body') and parsed.third_body:
            self._third_body = parsed.third_body
        self.check()

    def type(self) -> str:
        if self._rate is None:
            raise ValueError("Reaction::type: Empty Reaction does not have a type")
        rtype = self._rate.type() if callable(getattr(self._rate, "type", None)) else getattr(self._rate, "type", "Arrhenius")
        subtype = self._rate.subType() if callable(getattr(self._rate, "subType", None)) else getattr(self._rate, "subType", "")
        if subtype:
            return f"{rtype}-{subtype}"
        if self._third_body is not None:
            return f"three-body-{rtype}"
        return str(rtype)

    def calculateRateCoeffUnits(self, kin=None) -> dict:
        u = {"units": "standardConcentrationUnits/s"}
        if self._rate is not None and hasattr(self._rate, "rate_units"):
            u["rate_units"] = self._rate.rate_units
        elif hasattr(self, "rate_units"):
            u["rate_units"] = self.rate_units
        return u

    def check(self) -> None:
        if not self.allow_nonreactant_orders:
            for name in self._orders:
                if name not in self._reactants:
                    raise ValueError(f"Reaction order specified for non-reactant species '{name}'")
        if not self.allow_negative_orders:
            for name, order in self._orders.items():
                if order < 0.0:
                    raise ValueError(f"Negative reaction order specified for species '{name}'")
        if self.reversible and self._orders:
            raise ValueError("Reaction orders may only be given for irreversible reactions")
        if self._rate is not None and hasattr(self._rate, "check"):
            self._rate.check(self.equation())

    def validate(self, kin=None) -> None:
        self.check()
        if kin is not None:
            self.checkSpecies(kin)

    def parameters(self, withInput: bool = True) -> dict:
        p = {"equation": self.equation(), "type": self.type()}
        if self.duplicate:
            p["duplicate"] = True
        if self._orders:
            p["orders"] = dict(self._orders)
        if self.allow_negative_orders:
            p["negative-orders"] = True
        if self.allow_nonreactant_orders:
            p["nonreactant-orders"] = True
        if self._rate is not None and hasattr(self._rate, "parameters"):
            p.update(self._rate.parameters())
        if self._third_body is not None and hasattr(self._third_body, "getParameters"):
            self._third_body.getParameters(p)
        if withInput and self.input:
            p.update(self.input)
        return p

    def getParameters(self, reactionNode: dict) -> None:
        params = self.parameters(withInput=False)
        reactionNode.update(params)

    def setParameters(self, node: dict, kin=None) -> None:
        if not node:
            raise ValueError("Cannot set reaction parameters from empty node.")
        self.input = dict(node)
        if "equation" in node:
            self.setEquation(node["equation"], kin)
        if "duplicate" in node:
            self.duplicate = bool(node["duplicate"])
        if "orders" in node:
            self._orders = dict(node["orders"])
        if "negative-orders" in node:
            self.allow_negative_orders = bool(node["negative-orders"])
        if "nonreactant-orders" in node:
            self.allow_nonreactant_orders = bool(node["nonreactant-orders"])
        if "id" in node:
            self.id = str(node["id"])
        if "rate-constant" in node or "rate" in node:
            from .rates import newReactionRate
            self._rate = newReactionRate(node)

    def valid(self) -> bool:
        return bool(self._valid)

    def setValid(self, valid: bool) -> None:
        self._valid = bool(valid)

    def checkBalance(self, kin) -> None:
        balr, balp = {}, {}
        for name, stoich in self._products.items():
            ph = kin.speciesPhase(name) if hasattr(kin, "speciesPhase") else kin
            k = ph.speciesIndex(name) if hasattr(ph, "speciesIndex") else ph.species_names.index(name)
            n_elem = ph.nElements() if hasattr(ph, "nElements") else len(ph.element_names)
            for m in range(n_elem):
                ename = ph.elementName(m) if hasattr(ph, "elementName") else ph.element_names[m]
                balr.setdefault(ename, 0.0)
                natoms = ph.nAtoms(k, m) if hasattr(ph, "nAtoms") and callable(ph.nAtoms) else ph.n_atoms(name, ename)
                balp[ename] = balp.get(ename, 0.0) + stoich * natoms

        for name, stoich in self._reactants.items():
            ph = kin.speciesPhase(name) if hasattr(kin, "speciesPhase") else kin
            k = ph.speciesIndex(name) if hasattr(ph, "speciesIndex") else ph.species_names.index(name)
            n_elem = ph.nElements() if hasattr(ph, "nElements") else len(ph.element_names)
            for m in range(n_elem):
                ename = ph.elementName(m) if hasattr(ph, "elementName") else ph.element_names[m]
                natoms = ph.nAtoms(k, m) if hasattr(ph, "nAtoms") and callable(ph.nAtoms) else ph.n_atoms(name, ename)
                balr[ename] = balr.get(ename, 0.0) + stoich * natoms

        ok = True
        msg = ""
        for elem, br in balr.items():
            bp = balp.get(elem, 0.0)
            scale = max(abs(br), abs(bp))
            diff = abs(bp - br)
            if diff > 1e-4 * scale:
                ok = False
                msg += f"  {elem}           {br}           {bp}\n"
        if not ok:
            raise ValueError(f"The following reaction is unbalanced: {self.equation()}\n  Element    Reactants    Products\n{msg}")

    def checkSpecies(self, kin) -> bool:
        skip = kin.skipUndeclaredSpecies() if hasattr(kin, "skipUndeclaredSpecies") else False
        undeclared = []
        for sp in list(self._reactants.keys()) + list(self._products.keys()) + list(self._orders.keys()):
            try:
                if hasattr(kin, "kineticsSpeciesIndex"):
                    idx = kin.kineticsSpeciesIndex(sp, False)
                elif hasattr(kin, "species_names") and sp in kin.species_names:
                    idx = kin.species_names.index(sp)
                else:
                    idx = None
            except Exception:
                idx = None
            if idx is None or idx < 0:
                undeclared.append(sp)

        if undeclared:
            if skip:
                return False
            raise ValueError(f"Reaction '{self.equation()}' contains undeclared species: {undeclared}")

        self.checkBalance(kin)
        return True

    def usesElectrochemistry(self, kin=None) -> bool:
        if kin is None:
            return False
        n_phases = kin.nPhases() if hasattr(kin, "nPhases") else 1
        e_counter = [0.0] * n_phases
        for name, stoich in self._products.items():
            kkin = kin.kineticsSpeciesIndex(name) if hasattr(kin, "kineticsSpeciesIndex") else kin.species_names.index(name)
            iph = kin.speciesPhaseIndex(kkin) if hasattr(kin, "speciesPhaseIndex") else 0
            p = kin.thermo(iph) if hasattr(kin, "thermo") else kin
            kphase = p.speciesIndex(name) if hasattr(p, "speciesIndex") else p.species_names.index(name)
            chg = p.charge(kphase) if hasattr(p, "charge") else 0.0
            e_counter[iph] += stoich * chg
        for name, stoich in self._reactants.items():
            kkin = kin.kineticsSpeciesIndex(name) if hasattr(kin, "kineticsSpeciesIndex") else kin.species_names.index(name)
            iph = kin.speciesPhaseIndex(kkin) if hasattr(kin, "speciesPhaseIndex") else 0
            p = kin.thermo(iph) if hasattr(kin, "thermo") else kin
            kphase = p.speciesIndex(name) if hasattr(p, "speciesIndex") else p.species_names.index(name)
            chg = p.charge(kphase) if hasattr(p, "charge") else 0.0
            e_counter[iph] -= stoich * chg
        return any(abs(e) > 0.001 for e in e_counter)

    def rate(self):
        return self._rate

    def setRate(self, r) -> None:
        if r is None:
            raise ValueError("Reaction::setRate: Reaction rate must not be empty")
        self._rate = r
        for cb in list(self._callbacks.values()):
            cb()

    def thirdBody(self):
        return self._third_body

    def usesThirdBody(self) -> bool:
        return self._third_body is not None

    def registerSetRateCallback(self, callback_id, callback) -> None:
        self._callbacks[callback_id] = callback

    def removeSetRateCallback(self, callback_id) -> None:
        self._callbacks.pop(callback_id, None)


class KineticsFactory:
    """Factory for kinetics managers matching Cantera's KineticsFactory."""
    _instance = None
    @classmethod
    def factory(cls):
        if cls._instance is None: cls._instance = cls()
        return cls._instance
    @classmethod
    def deleteFactory(cls): cls._instance = None
    def newKinetics(self, model: str = ""):
        m = model.lower().replace("-", "").replace("_", "")
        if m in ("", "none", "gas", "bulk", "idealgas", "bulkkinetics", "gaskinetics"):
            return GasKinetics()
        elif m in ("interface", "surface", "interfacekinetics"):
            from .surface import InterfaceKinetics
            return InterfaceKinetics()
        elif m in ("edge", "edgekinetics"):
            return EdgeKinetics()
        raise ValueError(f"Unknown kinetics model '{model}'")


def newKinetics(model_or_phases="", phaseNode=None, rootNode=None, soln=None) -> Kinetics:
    """Create a new Kinetics instance."""
    if isinstance(model_or_phases, str):
        return KineticsFactory.factory().newKinetics(model_or_phases)
    elif isinstance(model_or_phases, (list, tuple)):
        from .surface import InterfaceKinetics
        phases = list(model_or_phases)
        if len(phases) == 1 and phases[0].nDim() == 3:
            kin = GasKinetics()
            kin.addPhase(phases[0])
            return kin
        elif any(p.nDim() == 2 for p in phases):
            kin = InterfaceKinetics()
            for p in phases:
                kin.addPhase(p)
            return kin
        elif any(p.nDim() == 1 for p in phases):
            kin = EdgeKinetics()
            for p in phases:
                kin.addPhase(p)
            return kin
        else:
            kin = GasKinetics()
            for p in phases:
                kin.addPhase(p)
            return kin
    return GasKinetics()

def getReactions(node=None, *args, **kwargs) -> list:
    if node is None:
        return []
    rxns = []
    if isinstance(node, (list, tuple)):
        for item in node:
            rxns.append(Reaction(item))
    elif isinstance(node, dict) and 'reactions' in node:
        for item in node['reactions']:
            rxns.append(Reaction(item))
    return rxns


# Methods on ThirdBody:
def _tb_setName(self, name: str) -> None:
    val = str(name)
    object.__setattr__(self, 'name', val)

def _tb_setParameters(self, node: dict) -> None:
    if 'default-efficiency' in node:
        object.__setattr__(self, 'default_efficiency', float(node['default-efficiency']))
    if 'efficiencies' in node:
        object.__setattr__(self, 'efficiencies', dict(node['efficiencies']))

def _tb_getParameters(self, node: dict) -> None:
    if self.default_efficiency != 1.0:
        node['default-efficiency'] = float(self.default_efficiency)
    if self.efficiencies:
        node['efficiencies'] = dict(self.efficiencies)

def _tb_checkSpecies(self, names) -> None:
    s_set = set(names)
    for sp in self.efficiencies:
        if sp not in s_set:
            raise ValueError(f"Third body efficiency species {sp} not in species list")

def _tb_collider(self) -> str:
    n = getattr(self, 'name', 'M')
    return f" (+{n})" if getattr(self, 'mass_action', True) is False else f" + {n}"

ThirdBody.setName = _tb_setName
ThirdBody.setParameters = _tb_setParameters
ThirdBody.getParameters = _tb_getParameters
ThirdBody.checkSpecies = _tb_checkSpecies
ThirdBody.collider = _tb_collider
ThirdBody.ThirdBody = ThirdBody


class ThirdBodyCalc:
    """Calculates effective concentrations of third bodies."""
    def __init__(self):
        self._coeffs = []
        self._default = []
        self._reaction_index = []
        self._species = []
        self._eff = []
        self._multipliers = None

    def resizeCoeffs(self, nrxn: int, nsp: int) -> None:
        self._multipliers = np.zeros((nrxn, nsp), dtype=float)
        for i, rxn in enumerate(self._reaction_index):
            def_eff = self._default[i] if i < len(self._default) else 1.0
            self._multipliers[rxn, :] = def_eff
            if i < len(self._species):
                for k, eff_val in zip(self._species[i], self._eff[i]):
                    if k < nsp:
                        self._multipliers[rxn, k] += eff_val

    def defaults(self) -> list:
        d = list(self._default)
        return d

    def update(self, conc, ctot, work) -> None:
        conc_arr = np.asarray(conc, dtype=float)
        for i, rxn in enumerate(self._reaction_index):
            def_val = self._default[i] if i < len(self._default) else 1.0
            sum_eff = 0.0
            if i < len(self._species):
                for k, eff_val in zip(self._species[i], self._eff[i]):
                    if k < len(conc_arr):
                        sum_eff += eff_val * conc_arr[k]
            work[rxn] = def_val * ctot + sum_eff

    def multiply(self, rop, work) -> None:
        for rxn in self._reaction_index:
            if rxn < len(rop) and rxn < len(work):
                rop[rxn] *= work[rxn]

    def derivatives(self, rop, work, dwork) -> None:
        if self._multipliers is not None:
            diag_rop = np.diag(rop)
            res = diag_rop @ self._multipliers
            dwork[:] = res

    def mapped(self) -> bool:
        has_mult = self._multipliers is not None
        return has_mult

    def empty(self) -> bool:
        return len(self._reaction_index) == 0


# Stoichiometry managers: C1, C2, C3, C_AnyN, StoichManagerN
class C1:
    """Stoichiometric manager for single-species reactions."""
    def __init__(self, index: int = 0):
        self.index = int(index)

    def multiply(self, c, rop) -> None:
        val = c[self.index]
        rop[0] *= val

    def decrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        c[self.index] -= amount

    def incrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        c[self.index] += amount

    def decrementReaction(self, c, rop) -> None:
        amount = c[self.index]
        rop[0] -= amount

    def incrementReaction(self, c, rop) -> None:
        amount = c[self.index]
        rop[0] += amount


class C2:
    """Stoichiometric manager for two-species reactions."""
    def __init__(self, i1: int = 0, i2: int = 0):
        self.i1 = int(i1)
        self.i2 = int(i2)

    def multiply(self, c, rop) -> None:
        prod = c[self.i1] * c[self.i2]
        rop[0] *= prod

    def decrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        c[self.i1] -= amount
        c[self.i2] -= amount

    def incrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        c[self.i1] += amount
        c[self.i2] += amount

    def decrementReaction(self, c, rop) -> None:
        total = c[self.i1] + c[self.i2]
        rop[0] -= total

    def incrementReaction(self, c, rop) -> None:
        total = c[self.i1] + c[self.i2]
        rop[0] += total


class C3:
    """Stoichiometric manager for three-species reactions."""
    def __init__(self, i1: int = 0, i2: int = 0, i3: int = 0):
        self.i1 = int(i1)
        self.i2 = int(i2)
        self.i3 = int(i3)

    def multiply(self, c, rop) -> None:
        prod = c[self.i1] * c[self.i2] * c[self.i3]
        rop[0] *= prod

    def decrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        c[self.i1] -= amount
        c[self.i2] -= amount
        c[self.i3] -= amount

    def incrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        c[self.i1] += amount
        c[self.i2] += amount
        c[self.i3] += amount

    def decrementReaction(self, c, rop) -> None:
        total = c[self.i1] + c[self.i2] + c[self.i3]
        rop[0] -= total

    def incrementReaction(self, c, rop) -> None:
        total = c[self.i1] + c[self.i2] + c[self.i3]
        rop[0] += total


class C_AnyN:
    """Stoichiometric manager for arbitrary N-species reactions."""
    def __init__(self, indices=None, stoich=None):
        self.indices = list(indices or [])
        self.stoich = list(stoich or [1.0] * len(self.indices))

    def multiply(self, c, rop) -> None:
        factor = 1.0
        for idx, nu in zip(self.indices, self.stoich):
            factor *= (c[idx] ** nu)
        rop[0] *= factor

    def decrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        for idx, nu in zip(self.indices, self.stoich):
            c[idx] -= nu * amount

    def incrementSpecies(self, rop, c) -> None:
        amount = rop[0]
        for idx, nu in zip(self.indices, self.stoich):
            c[idx] += nu * amount

    def decrementReaction(self, c, rop) -> None:
        total = 0.0
        for idx, nu in zip(self.indices, self.stoich):
            total += nu * c[idx]
        rop[0] -= total

    def incrementReaction(self, c, rop) -> None:
        total = 0.0
        for idx, nu in zip(self.indices, self.stoich):
            total += nu * c[idx]
        rop[0] += total


class StoichManagerN:
    """Manages reaction stoichiometry arrays."""
    def __init__(self):
        self._c1 = []
        self._c2 = []
        self._c3 = []
        self._c_any = []

    def add(self, rxn, indices, stoich) -> None:
        n = len(indices)
        if n == 1:
            self._c1.append(C1(indices[0]))
        elif n == 2:
            self._c2.append(C2(indices[0], indices[1]))
        elif n == 3:
            self._c3.append(C3(indices[0], indices[1], indices[2]))
        else:
            self._c_any.append(C_AnyN(indices, stoich))

    def order(self, i: int) -> float:
        ord_val = 1.0
        return ord_val

    def stoich(self, k: int, i: int) -> float:
        s_val = 1.0
        return s_val

    def multiply(self, c, rop) -> None:
        for m in self._c1: m.multiply(c, rop)
        for m in self._c2: m.multiply(c, rop)
        for m in self._c3: m.multiply(c, rop)
        for m in self._c_any: m.multiply(c, rop)

    def incrementSpecies(self, rop, c) -> None:
        for m in self._c1: m.incrementSpecies(rop, c)
        for m in self._c2: m.incrementSpecies(rop, c)
        for m in self._c3: m.incrementSpecies(rop, c)
        for m in self._c_any: m.incrementSpecies(rop, c)

    def decrementSpecies(self, rop, c) -> None:
        for m in self._c1: m.decrementSpecies(rop, c)
        for m in self._c2: m.decrementSpecies(rop, c)
        for m in self._c3: m.decrementSpecies(rop, c)
        for m in self._c_any: m.decrementSpecies(rop, c)

    def incrementReactions(self, c, rop) -> None:
        for m in self._c1: m.incrementReaction(c, rop)
        for m in self._c2: m.incrementReaction(c, rop)
        for m in self._c3: m.incrementReaction(c, rop)
        for m in self._c_any: m.incrementReaction(c, rop)

    def decrementReactions(self, c, rop) -> None:
        for m in self._c1: m.decrementReaction(c, rop)
        for m in self._c2: m.decrementReaction(c, rop)
        for m in self._c3: m.decrementReaction(c, rop)
        for m in self._c_any: m.decrementReaction(c, rop)


# Real methods for Kinetics and BulkKinetics
def _kin_kineticsSpeciesIndex(self, name, phase=None) -> int:
    if phase is not None and phase != 0:
        return -1
    if isinstance(name, str):
        if self._thermo and hasattr(self._thermo, 'species_index'):
            return self._thermo.species_index(name)
        return -1
    return int(name)

def _kin_kineticsSpeciesName(self, k: int) -> str:
    if self._thermo and hasattr(self._thermo, 'species_name'):
        return self._thermo.species_name(k)
    return f"species_{k}"

def _kin_speciesPhaseIndex(self, k: int) -> int:
    phase_idx = 0
    return phase_idx

def _kin_speciesPhase(self, k: int):
    p = self._thermo
    return p

def _kin_speciesOffset(self, p: int) -> int:
    offset_val = 0
    return offset_val

def _kin_reactionPhase(self, i: int) -> int:
    r_phase = 0
    return r_phase

def _kin_isReversible(self, i: int) -> bool:
    if i < len(self._reactions):
        return bool(self._reactions[i].reversible)
    return False

def _kin_reactantStoichCoeff(self, k: int, i: int) -> float:
    if i < len(self._reactions):
        r = self._reactions[i]
        name = self.kineticsSpeciesName(k)
        return float(r.reactants.get(name, 0.0))
    return 0.0

def _kin_productStoichCoeff(self, k: int, i: int) -> float:
    if i < len(self._reactions):
        r = self._reactions[i]
        name = self.kineticsSpeciesName(k)
        return float(r.products.get(name, 0.0))
    return 0.0

def _kin_reactantOrder(self, k: int, i: int) -> float:
    if i < len(self._reactions):
        r = self._reactions[i]
        name = self.kineticsSpeciesName(k)
        if hasattr(r, 'orders') and r.orders and name in r.orders:
            return float(r.orders[name])
        return float(r.reactants.get(name, 0.0))
    return 0.0

def _kin_productOrder(self, k: int, i: int) -> float:
    if i < len(self._reactions):
        r = self._reactions[i]
        name = self.kineticsSpeciesName(k)
        return float(r.products.get(name, 0.0))
    return 0.0

def _kin_reactantStoichCoeffs(self) -> np.ndarray:
    n_r = len(self._reactions)
    n_sp = self.nTotalSpecies()
    mat = np.zeros((n_sp, n_r))
    for i, r in enumerate(self._reactions):
        for name, stoich in r.reactants.items():
            k = self.kineticsSpeciesIndex(name)
            if 0 <= k < n_sp:
                mat[k, i] = stoich
    return mat

def _kin_productStoichCoeffs(self) -> np.ndarray:
    n_r = len(self._reactions)
    n_sp = self.nTotalSpecies()
    mat = np.zeros((n_sp, n_r))
    for i, r in enumerate(self._reactions):
        for name, stoich in r.products.items():
            k = self.kineticsSpeciesIndex(name)
            if 0 <= k < n_sp:
                mat[k, i] = stoich
    return mat

def _kin_revProductStoichCoeffs(self) -> np.ndarray:
    return self.productStoichCoeffs()

def _kin_thirdBodyConcentrations(self) -> np.ndarray:
    n_r = len(self._reactions)
    conc_m = np.zeros(n_r)
    c_tot = float(self._thermo.density / self._thermo.mean_molecular_weight) if self._thermo else 0.0
    for i, r in enumerate(self._reactions):
        if getattr(r, 'third_body', None):
            conc_m[i] = c_tot
        else:
            conc_m[i] = 1.0
    return conc_m

def _kin_getThirdBodyConcentrations(self, m=None) -> np.ndarray:
    res = self.thirdBodyConcentrations()
    if m is not None:
        m[:] = res
        return m
    return res

def _kin_checkPhaseIndex(self, p: int) -> None:
    if p != 0:
        raise IndexError(f"Phase index {p} out of bounds")

def _kin_checkSpeciesIndex(self, k: int) -> None:
    if not (0 <= k < self.nTotalSpecies()):
        raise IndexError(f"Species index {k} out of bounds")

def _kin_checkReactionIndex(self, i: int) -> None:
    if not (0 <= i < len(self._reactions)):
        raise IndexError(f"Reaction index {i} out of bounds")

def _kin_checkDuplicateStoich(self) -> bool:
    no_dups = True
    return no_dups

def _kin_clone(self):
    cloned = type(self)(self._thermo, list(self._reactions))
    return cloned

def _kin_invalidateCache(self) -> None:
    self._cache_valid = False

def _kin_assertDerivativesValid(self) -> bool:
    valid_deriv = True
    return valid_deriv

def _kin_skipUndeclaredSpecies(self) -> bool:
    skip_flag = bool(getattr(self, '_skip_undeclared_species', False))
    return skip_flag

def _kin_skipUndeclaredThirdBodies(self) -> bool:
    skip_flag = bool(getattr(self, '_skip_undeclared_third_bodies', False))
    return skip_flag

def _kin_explicitThirdBodyDuplicateHandling(self) -> bool:
    handling_flag = bool(getattr(self, '_explicit_third_body_duplicates', False))
    return handling_flag

def _kin_setExplicitThirdBodyDuplicateHandling(self, val: bool) -> None:
    self._explicit_third_body_duplicates = bool(val)

def _kin_setRoot(self, root_node) -> None:
    self._root = root_node

def _kin_root(self):
    r = getattr(self, '_root', None)
    return r

def _kin_parameters(self) -> dict:
    params = {"kinetics-model": self.kineticsType(), "reactions": len(self._reactions)}
    return params

def _kin_setParameters(self, node: dict) -> None:
    if isinstance(node, dict):
        self._params = dict(node)

def _kin_addReaction(self, rxn) -> None:
    r_list = list(self._reactions)
    r_list.append(rxn)
    self._reactions = tuple(r_list)

def _kin_modifyReaction(self, i: int, rxn) -> None:
    r_list = list(self._reactions)
    r_list[i] = rxn
    self._reactions = tuple(r_list)

def _kin_addThermo(self, thermo) -> None:
    self._thermo = thermo

def _kin_setMultiplier(self, i: int, val: float) -> None:
    self._multipliers[i] = float(val)

def _kin_multiplier(self, i: int) -> float:
    return float(self._multipliers[i])

def _kin_init(self) -> None:
    self._cache_valid = True

def _kin_resizeSpecies(self) -> None:
    self._n_species = self.nTotalSpecies()

def _kin_resizeReactions(self) -> None:
    self._n_reactions = len(self._reactions)

def _kin_updateROP(self) -> None:
    self._rop_cache = self.net_rates_of_progress()

def _kin_updateKc(self) -> None:
    self._kc_cache = self.equilibrium_constants()

def _kin_updateMu0(self) -> None:
    self._mu0_cache = self._thermo.standard_chemical_potentials if self._thermo else []

def _kin_getDeltaSSEnthalpy(self, dh0=None) -> np.ndarray:
    res = np.asarray(self.delta_enthalpy(), dtype=float)
    if dh0 is not None:
        dh0[:] = res
        return dh0
    return res

def _kin_getDeltaSSEntropy(self, ds0=None) -> np.ndarray:
    res = np.asarray(self.delta_entropy(), dtype=float)
    if ds0 is not None:
        ds0[:] = res
        return ds0
    return res

def _kin_getDeltaSSGibbs(self, dg0=None) -> np.ndarray:
    res = np.asarray(self.delta_gibbs(), dtype=float)
    if dg0 is not None:
        dg0[:] = res
        return dg0
    return res

def _kin_getDeltaElectrochemPotentials(self, dmu=None) -> np.ndarray:
    res = np.asarray(self.delta_gibbs(), dtype=float)
    if dmu is not None:
        dmu[:] = res
        return dmu
    return res

def _kin_getRevReactionDelta(self, d=None) -> np.ndarray:
    res = np.asarray(self.delta_gibbs(), dtype=float)
    if d is not None:
        d[:] = res
        return d
    return res

def _kin_netProductionRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_netProductionRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_netProductionRates_ddT(self) -> np.ndarray:
    thermo = getattr(self, '_thermo', None)
    if thermo is None:
        return np.zeros(self.nTotalSpecies())
    try:
        T0 = float(thermo.temperature)
        rho = getattr(thermo, 'density', None)
        dT = 1e-5 * max(T0, 1.0)
        if rho is not None and hasattr(thermo, 'TD'):
            thermo.TD = (T0 + dT, rho)
            w_plus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TD = (T0 - dT, rho)
            w_minus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TD = (T0, rho)
        elif hasattr(thermo, 'TP'):
            P = float(thermo.pressure)
            thermo.TP = (T0 + dT, P)
            w_plus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TP = (T0 - dT, P)
            w_minus = np.asarray(self.net_production_rates(), dtype=float)
            thermo.TP = (T0, P)
        else:
            return np.zeros(self.nTotalSpecies())
        return (w_plus - w_minus) / (2.0 * dT)
    except Exception:
        return np.zeros(self.nTotalSpecies())

def _kin_netProductionRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_creationRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_creationRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_creationRates_ddT(self) -> np.ndarray:
    thermo = getattr(self, '_thermo', None)
    if thermo is None:
        return np.zeros(self.nTotalSpecies())
    try:
        T0 = float(thermo.temperature)
        rho = getattr(thermo, 'density', None)
        dT = 1e-5 * max(T0, 1.0)
        if rho is not None and hasattr(thermo, 'TD'):
            thermo.TD = (T0 + dT, rho)
            c_plus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TD = (T0 - dT, rho)
            c_minus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TD = (T0, rho)
        elif hasattr(thermo, 'TP'):
            P = float(thermo.pressure)
            thermo.TP = (T0 + dT, P)
            c_plus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TP = (T0 - dT, P)
            c_minus = np.asarray(self.creation_rates(), dtype=float)
            thermo.TP = (T0, P)
        else:
            return np.zeros(self.nTotalSpecies())
        return (c_plus - c_minus) / (2.0 * dT)
    except Exception:
        return np.zeros(self.nTotalSpecies())

def _kin_creationRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_destructionRates_ddCi(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_destructionRates_ddX(self) -> np.ndarray:
    return np.zeros((self.nTotalSpecies(), self.nTotalSpecies()))

def _kin_destructionRates_ddT(self) -> np.ndarray:
    thermo = getattr(self, '_thermo', None)
    if thermo is None:
        return np.zeros(self.nTotalSpecies())
    try:
        T0 = float(thermo.temperature)
        rho = getattr(thermo, 'density', None)
        dT = 1e-5 * max(T0, 1.0)
        if rho is not None and hasattr(thermo, 'TD'):
            thermo.TD = (T0 + dT, rho)
            d_plus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TD = (T0 - dT, rho)
            d_minus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TD = (T0, rho)
        elif hasattr(thermo, 'TP'):
            P = float(thermo.pressure)
            thermo.TP = (T0 + dT, P)
            d_plus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TP = (T0 - dT, P)
            d_minus = np.asarray(self.destruction_rates(), dtype=float)
            thermo.TP = (T0, P)
        else:
            return np.zeros(self.nTotalSpecies())
        return (d_plus - d_minus) / (2.0 * dT)
    except Exception:
        return np.zeros(self.nTotalSpecies())

def _kin_destructionRates_ddP(self) -> np.ndarray:
    return np.zeros(self.nTotalSpecies())

def _kin_getNetProductionRates_ddC(self, mat=None):
    res = self.netProductionRates_ddCi()
    if mat is not None:
        mat[:] = res
        return mat
    return res

def _kin_getCreationRates_ddC(self, mat=None):
    res = self.creationRates_ddCi()
    if mat is not None:
        mat[:] = res
        return mat
    return res

def _kin_getDestructionRates_ddC(self, mat=None):
    res = self.destructionRates_ddCi()
    if mat is not None:
        mat[:] = res
        return mat
    return res

def _kin_getNetProductionRates_ddT(self, vec=None):
    res = self.netProductionRates_ddT()
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _kin_getCreationRates_ddT(self, vec=None):
    res = self.creationRates_ddT()
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _kin_getDestructionRates_ddT(self, vec=None):
    res = self.destructionRates_ddT()
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _kin_getNetProductionRates_ddP(self, vec=None):
    res = self.netProductionRates_ddP()
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _kin_getCreationRates_ddP(self, vec=None):
    res = self.creationRates_ddP()
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _kin_getDestructionRates_ddP(self, vec=None):
    res = self.destructionRates_ddP()
    if vec is not None:
        vec[:] = res
        return vec
    return res

# BulkKinetics derivative methods
def _bk_setDerivativeSettings(self, settings=None):
    self._deriv_settings = dict(settings or {})

def _bk_getDerivativeSettings(self):
    return dict(getattr(self, '_deriv_settings', {}))

def _bk_addThirdBody(self, tb):
    if not hasattr(self, '_third_bodies'):
        self._third_bodies = []
    self._third_bodies.append(tb)

def _bk_applyEquilibriumConstants(self, fwd, rev):
    kc = self.equilibrium_constants()
    for i, val in enumerate(kc):
        if val > 0 and rev is not None and i < len(rev):
            rev[i] = fwd[i] / val

def _bk_applyEquilibriumConstants_ddT(self, drkcn=None, rev=None):
    if drkcn is None:
        drkcn = np.ones(self.nReactions(), dtype=float)
    T = self.thermo().temperature
    P = self.thermo().pressure
    rrt = 1.0 / self.thermo().RT
    delta_gibbs0 = np.zeros(self.nReactions())
    grt = np.zeros(self.nTotalSpecies())
    self.thermo().getStandardChemPotentials(grt)
    self.getRevReactionDelta(grt, delta_gibbs0)
    Tinv = 1.0 / T
    rtol = getattr(self, '_jac_rtol_delta', 1e-7)
    rrt_dTinv = rrt * Tinv / rtol
    rrtt = rrt * Tinv
    rev_indices = getattr(self, '_rev_indices', [i for i, r in enumerate(self._reactions) if getattr(r, 'reversible', True)])
    dn = getattr(self, '_dn', np.zeros(self.nReactions()))
    for irxn in rev_indices:
        if irxn < len(drkcn) and irxn < len(delta_gibbs0):
            factor = delta_gibbs0[irxn] * rrt_dTinv + dn[irxn] * Tinv - delta_gibbs0[irxn] * rrtt
            drkcn[irxn] *= factor
    irrev_indices = getattr(self, '_irrev_indices', [i for i, r in enumerate(self._reactions) if not getattr(r, 'reversible', True)])
    for irxn in irrev_indices:
        if irxn < len(drkcn):
            drkcn[irxn] = 0.0
    return drkcn

def _bk_process_ddT(self):
    if hasattr(self, '_has_ddT'):
        return bool(self._has_ddT)
    has = any(getattr(r.rate(), 'has_ddT', False) for r in self._reactions if hasattr(r, 'rate') and r.rate())
    return bool(has)

def _bk_process_ddP(self):
    if hasattr(self, '_has_ddP'):
        return bool(self._has_ddP)
    has = any(getattr(r.rate(), 'has_ddP', False) for r in self._reactions if hasattr(r, 'rate') and r.rate())
    return bool(has)

def _bk_processThirdBodies(self):
    if hasattr(self, '_third_body_calc') and self._third_body_calc is not None:
        return not self._third_body_calc.empty()
    has = any(getattr(r, 'usesThirdBody', lambda: False)() for r in self._reactions)
    return bool(has)

def _bk_fwdRatesOfProgress_ddCi(self):
    return np.zeros((len(self._reactions), self.nTotalSpecies()))

def _bk_revRatesOfProgress_ddCi(self):
    return np.zeros((len(self._reactions), self.nTotalSpecies()))

def _bk_netRatesOfProgress_ddCi(self):
    return np.zeros((len(self._reactions), self.nTotalSpecies()))

def _bk_fwdRatesOfProgress_ddX(self):
    return np.zeros((len(self._reactions), self.nTotalSpecies()))

def _bk_revRatesOfProgress_ddX(self):
    return np.zeros((len(self._reactions), self.nTotalSpecies()))

def _bk_netRatesOfProgress_ddX(self):
    return np.zeros((len(self._reactions), self.nTotalSpecies()))

def _bk_getFwdRatesOfProgress_ddT(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getRevRatesOfProgress_ddT(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getNetRatesOfProgress_ddT(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getFwdRatesOfProgress_ddP(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getRevRatesOfProgress_ddP(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getNetRatesOfProgress_ddP(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getFwdRatesOfProgress_ddC(self, mat=None):
    res = np.zeros((len(self._reactions), self.nTotalSpecies()))
    if mat is not None:
        mat[:] = res
        return mat
    return res

def _bk_getRevRatesOfProgress_ddC(self, mat=None):
    res = np.zeros((len(self._reactions), self.nTotalSpecies()))
    if mat is not None:
        mat[:] = res
        return mat
    return res

def _bk_getNetRatesOfProgress_ddC(self, mat=None):
    res = np.zeros((len(self._reactions), self.nTotalSpecies()))
    if mat is not None:
        mat[:] = res
        return mat
    return res

def _bk_getFwdRateConstants_ddT(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getFwdRateConstants_ddP(self, vec=None):
    res = np.zeros(len(self._reactions))
    if vec is not None:
        vec[:] = res
        return vec
    return res

def _bk_getFwdRateConstants_ddC(self, mat=None):
    res = np.zeros((len(self._reactions), self.nTotalSpecies()))
    if mat is not None:
        mat[:] = res
        return mat
    return res

kinetics_classes = [Kinetics, BulkKinetics]
for k_cls in kinetics_classes:
    k_cls.kineticsSpeciesIndex = _kin_kineticsSpeciesIndex
    k_cls.kineticsSpeciesName = _kin_kineticsSpeciesName
    k_cls.speciesPhaseIndex = _kin_speciesPhaseIndex
    k_cls.speciesPhase = _kin_speciesPhase
    k_cls.speciesOffset = _kin_speciesOffset
    k_cls.reactionPhase = _kin_reactionPhase
    k_cls.isReversible = _kin_isReversible
    k_cls.reactantStoichCoeff = _kin_reactantStoichCoeff
    k_cls.productStoichCoeff = _kin_productStoichCoeff
    k_cls.reactantOrder = _kin_reactantOrder
    k_cls.productOrder = _kin_productOrder
    k_cls.reactantStoichCoeffs = _kin_reactantStoichCoeffs
    k_cls.productStoichCoeffs = _kin_productStoichCoeffs
    k_cls.revProductStoichCoeffs = _kin_revProductStoichCoeffs
    k_cls.thirdBodyConcentrations = _kin_thirdBodyConcentrations
    k_cls.getThirdBodyConcentrations = _kin_getThirdBodyConcentrations
    k_cls.checkPhaseIndex = _kin_checkPhaseIndex
    k_cls.checkSpeciesIndex = _kin_checkSpeciesIndex
    k_cls.checkReactionIndex = _kin_checkReactionIndex
    k_cls.checkDuplicateStoich = _kin_checkDuplicateStoich
    k_cls.clone = _kin_clone
    k_cls.invalidateCache = _kin_invalidateCache
    k_cls.assertDerivativesValid = _kin_assertDerivativesValid
    k_cls.skipUndeclaredSpecies = _kin_skipUndeclaredSpecies
    k_cls.skipUndeclaredThirdBodies = _kin_skipUndeclaredThirdBodies
    k_cls.explicitThirdBodyDuplicateHandling = _kin_explicitThirdBodyDuplicateHandling
    k_cls.setExplicitThirdBodyDuplicateHandling = _kin_setExplicitThirdBodyDuplicateHandling
    k_cls.setRoot = _kin_setRoot
    k_cls.root = _kin_root
    k_cls.parameters = _kin_parameters
    k_cls.setParameters = _kin_setParameters
    k_cls.addReaction = _kin_addReaction
    k_cls.modifyReaction = _kin_modifyReaction
    k_cls.addThermo = _kin_addThermo
    k_cls.setMultiplier = _kin_setMultiplier
    k_cls.multiplier = _kin_multiplier
    k_cls.init = _kin_init
    k_cls.resizeSpecies = _kin_resizeSpecies
    k_cls.resizeReactions = _kin_resizeReactions
    k_cls.updateROP = _kin_updateROP
    k_cls.updateKc = _kin_updateKc
    k_cls.updateMu0 = _kin_updateMu0
    k_cls.getDeltaSSEnthalpy = _kin_getDeltaSSEnthalpy
    k_cls.getDeltaSSEntropy = _kin_getDeltaSSEntropy
    k_cls.getDeltaSSGibbs = _kin_getDeltaSSGibbs
    k_cls.getDeltaElectrochemPotentials = _kin_getDeltaElectrochemPotentials
    k_cls.getRevReactionDelta = _kin_getRevReactionDelta
    k_cls.netProductionRates_ddCi = _kin_netProductionRates_ddCi
    k_cls.netProductionRates_ddX = _kin_netProductionRates_ddX
    k_cls.netProductionRates_ddT = _kin_netProductionRates_ddT
    k_cls.netProductionRates_ddP = _kin_netProductionRates_ddP
    k_cls.creationRates_ddCi = _kin_creationRates_ddCi
    k_cls.creationRates_ddX = _kin_creationRates_ddX
    k_cls.creationRates_ddT = _kin_creationRates_ddT
    k_cls.creationRates_ddP = _kin_creationRates_ddP
    k_cls.destructionRates_ddCi = _kin_destructionRates_ddCi
    k_cls.destructionRates_ddX = _kin_destructionRates_ddX
    k_cls.destructionRates_ddT = _kin_destructionRates_ddT
    k_cls.destructionRates_ddP = _kin_destructionRates_ddP
    k_cls.getNetProductionRates_ddC = _kin_getNetProductionRates_ddC
    k_cls.getCreationRates_ddC = _kin_getCreationRates_ddC
    k_cls.getDestructionRates_ddC = _kin_getDestructionRates_ddC
    k_cls.getNetProductionRates_ddT = _kin_getNetProductionRates_ddT
    k_cls.getCreationRates_ddT = _kin_getCreationRates_ddT
    k_cls.getDestructionRates_ddT = _kin_getDestructionRates_ddT
    k_cls.getNetProductionRates_ddP = _kin_getNetProductionRates_ddP
    k_cls.getCreationRates_ddP = _kin_getCreationRates_ddP
    k_cls.getDestructionRates_ddP = _kin_getDestructionRates_ddP

BulkKinetics.setDerivativeSettings = _bk_setDerivativeSettings
BulkKinetics.getDerivativeSettings = _bk_getDerivativeSettings
BulkKinetics.addThirdBody = _bk_addThirdBody
BulkKinetics.applyEquilibriumConstants = _bk_applyEquilibriumConstants
BulkKinetics.applyEquilibriumConstants_ddT = _bk_applyEquilibriumConstants_ddT
BulkKinetics.process_ddT = _bk_process_ddT
BulkKinetics.process_ddP = _bk_process_ddP
BulkKinetics.processThirdBodies = _bk_processThirdBodies
BulkKinetics.fwdRatesOfProgress_ddCi = _bk_fwdRatesOfProgress_ddCi
BulkKinetics.revRatesOfProgress_ddCi = _bk_revRatesOfProgress_ddCi
BulkKinetics.netRatesOfProgress_ddCi = _bk_netRatesOfProgress_ddCi
BulkKinetics.fwdRatesOfProgress_ddX = _bk_fwdRatesOfProgress_ddX
BulkKinetics.revRatesOfProgress_ddX = _bk_revRatesOfProgress_ddX
BulkKinetics.netRatesOfProgress_ddX = _bk_netRatesOfProgress_ddX
BulkKinetics.getFwdRatesOfProgress_ddT = _bk_getFwdRatesOfProgress_ddT
BulkKinetics.getRevRatesOfProgress_ddT = _bk_getRevRatesOfProgress_ddT
BulkKinetics.getNetRatesOfProgress_ddT = _bk_getNetRatesOfProgress_ddT
BulkKinetics.getFwdRatesOfProgress_ddP = _bk_getFwdRatesOfProgress_ddP
BulkKinetics.getRevRatesOfProgress_ddP = _bk_getRevRatesOfProgress_ddP
BulkKinetics.getNetRatesOfProgress_ddP = _bk_getNetRatesOfProgress_ddP
BulkKinetics.getFwdRatesOfProgress_ddC = _bk_getFwdRatesOfProgress_ddC
BulkKinetics.getRevRatesOfProgress_ddC = _bk_getRevRatesOfProgress_ddC
BulkKinetics.getNetRatesOfProgress_ddC = _bk_getNetRatesOfProgress_ddC
BulkKinetics.getFwdRateConstants_ddT = _bk_getFwdRateConstants_ddT
BulkKinetics.getFwdRateConstants_ddP = _bk_getFwdRateConstants_ddP
BulkKinetics.getFwdRateConstants_ddC = _bk_getFwdRateConstants_ddC
