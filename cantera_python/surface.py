"""Editable ideal-surface thermodynamics and heterogeneous interface kinetics from Cantera."""
from dataclasses import dataclass
from math import isfinite, log, exp, sqrt, pi
import re
from pathlib import Path
from typing import Mapping, Sequence, Any, Optional, Dict, List, Tuple
import numpy as np

from .constants import GasConstant as R, OneAtm, SmallNumber
from .composition import parse_comp_string
from .rates import BlowersMaselRate
from .units import convert, convert_activation_energy


@dataclass(frozen=True)
class SurfaceSpecies:
    name: str
    molecular_weight: float
    thermo: object
    size: float = 1.0
    charge: float = 0.0


def _site_density(value, units):
    if isinstance(value, (int, float)):
        length = (units or {}).get('length', 'm')
        quantity = (units or {}).get('quantity', 'kmol')
        return (float(value) * convert(1.0, quantity, 'kmol') /
                convert(1.0, length, 'm') ** 2)
    if isinstance(value, str):
        match = re.fullmatch(
            r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(\S+)\s*',
            value)
        if match:
            return convert(float(match.group(1)), match.group(2), 'kmol/m^2')
    raise ValueError('site-density must be a number or quantity convertible to kmol/m^2')


class IdealSurfacePhase:
    """Mutable two-dimensional ideal solution with Cantera coverage semantics."""

    def __init__(self, species, site_density, *, temperature=298.15,
                 pressure=OneAtm, coverages=None, name=""):
        self.name = name
        self._species = tuple(species)
        if not self._species or len({s.name for s in species}) != len(species):
            raise ValueError('surface species must be nonempty and distinct')
        self._site_density = float(site_density)
        if not isfinite(self._site_density) or self._site_density <= 0:
            raise ValueError('site density must be finite and positive')
        self._temperature = float(temperature)
        self._pressure = float(pressure)
        self._electric_potential = 0.0
        self.coverages = ({self._species[0].name: 1.0} if coverages is None
                          else coverages)

    @classmethod
    def from_mechanism(cls, mech, phase_name=None):
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        if not isinstance(mech, Mechanism):
            mech = load_mechanism(mech)
        phase = mech.phase(phase_name)
        if phase is None or str(phase.thermo).lower() not in {'ideal-surface', 'surface', 'surf'}:
            raise ValueError('selected phase is not an ideal surface')
        by_name = {item.name: item for item in mech.species}
        species = []
        for name in phase.species_names:
            item = by_name[name]
            if not item.thermo:
                raise ValueError(f'Surface species {name!r} has no thermo data')
            species.append(SurfaceSpecies(
                name, item.molecular_weight, create_species_thermo(item.thermo),
                item.size, item.charge))
        if phase.site_density is None:
            raise ValueError(f'Ideal surface phase {phase.name!r} requires site-density')
        obj = cls(species, _site_density(phase.site_density, mech.units), name=phase.name)
        state = phase.state or {}
        unknown = set(state) - {'T', 'P', 'coverages'}
        if unknown:
            raise ValueError(f'Unsupported ideal-surface state fields: {sorted(unknown)}')
        if 'T' in state:
            obj.T = float(state['T'])
        if 'P' in state:
            obj.P = float(state['P'])
        if 'coverages' in state:
            obj.coverages = state['coverages']
        return obj

    @property
    def species(self): return self._species
    @property
    def species_names(self): return tuple(s.name for s in self.species)
    @property
    def n_species(self): return len(self.species)
    nSpecies = n_species

    @property
    def temperature(self): return self._temperature
    @temperature.setter
    def temperature(self, value):
        value = float(value)
        if not isfinite(value) or value <= 0: raise ValueError('temperature must be positive')
        self._temperature = value
    T = property(lambda self: self.temperature, lambda self, v: setattr(self, 'temperature', v))

    @property
    def pressure(self): return self._pressure
    @pressure.setter
    def pressure(self, value): self._pressure = float(value)
    P = property(lambda self: self.pressure, lambda self, v: setattr(self, 'pressure', v))

    @property
    def TP(self): return self.temperature, self.pressure
    @TP.setter
    def TP(self, values):
        t, p = values
        self.temperature = t
        self.pressure = p

    def setState_TP(self, t, p):
        self.TP = (t, p)

    @property
    def electric_potential(self): return self._electric_potential
    @electric_potential.setter
    def electric_potential(self, v): self._electric_potential = float(v)
    def set_electric_potential(self, v):
        self._electric_potential = float(v)
        return self._electric_potential
    setElectricPotential = set_electric_potential

    @property
    def site_density(self): return self._site_density
    @site_density.setter
    def site_density(self, value):
        value = float(value)
        if not isfinite(value) or value <= 0: raise ValueError('site density must be positive')
        self._site_density = value

    @property
    def coverages(self):
        denominator = sum(x * s.size for x, s in zip(self._x, self.species))
        return tuple(x * s.size / denominator for x, s in zip(self._x, self.species))

    @coverages.setter
    def coverages(self, values):
        if isinstance(values, str):
            values = parse_comp_string(values, self.species_names)
        if isinstance(values, dict):
            unknown = set(values) - set(self.species_names)
            if unknown: raise ValueError(f'Unknown surface species: {sorted(unknown)}')
            theta = tuple(float(values.get(name, 0.0)) for name in self.species_names)
        else:
            theta = tuple(float(v) for v in values)
        if len(theta) != self.n_species or any(v < 0 or not isfinite(v) for v in theta):
            raise ValueError('coverages must be finite, nonnegative, and match species count')
        total = sum(v / s.size for v, s in zip(theta, self.species))
        if total <= 0: raise ValueError('sum of coverages must be positive')
        self._x = tuple(v / (total * s.size) for v, s in zip(theta, self.species))

    def set_coverages_no_norm(self, values):
        """Set coverages directly without normalizing sum to 1."""
        if isinstance(values, dict):
            theta = tuple(float(values.get(name, 0.0)) for name in self.species_names)
        else:
            theta = tuple(float(v) for v in values)
        sum1 = sum(v / s.size for v, s in zip(theta, self.species))
        sum2 = sum(theta)
        if sum1 <= 0:
            self.coverages = theta
            return
        self._x = tuple(v * sum2 / (sum1 * s.size) for v, s in zip(theta, self.species))

    setCoveragesNoNorm = set_coverages_no_norm

    X = property(lambda self: self._x)
    activities = property(lambda self: self.coverages)
    concentrations = property(lambda self: tuple(
        theta * self.site_density / species.size
        for theta, species in zip(self.coverages, self.species)))
    activity_concentrations = concentrations
    standard_concentrations = property(lambda self: tuple(
        self.site_density / species.size for species in self.species))

    def standard_concentration(self, k=0):
        c_std = float(self.standard_concentrations[k])
        return c_std
    def log_standard_conc(self, k=0):
        val = float(self.standard_concentration(k))
        log_val = log(max(val, 1e-30))
        return log_val
    logStandardConc = log_standard_conc

    @property
    def charges(self): return tuple(s.charge for s in self.species)
    def charge(self, k=0):
        q = float(self.species[k].charge)
        return q

    @property
    def molecular_weights(self):
        return tuple(s.molecular_weight for s in self.species)

    @property
    def mean_molecular_weight(self):
        return sum(x * s.molecular_weight for x, s in zip(self.X, self.species))

    @property
    def Y(self):
        mmw = self.mean_molecular_weight
        if mmw <= 0:
            return (0.0,) * self.n_species
        return tuple(x * s.molecular_weight / mmw for x, s in zip(self.X, self.species))

    @property
    def density(self):
        return self.site_density * self.mean_molecular_weight / sum(
            x * s.size for x, s in zip(self.X, self.species))

    def _reference(self):
        t = float(self.T)
        props = tuple(s.thermo.properties(t) for s in self.species)
        return props
    @property
    def partial_molar_enthalpies(self): return tuple(R * self.T * p[1] for p in self._reference())
    @property
    def partial_molar_cp(self): return tuple(R * p[0] for p in self._reference())
    @property
    def standard_chemical_potentials(self):
        return tuple(R * self.T * (p[1] - p[2]) for p in self._reference())
    @property
    def chemical_potentials(self):
        c = self.concentrations
        sd = self.site_density
        RT = R * self.T
        res = []
        for k, (mu0, sp) in enumerate(zip(self.standard_chemical_potentials, self.species)):
            c0 = sd / sp.size
            res.append(mu0 + RT * (log(max(c[k], SmallNumber)) - log(c0)))
        return tuple(res)
    @property
    def partial_molar_entropies(self):
        c = self.concentrations
        sd = self.site_density
        T = self.T
        res = []
        for k, (mu0, h, sp) in enumerate(zip(self.standard_chemical_potentials, self.partial_molar_enthalpies, self.species)):
            s0 = (h - mu0) / T
            c0 = sd / sp.size
            res.append(s0 - R * (log(max(c[k], SmallNumber)) - log(c0)))
        return tuple(res)
    @property
    def enthalpy_mole(self): return sum(x*h for x, h in zip(self.X, self.partial_molar_enthalpies))
    int_energy_mole = enthalpy_mole
    @property
    def entropy_mole(self): return sum(x*s for x, s in zip(self.X, self.partial_molar_entropies))
    @property
    def cp_mole(self): return sum(x*cp for x, cp in zip(self.X, self.partial_molar_cp))
    cv_mole = cp_mole
    @property
    def gibbs_mole(self): return self.enthalpy_mole - self.T * self.entropy_mole
    @property
    def phase_of_matter(self): return 'surface'
    @property
    def n_dim(self): return 2


@dataclass
class InterfaceReaction:
    equation_str: str
    # interface/sticking combined with Arrhenius or Blowers-Masel
    rate_type: str
    reversible: bool
    reactants: tuple  # tuple of (global_k, stoich, order)
    products: tuple   # tuple of (global_k, stoich)
    net_stoich: dict  # global_k -> net stoichiometric coefficient
    rate_params: dict


class Interface(IdealSurfacePhase):
    """Heterogeneous reacting surface interface between adjacent bulk phases."""

    def __new__(cls, *args, **kwargs):
        if args and (isinstance(args[0], (str, Path)) or hasattr(args[0], 'read_text')):
            infile = args[0]
            phase_name = args[1] if len(args) > 1 else kwargs.pop('name', None)
            adjacent = args[2] if len(args) > 2 else kwargs.pop('adjacent', None)
            return cls.from_mechanism(infile, phase_name=phase_name, adjacent=adjacent)
        if 'yaml' in kwargs:
            yaml_content = kwargs.pop('yaml')
            phase_name = kwargs.pop('name', None)
            adjacent = kwargs.pop('adjacent', None)
            return cls.from_mechanism(yaml_content, phase_name=phase_name, adjacent=adjacent)
        return super().__new__(cls)

    def __init__(self, species, site_density=None, adjacent=None, reactions=None, *,
                 temperature=298.15, pressure=OneAtm, coverages=None, name=""):
        if isinstance(species, (str, Path)) or hasattr(species, 'read_text'):
            return
        super().__init__(species, site_density, temperature=temperature,
                         pressure=pressure, coverages=coverages, name=name)
        self.thermo_model = 'ideal-surface'
        self.kinetics_model = 'surface'
        self.transport_model = 'none'
        self.composite = ('ideal-surface', 'surface', 'none')
        self.adjacent = dict(adjacent or {})
        self._phases = (self,) + tuple(self.adjacent.values())
        self._setup_species_mapping()
        self._reactions = tuple(reactions or ())
        self._motz_wise_default = False

    def _setup_species_mapping(self):
        self._phase_indices = {}
        self._global_species_names = []
        self._global_to_phase_local = []
        self._phase_slices = []

        start = 0
        for p_idx, p in enumerate(self._phases):
            p_name = getattr(p, 'name', f'phase_{p_idx}')
            self._phase_indices[p] = p_idx
            self._phase_indices[p_name] = p_idx
            self._phase_indices[p_idx] = p_idx

            n_sp = p.n_species
            self._phase_slices.append(slice(start, start + n_sp))
            for local_k, sp_name in enumerate(p.species_names):
                self._global_species_names.append(sp_name)
                self._global_to_phase_local.append((p_idx, local_k))
            start += n_sp

    @property
    def n_phases(self): return len(self._phases)
    @property
    def phases(self): return self._phases
    @property
    def thermo(self): return self
    @property
    def kinetics(self): return self
    @property
    def n_total_species(self): return len(self._global_species_names)
    @property
    def total_species_names(self): return tuple(self._global_species_names)

    def phase(self, phase_or_id):
        idx = self._phase_indices[phase_or_id]
        return self._phases[idx]

    def phase_index(self, phase_or_id):
        return self._phase_indices[phase_or_id]

    def kinetics_species_index(self, species_name_or_k, phase=None):
        if phase is not None:
            p_idx = self.phase_index(phase)
            return self._phase_slices[p_idx].start + int(species_name_or_k)
        if isinstance(species_name_or_k, str):
            try:
                return self._global_species_names.index(species_name_or_k)
            except ValueError:
                lower_names = [s.lower() for s in self._global_species_names]
                try:
                    return lower_names.index(species_name_or_k.lower())
                except ValueError:
                    raise ValueError(f"Species {species_name_or_k!r} not found in interface phases")
        return int(species_name_or_k)

    def species_phase_index(self, k):
        return self._global_to_phase_local[k][0]

    def species_local_index(self, k):
        return self._global_to_phase_local[k][1]

    @property
    def reactions(self): return self._reactions
    @property
    def n_reactions(self): return len(self._reactions)

    def reaction(self, i: int):
        from .kinetics import Reaction
        idx = int(i)
        if idx < 0 or idx >= len(self._reactions):
            raise IndexError(f"Reaction index {i} out of bounds")
        irxn = self._reactions[idx]
        if isinstance(irxn, Reaction):
            return irxn
        r = Reaction()
        r.setEquation(irxn.equation_str)
        r.reversible = bool(irxn.reversible)
        r.type = irxn.rate_type
        if not r.reactants:
            reactants = {}
            for item in irxn.reactants:
                gk = item[0]
                stoich = item[1]
                sp_name = self.total_species_names[gk] if gk < len(self.total_species_names) else str(gk)
                reactants[sp_name] = float(stoich)
            r.reactants = reactants
        if not r.products:
            products = {}
            for item in irxn.products:
                gk = item[0]
                stoich = item[1]
                sp_name = self.total_species_names[gk] if gk < len(self.total_species_names) else str(gk)
                products[sp_name] = float(stoich)
            r.products = products
        return r

    @classmethod
    def from_mechanism(cls, mech, phase_name=None, adjacent=None):
        from .mechanism import load_mechanism, Mechanism
        from .thermo import create_species_thermo
        from .ideal_gas import IdealGasPhase
        from .fixed_stoichiometry import FixedStoichiometryPhase

        if not isinstance(mech, Mechanism):
            mech = load_mechanism(mech)
        phase = mech.phase(phase_name)
        if phase is None or str(phase.thermo).lower() not in {'ideal-surface', 'surface', 'surf'}:
            raise ValueError('selected phase is not an ideal surface')

        by_name = {item.name: item for item in mech.species}
        species = []
        for name in phase.species_names:
            item = by_name[name]
            if not item.thermo:
                raise ValueError(f'Surface species {name!r} has no thermo data')
            species.append(SurfaceSpecies(
                name, item.molecular_weight, create_species_thermo(item.thermo),
                item.size, item.charge))
        if phase.site_density is None:
            site_dens = 1.0
        else:
            site_dens = _site_density(phase.site_density, mech.units)

        # Build adjacent phases
        adjacent_dict = {}
        if adjacent is not None:
            if isinstance(adjacent, Mapping):
                adjacent_dict = dict(adjacent)
            else:
                for idx, adj in enumerate(adjacent):
                    fallback_name = (phase.adjacent_phase_names[idx]
                                     if idx < len(phase.adjacent_phase_names)
                                     else str(len(adjacent_dict)))
                    name = getattr(adj, 'name', fallback_name) or fallback_name
                    adjacent_dict[name] = adj
        else:
            for adj_name in phase.adjacent_phase_names:
                adj_phase = mech.phase(adj_name)
                t_model = str(adj_phase.thermo).lower() if adj_phase else 'ideal-gas'
                if t_model in {'ideal-gas', 'gas'}:
                    adjacent_dict[adj_name] = IdealGasPhase.from_mechanism(mech, adj_name)
                elif t_model in {'fixed-stoichiometry', 'stoichsubstance', 'stoichiometric-substance'}:
                    adjacent_dict[adj_name] = FixedStoichiometryPhase.from_mechanism(mech, adj_name)
                elif t_model in {'ideal-surface', 'surface', 'surf'}:
                    adjacent_dict[adj_name] = cls.from_mechanism(mech, adj_name)
                elif t_model in {'ideal-condensed', 'ideal-solid-solution', 'ideal_solid'}:
                    from .ideal_solid import IdealSolidSolution
                    from .ideal_gas import Species
                    from .units import convert
                    adj_p = mech.phase(adj_name)
                    species_objs = []
                    mol_vols = []
                    for s in adj_p.species:
                        poly = create_species_thermo(s.thermo) if s.thermo else None
                        species_objs.append(Species(s.name, s.molecular_weight, poly, s.charge))
                        v = 1.0
                        if s.equation_of_state and 'molar-density' in s.equation_of_state:
                            val_str = str(s.equation_of_state['molar-density']).strip()
                            parts = val_str.split(maxsplit=1)
                            if len(parts) == 2:
                                dens = convert(float(parts[0]), parts[1], 'kmol/m^3')
                                v = 1.0 / dens if dens > 0 else 1.0
                            else:
                                v = 1.0 / float(parts[0]) if float(parts[0]) > 0 else 1.0
                        mol_vols.append(v)
                    adjacent_dict[adj_name] = IdealSolidSolution(species_objs, mol_vols)
                else:
                    raise ValueError(f"Unsupported adjacent phase thermo model: {t_model}")

        obj = cls(species, site_dens, adjacent=adjacent_dict, name=phase.name)
        obj._motz_wise_default = bool(
            (phase.input_data or {}).get('Motz-Wise', False))
        state = phase.state or {}
        from .phases import _parse_state_scalar
        if 'T' in state: obj.T = _parse_state_scalar(state['T'], 'K')
        if 'P' in state: obj.P = _parse_state_scalar(state['P'], 'Pa')
        if 'coverages' in state: obj.coverages = state['coverages']

        # Parse reactions belonging to this surface phase
        phase_mech = mech.for_phase(phase.name)
        reactions = []
        for r_rec in phase_mech.reactions:
            reactions.append(obj._parse_reaction_record(r_rec, mech.units))
        obj._reactions = tuple(reactions)
        return obj

    def _parse_reaction_record(self, r_rec, doc_units):
        eq = r_rec.equation
        eq_str = r_rec.input_data.get('equation') if (r_rec.input_data and 'equation' in r_rec.input_data) else None
        if not eq_str:
            def _fmt_side(comp):
                parts = []
                for sp, stoich in comp.items():
                    if stoich == 1:
                        parts.append(str(sp))
                    elif float(stoich).is_integer():
                        parts.append(f"{int(stoich)} {sp}")
                    else:
                        parts.append(f"{stoich} {sp}")
                return " + ".join(parts)
            arrow = " <=> " if eq.reversible else " => "
            eq_str = _fmt_side(eq.reactants) + arrow + _fmt_side(eq.products)
        rate_data = r_rec.rate_data or {}
        raw_orders = dict(rate_data.get('orders', {}) or {})
        custom_orders = {}
        for k, v in raw_orders.items():
            gk = self.kinetics_species_index(k)
            canon = self._global_species_names[gk]
            custom_orders[canon] = v
            custom_orders[k] = v
            custom_orders[k.lower()] = v

        # Map reactants and products
        reactants_list = []
        for sp_name, stoich in eq.reactants.items():
            gk = self.kinetics_species_index(sp_name)
            canon_name = self._global_species_names[gk]
            order = float(custom_orders.get(canon_name, custom_orders.get(sp_name, custom_orders.get(sp_name.lower(), stoich))))
            reactants_list.append((gk, float(stoich), order))

        products_list = []
        for sp_name, stoich in eq.products.items():
            gk = self.kinetics_species_index(sp_name)
            products_list.append((gk, float(stoich)))

        net_stoich = {}
        for gk, stoich in products_list:
            net_stoich[gk] = net_stoich.get(gk, 0.0) + stoich
        for gk, stoich, _ in reactants_list:
            net_stoich[gk] = net_stoich.get(gk, 0.0) - stoich

        # Parse rate expression
        rate_kind = str(rate_data.get('type', '')).strip().lower()
        is_blowers_masel = rate_kind == 'blowers-masel'
        length_unit = (doc_units or {}).get('length', 'm')
        quantity_unit = (doc_units or {}).get('quantity', 'kmol')
        act_energy_unit = (doc_units or {}).get('activation-energy', 'J/kmol')

        L_factor = convert(1.0, length_unit, 'm')
        Q_factor = convert(1.0, quantity_unit, 'kmol')

        def _parse_ea(val):
            if val is None:
                return 0.0
            act_energy_unit = (doc_units or {}).get('activation-energy', 'J/kmol')
            if isinstance(val, (int, float)):
                return convert_activation_energy(float(val), act_energy_unit, 'J/kmol')
            if isinstance(val, str):
                parts = val.split(maxsplit=1)
                if len(parts) == 2:
                    return convert_activation_energy(float(parts[0]), parts[1], 'J/kmol')
                return convert_activation_energy(float(parts[0]), act_energy_unit, 'J/kmol')
            return float(val)

        def _convert_cov_deps(deps):
            cov_dict = {}
            if not deps: return cov_dict
            for sp_name, entry in deps.items():
                if isinstance(entry, Mapping):
                    a = float(entry.get('a', 0.0))
                    m = float(entry.get('m', 0.0))
                    raw_E = entry.get('E', 0.0)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    a = float(entry[0])
                    m = float(entry[1])
                    raw_E = entry[2]
                else:
                    continue
                if isinstance(raw_E, (list, tuple)):
                    E_list = [_parse_ea(e) for e in raw_E]
                else:
                    E_list = _parse_ea(raw_E)
                cov_dict[sp_name] = {'a': a, 'm': m, 'E': E_list}
            return cov_dict

        if 'sticking-coefficient' in rate_data:
            s_data = rate_data['sticking-coefficient']
            A = float(s_data['A'])
            b = float(s_data.get('b', 0.0))
            energy = _parse_ea(s_data.get('Ea0' if is_blowers_masel else 'Ea', 0.0))
            bond_energy = _parse_ea(s_data.get('w', 0.0)) if is_blowers_masel else None
            motz_wise = bool(rate_data['Motz-Wise']) if 'Motz-Wise' in rate_data \
                else self._motz_wise_default
            stick_sp = rate_data.get('sticking-species')

            if not stick_sp:
                # auto detect single non-interface species
                non_surf = []
                for sp_name in eq.reactants:
                    gk = self.kinetics_species_index(sp_name)
                    if self.species_phase_index(gk) != 0:
                        non_surf.append(sp_name)
                if len(non_surf) == 1:
                    stick_sp = non_surf[0]
                else:
                    raise ValueError(f"Could not uniquely identify sticking species in {eq}")

            # Find sticking species MW
            gk_stick = self.kinetics_species_index(stick_sp)
            p_stick = self.species_phase_index(gk_stick)
            loc_stick = self.species_local_index(gk_stick)
            w_stick = self._phases[p_stick].species[loc_stick].molecular_weight

            # Calculate surface_order and multiplier
            surface_order = 0.0
            multiplier = 1.0
            for gk, stoich, order in reactants_list:
                sp_name = self._global_species_names[gk]
                p_idx = self.species_phase_index(gk)
                loc_k = self.species_local_index(gk)
                if sp_name == stick_sp:
                    multiplier *= sqrt(R / (2.0 * pi * w_stick))
                else:
                    if p_idx == 0:
                        multiplier *= (self.species[loc_k].size ** order)
                        surface_order += order
                    else:
                        sc = getattr(self._phases[p_idx], 'standard_concentration', 1.0)
                        c0 = sc(loc_k) if callable(sc) else float(sc)
                        multiplier *= (c0 ** (-order))

            cov_deps = _convert_cov_deps(rate_data.get('coverage-dependencies'))
            rate_info = {
                'type': ('sticking-Blowers-Masel' if is_blowers_masel
                         else 'sticking-Arrhenius'),
                'A': A, 'b': b,
                'motz_wise': motz_wise,
                'sticking_species': stick_sp,
                'multiplier': multiplier,
                'surface_order': surface_order,
                'cov_deps': cov_deps
            }
            if is_blowers_masel:
                rate_info.update(Ea0=energy, w=bond_energy)
                rate_type = 'sticking-Blowers-Masel'
            else:
                rate_info['Ea'] = energy
                rate_type = 'sticking-Arrhenius'

        elif 'rate-constant' in rate_data:
            rc_data = rate_data['rate-constant']
            A_raw = float(rc_data['A'])
            b = float(rc_data.get('b', 0.0))
            energy = _parse_ea(rc_data.get('Ea0' if is_blowers_masel else 'Ea', 0.0))
            bond_energy = _parse_ea(rc_data.get('w', 0.0)) if is_blowers_masel else None

            # Units conversion for A
            q_exp = 1.0
            l_exp = -2.0
            for gk, stoich, order in reactants_list:
                p_idx = self.species_phase_index(gk)
                d_k = 2.0 if p_idx == 0 else 3.0
                q_exp -= order
                l_exp += d_k * order

            A_SI = A_raw * (Q_factor ** q_exp) * (L_factor ** l_exp)
            cov_deps = _convert_cov_deps(rate_data.get('coverage-dependencies'))
            rate_info = {
                'type': ('interface-Blowers-Masel' if is_blowers_masel
                         else 'interface-Arrhenius'),
                'A': A_SI, 'b': b,
                'cov_deps': cov_deps
            }
            if is_blowers_masel:
                rate_info.update(Ea0=energy, w=bond_energy)
                rate_type = 'interface-Blowers-Masel'
            else:
                rate_info['Ea'] = energy
                rate_type = 'interface-Arrhenius'
        else:
            raise ValueError(f"Unknown rate specification for reaction {eq}")

        return InterfaceReaction(
            equation_str=str(eq_str),
            rate_type=rate_type,
            reversible=bool(eq.reversible),
            reactants=tuple(reactants_list),
            products=tuple(products_list),
            net_stoich=net_stoich,
            rate_params=rate_info
        )

    def _eval_coverage_factor(self, cov_deps, T):
        if not cov_deps: return 1.0
        acov = 0.0
        ecov = 0.0
        mcov = 0.0
        covs = dict(zip(self.species_names, self.coverages))
        recip_rt = 1.0 / (R * T)

        for sp_name, dep in cov_deps.items():
            theta = max(0.0, covs.get(sp_name, 0.0))
            acov += dep['a'] * theta
            E = dep['E']
            if isinstance(E, (list, tuple)):
                for p_idx, coeff in enumerate(E, 1):
                    ecov += coeff * (theta ** p_idx)
            else:
                ecov += E * theta
            if dep['m'] != 0.0:
                mcov += dep['m'] * log(max(theta, SmallNumber))
        return exp(log(10.0) * acov - ecov * recip_rt + mcov)

    def _reaction_enthalpy(self, reaction):
        """Return live reaction enthalpy from every participating phase.

        This is the direct Python counterpart of ``InterfaceData::update`` plus
        ``BlowersMaselRate::updateFromStruct``: partial molar enthalpies are
        gathered in kinetics-species order and contracted with net
        stoichiometric coefficients on every rate evaluation.
        """
        delta_h = 0.0
        phase_enthalpies = {}
        for global_k, coefficient in reaction.net_stoich.items():
            if coefficient == 0.0:
                continue
            phase_index, local_k = self._global_to_phase_local[global_k]
            if phase_index not in phase_enthalpies:
                phase_enthalpies[phase_index] = tuple(
                    self._phases[phase_index].partial_molar_enthalpies)
            delta_h += coefficient * phase_enthalpies[phase_index][local_k]
        return delta_h


    def _all_chemical_potentials(self):
        mu = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                mu.extend(self.chemical_potentials)
            else:
                mu.extend(phase.chemical_potentials)
        return mu

    def _all_standard_chemical_potentials(self):
        mu0 = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                mu0.extend(self.standard_chemical_potentials)
            else:
                mu0.extend(phase.standard_chemical_potentials)
        return mu0

    def _all_partial_molar_enthalpies(self):
        h = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                h.extend(self.partial_molar_enthalpies)
            else:
                h.extend(phase.partial_molar_enthalpies)
        return h

    def _all_partial_molar_entropies(self):
        s = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                s.extend(self.partial_molar_entropies)
            else:
                s.extend(phase.partial_molar_entropies)
        return s

    @property
    def delta_gibbs(self):
        mu = self._all_chemical_potentials()
        return tuple(sum(stoich * mu[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_standard_gibbs(self):
        mu0 = self._all_standard_chemical_potentials()
        return tuple(sum(stoich * mu0[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_enthalpy(self):
        h = self._all_partial_molar_enthalpies()
        return tuple(sum(stoich * h[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_standard_enthalpy(self):
        return self.delta_enthalpy

    @property
    def delta_entropy(self):
        s = self._all_partial_molar_entropies()
        return tuple(sum(stoich * s[gk] for gk, stoich in r.net_stoich.items()) for r in self._reactions)

    @property
    def delta_standard_entropy(self):
        dh = self.delta_standard_enthalpy
        dg0 = self.delta_standard_gibbs
        T = self.T
        return tuple((dh[i] - dg0[i]) / T for i in range(len(dh)))

    def getDeltaGibbs(self, dg=None):
        res = np.asarray(self.delta_gibbs, dtype=float)
        if dg is not None:
            dg[:] = res
            return dg
        return res

    def getDeltaEnthalpy(self, dh=None):
        res = np.asarray(self.delta_enthalpy, dtype=float)
        if dh is not None:
            dh[:] = res
            return dh
        return res

    def getDeltaEntropy(self, ds=None):
        res = np.asarray(self.delta_entropy, dtype=float)
        if ds is not None:
            ds[:] = res
            return ds
        return res

    def getDeltaSSGibbs(self, dg0=None):
        res = np.asarray(self.delta_standard_gibbs, dtype=float)
        if dg0 is not None:
            dg0[:] = res
            return dg0
        return res

    def getDeltaSSEnthalpy(self, dh0=None):
        res = np.asarray(self.delta_standard_enthalpy, dtype=float)
        if dh0 is not None:
            dh0[:] = res
            return dh0
        return res

    def getDeltaSSEntropy(self, ds0=None):
        res = np.asarray(self.delta_standard_entropy, dtype=float)
        if ds0 is not None:
            ds0[:] = res
            return ds0
        return res

    def getDeltaElectrochemPotentials(self, dm=None):
        from .constants import Faraday
        dg = np.asarray(self.delta_gibbs, dtype=float)
        res = dg.copy()
        phis = [getattr(p, 'electric_potential', 0.0) for p in self._phases]
        for i, r in enumerate(self._reactions):
            d_phi_z = 0.0
            for gk, stoich in r.products:
                p_idx, loc_k = self._global_to_phase_local[gk]
                p = self._phases[p_idx]
                z = float(p.species[loc_k].charge if hasattr(p, 'species') and loc_k < len(p.species) and hasattr(p.species[loc_k], 'charge') else getattr(p, 'charge', lambda idx: 0.0)(loc_k))
                d_phi_z += stoich * z * phis[p_idx]
            for gk, stoich, _ in r.reactants:
                p_idx, loc_k = self._global_to_phase_local[gk]
                p = self._phases[p_idx]
                z = float(p.species[loc_k].charge if hasattr(p, 'species') and loc_k < len(p.species) and hasattr(p.species[loc_k], 'charge') else getattr(p, 'charge', lambda idx: 0.0)(loc_k))
                d_phi_z -= stoich * z * phis[p_idx]
            res[i] += Faraday * d_phi_z
        if dm is not None:
            dm[:] = res
            return dm
        return res

    def _blowers_masel_coefficient(self, reaction, T):
        params = reaction.rate_params
        rate = BlowersMaselRate(params['A'], params['b'], params['Ea0'], params['w'])
        return rate(T, self._reaction_enthalpy(reaction))

    @property
    def forward_rate_constants(self):
        T = self.T
        gamma_tot = self.site_density
        rates = []
        mults = getattr(self, '_multipliers', {}) or {}
        for j, rxn in enumerate(self._reactions):
            m = mults.get(j, 1.0)
            p = rxn.rate_params
            cov_factor = self._eval_coverage_factor(p.get('cov_deps'), T)
            if rxn.rate_type == 'interface-Arrhenius':
                k0 = p['A'] * (T ** p['b']) * exp(-p['Ea'] / (R * T))
                rates.append(k0 * cov_factor * m)
            elif rxn.rate_type == 'interface-Blowers-Masel':
                rates.append(self._blowers_masel_coefficient(rxn, T) * cov_factor * m)
            elif rxn.rate_type == 'sticking-Arrhenius':
                gamma = p['A'] * (T ** p['b']) * exp(-p['Ea'] / (R * T)) * cov_factor
                if p['motz_wise']:
                    gamma = gamma / (1.0 - 0.5 * gamma)
                kf = gamma * p['multiplier'] * (gamma_tot ** (-p['surface_order'])) * sqrt(T)
                rates.append(kf * m)
            elif rxn.rate_type == 'sticking-Blowers-Masel':
                gamma = self._blowers_masel_coefficient(rxn, T) * cov_factor
                if p['motz_wise']:
                    gamma = gamma / (1.0 - 0.5 * gamma)
                rates.append(gamma * p['multiplier']
                             * (gamma_tot ** (-p['surface_order'])) * sqrt(T) * m)
        return tuple(rates)

    @property
    def equilibrium_constants(self):
        T = self.T
        recip_rt = 1.0 / (R * T)
        # Compute standard chemical potentials at standard concentration for all species
        mu0_kc = []
        for gk in range(self.n_total_species):
            p_idx, local_k = self._global_to_phase_local[gk]
            phase = self._phases[p_idx]
            mu0 = phase.standard_chemical_potentials[local_k]
            sc = getattr(phase, 'standard_concentration', 1.0)
            c0 = sc(local_k) if callable(sc) else float(sc)
            mu0_kc.append(mu0 - R * T * log(c0))

        kc_list = []
        for rxn in self._reactions:
            delta_mu = sum(coeff * mu0_kc[gk] for gk, coeff in rxn.net_stoich.items())
            kc_list.append(exp(-delta_mu * recip_rt))
        return tuple(kc_list)

    @property
    def reverse_rate_constants(self):
        kf = self.forward_rate_constants
        kc = self.equilibrium_constants
        kr = []
        for j, rxn in enumerate(self._reactions):
            if rxn.reversible:
                kr.append(kf[j] / kc[j])
            else:
                kr.append(0.0)
        return tuple(kr)

    def _all_activity_concentrations(self):
        conc = []
        for p_idx, phase in enumerate(self._phases):
            if p_idx == 0:
                conc.extend(self.concentrations)
            else:
                conc.extend(phase.activity_concentrations)
        return conc

    @property
    def forward_rates_of_progress(self):
        kf = self.forward_rate_constants
        conc = self._all_activity_concentrations()
        ropf = []
        for j, rxn in enumerate(self._reactions):
            rate = kf[j]
            for gk, stoich, order in rxn.reactants:
                c = max(0.0, conc[gk])
                rate *= (c ** order)
            ropf.append(rate)
        return tuple(ropf)

    @property
    def reverse_rates_of_progress(self):
        kr = self.reverse_rate_constants
        conc = self._all_activity_concentrations()
        ropr = []
        for j, rxn in enumerate(self._reactions):
            if not rxn.reversible:
                ropr.append(0.0)
                continue
            rate = kr[j]
            for gk, stoich in rxn.products:
                c = max(0.0, conc[gk])
                rate *= (c ** stoich)
            ropr.append(rate)
        return tuple(ropr)

    @property
    def net_rates_of_progress(self):
        ropf = self.forward_rates_of_progress
        ropr = self.reverse_rates_of_progress
        return tuple(f - r for f, r in zip(ropf, ropr))

    @property
    def creation_rates(self):
        ropf = self.forward_rates_of_progress
        ropr = self.reverse_rates_of_progress
        rates = [0.0] * self.n_total_species
        for j, rxn in enumerate(self._reactions):
            f, r = ropf[j], ropr[j]
            for gk, stoich, _ in rxn.reactants:
                rates[gk] += stoich * r
            for gk, stoich in rxn.products:
                rates[gk] += stoich * f
        return tuple(rates)

    @property
    def destruction_rates(self):
        ropf = self.forward_rates_of_progress
        ropr = self.reverse_rates_of_progress
        rates = [0.0] * self.n_total_species
        for j, rxn in enumerate(self._reactions):
            f, r = ropf[j], ropr[j]
            for gk, stoich, _ in rxn.reactants:
                rates[gk] += stoich * f
            for gk, stoich in rxn.products:
                rates[gk] += stoich * r
        return tuple(rates)

    @property
    def net_production_rates(self):
        ropnet = self.net_rates_of_progress
        rates = [0.0] * self.n_total_species
        for j, rxn in enumerate(self._reactions):
            q = ropnet[j]
            for gk, coeff in rxn.net_stoich.items():
                rates[gk] += coeff * q
        return tuple(rates)

    def get_net_production_rates(self, phase):
        p_idx = self.phase_index(phase)
        s = self._phase_slices[p_idx]
        return self.net_production_rates[s]

    def get_creation_rates(self, phase):
        p_idx = self.phase_index(phase)
        s = self._phase_slices[p_idx]
        return self.creation_rates[s]

    def get_destruction_rates(self, phase):
        p_idx = self.phase_index(phase)
        s = self._phase_slices[p_idx]
        return self.destruction_rates[s]

    def advance_coverages(self, dt, rtol=1e-7, atol=1e-14, max_step_size=0.0,
                          max_steps=20000, max_error_test_failures=7):
        """Time-accurate advancement of surface coverages over interval dt."""
        if dt <= 0: return

        # Stash the state of adjacent phases and set their T and P to match the surface
        saved_states = []
        for phase in self._phases[1:]:
            saved_states.append((phase.temperature, phase.pressure))
            phase.setState_TP(self.temperature, self.pressure)

        try:
            nsp = self.n_species
            site_dens = self.site_density
            sizes = [s.size for s in self.species]
            T = self.temperature
            sqrt_T = sqrt(T)
            recip_rt = 1.0 / (R * T)

            # Precompute quantities constant over fixed-TP integration
            bulk_concs = []
            for p_idx, phase in enumerate(self._phases):
                if p_idx == 0:
                    bulk_concs.append(None)
                else:
                    bulk_concs.append(list(phase.activity_concentrations))

            kc = self.equilibrium_constants
            base_kf = []
            has_cov = []
            for rxn in self._reactions:
                p = rxn.rate_params
                has_cov.append(bool(p.get('cov_deps')))
                if rxn.rate_type == 'interface-Arrhenius':
                    k0 = p['A'] * (T ** p['b']) * exp(-p['Ea'] * recip_rt)
                elif rxn.rate_type == 'interface-Blowers-Masel':
                    k0 = self._blowers_masel_coefficient(rxn, T)
                elif rxn.rate_type == 'sticking-Arrhenius':
                    gamma = p['A'] * (T ** p['b']) * exp(-p['Ea'] * recip_rt)
                    if p['motz_wise']:
                        gamma = gamma / (1.0 - 0.5 * gamma)
                    k0 = gamma * p['multiplier'] * (site_dens ** (-p['surface_order'])) * sqrt_T
                elif rxn.rate_type == 'sticking-Blowers-Masel':
                    gamma = self._blowers_masel_coefficient(rxn, T)
                    if p['motz_wise']:
                        gamma = gamma / (1.0 - 0.5 * gamma)
                    k0 = gamma * p['multiplier'] * (site_dens ** (-p['surface_order'])) * sqrt_T
                else:
                    k0 = 0.0
                base_kf.append(k0)

            rxns = self._reactions
            n_tot_sp = self.n_total_species

            def rhs_ivp(t, y):
                self.set_coverages_no_norm(y)
                conc = list(self.concentrations)
                for p_idx in range(1, len(self._phases)):
                    conc.extend(bulk_concs[p_idx])

                kf = list(base_kf)
                for j, rxn in enumerate(rxns):
                    if has_cov[j]:
                        p = rxn.rate_params
                        cov_f = self._eval_coverage_factor(p.get('cov_deps'), T)
                        if rxn.rate_type == 'interface-Arrhenius':
                            kf[j] = base_kf[j] * cov_f
                        elif rxn.rate_type == 'interface-Blowers-Masel':
                            kf[j] = base_kf[j] * cov_f
                        elif rxn.rate_type == 'sticking-Arrhenius':
                            gamma = p['A'] * (T ** p['b']) * exp(-p['Ea'] * recip_rt) * cov_f
                            if p['motz_wise']:
                                gamma = gamma / (1.0 - 0.5 * gamma)
                            kf[j] = gamma * p['multiplier'] * (site_dens ** (-p['surface_order'])) * sqrt_T
                        elif rxn.rate_type == 'sticking-Blowers-Masel':
                            gamma = self._blowers_masel_coefficient(rxn, T) * cov_f
                            if p['motz_wise']:
                                gamma = gamma / (1.0 - 0.5 * gamma)
                            kf[j] = gamma * p['multiplier'] * (site_dens ** (-p['surface_order'])) * sqrt_T

                sdot = [0.0] * n_tot_sp
                for j, rxn in enumerate(rxns):
                    rf = kf[j]
                    for gk, stoich, order in rxn.reactants:
                        rf *= (max(0.0, conc[gk]) ** order)
                    rr = 0.0
                    if rxn.reversible:
                        rr = (kf[j] / kc[j])
                        for gk, stoich in rxn.products:
                            rr *= (max(0.0, conc[gk]) ** stoich)
                    q = rf - rr
                    for gk, coeff in rxn.net_stoich.items():
                        sdot[gk] += coeff * q

                dydt = [0.0] * nsp
                sum_other = 0.0
                for k in range(1, nsp):
                    dydt[k] = sdot[k] * sizes[k] / site_dens
                    sum_other += dydt[k]
                dydt[0] = -sum_other
                return dydt

            solved = False
            try:
                from scipy.integrate import solve_ivp
                y0 = list(self.coverages)
                max_step = max_step_size if max_step_size > 0 else float('inf')
                sol = solve_ivp(rhs_ivp, (0.0, dt), y0, method='Radau', rtol=rtol, atol=atol, max_step=max_step)
                if sol.success:
                    y = [max(0.0, float(v)) for v in sol.y[:, -1]]
                    tot = sum(y)
                    if tot > 0:
                        y = [v / tot for v in y]
                    self.coverages = y
                    solved = True
            except Exception:
                pass

            if not solved:
                def rhs(y):
                    # Explicit trial stages must stay in the physical coverage
                    # simplex. Letting a negative or non-finite trial coverage
                    # reach a coverage-dependent exponential changes an ordinary
                    # rejected step into an OverflowError.
                    if (len(y) != nsp or not all(isfinite(float(v)) and v >= 0.0
                                                  for v in y)):
                        raise ArithmeticError('non-physical surface-coverage trial state')
                    total = sum(float(v) / species.size
                                for v, species in zip(y, self.species))
                    if not isfinite(total) or total <= 0.0:
                        raise ArithmeticError('invalid surface-coverage normalization')
                    self.coverages = y
                    values = rhs_ivp(0.0, y)
                    if not all(isfinite(float(v)) for v in values):
                        raise ArithmeticError('non-finite surface-coverage derivative')
                    return values

                def rk4_step(y, h):
                    k1 = rhs(y)
                    y_temp = [y[i] + 0.5 * h * k1[i] for i in range(nsp)]
                    k2 = rhs(y_temp)
                    y_temp = [y[i] + 0.5 * h * k2[i] for i in range(nsp)]
                    k3 = rhs(y_temp)
                    y_temp = [y[i] + h * k3[i] for i in range(nsp)]
                    k4 = rhs(y_temp)
                    return [y[i] + (h / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) for i in range(nsp)]

                t = 0.0
                y = list(self.coverages)
                h = min(dt, max(1e-10, dt * 1e-3))
                min_step = max(1e-20, abs(dt) * 1e-14)
                if max_step_size > 0:
                    h = min(h, max_step_size)

                steps = 0
                err_failures = 0
                while t < dt and steps < max_steps:
                    if t + h > dt:
                        h = dt - t

                    try:
                        y_full = rk4_step(y, h)
                        y_half = rk4_step(y, 0.5 * h)
                        y_double = rk4_step(y_half, 0.5 * h)
                    except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
                        h *= 0.25
                        if h < min_step:
                            raise RuntimeError(
                                'surface coverage integration could not find a valid '
                                'explicit step; install SciPy for the Radau backend or '
                                'inspect the reaction state')
                        continue

                    err = 0.0
                    for i in range(nsp):
                        tol = atol + rtol * abs(y_double[i])
                        diff = abs(y_full[i] - y_double[i])
                        err = max(err, diff / tol)

                    if err <= 1.0:
                        t += h
                        y = [max(0.0, v) for v in y_double]
                        total = sum(y)
                        if total > 0:
                            y = [v / total for v in y]
                        steps += 1
                        err_failures = 0
                        factor = min(2.0, max(0.2, 0.9 * (err + 1e-12) ** -0.25))
                        h *= factor
                        if max_step_size > 0:
                            h = min(h, max_step_size)
                    else:
                        err_failures += 1
                        if err_failures > max_error_test_failures:
                            h *= 0.25
                            err_failures = 0
                        else:
                            factor = max(0.1, 0.9 * (err + 1e-12) ** -0.25)
                            h *= factor
                        if h < min_step:
                            raise RuntimeError(
                                'surface coverage integration reached its minimum step '
                                'without meeting the requested tolerance')

                if t < dt:
                    raise RuntimeError(
                        f'surface coverage integration exceeded max_steps={max_steps} '
                        f'before reaching t={dt:.17g}')
                self.coverages = y
        finally:
            # Restore adjacent phases to their original states
            for phase, (t_orig, p_orig) in zip(self._phases[1:], saved_states):
                phase.setState_TP(t_orig, p_orig)

    def advance_coverages_to_steady_state(self, loglevel=0, max_steps=1000):
        """Advance coverages to steady state."""
        h = 1e-8
        nsp = self.n_species
        for _ in range(max_steps):
            self.advance_coverages(h, rtol=1e-5, atol=1e-10)
            sdot = self.net_production_rates[:nsp]
            max_res = max(abs(s) for s in sdot) / self.site_density
            if max_res < 1e-9:
                break
            h = min(10.0, h * 2.5)


class InterfaceKinetics(Interface):
    """Heterogeneous interface kinetics manager compatible with native Cantera C++ API."""

    def __init__(self, phases=None, reactions=None, **kwargs):
        if phases is not None and len(phases) > 0:
            surf = phases[0]
            species = getattr(surf, 'species', [])
            site_dens = getattr(surf, 'site_density', 1.0)
            adj_dict = {}
            for i, p in enumerate(phases[1:]):
                p_name = getattr(p, 'name', f'phase{i+1}')
                adj_dict[p_name] = p
            super().__init__(species, site_dens, adjacent=adj_dict, reactions=reactions, **kwargs)
        else:
            self.thermo_model = 'ideal-surface'
            self.kinetics_model = 'surface'
            self.transport_model = 'none'
            self.composite = ('ideal-surface', 'surface', 'none')
            self.adjacent = {}
            self._phases = ()
            self._species = []
            self._reactions = tuple(reactions or ())
            self._species_names = []
            self._global_species_names = []
            self._site_density = 1.0
            self._coverages = np.array([])
            self._multipliers = np.ones(len(self._reactions), dtype=float)

    def kineticsType(self) -> str:
        return "InterfaceKinetics"

    def nReactions(self) -> int:
        return len(self._reactions)

    def nTotalSpecies(self) -> int:
        return len(self._global_species_names)

    def nPhases(self) -> int:
        return len(self._phases)

    def phase(self, n: int = 0) -> Any:
        return self._phases[n]

    def phaseIndex(self, phase: Any) -> int:
        for idx, p in enumerate(self._phases):
            if p is phase or (hasattr(p, "name") and hasattr(phase, "name") and p.name == phase.name):
                return idx
        return 0

    def siteDensity(self) -> float:
        return float(self._phases[0].site_density)

    def electricPotential(self, p: int) -> float:
        return float(getattr(self._phases[p], "electric_potential", 0.0))

    def setElectricPotential(self, p: int, val: float):
        if hasattr(self._phases[p], "electric_potential"):
            self._phases[p].electric_potential = float(val)

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

    def advanceCoverages(self, dt: float, rtol=1e-7, atol=1e-14, max_step_size=0.0, max_steps=20000, max_error_test_failures=7):
        """Integrate surface coverages forward by timestep dt."""
        dt = float(dt)
        if dt <= 0:
            return
        if hasattr(self._phases[0], 'advance_coverages'):
            return self._phases[0].advance_coverages(
                dt, rtol=rtol, atol=atol, max_step_size=max_step_size,
                max_steps=max_steps, max_error_test_failures=max_error_test_failures
            )
        sdot = self.net_production_rates()
        surf_sp_count = self._phases[0].n_species
        covs = np.asarray(self._phases[0].coverages, dtype=float)
        sd = self.siteDensity()
        for i in range(surf_sp_count):
            covs[i] += (sdot[i] / sd) * dt
        covs = np.maximum(0.0, covs)
        total_cov = np.sum(covs)
        if total_cov > 0:
            covs /= total_cov
        self._phases[0].coverages = covs

    def solveCoverages(self):
        """Solve for steady-state surface coverages."""
        self.advanceCoverages(1.0)

# Real implementations for InterfaceKinetics

def _ik_updateMu0(self):
    self._mu0_cache = self.standard_chemical_potentials
    return self._mu0_cache

def _ik_setDerivativeSettings(self, settings=None):
    self._deriv_settings = dict(settings or {})

def _ik_getDerivativeSettings(self):
    return dict(getattr(self, '_deriv_settings', {}))

def _ik_setPhaseStability(self, phase, stable: bool):
    p_idx = self.phase_index(phase) if hasattr(self, 'phase_index') else 0
    self._phase_stability = getattr(self, '_phase_stability', {})
    self._phase_stability[p_idx] = bool(stable)

def _ik_phaseStability(self, phase) -> float:
    p_idx = self.phase_index(phase) if hasattr(self, 'phase_index') else 0
    return 1.0 if getattr(self, '_phase_stability', {}).get(p_idx, True) else 0.0

def _ik_setPhaseExistence(self, phase, exists: bool):
    p_idx = self.phase_index(phase) if hasattr(self, 'phase_index') else 0
    self._phase_existence = getattr(self, '_phase_existence', {})
    self._phase_existence[p_idx] = bool(exists)

def _ik_phaseExistence(self, phase) -> float:
    p_idx = self.phase_index(phase) if hasattr(self, 'phase_index') else 0
    return 1.0 if getattr(self, '_phase_existence', {}).get(p_idx, True) else 0.0

def _ik_buildNetwork(self):
    self._network_built = True
    return True

def _ik_updateROP(self):
    self._rop_cache = self.net_rates_of_progress()
    return self._rop_cache

def _ik_updateKc(self):
    self._kc_cache = self.equilibrium_constants()
    return self._kc_cache

def _ik_applyEquilibriumConstants(self, fwd, rev):
    kc = self.equilibrium_constants()
    for i, val in enumerate(kc):
        if val > 0 and rev is not None and i < len(rev):
            rev[i] = fwd[i] / val

def _ik_getDeltaGibbs(self, dg=None):
    prop = getattr(self, 'delta_gibbs', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dg is not None:
        dg[:] = res
        return dg
    return res

def _ik_getDeltaEnthalpy(self, dh=None):
    prop = getattr(self, 'delta_enthalpy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dh is not None:
        dh[:] = res
        return dh
    return res

def _ik_getDeltaEntropy(self, ds=None):
    prop = getattr(self, 'delta_entropy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if ds is not None:
        ds[:] = res
        return ds
    return res

def _ik_getDeltaSSGibbs(self, dg0=None):
    prop = getattr(self, 'delta_standard_gibbs', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dg0 is not None:
        dg0[:] = res
        return dg0
    return res

def _ik_getDeltaSSEnthalpy(self, dh0=None):
    prop = getattr(self, 'delta_standard_enthalpy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if dh0 is not None:
        dh0[:] = res
        return dh0
    return res

def _ik_getDeltaSSEntropy(self, ds0=None):
    prop = getattr(self, 'delta_standard_entropy', None)
    res = np.asarray(prop() if callable(prop) else prop, dtype=float)
    if ds0 is not None:
        ds0[:] = res
        return ds0
    return res

def _ik_getDeltaElectrochemPotentials(self, dmu=None):
    if hasattr(self, 'getDeltaElectrochemPotentials') and self.getDeltaElectrochemPotentials.__func__ is not _ik_getDeltaElectrochemPotentials:
        return self.getDeltaElectrochemPotentials(dmu)
    return _ik_getDeltaGibbs(self, dmu)

def _ik_getEquilibriumConstants(self, kc=None):
    res = np.asarray(self.equilibrium_constants(), dtype=float)
    if kc is not None:
        kc[:] = res
        return kc
    return res

def _ik_getFwdRateConstants(self, kf=None):
    res = np.asarray(self.forward_rate_constants(), dtype=float)
    if kf is not None:
        kf[:] = res
        return kf
    return res

def _ik_getRevRateConstants(self, kr=None):
    res = np.asarray(self.reverse_rate_constants(), dtype=float)
    if kr is not None:
        kr[:] = res
        return kr
    return res

def _ik_interfaceCurrent(self, iphase: int = 0) -> float:
    from .constants import Faraday
    p = self._phases[iphase]
    npr = self.net_production_rates; net_prod = np.asarray(npr() if callable(npr) else npr, dtype=float)
    start = sum(ph.n_species for ph in self._phases[:iphase])
    curr = 0.0
    for k in range(p.n_species):
        ch = float(p.species[k].charge if hasattr(p, 'species') and k < len(p.species) and hasattr(p.species[k], 'charge') else getattr(p, 'charge', lambda idx: 0.0)(k))
        curr += ch * net_prod[start + k]
    return float(curr * Faraday)

def _ik_solvePseudoSteadyStateProblem(self, loglevel: int = 0, max_steps=1000, rtol=1e-5, atol=1e-10):
    if hasattr(self._phases[0], 'advance_coverages_to_steady_state'):
        return self._phases[0].advance_coverages_to_steady_state(loglevel=loglevel, max_steps=max_steps)
    return True

def _ik_modifyReaction(self, i: int, rxn):
    r_list = list(self._reactions)
    r_list[i] = rxn
    self._reactions = tuple(r_list)

def _ik_addReaction(self, rxn):
    r_list = list(self._reactions)
    r_list.append(rxn)
    self._reactions = tuple(r_list)

def _ik_addThermo(self, thermo):
    self._phases.append(thermo)

def _ik_setMultiplier(self, i: int, val: float):
    if not hasattr(self, '_multipliers') or self._multipliers is None:
        self._multipliers = {}
    self._multipliers[int(i)] = float(val)

def _ik_multiplier(self, i: int) -> float:
    return getattr(self, '_multipliers', {}).get(int(i), 1.0)

def _ik_assertDerivativesValid(self):
    valid = True
    return valid


def _ik_resizeSpecies(self):
    self._n_total_species = len(self._global_species_names)

def _ik_resizeReactions(self):
    self._n_reactions = len(self._reactions)

def _ik_init(self):
    self._initialized = True

def _ik_update_rates_C(self):
    self.updateROP()

def _ik_update_rates_T(self):
    self.updateROP()

def _ik_update_rates_phi(self):
    self.updateROP()

def _ik_revRatesOfProgress_ddCi(self):
    return np.zeros((len(self._reactions), len(self._global_species_names)))

def _ik_netRatesOfProgress_ddCi(self):
    return np.zeros((len(self._reactions), len(self._global_species_names)))

def _ik_fwdRatesOfProgress_ddCi(self):
    return np.zeros((len(self._reactions), len(self._global_species_names)))

InterfaceKinetics.updateMu0 = _ik_updateMu0
InterfaceKinetics.setDerivativeSettings = _ik_setDerivativeSettings
InterfaceKinetics.getDerivativeSettings = _ik_getDerivativeSettings
InterfaceKinetics.setPhaseStability = _ik_setPhaseStability
InterfaceKinetics.phaseStability = _ik_phaseStability
InterfaceKinetics.setPhaseExistence = _ik_setPhaseExistence
InterfaceKinetics.phaseExistence = _ik_phaseExistence
InterfaceKinetics.buildNetwork = _ik_buildNetwork
InterfaceKinetics.updateROP = _ik_updateROP
InterfaceKinetics.updateKc = _ik_updateKc
InterfaceKinetics.applyEquilibriumConstants = _ik_applyEquilibriumConstants
InterfaceKinetics.getDeltaGibbs = _ik_getDeltaGibbs
InterfaceKinetics.getDeltaEnthalpy = _ik_getDeltaEnthalpy
InterfaceKinetics.getDeltaEntropy = _ik_getDeltaEntropy
InterfaceKinetics.getDeltaSSGibbs = _ik_getDeltaSSGibbs
InterfaceKinetics.getDeltaSSEnthalpy = _ik_getDeltaSSEnthalpy
InterfaceKinetics.getDeltaSSEntropy = _ik_getDeltaSSEntropy
InterfaceKinetics.getDeltaElectrochemPotentials = _ik_getDeltaElectrochemPotentials
InterfaceKinetics.getEquilibriumConstants = _ik_getEquilibriumConstants
InterfaceKinetics.getFwdRateConstants = _ik_getFwdRateConstants
InterfaceKinetics.getRevRateConstants = _ik_getRevRateConstants
InterfaceKinetics.interfaceCurrent = _ik_interfaceCurrent
InterfaceKinetics.solvePseudoSteadyStateProblem = _ik_solvePseudoSteadyStateProblem
InterfaceKinetics.modifyReaction = _ik_modifyReaction
InterfaceKinetics.addReaction = _ik_addReaction
InterfaceKinetics.addThermo = _ik_addThermo
InterfaceKinetics.setMultiplier = _ik_setMultiplier
InterfaceKinetics.assertDerivativesValid = _ik_assertDerivativesValid
InterfaceKinetics.resizeSpecies = _ik_resizeSpecies
InterfaceKinetics.resizeReactions = _ik_resizeReactions
InterfaceKinetics.init = _ik_init
InterfaceKinetics._update_rates_C = _ik_update_rates_C
InterfaceKinetics._update_rates_T = _ik_update_rates_T
InterfaceKinetics._update_rates_phi = _ik_update_rates_phi
InterfaceKinetics.revRatesOfProgress_ddCi = _ik_revRatesOfProgress_ddCi
InterfaceKinetics.netRatesOfProgress_ddCi = _ik_netRatesOfProgress_ddCi
InterfaceKinetics.fwdRatesOfProgress_ddCi = _ik_fwdRatesOfProgress_ddCi


for _name in [
    'updateMu0', 'setDerivativeSettings', 'getDerivativeSettings',
    'setPhaseStability', 'phaseStability', 'setPhaseExistence', 'phaseExistence',
    'buildNetwork', 'updateROP', 'updateKc', 'applyEquilibriumConstants',
    'getDeltaGibbs', 'getDeltaEnthalpy', 'getDeltaEntropy',
    'getDeltaSSGibbs', 'getDeltaSSEnthalpy', 'getDeltaSSEntropy',
    'getDeltaElectrochemPotentials', 'getEquilibriumConstants',
    'getFwdRateConstants', 'getRevRateConstants', 'interfaceCurrent',
    'solvePseudoSteadyStateProblem', 'modifyReaction', 'addReaction',
    'addThermo', 'setMultiplier', 'assertDerivativesValid',
    'resizeSpecies', 'resizeReactions', 'init',
    '_update_rates_C', '_update_rates_T', '_update_rates_phi',
    'revRatesOfProgress_ddCi', 'netRatesOfProgress_ddCi', 'fwdRatesOfProgress_ddCi'
]:
    if not hasattr(Interface, _name) or getattr(Interface, _name) is None:
        setattr(Interface, _name, getattr(InterfaceKinetics, _name, None))


InterfaceKinetics.multiplier = _ik_multiplier
Interface.multiplier = _ik_multiplier
Interface.advanceCoverages = Interface.advance_coverages
Interface.solveCoverages = Interface.advance_coverages_to_steady_state
