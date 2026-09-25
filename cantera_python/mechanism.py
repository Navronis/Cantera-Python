"""Basic, native-free mechanism document validation and construction data.

The input mapping mirrors the ``species`` and ``reactions`` portions of a
Cantera YAML document after YAML decoding. Sources: ``Species::newSpecies``
in ``src/thermo/Species.cpp``, ``parseReactionEquation`` and
``Reaction::checkBalance`` in ``src/kinetics/Reaction.cpp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from math import isclose
import os
from pathlib import Path
from typing import Mapping

from .elements import COMMON_ELEMENTS, ElementRegistry
from .reaction_equation import ParsedReactionEquation, parse_reaction_equation


class MechanismError(ValueError):
    """A basic species/reaction mechanism document is invalid."""


@dataclass(frozen=True)
class MechanismSpecies:
    name: str
    composition: dict[str, float]
    molecular_weight: float
    charge: float = 0.0
    size: float = 1.0
    thermo: dict | None = None
    transport: dict | None = None
    equation_of_state: dict | None = None
    input_data: dict | None = None


@dataclass(frozen=True)
class MechanismReaction:
    equation: ParsedReactionEquation
    duplicate: bool = False
    rate_data: dict | None = None
    section: str = 'reactions'
    input_data: dict | None = None


@dataclass(frozen=True)
class MechanismPhase:
    name: str
    thermo: str | None
    kinetics: str | None
    transport: str | None
    species_names: tuple[str, ...]
    reaction_sections: tuple[str, ...]
    state: dict | None = None
    reaction_rules: tuple[str, ...] = ()
    skip_undeclared_third_bodies: bool = False
    explicit_third_body_duplicates: str = 'warn'
    adjacent_phase_names: tuple[str, ...] = ()
    site_density: object = None
    input_data: dict | None = None
    species: tuple[MechanismSpecies, ...] = ()


@dataclass(frozen=True)
class Mechanism:
    species: tuple[MechanismSpecies, ...]
    reactions: tuple[MechanismReaction, ...]
    # Document-level unit system (YAML top-level `units` mapping), needed so
    # rate constants constructed later convert to SI exactly as pinned
    # Reaction.cpp does when parsing the same document.
    units: dict | None = None
    phases: tuple[MechanismPhase, ...] = ()
    reaction_sections: dict[str, tuple[MechanismReaction, ...]] | None = None
    input_data: dict | None = None

    @property
    def species_names(self):
        return tuple(item.name for item in self.species)

    def phase(self, name=None):
        if not self.phases:
            if name is not None:
                raise MechanismError(f'Unknown phase {name!r}; document defines no phases')
            return None
        if name is None:
            if len(self.phases) != 1:
                raise MechanismError('phase_name is required for a document with multiple phases')
            return self.phases[0]
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise MechanismError(f'Unknown phase {name!r}; available phases: {[p.name for p in self.phases]}')

    def for_phase(self, name=None):
        phase = self.phase(name)
        if phase is None:
            return self
        selected = set(phase.species_names)
        allowed = set(selected)
        for adjacent_name in phase.adjacent_phase_names:
            adjacent = self.phase(adjacent_name)
            allowed.update(adjacent.species_names)
        if phase.species:
            phase_species_dict = {s.name: s for s in phase.species}
            for adjacent_name in phase.adjacent_phase_names:
                adj = self.phase(adjacent_name)
                if adj and adj.species:
                    for s in adj.species:
                        phase_species_dict.setdefault(s.name, s)
                elif adj:
                    for s in self.species:
                        if s.name in adj.species_names:
                            phase_species_dict.setdefault(s.name, s)
            species = tuple(phase_species_dict.values())
        else:
            species = tuple(s for s in self.species if s.name in allowed)
            if len(species) != len(allowed):
                missing = sorted(allowed - {s.name for s in species})
                raise MechanismError(f'Phase {phase.name!r} references unknown species: {missing}')
        section_rules = tuple(zip(phase.reaction_sections, phase.reaction_rules))
        def in_phase(names):
            return all(name in allowed or name == 'M' or name.startswith('(+')
                       for name in names)
        reactions = []
        for section, rule in section_rules:
            for reaction in (self.reaction_sections or {}).get(section, ()):
                declared = (in_phase(reaction.equation.reactants)
                            and in_phase(reaction.equation.products))
                if rule == 'declared-species' and not declared:
                    continue
                if rule == 'all' and not declared:
                    names = set(reaction.equation.reactants) | set(reaction.equation.products)
                    missing = sorted(name for name in names
                                     if name not in selected and name != 'M'
                                     and not name.startswith('(+'))
                    raise MechanismError(
                        f'Reaction section {section!r} contains undeclared species: {missing}')
                reactions.append(reaction)
        reactions = tuple(reactions)
        sections = set(phase.reaction_sections)
        return Mechanism(
            species=species,
            reactions=reactions,
            units=self.units,
            phases=(phase,),
            reaction_sections={
                section: tuple(r for r in reactions if r.section == section)
                for section in sections
            },
            input_data=self.input_data,
        )

    def to_dict(self):
        """Return a self-contained YAML-compatible mechanism mapping."""
        # Start with user-defined top-level metadata. Reconstructed scientific
        # sections below replace their source counterparts so a phase-scoped
        # mechanism remains self-contained while arbitrary nested metadata is
        # retained.
        source = deepcopy(self.input_data or {})
        dynamic_sections = {'units', 'phases', 'species'}
        dynamic_sections.update((self.reaction_sections or {}).keys())
        dynamic_sections.update(
            str(key) for key, value in source.items()
            if isinstance(value, (list, tuple)) and value
            and all(isinstance(item, Mapping) and 'equation' in item
                    for item in value)
        )
        document = {
            key: value for key, value in source.items()
            if key not in dynamic_sections
        }
        if self.units:
            document['units'] = dict(self.units)
        if self.phases:
            phase_records = []
            for phase in self.phases:
                record = deepcopy(phase.input_data or {})
                record['name'] = phase.name
                if phase.thermo is not None:
                    record['thermo'] = phase.thermo
                if phase.kinetics is not None:
                    record['kinetics'] = phase.kinetics
                if phase.transport is not None:
                    record['transport'] = phase.transport
                record['species'] = list(phase.species_names)
                if not phase.reaction_sections:
                    record['reactions'] = 'none'
                elif (phase.reaction_sections == ('reactions',)
                      and phase.reaction_rules == ('all',)):
                    record['reactions'] = 'all'
                else:
                    record['reactions'] = [
                        {section: rule} if rule != 'all' else section
                        for section, rule in zip(phase.reaction_sections,
                                                 phase.reaction_rules)
                    ]
                if phase.state:
                    record['state'] = dict(phase.state)
                if phase.skip_undeclared_third_bodies:
                    record['skip-undeclared-third-bodies'] = True
                if phase.explicit_third_body_duplicates != 'warn':
                    record['explicit-third-body-duplicates'] = phase.explicit_third_body_duplicates
                if phase.adjacent_phase_names:
                    record['adjacent-phases'] = list(phase.adjacent_phase_names)
                if phase.site_density is not None:
                    record['site-density'] = phase.site_density
                phase_records.append(record)
            document['phases'] = phase_records
        species_records = []
        for species in self.species:
            record = deepcopy(species.input_data or {})
            record['name'] = species.name
            record['composition'] = dict(species.composition)
            if species.charge != -species.composition.get('E', 0.0):
                record['charge'] = species.charge
            if species.size != 1.0:
                record['size'] = species.size
            if species.thermo is not None:
                record['thermo'] = dict(species.thermo)
            if species.transport is not None:
                record['transport'] = dict(species.transport)
            if species.equation_of_state is not None:
                record['equation-of-state'] = deepcopy(species.equation_of_state)
            species_records.append(record)
        document['species'] = species_records
        sections = self.reaction_sections or {'reactions': self.reactions}
        for section, reactions in sections.items():
            records = []
            for reaction in reactions:
                record = deepcopy(reaction.input_data or {})
                record['equation'] = _format_reaction_equation(reaction.equation)
                if reaction.duplicate:
                    record['duplicate'] = True
                else:
                    record.pop('duplicate', None)
                if reaction.rate_data:
                    record.update(reaction.rate_data)
                records.append(record)
            document[section] = records
        return document

    def to_yaml(self, path=None):
        """Serialize this mechanism to YAML text and optionally write *path*."""
        try:
            from io import StringIO
            from ruamel.yaml import YAML
            stream = StringIO()
            YAML().dump(self.to_dict(), stream)
            text = stream.getvalue()
        except ImportError as error:
            raise MechanismError('ruamel.yaml is required for YAML serialization') from error
        if path is not None:
            from pathlib import Path
            Path(path).write_text(text, encoding='utf-8')
        return text


def _format_reaction_equation(equation):
    def side(values):
        terms = []
        for name, coefficient in values.items():
            if coefficient == 1.0:
                terms.append(name)
            else:
                terms.append(f'{coefficient:g} {name}')
        return ' + '.join(terms)
    arrow = '<=>' if equation.reversible else '=>'
    return f'{side(equation.reactants)} {arrow} {side(equation.products)}'


def _as_document(data, search_paths=None):
    from pathlib import Path
    if isinstance(data, (str, Path)):
        try:
            p = Path(data)
            if p.is_file():
                data = p.read_text(encoding='utf-8', errors='ignore')
            elif isinstance(data, str) and '\n' not in data:
                # Check candidate search directories for filename
                candidate_paths = list(search_paths or [])
                try:
                    from . import DATA_DIR
                    candidate_paths.append(DATA_DIR)
                except ImportError:
                    pass
                candidate_paths.append(Path(__file__).resolve().parent / 'data')
                candidate_paths.append(Path(__file__).resolve().parent / 'data')
                if 'CANTERA_DATA' in os.environ:
                    for p_env in os.environ['CANTERA_DATA'].split(os.pathsep):
                        candidate_paths.append(Path(p_env))
                for cp in candidate_paths:
                    cand = Path(cp) / data
                    if cand.is_file():
                        data = cand.read_text(encoding='utf-8', errors='ignore')
                        break
        except OSError:
            pass
    elif hasattr(data, 'read_text'):
        data = data.read_text(encoding='utf-8', errors='ignore')
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            try:
                try:
                    from ruamel.yaml import YAML
                    yaml_loader = YAML(typ='safe')
                    data = yaml_loader.load(data)
                except ImportError:
                    import yaml
                    data = yaml.safe_load(data)
            except Exception as error:
                raise MechanismError(f'Failed to parse mechanism text as JSON or YAML: {error}') from error
    if not isinstance(data, Mapping):
        raise MechanismError('Mechanism document must be a mapping or JSON/YAML mapping string')
    return data


def _species(record, elements):
    if not isinstance(record, Mapping):
        raise MechanismError('Each species entry must be a mapping')
    name = record.get('name')
    if name is False:
        name = 'NO'
    elif name is True:
        name = 'YES'
    elif not isinstance(name, str) or not name:
        raise MechanismError('Species must provide a nonempty name')
    composition = record.get('composition')
    if composition is None:
        composition = {}
    if not isinstance(composition, Mapping):
        raise MechanismError(f'Species {name!r} must provide an elemental composition mapping')
    try:
        normalized = {}
        for key, value in composition.items():
            k = 'NO' if key is False else ('YES' if key is True else str(key))
            normalized[k] = float(value)
        molecular_weight = elements.molecular_weight(normalized)
        # Cantera's pseudo-element E counts electrons: a positive E count is
        # a negative species charge, while E:-1 denotes a singly charged ion.
        inferred_charge = -normalized.get('E', 0.0)
        charge = float(record.get('charge', inferred_charge))
        size = float(record.get('size', 1.0))
    except (TypeError, ValueError) as error:
        raise MechanismError(f'Invalid data for species {name!r}: {error}') from error
    thermo = record.get('thermo')
    if thermo is not None and not isinstance(thermo, Mapping):
        raise MechanismError(f'Species {name!r} thermo must be a mapping when provided')
    transport = record.get('transport')
    if transport is not None and not isinstance(transport, Mapping):
        raise MechanismError(f'Species {name!r} transport must be a mapping when provided')
    eos = record.get('equation-of-state')
    return MechanismSpecies(name, normalized, molecular_weight, charge, size,
                            None if thermo is None else dict(thermo),
                            None if transport is None else dict(transport),
                            None if eos is None else dict(eos),
                            dict(record))


def _check_balance(reaction, species_by_name):
    reactant_atoms, product_atoms = {}, {}
    for side, totals in ((reaction.reactants, reactant_atoms), (reaction.products, product_atoms)):
        for name, coefficient in side.items():
            if name == 'M' or name.startswith('(+'):
                continue
            for element, count in species_by_name[name].composition.items():
                totals[element] = totals.get(element, 0.0) + coefficient * count
    for element in set(reactant_atoms) | set(product_atoms):
        left, right = reactant_atoms.get(element, 0.0), product_atoms.get(element, 0.0)
        scale = max(abs(left), abs(right))
        if scale and not isclose(left, right, rel_tol=1e-4, abs_tol=0.0):
            raise MechanismError(f'Unbalanced reaction for {element}: reactants={left}, products={right}')


def load_mechanism(data, *, search_paths=None, elements: ElementRegistry = COMMON_ELEMENTS) -> Mechanism:
    """Load basic species/reactions from a decoded YAML mapping or JSON text.

    The result preserves raw ``thermo`` and rate fields for later factory work.
    Supports resolving imported species from referenced files via ``search_paths``.
    """
    from pathlib import Path
    document = _as_document(data)

    # Collect candidate search directories
    paths = list(search_paths or [])
    default_data_dirs = [
        Path(__file__).resolve().parent / 'data',
        Path(__file__).resolve().parent / 'data',
    ]
    for d_dir in default_data_dirs:
        if d_dir.is_dir() and d_dir not in paths:
            paths.append(d_dir)
    try:
        p = Path(data)
        if p.is_file():
            parent = p.resolve().parent
            if parent not in paths:
                paths.append(parent)
            if parent.parent not in paths:
                paths.append(parent.parent)
        elif isinstance(data, str) and '\n' not in data:
            for p_dir in paths:
                cand = Path(p_dir) / data
                if cand.is_file():
                    parent = cand.resolve().parent
                    if parent not in paths:
                        paths.append(parent)
                    if parent.parent not in paths:
                        paths.append(parent.parent)
                    break
    except (OSError, TypeError):
        pass
    if hasattr(data, 'parent') and data.parent not in paths:
        paths.append(data.parent)
        if hasattr(data.parent, 'parent') and data.parent.parent not in paths:
            paths.append(data.parent.parent)
    # Default common data directories
    try:
        from . import DATA_DIR
        if DATA_DIR not in paths:
            paths.append(DATA_DIR)
    except ImportError:
        pass
    if 'CANTERA_DATA' in os.environ:
        for p_env in os.environ['CANTERA_DATA'].split(os.pathsep):
            p_path = Path(p_env)
            if p_path.is_dir() and p_path not in paths:
                paths.append(p_path)

    species_records = list(document.get('species', ()) or ())
    existing_names = set()
    for s in species_records:
        if isinstance(s, Mapping) and s.get('name'):
            s_name = 'NO' if s['name'] is False else ('YES' if s['name'] is True else str(s['name']))
            existing_names.add(s_name)

    # Resolve species from phases and external files (e.g. gri30.yaml/species or local sections)
    for phase in document.get('phases', ()) or ():
        if not isinstance(phase, Mapping):
            continue
        for spec_entry in phase.get('species', ()) or ():
            if isinstance(spec_entry, Mapping):
                for key, spec_list in spec_entry.items():
                    if '/' in key:
                        ext_file, section = key.split('/', 1)
                        target_path = None
                        for sp in paths:
                            candidate = Path(sp) / ext_file
                            if candidate.exists():
                                target_path = candidate
                                break
                        if target_path and target_path.exists():
                            ext_doc = _as_document(target_path)
                            ext_species = ext_doc.get('species', ()) or ()
                            if spec_list == 'all':
                                for es in ext_species:
                                    if isinstance(es, Mapping) and es.get('name') not in existing_names:
                                        species_records.append(es)
                                        existing_names.add(es['name'])
                            elif isinstance(spec_list, (list, tuple)):
                                ext_map = {es['name']: es for es in ext_species if isinstance(es, Mapping)}
                                for name in spec_list:
                                    if name in ext_map and name not in existing_names:
                                        species_records.append(ext_map[name])
                                        existing_names.add(name)
                    elif key in document:
                        sec_species = document.get(key, ()) or ()
                        if spec_list == 'all':
                            for es in sec_species:
                                if isinstance(es, Mapping) and es.get('name') not in existing_names:
                                    species_records.append(es)
                                    existing_names.add(es['name'])
                        elif isinstance(spec_list, (list, tuple)):
                            sec_map = {es['name']: es for es in sec_species if isinstance(es, Mapping)}
                            for name in spec_list:
                                if name in sec_map and name not in existing_names:
                                    species_records.append(sec_map[name])
                                    existing_names.add(name)

    reaction_sections = {}
    for section, records in document.items():
        if isinstance(records, (list, tuple)) and records and all(
                isinstance(record, Mapping) and 'equation' in record for record in records):
            reaction_sections[str(section)] = records
    if 'reactions' not in reaction_sections:
        records = document.get('reactions', ()) or ()
        if not isinstance(records, (list, tuple)):
            raise MechanismError('reactions must be a list')
        reaction_sections['reactions'] = records
    # Resolve phase reaction-section imports such as
    # ``gri30.yaml/reactions: all`` while preserving the full source key so
    # phase selection remains unambiguous.
    for phase_record in document.get('phases', ()) or ():
        if not isinstance(phase_record, Mapping):
            continue
        selection = phase_record.get('reactions', ()) or ()
        entries = selection if isinstance(selection, (list, tuple)) else (selection,)
        for entry in entries:
            keys = entry.keys() if isinstance(entry, Mapping) else (entry,)
            for key in keys:
                if not isinstance(key, str) or '/' not in key or key in reaction_sections:
                    continue
                ext_file, section = key.split('/', 1)
                target_path = next((Path(sp) / ext_file for sp in paths
                                    if (Path(sp) / ext_file).exists()), None)
                if target_path is None:
                    raise MechanismError(f'Cannot resolve imported reaction section {key!r}')
                ext_doc = _as_document(target_path)
                records = ext_doc.get(section, ()) or ()
                if not isinstance(records, (list, tuple)):
                    raise MechanismError(f'Imported reaction section {key!r} is not a list')
                reaction_sections[key] = records
    if not isinstance(species_records, (list, tuple)):
        raise MechanismError('species must be a list')
    species = tuple(_species(record, elements) for record in species_records)
    species_names = [item.name for item in species]
    duplicates = sorted({name for name in species_names if species_names.count(name) > 1})
    if duplicates:
        raise MechanismError(f'Duplicate species definitions: {duplicates}')
    species_by_name = {item.name: item for item in species}

    # Support electron alias between 'e' and 'E' if only one is defined
    if 'e' in species_by_name and 'E' not in species_by_name:
        species_by_name['E'] = species_by_name['e']
    elif 'E' in species_by_name and 'e' not in species_by_name:
        species_by_name['e'] = species_by_name['E']

    # A section selected with ``declared-species`` is filtered while loading;
    # reactions containing species outside the phase are omitted instead of
    # making the entire external section invalid (KineticsFactory::addReactions).
    section_rules = {}
    for phase_record in document.get('phases', ()) or ():
        if not isinstance(phase_record, Mapping):
            continue
        if 'reactions' not in phase_record:
            if str(phase_record.get('kinetics', 'none')).lower() not in ('none', 'kinetics'):
                section_rules.setdefault('reactions', set()).add('all')
            continue
        selection = phase_record.get('reactions')
        if isinstance(selection, str):
            if selection in ('all', 'declared-species'):
                section_rules.setdefault('reactions', set()).add(selection)
        elif isinstance(selection, (list, tuple)):
            for entry in selection:
                if isinstance(entry, str):
                    section_rules.setdefault(entry, set()).add('all')
                elif isinstance(entry, Mapping) and len(entry) == 1:
                    section, rule = next(iter(entry.items()))
                    section_rules.setdefault(str(section), set()).add(str(rule))
    filter_undeclared = {
        section for section, rules in section_rules.items()
        if rules and rules <= {'declared-species', 'none'}
    }

    parsed_sections = {}
    for section, records in reaction_sections.items():
        parsed = []
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get('equation'), str):
                raise MechanismError('Each reaction entry must provide an equation string')
            try:
                equation = parse_reaction_equation(
                    record['equation'], None if section in filter_undeclared else species_by_name)
            except ValueError as error:
                raise MechanismError(f'Invalid reaction {record["equation"]!r}: {error}') from error
            ordinary_names = {
                name for name in set(equation.reactants) | set(equation.products)
                if name != 'M' and not name.startswith('(+')
            }
            unknown = ordinary_names - set(species_by_name)
            if unknown and section in filter_undeclared:
                continue
            if unknown:
                raise MechanismError(
                    f'Invalid reaction {record["equation"]!r}: unknown species {sorted(unknown)}')
            _check_balance(equation, species_by_name)
            rate_data = {key: value for key, value in record.items()
                         if key not in {'equation', 'duplicate'}}
            parsed.append(MechanismReaction(
                equation, bool(record.get('duplicate', False)), rate_data or None,
                section, deepcopy(dict(record))))
        parsed_sections[section] = tuple(parsed)
    reactions = list(parsed_sections.get('reactions', ()))

    phases = []
    for record in document.get('phases', ()) or ():
        if not isinstance(record, Mapping) or not isinstance(record.get('name'), str):
            raise MechanismError('Each phase entry must provide a name')
        phase_species = []
        phase_species_objs = []
        for entry in record.get('species', ()) or ():
            if isinstance(entry, str):
                if entry == 'all':
                    for item in species:
                        phase_species.append(item.name)
                        phase_species_objs.append(item)
                elif entry in species_by_name:
                    phase_species.append(entry)
                    phase_species_objs.append(species_by_name[entry])
            elif isinstance(entry, Mapping):
                for key, names in entry.items():
                    sec_list = []
                    if '/' in key:
                        ext_file, section = key.split('/', 1)
                        for sp in paths:
                            cand = Path(sp) / ext_file
                            if cand.exists():
                                ext_doc = _as_document(cand)
                                sec_list = ext_doc.get(section if section != 'species' else 'species', ()) or ()
                                break
                    elif key in document:
                        sec_list = document.get(key, ()) or ()
                    sec_map = {es['name']: es for es in sec_list if isinstance(es, Mapping) and es.get('name')}
                    if names == 'all':
                        for s_name, s_data in sec_map.items():
                            phase_species.append(s_name)
                            phase_species_objs.append(_species(s_data, elements))
                    elif isinstance(names, (list, tuple)):
                        for name in names:
                            s_name = str(name)
                            if s_name in sec_map:
                                phase_species.append(s_name)
                                phase_species_objs.append(_species(sec_map[s_name], elements))
                            elif s_name in species_by_name:
                                phase_species.append(s_name)
                                phase_species_objs.append(species_by_name[s_name])
        selected_sections, selected_rules = [], []
        has_selection = 'reactions' in record
        selection = record.get('reactions')
        kinetics_model = str(record.get('kinetics', 'none')).lower()
        if not has_selection:
            if kinetics_model not in ('none', 'kinetics') and 'reactions' in reaction_sections:
                selected_sections, selected_rules = ['reactions'], ['all']
        elif selection in ('none', None):
            pass
        elif isinstance(selection, str):
            if selection not in ('all', 'declared-species'):
                raise MechanismError(
                    f'Unknown rule {selection!r} for the default reaction section')
            if 'reactions' not in reaction_sections:
                raise MechanismError(
                    f'Phase {record["name"]!r} references missing default reaction section')
            selected_sections, selected_rules = ['reactions'], [selection]
        elif isinstance(selection, (list, tuple)):
            for entry in selection:
                if isinstance(entry, str):
                    selected_sections.append(entry)
                    selected_rules.append('all')
                elif isinstance(entry, Mapping):
                    if len(entry) != 1:
                        raise MechanismError('Each reaction-section mapping must contain one item')
                    section, rule = next(iter(entry.items()))
                    if rule not in ('all', 'declared-species', 'none'):
                        raise MechanismError(
                            f'Unknown rule {rule!r} for reaction section {section!r}')
                    if rule != 'none':
                        selected_sections.append(str(section))
                        selected_rules.append(str(rule))
                else:
                    raise MechanismError('Reaction section entries must be strings or mappings')
        else:
            raise MechanismError('Phase reactions must be a string or list')
        unknown_sections = set(selected_sections) - set(reaction_sections)
        if unknown_sections:
            raise MechanismError(f'Phase {record["name"]!r} references unknown reaction sections: {sorted(unknown_sections)}')
        state = record.get('state')
        if state is not None and not isinstance(state, Mapping):
            raise MechanismError(f'Phase {record["name"]!r} state must be a mapping')
        skip_undeclared_third_bodies = record.get('skip-undeclared-third-bodies', False)
        if not isinstance(skip_undeclared_third_bodies, bool):
            raise MechanismError(
                f'Phase {record["name"]!r} skip-undeclared-third-bodies must be boolean')
        explicit_third_body_duplicates = record.get(
            'explicit-third-body-duplicates', 'warn')
        if explicit_third_body_duplicates not in {
                'warn', 'error', 'mark-duplicate', 'modify-efficiency'}:
            raise MechanismError(
                f"Invalid explicit-third-body-duplicates flag "
                f"{explicit_third_body_duplicates!r} for phase {record['name']!r}")
        adjacent_phase_names = record.get('adjacent-phases', ()) or ()
        if (not isinstance(adjacent_phase_names, (list, tuple))
                or not all(isinstance(name, str) for name in adjacent_phase_names)):
            raise MechanismError(
                f'Phase {record["name"]!r} adjacent-phases must be a list of names')
        phases.append(MechanismPhase(
            record['name'], record.get('thermo'), record.get('kinetics'), record.get('transport'),
            tuple(dict.fromkeys(phase_species)), tuple(dict.fromkeys(selected_sections)),
            dict(state) if state is not None else None, tuple(selected_rules),
            skip_undeclared_third_bodies, explicit_third_body_duplicates,
            tuple(adjacent_phase_names), record.get('site-density'),
            deepcopy(dict(record)),
            tuple(phase_species_objs)))
    phase_names = [phase.name for phase in phases]
    duplicate_phases = sorted({name for name in phase_names if phase_names.count(name) > 1})
    if duplicate_phases:
        raise MechanismError(f'Duplicate phase definitions: {duplicate_phases}')
    known_phases = set(phase_names)
    for phase in phases:
        unknown_adjacent = set(phase.adjacent_phase_names) - known_phases
        if unknown_adjacent:
            raise MechanismError(
                f'Phase {phase.name!r} references unknown adjacent phases: '
                f'{sorted(unknown_adjacent)}')
    units = document.get('units') or {}
    units = dict(units) if isinstance(units, Mapping) else {}
    return Mechanism(
        species=species,
        reactions=tuple(reactions),
        units=units,
        phases=tuple(phases),
        reaction_sections=parsed_sections,
        input_data=deepcopy(dict(document)),
    )
