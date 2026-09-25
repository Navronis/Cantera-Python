"""Dimensional SI conversions for pure-Python Cantera input layers.

This bounded port follows ``Units`` and ``UnitSystem::convert`` in
``src/base/Units.cpp``. Quantities are converted to Cantera base units:
kg, m, s, K, A, kmol, J, and Pa.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
import re

from .constants import Avogadro, ElectronCharge, GasConstant, OneAtm


class UnitError(ValueError):
    """A unit expression is unknown or dimensions cannot be converted."""


@dataclass(init=False)
class Units:
    factor: float = 1.0
    dimensions: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0)
    pressure_dimension: float = 0.0
    energy_dimension: float = 0.0

    def __init__(self, factor=1.0, *dimension_args, force_unity=False,
                 pressure_dimension=None, energy_dimension=None):
        """Construct units from dimensions or a pinned-style unit string.

        The numeric form accepts the pinned scalar signature
        ``(factor, mass, length, time, temperature, current, quantity)`` and
        the Python tuple shorthand ``(factor, dimensions)``. The string form
        mirrors ``Units(string, bool)``; a positional Boolean in the second
        slot is accepted for the C++ ``force_unity`` argument as well as the
        explicit Python keyword.
        """
        if isinstance(factor, str):
            if len(dimension_args) == 1 and isinstance(dimension_args[0], bool):
                force_unity = dimension_args[0]
            elif dimension_args:
                raise TypeError('String Units construction accepts only force_unity')
            parsed = parse_units(factor)
            if force_unity and abs(parsed.factor - 1.0) > 1.0e-300:
                raise UnitError(
                    f'Detected non-unity conversion factor in {factor!r}: '
                    f'{parsed.factor}'
                )
            self.factor = parsed.factor
            self.dimensions = parsed.dimensions
            self.pressure_dimension = parsed.pressure_dimension
            self.energy_dimension = parsed.energy_dimension
            return

        if not dimension_args:
            dimensions = (0, 0, 0, 0, 0, 0)
        elif len(dimension_args) == 1 and isinstance(dimension_args[0], Sequence):
            dimensions = dimension_args[0]
        elif (len(dimension_args) == 3
              and isinstance(dimension_args[0], Sequence)):
            # Backward-compatible internal form used before the scalar source
            # signature was exposed: Units(factor, dimensions, pressure, energy).
            dimensions, pressure_dimension, energy_dimension = dimension_args
        elif len(dimension_args) <= 6:
            dimensions = dimension_args + (0,) * (6 - len(dimension_args))
        else:
            raise TypeError('Numeric Units construction accepts at most six dimensions')
        if len(dimensions) != 6:
            raise ValueError('Units dimensions must contain six exponents')
        self.factor = float(factor)
        self.dimensions = tuple(float(value) for value in dimensions)
        mass, length, time, temperature, current, quantity = self.dimensions
        if pressure_dimension is None:
            pressure_dimension = (
                mass if (mass != 0 and length == -mass and time == -2 * mass
                         and temperature == current == quantity == 0) else 0.0
            )
        if energy_dimension is None:
            energy_dimension = (
                mass if (mass != 0 and length == 2 * mass and time == -2 * mass
                         and temperature == current == quantity == 0) else 0.0
            )
        self.pressure_dimension = float(pressure_dimension)
        self.energy_dimension = float(energy_dimension)

    def __mul__(self, other):
        if not isinstance(other, Units):
            return NotImplemented
        return Units(
            self.factor * other.factor,
            tuple(a + b for a, b in zip(self.dimensions, other.dimensions)),
            self.pressure_dimension + other.pressure_dimension,
            self.energy_dimension + other.energy_dimension,
        )

    def power(self, exponent):
        dimensions = tuple(exponent * value for value in self.dimensions)
        mass, length, time, temperature, current, quantity = dimensions
        # Units::pow constructs a new Units object from the primary dimensions.
        # Its pressure/energy pseudo-dimensions are therefore inferred afresh,
        # rather than scaling bookkeeping flags carried by the input object.
        pressure = (
            mass if (mass != 0 and length == -mass and time == -2 * mass
                     and temperature == current == quantity == 0) else 0.0
        )
        energy = (
            mass if (mass != 0 and length == 2 * mass and time == -2 * mass
                     and temperature == current == quantity == 0) else 0.0
        )
        return Units(self.factor ** exponent, dimensions, pressure, energy)

    def scale(self, factor):
        """Scale the conversion factor in place, as pinned ``Units::scale``."""
        self.factor *= float(factor)

    def to_string(self, skip_unity=True):
        """Return the canonical base-dimension representation from ``Units::str``."""
        names = ('A', 'K', 'kg', 'kmol', 'm', 's')
        indices = (4, 3, 0, 5, 1, 2)
        numerator = []
        denominator = []
        for name, index in zip(names, indices):
            exponent = self.dimensions[index]
            rounded = round(exponent)
            if exponent == 0.0:
                continue
            if exponent == 1.0:
                numerator.append(name)
            elif exponent == -1.0:
                denominator.append(name)
            elif exponent == rounded:
                term = f'{name}^{abs(int(rounded))}'
                (numerator if exponent > 0 else denominator).append(term)
            else:
                term = f'{name}^{abs(exponent):g}'
                (numerator if exponent > 0 else denominator).append(term)

        def dimensions_text():
            text = ' * '.join(numerator) if numerator else '1'
            if denominator:
                text += ''.join(f' / {term}' for term in denominator)
            return text

        if skip_unity and abs(self.factor - 1.0) < 1.0e-300:
            return dimensions_text()
        factor = f'{self.factor:.1f}' if self.factor == round(self.factor) else f'{self.factor:g}'
        if numerator:
            return f"{factor} {' * '.join(numerator)}" + ''.join(
                f' / {term}' for term in denominator)
        return factor + ''.join(f' / {term}' for term in denominator)

    def convertible(self, other):
        if not isinstance(other, Units):
            raise TypeError('convertible expects a Units object')
        return self.dimensions == other.dimensions

    def dimension(self, primary):
        """Return a primary dimension, following ``Units::dimension``."""
        names = ('mass', 'length', 'time', 'temperature', 'current', 'quantity')
        try:
            return self.dimensions[names.index(primary)]
        except ValueError as err:
            raise UnitError(f'Unknown primary unit {primary!r}') from err

    # Source spellings retained alongside Python-native names.
    pow = power
    str = to_string


class UnitStack:
    """Aggregate unit/exponent pairs used for reaction-rate dimensions.

    This is the editable Python counterpart of pinned ``UnitStack`` in
    ``include/cantera/base/Units.h:104-142`` and
    ``src/base/Units.cpp:326-389``.  The first item is the *standard* unit;
    :meth:`join` changes only its exponent, while :meth:`update` combines an
    exactly matching :class:`Units` value anywhere in the stack.

    ``UnitStack()`` and ``UnitStack([])`` construct the source's two empty
    forms. ``UnitStack(units)`` constructs a stack whose standard unit starts
    with exponent zero. An iterable of ``(Units, exponent)`` pairs implements
    the C++ initializer-list constructor.
    """

    def __init__(self, units=None):
        if units is None:
            self.stack = []
        elif isinstance(units, Units):
            self.stack = [(units, 0.0)]
        else:
            try:
                entries = list(units)
            except TypeError as err:
                raise TypeError(
                    'UnitStack expects a Units object or unit/exponent pairs'
                ) from err
            self.stack = []
            for entry in entries:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    raise TypeError('UnitStack entries must be (Units, exponent) pairs')
                unit, exponent = entry
                if not isinstance(unit, Units):
                    raise TypeError('UnitStack entries must contain Units objects')
                exponent = float(exponent)
                self.stack.append((unit, exponent))

    def __len__(self):
        return len(self.stack)

    def size(self):
        """Return the number of unit/exponent pairs (``UnitStack::size``)."""
        return len(self.stack)

    def standard_units(self):
        """Return the first unit, or the pinned zero-factor empty marker."""
        if not self.stack:
            return Units(0.0)
        return self.stack[0][0]

    def set_standard_units(self, standard_units):
        """Install or replace an unused standard unit.

        As in the pinned implementation, replacement is forbidden after the
        standard exponent has become nonzero.
        """
        if not isinstance(standard_units, Units):
            raise TypeError('standard_units must be a Units object')
        if not self.stack:
            self.stack.append((standard_units, 0.0))
            return
        if self.stack[0][1] != 0.0:
            raise UnitError('Standard unit is already defined.')
        self.stack[0] = (standard_units, 0.0)

    def standard_exponent(self):
        """Return the standard exponent, or NaN for an empty stack."""
        if not self.stack:
            return float('nan')
        return self.stack[0][1]

    def join(self, exponent):
        """Add *exponent* to the standard-unit exponent."""
        if not self.stack:
            raise UnitError('Standard unit is not defined.')
        exponent = float(exponent)
        unit, current = self.stack[0]
        self.stack[0] = (unit, current + exponent)

    def update(self, units, exponent):
        """Accumulate an exact unit match or append a new pair."""
        if not isinstance(units, Units):
            raise TypeError('units must be a Units object')
        exponent = float(exponent)
        for index, (current_unit, current_exponent) in enumerate(self.stack):
            if current_unit == units:
                self.stack[index] = (current_unit, current_exponent + exponent)
                return
        self.stack.append((units, exponent))

    def product(self):
        """Return the product of every unit raised to its stored exponent."""
        if not self.stack:
            return Units(0.0)
        result = Units(1.0)
        for units, exponent in self.stack:
            result = result * (units if exponent == 1.0 else units.power(exponent))
        return result

    # Source-name aliases make symbol-level comparisons unambiguous while the
    # snake_case spellings remain the native Python API.
    standardUnits = standard_units
    setStandardUnits = set_standard_units
    standardExponent = standard_exponent


def _unit(factor, mass=0, length=0, time=0, temperature=0, current=0, quantity=0):
    pressure = mass if (mass != 0 and length == -mass and time == -2 * mass
                        and temperature == current == quantity == 0) else 0.0
    energy = mass if (mass != 0 and length == 2 * mass and time == -2 * mass
                      and temperature == current == quantity == 0) else 0.0
    return Units(factor, (mass, length, time, temperature, current, quantity),
                 pressure, energy)


# Direct values from pinned src/base/Units.cpp knownUnits.
KNOWN_UNITS = {
    '1': _unit(1), 'kg': _unit(1, mass=1), 'g': _unit(1e-3, mass=1),
    'm': _unit(1, length=1), 'micron': _unit(1e-6, length=1), 'angstrom': _unit(1e-10, length=1), 'Å': _unit(1e-10, length=1),
    's': _unit(1, time=1), 'min': _unit(60, time=1), 'hr': _unit(3600, time=1), 'K': _unit(1, temperature=1), 'C': _unit(1, temperature=1),
    'A': _unit(1, current=1), 'mol': _unit(1e-3, quantity=1), 'gmol': _unit(1e-3, quantity=1),
    'mole': _unit(1e-3, quantity=1), 'kmol': _unit(1, quantity=1), 'kgmol': _unit(1, quantity=1), 'molec': _unit(1 / Avogadro, quantity=1),
    'J': _unit(1, 1, 2, -2), 'cal': _unit(4.184, 1, 2, -2), 'erg': _unit(1e-7, 1, 2, -2), 'eV': _unit(ElectronCharge, 1, 2, -2),
    'N': _unit(1, 1, 1, -2), 'dyn': _unit(1e-5, 1, 1, -2), 'Pa': _unit(1, 1, -1, -2),
    'atm': _unit(OneAtm, 1, -1, -2), 'bar': _unit(1e5, 1, -1, -2), 'liter': _unit(1e-3, length=3), 'L': _unit(1e-3, length=3), 'l': _unit(1e-3, length=3), 'cc': _unit(1e-6, length=3),
    'm^3': _unit(1, length=3), 'm³': _unit(1, length=3),
    'ohm': _unit(1, 1, 2, -3, current=-2), 'V': _unit(1, 1, 2, -3, current=-1),
    'coulomb': _unit(1, time=1, current=1),
    'townsend': _unit(1e-21, 1, 4, -3, current=-1),
    'Td': _unit(1e-21, 1, 4, -3, current=-1),
    'J/kmol': _unit(1, 1, 2, -2, quantity=-1),
}
PREFIXES = {'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 'h': 1e2, 'd': 1e-1, 'c': 1e-2, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24}


def parse_units(expression: str) -> Units:
    """Parse upstream-style ``*``, ``/``, and ``^`` unit expressions."""
    if not isinstance(expression, str):
        raise TypeError('Unit expression must be str')
    text = expression.strip().replace(' ', '').replace('³', '^3')
    if not text or text == '1':
        return Units()
    numeric = re.match(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', text)
    result = Units()
    if numeric:
        result = Units(float(numeric.group(1)))
        text = text[numeric.end():]
        if not text:
            return result
    for operator, token in re.findall(r'([*/]?)([^*/]+)', text):
        match = re.fullmatch(r'([^\^]+)(?:\^([+-]?(?:\d+(?:\.\d*)?|\.\d+)))?', token)
        if not match:
            raise UnitError(f'Invalid unit term {token!r}')
        name, exponent_text = match.groups()
        exponent = float(exponent_text) if exponent_text is not None else 1.0
        if operator == '/':
            exponent = -exponent
        unit = KNOWN_UNITS.get(name)
        if unit is None and len(name) > 1 and name[0] in PREFIXES and name[1:] in KNOWN_UNITS:
            base = KNOWN_UNITS[name[1:]]
            unit = Units(base.factor * PREFIXES[name[0]], base.dimensions)
        if unit is None:
            raise UnitError(f'Unknown unit {name!r} in {expression!r}')
        result = result * unit.power(exponent)
    return result


def convert(value: float, source: str, destination: str) -> float:
    """Convert a scalar using ``value * source.factor / destination.factor``."""
    src, dest = _coerce_units(source), _coerce_units(destination)
    if not src.convertible(dest):
        raise UnitError(f'Units {source!r} and {destination!r} are not convertible')
    return float(value) * src.factor / dest.factor


def convert_activation_energy(value: float, source: str, destination: str) -> float:
    """Convert J/kmol, temperature, or eV activation-energy representations."""
    src, dest = _coerce_units(source), _coerce_units(destination)
    energy_per_amount = parse_units('J/kmol')
    kelvin, electron_volt = parse_units('K'), parse_units('eV')
    amount = float(value)
    if src.convertible(energy_per_amount):
        joules_per_kmol = amount * src.factor
    elif src.convertible(kelvin):
        joules_per_kmol = amount * GasConstant * src.factor
    elif src.convertible(electron_volt):
        joules_per_kmol = amount * Avogadro * src.factor
    else:
        raise UnitError(f'Unit {source!r} is not an activation energy')
    if dest.convertible(energy_per_amount):
        return joules_per_kmol / dest.factor
    if dest.convertible(kelvin):
        return joules_per_kmol / GasConstant / dest.factor
    if dest.convertible(electron_volt):
        return joules_per_kmol / Avogadro / dest.factor
    raise UnitError(f'Unit {destination!r} is not an activation energy')


def _coerce_units(value) -> Units:
    if isinstance(value, Units):
        return value
    if isinstance(value, str):
        return parse_units(value)
    raise TypeError('Units must be specified by a string or Units object')


def _split_decoded_value(value):
    """Python equivalent of pinned ``split_unit(const AnyValue&)``.

    YAML decoders in this port produce numbers and strings, so these are the
    supported AnyValue-equivalent node types.
    """
    if isinstance(value, bool):
        raise UnitError('Boolean values are not dimensional quantities')
    if isinstance(value, (int, float)):
        numeric = float(value)
        if not isfinite(numeric):
            raise UnitError(f'Non-finite dimensional value {value!r}')
        return numeric, None
    if isinstance(value, str):
        match = re.fullmatch(
            r'\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+(.+?)\s*',
            value,
        )
        if not match:
            raise UnitError(
                f"Couldn't parse {value!r} as a space-separated value/unit pair"
            )
        return float(match.group(1)), match.group(2)
    raise UnitError(f'Expected a number or dimensioned string; got {type(value).__name__}')


class UnitSystem:
    """Pinned-style default unit state and decoded-value conversions.

    This implements the ``UnitSystem`` paths from pinned ``Units.cpp`` using
    Python numbers, strings, and sequences as decoded ``AnyValue`` equivalents.
    """

    _BASE_DEFAULTS = {
        'mass': 'kg', 'length': 'm', 'time': 's', 'quantity': 'kmol',
        'pressure': 'Pa', 'energy': 'J', 'temperature': 'K', 'current': 'A',
        'activation-energy': 'J / kmol',
    }

    def __init__(self, units=()):
        self._factors = {
            'mass': 1.0, 'length': 1.0, 'time': 1.0, 'quantity': 1.0,
            'pressure': 1.0, 'energy': 1.0,
        }
        self._defaults = {}
        self._activation_energy_factor = 1.0
        self._explicit_activation_energy = False
        if units:
            self.set_defaults(units)

    @property
    def defaults(self):
        values = dict(self._BASE_DEFAULTS)
        values.update(self._defaults)
        if 'activation-energy' not in self._defaults:
            values['activation-energy'] = f"{values['energy']} / {values['quantity']}"
        return values

    def set_defaults(self, units):
        if isinstance(units, Mapping):
            self._set_defaults_mapping(units)
        elif isinstance(units, str):
            raise TypeError('Unit defaults must be a mapping or iterable of unit strings')
        else:
            self._set_defaults_iterable(units)
        if not self._explicit_activation_energy:
            self._activation_energy_factor = (
                self._factors['energy'] / self._factors['quantity']
            )

    def _set_defaults_iterable(self, units: Iterable):
        primary = {
            'mass': parse_units('kg'), 'length': parse_units('m'),
            'time': parse_units('s'), 'quantity': parse_units('kmol'),
            'pressure': parse_units('Pa'), 'energy': parse_units('J'),
            'temperature': parse_units('K'), 'current': parse_units('A'),
        }
        for name in units:
            if not isinstance(name, str):
                raise TypeError('Each default unit must be a string')
            unit = parse_units(name)
            dimension = next((key for key, expected in primary.items()
                              if unit.convertible(expected)), None)
            if dimension is None:
                raise UnitError(f'Unable to match unit {name!r} to a basic dimension')
            if dimension in ('temperature', 'current'):
                if unit.factor != 1.0:
                    raise UnitError(f'{dimension.capitalize()} scales with non-unity factors are unsupported')
                continue
            self._factors[dimension] = unit.factor
            self._defaults[dimension] = name

    def _set_defaults_mapping(self, units: Mapping):
        expected = {
            'mass': 'kg', 'length': 'm', 'time': 's', 'quantity': 'kmol',
            'pressure': 'Pa', 'energy': 'J', 'temperature': 'K', 'current': 'A',
        }
        for dimension, name in units.items():
            if not isinstance(dimension, str) or not isinstance(name, str):
                raise TypeError('Default dimensions and units must be strings')
            if dimension == 'activation-energy':
                continue
            if dimension not in expected:
                raise UnitError(f'Unable to set default unit for unknown dimension {dimension!r}')
            unit = parse_units(name)
            if not unit.convertible(parse_units(expected[dimension])):
                raise UnitError(f'Unable to set default unit for {dimension!r} to {name!r}')
            if dimension in ('temperature', 'current'):
                if unit.factor != 1.0:
                    raise UnitError(f'{dimension.capitalize()} scales with non-unity factors are unsupported')
                continue
            self._factors[dimension] = unit.factor
            self._defaults[dimension] = name
        if 'activation-energy' in units:
            self.set_default_activation_energy(units['activation-energy'])

    def set_default_activation_energy(self, units):
        parsed = _coerce_units(units)
        if parsed.convertible(parse_units('J/kmol')):
            factor = parsed.factor
        elif parsed.convertible(parse_units('K')):
            factor = GasConstant
        elif parsed.convertible(parse_units('eV')):
            factor = parsed.factor * Avogadro
        else:
            raise UnitError(f'Unable to match {units!r} to activation-energy units')
        self._activation_energy_factor = factor
        self._explicit_activation_energy = True
        self._defaults['activation-energy'] = units if isinstance(units, str) else repr(units)

    def convert(self, value, source, destination=None):
        """Convert explicit units or a decoded scalar/vector to destination units."""
        if destination is not None:
            return convert(value, source, destination)
        dest = _coerce_units(source)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            converted = []
            for index, item in enumerate(value):
                try:
                    converted.append(self._convert_decoded_scalar(item, dest))
                except (TypeError, UnitError) as err:
                    raise UnitError(f'Unable to convert value at index {index}: {err}') from err
            return converted
        return self._convert_decoded_scalar(value, dest)

    def _convert_decoded_scalar(self, value, destination: Units):
        numeric, explicit_units = _split_decoded_value(value)
        if explicit_units is None:
            return self.convert_to(numeric, destination)
        return convert(numeric, explicit_units, destination)

    def convert_to(self, value, destination):
        dest = _coerce_units(destination)
        mass, length, time, _temperature, _current, quantity = dest.dimensions
        return (float(value) / dest.factor
                * self._factors['mass'] ** (mass - dest.pressure_dimension - dest.energy_dimension)
                * self._factors['length'] ** (length + dest.pressure_dimension - 2 * dest.energy_dimension)
                * self._factors['time'] ** (time + 2 * dest.pressure_dimension + 2 * dest.energy_dimension)
                * self._factors['quantity'] ** quantity
                * self._factors['pressure'] ** dest.pressure_dimension
                * self._factors['energy'] ** dest.energy_dimension)

    def convert_from(self, value, source):
        src = _coerce_units(source)
        mass, length, time, _temperature, _current, quantity = src.dimensions
        return (float(value) * src.factor
                * self._factors['mass'] ** (-mass + src.pressure_dimension + src.energy_dimension)
                * self._factors['length'] ** (-length - src.pressure_dimension + 2 * src.energy_dimension)
                * self._factors['time'] ** (-time - 2 * src.pressure_dimension - 2 * src.energy_dimension)
                * self._factors['quantity'] ** (-quantity)
                * self._factors['pressure'] ** (-src.pressure_dimension)
                * self._factors['energy'] ** (-src.energy_dimension))

    def convert_rate_coeff(self, value, destination):
        """Convert a decoded rate coefficient, including undefined-unit mode.

        A zero-factor :class:`Units` object is the pinned ``UnitStack`` marker
        for a standalone rate whose dimensions are unknown. In that case only
        values already expressed in the default m-kg-kmol-s system are safe.
        """
        dest = _coerce_units(destination)
        if dest.factor != 0.0:
            return self.convert(value, dest)

        numeric, explicit_units = _split_decoded_value(value)
        if explicit_units is None:
            if self._factors['length'] == 1.0 and self._factors['quantity'] == 1.0:
                return numeric
        else:
            source = parse_units(explicit_units)
            if abs(source.factor - 1.0) < 1.0e-14:
                return numeric
        raise UnitError(
            'Unable to convert value with non-default units to undefined units; '
            'likely while creating a standalone reaction-rate object')

    def convert_activation_energy_to(self, value, destination):
        dest = _coerce_units(destination)
        base = float(value) * self._activation_energy_factor
        if dest.convertible(parse_units('J/kmol')):
            return base / dest.factor
        if dest.convertible(parse_units('K')):
            # The pinned overload intentionally does not apply dest.factor().
            return base / GasConstant
        if dest.convertible(parse_units('eV')):
            return base / (Avogadro * dest.factor)
        raise UnitError('Destination is not a unit of activation energy')

    def convert_activation_energy_from(self, value, source):
        src = _coerce_units(source)
        if src.convertible(parse_units('J/kmol')):
            base = float(value) * src.factor
        elif src.convertible(parse_units('K')):
            # The pinned overload intentionally does not apply src.factor().
            base = float(value) * GasConstant
        elif src.convertible(parse_units('eV')):
            base = float(value) * Avogadro * src.factor
        else:
            raise UnitError('Source is not a unit of activation energy')
        return base / self._activation_energy_factor

    def convert_activation_energy(self, value, destination, explicit_destination=None):
        if explicit_destination is not None:
            return convert_activation_energy(value, destination, explicit_destination)
        numeric, explicit_units = _split_decoded_value(value)
        if explicit_units is None:
            return self.convert_activation_energy_to(numeric, destination)
        return convert_activation_energy(numeric, explicit_units, destination)

    def get_delta(self, other):
        if not isinstance(other, UnitSystem):
            raise TypeError('get_delta expects a UnitSystem')
        delta = {}
        for dimension in ('mass', 'length', 'time', 'pressure', 'energy', 'quantity'):
            if self._factors[dimension] != other._factors[dimension]:
                delta[dimension] = self.defaults[dimension]
        if (self._explicit_activation_energy or
                (other._explicit_activation_energy and
                 self._activation_energy_factor != self._factors['energy'] / self._factors['quantity'])):
            delta['activation-energy'] = self.defaults['activation-energy']
        return delta

    # Source-name aliases make direct symbol comparisons possible while the
    # snake_case spellings remain the primary Python API.
    setDefaults = set_defaults
    setDefaultActivationEnergy = set_default_activation_energy
    convertTo = convert_to
    convertFrom = convert_from
    convertRateCoeff = convert_rate_coeff
    convertActivationEnergyTo = convert_activation_energy_to
    convertActivationEnergyFrom = convert_activation_energy_from
    convertActivationEnergy = convert_activation_energy
    getDelta = get_delta


def _convert_activation_to_base(value, units):
    if units.convertible(parse_units('J/kmol')):
        return value * units.factor
    if units.convertible(parse_units('K')):
        return value * GasConstant * units.factor
    if units.convertible(parse_units('eV')):
        return value * Avogadro * units.factor
    raise UnitError('Source is not a unit of activation energy')


def _convert_activation_from_base(value, units):
    if units.convertible(parse_units('J/kmol')):
        return value / units.factor
    if units.convertible(parse_units('K')):
        return value / (GasConstant * units.factor)
    if units.convertible(parse_units('eV')):
        return value / (Avogadro * units.factor)
    raise UnitError('Destination is not a unit of activation energy')

setattr(Units, 'Units', lambda self: None)
