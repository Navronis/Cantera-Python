"""Composition-string parsing and species-name lookup.

This module ports the small, shared input layer used by Cantera phases.  It is
kept independent from YAML readers and native Cantera so higher-level pure
Python thermo, kinetics, reactor, and one-dimensional code can use the same
composition syntax.

Sources: ``src/base/stringUtils.cpp`` (``parseCompString``) and
``src/thermo/Phase.cpp`` (``speciesIndex`` and ``addSpeciesAlias``).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math


class CompositionError(ValueError):
    """A malformed composition or invalid species-name lookup."""


_SEPARATORS = ", ;\n\t"
_WHITESPACE = " \t\n"


def _next_non_separator(text: str, start: int) -> int:
    while start < len(text) and text[start] in _SEPARATORS:
        start += 1
    return start


def _parse_float(text: str) -> float:
    """Parse one finite C-locale-style scalar used in a composition map."""
    try:
        value = float(text)
    except (TypeError, ValueError) as error:
        raise CompositionError(f"Invalid composition value: {text!r}") from error
    if not math.isfinite(value):
        raise CompositionError(f"Composition value must be finite: {text!r}")
    return value


def parse_comp_string(text: str, names: Iterable[str] = ()) -> dict[str, float]:
    """Parse Cantera ``name:value`` composition syntax into a mapping.

    Commas, semicolons, spaces, tabs, and newlines may separate fields.  When
    *names* is supplied, the return value includes every listed name with a
    zero value and rejects unknown names. Species names containing ``:`` are
    handled using the same retry rule as upstream ``parseCompString``.
    """
    if not isinstance(text, str):
        raise TypeError("Composition string must be str")
    valid_names = tuple(names)
    if len(set(valid_names)) != len(valid_names):
        raise ValueError("Valid species names must be distinct")
    result = {name: 0.0 for name in valid_names}

    start = left = stop = 0
    text_length = len(text)
    while stop < text_length:
        colon = text.find(":", left)
        if colon < 0:
            break
        value_start = colon + 1
        while value_start < text_length and text[value_start] in _WHITESPACE:
            value_start += 1
        stop = value_start
        while stop < text_length and text[stop] not in _SEPARATORS:
            stop += 1
        name = text[start:colon].strip()
        if valid_names and name not in result:
            raise CompositionError(f"unknown species {name!r}")
        value_text = text[value_start:stop]
        try:
            value = _parse_float(value_text)
        except CompositionError:
            # An unsuccessful parse may mean that the first colon belongs to
            # a species name such as ``phase:species``. Upstream only retries
            # when no whitespace occurs in that candidate name.
            candidate = text[start:stop]
            if any(char in _WHITESPACE for char in candidate):
                raise
            if ":" in value_text:
                left = colon + 1
                stop = 0
                continue
            raise
        # Preserve upstream's nonzero duplicate test, including the permitted
        # ``A:0, A:1`` overwrite behavior.
        if result.get(name, 0.0) != 0.0:
            raise CompositionError(f"Duplicate key: {name!r}")
        result[name] = value
        start = _next_non_separator(text, stop + 1)
        left = start

    if left != start:
        raise CompositionError(f"Unable to parse key-value pair: {text[start:stop]!r}")
    if stop < text_length and text[stop:].strip():
        raise CompositionError(f"Found non-key:value data in composition string: {text[stop:]!r}")
    return result


class SpeciesRegistry:
    """Ordered species names with Cantera-style aliases and lookup semantics."""

    def __init__(self, names: Iterable[str], *, case_sensitive: bool = True):
        self._names = tuple(names)
        if not self._names or any(not isinstance(name, str) or not name for name in self._names):
            raise ValueError("Provide one or more nonempty species names")
        if len(set(self._names)) != len(self._names):
            raise ValueError("Species names must be distinct")
        self.case_sensitive = bool(case_sensitive)
        self._indices = {name: index for index, name in enumerate(self._names)}

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def species_index(self, name: str, *, raise_on_missing: bool = True) -> int | None:
        index = self._indices.get(name)
        if index is None and not self.case_sensitive:
            matches = {value for key, value in self._indices.items() if key.lower() == name.lower()}
            if len(matches) == 1:
                index = matches.pop()
            elif len(matches) > 1:
                raise CompositionError(
                    f"Lowercase species name {name!r} is not unique; enable case-sensitive lookup")
        if index is None and raise_on_missing:
            raise CompositionError(f"Species {name!r} not found")
        return index

    def add_alias(self, name: str, alias: str) -> None:
        if self.species_index(alias, raise_on_missing=False) is not None:
            raise CompositionError(f"Invalid alias {alias!r}: species already exists")
        index = self.species_index(name, raise_on_missing=False)
        if index is None:
            raise CompositionError(f"Unable to add alias {alias!r}: original species {name!r} not found")
        self._indices[alias] = index

    def amounts(self, composition: str | Mapping[str, float]) -> tuple[float, ...]:
        """Return registry-ordered amounts, setting omitted names to zero."""
        values = (parse_comp_string(composition, self._indices)
                  if isinstance(composition, str) else dict(composition))
        unknown = set(values).difference(self._indices)
        if unknown:
            raise CompositionError(f"Unknown species: {sorted(unknown)!r}")
        amounts = [0.0] * len(self.names)
        for name, value in values.items():
            scalar = _parse_float(value)
            amounts[self._indices[name]] = scalar
        return tuple(amounts)
