"""Reaction-equation parsing and stoichiometric duplicate comparison.

Sources: ``parseReactionEquation`` in ``src/kinetics/Reaction.cpp`` and
``Kinetics::checkDuplicateStoich`` in ``src/kinetics/Kinetics.cpp``. This is
input and validation infrastructure only; rate evaluation remains in
``cantera_python.kinetics``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
import re
from typing import Mapping


class ReactionEquationError(ValueError):
    """A reaction equation cannot be interpreted with the upstream grammar."""


@dataclass(frozen=True)
class ParsedReactionEquation:
    reactants: dict[str, float]
    products: dict[str, float]
    reversible: bool

    @property
    def net_stoichiometry(self) -> dict[str, float]:
        names = set(self.reactants) | set(self.products)
        return {name: self.products.get(name, 0.0) - self.reactants.get(name, 0.0)
                for name in names if self.products.get(name, 0.0) != self.reactants.get(name, 0.0)}


def parse_reaction_equation(equation: str, species_names=None) -> ParsedReactionEquation:
    """Parse the whitespace-tokenized syntax used by Cantera Reaction.cpp.

    Coefficients therefore require a space before the species (``2 H2``), as
    in the upstream tokenizer. ``<=>`` and ``=`` mark reversible reactions;
    ``=>`` marks an irreversible reaction. When *species_names* is supplied,
    ordinary unrecognized species raise an error. The special third-body
    token ``(+M)`` is retained as a stoichiometric name for a future third-body
    constructor, mirroring the parser's intermediate representation.
    """
    if not isinstance(equation, str):
        raise TypeError("Reaction equation must be str")
    normalized_eq = re.sub(r'\(\s*\+\s*([^)]+?)\s*\)', r'(+\1)', equation.strip())
    tokens = normalized_eq.split()
    if not tokens:
        raise ReactionEquationError("Reaction equation is empty")
    allowed = None if species_names is None else set(species_names)
    reactants, products = {}, {}
    current, reversible, seen_arrow = reactants, None, False
    part = []

    def add_part(fields, target):
        if not fields:
            raise ReactionEquationError(f"Missing species in reaction equation {equation!r}")
        if len(fields) == 1:
            coefficient, name = 1.0, fields[0]
        elif len(fields) == 2:
            try:
                coefficient = float(fields[0])
            except ValueError as error:
                raise ReactionEquationError(f"Invalid stoichiometric coefficient {fields[0]!r}") from error
            name = fields[1]
        else:
            raise ReactionEquationError(f"Cannot parse reaction term {' '.join(fields)!r}")
        if not isfinite(coefficient) or coefficient < 0:
            raise ReactionEquationError("Stoichiometric coefficients must be finite and nonnegative")
        mass_action = not (name.startswith("(+") and name.endswith(")"))
        if allowed is not None and mass_action and name != "M" and name not in allowed:
            raise ReactionEquationError(f"Unknown reaction species {name!r}")
        target[name] = target.get(name, 0.0) + coefficient

    for token in tokens + ["+"]:
        if token in ("<=>", "=", "=>"):
            if seen_arrow:
                raise ReactionEquationError("Reaction equation contains more than one arrow")
            add_part(part, current)
            part = []
            current = products
            reversible = token != "=>"
            seen_arrow = True
        elif token == "+" or token.startswith("(+"):
            # ``(+M)`` terminates the preceding term in the C++ parser, and is
            # then carried as the next terminal token. Preserve it explicitly.
            if token.startswith("(+"):
                add_part(part, current)
                part = [token]
            else:
                add_part(part, current)
                part = []
        else:
            part.append(token)
    if not seen_arrow or not reactants or not products:
        raise ReactionEquationError("Reaction equation must contain reactants, products, and one arrow")
    return ParsedReactionEquation(reactants, products, bool(reversible))


def duplicate_stoichiometry_ratio(first: Mapping[str, float], second: Mapping[str, float], *,
                                  rtol: float = 1e-8) -> float:
    """Return the source-style multiple between two signed net stoichiometries.

    A positive result means the same direction; a negative result means the
    reverse direction; zero means the reactions are not stoichiometric
    duplicates. This ports the two passes in ``checkDuplicateStoich``.
    """
    one = {name: float(value) for name, value in first.items() if float(value) != 0.0}
    two = {name: float(value) for name, value in second.items() if float(value) != 0.0}
    if not one or not two or any(not isfinite(value) for value in (*one.values(), *two.values())):
        return 0.0
    keys = set(one) | set(two)
    anchor = next(iter(one))
    for sign in (1.0, -1.0):
        if anchor not in two:
            continue
        ratio = two[anchor] / (sign * one[anchor])
        if ratio == 0.0:
            continue
        if all(name in one and name in two and isclose(two[name], sign * ratio * one[name],
                                                       rel_tol=rtol, abs_tol=0.0)
               for name in keys):
            return sign * ratio
    return 0.0


def are_duplicate_reactions(first: ParsedReactionEquation, second: ParsedReactionEquation) -> bool:
    """Apply the direction rule used by ``Kinetics::checkDuplicates``."""
    ratio = duplicate_stoichiometry_ratio(first.net_stoichiometry, second.net_stoichiometry)
    return ratio != 0.0 and not (ratio < 0.0 and not first.reversible and not second.reversible)
