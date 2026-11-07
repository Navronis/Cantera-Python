"""Failure semantics tests for Kinetics Family 4.

Verifies source-exact exceptions and edge-case behavior for:
- ReactionPathDiagram error handling
- Reaction validation, balance checks, and species verification
- Group component sign consistency and validity
- Kinetics and ReactionRate factory error modes
"""

import pytest
import cantera_python as ct
from cantera_python.kinetics import (
    Reaction, Group, SpeciesNode, Path as KPath,
    ReactionPathDiagram, newReactionPathDiagram, newKinetics
)
from cantera_python.rates import newReactionRate, ArrheniusRate


def test_unknown_flow_type():
    sol = ct.Solution("h2o2.yaml")
    diag = newReactionPathDiagram(sol.kinetics, "H")
    with pytest.raises(ValueError, match="Unknown flow type"):
        diag.setFlowType("InvalidFlowType")


def test_reaction_empty_type():
    rxn = Reaction()
    with pytest.raises(ValueError, match="Empty Reaction does not have a type"):
        rxn.type()


def test_reaction_set_rate_none():
    rxn = Reaction()
    with pytest.raises(ValueError, match="Reaction rate must not be empty"):
        rxn.setRate(None)


def test_reaction_unbalanced():
    rxn = Reaction()
    rxn.setEquation("H2 + O2 => H2O")  # unbalanced: 2 H, 2 O => 2 H, 1 O
    sol = ct.Solution("h2o2.yaml")
    with pytest.raises(ValueError, match="unbalanced"):
        rxn.checkBalance(sol.kinetics)


def test_reaction_undeclared_species():
    rxn = Reaction()
    rxn.setEquation("H2 + UNKNOWN_SPEC => H2O")
    sol = ct.Solution("h2o2.yaml")
    with pytest.raises(ValueError, match="undeclared species"):
        rxn.checkSpecies(sol.kinetics)


def test_group_mixed_sign_invalidation():
    # Mixed positive and negative atoms invalidates the transfer group
    g = Group([1, -1, 0])
    assert not g.valid()
    assert g.sign() == -999


def test_group_all_negative_valid():
    # All negative is a valid reverse group (sign = -1)
    g = Group([-1, -2, 0])
    assert g.valid()
    assert g.sign() == -1
    assert g.nAtoms() == 3


def test_group_empty():
    g = Group()
    assert not g.valid()
    assert g.sign() == -999
    assert g.size() == 0
    assert g.nAtoms() == 0


def test_kinetics_factory_unknown():
    with pytest.raises(ValueError, match="Unknown kinetics model"):
        newKinetics("NonExistentKineticsModel_12345")


def test_reaction_rate_factory_default():
    rate = newReactionRate("UnknownRateType_ABC")
    assert isinstance(rate, ArrheniusRate)
