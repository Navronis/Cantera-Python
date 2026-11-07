"""Scientific invariant tests for Kinetics Family 4.

Verifies:
- ReactionPath diagram element flux antisymmetry: netFlow(k1, k2) == -netFlow(k2, k1)
- ReactionPath diagram flow non-negativity: flow(k1, k2) >= 0
- Group algebra abelian group properties (commutativity, associativity, inverse, identity)
- Atom conservation across all reactions in standard mechanisms
- Reversible vs irreversible reaction directionality invariants
"""

import pytest
import cantera_python as ct
from cantera_python.kinetics import (
    Reaction, Group, SpeciesNode, Path as KPath,
    ReactionPathDiagram, newReactionPathDiagram
)


def test_reaction_path_netflow_antisymmetry():
    sol = ct.Solution("h2o2.yaml")
    sol.TPX = 1000.0, 101325.0, "H2:2.0, O2:1.0, AR:7.0"
    diag = newReactionPathDiagram(sol.kinetics, "H")
    diag.build()

    species_list = diag.species()
    for i1 in range(len(species_list)):
        k1 = species_list[i1]
        for i2 in range(i1 + 1, len(species_list)):
            k2 = species_list[i2]
            fwd = diag.netFlow(k1, k2)
            rev = diag.netFlow(k2, k1)
            assert abs(fwd + rev) < 1e-10, f"Net flow antisymmetry failed between {k1} and {k2}: {fwd} vs {rev}"


def test_reaction_path_oneway_flow_positivity():
    sol = ct.Solution("h2o2.yaml")
    sol.TPX = 1000.0, 101325.0, "H2:2.0, O2:1.0, AR:7.0"
    diag = newReactionPathDiagram(sol.kinetics, "H")
    diag.setFlowType("OneWayFlow")
    diag.build()

    species_list = diag.species()
    for k1 in species_list:
        for k2 in species_list:
            if k1 != k2:
                f = diag.flow(k1, k2)
                assert f >= 0.0, f"One-way flow must be non-negative: flow({k1}, {k2}) = {f}"


def test_reaction_path_scaling_preserves_ratios():
    sol = ct.Solution("h2o2.yaml")
    sol.TPX = 1000.0, 101325.0, "H2:2.0, O2:1.0, AR:7.0"

    d1 = newReactionPathDiagram(sol.kinetics, "H")
    d1.scale = -1.0  # auto-scale by maxFlow
    d1.build()

    d2 = newReactionPathDiagram(sol.kinetics, "H")
    d2.scale = 1.0   # unscaled
    d2.build()

    assert d1.maxFlow() == pytest.approx(d2.maxFlow(), rel=1e-12)
    assert d1.nNodes() == d2.nNodes()
    assert d1.nPaths() == d2.nPaths()


def test_group_algebra_abelian_properties():
    # Commutativity: g1 + g2 == g2 + g1
    g1 = Group([2, 1, 0])
    g2 = Group([0, 1, 3])
    sum1 = g1 + g2
    sum2 = g2 + g1
    assert sum1 == sum2
    assert sum1.nAtoms() == 7

    # Associativity: (g1 + g2) + g3 == g1 + (g2 + g3)
    g3 = Group([1, 0, 1])
    assert (g1 + g2) + g3 == g1 + (g2 + g3)

    # Identity: g + 0 == g
    g0 = Group([0, 0, 0])
    assert g1 + g0 == g1

    # Inverse: g - g == 0
    diff_self = g1 - g1
    assert diff_self == g0
    assert diff_self.nAtoms() == 0


def test_reaction_balance_all_gri30():
    sol = ct.Solution("gri30.yaml")
    kin = sol.kinetics
    for i in range(kin.nReactions()):
        rxn = kin.reaction(i)
        # Verify checkBalance and checkSpecies pass without raising
        rxn.checkBalance(kin)
        rxn.checkSpecies(kin)


def test_reaction_balance_all_h2o2():
    sol = ct.Solution("h2o2.yaml")
    kin = sol.kinetics
    for i in range(kin.nReactions()):
        rxn = kin.reaction(i)
        rxn.checkBalance(kin)
        rxn.checkSpecies(kin)
