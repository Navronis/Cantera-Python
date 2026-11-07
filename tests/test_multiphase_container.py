"""Pinned-native tests for the MultiPhase state container."""
import json
from pathlib import Path

import numpy as np
import pytest

import cantera_python as ct
from cantera_python.vcs import MultiPhase, VCSError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "cantera-original" / "data"
REFERENCE = ROOT / "tests" / "fixtures" / "pinned_multiphase_container.json"


def _mixture():
    gas = ct.Solution(DATA / "gri30.yaml")
    gas.TPX = 1200.0, ct.OneAtm, "CH4:1.0, O2:2.0, N2:7.52"
    graphite = ct.Solution(DATA / "graphite.yaml")
    graphite.TP = gas.TP
    mixture = MultiPhase()
    mixture.add_phase(gas, 1.0)
    mixture.add_phase(graphite, 0.2)
    return mixture


def test_multiphase_species_and_element_aggregates_match_pinned_native():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert reference["pinned_commit"] == "726522be4e2a13454d8415b7ef799d621f665cf3"
    mixture = _mixture()

    assert mixture.nPhases() == len(reference["state"]["phase_names"])
    assert mixture.phase_names == tuple(reference["state"]["phase_names"])
    assert mixture.species_names() == reference["species_names"]
    assert mixture.element_names == tuple(reference["element_names"])
    np.testing.assert_allclose(
        mixture.phase_moles, reference["state"]["phase_moles"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        mixture._species_moles(), reference["species_moles"], rtol=2e-14, atol=2e-16)

    for element_index, element in enumerate(reference["element_names"]):
        native_amount = reference["element_moles"][element]
        assert mixture.elementIndex(element) == element_index
        assert mixture.elementMoles(element_index) == pytest.approx(
            native_amount, rel=2e-14, abs=2e-16)
    np.testing.assert_allclose(
        [mixture.phaseCharge(i) for i in range(mixture.nPhases())],
        reference["phase_charge"], rtol=2e-14, atol=1e-30)
    assert mixture.charge == pytest.approx(reference["charge"], abs=1e-30)

    for k, name in enumerate(reference["species_names"]):
        phase_index = 0 if k < len(mixture.phase(0).species_names) else 1
        local_index = k if phase_index == 0 else k - len(mixture.phase(0).species_names)
        assert mixture.speciesName(k) == name
        assert mixture.speciesPhaseIndex(k) == phase_index
        assert mixture.moleFraction(k) == pytest.approx(
            mixture.phase(phase_index).X[local_index], rel=0.0, abs=1e-15)
        assert mixture.speciesMoles(k) == pytest.approx(
            reference["species_moles"][k], rel=2e-14, abs=2e-16)


def test_multiphase_get_and_set_species_moles_use_flat_species_order():
    mixture = _mixture()
    output = np.full(mixture.nSpecies() + 2, -1.0)
    mixture.getMoles(output)
    np.testing.assert_allclose(output[:mixture.nSpecies()], mixture._species_moles())
    assert output[-2:].tolist() == [-1.0, -1.0]

    amounts = np.zeros(mixture.nSpecies())
    amounts[mixture.phase(0).species_index("H2")] = 2.0
    amounts[mixture.phase(0).species_index("O2")] = 3.0
    amounts[-1] = 0.25  # graphite is the final species in the flattened list
    mixture.setMoles(amounts)

    assert mixture.phase_moles == pytest.approx([5.0, 0.25])
    assert mixture.phase(0).X[mixture.phase(0).species_index("H2")] == pytest.approx(0.4)
    assert mixture.phase(0).X[mixture.phase(0).species_index("O2")] == pytest.approx(0.6)
    assert mixture.speciesMoles(mixture.nSpecies() - 1) == pytest.approx(0.25)


def test_multiphase_init_freezes_layout_and_uploads_phase_composition_explicitly():
    mixture = _mixture()
    mixture.init()
    h2 = mixture.phase(0).species_index("H2")
    initial_fraction = mixture.moleFraction(h2)

    # The phase object is shared, while MultiPhase retains its own composition
    # until uploadMoleFractionsFromPhases is called, as in native MultiPhase.
    mixture.phase(0).TPX = mixture.T, mixture.P, "H2:1.0"
    assert mixture.moleFraction(h2) == pytest.approx(initial_fraction)
    mixture.phase(0)  # updatePhases restores the mixture's stored composition
    assert mixture.phase(0).X[h2] == pytest.approx(initial_fraction)

    mixture.phase(0).TPX = mixture.T, mixture.P, "H2:1.0"
    mixture.uploadMoleFractionsFromPhases()
    assert mixture.moleFraction(h2) == pytest.approx(1.0)
    assert mixture.speciesMoles(h2) == pytest.approx(1.0)

    with pytest.raises(VCSError, match="after init"):
        mixture.add_phase(ct.Solution(DATA / "graphite.yaml"), 0.1)
