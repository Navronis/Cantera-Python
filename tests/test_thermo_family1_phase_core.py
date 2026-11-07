"""Comprehensive unit tests for Thermodynamics Family 1: Phase Core State & Multi-Component Foundation."""
import pytest
import numpy as np

from cantera_python.phases import Phase

class SpeciesSpec:
    def __init__(self, name: str, mw: float, composition: dict = None, charge: float = 0.0):
        self.name = name
        self.molecular_weight = mw
        self.composition = composition or {}
        self.charge = charge

@pytest.fixture
def test_phase():
    ph = Phase("combustion_gas")
    # Add elements
    ph.addElement("H", 1.008)
    ph.addElement("C", 12.011)
    ph.addElement("O", 15.999)
    ph.addElement("N", 14.007)
    ph.addElement("Ar", 39.948)

    # Add species
    ph.addSpecies(SpeciesSpec("CH4", 16.043, {"C": 1.0, "H": 4.0}))
    ph.addSpecies(SpeciesSpec("O2", 31.998, {"O": 2.0}))
    ph.addSpecies(SpeciesSpec("N2", 28.013, {"N": 2.0}))
    ph.addSpecies(SpeciesSpec("H2O", 18.015, {"H": 2.0, "O": 1.0}))
    ph.addSpecies(SpeciesSpec("CO2", 44.010, {"C": 1.0, "O": 2.0}))
    return ph

def test_phase_element_methods(test_phase):
    ph = test_phase

    assert ph.name == "combustion_gas"
    assert ph.nElements() == 5
    el_names = ph.elementNames()
    assert el_names == ["H", "C", "O", "N", "Ar"]

    for m in range(5):
        name = ph.elementName(m)
        assert ph.elementIndex(name) == m
        assert ph.atomicWeight(m) > 0.0
        assert ph.atomicNumber(m) > 0
        assert ph.entropyElement298(m) > 0.0

    at_weights = ph.atomicWeights()
    assert len(at_weights) == 5
    assert pytest.approx(at_weights[0], rel=1e-5) == 1.008

    # Bounds check
    with pytest.raises(IndexError):
        ph.checkElementIndex(-1)
    with pytest.raises(IndexError):
        ph.checkElementIndex(10)

def test_phase_species_methods(test_phase):
    ph = test_phase

    assert ph.nSpecies() == 5
    sp_names = ph.speciesNames()
    assert sp_names == ["CH4", "O2", "N2", "H2O", "CO2"]

    for k in range(5):
        sname = ph.speciesName(k)
        assert ph.speciesIndex(sname) == k
        assert ph.molecularWeight(k) > 0.0

    mw_arr = ph.molecularWeights()
    assert len(mw_arr) == 5
    assert pytest.approx(mw_arr[0], rel=1e-5) == 16.043

    # Bounds check
    with pytest.raises(IndexError):
        ph.checkSpeciesIndex(-1)
    with pytest.raises(IndexError):
        ph.checkSpeciesIndex(25)

    # species access by index and name
    sp_obj = ph.species(0)
    assert getattr(sp_obj, "name", str(sp_obj)) == "CH4"
    sp_obj_by_name = ph.species("CH4")
    assert getattr(sp_obj_by_name, "name", str(sp_obj_by_name)) == "CH4"

def test_phase_state_coordinates_and_conversions(test_phase):
    ph = test_phase

    T_target = 1200.0
    rho_target = 0.35
    ph.setState_TD(T_target, rho_target)

    assert pytest.approx(ph.temperature(), rel=1e-10) == T_target
    assert pytest.approx(ph.density(), rel=1e-10) == rho_target

    # Set mole fractions
    ph.setMoleFractionsByName({"CH4": 0.1, "O2": 0.2, "N2": 0.7})
    mmw = ph.meanMolecularWeight()
    assert mmw > 0.0
    molar_dens = ph.molarDensity()
    assert pytest.approx(molar_dens, rel=1e-10) == rho_target / mmw
    assert pytest.approx(ph.molarVolume(), rel=1e-10) == 1.0 / molar_dens

def test_phase_composition_and_stoichiometry(test_phase):
    ph = test_phase

    comp = {"CH4": 0.095, "O2": 0.19, "N2": 0.715}
    ph.setMoleFractionsByName(comp)

    # Normalization invariants
    x = ph.moleFractions()
    assert pytest.approx(np.sum(x), rel=1e-12) == 1.0

    y = ph.massFractions()
    assert pytest.approx(np.sum(y), rel=1e-12) == 1.0

    k_ch4 = ph.speciesIndex("CH4")
    k_o2 = ph.speciesIndex("O2")
    k_n2 = ph.speciesIndex("N2")

    assert pytest.approx(ph.moleFraction(k_ch4), rel=1e-10) == 0.095
    assert pytest.approx(ph.moleFraction(k_o2), rel=1e-10) == 0.19
    assert pytest.approx(ph.moleFraction(k_n2), rel=1e-10) == 0.715

    # Concentrations
    c = ph.concentrations()
    assert pytest.approx(np.sum(c), rel=1e-10) == ph.molarDensity()

    c_buf = np.zeros(ph.nSpecies())
    ph.getConcentrations(c_buf)
    assert np.allclose(c_buf, c)

    # Stoichiometry nAtoms
    m_c = ph.elementIndex("C")
    m_h = ph.elementIndex("H")
    m_o = ph.elementIndex("O")

    assert ph.nAtoms(k_ch4, m_c) == 1.0
    assert ph.nAtoms(k_ch4, m_h) == 4.0
    assert ph.nAtoms(k_ch4, m_o) == 0.0
    assert ph.nAtoms(k_o2, m_o) == 2.0

def test_phase_state_save_and_restore(test_phase):
    ph = test_phase

    ph.setState_TD(850.0, 0.45)
    ph.setMoleFractionsByName({"CH4": 0.2, "O2": 0.4, "N2": 0.4})

    state_saved = np.zeros(ph.stateSize())
    ph.saveState(state_saved)
    assert len(state_saved) == ph.stateSize()
    assert pytest.approx(state_saved[0], rel=1e-10) == 850.0
    assert pytest.approx(state_saved[1], rel=1e-10) == 0.45

    # Change state
    ph.setState_TD(300.0, 1.2)
    ph.setMoleFractionsByName({"CO2": 1.0})

    assert pytest.approx(ph.temperature(), rel=1e-10) == 300.0
    assert pytest.approx(ph.moleFraction(ph.speciesIndex("CO2")), rel=1e-10) == 1.0

    # Restore state
    ph.restoreState(state_saved)
    assert pytest.approx(ph.temperature(), rel=1e-10) == 850.0
    assert pytest.approx(ph.density(), rel=1e-10) == 0.45
    assert pytest.approx(ph.moleFraction(ph.speciesIndex("CH4")), rel=1e-10) == 0.2
