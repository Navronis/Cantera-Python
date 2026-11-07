import numpy as np
import pytest
import cantera_python as ct
from cantera_python.kinetics import Kinetics, BulkKinetics, ReactionPathDiagram
from cantera_python.surface import InterfaceKinetics

def test_bulk_kinetics_and_rates():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1500.0, ct.OneAtm, "CH4:1.0, O2:2.0, H:0.01"
    kin = gas.kinetics

    assert isinstance(kin, BulkKinetics)
    assert isinstance(kin, Kinetics)
    assert kin.kineticsType() == "BulkKinetics"
    assert kin.nReactions() == 325
    assert kin.nTotalSpecies() == 53

    # Test get-methods with buffers
    fwd = np.zeros(kin.nReactions())
    kin.getFwdRatesOfProgress(fwd)
    assert np.any(fwd > 0.0)

    wdot = np.zeros(kin.nTotalSpecies())
    kin.getNetProductionRates(wdot)
    assert len(wdot) == 53

def test_reaction_path_diagram():
    gas = ct.Solution("gri30.yaml")
    gas.TPX = 1800.0, ct.OneAtm, "CH4:1.0, O2:2.0"
    kin = gas.kinetics

    diag = ReactionPathDiagram(kin, element="C")
    diag.setThreshold(1e-5)
    diag.setBoldThreshold(0.05)
    assert diag.threshold() == 1e-5
    assert diag.boldThreshold() == 0.05

    dot_str = diag.export_dot()
    assert 'digraph' in dot_str
    assert 'CH4' in dot_str

def test_interface_kinetics():
    # Verify InterfaceKinetics class can be instantiated and has expected methods
    surf_sp = [ct.SurfaceSpecies("Pt(s)", 195.08, None, size=1.0)]
    surf_phase = ct.IdealSurfacePhase(surf_sp, site_density=2.7e-9)
    ik = InterfaceKinetics(phases=[surf_phase], reactions=[])

    assert ik.kineticsType() == "InterfaceKinetics"
    assert ik.nReactions() == 0
    assert ik.nTotalSpecies() == 1
    assert ik.nPhases() == 1
    assert ik.siteDensity() == 2.7e-9
    assert ik.electricPotential(0) == 0.0
    ik.setElectricPotential(0, 0.5)
    assert ik.electricPotential(0) == 0.5
