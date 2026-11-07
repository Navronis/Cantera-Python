"""Comprehensive unit tests for Thermodynamics Family 4: Ideal, Solid, Surface, Edge, and Condensed Phase Models."""
import pytest
import numpy as np
from cantera_python import (
    Solution,
    IdealGasPhase,
    IdealSolidSolnPhase,
    SingleSpeciesTP,
    SurfPhase,
    CoverageDependentSurfPhase,
    EdgePhase,
    MetalPhase,
    StoichSubstance,
)

def test_ideal_gas_phase_thermo_consistency():
    gas = Solution("gri30.yaml", "gri30").thermo
    gas.setState_TPX(1200.0, 2.0 * 101325.0, "CH4:0.1, O2:0.2, N2:0.7")

    # Compressible check
    assert gas.isCompressible() is True
    assert gas.isIdeal() is True

    # Equation of state: P = rho * R * T / mw
    P = gas.pressure()
    rho = gas.density()
    mw = gas.meanMolecularWeight()
    T = gas.temperature()
    assert abs(P - rho * 8314.46261815324 * T / mw) / P < 1e-12

    # Thermodynamic identity: H = U + P * V
    h_mol = gas.enthalpy_mole()
    u_mol = gas.intEnergy_mole()
    v_mol = gas.molarVolume()
    assert abs(h_mol - (u_mol + P * v_mol)) / abs(h_mol) < 1e-12

    # Thermodynamic identity: G = H - T * S
    g_mol = gas.gibbs_mole()
    s_mol = gas.entropy_mole()
    assert abs(g_mol - (h_mol - T * s_mol)) / abs(g_mol) < 1e-12

    # Partial molar consistency: sum(X_k * hbar_k) == h
    hbar = np.zeros(gas.nSpecies())
    gas.getPartialMolarEnthalpies(hbar)
    assert abs(gas.mean_X(hbar) - h_mol) / abs(h_mol) < 1e-12

    # Chemical potentials: sum(X_k * mu_k) == g
    mu = np.zeros(gas.nSpecies())
    gas.getChemPotentials(mu)
    assert abs(gas.mean_X(mu) - g_mol) / abs(g_mol) < 1e-12

def test_ideal_solid_soln_phase():
    solid = Solution("thermo-models.yaml", "IdealSolidSolnPhase").thermo
    assert isinstance(solid, IdealSolidSolnPhase)
    solid.setState_TP(1100.0, 5.0 * 101325.0)

    # Incompressible check
    assert solid.isCompressible() is False
    assert solid.isIdeal() is True

    # Thermodynamic identity: H = U + P * V
    P = solid.pressure()
    v_mol = solid.molarVolume()
    h_mol = solid.enthalpy_mole()
    u_mol = solid.intEnergy_mole()
    assert abs(h_mol - (u_mol + P * v_mol)) / abs(h_mol) < 1e-12

    # Thermodynamic identity: G = H - T * S
    T = solid.temperature()
    g_mol = solid.gibbs_mole()
    s_mol = solid.entropy_mole()
    assert abs(g_mol - (h_mol - T * s_mol)) / abs(g_mol) < 1e-12

    # Partial molar consistency
    hbar = np.zeros(solid.nSpecies())
    solid.getPartialMolarEnthalpies(hbar)
    assert abs(solid.mean_X(hbar) - h_mol) / abs(h_mol) < 1e-12

    mu = np.zeros(solid.nSpecies())
    solid.getChemPotentials(mu)
    assert abs(solid.mean_X(mu) - g_mol) / abs(g_mol) < 1e-12

def test_stoich_substance_and_single_species():
    stoich = Solution("thermo-models.yaml", "NaCl(s)").thermo
    assert isinstance(stoich, (StoichSubstance, SingleSpeciesTP))
    assert stoich.nSpecies() == 1
    stoich.setState_TP(500.0, 101325.0)

    assert stoich.isCompressible() is False
    assert stoich.moleFraction(0) == 1.0

    # For pure substance, mu = g
    mu = np.zeros(1)
    stoich.getChemPotentials(mu)
    assert abs(mu[0] - stoich.gibbs_mole()) / abs(stoich.gibbs_mole()) < 1e-12

    # Mass to mole consistency
    mw = stoich.meanMolecularWeight()
    assert abs(stoich.enthalpy_mass() * mw - stoich.enthalpy_mole()) < 1e-10
    assert abs(stoich.entropy_mass() * mw - stoich.entropy_mole()) < 1e-10

def test_surf_phase():
    surf = Solution("surface-phases.yaml", "Pt-surf").thermo
    assert isinstance(surf, SurfPhase)
    assert surf.molarVolume() == 0.0

    surf.setState_TP(900.0, 101325.0)
    cov = np.zeros(surf.nSpecies())
    cov[0] = 0.7
    cov[1] = 0.3
    surf.setCoverages(cov)

    curr_cov = np.zeros(surf.nSpecies())
    surf.getCoverages(curr_cov)
    assert np.allclose(curr_cov, cov, atol=1e-14)

    # Site density must be positive
    assert surf.siteDensity() > 0.0

    # Thermodynamic identity: G = H - T * S
    T = surf.temperature()
    h_mol = surf.enthalpy_mole()
    s_mol = surf.entropy_mole()
    g_mol = surf.gibbs_mole()
    assert abs(g_mol - (h_mol - T * s_mol)) / abs(g_mol) < 1e-12

def test_coverage_dependent_surf_phase():
    covdep = Solution("copt_covdepsurf_example.yaml", "covdep").thermo
    assert isinstance(covdep, (CoverageDependentSurfPhase, SurfPhase))
    covdep.setState_TP(800.0, 101325.0)

    # Coverages should sum to 1
    cov = np.zeros(covdep.nSpecies())
    cov[0] = 0.5
    cov[1] = 0.5
    covdep.setCoverages(cov)

    h_mol = covdep.enthalpy_mole()
    s_mol = covdep.entropy_mole()
    g_mol = covdep.gibbs_mole()
    T = covdep.temperature()
    assert abs(g_mol - (h_mol - T * s_mol)) / abs(g_mol) < 1e-12

def test_edge_phase():
    edge = Solution("surface-phases.yaml", "TPB").thermo
    assert isinstance(edge, (EdgePhase, SurfPhase))
    edge.setState_TP(1000.0, 101325.0)
    assert edge.siteDensity() > 0.0

    mu = np.zeros(edge.nSpecies())
    edge.getChemPotentials(mu)
    assert len(mu) == edge.nSpecies()

def test_metal_phase():
    metal = Solution("thermo-models.yaml", "Metal").thermo
    assert isinstance(metal, (MetalPhase, SingleSpeciesTP))
    assert metal.nSpecies() == 1
    metal.setState_TP(300.0, 101325.0)
    assert metal.density() > 0.0
    assert abs(metal.moleFraction(0) - 1.0) < 1e-14

def test_invalid_state_failure_semantics():
    gas = Solution("gri30.yaml", "gri30").thermo
    with pytest.raises(Exception):
        gas.setState_TP(-50.0, 101325.0)

    with pytest.raises(Exception):
        gas.setState_TP(300.0, -100.0)
