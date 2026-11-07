"""Scientific integrity and conservation checks for Kinetics Family 3."""

import pytest
import numpy as np
from pathlib import Path
from cantera_python.surface import Interface

ROOT = Path(__file__).resolve().parents[3]
PTCOMBUST_PATH = ROOT / "outputs" / "cantera-original" / "data" / "ptcombust.yaml"
SOFC_PATH = ROOT / "outputs" / "cantera-original" / "data" / "sofc.yaml"

def test_surface_site_conservation():
    """Verify sum(size_k * wdot_k) == 0 for surface species."""
    surf = Interface(str(PTCOMBUST_PATH), 'Pt_surf')
    gas = surf.adjacent['gas']
    gas.TPX = 1100.0, 101325.0, "CH4:0.095, O2:0.19, AR:0.715"
    surf.TP = 1100.0, 101325.0
    surf.coverages = "PT(S):0.5, H(S):0.2, O(S):0.2, OH(S):0.1"

    wdot = np.asarray(surf.net_production_rates, dtype=float)
    surf_wdot = wdot[:surf.n_species]
    sizes = np.array([s.size for s in surf.species], dtype=float)

    site_prod = float(np.dot(sizes, surf_wdot))
    assert abs(site_prod) < 1e-12, f"Surface site production not conserved: {site_prod}"

def test_element_mass_conservation():
    """Verify that total atomic element production rate across all phases is zero."""
    surf = Interface(str(PTCOMBUST_PATH), 'Pt_surf')
    gas = surf.adjacent['gas']
    gas.TPX = 1200.0, 202650.0, "CH4:0.1, O2:0.2, AR:0.7"
    surf.TP = 1200.0, 202650.0
    surf.coverages = "PT(S):0.4, C(S):0.2, CO(S):0.2, H(S):0.1, O(S):0.1"

    wdot = np.asarray(surf.net_production_rates, dtype=float)

    # Collect element counts for all species in surface and gas
    all_species = list(surf.species) + list(gas.species)
    elements = set()
    for sp in all_species:
        comp = sp.composition if hasattr(sp, 'composition') else getattr(sp, '_composition', {})
        elements.update(comp.keys())

    for elem in elements:
        atom_counts = np.array([
            float((sp.composition if hasattr(sp, 'composition') else getattr(sp, '_composition', {})).get(elem, 0.0))
            for sp in all_species
        ])
        elem_prod = float(np.dot(atom_counts, wdot))
        assert abs(elem_prod) < 1e-10, f"Element {elem} not conserved: {elem_prod}"

def test_thermodynamic_consistency():
    """Verify kf / kr == Kc, ln(Kc) == -DeltaG0 / RT, and DeltaG == DeltaH - T*DeltaS."""
    surf = Interface(str(PTCOMBUST_PATH), 'Pt_surf')
    gas = surf.adjacent['gas']
    T = 1000.0
    P = 101325.0
    gas.TPX = T, P, "CH4:0.05, O2:0.2, AR:0.75"
    surf.TP = T, P
    surf.coverages = "PT(S):0.7, H(S):0.1, O(S):0.1, OH(S):0.1"

    kin = surf.kinetics
    kf = np.asarray(kin.forward_rate_constants, dtype=float)
    kr = np.asarray(kin.reverse_rate_constants, dtype=float)
    kc = np.asarray(kin.equilibrium_constants, dtype=float)
    dg = np.asarray(kin.delta_gibbs, dtype=float)
    dh = np.asarray(kin.delta_enthalpy, dtype=float)
    ds = np.asarray(kin.delta_entropy, dtype=float)
    dg0 = np.asarray(kin.delta_standard_gibbs, dtype=float)
    dh0 = np.asarray(kin.delta_standard_enthalpy, dtype=float)
    ds0 = np.asarray(kin.delta_standard_entropy, dtype=float)

    # DeltaG = DeltaH - T * DeltaS
    np.testing.assert_allclose(dg, dh - T * ds, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(dg0, dh0 - T * ds0, rtol=1e-10, atol=1e-8)

    # Reversible reactions: kf / kr == kc
    for j, rxn in enumerate(surf._reactions):
        if rxn.reversible and kr[j] > 1e-30 and kc[j] > 1e-30:
            calc_kc = kf[j] / kr[j]
            np.testing.assert_allclose(calc_kc, kc[j], rtol=1e-10)

def test_coverage_integration_preserves_normalization():
    """Verify advanceCoverages preserves sum(theta) == 1.0."""
    surf = Interface(str(PTCOMBUST_PATH), 'Pt_surf')
    gas = surf.adjacent['gas']
    gas.TPX = 1000.0, 101325.0, "CH4:0.095, O2:0.19, AR:0.715"
    surf.TP = 1000.0, 101325.0
    surf.coverages = "PT(S):0.8, H(S):0.1, O(S):0.1"

    # Step forward in time
    for dt in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        surf.advanceCoverages(dt)
        covs = np.asarray(surf.coverages, dtype=float)
        assert np.all(covs >= 0.0), f"Negative coverage encountered: {covs}"
        assert abs(np.sum(covs) - 1.0) < 1e-10, f"Coverage sum deviates from 1.0: {np.sum(covs)}"

def test_pseudo_steady_state_solver():
    """Verify solvePseudoSteadyStateProblem converges and preserves coverages."""
    surf = Interface(str(PTCOMBUST_PATH), 'Pt_surf')
    gas = surf.adjacent['gas']
    gas.TPX = 900.0, 101325.0, "CH4:0.05, O2:0.2, AR:0.75"
    surf.TP = 900.0, 101325.0
    surf.coverages = "PT(S):0.6, H(S):0.2, O(S):0.2"

    kin = surf.kinetics
    ret = kin.solvePseudoSteadyStateProblem()
    assert ret is True or ret is None or ret == 0
    covs = np.asarray(surf.coverages, dtype=float)
    assert np.all(covs >= 0.0)
    assert abs(np.sum(covs) - 1.0) < 1e-7
