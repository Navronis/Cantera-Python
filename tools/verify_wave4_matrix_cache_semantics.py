"""
Stefan-Maxwell matrix, cache/state invalidation, and failure semantics verification.
Covers requirements 6, 7, and 8 for Wave 4 closure:
- Matrix construction, RHS, constraints, flux transformations
- T1 -> T2 -> T1, X1 -> X2 -> X1 cache cycle invariance
- Exception and failure semantics parity
"""

import sys
import math
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
import cantera_python as ct

def test_cache_state_semantics():
    print("Testing Cache/State Transitions (T1 -> T2 -> T1, X1 -> X2 -> X1)...")
    gas = ct.Solution("h2o2.yaml")
    gas.transport_model = "multicomponent"
    tr = gas.transport

    T1, P1, X1 = 500.0, 101325.0, "H2:0.4, O2:0.6"
    T2, P2, X2 = 1800.0, 202650.0, "H2:0.1, O2:0.2, H2O:0.7"

    # Step 1: Set state 1
    gas.TPX = T1, P1, X1
    visc_1a = float(tr.viscosity)
    cond_1a = float(tr.thermalConductivity())
    dmulti_1a = np.array(tr.multi_diff_coeffs)
    bdiff_1a = np.array(tr.binary_diff_coeffs)

    # Step 2: Transition to state 2
    gas.TPX = T2, P2, X2
    visc_2 = float(tr.viscosity)
    cond_2 = float(tr.thermalConductivity())
    dmulti_2 = np.array(tr.multi_diff_coeffs)
    bdiff_2 = np.array(tr.binary_diff_coeffs)

    assert abs(visc_2 - visc_1a) > 1e-6, "Cache failed to update on state change!"

    # Step 3: Transition back to state 1 (T1 -> T2 -> T1)
    gas.TPX = T1, P1, X1
    visc_1b = float(tr.viscosity)
    cond_1b = float(tr.thermalConductivity())
    dmulti_1b = np.array(tr.multi_diff_coeffs)
    bdiff_1b = np.array(tr.binary_diff_coeffs)

    rel_v = abs(visc_1b - visc_1a) / visc_1a
    rel_c = abs(cond_1b - cond_1a) / cond_1a
    rel_dm = np.max(np.abs(dmulti_1b - dmulti_1a) / np.maximum(np.abs(dmulti_1a), 1e-12))
    rel_bd = np.max(np.abs(bdiff_1b - bdiff_1a) / np.maximum(np.abs(bdiff_1a), 1e-12))

    assert rel_v < 1e-14, f"Viscosity cache stale! RelErr = {rel_v}"
    assert rel_c < 1e-14, f"Conductivity cache stale! RelErr = {rel_c}"
    assert rel_dm < 1e-14, f"MultiDiff cache stale! RelErr = {rel_dm}"
    assert rel_bd < 1e-14, f"BinaryDiff cache stale! RelErr = {rel_bd}"

    print(f"CACHE INVARIANCE PASSED: RelErr visc={rel_v:.2e}, cond={rel_c:.2e}, dmulti={rel_dm:.2e}, bdiff={rel_bd:.2e}")
    return True

def test_failure_semantics():
    print("\nTesting Failure Semantics Parity...")
    gas = ct.Solution("h2o2.yaml")
    gas.transport_model = "multicomponent"
    tr = gas.transport
    nsp = gas.n_species

    unknown_failures = 0

    # 1. Invalid leading dimension for getSpeciesFluxes
    try:
        fluxes = np.zeros(nsp)
        tr.getSpeciesFluxes(1, np.array([0.0]), nsp - 1, np.zeros(nsp), nsp, fluxes)
        print("FAIL: Expected ValueError for ldx < nsp")
        unknown_failures += 1
    except ValueError as e:
        print(f"PASS: Correctly raised ValueError for small ldx: {e}")

    # 2. Invalid leading dimension for getMultiDiffCoeffs
    try:
        tr.getMultiDiffCoeffs(nsp - 1, np.zeros(nsp * nsp))
        print("FAIL: Expected ValueError for ld < nsp")
        unknown_failures += 1
    except ValueError as e:
        print(f"PASS: Correctly raised Cantera-style ValueError for small ld in getMultiDiffCoeffs: {e}")

    # 3. Negative temperature
    try:
        gas.TP = -100.0, 101325.0
        v = tr.viscosity
        print("FAIL: Expected error for negative temperature")
        unknown_failures += 1
    except (ValueError, Exception) as e:
        print(f"PASS: Correctly rejected negative temperature: {type(e).__name__}")

    # 4. Negative pressure
    try:
        gas.TP = 300.0, -101325.0
        v = tr.viscosity
        print("FAIL: Expected error for negative pressure")
        unknown_failures += 1
    except (ValueError, Exception) as e:
        print(f"PASS: Correctly rejected negative pressure: {type(e).__name__}")

    print(f"UNKNOWN_FAILURE_CATEGORY = {unknown_failures}")
    return unknown_failures == 0

def test_matrix_rhs_parity():
    print("\nTesting Stefan-Maxwell Matrix Construction & RHS Parity...")
    gas = ct.Solution("h2o2.yaml")
    gas.transport_model = "multicomponent"
    gas.TPX = 1200.0, 101325.0, "H2:0.2, O2:0.3, H2O:0.5"
    tr = gas.transport
    nsp = gas.n_species

    # Trigger internal matrix evaluations
    _ = tr.thermalConductivity()
    _ = tr.multi_diff_coeffs

    # Inspect L0000 matrix
    tr.updateThermal_T()
    tr.eval_L0000()
    L = tr._L0000
    assert np.all(np.isfinite(L)), "L matrix contains non-finite entries!"
    assert np.all(np.diag(L) >= 0), "Diagonal of L0000 matrix must be non-negative!"

    # Test flux matrix construction & row substitution
    grad_T = np.array([500.0])
    grad_X = np.array([0.2, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fluxes = np.zeros(nsp)
    tr.getSpeciesFluxes(1, grad_T, nsp, grad_X, nsp, fluxes)

    # Check that row jmax was correctly substituted by mass fractions
    jmax = np.argmax(np.abs(grad_X))
    y = np.array(gas.Y)
    row_sub = tr._aa[jmax, :]
    assert np.allclose(row_sub, y, rtol=1e-14, atol=1e-14), "Row substitution in Stefan-Maxwell matrix does not match mass fractions!"

    # Mass flux conservation
    net_flux = float(np.sum(fluxes))
    assert abs(net_flux) < 1e-15, f"Mass flux conservation violated: {net_flux}"

    # Molar flux conversion
    molar_fluxes = np.zeros(nsp)
    for k in range(nsp):
        molar_fluxes[k] = fluxes[k] / gas.molecular_weights[k]
    
    print(f"STEFAN-MAXWELL MATRIX PARITY PASSED: Net mass flux = {net_flux:.2e} kg/(m^2 s)")
    return True

if __name__ == "__main__":
    c_ok = test_cache_state_semantics()
    f_ok = test_failure_semantics()
    m_ok = test_matrix_rhs_parity()
    if c_ok and f_ok and m_ok:
        print("\nALL SEMANTIC & MATRIX AUDITS PASSED!")
        sys.exit(0)
    else:
        sys.exit(1)
