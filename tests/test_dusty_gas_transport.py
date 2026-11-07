"""Pinned-equation tests for the pure-Python dusty-gas transport model."""
from pathlib import Path

import numpy as np
import pytest

import cantera_python as ct
from cantera_python.constants import Pi, Tiny


DATA_DIR = Path(__file__).resolve().parents[2] / "cantera-original" / "data"
H2O2_PATH = DATA_DIR / "h2o2.yaml"


@pytest.fixture
def phase():
    gas = ct.DustyGas(H2O2_PATH)
    gas.TPX = 500.0, ct.OneAtm, "O2:2.0, H2:1.0, H2O:1.0"
    gas.porosity = 0.2
    gas.tortuosity = 0.3
    gas.mean_pore_radius = 1e-4
    gas.mean_particle_diameter = 5e-4
    return gas


def test_factory_and_model(phase):
    assert phase.transport_model == "DustyGas"
    assert isinstance(phase.transport, ct.DustyGasTransport)
    assert isinstance(phase.transport.gasTransport(), ct.MultiTransport)


def test_resistance_matrix_matches_pinned_equations(phase):
    transport = phase.transport
    x = np.maximum(Tiny, np.asarray(phase.X))
    binary = (0.2 / 0.3) * np.asarray(
        transport.gasTransport().binary_diffusion_coefficients())
    mw = np.asarray(phase.molecular_weights)
    knudsen = ((2.0 / 3.0) * (1e-4 * 0.2 / 0.3)
                * np.sqrt(8.0 * ct.GasConstant * phase.T / (Pi * mw)))
    resistance = np.empty((phase.n_species, phase.n_species))
    for k in range(phase.n_species):
        resistance[k, :] = -x[k] / binary[k, :]
        resistance[k, k] = (1.0 / knudsen[k]
                            + sum(x[j] / binary[k, j]
                                  for j in range(phase.n_species) if j != k))
    actual = phase.multi_diff_coeffs
    # Independent NumPy and Python factorizations can differ by several ulps.
    # Check the inverse against NumPy and the defining system residual.
    np.testing.assert_allclose(
        actual, np.linalg.inv(resistance), rtol=3e-14, atol=0.0)
    np.testing.assert_allclose(
        resistance @ actual, np.eye(phase.n_species), rtol=0.0, atol=4e-14)


def test_porosity_and_tortuosity_scaling(phase):
    reference = phase.multi_diff_coeffs
    phase.porosity = 0.4
    np.testing.assert_allclose(phase.multi_diff_coeffs, 2.0 * reference, rtol=2e-14)

    phase.porosity = 0.2
    phase.tortuosity = 0.6
    np.testing.assert_allclose(phase.multi_diff_coeffs, 0.5 * reference, rtol=2e-14)


def test_column_major_coefficient_export(phase):
    matrix = phase.multi_diff_coeffs
    leading_dimension = phase.n_species + 3
    output = np.full(leading_dimension * phase.n_species, np.nan)
    phase.getMultiDiffCoeffs(leading_dimension, output)
    for i in range(phase.n_species):
        for j in range(phase.n_species):
            assert output[leading_dimension * j + i] == matrix[i, j]
    with pytest.raises(ValueError, match="ld is too small"):
        phase.getMultiDiffCoeffs(phase.n_species - 1, output)


def test_molar_fluxes_zero_and_species_direction(phase):
    t1, rho1, y1 = phase.TDY
    zero = phase.molar_fluxes(t1, t1, rho1, rho1, y1, y1, 1e-4)
    np.testing.assert_allclose(zero, 0.0, rtol=0.0, atol=0.0)

    phase.TPX = 500.0, ct.OneAtm, "O2:2.0, H2:1.001, H2O:0.999"
    t2, rho2, y2 = phase.T, phase.density, np.asarray(phase.Y)
    flux = phase.molar_fluxes(t1, t2, rho1, rho2, y1, y2, 1e-4)
    assert flux[phase.species_index("H2")] < 0.0
    assert flux[phase.species_index("H2O")] > 0.0
    assert np.all(np.isfinite(flux))


def test_explicit_permeability_changes_pressure_driven_flux(phase):
    y = np.asarray(phase.Y)
    t = phase.T
    rho = phase.density
    reference = phase.molar_fluxes(t, t, rho, 1.01 * rho, y, y, 1e-3)
    phase.permeability = 0.0
    no_darcy = phase.molar_fluxes(t, t, rho, 1.01 * rho, y, y, 1e-3)
    assert not np.allclose(reference, no_darcy, rtol=1e-12, atol=0.0)


def test_thermal_conductivity_delegates_to_multicomponent_gas(phase):
    gas = ct.Solution(H2O2_PATH, transport="multicomponent")
    gas.TPX = phase.TPX
    assert phase.thermal_conductivity == gas.thermal_conductivity


def test_dusty_parameters_are_write_only_like_native_api(phase):
    for name in ("porosity", "tortuosity", "mean_pore_radius",
                 "mean_particle_diameter", "permeability"):
        with pytest.raises(AttributeError):
            getattr(phase, name)


def test_cache_invalidation_and_recompute_semantics(phase):
    transport = phase.transport
    m1 = transport.multi_diff_coeffs.copy()

    # Temperature change
    phase.TP = 800.0, phase.P
    transport.updateTransport_T()
    assert not transport._knudsen_ok
    assert not transport._bulk_ok
    m2 = transport.multi_diff_coeffs
    assert not np.allclose(m1, m2, rtol=1e-5)

    # Composition change
    phase.TPX = 800.0, phase.P, "O2:1.0, H2:3.0"
    transport.updateTransport_C()
    assert not transport._bulk_ok
    m3 = transport.multi_diff_coeffs
    assert not np.allclose(m2, m3, rtol=1e-5)


def test_knudsen_diffusion_direct_evaluation(phase):
    transport = phase.transport
    kd = transport.knudsen_diff_coeffs
    assert len(kd) == phase.n_species
    assert np.all(kd > 0.0)
    # Check square-root molecular weight ratio for O2 and H2
    idx_h2 = phase.species_index("H2")
    idx_o2 = phase.species_index("O2")
    mw_h2 = phase.molecular_weights[idx_h2]
    mw_o2 = phase.molecular_weights[idx_o2]
    ratio = kd[idx_h2] / kd[idx_o2]
    expected_ratio = np.sqrt(mw_o2 / mw_h2)
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-12)


def test_molar_fluxes_array_guards(phase):
    transport = phase.transport
    nsp = phase.n_species
    state1 = np.zeros(nsp + 1)  # Too small
    state2 = np.zeros(nsp + 2)
    fluxes = np.zeros(nsp)
    with pytest.raises(Exception):
        transport.getMolarFluxes(state1, state2, 1e-3, fluxes)

    with pytest.raises(Exception):
        transport.getMolarFluxes(state2, state2, 1e-3, np.zeros(nsp - 1))
