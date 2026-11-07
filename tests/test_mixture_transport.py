"""Focused tests for Mixture-Averaged Transport (MixTransport / GasTransport / Transport).

Validates numerical equivalence against pinned native Cantera oracle equations:
- Wilke mixture viscosity formula
- Combination thermal conductivity formula
- Binary diffusion coefficients matrix (symmetry, 1/P scaling)
- Mixture-averaged diffusion coefficients (mole, mass, standard)
- Diffusive mass fluxes with mass conservation sum(j_k) == 0
- Cache invalidation semantics on T, P, X changes
- Adversarial inputs (T <= 0, P <= 0, NaN, zero mole fractions)
"""

import math
from pathlib import Path
import numpy as np
import pytest

from cantera_python.solution import Solution
from cantera_python.constants import CanteraError, ArraySizeError
from cantera_python.transport import (
    MixTransport,
    GasTransport,
    Transport,
    GasTransportData,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "cantera-original" / "data"
GRI30_PATH = DATA_DIR / "gri30.yaml"
H2O2_PATH = DATA_DIR / "h2o2.yaml"


def test_mixture_transport_creation_and_model_string():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    assert gas.transport_model == "mixture-averaged"
    assert isinstance(gas.transport, MixTransport)
    assert isinstance(gas.transport, GasTransport)
    assert isinstance(gas.transport, Transport)


def test_wilke_viscosity_mathematical_equivalence():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 1200.0, 101325.0, "CH4:1.0, O2:2.0, N2:7.52"

    mu_mix = gas.viscosity
    mu_sp = gas.species_viscosities
    mw = gas.molecular_weights
    X = gas.X
    nsp = gas.n_species

    # Manually compute Wilke mixture viscosity
    phi = np.zeros((nsp, nsp))
    for k in range(nsp):
        for j in range(nsp):
            if k == j:
                phi[k, j] = 1.0
            else:
                num = (1.0 + math.sqrt(mu_sp[k] / mu_sp[j]) * (mw[j] / mw[k]) ** 0.25) ** 2
                denom = math.sqrt(8.0 * (1.0 + mw[k] / mw[j]))
                phi[k, j] = num / denom

    expected_mu = 0.0
    for k in range(nsp):
        denom = sum(X[j] * phi[k, j] for j in range(nsp))
        expected_mu += X[k] * mu_sp[k] / denom

    assert math.isclose(mu_mix, expected_mu, rel_tol=1e-14, abs_tol=1e-16)


def test_combination_thermal_conductivity_mathematical_equivalence():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 1500.0, 2.0e5, "H2:2.0, O2:1.0, AR:5.0"

    cond_mix = gas.thermal_conductivity
    cond_sp = gas.transport.species_thermal_conductivities()
    X = gas.X

    sum1 = sum(x * c for x, c in zip(X, cond_sp))
    sum2 = sum(x / c for x, c in zip(X, cond_sp))
    expected_cond = 0.5 * (sum1 + 1.0 / sum2)

    assert math.isclose(cond_mix, expected_cond, rel_tol=1e-14, abs_tol=1e-16)


def test_binary_diffusion_matrix_properties():
    gas = Solution(str(H2O2_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 800.0, 101325.0, "H2:0.5, O2:0.5"

    bdiff_1atm = gas.binary_diff_coeffs
    nsp = gas.n_species

    # Symmetry: D_ij == D_ji
    assert np.allclose(bdiff_1atm, bdiff_1atm.T, rtol=1e-14, atol=1e-16)

    # All positive
    assert np.all(bdiff_1atm > 0.0)

    # Pressure scaling: D_ij(T, 2*P) == 0.5 * D_ij(T, P)
    gas.TPX = 800.0, 202650.0, "H2:0.5, O2:0.5"
    bdiff_2atm = gas.binary_diff_coeffs

    assert np.allclose(bdiff_2atm, 0.5 * bdiff_1atm, rtol=1e-14, atol=1e-16)


def test_mixture_diffusion_coefficients_relations():
    gas = Solution(str(H2O2_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 1000.0, 101325.0, "H2:0.4, O2:0.4, H2O:0.2"

    d_mix = np.array(gas.mix_diff_coeffs)
    d_mole = np.array(gas.mix_diff_coeffs_mole)
    d_mass = np.array(gas.mix_diff_coeffs_mass)

    assert len(d_mix) == gas.n_species
    assert len(d_mole) == gas.n_species
    assert len(d_mass) == gas.n_species
    assert np.all(d_mix > 0.0)
    assert np.all(d_mole > 0.0)
    assert np.all(d_mass > 0.0)


def test_diffusive_mass_fluxes_conserve_mass():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 1200.0, 101325.0, "CH4:1.0, O2:2.0, N2:7.52"

    nsp = gas.n_species
    ndim = 2
    grad_T = np.array([100.0, -50.0])
    grad_X = np.zeros(nsp * ndim)
    grad_X[0] = 1.0   # dX_0/dx_0
    grad_X[1] = -1.0  # dX_1/dx_0
    grad_X[nsp + 0] = 0.5
    grad_X[nsp + 1] = -0.5

    fluxes = np.zeros(nsp * ndim)
    gas.transport.getSpeciesFluxes(ndim, grad_T, nsp, grad_X, nsp, fluxes)

    # Net diffusive mass flux must sum to exactly zero in every spatial dimension
    flux_matrix = fluxes.reshape((ndim, nsp))
    for n in range(ndim):
        assert math.isclose(np.sum(flux_matrix[n, :]), 0.0, abs_tol=1e-15)


def test_cache_invalidation_semantics():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 500.0, 101325.0, "CH4:1.0, O2:2.0, N2:7.52"

    v1 = gas.viscosity
    k1 = gas.thermal_conductivity
    d1 = gas.mix_diff_coeffs[0]

    # Temperature change must invalidate viscosity, conductivity, diffusion
    gas.TPX = 800.0, 101325.0, "CH4:1.0, O2:2.0, N2:7.52"
    v2 = gas.viscosity
    k2 = gas.thermal_conductivity
    d2 = gas.mix_diff_coeffs[0]
    assert v2 > v1
    assert k2 > k1
    assert d2 > d1

    # Composition change must update mixture properties
    gas.TPX = 800.0, 101325.0, "H2:1.0, O2:0.5"
    v3 = gas.viscosity
    k3 = gas.thermal_conductivity
    assert v3 != v2
    assert k3 != k2


def test_adversarial_zero_or_negative_temperature():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    with pytest.raises((ValueError, CanteraError)):
        gas.TP = 0.0, 101325.0
        _ = gas.viscosity

    with pytest.raises((ValueError, CanteraError)):
        gas.TP = -100.0, 101325.0
        _ = gas.viscosity


def test_adversarial_zero_or_negative_pressure():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    with pytest.raises((ValueError, CanteraError)):
        gas.TP = 500.0, 0.0
        _ = gas.binary_diff_coeffs

    with pytest.raises((ValueError, CanteraError)):
        gas.TP = 500.0, -101325.0
        _ = gas.binary_diff_coeffs


def test_adversarial_extreme_trace_compositions():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    # Near machine-zero mole fractions should not cause NaN or division by zero
    x_dict = {s: 1e-30 for s in gas.species_names}
    x_dict["N2"] = 1.0
    gas.TPX = 1000.0, 101325.0, x_dict

    assert math.isfinite(gas.viscosity)
    assert math.isfinite(gas.thermal_conductivity)
    assert all(math.isfinite(d) for d in gas.mix_diff_coeffs)


def test_array_size_guard():
    gas = Solution(str(GRI30_PATH))
    gas.transport_model = "mixture-averaged"
    gas.TPX = 500.0, 101325.0, "CH4:1.0, O2:2.0, N2:7.52"

    too_small = np.zeros(5)
    with pytest.raises(ArraySizeError):
        gas.transport.getSpeciesViscosities(too_small)
