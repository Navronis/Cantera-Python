"""Tests for pure-Python Multicomponent Transport (MultiTransport).

Ported directly from:
- outputs/cantera-original/test/python/test_transport.py (test_multicomponent)
- outputs/cantera-original/test/transport/gasTransportTest.cpp
- outputs/cantera-original/test/transport/transportFromScratch.cpp
"""
from pathlib import Path
import json
import math
import numpy as np
import pytest

import cantera_python as ct
from cantera_python.solution import Solution
from cantera_python.constants import GasConstant
from cantera_python.transport import (
    MultiTransport,
    MixTransport,
    _invert_dense_partial_pivot,
    _polyfit_householder_pivoted,
    _solve_dense_partial_pivot,
)

DATA_DIR = Path(__file__).resolve().parents[2] / 'cantera-original' / 'data'
GRI30_PATH = DATA_DIR / 'gri30.yaml'
H2O2_PATH = DATA_DIR / 'h2o2.yaml'
PINNED_TRACE_REFERENCE = (
    Path(__file__).resolve().parent
    / 'fixtures'
    / 'pinned_multitransport_trace_heavy.json'
)


def test_trace_heavy_multitransport_public_properties_match_pinned_native():
    """Internal LU vectors are ill-conditioned; public transport outputs still match."""
    reference = json.loads(PINNED_TRACE_REFERENCE.read_text(encoding='utf-8'))
    assert reference['provenance']['pinned_commit'] == (
        '726522be4e2a13454d8415b7ef799d621f665cf3'
    )
    phase = Solution(GRI30_PATH)
    phase.TPX = (
        reference['state']['temperature_K'],
        reference['state']['pressure_Pa'],
        reference['state']['mole_fractions'],
    )
    assert list(phase.species_names) == reference['state']['species_names']
    phase.transport_model = 'multicomponent'

    expected = reference['native_expected']
    assert math.isclose(
        phase.thermal_conductivity,
        expected['thermal_conductivity_W_m_K'],
        rel_tol=1e-12,
        abs_tol=1e-14,
    )
    np.testing.assert_allclose(
        phase.thermal_diff_coeffs,
        expected['thermal_diffusion_coefficients'],
        rtol=1e-11,
        atol=3e-20,
    )


def test_get_mass_fluxes_leaves_thermo_at_native_midpoint_state():
    """MultiTransport.cpp leaves the thermo object at the averaged state."""
    class Thermo:
        n_species = 2
        molecular_weights = np.array([2.0, 28.0])

        def __init__(self):
            self.temperature = 777.0
            self.pressure = 123456.0
            self.X = np.array([0.25, 0.75])

        def setState_TPX(self, temperature, pressure, mole_fractions):
            self.temperature = float(temperature)
            self.pressure = float(pressure)
            self.X = np.asarray(mole_fractions, dtype=float)

    thermo = Thermo()
    transport = MultiTransport.__new__(MultiTransport)
    transport._thermo = thermo
    transport.getSpeciesFluxes = lambda ndim, grad_t, ldx, grad_x, ldf, fluxes: None

    state1 = np.array([300.0, 1.0, 0.5, 0.5])
    state2 = np.array([500.0, 2.0, 0.9, 0.1])
    fluxes = np.zeros(2)
    transport.getMassFluxes(state1, state2, 0.25, fluxes)

    y1, y2 = state1[2:], state2[2:]
    x1 = y1 / thermo.molecular_weights
    x1 /= x1.sum()
    x2 = y2 / thermo.molecular_weights
    x2 /= x2.sum()
    mean_mw1 = 1.0 / np.sum(y1 / thermo.molecular_weights)
    mean_mw2 = 1.0 / np.sum(y2 / thermo.molecular_weights)
    p1 = state1[1] * (GasConstant / mean_mw1) * state1[0]
    p2 = state2[1] * (GasConstant / mean_mw2) * state2[0]
    assert thermo.temperature == 400.0
    assert thermo.pressure == pytest.approx(0.5 * (p1 + p2))
    np.testing.assert_allclose(thermo.X, 0.5 * (x1 + x2), rtol=0.0, atol=1e-15)


def test_get_mass_fluxes_preserves_midpoint_state_when_flux_solve_fails():
    """The native routine does not roll phase state back after a solver error."""
    class Thermo:
        n_species = 2
        molecular_weights = np.array([2.0, 28.0])

        def __init__(self):
            self.temperature = 777.0
            self.pressure = 123456.0
            self.X = np.array([0.25, 0.75])

        def setState_TPX(self, temperature, pressure, mole_fractions):
            self.temperature = float(temperature)
            self.pressure = float(pressure)
            self.X = np.asarray(mole_fractions, dtype=float)

    thermo = Thermo()
    transport = MultiTransport.__new__(MultiTransport)
    transport._thermo = thermo

    def fail(*args):
        raise RuntimeError("synthetic flux solve failure")

    transport.getSpeciesFluxes = fail
    with pytest.raises(RuntimeError, match="synthetic flux solve failure"):
        transport.getMassFluxes(
            np.array([300.0, 1.0, 0.5, 0.5]),
            np.array([500.0, 2.0, 0.9, 0.1]), 0.25, np.zeros(2))
    assert thermo.temperature == 400.0


class TestMultiTransportPythonAPI:
    """Test suite matching test_multicomponent in test_transport.py."""

    def test_python_partial_pivot_lu_solve_and_inverse(self):
        matrix = np.array([[0.0, 2.0, 1.0], [1.0, 1.0, 0.0],
                           [2.0, 0.0, 1.0]])
        expected = np.array([1.0, 2.0, -1.0])
        rhs = matrix @ expected
        solution = _solve_dense_partial_pivot(
            matrix, rhs, "test partial pivot")
        inverse = _invert_dense_partial_pivot(matrix, "test inverse")

        np.testing.assert_allclose(solution, expected, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(matrix @ inverse, np.eye(3),
                                   rtol=0.0, atol=1e-15)

    def test_python_partial_pivot_lu_rejects_exact_singularity(self):
        with pytest.raises(ValueError, match="singular at pivot"):
            _solve_dense_partial_pivot(
                [[1.0, 2.0], [2.0, 4.0]], [3.0, 6.0],
                "test singular matrix")

    def test_python_pivoted_qr_recovers_weighted_polynomial(self):
        x = np.array([-2.0, -1.0, -0.25, 0.5, 1.5, 2.0])
        y = 3.0 - 2.0 * x + 0.5 * x * x
        coeffs = _polyfit_householder_pivoted(
            x, y, 2, weights=np.array([0.5, 1.0, 2.0, 0.75, 1.5, 0.25]))
        np.testing.assert_allclose(coeffs, [3.0, -2.0, 0.5],
                                   rtol=0.0, atol=2e-15)

    def test_transport_array_contracts_match_native_errors(self):
        phase = ct.Solution(H2O2_PATH)
        phase.transport_model = "multicomponent"
        nsp = phase.n_species
        grad_t = np.zeros(1)
        grad_x = np.zeros(nsp)
        fluxes = np.zeros(nsp)

        with pytest.raises(ValueError, match="ldx is too small"):
            phase.get_species_fluxes(1, grad_t, nsp - 1, grad_x, nsp, fluxes)
        with pytest.raises(ValueError, match="ldf is too small"):
            phase.get_species_fluxes(1, grad_t, nsp, grad_x, nsp - 1, fluxes)
        with pytest.raises(ValueError, match="grad_T array is too small"):
            phase.get_species_fluxes(2, grad_t, nsp, np.zeros(2 * nsp), nsp,
                                     np.zeros(2 * nsp))
        with pytest.raises(ValueError, match="grad_X array is too small"):
            phase.get_species_fluxes(1, grad_t, nsp, grad_x[:-1], nsp, fluxes)
        with pytest.raises(ValueError, match="fluxes array is too small"):
            phase.get_species_fluxes(1, grad_t, nsp, grad_x, nsp, fluxes[:-1])

        with pytest.raises(ValueError, match="ld is too small"):
            phase.get_multi_diff_coeffs(nsp - 1, np.zeros(nsp * nsp))
        with pytest.raises(ValueError, match="diffusion coefficients array is too small"):
            phase.get_multi_diff_coeffs(nsp, np.zeros(nsp * nsp - 1))

    def test_multicomponent_flux_conservation(self):
        phase = ct.Solution(H2O2_PATH)
        assert phase.transport_model == 'mixture-averaged'

        # Switch transport model to multicomponent
        phase.transport_model = 'multicomponent'
        assert phase.transport_model == 'multicomponent'

        Dmulti = phase.multi_diff_coeffs
        assert Dmulti.shape == (phase.n_species, phase.n_species)
        assert np.all(Dmulti.diagonal() == 0)

        # The sum of mass fluxes should equal zero for any consistent vector of mole
        # fraction gradients (that is, where the sum of mole fraction gradients is zero)
        W = np.array(phase.molecular_weights)
        W_mean = phase.mean_molecular_weight
        rho = phase.density

        net_flux = 0.0
        flux_scale = 0.0
        grad_X = np.array([0.1, 0.2, -0.3, 0.4, -0.6, 0.2, 0.1, -0.2, 0.5, -0.4])
        rng = np.random.default_rng(123)

        for _ in range(10):
            rng.shuffle(grad_X)
            for k in range(phase.n_species):
                flux_k = rho * W[k] / W_mean**2 * np.sum(W * Dmulti[k, :] * grad_X)
                net_flux += flux_k
                flux_scale += abs(flux_k)

            assert abs(net_flux) <= 1e-14 * flux_scale

    def test_multicomponent_switch_back(self):
        phase = ct.Solution(H2O2_PATH)
        phase.transport_model = 'multicomponent'
        assert phase.transport_model == 'multicomponent'
        assert isinstance(phase.transport, MultiTransport)

        phase.transport_model = 'mixture-averaged'
        assert phase.transport_model == 'mixture-averaged'
        assert isinstance(phase.transport, MixTransport)

        phase.transport_model = 'none'
        assert phase.transport_model == 'none'
        assert phase.transport is None


class TestGasTransportGTest:
    """Test suite ported from GasTransportTest in gasTransportTest.cpp."""

    @pytest.fixture(autouse=True)
    def setup_gas(self):
        self.soln = ct.Solution(GRI30_PATH)
        self.thermo = self.soln.thermo
        self.nsp = self.soln.n_species

        self.T0 = 1500.0
        self.P0 = 1e5
        self.T1 = 1300.0
        self.dist = 0.1
        self.T2 = 1000.0
        self.T3 = 1200.0

        self.X0 = [
            0.269205, 0.000107082, 1.36377e-09, 4.35475e-10, 4.34036e-06, 0.192249,
            3.59356e-13, 2.78061e-12, 4.7406e-18, 4.12955e-17, 2.58549e-14, 8.96502e-16,
            6.09056e-11, 7.56752e-09, 0.192253, 0.0385036, 1.49596e-08, 2.22378e-08,
            1.43096e-13, 1.45312e-15, 1.96948e-12, 8.41937e-19, 3.18852e-13, 7.93625e-18,
            3.20653e-15, 1.15149e-19, 1.61189e-18, 1.4719e-15, 5.24728e-13, 6.90582e-17,
            6.37248e-12, 5.93728e-11, 2.71219e-09, 2.66645e-06, 6.57142e-11, 9.52453e-08,
            1.26006e-14, 3.49802e-12, 1.19232e-11, 7.17782e-13, 1.85347e-07, 8.25325e-14,
            5.00914e-20, 1.54407e-16, 3.07176e-11, 4.93198e-08, 4.84792e-12, 0.307675,
            0.0, 6.21649e-29, 8.42393e-28, 6.77865e-18, 2.19225e-16
        ]

        self.soln.TPX = self.T0, self.P0, self.X0
        self.soln.transport_model = 'multicomponent'

    def setup_fluxes(self):
        X2 = np.zeros(self.nsp)
        X2[0] = 0.25; X2[5] = 0.17; X2[14] = 0.15; X2[15] = 0.05; X2[47] = 0.38
        X3 = np.zeros(self.nsp)
        X3[0] = 0.27; X3[5] = 0.15; X3[14] = 0.18; X3[15] = 0.06; X3[47] = 0.36

        grad_T = np.array([(self.T2 - self.T1) / self.dist, (self.T3 - self.T1) / self.dist])
        grad_X = np.zeros((self.nsp, 2))
        for k in range(self.nsp):
            grad_X[k, 0] = (X2[k] - self.X0[k]) / self.dist
            grad_X[k, 1] = (X3[k] - self.X0[k]) / self.dist
        return grad_T, grad_X

    def test_multicomponent_diffusion_coefficients(self):
        """Matches TEST_F(GasTransportTest, multicomponentDiffusionCoefficients)."""
        D_H2_X_ref = [
            0.000000e+00, 2.069166e-02, 1.533272e-03, 7.984950e-04, 1.450324e-03,
            1.363695e-03, 7.766991e-04, 7.561488e-04, 1.968227e-03, 1.851414e-03,
            1.681716e-03, 1.681716e-03, 1.576561e-03, 1.486114e-03, 8.951919e-04,
            5.900643e-04, 8.577657e-04, 8.314705e-04, 8.055783e-04, 8.055783e-04,
            7.831462e-04, 9.755145e-04, 9.407016e-04, 9.084341e-04, 8.794914e-04,
            8.454421e-04, 8.193710e-04, 6.691079e-04, 6.070352e-04, 6.070352e-04,
            1.707391e-03, 1.629845e-03, 1.536190e-03, 1.424771e-03, 8.652237e-04,
            8.420368e-04, 5.745784e-04, 5.889965e-04, 8.196214e-04, 9.534937e-04,
            9.135834e-04, 8.834312e-04, 6.690525e-04, 6.009906e-04, 6.009906e-04,
            6.009906e-04, 6.137995e-04, 8.959026e-04, 6.575936e-04, 5.802367e-04,
            5.681411e-04, 5.942564e-04, 5.820537e-04
        ]
        D_X_H2_ref = [
            0.000000e+00, 1.737090e-02, 4.930150e-03, 3.253679e-03, 4.846513e-03,
            3.623689e-03, 3.235537e-03, 3.215876e-03, 4.560725e-03, 5.237593e-03,
            3.543842e-03, 3.543842e-03, 3.472788e-03, 3.462544e-03, 2.547296e-03,
            2.446430e-03, 2.768576e-03, 2.748195e-03, 2.678224e-03, 2.678224e-03,
            2.692416e-03, 2.707123e-03, 2.682446e-03, 2.659347e-03, 2.656733e-03,
            2.437642e-03, 2.419780e-03, 4.081771e-03, 2.345333e-03, 2.345333e-03,
            4.352925e-03, 5.189638e-03, 5.080782e-03, 3.834139e-03, 3.120262e-03,
            3.171634e-03, 2.844790e-03, 2.574628e-03, 3.222283e-03, 3.133609e-03,
            2.740090e-03, 2.717702e-03, 4.081698e-03, 2.584837e-03, 2.584837e-03,
            2.584836e-03, 2.595690e-03, 2.178005e-03, 3.172576e-03, 1.892944e-03,
            1.885366e-03, 2.335541e-03, 2.326147e-03
        ]

        self.soln.TPX = self.T1, self.P0, self.X0
        multi_diff = self.soln.multi_diff_coeffs
        kH2 = self.thermo.species_index("H2")

        for k in range(self.nsp):
            tol = max(1e-6, 0.02 * abs(D_H2_X_ref[k]))
            assert abs(multi_diff[kH2, k] - D_H2_X_ref[k]) < tol, f"D_H2_X k={k}"
            tol2 = max(1e-6, 0.02 * abs(D_X_H2_ref[k]))
            assert abs(multi_diff[k, kH2] - D_X_H2_ref[k]) < tol2, f"D_X_H2 k={k}"

    def test_thermal_diff_coeffs(self):
        """Matches TEST_F(GasTransportTest, thermalDiffCoeffs)."""
        thermal_diff_ref = [
            -1.749421e-06, -5.795523e-10, -1.036581e-14,  2.976174e-15, -2.877580e-11,
            -8.681002e-07,  2.942992e-18,  2.437224e-17, -3.075226e-22, -7.242761e-22,
            -1.227900e-19, -4.193045e-21, -2.609672e-16, -3.238708e-14,  7.585519e-07,
             7.037944e-07,  7.190017e-14,  1.204366e-13,  8.710654e-19,  8.966277e-21,
             1.419255e-17, -2.828291e-22,  1.092672e-18,  2.067526e-23,  1.607623e-20,
             -4.376819e-22, -2.041513e-22,  1.158367e-20,  8.472501e-18,  1.273332e-21,
             -4.408996e-17, -4.943295e-16, -2.097570e-14, -1.207527e-11,  3.627187e-16,
              5.204528e-13,  2.546007e-19,  6.566709e-17,  7.782264e-17,  2.117999e-18,
              6.059605e-13,  3.289296e-19,  5.920676e-22,  2.653811e-21,  5.506497e-16,
              8.840154e-13,  8.195568e-17,  1.155794e-06,  2.282194e-23,  7.479437e-23,
             -1.400893e-22,  1.968555e-22,  4.414863e-21
        ]

        self.soln.TPX = self.T0, self.P0, self.X0
        dt = self.soln.thermal_diff_coeffs

        for k in range(self.nsp):
            tol = max(1e-15, 0.06 * abs(thermal_diff_ref[k]))
            assert abs(dt[k] - thermal_diff_ref[k]) < tol, f"thermal_diff k={k}"

    def test_thermal_conductivity_multi(self):
        """Matches TEST_F(GasTransportTest, thermalConductivity_multi)."""
        cond_ref = [
            6.336474e-02, 7.586649e-02, 8.779171e-02, 9.964744e-02, 1.116838e-01,
            1.237784e-01, 1.353296e-01, 1.472053e-01, 1.589833e-01, 1.706627e-01
        ]
        for i in range(10):
            T = 400.0 + 100.0 * i
            self.soln.TP = T, self.P0
            cond = self.soln.thermal_conductivity
            assert abs(cond - cond_ref[i]) / cond_ref[i] < 0.15, f"thermalCond T={T}"

    def test_get_species_fluxes_multi(self):
        """Matches TEST_F(GasTransportTest, getSpeciesFluxes_multi)."""
        grad_T, grad_X = self.setup_fluxes()

        self.soln.TPX = self.T1, self.P0, self.X0
        flat_fluxes = np.zeros(self.nsp * 2)
        self.soln.get_species_fluxes(2, grad_T, self.nsp, grad_X.flatten(order='F'), self.nsp, flat_fluxes)
        flux0 = flat_fluxes[:self.nsp]
        flux1 = flat_fluxes[self.nsp:]

        # Enforce sum of mass fluxes == 0
        assert abs(np.sum(flux0)) < 1e-15
        assert abs(np.sum(flux1)) < 1e-15

    def test_get_fluxes_mass_and_molar_consistency(self):
        """Matches TEST_F(GasTransportTest, getFluxes_multi)."""
        fluxS = np.zeros(self.nsp)
        fluxMass = np.zeros(self.nsp)
        fluxMole = np.zeros(self.nsp)

        self.soln.TPX = self.T2, self.P0, "H2:0.25, H:0.0001, H2O:0.17, CO:0.15, CO2:0.05, NO:0.001, N2: 0.38"
        state2 = [self.T2, self.thermo.density] + list(self.thermo.Y)
        X2 = np.array(self.thermo.X)

        self.soln.TPX = self.T3, self.P0, "H2:0.27, H2O:0.18, CO:0.13, CO2:0.04, N2: 0.38"
        state3 = [self.T3, self.thermo.density] + list(self.thermo.Y)
        X3 = np.array(self.thermo.X)

        grad_T = np.array([(self.T3 - self.T2) / self.dist])
        X1 = 0.5 * (X2 + X3)
        grad_X = (X3 - X2) / self.dist

        self.soln.TPX = 0.5 * (self.T2 + self.T3), self.P0, X1
        self.soln.get_species_fluxes(1, grad_T, self.nsp, grad_X, self.nsp, fluxS)
        self.soln.get_mass_fluxes(state2, state3, self.dist, fluxMass)
        self.soln.get_molar_fluxes(state2, state3, self.dist, fluxMole)

        for k in range(self.nsp):
            Wk = self.thermo.molecular_weight(k)
            tol = max(1e-14, 1e-4 * abs(fluxS[k]))
            assert abs(fluxMass[k] - fluxS[k]) < tol, f"mass flux k={k}"
            assert abs(fluxMole[k] * Wk - fluxS[k]) < tol, f"molar flux k={k}"

        assert abs(np.sum(fluxMass)) < 1e-15
