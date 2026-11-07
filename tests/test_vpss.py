import math
import numpy as np
import pytest
from cantera_python.constants import GasConstant, OneAtm
from cantera_python.vpss import (
    VPStandardStateTP,
    MolalityVPSSTP,
    MargulesVPSSTP,
    RedlichKisterVPSSTP,
    PDSS_HKF,
    LatticeSolidPhase,
    VPSSError,
)
from cantera_python.electrolytes import HMWSoln

def test_margules_vpsstp_thermo():
    # Symmetric Margules binary mixture of species A and B
    phase = MargulesVPSSTP(["A", "B"], temperature=300.0, pressure=OneAtm)
    h0 = 10000.0  # J/kmol
    phase.setBinaryInteraction("A", "B", h0=h0)
    phase.set_mole_fractions([0.5, 0.5])

    # G^E = X_A * X_B * h0 = 0.5 * 0.5 * 10000 = 2500 J/kmol
    GE_expected = 0.25 * h0
    assert np.isclose(phase.gibbs_excess(), GE_expected, rtol=1e-6)

    # ln(gamma_A) = h0 * X_B^2 / (RT) = 10000 * 0.25 / (R * 300)
    RT = GasConstant * 300.0
    gamma_A_expected = math.exp(0.25 * h0 / RT)
    gamma_B_expected = math.exp(0.25 * h0 / RT)

    gamma = phase.getActivityCoefficients()
    assert np.isclose(gamma[0], gamma_A_expected, rtol=1e-5)
    assert np.isclose(gamma[1], gamma_B_expected, rtol=1e-5)


def test_margules_log_activity_mole_derivative_matches_finite_difference():
    phase = MargulesVPSSTP(["A", "B", "C"], temperature=420.0,
                           pressure=OneAtm + 2.0e6)
    phase.setBinaryInteraction("A", "B", h0=8200.0, s0=1.7,
                               h1=-600.0, s1=0.21, vh0=2.0e-5,
                               vs0=2.0e-8, vh1=-1.0e-5, vs1=-1.0e-8)
    phase.setBinaryInteraction("A", "C", h0=-3100.0, s0=-0.4,
                               h1=1200.0, s1=0.13)
    phase.setBinaryInteraction("B", "C", h0=5500.0, s0=0.8,
                               h1=900.0, s1=-0.17)
    amounts = np.array([0.7, 1.4, 2.1])
    phase.set_mole_fractions(amounts / amounts.sum())

    analytic = np.zeros((3, 3))
    phase.getdlnActCoeffdlnN(analytic)
    ln_gamma = np.log(phase.getActivityCoefficients())
    assert np.isclose(phase.gibbs_excess(),
                      GasConstant * phase.temperature() * np.dot(phase._X, ln_gamma),
                      rtol=2.0e-14, atol=1.0e-10)
    step = 1.0e-6
    for j in range(3):
        perturbed = amounts.copy()
        perturbed[j] += step
        phase.set_mole_fractions(perturbed / perturbed.sum())
        plus = np.log(phase.getActivityCoefficients())
        perturbed[j] -= 2.0 * step
        phase.set_mole_fractions(perturbed / perturbed.sum())
        minus = np.log(phase.getActivityCoefficients())
        phase.set_mole_fractions(amounts / amounts.sum())
        # The model contract is n_j * d(ln gamma_i) / dn_j.
        numerical = amounts[j] * (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(analytic[:, j], numerical,
                                   rtol=2.0e-7, atol=2.0e-10)

    # VCS rescales the logarithmic mole derivative by N / n_j. Confirm the
    # phase-local adapter applies that native conversion for a positive state.
    from cantera_python.vcs import vcs_VolPhase
    from types import SimpleNamespace
    adapter = object.__new__(vcs_VolPhase)
    adapter._ideal = False
    adapter._tp = SimpleNamespace(_thermo=phase)
    adapter._species_names = ("A", "B", "C")
    adapter._x = amounts / amounts.sum()
    adapter._total_moles = float(amounts.sum())
    jacobian = adapter.np_dLnActCoeffdMolNumber()
    expected = analytic * amounts.sum() / amounts[np.newaxis, :]
    np.testing.assert_allclose(jacobian, expected, rtol=1.0e-14, atol=1.0e-14)

def test_redlich_kister_vpsstp():
    phase = RedlichKisterVPSSTP(["A", "B"], temperature=300.0, pressure=OneAtm)
    # A0 = 8000 J/kmol
    phase.setBinaryInteraction("A", "B", coeffs_H=[8000.0])
    phase.set_mole_fractions([0.6, 0.4])

    # G^E = X_A * X_B * A0 = 0.6 * 0.4 * 8000 = 1920 J/kmol
    assert np.isclose(phase.gibbs_excess(), 0.24 * 8000.0, rtol=1e-6)
    gamma = phase.getActivityCoefficients()
    assert len(gamma) == 2
    assert gamma[0] > 1.0 and gamma[1] > 1.0


def test_redlich_kister_source_activity_jacobian_and_vcs_scaling():
    phase = RedlichKisterVPSSTP(["A", "B"], temperature=300.0,
                                pressure=OneAtm)
    phase.setBinaryInteraction("A", "B", coeffs_H=[8000.0])
    x = np.array([0.6, 0.4])
    phase.set_mole_fractions(x)
    ln_gamma = np.log(phase.getActivityCoefficients())
    expected_ln_gamma = np.array([x[1] ** 2, x[0] ** 2]) * 8000.0 / (
        GasConstant * phase.temperature())
    np.testing.assert_allclose(ln_gamma, expected_ln_gamma, rtol=5.0e-13)

    # Match the pinned s_update_dlnActCoeff_dX_ result for the m=0 branch.
    derivative = np.zeros((2, 2))
    phase.getdlnActCoeffdlnN(derivative)
    native_matrix = 8000.0 * np.array([[-x[1], x[1]], [x[0], -x[0]]])
    np.testing.assert_allclose(derivative, native_matrix, rtol=1e-14)

    from cantera_python.vcs import vcs_VolPhase
    from types import SimpleNamespace
    adapter = object.__new__(vcs_VolPhase)
    adapter._ideal = False
    adapter._tp = SimpleNamespace(_thermo=phase)
    adapter._species_names = ("A", "B")
    adapter._x = x.copy()
    adapter._total_moles = 5.0
    np.testing.assert_allclose(
        adapter.np_dLnActCoeffdMolNumber(),
        native_matrix * 5.0 / (5.0 * x[np.newaxis, :]), rtol=1e-14)


def test_redlich_kister_higher_order_polynomial_matches_pinned_equations():
    temperature = 430.0
    phase = RedlichKisterVPSSTP(["A", "B", "C"], temperature=temperature,
                                pressure=OneAtm)
    enthalpy = [8000.0, -1200.0, 350.0]
    entropy = [1.2, -0.2, 0.04]
    phase.setBinaryInteraction("A", "B", coeffs_H=enthalpy,
                               coeffs_S=entropy)
    x = np.array([0.2, 0.3, 0.5])
    phase.set_mole_fractions(x)

    # Independent scalar transcription of the pinned m=0..2 polynomial sums.
    dx = x[0] - x[1]
    scaled = [(h - temperature * s) / (GasConstant * temperature)
              for h, s in zip(enthalpy, entropy)]
    raw = [h - temperature * s for h, s in zip(enthalpy, entropy)]
    total = sum(a * dx**m for m, a in enumerate(scaled))
    first = sum(m * scaled[m] * dx**(m - 1) for m in range(1, 3))
    weighted = sum((m + 1) * scaled[m] * dx**m for m in range(3))
    xa, xb = x[:2]
    xaxb = xa * xb
    expected_ln_gamma = np.array([
        (1.0 - xa) * xb * total + xaxb * first * (1.0 - xa + xb),
        (1.0 - xb) * xa * total + xaxb * first * (-1.0 + xb - xa),
        -xaxb * weighted,
    ])
    np.testing.assert_allclose(np.log(phase.getActivityCoefficients()),
                               expected_ln_gamma, rtol=2e-14, atol=1e-16)

    # Check every Jacobian column against the pinned source's polynomial sums,
    # including both non-component rows and the m >= 2 second derivative.
    s0 = sum(raw[m] * dx**m for m in range(3))
    s1 = sum(m * raw[m] * dx**(m - 1) for m in range(1, 3))
    s2 = sum((m + 1) * raw[m] * dx**m for m in range(3))
    s2_m1 = sum(m * (m + 1) * raw[m] * dx**(m - 1)
                for m in range(1, 3))
    s1_m2 = sum(m * (m - 1) * raw[m] * dx**(m - 2)
                for m in range(2, 3))
    expected_jacobian = np.zeros((3, 3))
    expected_jacobian[0, 0] = (
        -xb * s0 + (1.0 - xa) * xb * s1
        + xb * s1 * (1.0 - 2.0 * xa + xb)
        + xa * xb * s1_m2 * (1.0 - xa + xb))
    expected_jacobian[0, 1] = (
        (1.0 - xa) * s0 - (1.0 - xa) * xb * s1
        + xa * s1 * (1.0 + 2.0 * xb - xa)
        - xa * xb * s1_m2 * (1.0 - xa + xb))
    expected_jacobian[1, 0] = (
        (1.0 - xb) * s0 + (1.0 - xa) * xb * s1
        + xb * s1 * (1.0 - 2.0 * xa + xb)
        + xa * xb * s1_m2 * (1.0 - xa + xb))
    expected_jacobian[1, 1] = (
        -xa * s0 - (1.0 - xb) * xa * s1
        + xa * s1 * (xb - xa - (1.0 - xb))
        - xa * xb * s1_m2 * (-xa - (1.0 - xb)))
    expected_jacobian[2, 0] = -xb * s2 - xa * xb * s2_m1
    expected_jacobian[2, 1] = -xa * s2 + xa * xb * s2_m1
    jacobian = np.zeros((3, 3))
    phase.getdlnActCoeffdlnN(jacobian)
    np.testing.assert_allclose(jacobian, expected_jacobian,
                               rtol=2e-14, atol=1e-12)

def test_molality_vpsstp():
    phase = MolalityVPSSTP(["H2O", "Na+", "Cl-", "H+"], solvent_name="H2O")
    phase.setMolalitiesByName({"Na+": 1.5, "Cl-": 1.5, "H+": 1e-4})

    m = phase.getMolalities()
    assert np.isclose(m[phase.species_index("Na+")], 1.5, rtol=1e-5)
    assert np.isclose(m[phase.species_index("Cl-")], 1.5, rtol=1e-5)
    assert np.isclose(phase.molality("H2O"), 0.0)

    # Activity convention should be molality (1)
    assert phase.activityConvention() == 1
    # pH should be around 4.0
    assert np.isclose(phase.pH(), 4.0, atol=0.1)

def test_pdss_hkf():
    # Na+ ion HKF parameters
    hkf = PDSS_HKF("Na+", charge=1,
                   a1=0.1839, a2=-228.5, a3=3.256, a4=-27260.0,
                   c1=18.18, c2=-29810.0, omega_pr=33060.0)
    hkf.set_state_TP(298.15, OneAtm)
    v0 = hkf.molarVolume()
    assert np.isfinite(v0)
    cp = hkf.cp_R()
    assert np.isfinite(cp)

def test_lattice_solid_phase():
    phase = LatticeSolidPhase(["Li_site", "Vacant_site"], site_density=20.0)
    assert np.isclose(phase.siteDensity(), 20.0)
    assert np.isclose(phase.molarVolume(), 0.05)

def test_hmw_soln_ternary_and_derivatives():
    soln = HMWSoln(cations=["Na+", "K+"], anions=["Cl-", "SO4-2"])
    soln.set_charge("SO4-2", -2)
    soln.set_molalities({"Na+": 1.0, "K+": 0.5, "Cl-": 1.5, "SO4-2": 0.25})

    # Set ternary parameters
    soln.set_theta("Na+", "K+", theta=0.012)
    soln.set_psi("Na+", "K+", "Cl-", psi=-0.0018)

    # Check Debye-Huckel temperature/pressure derivatives
    AL = soln.A_L()
    AJ = soln.A_J()
    AV = soln.A_V()
    assert np.isfinite(AL)
    assert np.isfinite(AJ)
    assert np.isfinite(AV)

    # Check single-ion activity coefficients
    gammas = soln.single_ion_activity_coefficients()
    for ion in ("Na+", "K+", "Cl-", "SO4-2"):
        assert ion in gammas
        assert gammas[ion] > 0.0
