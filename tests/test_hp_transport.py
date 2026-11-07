from pathlib import Path

import numpy as np
import pytest

import cantera_python as ct
from cantera_python.transport import (
    HighPressureGasTransport,
    takahashi_correction_factor,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "cantera-original" / "test" / "data"


def test_takahashi_scalar_table_edges_and_interpolation():
    assert takahashi_correction_factor(0.099, 1.2) == 1.0
    # Table rows at reduced pressure 1.0 and 1.2, evaluated at Tr=1.
    at_10 = takahashi_correction_factor(1.0, 1.0)
    at_12 = takahashi_correction_factor(1.2, 1.0)
    at_11 = takahashi_correction_factor(1.1, 1.0)
    assert at_11 == pytest.approx(0.5 * (at_10 + at_12), rel=2e-15)
    assert takahashi_correction_factor(8.0, 1.0) == pytest.approx(
        takahashi_correction_factor(5.0, 1.0), rel=0.0, abs=0.0)


def test_high_pressure_takahashi_factors_match_pinned_source_case():
    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.TPX = 350.0, 2.0e7, "CH4:0.755, CO2:0.245"
    transport = HighPressureGasTransport(gas)
    assert transport.transport_model == "high-pressure"

    np.testing.assert_allclose(transport._tcrit, [190.7, 304.2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(transport._pcrit, [4.63e6, 7.39e6], rtol=0, atol=1e-6)
    transport.updateCorrectionFactors()
    np.testing.assert_allclose(
        transport._takahashi_factors,
        [[0.9329854202723253, 0.8772256889321317],
         [0.8772256889321317, 0.5285432718135198]],
        rtol=2e-14, atol=1e-15,
    )


def test_mixture_averaged_binary_diffusion_uses_pinned_fit():
    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.TPX = 350.0, 2.0e7, "CH4:0.755, CO2:0.245"

    # Extracted from the native transport polynomials and binary_diff_coeffs
    # at pinned Cantera commit 726522be4e2a13454d8415b7ef799d621f665cf3.
    expected = np.array([
        [1.5461116873052805e-07, 1.1621170828659289e-07],
        [1.1621170828659289e-07, 7.607025023060068e-08],
    ])
    np.testing.assert_allclose(gas.transport.binary_diff_coeffs, expected,
                               rtol=1e-12, atol=0.0)


def test_mixture_averaged_viscosity_uses_pinned_species_fits():
    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.TPX = 350.0, 2.0e7, "CH4:0.755, CO2:0.245"

    # Native Cantera mixture viscosity at the pinned commit and state above.
    assert gas.transport.viscosity == pytest.approx(
        1.4803934074984754e-05, rel=2e-14, abs=0.0)


def test_high_pressure_transport_uses_exact_pair_factors():
    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.TPX = 350.0, 2.0e7, "CH4:0.755, CO2:0.245"
    transport = HighPressureGasTransport(gas)
    low_pressure_model = ct.MixTransport(gas)
    low = np.asarray(low_pressure_model.binary_diffusion_coefficients())
    corrected = np.asarray(transport.binary_diffusion_coefficients())
    np.testing.assert_allclose(corrected, low * transport._takahashi_factors,
                               rtol=2e-15, atol=0.0)


def test_high_pressure_binary_diffusion_matches_pinned_native_case():
    path = DATA_DIR / "methane_co2.yaml"
    gas = ct.Solution(path, phase_name="methane_co2")
    gas.TPX = 350.0, 2.0e7, "CH4:0.755, CO2:0.245"
    transport = HighPressureGasTransport.from_mechanism(path, thermo=gas)

    expected = np.array([
        [1.442499662368471e-07, 1.0194389586368636e-07],
        [1.0194389586368636e-07, 4.0206418944554845e-08],
    ])
    np.testing.assert_allclose(transport.binary_diffusion_coefficients(),
                               expected, rtol=2e-14, atol=0.0)


def test_high_pressure_transport_rejects_missing_critical_properties():
    gas = ct.Solution(DATA_DIR / "methane_co2_noCritProp.yaml",
                      phase_name="methane_co2")
    with pytest.raises(ValueError, match="must have critical properties defined"):
        HighPressureGasTransport(gas)


def test_high_pressure_critical_property_accessors_and_model_string():
    path = DATA_DIR / "methane_co2.yaml"
    gas = ct.Solution(path, phase_name="methane_co2")
    gas.TPX = 350.0, 2.0e7, "CH4:0.755, CO2:0.245"
    transport = HighPressureGasTransport(gas)

    assert transport.transportModel() == "HighPressure"
    assert transport.Tcrit_i(0) == pytest.approx(190.7, abs=1e-6)
    assert transport.Tcrit_i(1) == pytest.approx(304.2, abs=1e-6)
    assert transport.Pcrit_i(0) == pytest.approx(4.63e6, abs=1e-1)
    assert transport.Pcrit_i(1) == pytest.approx(7.39e6, abs=1e-1)
    assert transport.Vcrit_i(0) > 0.0
    assert transport.Zcrit_i(0) > 0.0


def test_high_pressure_test_csv_100_points():
    ref_file = DATA_DIR / "HighPressureTest.csv"
    assert ref_file.exists(), f"Reference file {ref_file} not found"
    ref_data = np.loadtxt(ref_file, delimiter=",")
    pressures = ref_data[:, 0]
    expected_visc = ref_data[:, 1]
    expected_cond = ref_data[:, 2]

    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.transport_model = "high-pressure"

    for idx, p in enumerate(pressures):
        gas.TPX = 350.0, p, "CH4:0.755, CO2:0.245"
        assert gas.viscosity == pytest.approx(expected_visc[idx], rel=1e-5)
        assert gas.thermal_conductivity == pytest.approx(expected_cond[idx], rel=1e-5)


def test_high_pressure_chung_test_csv_100_points():
    ref_file = DATA_DIR / "HighPressureChungTest.csv"
    assert ref_file.exists(), f"Reference file {ref_file} not found"
    ref_data = np.loadtxt(ref_file, delimiter=",")
    pressures = ref_data[:, 0]
    expected_visc = ref_data[:, 1]
    expected_cond = ref_data[:, 2]

    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.transport_model = "high-pressure-Chung"

    for idx, p in enumerate(pressures):
        gas.TPX = 350.0, p, "CH4:0.755, CO2:0.245"
        assert gas.viscosity == pytest.approx(expected_visc[idx], rel=1e-5)
        assert gas.thermal_conductivity == pytest.approx(expected_cond[idx], rel=1e-5)


def test_high_pressure_multispecies_co2_rk():
    rk_file = DATA_DIR / "co2_RK_example.yaml"
    assert rk_file.exists()
    gas = ct.Solution(rk_file, phase_name="carbondioxide")
    gas.transport_model = "high-pressure"
    gas.TPX = 400.0, 5.0e6, "CO2:0.99, H2:0.01"

    # Evaluates dense gas properties without errors
    visc = gas.viscosity
    cond = gas.thermal_conductivity
    bin_diff = gas.binary_diff_coeffs
    mix_diff = gas.mix_diff_coeffs

    assert visc > 0.0
    assert cond > 0.0
    assert bin_diff.shape == (7, 7)
    assert len(mix_diff) == 7

    # Check Chung model as well
    gas.transport_model = "high-pressure-Chung"
    assert gas.viscosity > 0.0
    assert gas.thermal_conductivity > 0.0


def test_high_pressure_failure_semantics():
    gas = ct.Solution(DATA_DIR / "methane_co2.yaml", phase_name="methane_co2")
    gas.transport_model = "high-pressure"

    with pytest.raises(Exception):
        gas.TPX = 300.0, -1.0e5, "CH4:1.0"
        _ = gas.viscosity

    with pytest.raises(Exception):
        gas.TPX = -50.0, 1.0e5, "CH4:1.0"
        _ = gas.viscosity

    with pytest.raises(Exception):
        gas.transport_model = "invalid_hp_transport_model"
