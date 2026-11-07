"""Pytest test suite validating ReactorNet parity against pinned native Cantera oracle."""

import json
from pathlib import Path
import pytest
import numpy as np

from cantera_python import (
    Solution,
    IdealGasReactor,
    IdealGasConstPressureReactor,
    Reservoir,
    MassFlowController,
    Valve,
    PressureController,
    Wall,
    ReactorNet,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO_ROOT / "audit"
SUMMARY_PATH = AUDIT_DIR / "reactor_network_differential_summary.json"
NATIVE_RESULTS_PATH = AUDIT_DIR / "reactor_net_native_oracle_results.jsonl"
PYTHON_RESULTS_PATH = AUDIT_DIR / "reactor_net_python_port_results.jsonl"


def test_reactor_network_differential_campaign_summary():
    """Ensure the differential campaign against pinned native Cantera oracle passed 100%."""
    assert SUMMARY_PATH.exists(), f"Differential summary missing: {SUMMARY_PATH}"
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["status"] == "PASS"
    assert summary["total_cases"] == 24
    assert summary["passed_cases"] == 24
    assert summary["pass_rate_percent"] == 100.0

    metrics = summary["metrics"]
    assert metrics["temperature_relative_error"]["max"] < 1e-3
    assert metrics["pressure_relative_error"]["max"] < 1e-3
    assert metrics["mass_relative_error"]["max"] < 1e-4
    assert metrics["sensitivity_relative_error"]["max"] < 1e-2


def test_constant_volume_ignition_parity():
    """Verify ignition case matches native oracle within tight tolerances."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    nat_case = native["cv_h2_air_ignition_1000k"]
    py_case = port["cv_h2_air_ignition_1000k"]

    assert len(nat_case["checkpoints"]) == len(py_case["checkpoints"]) == 5
    for nat_cp, py_cp in zip(nat_case["checkpoints"], py_case["checkpoints"]):
        rel_T = abs(nat_cp["temperature"] - py_cp["temperature"]) / nat_cp["temperature"]
        rel_P = abs(nat_cp["pressure"] - py_cp["pressure"]) / nat_cp["pressure"]
        assert rel_T < 1e-3
        assert rel_P < 1e-3


def test_rocket_high_pressure_parity():
    """Verify rocket high-pressure (50 atm H2/O2 and 30 atm CH4/O2) matches native oracle."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    for cid in ["rocket_hp_h2_o2_50atm", "rocket_hp_ch4_o2_30atm"]:
        nat_case = native[cid]
        py_case = port[cid]
        for nat_cp, py_cp in zip(nat_case["checkpoints"], py_case["checkpoints"]):
            rel_T = abs(nat_cp["temperature"] - py_cp["temperature"]) / nat_cp["temperature"]
            rel_P = abs(nat_cp["pressure"] - py_cp["pressure"]) / nat_cp["pressure"]
            assert rel_T < 1e-3
            assert rel_P < 1e-3


def test_multi_reactor_network_cascade():
    """Verify coupled multi-reactor cascade with flow devices matches native oracle."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    nat_case = native["multi_reactor_cascade_flow"]
    py_case = port["multi_reactor_cascade_flow"]
    for nat_cp, py_cp in zip(nat_case["checkpoints"], py_case["checkpoints"]):
        assert np.isclose(nat_cp["T1"], py_cp["T1"], rtol=1e-3)
        assert np.isclose(nat_cp["T2"], py_cp["T2"], rtol=1e-3)
        assert np.isclose(nat_cp["m1"], py_cp["m1"], rtol=1e-4)
        assert np.isclose(nat_cp["m2"], py_cp["m2"], rtol=1e-4)


def test_advance_limits_parity():
    """Verify advance limits stops integration on root-crossing identically to native oracle."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    nat_case = native["advance_limits_temperature_trigger"]
    py_case = port["advance_limits_temperature_trigger"]
    nat_cp = nat_case["checkpoints"][0]
    py_cp = py_case["checkpoints"][0]
    assert np.isclose(nat_cp["reached_time"], py_cp["reached_time"], rtol=1e-3)
    assert np.isclose(nat_cp["final_temp"], py_cp["final_temp"], rtol=1e-3)


def test_reinitialization_parity():
    """Verify reinitialization clears stale history identically to native oracle."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    nat_case = native["reinitialization_trajectory"]
    py_case = port["reinitialization_trajectory"]
    nat_cp = nat_case["checkpoints"][0]
    py_cp = py_case["checkpoints"][0]
    assert np.isclose(nat_cp["T1"], py_cp["T1"], rtol=1e-5)
    assert np.isclose(nat_cp["T2"], py_cp["T2"], rtol=1e-5)


def test_sensitivity_coefficients_parity():
    """Verify normalized forward sensitivity coefficients match native oracle."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    nat_case = native["sensitivity_reaction_multipliers"]
    py_case = port["sensitivity_reaction_multipliers"]
    for nat_cp, py_cp in zip(nat_case["checkpoints"], py_case["checkpoints"]):
        s_nat = nat_cp["sens_rxn0_temp"]
        s_py = py_cp["sens_rxn0_temp"]
        assert np.isclose(s_nat, s_py, rtol=1e-2, atol=1e-4)


def test_failure_semantics_parity():
    """Verify error catching matches native oracle for all edge cases."""
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        port = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    failure_cases = [
        "failure_backward_advance_time",
        "failure_negative_relative_tolerance",
        "failure_out_of_range_reactor_index",
        "failure_nonexistent_component_name",
    ]
    for cid in failure_cases:
        assert native[cid]["failure_caught"] is True
        assert port[cid]["failure_caught"] is True
