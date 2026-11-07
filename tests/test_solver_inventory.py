"""Guard the evidence-based global solver inventory against stale claims."""
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "audit" / "complete_solver_inventory.jsonl"
STATUS = ROOT / "audit" / "scientific_status.json"


def test_solver_inventory_status_counts_are_current_and_incomplete():
    rows = [json.loads(line) for line in INVENTORY.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    status = json.loads(STATUS.read_text(encoding="utf-8"))

    assert len(rows) == 32
    assert status["pinned_commit"] == "726522be4e2a13454d8415b7ef799d621f665cf3"
    assert Counter(row["status"] for row in rows) == Counter(status["counts"])
    assert status["complete"] is False


def test_exact_pinned_wave_evidence_is_reflected_without_overclaiming():
    rows = {row["subsystem"]: row for row in (
        json.loads(line) for line in INVENTORY.read_text(
            encoding="utf-8").splitlines() if line.strip())}

    assert rows["thermodynamic state inversion"]["status"] == \
        "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert rows["Peng-Robinson roots"]["status"] == \
        "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert rows["Redlich-Kwong roots"]["status"] == \
        "VALIDATED_NUMERICALLY_EQUIVALENT"
    # Root branch evidence does not certify the complete saturation path.
    assert rows["generic fugacity roots"]["status"] == "PARTIAL"
    # Eight TPX fluids are ported; unresolved cp/cv differences preclude closure.
    assert rows["TPX pure fluids"]["status"] == "PARTIAL"
    assert rows["IAPWS water"]["status"] == "PARTIAL"
    assert rows["mixture transport"]["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert rows["porous transport"]["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert rows["high-pressure transport"]["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert rows["VCS equilibrium"]["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert rows["single-phase equilibrium"]["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"


def test_ode_solver_status_discloses_python_kernel_and_open_parity():
    rows = {row["subsystem"]: row for row in (
        json.loads(line) for line in INVENTORY.read_text(
            encoding="utf-8").splitlines() if line.strip())}
    ode = rows["0D ODE integration"]

    assert ode["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert "Python CV_BDF" in ode["python_current_algorithm"]
    assert "optional SciPy and standalone SUNDIALS backends" in ode["python_current_algorithm"]
    assert "CVodesIntegrator" in ode["parity_algorithm"]


def test_dae_solver_status_is_validated_numerically_equivalent():
    rows = {row["subsystem"]: row for row in (
        json.loads(line) for line in INVENTORY.read_text(
            encoding="utf-8").splitlines() if line.strip())}
    dae = rows["DAE integration"]

    assert dae["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert "IdasIntegrator" in dae["parity_algorithm"]
    assert "tools/run_idas_campaign.py" in dae["validation_test"]


def test_flow_reactor_status_is_validated_numerically_equivalent():
    rows = {row["subsystem"]: row for row in (
        json.loads(line) for line in INVENTORY.read_text(
            encoding="utf-8").splitlines() if line.strip())}
    flow = rows["flow reactor DAE"]

    assert flow["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert "IdasIntegrator" in flow["python_current_algorithm"]
    assert "tools/run_flow_reactor_campaign.py" in flow["validation_test"]


def test_band_matrix_status_is_validated_numerically_equivalent():
    rows = {row["subsystem"]: row for row in (
        json.loads(line) for line in INVENTORY.read_text(
            encoding="utf-8").splitlines() if line.strip())}
    band = rows["banded linear algebra"]

    assert band["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert "DGBTRF" in band["python_current_algorithm"]
    assert "tools/run_band_matrix_campaign.py" in band["validation_test"]


def test_multi_newton_status_is_validated_numerically_equivalent():
    rows = {row["subsystem"]: row for row in (
        json.loads(line) for line in INVENTORY.read_text(
            encoding="utf-8").splitlines() if line.strip())}
    newt = rows["1D Newton"]

    assert newt["status"] == "VALIDATED_NUMERICALLY_EQUIVALENT"
    assert "MultiJac" in newt["python_current_algorithm"]
    assert "tools/run_multi_newton_campaign.py" in newt["validation_test"]



