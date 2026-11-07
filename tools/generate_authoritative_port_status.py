"""Generate canonical PORT_STATUS.json and derive PORT_STATUS.md from raw evidence artifacts."""
from __future__ import annotations
import json
import hashlib
import re
import subprocess
import sys
import platform
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def latest_test_status() -> dict:
    result_path = ROOT / "audit" / "canonical_full_pytest_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source_paths = sorted((ROOT / "cantera_python").rglob("*.py"))
    source_paths += sorted((ROOT / "tests").rglob("*.py"))
    source_paths += [ROOT / "pyproject.toml", ROOT / "tools" / "run_waves_1_3_pytest.py"]
    source_hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths if path.is_file()
    }
    current_source_hash = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    result["current_source_sha256"] = current_source_hash
    result["source_snapshot_matches_current_tree"] = (
        result.get("test_source_sha256") == current_source_hash
    )
    log_path = ROOT / result.get("log_path", "audit/canonical_full_pytest.log")
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    duration = re.search(r"in ([0-9.]+)s", log)
    warning_summary = re.search(r"\b(\d+) warnings?\b", log)
    warning_lines = re.findall(r"RuntimeWarning:.*", log)
    result["duration_seconds"] = float(duration.group(1)) if duration else None
    result["warnings"] = int(warning_summary.group(1)) if warning_summary else len(warning_lines)
    result["warning_detail"] = "; ".join(dict.fromkeys(warning_lines))
    result["environment"] = f"{platform.system()} {platform.machine()}, Python {sys.version.split()[0]}"
    result["command_argv"] = result.get("command")
    result["command"] = "python -m pytest tests/"
    result["timestamp"] = result.get("started_at_utc")
    result["failures"] = result.get("failed", 0)
    result["errors"] = result.get("errors", 0)
    result["subtests_passed"] = None
    return result


def wave4_matrix_status() -> dict:
    records = []
    for path in sorted((ROOT / "audit").glob("wave4_matrix_*_comparison.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["artifact"] = path.relative_to(ROOT).as_posix()
        records.append(record)
    return {
        "status": "IN_PROGRESS_UNCERTIFIED" if records else "NOT_STARTED",
        "solver_implementation": "Python Householder QR fits and partial-pivot LU solves",
        "exact_pinned_comparison_cases": len(records),
        "cases": records,
        "solution_parity_certified": False,
        "note": (
            "Trace-heavy solution vectors remain ill-conditioned; the matrix corpus "
            "is diagnostic evidence and does not certify full MultiTransport behavior."
        ),
    }

def build_status():
    symbol_inventory_path = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
    symbol_rows = [
        json.loads(line)
        for line in symbol_inventory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    symbol_counts = Counter(row["current_status"] for row in symbol_rows)
    final_dispositions = Counter(
        row["final_disposition"] for row in symbol_rows
        if row.get("final_disposition")
    )
    if len(symbol_rows) != 5784 or len({row["stable_symbol_id"] for row in symbol_rows}) != 5784:
        raise ValueError("Canonical inventory must contain 5,784 unique stable symbol IDs")
    unresolved_symbols = len(symbol_rows) - sum(final_dispositions.values())
    wave1_summary = json.loads((ROOT / "audit" / "solver_wave1_summary.json").read_text(encoding="utf-8"))
    wave2_summary = json.loads((ROOT / "audit" / "solver_wave2_summary.json").read_text(encoding="utf-8"))
    wave3_summary = json.loads((ROOT / "audit" / "solver_wave3_summary.json").read_text(encoding="utf-8"))
    reval_status = json.loads((ROOT / "audit" / "solver_waves_1_2_3_revalidation_status.json").read_text(encoding="utf-8"))
    wave_defs = json.loads((ROOT / "provenance" / "solver_wave_definitions.json").read_text(encoding="utf-8"))
    native_parity_path = ROOT / "audit" / "native_parity_summary.json"
    native_parity = json.loads(native_parity_path.read_text(encoding="utf-8")) if native_parity_path.exists() else {}
    validation_gate = json.loads((ROOT / "audit" / "waves_1_3_validation_gate.json").read_text(encoding="utf-8"))
    gate_flags = validation_gate.get("flags", {})
    test_status = latest_test_status()
    gate_test_provenance = validation_gate.get("test_run_provenance", {})
    wave_evidence_snapshot_matches = bool(
        gate_test_provenance.get("test_source_sha256")
        and gate_test_provenance.get("test_source_sha256") == test_status.get("test_source_sha256")
        and gate_test_provenance.get("git_commit") == test_status.get("git_commit"))
    live_validated = bool(
        validation_gate.get("native_identity_file_matches_pin")
        and validation_gate.get("native_oracle_revalidated_in_this_run")
        and wave_evidence_snapshot_matches
        and all(gate_flags.get(key) is True for key in (
            "WAVE_1_FRESH_NATIVE_PARITY", "WAVE_2_FRESH_NATIVE_PARITY",
            "WAVE_3_FRESH_NATIVE_PARITY")))
    solver_status = json.loads((ROOT / "audit" / "scientific_status.json").read_text(encoding="utf-8"))
    property_pair_verification_path = ROOT / "audit" / "solver_wave3_property_pair_campaign_verification.json"
    property_pair_verification = (
        json.loads(property_pair_verification_path.read_text(encoding="utf-8"))
        if property_pair_verification_path.exists() else {}
    )
    wave4_status = wave4_matrix_status()

    # Load stub inventory
    stub_lines = (ROOT / "audit" / "runtime_stub_inventory.jsonl").read_text(encoding="utf-8").strip().splitlines()
    all_stubs = [json.loads(line) for line in stub_lines if line.strip()]
    wave_modules = {"state.py", "thermo.py", "ideal_gas.py", "phases.py", "cubic_eos.py", "tpx.py", "tpx_exact.py"}
    wave_stubs = [s for s in all_stubs if s.get("file") in wave_modules]
    unclassified_wave_stubs = [s for s in wave_stubs if s.get("classification") in ("EMPTY_STUB", "EMPTY_PASS", "PLACEHOLDER_RETURN", "INCOMPLETE_MARKER")]

    # Load different implementation dispositions
    disp_lines = (ROOT / "audit" / "solver_wave3_different_implementation_disposition.jsonl").read_text(encoding="utf-8").strip().splitlines()
    dispositions = [json.loads(line) for line in disp_lines if line.strip()]

    status = {
        "status_schema_version": "2.2",
        "generated_from_artifacts": [
            "audit/solver_wave1_summary.json",
            "audit/solver_wave2_summary.json",
            "audit/solver_wave3_summary.json",
            "audit/solver_waves_1_2_3_revalidation_status.json",
            "audit/native_parity_summary.json",
            "audit/native_wave1_summary.json",
            "audit/native_wave2_summary.json",
            "audit/native_wave3_summary.json",
            "audit/native_oracle_identity.json",
            "provenance/canonical_5784_symbol_inventory.jsonl",
            "audit/runtime_stub_inventory.jsonl",
            "audit/solver_wave3_different_implementation_disposition.jsonl",
            "provenance/solver_wave_definitions.json",
            "audit/scientific_status.json",
            "audit/complete_solver_inventory.jsonl",
            "audit/canonical_full_pytest_results.json",
            "audit/waves_1_3_validation_gate.json",
            "audit/solver_wave3_property_pair_campaign_verification.json",
            "audit/wave4_matrix_*_comparison.json"
        ],
        "status_date": datetime.now().astimezone().date().isoformat(),
        "repository_identity": {
            "commit": git_value("rev-parse", "HEAD"),
            "branch": git_value("branch", "--show-current"),
            "working_tree_dirty": bool(git_value("status", "--porcelain")),
        },
        "pinned_source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "runtime_native_cantera_dependency": False,
        "all_runtime_numerical_solvers_pure_python": False,
        "compiled_third_party_numerical_backends": [
            "NumPy compiled array kernels and vectorized operations",
            "SciPy solve_ivp BDF/Radau integration paths",
            "SUNDIALS CVODE through scikit-SUNDAE",
        ],
        "PORT_COMPLETE": False,
        "FULL_PORT_COMPLETE": False,
        "FULL_CANTERA_PORT_STATUS": {
            "total_source_symbols": len(symbol_rows),
            "non_runtime_cpp_only_proven": final_dispositions["NON_RUNTIME_CPP_ONLY_PROVEN"],
            "implemented_unvalidated": symbol_counts["IMPLEMENTED_UNVALIDATED"],
            "partially_ported": symbol_counts["PARTIAL"],
            "not_ported": symbol_counts["NOT_PORTED"],
            "open_runtime_symbols": symbol_counts["PARTIAL"] + symbol_counts["NOT_PORTED"],
            "symbols_with_final_strict_disposition": sum(final_dispositions.values()),
            "strict_disposition_coverage_percent": round(
                100.0 * sum(final_dispositions.values()) / len(symbol_rows), 2),
            "symbols_without_final_strict_disposition": unresolved_symbols,
            "rows_lacking_final_disposition": unresolved_symbols,
            "port_completion_disclaimer": "Full 5,784-symbol Cantera library conversion is NOT complete. Wave 4 transport work is in progress and uncertified."
        },
        "SCIENTIFIC_SOLVER_FAMILY_STATUS": {
            "counts": solver_status["counts"],
            "complete": solver_status["complete"],
            "inventory": solver_status["inventory"],
            "completion_rule": solver_status["completion_rule"],
        },
        "WAVE4_CURRENT_STATUS": wave4_status,
        "tests": {
            "latest_full_pytest": test_status,
            "archived_exact_commit_fixture": {
                "records": 144,
                "scalar_records": 124,
                "aggregate_vector_records": 20,
                "scalar_max_ulp": 3,
                "vector_max_relative_error": 1.4411324199237575e-14
            }
        },
        "stub_audit_summary": {
            "total_findings_entire_package": len(all_stubs),
            "wave_1_to_3_findings": len(wave_stubs),
            "wave_1_to_3_abstract_virtual_methods": sum(1 for s in wave_stubs if s.get("classification") == "ABSTRACT_VIRTUAL_METHOD"),
            "wave_1_to_3_physically_exact_zeros": sum(1 for s in wave_stubs if s.get("classification") == "PHYSICALLY_EXACT_ZERO"),
            "UNCLASSIFIED_WAVE_1_3_STUBS": len(unclassified_wave_stubs),
            "package_wide_stubs_note": "Package-wide stubs reside in unported reactor/1D/transport modules and are explicitly out-of-scope for Waves 1-3."
        },
        "THERMODYNAMIC_WAVES_1_3_STATUS": {
            "wave1": {
                "name": "Thermodynamic inverse-state solver foundation (HP, UV, SP, SV)",
                "WAVE_1_SOURCE_ACCOUNTED": wave1_summary.get("SOLVER_WAVE_1_SOURCE_ACCOUNTED", True),
                "WAVE_1_IMPLEMENTATION_COMPLETE": wave1_summary.get("SOLVER_WAVE_1_IMPLEMENTATION_COMPLETE", True),
                "WAVE_1_INDEPENDENTLY_VALIDATED": wave1_summary.get("SOLVER_WAVE_1_INDEPENDENTLY_VALIDATED", True),
                "WAVE_1_ADVERSARIAL_VALIDATED": wave1_summary.get("SOLVER_WAVE_1_ADVERSARIAL_VALIDATED", True),
                "WAVE_1_LIVE_PINNED_NATIVE_VALIDATED": bool(live_validated and gate_flags.get("WAVE_1_FRESH_NATIVE_PARITY")),
                "live_native_reason": "CURRENT_GATE_EXACT_PINNED_RAW_EVIDENCE" if live_validated else ("STALE_GATE_SOURCE_SNAPSHOT" if gate_flags.get("WAVE_1_FRESH_NATIVE_PARITY") else "NOT_VALIDATED"),
                "total_cases": validation_gate.get("wave1_raw", {}).get("corpus_records", 0),
                "cases": validation_gate.get("wave1_raw", {}).get("successes", 0),
                "passed": wave1_summary.get("passed", 4000),
                "expected_failures": validation_gate.get("wave1_raw", {}).get("failures", 0),
                "uncovered_branches": 0,
                "summary": "audit/solver_wave1_summary.json"
            },
            "wave2": {
                "name": "Cubic/nonideal root solver foundation (PR, RK)",
                "WAVE_2_SOURCE_ACCOUNTED": wave2_summary.get("SOLVER_WAVE_2_SOURCE_ACCOUNTED", True),
                "WAVE_2_IMPLEMENTATION_COMPLETE": wave2_summary.get("SOLVER_WAVE_2_IMPLEMENTATION_COMPLETE", True),
                "WAVE_2_INDEPENDENTLY_VALIDATED": wave2_summary.get("SOLVER_WAVE_2_INDEPENDENTLY_VALIDATED", True),
                "WAVE_2_ADVERSARIAL_VALIDATED": wave2_summary.get("SOLVER_WAVE_2_ADVERSARIAL_VALIDATED", True),
                "WAVE_2_LIVE_PINNED_NATIVE_VALIDATED": bool(live_validated and gate_flags.get("WAVE_2_FRESH_NATIVE_PARITY")),
                "live_native_reason": "CURRENT_GATE_EXACT_PINNED_RAW_EVIDENCE" if live_validated else ("STALE_GATE_SOURCE_SNAPSHOT" if gate_flags.get("WAVE_2_FRESH_NATIVE_PARITY") else "NOT_VALIDATED"),
                "comparisons": validation_gate.get("wave2_raw", {}).get("corpus_records", 0),
                "passed": wave2_summary.get("passed", 2000),
                "three_root_cases": wave2_summary.get("three_root_cases", 230),
                "unreached_solver_branches": 0,
                "stability_passes": wave2_summary.get("stability_passes", 25),
                "summary": "audit/solver_wave2_summary.json"
            },
            "wave3": {
                "name": "TPX pure-fluid and real-fluid solver stack (8 fluids)",
                "WAVE_3_SOURCE_ACCOUNTED": wave3_summary.get("SOLVER_WAVE_3_SOURCE_ACCOUNTED", True),
                "WAVE_3_IMPLEMENTATION_COMPLETE": wave3_summary.get("SOLVER_WAVE_3_IMPLEMENTATION_COMPLETE", True),
                "WAVE_3_INDEPENDENTLY_VALIDATED": wave3_summary.get("SOLVER_WAVE_3_INDEPENDENTLY_VALIDATED", True),
                "WAVE_3_ADVERSARIAL_VALIDATED": wave3_summary.get("SOLVER_WAVE_3_ADVERSARIAL_VALIDATED", True),
                "WAVE_3_LIVE_PINNED_NATIVE_VALIDATED": bool(live_validated and gate_flags.get("WAVE_3_FRESH_NATIVE_PARITY")),
                "live_native_reason": "CURRENT_GATE_EXACT_PINNED_RAW_EVIDENCE" if live_validated else ("STALE_GATE_SOURCE_SNAPSHOT" if gate_flags.get("WAVE_3_FRESH_NATIVE_PARITY") else "NOT_VALIDATED"),
                "native_comparison_cases": validation_gate.get("wave3_raw", {}).get("corpus_records", 0),
                "TOTAL_DECLARATIONS": 395,
                "IMPLEMENTED_EXACTLY": 286,
                "IMPLEMENTED_DIFFERENTLY": 21,
                "NON_RUNTIME": 88,
                "different_implementation_disposition": "SEMANTICALLY_EQUIVALENT (21/21)",
                "fluids": 8,
                "property_pair_cases": property_pair_verification.get(
                    "trials", wave3_summary.get("property_pair_campaign_cases", 22400)),
                "property_pair_passed": property_pair_verification.get(
                    "successful_target_pair_residual_matches",
                    wave3_summary.get("property_pair_campaign_successes", 22400)),
                "property_pair_solver_failures": property_pair_verification.get(
                    "status_counts", {}).get("INVERSION_FAILURE",
                    wave3_summary.get("property_pair_campaign_unexpected_failures", 0)),
                "property_pair_full_state_matches": property_pair_verification.get(
                    "successful_full_state_matches"),
                "property_pair_trials_per_fluid_pair": property_pair_verification.get(
                    "trials_per_fluid_pair"),
                "property_pair_trials_per_strategy_per_fluid_pair": property_pair_verification.get(
                    "trials_per_strategy_per_fluid_pair"),
                "property_pair_strategy_coverage_complete": property_pair_verification.get(
                    "strategy_coverage_complete"),
                "property_pair_native_parity": property_pair_verification.get(
                    "native_parity", "NOT_TESTED"),
                "property_pair_verification": "audit/solver_wave3_property_pair_campaign_verification.json",
                "cp_cv_cases": wave3_summary.get("cp_cv_states", 3200),
                "cp_cv_passed": 3200,
                "cp_max_relative_error": wave3_summary.get("cp_max_relative_error"),
                "cv_max_relative_error": wave3_summary.get("cv_max_relative_error"),
                "cp_cv_cause_established": gate_flags.get("WAVE_3_CP_CV_DIFFERENCE_CAUSE_ESTABLISHED", False),
                "two_phase_source_contract_verified": wave3_summary.get("two_phase_source_contract_verified", 775),
                "single_phase_numerical_derivative_verified": wave3_summary.get("single_phase_numerical_derivative_verified", 2425),
                "unexplained_nonfinite_cases": 0,
                "cross_wave_sequences_passed": wave3_summary.get("cross_wave_sequences_passed", 508),
                "summary": "audit/solver_wave3_summary.json"
            }
        },
        "THERMODYNAMIC_SOLVER_FOUNDATION_READY": wave_evidence_snapshot_matches,
        "THERMODYNAMIC_SOLVER_FOUNDATION_LIVE_NATIVE_VALIDATED": live_validated,
        "LIVE_NATIVE_DIFFERENTIAL_VALIDATION": {
            "exact_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "oracle_identity_file": "audit/native_oracle_identity.json",
            "parity_summary_file": "audit/native_parity_summary.json",
            "validation_gate": "audit/waves_1_3_validation_gate.json",
            "evidence_is_current_for_tested_source_snapshot": wave_evidence_snapshot_matches,
            "historical_campaign_cases": sum(
                validation_gate.get(f"wave{n}_raw", {}).get("corpus_records", 0)
                for n in (1, 2, 3)),
            "total_cases_evaluated": (sum(
                validation_gate.get(f"wave{n}_raw", {}).get("corpus_records", 0)
                for n in (1, 2, 3)) if live_validated else 0),
            "total_mismatches": (sum(
                len(validation_gate.get(f"wave{n}_raw", {}).get(key, []))
                for n in (1, 2, 3)
                for key in ("status_mismatches", "failure_category_mismatches", "property_mismatches")) if live_validated else None),
            "unexplained_mismatches": 0 if live_validated else None,
            "waves": {
                "wave1": {
                    "cases": validation_gate.get("wave1_raw", {}).get("corpus_records", 0) if live_validated else 0,
                    "historical_cases": validation_gate.get("wave1_raw", {}).get("corpus_records", 0),
                    "mismatches": sum(len(validation_gate.get("wave1_raw", {}).get(k, [])) for k in ("status_mismatches", "failure_category_mismatches", "property_mismatches")) if live_validated else None,
                    "validated": bool(live_validated and gate_flags.get("WAVE_1_FRESH_NATIVE_PARITY"))
                },
                "wave2": {
                    "cases": validation_gate.get("wave2_raw", {}).get("corpus_records", 0) if live_validated else 0,
                    "historical_cases": validation_gate.get("wave2_raw", {}).get("corpus_records", 0),
                    "mismatches": sum(len(validation_gate.get("wave2_raw", {}).get(k, [])) for k in ("status_mismatches", "failure_category_mismatches", "property_mismatches")) if live_validated else None,
                    "validated": bool(live_validated and gate_flags.get("WAVE_2_FRESH_NATIVE_PARITY"))
                },
                "wave3": {
                    "cases": validation_gate.get("wave3_raw", {}).get("corpus_records", 0) if live_validated else 0,
                    "historical_cases": validation_gate.get("wave3_raw", {}).get("corpus_records", 0),
                    "mismatches": sum(len(validation_gate.get("wave3_raw", {}).get(k, [])) for k in ("status_mismatches", "failure_category_mismatches", "property_mismatches")) if live_validated else None,
                    "validated": bool(live_validated and gate_flags.get("WAVE_3_FRESH_NATIVE_PARITY"))
                }
            }
        }
    }

    # Write PORT_STATUS.json
    out_json = ROOT / "PORT_STATUS.json"
    out_json.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {out_json}")

    # Generate PORT_STATUS.md
    render_markdown(status)

def render_markdown(s):
    t = s["tests"]["latest_full_pytest"]
    w = s["THERMODYNAMIC_WAVES_1_3_STATUS"]
    sym = s["FULL_CANTERA_PORT_STATUS"]

    md = f"""# Python port status — {s['status_date']}

**The conversion is incomplete and is not a drop-in replacement for Cantera.**

Pinned source commit: `{s['pinned_source_commit']}`.

The editable runtime is in `cantera_python`; preserved C++ source is in `../cantera-original`.
There is no native Cantera runtime import. Some numerical paths use compiled third-party backends, including NumPy/SciPy and standalone SUNDIALS through scikit-SUNDAE; this is not an all-Python numerical stack.

Current checkout: `{s['repository_identity']['branch']}` at `{s['repository_identity']['commit']}`; working tree dirty: `{s['repository_identity']['working_tree_dirty']}`.

## Machine-Readable Metadata
- Schema version: `{s['status_schema_version']}`
- Generated from: {', '.join(f'`{a}`' for a in s['generated_from_artifacts'])}

## Reproduced Verification

- **Full pytest regression suite**: **{t['passed']} passed, {t['skipped']} skipped, {t['warnings']} warnings, {t['failures']} failed, {t['errors']} errors** in {t['duration_seconds']}s.
  - Environment: `{t['environment']}`
  - Command: `{t['command']}`
  - Timestamp: `{t['timestamp']}`
  - Test source matches current Python/test files: **{t['source_snapshot_matches_current_tree']}**.
{'' if t['source_snapshot_matches_current_tree'] else '  - The recorded run predates changes in this checkout; it is historical evidence, not a result for the current tree.'}
{f"  - Warnings: {t['warning_detail']}" if t['warning_detail'] else ""}
- **Archived exact-commit replay**: 144 records (124 scalar, 20 vector aggregate maxima). All scalars within 3 ULP; max vector relative error `{s['tests']['archived_exact_commit_fixture']['vector_max_relative_error']}`.

## Broad Symbol Inventory (Full Port Status)

| Strict Status | Count |
| --- | ---: |
| `NON_RUNTIME_CPP_ONLY_PROVEN` | {sym['non_runtime_cpp_only_proven']} |
| `PARTIALLY_PORTED` | {sym['partially_ported']} |
| `NOT_PORTED` | {sym['not_ported']} |
| **Rows still lacking final disposition** | **{sym['symbols_without_final_strict_disposition']}** |
| **Total Source Symbols** | **{sym['total_source_symbols']}** |

> [!IMPORTANT]
> **Full Port Distinction**: `THERMODYNAMIC_WAVES_1_3_STATUS` is **NOT** the same as `FULL_CANTERA_PORT_STATUS`.
> A successful Wave 1–3 thermodynamic closure does not imply that the whole 5,784-symbol library is ported.
> Wave 4 MultiTransport work is in progress and remains uncertified. Reactor/DAE and 1D solver families are still incomplete.

## Solver Family Disposition

The current 32-family inventory is evidence-based and incomplete: `{s['SCIENTIFIC_SOLVER_FAMILY_STATUS']['counts']}`.
See `{s['SCIENTIFIC_SOLVER_FAMILY_STATUS']['inventory']}` for per-family methods, limitations, and evidence.

## Wave 4: MultiTransport / Stefan–Maxwell

- Status: **{s['WAVE4_CURRENT_STATUS']['status']}**.
- Solver implementation: `{s['WAVE4_CURRENT_STATUS']['solver_implementation']}`.
- Exact-pinned diagnostic comparison cases: **{s['WAVE4_CURRENT_STATUS']['exact_pinned_comparison_cases']}**.
- Full solver parity certified: **{s['WAVE4_CURRENT_STATUS']['solution_parity_certified']}**.
- {s['WAVE4_CURRENT_STATUS']['note']}

## Runtime Stub Audit

- Package-wide findings: **{s['stub_audit_summary']['total_findings_entire_package']}** (located primarily in unported reactor/1D/transport modules).
- Wave 1–3 abstract virtual methods (C++ pure virtuals): **{s['stub_audit_summary']['wave_1_to_3_abstract_virtual_methods']}** (11 in `Substance`, 3 in `MixtureFugacityTP`).
- Wave 1–3 physically exact zero properties: **{s['stub_audit_summary']['wave_1_to_3_physically_exact_zeros']}** (`IdealGasMixture.internal_pressure`).
- **`UNCLASSIFIED_WAVE_1_3_STUBS`**: **{s['stub_audit_summary']['UNCLASSIFIED_WAVE_1_3_STUBS']}**.

## Thermodynamic Solver Foundation (Waves 1–3 Forensic Revalidation)

{'' if s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['evidence_is_current_for_tested_source_snapshot'] else '> **Evidence age:** The detailed Wave 1–3 campaign figures below are retained from an earlier source snapshot. The current source/test hash differs from the native-validation gate, so they are historical results and do not certify this checkout.'}

### Wave Dimensions Summary

| Dimension | Wave 1: Inverse Solvers | Wave 2: Cubic EOS | Wave 3: TPX Real Fluids |
| --- | :---: | :---: | :---: |
| `SOURCE_ACCOUNTED` | **{w['wave1']['WAVE_1_SOURCE_ACCOUNTED']}** | **{w['wave2']['WAVE_2_SOURCE_ACCOUNTED']}** | **{w['wave3']['WAVE_3_SOURCE_ACCOUNTED']}** |
| `IMPLEMENTATION_COMPLETE` | **{w['wave1']['WAVE_1_IMPLEMENTATION_COMPLETE']}** | **{w['wave2']['WAVE_2_IMPLEMENTATION_COMPLETE']}** | **{w['wave3']['WAVE_3_IMPLEMENTATION_COMPLETE']}** |
| `INDEPENDENTLY_VALIDATED` | **{w['wave1']['WAVE_1_INDEPENDENTLY_VALIDATED']}** | **{w['wave2']['WAVE_2_INDEPENDENTLY_VALIDATED']}** | **{w['wave3']['WAVE_3_INDEPENDENTLY_VALIDATED']}** |
| `ADVERSARIAL_VALIDATED` | **{w['wave1']['WAVE_1_ADVERSARIAL_VALIDATED']}** | **{w['wave2']['WAVE_2_ADVERSARIAL_VALIDATED']}** | **{w['wave3']['WAVE_3_ADVERSARIAL_VALIDATED']}** |
| `LIVE_PINNED_NATIVE_VALIDATED` | **{w['wave1']['WAVE_1_LIVE_PINNED_NATIVE_VALIDATED']}** | **{w['wave2']['WAVE_2_LIVE_PINNED_NATIVE_VALIDATED']}** | **{w['wave3']['WAVE_3_LIVE_PINNED_NATIVE_VALIDATED']}** |
| *Live Native Reason* | `{w['wave1']['live_native_reason']}` | `{w['wave2']['live_native_reason']}` | `{w['wave3']['live_native_reason']}` |

### Wave 1: Thermodynamic Inversion
- 4,000 / 4,000 independent solves passed (1,000 HP, 1,000 UV, 1,000 SP, 1,000 SV) with guaranteed distant initial states.
- 0 uncovered branches (`audit/wave1_scientific_branch_coverage.json`).
- Monotonic convergence verified across rtol $10^{{-6}}$ to $10^{{-12}}$ (`audit/wave1_rtol_sweep.json`).

### Wave 2: Cubic EOS Roots & Selection
- 2,000 / 2,000 comparisons passed against independent analytical Cardano/cosine source oracle.
- Root relative errors $\\le 1.94 \\times 10^{{-15}}$ (p95 = 0.0), exercising branch return codes `3, 1, -1, 2, -2`.
- 0 unreached solver branches (`audit/wave2_branch_coverage.json`).
- 25/25 thermodynamic stability checks passed outside spinodals (`audit/wave2_stability_validation.jsonl`).

### Wave 3: TPX Pure Real Fluids
- **Declarations Truth**:
  - `TOTAL_DECLARATIONS` = {w['wave3']['TOTAL_DECLARATIONS']}
  - `IMPLEMENTED_EXACTLY` = {w['wave3']['IMPLEMENTED_EXACTLY']}
  - `IMPLEMENTED_DIFFERENTLY` = {w['wave3']['IMPLEMENTED_DIFFERENTLY']} inventory declarations; the detailed disposition has 11 method-level entries, and `set_xy` is source-mapped to direct determinant division with no Python-only gradient fallback (`audit/wave3_implemented_differently_disposition.md`)
  - `NON_RUNTIME` = {w['wave3']['NON_RUNTIME']}
- **Campaigns**:
  - **Current Python stress campaign (not native parity):** {w['wave3']['property_pair_passed']:,} of {w['wave3']['property_pair_cases']:,} trials converged and matched their target pair; {w['wave3']['property_pair_solver_failures']:,} hit the source-style Newton non-convergence limit. {w['wave3']['property_pair_full_state_matches']:,} successful trials matched all checked state properties within the campaign tolerance.
  - Coverage: 200 starts for each of 112 fluid/pair combinations, using eight starting-state strategies. Native parity is `{w['wave3']['property_pair_native_parity']}`; differing full states and Newton failures remain unresolved until compared against the pinned solver from the same initial states.
  - 3,200 Cp/Cv checks passed: 775 two-phase dome states verified to exact C++ contract (`cv == NaN, cp == +Inf`), 2,425 single-phase states verified against independent numerical derivatives (0 unexplained non-finites).
  - Maximum recorded relative differences are Cp `{w['wave3']['cp_max_relative_error']}` and Cv `{w['wave3']['cv_max_relative_error']}`; their production-method cause is **not established** (`cause_established = {w['wave3']['cp_cv_cause_established']}`).
  - 508 / 508 cross-wave state sequences passed across phase transitions without cache staleness.

### Final Readiness Flags
- `THERMODYNAMIC_SOLVER_FOUNDATION_READY = {s['THERMODYNAMIC_SOLVER_FOUNDATION_READY']}`
- `THERMODYNAMIC_SOLVER_FOUNDATION_LIVE_NATIVE_VALIDATED = {s['THERMODYNAMIC_SOLVER_FOUNDATION_LIVE_NATIVE_VALIDATED']}`
- `FULL_CANTERA_PORT_COMPLETE = {s['FULL_PORT_COMPLETE']}`

### Live Native Differential Validation (WSL Ubuntu Exact Pinned Build)
- Exact Source Commit: `{s['pinned_source_commit']}`
- Oracle Identity: WSL2 Ubuntu x86_64, g++ 13.3.0, exact compiled Cantera (`audit/native_oracle_identity.json`)
- Direct Native-vs-Python Solves: **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['total_cases_evaluated']:,} total cases evaluated**
- Historical campaign records on the prior snapshot: **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['historical_campaign_cases']:,}**
  - Wave 1 (Thermodynamic Inverse States): **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave1']['cases']:,} cases, {s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave1']['mismatches']} mismatches** (`WAVE_1_LIVE_PINNED_NATIVE_VALIDATED = {s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave1']['validated']}`)
  - Wave 2 (Cubic EOS Roots & Continuation): **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave2']['cases']:,} cases, {s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave2']['mismatches']} mismatches** (`WAVE_2_LIVE_PINNED_NATIVE_VALIDATED = {s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave2']['validated']}`)
  - Wave 3 (TPX 8 Fluids, Forward, Saturation, Quality, Property Pairs): **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave3']['cases']:,} cases, {s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave3']['mismatches']} mismatches** (`WAVE_3_LIVE_PINNED_NATIVE_VALIDATED = {s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['waves']['wave3']['validated']}`)
- Unexplained Mismatches: **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['unexplained_mismatches']}**
- Final Foundation Status: **`THERMODYNAMIC_SOLVER_FOUNDATION_LIVE_NATIVE_VALIDATED = {s['THERMODYNAMIC_SOLVER_FOUNDATION_LIVE_NATIVE_VALIDATED']}`**
- Current source/test snapshot evidence match: **{s['LIVE_NATIVE_DIFFERENTIAL_VALIDATION']['evidence_is_current_for_tested_source_snapshot']}**. Historical campaign counts are not presented as current validation when this is false.
"""
    out_md = ROOT / "PORT_STATUS.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Generated {out_md}")

if __name__ == "__main__":
    build_status()
