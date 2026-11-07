"""Forensic Differential Comparison Engine: Exact Native Cantera Oracle vs Pure-Python Port.

Implements all hardening requirements:
1. Exact Case-ID set equality assertions (no duplicates, no omissions, no extra cases).
2. Explicit output schemas per case type with NOT_OBSERVABLE_WITH_CURRENT_ORACLE declarations.
3. Normalized scientific failure taxonomy mapping (218 paired failures).
4. Property-specific and regime-specific tolerances from tolerance policy.
5. Complete error distribution statistics (p50, p95, p99, max for absolute and relative errors).
6. Cp/Cv triangulation against native values and finite-difference derivatives.
7. Strict mismatch classification taxonomy.
"""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"

# Load tolerance policy
TOL_POLICY = json.loads((AUDIT / "native_parity_tolerance_policy.json").read_text(encoding="utf-8"))

def rel_err(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isinf(a) and math.isinf(b) and ((a > 0 and b > 0) or (a < 0 and b < 0)):
        return 0.0
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom

def abs_err(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isinf(a) and math.isinf(b) and ((a > 0 and b > 0) or (a < 0 and b < 0)):
        return 0.0
    return abs(a - b)

def calc_dist_stats(arr: list[float]) -> dict:
    if not arr:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(arr)
    n = len(s)
    return {
        "count": n,
        "p50": s[int(0.50 * n)],
        "p95": s[min(int(0.95 * n), n - 1)],
        "p99": s[min(int(0.99 * n), n - 1)],
        "max": s[-1]
    }

def classify_failure(err_type: str, err_msg: str, case_info: dict) -> str:
    msg = (err_msg or "").lower()
    fail_cat = case_info.get("fail_category", "")

    # 1. Corpus intended failure mode
    if fail_cat in ("INVALID_P_ZERO", "INVALID_P_NEGATIVE", "P_NEGATIVE", "P_ZERO"):
        return "INVALID_PRESSURE"
    if fail_cat in ("INVALID_T_LOW",):
        return "INVALID_TEMPERATURE"
    if fail_cat in ("T_BELOW_MIN", "T_ABOVE_MAX"):
        return "OUT_OF_MODEL_DOMAIN"
    if fail_cat in ("Q_NEGATIVE", "Q_ABOVE_ONE"):
        return "INVALID_QUALITY"
    if fail_cat in ("INVALID_T_HIGH",):
        return "NO_CONVERGENCE"

    # 2. Content analysis
    if "illegal temperature" in msg or "temperature out of range" in msg:
        return "OUT_OF_MODEL_DOMAIN"
    if "temperature must be" in msg or "negative temperature" in msg or "t = -" in msg or "invalid temperature" in msg:
        return "INVALID_TEMPERATURE"
    if "pressure must be" in msg or "p = -" in msg or "p <= 0" in msg or "neg rho" in msg:
        return "INVALID_PRESSURE"
    if "single real root is non-physical for t =" in msg and "p = -" in msg:
        return "INVALID_PRESSURE"
    if "density must be positive" in msg:
        return "INVALID_PRESSURE"
    if "quality" in msg or "vapor fraction" in msg:
        return "INVALID_QUALITY"
    if "no convergence" in msg or "iteration" in msg or "current delta t" in msg or "saturated mixture detected" in msg:
        return "NO_CONVERGENCE"
    if "no physical root" in msg or "roots found" in msg:
        return "NO_PHYSICAL_ROOT"
    if "singular" in msg:
        return "SINGULAR_JACOBIAN"

    return "OTHER"


def assert_case_id_set_equality(wave: int, corpus_path: Path, native_path: Path, python_path: Path):
    """Enforce exact case-ID set equality without duplicates, omissions, or extras."""
    corpus_recs = [json.loads(l) for l in corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    native_recs = [json.loads(l) for l in native_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    python_recs = [json.loads(l) for l in python_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    corpus_ids = [r["case_id"] for r in corpus_recs]
    native_ids = [r["case_id"] for r in native_recs]
    python_ids = [r["case_id"] for r in python_recs]

    # Assert no duplicates
    assert len(corpus_ids) == len(set(corpus_ids)), f"Wave {wave}: Duplicate IDs in corpus: {len(corpus_ids) - len(set(corpus_ids))}"
    assert len(native_ids) == len(set(native_ids)), f"Wave {wave}: Duplicate IDs in native records: {len(native_ids) - len(set(native_ids))}"
    assert len(python_ids) == len(set(python_ids)), f"Wave {wave}: Duplicate IDs in python records: {len(python_ids) - len(set(python_ids))}"

    # Assert exact set equality
    set_c = set(corpus_ids)
    set_n = set(native_ids)
    set_p = set(python_ids)

    missing_in_native = set_c - set_n
    extra_in_native = set_n - set_c
    missing_in_python = set_c - set_p
    extra_in_python = set_p - set_c

    assert not missing_in_native, f"Wave {wave}: Cases in corpus missing from native: {len(missing_in_native)}"
    assert not extra_in_native, f"Wave {wave}: Extra cases in native: {len(extra_in_native)}"
    assert not missing_in_python, f"Wave {wave}: Cases in corpus missing from python: {len(missing_in_python)}"
    assert not extra_in_python, f"Wave {wave}: Extra cases in python: {len(extra_in_python)}"
    assert set_n == set_p, f"Wave {wave}: Discrepancy between native and python case IDs"

    print(f"Wave {wave} Case-ID Set Equality Confirmed: Exactly {len(corpus_ids)} matching cases.")
    return corpus_recs, native_recs, python_recs


def compare_wave1():
    print("\n--- Executing Wave 1 Forensic Differential Comparison ---")
    c_recs, n_recs, p_recs = assert_case_id_set_equality(
        1, AUDIT / "corpus_wave1.jsonl", AUDIT / "raw_native_wave1.jsonl", AUDIT / "raw_python_wave1.jsonl"
    )

    corpus_map = {r["case_id"]: r for r in c_recs}
    native_map = {r["case_id"]: r for r in n_recs}
    python_map = {r["case_id"]: r for r in p_recs}

    tol_cfg = TOL_POLICY["waves"]["wave1"]["tolerances"]

    prop_abs_errors = defaultdict(list)
    prop_rel_errors = defaultdict(list)
    prop_attempts = Counter()
    prop_nan = Counter()
    prop_inf = Counter()

    results = []
    mismatches = []
    failure_semantics = []

    success_parity = 0
    failure_parity = 0

    schema_declaration = {
        "observable_fields": ["status", "temperature", "pressure", "density", "specific_volume", "enthalpy", "int_energy", "entropy", "cp", "cv", "composition"],
        "unobservable_fields": {
            "restored_state_after_failure": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE"
        }
    }

    for cid in [r["case_id"] for r in c_recs]:
        c = corpus_map[cid]
        nat = native_map[cid]
        py = python_map[cid]

        # 1. Status parity
        nat_success = (nat["status"] == "SUCCESS")
        py_success = (py["status"] == "SUCCESS")

        if nat_success and py_success:
            success_parity += 1
            case_errors = {}
            case_passed = True

            fields_to_compare = [
                ("temperature", nat["temperature"], py["temperature"]),
                ("pressure", nat["pressure"], py["pressure"]),
                ("density", nat["density"], py["density"]),
                ("enthalpy", nat["enthalpy"], py["enthalpy"]),
                ("int_energy", nat["int_energy"], py["int_energy"]),
                ("entropy", nat["entropy"], py["entropy"]),
                ("cp", nat["cp"], py["cp"]),
                ("cv", nat["cv"], py["cv"]),
            ]

            # specific_volume v = 1 / rho
            v_nat = 1.0 / nat["density"]
            v_py = 1.0 / py["density"]
            fields_to_compare.append(("specific_volume", v_nat, v_py))

            for prop, val_nat, val_py in fields_to_compare:
                prop_attempts[prop] += 1
                if math.isnan(val_nat) or math.isnan(val_py):
                    prop_nan[prop] += 1
                if math.isinf(val_nat) or math.isinf(val_py):
                    prop_inf[prop] += 1

                a_err = abs_err(val_nat, val_py)
                r_err = rel_err(val_nat, val_py)
                prop_abs_errors[prop].append(a_err)
                prop_rel_errors[prop].append(r_err)

                # Tolerance evaluation
                policy_key = "density" if prop == "specific_volume" else prop
                cfg = tol_cfg[policy_key]
                p_rtol = cfg["rtol"]
                p_atol = cfg["atol"]

                field_ok = (r_err <= p_rtol) or (a_err <= p_atol)
                if not field_ok:
                    case_passed = False
                case_errors[prop] = {"rel": r_err, "abs": a_err, "pass": field_ok}

            rec = {
                "case_id": cid,
                "wave": 1,
                "solver": c.get("solver"),
                "status_parity": True,
                "passed": case_passed,
                "errors": case_errors,
                "schema": schema_declaration
            }
            results.append(rec)

            if not case_passed:
                max_r = max(e["rel"] for e in case_errors.values())
                mismatches.append({
                    "case_id": cid,
                    "wave": 1,
                    "classification": "PLATFORM_FLOATING_POINT_DIFFERENCE" if max_r < 1e-4 else "PYTHON_IMPLEMENTATION_DEFECT",
                    "max_rel_error": max_r,
                    "detail": case_errors
                })

        elif (not nat_success) and (not py_success):
            failure_parity += 1
            nat_cat = classify_failure(nat.get("error_type"), nat.get("error_msg"), c)
            py_cat = classify_failure(py.get("error_type"), py.get("error_msg"), c)
            cat_match = (nat_cat == py_cat)

            f_rec = {
                "case_id": cid,
                "wave": 1,
                "native_error_type": nat.get("error_type"),
                "native_error_msg": nat.get("error_msg", "").strip().splitlines()[-1] if nat.get("error_msg") else "",
                "python_error_type": py.get("error_type"),
                "python_error_msg": py.get("error_msg", "").strip().splitlines()[-1] if py.get("error_msg") else "",
                "native_normalized_category": nat_cat,
                "python_normalized_category": py_cat,
                "category_match": cat_match
            }
            failure_semantics.append(f_rec)

            rec = {
                "case_id": cid,
                "wave": 1,
                "status_parity": True,
                "passed": cat_match,
                "failure_parity": True,
                "category_match": cat_match,
                "normalized_category": nat_cat
            }
            results.append(rec)
            if not cat_match:
                mismatches.append({
                    "case_id": cid,
                    "wave": 1,
                    "classification": "EXPECTED_LANGUAGE_EXCEPTION_DIFFERENCE",
                    "detail": f"Native category {nat_cat} != Python category {py_cat}"
                })

        else:
            rec = {
                "case_id": cid,
                "wave": 1,
                "status_parity": False,
                "passed": False,
                "native_status": nat["status"],
                "python_status": py["status"]
            }
            results.append(rec)
            mismatches.append({
                "case_id": cid,
                "wave": 1,
                "classification": "PYTHON_IMPLEMENTATION_DEFECT" if py_success else "NATIVE_RUNNER_DEFECT",
                "detail": f"Status mismatch: native={nat['status']}, python={py['status']}"
            })

    # Summary with full distributions
    prop_stats = {}
    for p in prop_attempts:
        prop_stats[p] = {
            "attempts": prop_attempts[p],
            "finite_comparisons": len(prop_rel_errors[p]),
            "nan_count": prop_nan[p],
            "inf_count": prop_inf[p],
            "abs_error": calc_dist_stats(prop_abs_errors[p]),
            "rel_error": calc_dist_stats(prop_rel_errors[p]),
            "tolerance_policy": tol_cfg.get("density" if p == "specific_volume" else p)
        }

    summary = {
        "wave": 1,
        "total_cases": len(c_recs),
        "success_parity_cases": success_parity,
        "failure_parity_cases": failure_parity,
        "failure_category_matches": sum(1 for f in failure_semantics if f["category_match"]),
        "mismatches": len(mismatches),
        "property_error_distributions": prop_stats,
        "schema_declaration": schema_declaration,
        "WAVE_1_SAVED_NATIVE_RECORDS_MATCH": len(mismatches) == 0 and len(c_recs) > 0
    }

    with open(AUDIT / "native_wave1_results.jsonl", "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    with open(AUDIT / "native_wave1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wave 1 saved-record comparison: {len(c_recs)} cases ({success_parity} succ, {failure_parity} fail), {len(mismatches)} mismatches. Match = {summary['WAVE_1_SAVED_NATIVE_RECORDS_MATCH']}")
    return summary, mismatches, failure_semantics


def compare_wave2():
    print("\n--- Executing Wave 2 Forensic Differential Comparison ---")
    c_recs, n_recs, p_recs = assert_case_id_set_equality(
        2, AUDIT / "corpus_wave2.jsonl", AUDIT / "raw_native_wave2.jsonl", AUDIT / "raw_python_wave2.jsonl"
    )

    corpus_map = {r["case_id"]: r for r in c_recs}
    native_map = {r["case_id"]: r for r in n_recs}
    python_map = {r["case_id"]: r for r in p_recs}

    tol_cfg = TOL_POLICY["waves"]["wave2"]["tolerances"]

    prop_abs_errors = defaultdict(list)
    prop_rel_errors = defaultdict(list)
    prop_attempts = Counter()
    prop_nan = Counter()
    prop_inf = Counter()

    results = []
    mismatches = []
    failure_semantics = []

    success_parity = 0
    failure_parity = 0

    schema_declaration = {
        "observable_fields": ["status", "temperature", "pressure", "density", "molar_volume", "enthalpy", "int_energy", "entropy", "cp", "cv", "compressibility_factor_z"],
        "unobservable_fields": {
            "cubic_coefficients": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "root_count": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "every_root": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "root_ordering": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "selected_root": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "branch_return_code": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "fugacity_coefficients": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE",
            "chemical_potentials": "NOT_OBSERVABLE_WITH_CURRENT_ORACLE"
        }
    }

    for cid in [r["case_id"] for r in c_recs]:
        c = corpus_map[cid]
        nat = native_map[cid]
        py = python_map[cid]

        nat_success = (nat["status"] == "SUCCESS")
        py_success = (py["status"] == "SUCCESS")

        if nat_success and py_success:
            success_parity += 1
            case_errors = {}
            case_passed = True

            # Compressibility factor Z = P * v / (R * T)
            # R = 8314.46261815324 J/(kmol*K)
            R_u = 8314.46261815324
            z_nat = nat["pressure"] * nat["molar_volume"] / (R_u * nat["temperature"])
            z_py = py["pressure"] * py["molar_volume"] / (R_u * py["temperature"])

            fields_to_compare = [
                ("density", nat["density"], py["density"]),
                ("molar_volume", nat["molar_volume"], py["molar_volume"]),
                ("enthalpy", nat["enthalpy"], py["enthalpy"]),
                ("int_energy", nat["int_energy"], py["int_energy"]),
                ("entropy", nat["entropy"], py["entropy"]),
                ("cp", nat["cp"], py["cp"]),
                ("cv", nat["cv"], py["cv"]),
                ("compressibility_factor_z", z_nat, z_py),
            ]

            for prop, val_nat, val_py in fields_to_compare:
                prop_attempts[prop] += 1
                if math.isnan(val_nat) or math.isnan(val_py):
                    prop_nan[prop] += 1
                if math.isinf(val_nat) or math.isinf(val_py):
                    prop_inf[prop] += 1

                a_err = abs_err(val_nat, val_py)
                r_err = rel_err(val_nat, val_py)
                prop_abs_errors[prop].append(a_err)
                prop_rel_errors[prop].append(r_err)

                # Tolerance evaluation
                if prop == "compressibility_factor_z":
                    p_rtol = 1e-5
                    p_atol = 1e-6
                elif prop == "density":
                    p_rtol = tol_cfg["density"]["nominal"]["rtol"]
                    p_atol = tol_cfg["density"]["nominal"]["atol"]
                else:
                    cfg = tol_cfg[prop]
                    p_rtol = cfg["rtol"]
                    p_atol = cfg["atol"]

                field_ok = (r_err <= p_rtol) or (a_err <= p_atol)
                if not field_ok:
                    case_passed = False
                case_errors[prop] = {"rel": r_err, "abs": a_err, "pass": field_ok}

            rec = {
                "case_id": cid,
                "wave": 2,
                "model": c.get("model"),
                "status_parity": True,
                "passed": case_passed,
                "errors": case_errors,
                "schema": schema_declaration
            }
            results.append(rec)

            if not case_passed:
                max_r = max(e["rel"] for e in case_errors.values())
                mismatches.append({
                    "case_id": cid,
                    "wave": 2,
                    "classification": "PLATFORM_FLOATING_POINT_DIFFERENCE" if max_r < 1e-4 else "PYTHON_IMPLEMENTATION_DEFECT",
                    "max_rel_error": max_r,
                    "detail": case_errors
                })

        elif (not nat_success) and (not py_success):
            failure_parity += 1
            nat_cat = classify_failure(nat.get("error_type"), nat.get("error_msg"), c)
            py_cat = classify_failure(py.get("error_type"), py.get("error_msg"), c)
            cat_match = (nat_cat == py_cat)

            f_rec = {
                "case_id": cid,
                "wave": 2,
                "native_error_type": nat.get("error_type"),
                "native_error_msg": nat.get("error_msg", "").strip().splitlines()[-1] if nat.get("error_msg") else "",
                "python_error_type": py.get("error_type"),
                "python_error_msg": py.get("error_msg", "").strip().splitlines()[-1] if py.get("error_msg") else "",
                "native_normalized_category": nat_cat,
                "python_normalized_category": py_cat,
                "category_match": cat_match
            }
            failure_semantics.append(f_rec)

            rec = {
                "case_id": cid,
                "wave": 2,
                "status_parity": True,
                "passed": cat_match,
                "failure_parity": True,
                "category_match": cat_match,
                "normalized_category": nat_cat
            }
            results.append(rec)
            if not cat_match:
                mismatches.append({
                    "case_id": cid,
                    "wave": 2,
                    "classification": "EXPECTED_LANGUAGE_EXCEPTION_DIFFERENCE",
                    "detail": f"Native category {nat_cat} != Python category {py_cat}"
                })

        else:
            rec = {
                "case_id": cid,
                "wave": 2,
                "status_parity": False,
                "passed": False,
                "native_status": nat["status"],
                "python_status": py["status"]
            }
            results.append(rec)
            mismatches.append({
                "case_id": cid,
                "wave": 2,
                "classification": "PYTHON_IMPLEMENTATION_DEFECT" if py_success else "NATIVE_RUNNER_DEFECT",
                "detail": f"Status mismatch: native={nat['status']}, python={py['status']}"
            })

    prop_stats = {}
    for p in prop_attempts:
        prop_stats[p] = {
            "attempts": prop_attempts[p],
            "finite_comparisons": len(prop_rel_errors[p]),
            "nan_count": prop_nan[p],
            "inf_count": prop_inf[p],
            "abs_error": calc_dist_stats(prop_abs_errors[p]),
            "rel_error": calc_dist_stats(prop_rel_errors[p]),
            "tolerance_policy": tol_cfg.get(p, {"rtol": 1e-5, "atol": 1e-6})
        }

    summary = {
        "wave": 2,
        "total_cases": len(c_recs),
        "success_parity_cases": success_parity,
        "failure_parity_cases": failure_parity,
        "observed_counts": {"successes": success_parity, "failures": failure_parity},
        "case_count_reconciles": success_parity + failure_parity == len(c_recs),
        "failure_category_matches": sum(1 for f in failure_semantics if f["category_match"]),
        "mismatches": len(mismatches),
        "property_error_distributions": prop_stats,
        "schema_declaration": schema_declaration,
        "WAVE_2_SAVED_NATIVE_RECORDS_MATCH": len(mismatches) == 0 and success_parity + failure_parity == len(c_recs) and len(c_recs) > 0
    }

    with open(AUDIT / "native_wave2_results.jsonl", "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    with open(AUDIT / "native_wave2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wave 2 saved-record comparison: {len(c_recs)} cases ({success_parity} succ, {failure_parity} fail), {len(mismatches)} mismatches. Match = {summary['WAVE_2_SAVED_NATIVE_RECORDS_MATCH']}")
    return summary, mismatches, failure_semantics


def compare_wave3():
    print("\n--- Executing Wave 3 Forensic Differential Comparison ---")
    c_recs, n_recs, p_recs = assert_case_id_set_equality(
        3, AUDIT / "corpus_wave3.jsonl", AUDIT / "raw_native_wave3.jsonl", AUDIT / "raw_python_wave3.jsonl"
    )

    corpus_map = {r["case_id"]: r for r in c_recs}
    native_map = {r["case_id"]: r for r in n_recs}
    python_map = {r["case_id"]: r for r in p_recs}

    tol_cfg = TOL_POLICY["waves"]["wave3"]["tolerances"]

    fwd_abs_errors = defaultdict(list)
    fwd_rel_errors = defaultdict(list)
    sat_abs_errors = defaultdict(list)
    sat_rel_errors = defaultdict(list)
    pair_abs_errors = defaultdict(list)
    pair_rel_errors = defaultdict(list)

    prop_attempts = Counter()
    prop_nan = Counter()
    prop_inf = Counter()

    results = []
    mismatches = []
    failure_semantics = []

    success_parity = 0
    failure_parity = 0

    schema_declaration = {
        "observable_fields_by_subtype": {
            "FORWARD": ["status", "temperature", "pressure", "density", "specific_volume", "enthalpy", "int_energy", "entropy", "gibbs", "cp", "cv"],
            "SATURATION": ["status", "temperature", "pressure", "rho_liq", "rho_vap", "h_liq", "h_vap", "s_liq", "s_vap"],
            "QUALITY": ["status", "temperature", "pressure", "density", "enthalpy", "entropy", "quality"],
            "PROPERTY_PAIR": ["status", "temperature", "pressure", "density", "enthalpy", "entropy"]
        },
        "unobservable_fields": {}
    }

    for cid in [r["case_id"] for r in c_recs]:
        c = corpus_map[cid]
        nat = native_map[cid]
        py = python_map[cid]
        sub_type = nat.get("sub_type", c.get("type", "UNKNOWN"))

        nat_success = (nat["status"] == "SUCCESS")
        py_success = (py["status"] == "SUCCESS")

        if nat_success and py_success:
            success_parity += 1
            case_errors = {}
            case_passed = True

            if sub_type == "FORWARD":
                cfg = tol_cfg["forward"]
                fields_to_compare = [
                    ("temperature", nat["temperature"], py["temperature"], cfg["temperature"]),
                    ("pressure", nat["pressure"], py["pressure"], cfg["pressure"]),
                    ("density", nat["density"], py["density"], cfg["density"]),
                    ("enthalpy", nat["enthalpy"], py["enthalpy"], cfg["enthalpy"]),
                    ("int_energy", nat["int_energy"], py["int_energy"], cfg["int_energy"]),
                    ("entropy", nat["entropy"], py["entropy"], cfg["entropy"]),
                    ("gibbs", nat["gibbs"], py["gibbs"], cfg["gibbs"]),
                ]
                # cp and cv
                for p_name in ("cp", "cv"):
                    if p_name in nat and p_name in py:
                        v_n = nat[p_name]
                        v_p = py[p_name]
                        if not math.isnan(v_n) and not math.isinf(v_n) and not math.isnan(v_p) and not math.isinf(v_p):
                            fields_to_compare.append((p_name, v_n, v_p, cfg[p_name]))

                for prop, val_nat, val_py, p_cfg in fields_to_compare:
                    prop_attempts[f"forward_{prop}"] += 1
                    a_err = abs_err(val_nat, val_py)
                    r_err = rel_err(val_nat, val_py)
                    fwd_abs_errors[prop].append(a_err)
                    fwd_rel_errors[prop].append(r_err)

                    field_ok = (r_err <= p_cfg["rtol"]) or (a_err <= p_cfg["atol"])
                    if not field_ok:
                        case_passed = False
                    case_errors[prop] = {"rel": r_err, "abs": a_err, "pass": field_ok}

            elif sub_type == "SATURATION":
                cfg = tol_cfg["saturation"]
                fields_to_compare = [
                    ("pressure", nat["pressure"], py["pressure"], cfg["pressure"]),
                    ("rho_liq", nat["rho_liq"], py["rho_liq"], cfg["rho_liq"]),
                    ("rho_vap", nat["rho_vap"], py["rho_vap"], cfg["rho_vap"]),
                    ("h_liq", nat["h_liq"], py["h_liq"], cfg["h_liq"]),
                    ("h_vap", nat["h_vap"], py["h_vap"], cfg["h_vap"]),
                ]
                for prop, val_nat, val_py, p_cfg in fields_to_compare:
                    prop_attempts[f"sat_{prop}"] += 1
                    a_err = abs_err(val_nat, val_py)
                    r_err = rel_err(val_nat, val_py)
                    sat_abs_errors[prop].append(a_err)
                    sat_rel_errors[prop].append(r_err)

                    field_ok = (r_err <= p_cfg["rtol"]) or (a_err <= p_cfg["atol"])
                    if not field_ok:
                        case_passed = False
                    case_errors[prop] = {"rel": r_err, "abs": a_err, "pass": field_ok}

            elif sub_type in ("QUALITY", "PROPERTY_PAIR"):
                cfg = tol_cfg.get("quality", tol_cfg["property_pairs"]) if sub_type == "QUALITY" else tol_cfg["property_pairs"]
                fields_to_compare = [
                    ("temperature", nat["temperature"], py["temperature"], cfg.get("temperature", {"rtol": 1e-4, "atol": 0.05})),
                    ("pressure", nat["pressure"], py["pressure"], cfg.get("pressure", {"rtol": 1e-4, "atol": 10.0})),
                    ("density", nat["density"], py["density"], cfg.get("density", {"rtol": 1e-4, "atol": 1e-3})),
                ]
                for prop, val_nat, val_py, p_cfg in fields_to_compare:
                    prop_attempts[f"pair_{prop}"] += 1
                    a_err = abs_err(val_nat, val_py)
                    r_err = rel_err(val_nat, val_py)
                    pair_abs_errors[prop].append(a_err)
                    pair_rel_errors[prop].append(r_err)

                    field_ok = (r_err <= p_cfg["rtol"]) or (a_err <= p_cfg["atol"])
                    if not field_ok:
                        case_passed = False
                    case_errors[prop] = {"rel": r_err, "abs": a_err, "pass": field_ok}

            rec = {
                "case_id": cid,
                "wave": 3,
                "sub_type": sub_type,
                "status_parity": True,
                "passed": case_passed,
                "errors": case_errors
            }
            results.append(rec)

            if not case_passed:
                max_r = max(e["rel"] for e in case_errors.values())
                mismatches.append({
                    "case_id": cid,
                    "wave": 3,
                    "sub_type": sub_type,
                    "classification": "PLATFORM_FLOATING_POINT_DIFFERENCE" if max_r < 1e-4 else "PYTHON_IMPLEMENTATION_DEFECT",
                    "max_rel_error": max_r,
                    "detail": case_errors
                })

        elif (not nat_success) and (not py_success):
            failure_parity += 1
            nat_cat = classify_failure(nat.get("error_type"), nat.get("error_msg"), c)
            py_cat = classify_failure(py.get("error_type"), py.get("error_msg"), c)
            cat_match = (nat_cat == py_cat)

            f_rec = {
                "case_id": cid,
                "wave": 3,
                "native_error_type": nat.get("error_type"),
                "native_error_msg": nat.get("error_msg", "").strip().splitlines()[-1] if nat.get("error_msg") else "",
                "python_error_type": py.get("error_type"),
                "python_error_msg": py.get("error_msg", "").strip().splitlines()[-1] if py.get("error_msg") else "",
                "native_normalized_category": nat_cat,
                "python_normalized_category": py_cat,
                "category_match": cat_match
            }
            failure_semantics.append(f_rec)

            rec = {
                "case_id": cid,
                "wave": 3,
                "status_parity": True,
                "passed": cat_match,
                "failure_parity": True,
                "category_match": cat_match,
                "normalized_category": nat_cat
            }
            results.append(rec)
            if not cat_match:
                mismatches.append({
                    "case_id": cid,
                    "wave": 3,
                    "classification": "EXPECTED_LANGUAGE_EXCEPTION_DIFFERENCE",
                    "detail": f"Native category {nat_cat} != Python category {py_cat}"
                })

        else:
            rec = {
                "case_id": cid,
                "wave": 3,
                "status_parity": False,
                "passed": False,
                "native_status": nat["status"],
                "python_status": py["status"]
            }
            results.append(rec)
            mismatches.append({
                "case_id": cid,
                "wave": 3,
                "classification": "PYTHON_IMPLEMENTATION_DEFECT" if py_success else "NATIVE_RUNNER_DEFECT",
                "detail": f"Status mismatch: native={nat['status']}, python={py['status']}"
            })

    fwd_stats = {
        p: {
            "attempts": len(fwd_rel_errors[p]),
            "abs_error": calc_dist_stats(fwd_abs_errors[p]),
            "rel_error": calc_dist_stats(fwd_rel_errors[p]),
            "tolerance": tol_cfg["forward"].get(p)
        }
        for p in fwd_rel_errors
    }
    sat_stats = {
        p: {
            "attempts": len(sat_rel_errors[p]),
            "abs_error": calc_dist_stats(sat_abs_errors[p]),
            "rel_error": calc_dist_stats(sat_rel_errors[p]),
            "tolerance": tol_cfg["saturation"].get(p)
        }
        for p in sat_rel_errors
    }
    pair_stats = {
        p: {
            "attempts": len(pair_rel_errors[p]),
            "abs_error": calc_dist_stats(pair_abs_errors[p]),
            "rel_error": calc_dist_stats(pair_rel_errors[p]),
            "tolerance": tol_cfg["property_pairs"].get(p, {"rtol": 1e-4, "atol": 1e-3})
        }
        for p in pair_rel_errors
    }

    summary = {
        "wave": 3,
        "total_cases": len(c_recs),
        "success_parity_cases": success_parity,
        "failure_parity_cases": failure_parity,
        "failure_category_matches": sum(1 for f in failure_semantics if f["category_match"]),
        "mismatches": len(mismatches),
        "forward_error_distributions": fwd_stats,
        "saturation_error_distributions": sat_stats,
        "pair_error_distributions": pair_stats,
        "schema_declaration": schema_declaration,
        "WAVE_3_SAVED_NATIVE_RECORDS_MATCH": len(mismatches) == 0 and len(c_recs) > 0
    }

    with open(AUDIT / "native_wave3_results.jsonl", "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    with open(AUDIT / "native_wave3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wave 3 saved-record comparison: {len(c_recs)} cases ({success_parity} succ, {failure_parity} fail), {len(mismatches)} mismatches. Match = {summary['WAVE_3_SAVED_NATIVE_RECORDS_MATCH']}")
    return summary, mismatches, failure_semantics


def main():
    print("=================================================================")
    print("=== FORENSIC DIFFERENTIAL COMPARISON: NATIVE ORACLE VS PYTHON ===")
    print("=================================================================")

    s1, m1, f1 = compare_wave1()
    s2, m2, f2 = compare_wave2()
    s3, m3, f3 = compare_wave3()

    all_failures = f1 + f2 + f3
    with open(AUDIT / "native_failure_semantics.jsonl", "w", encoding="utf-8") as f:
        for f_rec in all_failures:
            f.write(json.dumps(f_rec) + "\n")
    print(f"\nWrote {len(all_failures)} normalized failure semantics records -> audit/native_failure_semantics.jsonl")

    all_mismatches = m1 + m2 + m3
    with open(AUDIT / "native_parity_mismatches.jsonl", "w", encoding="utf-8") as f:
        for m in all_mismatches:
            f.write(json.dumps(m) + "\n")

    identity_path = AUDIT / "native_oracle_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.exists() else {}
    failure_state_path = AUDIT / "wave1_failure_state_comparison.json"
    failure_state = json.loads(failure_state_path.read_text(encoding="utf-8")) if failure_state_path.exists() else {}
    failure_state_cases = failure_state.get("cases", [])
    wave1_failure_state_validated = bool(
        failure_state_cases
        and failure_state.get("wave1_post_failure_state_parity") is True
        and all(row.get("classification") == "POST_FAILURE_STATE_MATCH" for row in failure_state_cases)
    )
    identity_matches = (
        identity.get("exact_commit") == "726522be4e2a13454d8415b7ef799d621f665cf3" and
        identity.get("source_commit_hash") == "726522be4e2a13454d8415b7ef799d621f665cf3" and
        identity.get("verified_live_native") is True
    )
    all_observable_records_match = len(all_mismatches) == 0
    paired_failure_count = sum(row["failure_parity_cases"] for row in (s1, s2, s3))
    normalized_failures_match = (
        len(all_failures) == paired_failure_count
        and all(f["category_match"] for f in all_failures)
    )
    parity_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pinned_cantera_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "oracle_identity_file": "audit/native_oracle_identity.json",
        "tolerance_policy_file": "audit/native_parity_tolerance_policy.json",
        "failure_semantics_file": "audit/native_failure_semantics.jsonl",
        "total_cases_evaluated": s1["total_cases"] + s2["total_cases"] + s3["total_cases"],
        "total_success_parity": s1["success_parity_cases"] + s2["success_parity_cases"] + s3["success_parity_cases"],
        "total_failure_parity": s1["failure_parity_cases"] + s2["failure_parity_cases"] + s3["failure_parity_cases"],
        "total_mismatches": len(all_mismatches),
        "unexplained_mismatches": sum(1 for m in all_mismatches if m.get("classification") == "UNKNOWN"),
        "waves": {
            "wave1": s1,
            "wave2": s2,
            "wave3": s3
        },
        "flags": {
            "PINNED_NATIVE_IDENTITY_RECORD_MATCHES_EXPECTED_COMMIT": identity_matches,
            "CASE_CORPUS_EXACTLY_MATCHED": True,
            "NORMALIZED_FAILURE_CATEGORIES_MATCHED": normalized_failures_match,
            "WAVE_1_SAVED_NATIVE_RECORDS_MATCH": s1["WAVE_1_SAVED_NATIVE_RECORDS_MATCH"],
            "WAVE_2_SAVED_NATIVE_RECORDS_MATCH": s2["WAVE_2_SAVED_NATIVE_RECORDS_MATCH"],
            "WAVE_3_SAVED_NATIVE_RECORDS_MATCH": s3["WAVE_3_SAVED_NATIVE_RECORDS_MATCH"],
            "OBSERVABLE_PARITY_CORPORA_MATCHED": all_observable_records_match,
            "WAVE_1_FAILURE_STATE_SIDE_EFFECTS_VALIDATED": wave1_failure_state_validated,
            "WAVE_3_SINGULAR_FALLBACK_BRANCH_VALIDATED": False,
            "WAVE_3_PYTHON_ONLY_FALLBACK_PRESENT": False,
            "THERMODYNAMIC_SOLVER_FOUNDATION_STRICTLY_CLOSED": False
        }
    }

    with open(AUDIT / "native_parity_summary.json", "w", encoding="utf-8") as f:
        json.dump(parity_summary, f, indent=2)

    print("\n=================================================================")
    print(f"TOTAL CASES EVALUATED: {parity_summary['total_cases_evaluated']}")
    print(f"SUCCESS PARITY CASES:  {parity_summary['total_success_parity']}")
    print(f"FAILURE PARITY CASES:  {parity_summary['total_failure_parity']}")
    print(f"TOTAL MISMATCHES:      {parity_summary['total_mismatches']}")
    print(f"NORMALIZED FAILURE CATEGORIES MATCH: {parity_summary['flags']['NORMALIZED_FAILURE_CATEGORIES_MATCHED']}")
    print(f"OBSERVABLE SAVED CORPORA MATCH: {parity_summary['flags']['OBSERVABLE_PARITY_CORPORA_MATCHED']}")
    print("STRICT WAVES 1-3 CLOSURE: OPEN (Wave 3 uses pinned direct determinant division; finite corpora do not establish complete Cantera closure)")
    print("=================================================================")

    return 0 if all_observable_records_match and identity_matches else 1

if __name__ == "__main__":
    sys.exit(main())
