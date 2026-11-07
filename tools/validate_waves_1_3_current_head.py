#!/usr/bin/env python3
"""Evidence-derived Waves 1-3 gate.

The gate recomputes set equality, outcomes, tolerances, root accounting and
activation counts from raw records. It treats missing fresh native runs or a
missing regression run as OPEN evidence, never as a pass.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = AUDIT / "waves_1_3_validation_gate.json"
PIN = "726522be4e2a13454d8415b7ef799d621f665cf3"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def git_text(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_native_rerun_provenance(snapshot_id, oracle_identity):
    """Require an explicit, hash-bound run record; row counts are not provenance."""
    path = AUDIT / "waves_1_3_native_rerun_provenance.json"
    data = read_json(path) or {}
    expected = {
        "audit/raw_native_wave1.jsonl": sha(AUDIT / "raw_native_wave1.jsonl"),
        "audit/raw_python_wave1.jsonl": sha(AUDIT / "raw_python_wave1.jsonl"),
        "audit/raw_native_wave2.jsonl": sha(AUDIT / "raw_native_wave2.jsonl"),
        "audit/raw_python_wave2.jsonl": sha(AUDIT / "raw_python_wave2.jsonl"),
        "audit/raw_native_wave2_internal.jsonl": sha(AUDIT / "raw_native_wave2_internal.jsonl"),
        "audit/raw_python_wave2_internal.jsonl": sha(AUDIT / "raw_python_wave2_internal.jsonl"),
        "audit/raw_native_wave3.jsonl": sha(AUDIT / "raw_native_wave3.jsonl"),
        "audit/raw_python_wave3.jsonl": sha(AUDIT / "raw_python_wave3.jsonl"),
        "audit/raw_native_wave3_singular.jsonl": sha(AUDIT / "raw_native_wave3_singular.jsonl"),
        "audit/raw_python_wave3_singular.jsonl": sha(AUDIT / "raw_python_wave3_singular.jsonl"),
        "audit/wave3_cp_cv_step_sweep.jsonl": sha(AUDIT / "wave3_cp_cv_step_sweep.jsonl"),
    }
    recorded = data.get("raw_result_sha256", {})
    source_paths = [
        "cantera_python/state.py", "cantera_python/thermo.py", "cantera_python/ideal_gas.py",
        "cantera_python/cubic_eos.py", "cantera_python/tpx.py", "cantera_python/tpx_exact.py",
        "tools/run_native_oracle_runner.py", "tools/run_python_port_runner.py",
        "tools/run_native_cubic_internal_probe.cpp", "tools/run_python_cubic_internal_probe.py",
        "tools/wave1_failure_state_probe.py", "tools/test_wave3_singular_differential.py",
        "tools/wave3_cp_cv_step_sweep.py", "tools/generate_wave3_fallback_activation_corpus.py",
    ]
    current_source_hashes = {name: sha(ROOT / name) for name in source_paths}
    rehash = read_json(AUDIT / "native_oracle_rehash_verification.json") or {}
    checks = {
        "provenance_file_present": path.is_file(),
        "repository_snapshot_matches": data.get("waves_1_3_snapshot_id") == snapshot_id,
        "git_commit_matches_live_head": data.get("git_commit") == git_text("rev-parse", "HEAD"),
        "pinned_source_matches": data.get("pinned_native_commit") == PIN
            and oracle_identity.get("exact_commit") == PIN,
        "run_is_recent": bool(data.get("run_started_at_utc"))
            and (datetime.now(timezone.utc) - datetime.fromisoformat(data["run_started_at_utc"]).astimezone(timezone.utc)).total_seconds() < 24 * 3600,
        "all_required_runs_complete": set(data.get("completed_runs", []))
            >= {"wave1_native", "wave1_python", "wave2_public_native", "wave2_public_python",
                "wave2_internal_native", "wave2_internal_python", "wave3_native", "wave3_python",
                "wave3_singular_native", "wave3_singular_python", "wave3_fallback", "wave3_cp_cv_step_sweep"},
        "raw_native_hashes_match": all(recorded.get(name) == digest
            for name, digest in expected.items()),
        "execution_source_hashes_match": data.get("execution_source_sha256") == current_source_hashes,
        "native_probe_binary_hash_matches": data.get("native_probe_binary_sha256")
            == sha(ROOT / "tools" / "run_native_cubic_internal_probe"),
        "native_binary_hash_matches": data.get("native_library_sha256")
            == oracle_identity.get("native_library_static_sha256")
            and data.get("native_extension_sha256")
            == oracle_identity.get("native_extension_sha256"),
        "native_binary_hashes_rehashed": rehash.get("verified") is True
            and rehash.get("pinned_native_commit") == PIN
            and rehash.get("native_library_sha256") == oracle_identity.get("native_library_static_sha256")
            and rehash.get("native_extension_sha256") == oracle_identity.get("native_extension_sha256"),
    }
    return {"verified": all(checks.values()), "checks": checks,
            "path": str(path.relative_to(ROOT)), "expected_raw_result_sha256": expected}


def validate_test_run_provenance(current_head, test_result, log_path):
    path = AUDIT / "waves_1_3_test_run_provenance.json"
    data = read_json(path) or {}
    source_paths = sorted((ROOT / "cantera_python").rglob("*.py"))
    source_paths += sorted((ROOT / "tests").rglob("*.py"))
    source_paths += [ROOT / "pyproject.toml", ROOT / "tools" / "run_waves_1_3_pytest.py"]
    current_sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths if p.is_file()}
    current_digest = hashlib.sha256(json.dumps(current_sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checks = {
        "provenance_file_present": path.is_file(),
        "git_commit_matches_live_head": data.get("git_commit") == current_head,
        "source_hashes_match_live_tree": data.get("source_files") == current_sources
            and data.get("test_source_sha256") == current_digest,
        "test_exit_code_zero": data.get("exit_code") == 0 and test_result.get("exit_code") == 0,
        "log_hash_matches": log_path.is_file() and data.get("log_sha256") == sha(log_path)
            and test_result.get("log_sha256") == sha(log_path),
        "parsed_test_counts_match": all(data.get(key) == test_result.get(key)
            for key in ("collected", "passed", "skipped", "failed", "errors")),
    }
    return {"verified": all(checks.values()), "checks": checks,
            "path": str(path.relative_to(ROOT)), "test_source_sha256": current_digest}


def index(records, label):
    mapping = {row["case_id"]: row for row in records}
    if len(mapping) != len(records):
        raise ValueError(f"Duplicate case IDs in {label}")
    return mapping


def tolerance(policy, wave, subtype, field):
    waves = policy.get("waves", {})
    table = waves.get(wave, {})
    if wave == "wave3":
        families = table.get("tolerances", {})
        table = families.get(subtype.lower(), families.get("forward", {}))
        tol = table.get(field)
    else:
        tol = table.get("tolerances", {}).get(field)
    if not tol:
        return None
    # Wave 2 allows separate nominal and near-spinodal tolerance regimes.
    if "rtol" not in tol:
        tol = tol.get("near_spinodal", tol.get("nominal", {}))
    return float(tol["rtol"]), float(tol["atol"])


def failure_category(message: str, row: dict) -> str:
    """Classify the terminal failure diagnostic, not exception stack context."""
    msg = message.lower()
    if any(token in msg for token in ("no convergence", "failed to converge", "maximum iterations", "iteration limit")):
        return "NO_CONVERGENCE"
    if "ph::settemperature" in msg or "temperature must be" in msg or "negative temperature" in msg:
        return "INVALID_TEMPERATURE"
    if "ph::setdensity" in msg or "density must be positive" in msg or "pressure must be finite and positive" in msg:
        return "INVALID_PRESSURE_OR_DENSITY"
    if "volume" in msg:
        return "INVALID_VOLUME"
    return str(row.get("error_type", "OTHER"))


def compare_raw(corpus_path, native_path, python_path, wave, policy):
    corpus, native, python = (read_jsonl(ROOT / p) for p in (corpus_path, native_path, python_path))
    cm, nm, pm = index(corpus, corpus_path), index(native, native_path), index(python, python_path)
    ids_match = cm.keys() == nm.keys() == pm.keys()
    status_mismatches, category_mismatches, property_mismatches = [], [], []
    compared_fields = collections.Counter()
    max_relative = collections.Counter()
    successes = failures = 0
    for cid in sorted(cm.keys() & nm.keys() & pm.keys()):
        c, n, p = cm[cid], nm[cid], pm[cid]
        ns = str(n.get("status", "")).upper()
        ps = str(p.get("status", "")).upper()
        if ns != ps:
            status_mismatches.append(cid)
            continue
        if ns not in ("SUCCESS", "OK", "PASSED"):
            failures += 1
            if wave == "wave1":
                # The corpus category supplies intended diagnostic class; compare
                # independent error text/type classification as well.
                nmsg = str(n.get("error_msg", "")).lower()
                pmsg = str(p.get("error_msg", "")).lower()
                nc, pc = failure_category(nmsg, n), failure_category(pmsg, p)
                if nc != pc:
                    category_mismatches.append({"case_id": cid, "native": nc, "python": pc})
            continue
        successes += 1
        field_aliases = {
            "temperature": "temperature", "pressure": "pressure", "density": "density",
            "enthalpy": "enthalpy", "int_energy": "int_energy", "entropy": "entropy",
            "cp": "cp", "cv": "cv", "molar_volume": "molar_volume", "gibbs": "gibbs",
        }
        subtype = str(c.get("sub_type", c.get("type", "forward"))).replace("_", "").lower()
        for key, field in field_aliases.items():
            if key not in n or key not in p:
                continue
            limits = tolerance(policy, wave, subtype, field)
            if limits is None:
                continue
            rtol, atol = limits
            try:
                a, b = float(n[key]), float(p[key])
            except (TypeError, ValueError):
                continue
            if math.isnan(a) and math.isnan(b):
                continue
            err = abs(a-b)
            rel = err / max(abs(a), abs(b), 1e-300)
            compared_fields[key] += 1
            max_relative[key] = max(max_relative[key], rel)
            if not math.isfinite(err) or err > atol + rtol * abs(a):
                property_mismatches.append({"case_id": cid, "field": key, "native": a, "python": b, "absolute_error": err, "relative_error": rel, "rtol": rtol, "atol": atol})
    return {
        "corpus_records": len(corpus), "native_records": len(native), "python_records": len(python),
        "case_id_sets_match": ids_match, "successes": successes, "failures": failures,
        "status_mismatches": status_mismatches, "failure_category_mismatches": category_mismatches,
        "failure_category_parity_observed_in_saved_records": bool(failures > 0 and not category_mismatches),
        "property_mismatches": property_mismatches, "compared_field_counts": dict(compared_fields),
        "max_relative_error": dict(max_relative),
        "raw_sha256": {"corpus": sha(ROOT / corpus_path), "native": sha(ROOT / native_path), "python": sha(ROOT / python_path)},
    }


def wave2_internal():
    native = index(read_jsonl(AUDIT / "raw_native_wave2_internal.jsonl"), "native Wave 2 internal")
    python = index(read_jsonl(AUDIT / "raw_python_wave2_internal.jsonl"), "Python Wave 2 internal")
    code_n, code_p = collections.Counter(), collections.Counter()
    single_n, single_p, selected_n = collections.Counter(), collections.Counter(), collections.Counter()
    branch_n, branch_p = collections.Counter(), collections.Counter()
    return_mismatch = root_count_mismatch = root_value_mismatch = selected_mismatch = order_mismatch = 0
    selected_state_mismatches = 0
    max_root_relative = max_volume_relative = max_density_relative = 0.0
    case_ids_match = native.keys() == python.keys()
    for cid in native.keys() & python.keys():
        n, p = native[cid], python[cid]
        if n.get("status") != p.get("status"):
            return_mismatch += 1
            continue
        if n.get("status") != "SUCCESS":
            continue
        nc, pc = int(n["nsol"]), int(p["nsol"])
        code_n[nc] += 1; code_p[pc] += 1
        return_mismatch += nc != pc
        nr, pr = abs(nc), abs(pc)
        if nr == 1:
            single_n[nc] += 1
            branch_n["single_root_gas_default" if nc == 1 else "single_root_liquid_or_dense_supercritical"] += 1
        if pr == 1:
            single_p[pc] += 1
            branch_p["single_root_gas_default" if pc == 1 else "single_root_liquid_or_dense_supercritical"] += 1
        root_count_mismatch += nr != pr
        nroots = [float(x) for x in n["vroots"][:nr]]
        proots = [float(x) for x in p["vroots"][:pr]]
        order_mismatch += (nroots != sorted(nroots)) or (proots != sorted(proots)) if nr == 3 and pr == 3 else 0
        if nr != pr or len(nroots) != len(proots):
            root_value_mismatch += 1
            continue
        root_failed = False
        for a, b in zip(nroots, proots):
            rel = abs(a-b) / max(abs(a), 1e-300)
            max_root_relative = max(max_root_relative, rel)
            if abs(a-b) > 1e-7 + 1e-6*abs(a):
                root_failed = True
        root_value_mismatch += root_failed
        if nr == 3:
            def selected(roots, mv):
                return min(range(len(roots)), key=lambda i: abs(roots[i]-float(mv)))
            ni = selected(nroots, n["molar_volume"])
            pi = selected(proots, p["molar_volume"])
            selected_n[ni] += 1
            selected_mismatch += ni != pi
            branch_n[{0: "three_root_selected_liquid_lowest_volume", 1: "three_root_selected_middle_volume", 2: "three_root_selected_vapor_highest_volume"}[ni]] += 1
            branch_p[{0: "three_root_selected_liquid_lowest_volume", 1: "three_root_selected_middle_volume", 2: "three_root_selected_vapor_highest_volume"}[pi]] += 1
        for field, counter in (("molar_volume", "volume"), ("density", "density")):
            a, b = float(n[field]), float(p[field])
            rel = abs(a-b) / max(abs(a), abs(b), 1e-300)
            if counter == "volume":
                max_volume_relative = max(max_volume_relative, rel)
            else:
                max_density_relative = max(max_density_relative, rel)
            if not math.isfinite(a-b) or abs(a-b) > 1e-7 + 1e-6*abs(a):
                selected_state_mismatches += 1
    return {
        "case_ids_match": case_ids_match,
        "native_return_code_counts": {str(k): code_n[k] for k in sorted(code_n)},
        "python_return_code_counts": {str(k): code_p[k] for k in sorted(code_p)},
        "native_single_root_return_code_counts": {str(k): single_n[k] for k in sorted(single_n)},
        "python_single_root_return_code_counts": {str(k): single_p[k] for k in sorted(single_p)},
        "native_three_root_selected_index_counts": {str(k): selected_n[k] for k in sorted(selected_n)},
        "native_source_defined_physical_branch_counts": dict(branch_n),
        "python_source_defined_physical_branch_counts": dict(branch_p),
        "return_code_count_dictionaries_match": code_n == code_p,
        "return_code_mismatches": return_mismatch,
        "root_count_mismatches": root_count_mismatch,
        "root_value_mismatch_cases": root_value_mismatch,
        "selected_root_mismatches": selected_mismatch,
        "root_order_mismatches": order_mismatch,
        "selected_volume_density_mismatches": selected_state_mismatches,
        "root_tolerance": {"rtol": 1e-6, "atol": 1e-7},
        "max_root_relative_error": max_root_relative,
        "max_selected_volume_relative_error": max_volume_relative,
        "max_selected_density_relative_error": max_density_relative,
        "records": len(native),
        "raw_sha256": {name: sha(AUDIT / name) for name in ("raw_native_wave2_internal.jsonl", "raw_python_wave2_internal.jsonl")},
    }


def main():
    # Refresh identity from current files, then bind every derived report to it.
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_waves_1_3_repository_identity.py")], cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
    subprocess.run([sys.executable, str(ROOT / "tools" / "bind_waves_1_3_outputs.py")], cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
    identity = read_json(AUDIT / "waves_1_3_repository_identity.json") or {}
    policy = read_json(AUDIT / "native_parity_tolerance_policy.json") or {}
    oracle_identity = read_json(AUDIT / "native_oracle_identity.json") or {}

    wave1 = compare_raw("audit/corpus_wave1.jsonl", "audit/raw_native_wave1.jsonl", "audit/raw_python_wave1.jsonl", "wave1", policy)
    wave2 = compare_raw("audit/corpus_wave2.jsonl", "audit/raw_native_wave2.jsonl", "audit/raw_python_wave2.jsonl", "wave2", policy)
    wave3 = compare_raw("audit/corpus_wave3.jsonl", "audit/raw_native_wave3.jsonl", "audit/raw_python_wave3.jsonl", "wave3", policy)
    w2internal = wave2_internal()

    singular_native = index(read_jsonl(AUDIT / "raw_native_wave3_singular.jsonl"), "native Wave 3 singular")
    singular_python = index(read_jsonl(AUDIT / "raw_python_wave3_singular.jsonl"), "Python Wave 3 singular")
    singular_common = collections.Counter()
    singular_max = collections.Counter()
    singular_missing_native = collections.Counter()
    for cid in singular_native.keys() & singular_python.keys():
        n, p = singular_native[cid], singular_python[cid]
        if n.get("status") != "SUCCESS" or p.get("status") != "SUCCESS":
            continue
        for key in ("T", "P", "v", "h", "s", "rho", "u", "quality"):
            nv = n.get(key)
            pv = p.get(key)
            if key == "rho":
                nv = 1.0 / float(n["v"]) if "v" in n else None
                pv = 1.0 / float(p["v"]) if "v" in p else None
            if nv is None:
                singular_missing_native[key] += 1
                continue
            if pv is None:
                continue
            rel = abs(float(nv)-float(pv))/max(abs(float(nv)),abs(float(pv)),1e-300)
            singular_common[key] += 1
            singular_max[key] = max(singular_max[key], rel)
    fallback = read_json(AUDIT / "wave3_fallback_activation_probe.json") or {}
    singular_comparison = read_json(AUDIT / "wave3_singular_jacobian_comparison.json") or {}
    singular_py_rows = list(singular_python.values())
    activated_existing = sum(int(r.get("fallback_activation_count", int(bool(r.get("singular_fallback_activated"))))) for r in singular_py_rows)
    activated_targeted = int(fallback.get("activated_count", 0))
    disposition_json = read_json(AUDIT / "wave3_implemented_differently_audit.json") or {}
    dispositions = disposition_json.get("audit_details", [])
    unresolved_diff = [r for r in dispositions if any(word in str(r.get("classification", "")).upper() for word in ("UNVERIFIED", "UNJUSTIFIED", "DIFFERENT_ALGORITHM"))]
    perturbation = read_json(AUDIT / "wave3_cp_cv_perturbation_results.json") or {}
    test_result = read_json(AUDIT / "waves_1_3_test_results.json") or {}
    post_failure_path = AUDIT / "wave1_failure_state_comparison.json"
    post_failure = read_json(post_failure_path) if post_failure_path.exists() else None
    tracked_implementation_hashes = {}
    for rel in ("cantera_python/state.py", "cantera_python/ideal_gas.py", "cantera_python/cubic_eos.py", "cantera_python/tpx.py"):
        tracked_implementation_hashes[rel] = sha(ROOT / rel)

    snapshot_id = identity.get("waves_1_3_snapshot_id")
    library_hash_valid = bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(oracle_identity.get("native_library_static_sha256", ""))))
    extension_hash_valid = bool(re.fullmatch(r"[0-9a-fA-F]{64}", str(oracle_identity.get("native_extension_sha256", ""))))
    native_probe_source = ROOT / "tools" / "run_native_cubic_internal_probe.cpp"
    native_probe_binary = ROOT / "tools" / "run_native_cubic_internal_probe"
    pinned_source_root = ROOT.parent / "cantera-original"
    pinned_source_commit_result = subprocess.run(
        ["git", "-C", str(pinned_source_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) if pinned_source_root.is_dir() else None
    pinned_source_commit = pinned_source_commit_result.stdout.strip() if pinned_source_commit_result and pinned_source_commit_result.returncode == 0 else None
    pinned_source_matches = pinned_source_commit == PIN
    native_identity_present = bool(pinned_source_matches and oracle_identity.get("exact_commit") == PIN and oracle_identity.get("NATIVE_ORACLE_IDENTITY_VERIFIED") is True and library_hash_valid and extension_hash_valid and native_probe_source.is_file() and native_probe_binary.is_file())

    waves_1_3_oracle = read_json(AUDIT / "waves_1_3_native_oracle_identity.json") or {}
    oracle_identity_pin_match = bool(oracle_identity.get("exact_commit") == PIN and waves_1_3_oracle.get("exact_commit") == PIN)
    rehash = read_json(AUDIT / "native_oracle_rehash_verification.json") or {}
    native_files_rehashed = bool(
        rehash.get("verified") is True
        and rehash.get("pinned_native_commit") == PIN
        and rehash.get("native_library_sha256") == oracle_identity.get("native_library_static_sha256")
        and rehash.get("native_extension_sha256") == oracle_identity.get("native_extension_sha256")
        and rehash.get("source_checkout_clean") is True
    )
    native_oracle_verified = bool(
        oracle_identity_pin_match
        and (oracle_identity.get("NATIVE_ORACLE_IDENTITY_VERIFIED") is True or waves_1_3_oracle.get("NATIVE_ORACLE_RESTORED") is True)
        and library_hash_valid
        and extension_hash_valid
        and pinned_source_matches
        and native_files_rehashed
    )
    native_run_provenance = validate_native_rerun_provenance(snapshot_id, oracle_identity)
    native_rechecked = bool(native_oracle_verified and native_run_provenance["verified"])
    # An unobserved branch is not proven unreachable merely because a finite
    # corpus failed to trigger it. If the branch was removed in favor of the
    # pinned source algorithm, verify that source mapping directly instead.
    fallback_proof = fallback.get("formal_unreachable_proof", {})
    fallback_unreachable_proven = bool(
        fallback_proof.get("verified") is True
        and fallback_proof.get("method")
        and fallback_proof.get("source_sha256")
        and fallback_proof.get("proof_artifact_sha256")
    )
    w1_post = "NOT_OBSERVABLE" if post_failure is None else ("TRUE" if post_failure.get("wave1_post_failure_state_parity") else "NOT_EQUIVALENT")
    source_formula = source_formula_check()
    wave1_native_record_evidence = bool(wave1["case_id_sets_match"] and not wave1["property_mismatches"] and not wave1["status_mismatches"])
    wave1_category_record_evidence = bool(wave1["failures"] > 0 and not wave1["failure_category_mismatches"])
    test_log_path = ROOT / str(test_result.get("log_path", "audit/waves_1_3_full_pytest.log"))
    test_log = test_log_path.read_text(encoding="utf-8", errors="replace") if test_log_path.is_file() else ""
    jacobian_source = source_jacobian_check(fallback, test_log)
    inverse_source = source_inverse_solver_check()
    test_summary = re.search(r"(\d+) passed(?:,\s*(\d+) skipped)?(?:,\s*(\d+) failed)?(?:,\s*(\d+) error)?", test_log, re.I)
    actual_head = git_text("rev-parse", "HEAD")
    test_run_provenance = validate_test_run_provenance(actual_head, test_result, test_log_path)
    regression_fresh = bool(test_result.get("started_at_utc") and test_run_provenance["verified"])
    fluid_observed_names = {
        "corpus": sorted({str(r.get("fluid")) for r in read_jsonl(AUDIT / "corpus_wave3.jsonl") if r.get("fluid")}),
        "native": sorted({str(r.get("fluid")) for r in read_jsonl(AUDIT / "raw_native_wave3.jsonl") if r.get("fluid")}),
        "python": sorted({str(r.get("fluid")) for r in read_jsonl(AUDIT / "raw_python_wave3.jsonl") if r.get("fluid")}),
    }
    normalize_fluid = lambda value: "".join(char.lower() for char in str(value) if char.isalnum())
    fluid_sets = {
        engine: sorted({normalize_fluid(name) for name in names})
        for engine, names in fluid_observed_names.items()
    }
    fluid_coverage_match = bool(fluid_sets["corpus"] and fluid_sets["corpus"] == fluid_sets["native"] == fluid_sets["python"])
    declared_diff_count = int(disposition_json.get("breakdown", {}).get("IMPLEMENTED_DIFFERENTLY", -1))
    dispositions_complete = declared_diff_count >= 0 and len(dispositions) == declared_diff_count
    allowed_dispositions = {
        "SEMANTICALLY_PROVEN_EQUIVALENT", "SOURCE_EQUIVALENT",
        "OBSERVABLY_EQUIVALENT_WITH_DIFFERENT_NUMERICS", "INTENTIONAL_NUMERICAL_STABILIZATION",
    }
    set_xy_dispositions = [row for row in dispositions if row.get("name") == "set_xy"]
    set_xy_disposition_current = bool(
        len(set_xy_dispositions) == 1
        and str(set_xy_dispositions[0].get("classification", "")).upper() == "SOURCE_EQUIVALENT"
        and jacobian_source.get("verified")
    )
    dispositions_justified = bool(dispositions_complete and not unresolved_diff and all(
        str(row.get("classification", "")).upper() in allowed_dispositions for row in dispositions
    ) and set_xy_disposition_current)
    w3_inventory_rows = read_jsonl(AUDIT / "solver_wave3_source_inventory.jsonl")
    inventory_diff_count = sum(row.get("disposition") == "IMPLEMENTED_DIFFERENTLY" for row in w3_inventory_rows)
    disposition_md = (AUDIT / "wave3_implemented_differently_disposition.md").read_text(encoding="utf-8") if (AUDIT / "wave3_implemented_differently_disposition.md").is_file() else ""
    disposition_table_rows = len(re.findall(r"^\|\s*\d+\s*\|", disposition_md, re.M))
    fallback_instrumentation_complete = bool(
        fallback.get("attempt_count") == len(fallback.get("all_attempts", []))
        and all(all(key in row for key in (
            "fallback_activated", "fallback_activation_count", "fallback_threshold",
            "determinants", "fallback_iteration_numbers", "fallback_pre_states",
            "fallback_proposed_steps", "fallback_post_states", "fallback_events",
        )) for row in fallback.get("all_attempts", []))
    )
    singular_saved_observables_complete = bool(
        singular_comparison.get("cases") == len(singular_native) == len(singular_python)
        and all(key in singular_comparison.get("error_distributions", {}) for key in ("T", "P", "v", "rho", "h", "s"))
        and all(key in singular_comparison.get("unavailable_native_observables", {}) for key in ("internal_energy", "quality"))
        and "target_property_residuals" in singular_comparison
    )

    wave1_fresh_native_parity = bool(wave1_native_record_evidence and native_rechecked)
    wave1_failure_category_parity = bool(wave1_category_record_evidence and native_rechecked)
    wave1_success_source_parity = bool(inverse_source["verified"])
    wave1_current_certified = bool(
        wave1_native_record_evidence and wave1_category_record_evidence
        and wave1_success_source_parity and w1_post == "TRUE" and native_rechecked
    )

    wave2_fresh_native_parity = bool(wave2["case_id_sets_match"] and not wave2["property_mismatches"] and not wave2["status_mismatches"] and native_rechecked)
    wave2_return_code_parity = bool(w2internal["case_ids_match"] and w2internal["return_code_count_dictionaries_match"] and not w2internal["return_code_mismatches"])
    wave2_internal_root_numerical_parity = bool(w2internal["case_ids_match"] and not w2internal["root_count_mismatches"] and not w2internal["root_value_mismatch_cases"] and not w2internal["selected_root_mismatches"] and not w2internal["root_order_mismatches"] and not w2internal["selected_volume_density_mismatches"])
    wave2_branch_report_corrected = bool(
        w2internal["return_code_count_dictionaries_match"]
        and set(w2internal["native_return_code_counts"]).issuperset({"-1", "1", "3"})
        and w2internal["native_single_root_return_code_counts"] == w2internal["python_single_root_return_code_counts"]
        and w2internal["native_source_defined_physical_branch_counts"] == w2internal["python_source_defined_physical_branch_counts"]
        and not w2internal["selected_root_mismatches"]
    )
    wave2_current_certified = bool(native_rechecked and w2internal["case_ids_match"] and not w2internal["root_count_mismatches"] and not w2internal["root_value_mismatch_cases"] and not w2internal["selected_root_mismatches"] and not wave2["property_mismatches"])

    wave3_fresh_native_parity = bool(wave3["case_id_sets_match"] and not wave3["property_mismatches"] and not wave3["status_mismatches"] and native_rechecked and fluid_coverage_match)
    wave3_failure_category_parity = bool(wave3["case_id_sets_match"] and not wave3["failure_category_mismatches"] and native_rechecked)
    wave3_fallback_found = bool(activated_targeted > 0)
    no_python_fallback = fallback.get("python_only_fallback_present") is False
    if activated_targeted > 0:
        wave3_fallback_validated = "ACTIVATED"
    elif fallback_unreachable_proven:
        wave3_fallback_validated = "UNREACHABLE_PROVEN"
    elif no_python_fallback and jacobian_source["verified"]:
        wave3_fallback_validated = "NOT_APPLICABLE_DIRECT_SOURCE_ALGORITHM_VERIFIED"
    else:
        wave3_fallback_validated = False
    wave3_formula_verified = bool(source_formula["verified"])
    wave3_cause_established = bool(
        perturbation.get("cause_conclusion") == "CAUSE_ESTABLISHED"
        and perturbation.get("waves_1_3_snapshot_id") == snapshot_id
        and perturbation.get("status", "").startswith("RUN_COMPLETE")
        and native_rechecked
    )
    wave3_jacobian_path_disposed = bool(
        activated_targeted > 0 or fallback_unreachable_proven
        or (no_python_fallback and jacobian_source["verified"])
    )
    wave3_current_certified = bool(native_rechecked and fluid_coverage_match and wave3_jacobian_path_disposed and source_formula["verified"] and perturbation.get("status", "").startswith("RUN_COMPLETE") and dispositions_justified and wave3_failure_category_parity)

    full_regression_zero_failures = bool(regression_fresh and test_result.get("exit_code") == 0 and test_result.get("failed", -1) == 0 and test_result.get("errors", -1) == 0)
    waves_1_3_gate_derived = bool(
        wave1["corpus_records"] > 0 and wave2["corpus_records"] > 0 and wave3["corpus_records"] > 0
        and wave1["case_id_sets_match"] and wave2["case_id_sets_match"] and wave3["case_id_sets_match"]
        and w2internal["records"] > 0 and len(singular_native) > 0 and len(singular_python) > 0
        and fallback_instrumentation_complete and singular_saved_observables_complete
        and regression_fresh and test_result.get("exit_code") == 0 and native_identity_present
        and native_rechecked
    )

    actual_head = git_text("rev-parse", "HEAD")
    actual_branch = git_text("branch", "--show-current")
    actual_status = git_text("status", "--porcelain=v1", "--untracked-files=all") or ""
    identity_exclusions = set(identity.get("identity_hash_exclusions", []))
    def status_path(line):
        # git_text strips the complete command output, which removes the
        # leading status column's space on its first line. Handle both the
        # normal three-character porcelain prefix and that first-line form.
        offset = 3 if len(line) > 2 and line[2].isspace() else 2
        path = line[offset:].split(" -> ")[-1].strip().strip('"')
        return path.replace("\\", "/")
    filtered_status = "\n".join(
        line for line in actual_status.splitlines()
        if status_path(line) not in identity_exclusions
    )
    identity_matches_worktree = bool(
        identity.get("git_commit") == actual_head
        and identity.get("branch") == actual_branch
        and identity.get("status_porcelain_excluding_self_referential_certification_outputs") == filtered_status
    )
    is_clean_commit = bool(
        identity_matches_worktree and not filtered_status
        and identity.get("clean") is True
        and identity.get("snapshot_scope_clean") is True
        and identity.get("snapshot_kind") == "CLEAN_COMMITTED_SNAPSHOT"
    )
    committed_snapshot_certified = bool(
        is_clean_commit
        and wave1_current_certified
        and wave2_current_certified
        and wave3_current_certified
        and full_regression_zero_failures
        and waves_1_3_gate_derived
    )
    ready_for_wave4_freeze = committed_snapshot_certified

    flags = {
        "PINNED_NATIVE_ORACLE_RESTORED": native_oracle_verified,
        "FULL_PYTHON_TEST_ENV_RESTORED": bool(regression_fresh and test_result.get("pytest_available")),
        "WAVE_1_FRESH_NATIVE_PARITY": wave1_fresh_native_parity,
        "WAVE_1_SUCCESS_PATH_SOURCE_PARITY": wave1_success_source_parity,
        "WAVE_1_FAILURE_CATEGORY_PARITY": wave1_failure_category_parity,
        "WAVE_1_POST_FAILURE_STATE_PARITY": w1_post,
        "WAVE_1_CURRENT_SNAPSHOT_CERTIFIED": wave1_current_certified,
        "WAVE_2_FRESH_NATIVE_PARITY": wave2_fresh_native_parity,
        "WAVE_2_RETURN_CODE_PARITY": wave2_return_code_parity,
        "WAVE_2_INTERNAL_ROOT_NUMERICAL_PARITY": wave2_internal_root_numerical_parity,
        "WAVE_2_CURRENT_SNAPSHOT_CERTIFIED": wave2_current_certified,
        "WAVE_3_FRESH_NATIVE_PARITY": wave3_fresh_native_parity,
        "WAVE_3_FALLBACK_ACTIVATION_FOUND": wave3_fallback_found,
        "WAVE_3_FALLBACK_ACTIVATION_VALIDATED": wave3_fallback_found,
        "WAVE_3_FALLBACK_BEHAVIOR_VALIDATED": wave3_fallback_validated,
        "WAVE_3_FAILURE_CATEGORY_PARITY": wave3_failure_category_parity,
        "WAVE_3_DIRECT_JACOBIAN_SOURCE_PARITY": jacobian_source["verified"],
        "WAVE_3_CP_CV_FORMULA_VERIFIED": wave3_formula_verified,
        "WAVE_3_CP_CV_DIFFERENCE_CAUSE_ESTABLISHED": wave3_cause_established,
        "WAVE_3_DIFFERENT_IMPLEMENTATIONS_JUSTIFIED": dispositions_justified,
        "WAVE_3_CURRENT_SNAPSHOT_CERTIFIED": wave3_current_certified,
        "FULL_REGRESSION_ZERO_FAILURES": full_regression_zero_failures,
        "WAVES_1_3_VALIDATION_GATE_DERIVED_FROM_FRESH_RAW_EVIDENCE": waves_1_3_gate_derived,
        "WAVES_1_3_VALIDATION_GATE_DERIVED_FROM_RAW_EVIDENCE": waves_1_3_gate_derived,
        "WAVES_1_3_CLEAN_COMMIT_CREATED": is_clean_commit,
        "WAVES_1_3_COMMITTED_SNAPSHOT_BOUND": is_clean_commit,
        "WAVES_1_3_COMMITTED_SNAPSHOT_CERTIFIED": committed_snapshot_certified,
        "THERMODYNAMIC_SOLVER_FOUNDATION_READY_FOR_WAVE_4_FINAL_FREEZE": ready_for_wave4_freeze,
        "WAVE_1_SUCCESS_PATH_NATIVE_PARITY": wave1_fresh_native_parity,
        "WAVE_2_BRANCH_REPORT_CORRECTED": wave2_branch_report_corrected,
    }
    final_cert = committed_snapshot_certified
    limitations = [
        "Repository identity and live Git status are included in this report; a dirty worktree is not represented as a committed snapshot.",
    ]
    if w1_post != "TRUE":
        matched = (post_failure or {}).get("classification_counts", {}).get("POST_FAILURE_STATE_MATCH", 0)
        total = len((post_failure or {}).get("cases", []))
        limitations.append(f"Wave 1 post-failure state parity is {w1_post}: {matched}/{total} probe states match.")
    if not fallback_unreachable_proven and activated_targeted == 0 and not no_python_fallback:
        limitations.append("Finite targeted attempts with no Wave 3 fallback activation do not prove the branch unreachable; a formal proof artifact is required.")
    if no_python_fallback:
        limitations.append("The Python-only gradient fallback is absent and the direct determinant division is source-mapped; finite targeted samples do not establish coverage of every singular thermodynamic state.")
    if not wave3_cause_established:
        limitations.append("The cp/cv step sweep measures manual estimators and does not establish the cause of production cp/cv differences.")

    report = {
        "waves_1_3_snapshot_id": identity.get("waves_1_3_snapshot_id"),
        "repository_identity": {
            "commit": identity.get("git_commit"), "branch": identity.get("branch"),
            "clean": identity.get("clean"), "snapshot_kind": identity.get("snapshot_kind"),
            "working_tree_diff_sha256": identity.get("working_tree_diff_sha256"),
            "tracked_worktree_tree_sha256": identity.get("tracked_worktree_tree_sha256"),
        },
        "pinned_native_commit": PIN,
        "native_identity_file_matches_pin": oracle_identity.get("exact_commit") == PIN and oracle_identity.get("NATIVE_ORACLE_IDENTITY_VERIFIED") is True,
        "native_oracle_revalidated_in_this_run": native_rechecked,
        "native_rerun_provenance": native_run_provenance,
        "test_run_provenance": test_run_provenance,
        "git_identity_matches_live_worktree": identity_matches_worktree,
        "live_git_head": actual_head,
        "live_git_branch": actual_branch,
        "live_git_status_porcelain": actual_status,
        "live_git_status_excluding_identity_hash_exclusions": filtered_status,
        "current_python_hashes": tracked_implementation_hashes,
        "wave1_raw": wave1,
        "wave1_saved_record_evidence": {
            "successful_observable_parity_in_saved_records": wave1_native_record_evidence,
            "failure_category_parity_in_saved_records": wave1_category_record_evidence,
            "native_rerun_current_snapshot": native_rechecked,
            "failure_category_counts_native": dict(collections.Counter(
                failure_category(str(row.get("error_msg", "")), row)
                for row in read_jsonl(AUDIT / "raw_native_wave1.jsonl")
                if str(row.get("status", "")).upper() not in ("SUCCESS", "OK", "PASSED")
            )),
            "failure_category_counts_python": dict(collections.Counter(
                failure_category(str(row.get("error_msg", "")), row)
                for row in read_jsonl(AUDIT / "raw_python_wave1.jsonl")
                if str(row.get("status", "")).upper() not in ("SUCCESS", "OK", "PASSED")
            )),
        },
        "wave1_post_failure_state": post_failure if post_failure is not None else {"classification": "NOT_OBSERVABLE", "reason": "native and Python before/after failure-state raw records are absent"},
        "wave1_source_mapping": inverse_source,
        "wave2_raw": wave2,
        "wave2_internal_raw": w2internal,
        "wave3_raw": wave3,
        "wave3_fluid_coverage": {
            "observed_names": fluid_observed_names,
            "normalized_names": fluid_sets,
            "sets_match": fluid_coverage_match,
        },
        "wave3_different_implementation_disposition_count_computed": len(dispositions),
        "wave3_different_implementation_disposition_source": "audit/wave3_implemented_differently_audit.json",
        "wave3_different_implementation_dispositions": {
            "declared_count_from_breakdown": declared_diff_count,
            "inventory_count_from_source_rows": inventory_diff_count,
            "roster_rows_in_disposition_document": disposition_table_rows,
            "detailed_records_count": len(dispositions),
            "all_declared_records_present": dispositions_complete,
            "all_have_accepted_classification": dispositions_justified,
            "set_xy_disposition_matches_current_source": set_xy_disposition_current,
            "set_xy_source_mapping_verified": jacobian_source.get("verified", False),
            "unresolved_method_names": disposition_json.get("record_accounting", {}).get("unresolved_method_names", []),
        },
        "wave3_unresolved_different_implementation_records": unresolved_diff,
        "wave3_singular": {
            "native_records": len(singular_native), "python_records": len(singular_python),
            "common_observables_count": dict(singular_common), "max_relative_error": dict(singular_max),
            "native_unavailable_observables": dict(singular_missing_native),
            "fallback_activations_existing_88": activated_existing,
            "targeted_fallback_cases": fallback.get("attempt_count", 0),
            "targeted_fallback_activations": activated_targeted,
            "fallback_instrumentation_records_complete": fallback_instrumentation_complete,
            "saved_all_observable_comparison_complete": singular_saved_observables_complete,
            "saved_error_distributions": singular_comparison.get("error_distributions", {}),
            "saved_property_residuals": singular_comparison.get("target_property_residuals", {}),
            "native_missing_observable_counts": singular_comparison.get("unavailable_native_observables", {}),
            "targeted_min_abs_determinant": min((r.get("determinant_summary", {}).get("minimum_absolute_finite") for r in fallback.get("all_attempts", []) if r.get("determinant_summary") and r["determinant_summary"].get("minimum_absolute_finite") is not None), default=None),
        },
        "wave3_jacobian_source_mapping": jacobian_source,
        "wave3_cp_cv": {
            **source_formula,
            "perturbation_status": perturbation.get("status"),
            "cause_conclusion": perturbation.get("cause_conclusion", "CAUSE_NOT_FULLY_ESTABLISHED"),
        },
        "regression": {
            **test_result,
            "log_exists": test_log_path.is_file(),
            "log_sha256": sha(test_log_path) if test_log_path.is_file() else None,
            "summary_parsed_from_log": [int(value or 0) for value in test_summary.groups()] if test_summary else None,
            "fresh_run_record_present": regression_fresh,
            "full_suite_passed": bool(regression_fresh and test_result.get("exit_code") == 0 and test_summary and not test_summary.group(3) and not test_summary.group(4)),
        },
        "native_identity_validation": {
            "identity_commit_matches_pin": oracle_identity.get("exact_commit") == PIN,
            "pinned_source_checkout_commit": pinned_source_commit,
            "pinned_source_checkout_matches": pinned_source_matches,
            "library_recorded_hash_valid": library_hash_valid,
            "extension_recorded_hash_valid": extension_hash_valid,
            "native_library_bytes_rehashed_here": native_files_rehashed,
            "native_binary_files_exist_and_rehashed_here": native_files_rehashed,
            "native_probe_source_sha256": sha(native_probe_source),
            "native_probe_binary_sha256": sha(native_probe_binary),
            "identity_artifacts_present": native_identity_present,
        },
        "waves_1_3_snapshot_id": identity.get("waves_1_3_snapshot_id"),
        "flags": flags,
        "WAVES_1_3_COMMITTED_SNAPSHOT_CERTIFIED": final_cert,
        "limitations": limitations,
    }
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(flags, indent=2))
    print(f"WAVES_1_3_COMMITTED_SNAPSHOT_CERTIFIED = {str(final_cert).upper()}")
    print(f"THERMODYNAMIC_SOLVER_FOUNDATION_READY_FOR_WAVE_4_FINAL_FREEZE = {str(ready_for_wave4_freeze).upper()}")
    return 0 if final_cert else 1


def source_formula_check():
    """Check mapped source equations directly; do not trust report-file presence."""
    native_path = ROOT.parent / "cantera-original" / "src" / "tpx" / "Sub.cpp"
    python_path = ROOT / "cantera_python" / "tpx.py"
    audit_path = AUDIT / "wave3_cp_cv_formula_audit.md"
    if not (native_path.is_file() and python_path.is_file() and audit_path.is_file()):
        return {"verified": False, "reason": "source or formula audit file missing"}
    native = native_path.read_text(encoding="utf-8")
    python = python_path.read_text(encoding="utf-8")
    audit = audit_path.read_text(encoding="utf-8")
    checks = {
        "native_nominal_step": "dt = 1.e-4 * T" in native or "dt = 1e-4 * T" in native,
        "python_nominal_step": "dt = 1e-4 * self.T" in python or "dt = 1.e-4 * self.T" in python,
        "native_cv_entropy_difference": "return T * (s2 - s1) / (T2 - T1);" in native,
        "python_cv_entropy_difference": "s2 - s1" in python,
        "native_cp_entropy_difference": "(s2 - s1)" in native,
        "python_cp_entropy_difference": "s2 - s1" in python,
        "audit_records_source_line_mapping": "Sub.cpp:44-80" in audit and "tpx.py:301-324" in audit and "Sub.cpp:82-132" in audit and "tpx.py:326-364" in audit,
        "audit_states_held_and_restored": "specific volume fixed" in audit and "pressure" in audit and "restore" in audit.lower(),
    }
    return {"verified": all(checks.values()), "checks": checks,
            "native_source_sha256": sha(native_path), "python_source_sha256": sha(python_path),
            "audit_sha256": sha(audit_path),
            "native_source_path": str(native_path.relative_to(ROOT.parent)),
            "python_source_path": str(python_path.relative_to(ROOT))}


def source_inverse_solver_check():
    """Check the Wave 1 Python update rules against the pinned native source."""
    native_path = ROOT.parent / "cantera-original" / "src" / "thermo" / "ThermoPhase.cpp"
    python_state_path = ROOT / "cantera_python" / "state.py"
    python_phase_path = ROOT / "cantera_python" / "ideal_gas.py"
    paths = (native_path, python_state_path, python_phase_path)
    if not all(path.is_file() for path in paths):
        return {"verified": False, "reason": "native or Python inverse-solver source is missing"}
    native = re.sub(r"\s+", "", native_path.read_text(encoding="utf-8"))
    python = re.sub(r"\s+", "", python_state_path.read_text(encoding="utf-8"))
    phase = re.sub(r"\s+", "", python_phase_path.read_text(encoding="utf-8"))
    checks = {
        "native_and_python_hp_uv_newton_step": (
            "clip((Htarget-Hold)/cpd,-100.0,100.0)" in native
            and "_clip((target-old_value)/old_deriv,-100.0,100.0)" in python
        ),
        "native_and_python_sp_sv_newton_step": (
            "clip((Starget-Sold)*Told/cpd,-100.0,100.0)" in native
            and "_clip((target-old_value)*told/old_deriv,-100.0,100.0)" in python
        ),
        "distinct_hp_uv_and_sp_sv_bracket_rules": (
            "Tnew<(0.75*Tbot+0.25*Told)" in native
            and "tnew<(0.75*bot_t+0.25*told)" in python
            and "Tnew<Tbot" in native and "tnew<bot_t" in python
            and "Tnew>(0.75*Ttop+0.25*Told)" in native
            and "tnew>(0.75*top_t+0.25*told)" in python
            and "Tnew>Ttop" in native and "tnew>top_t" in python
        ),
        "negative_heat_capacity_retry": (
            "for(intits=0;its<10;its++)" in native and "for_inrange(10):" in python
            and "dt*=0.25" in native and "dt*=0.25" in python
        ),
        "hp_uv_only_one_third_temperature_floor": (
            "if(Tnew<Told/3.0)" in native
            and "quantityin('H','U')andtnew<told/3.0" in python
        ),
        "source_temperature_bounds_and_convergence_tests": (
            "clip(minTemp(),SmallNumber,100.0)" in native
            and "min(100.0,max(min_temp,1e-300))" in python
            and "fabs((Herr)/denom)" in native and "fabs((Serr*Tnew)/denom)" in native
            and "abs(error/denom)" in python and "abs(error*tnew/denom)" in python
            and "fabs(dt/Tnew)<rtol" in native and "abs(dt/tnew)<rtol" in python
        ),
        "fixed_pressure_or_specific_volume_constraints": (
            "setState_conditional_TP(Tnew,p,!doUV)" in native
            and "ifdo_volume:p=R*t/(phase.mean_molecular_weight*second)else:p=second" in python
            and "deriv_name='cv_mass'ifdo_volumeelse'cp_mass'" in python
        ),
        "native_and_python_default_iteration_limit": (
            "for(intn=0;n<500;n++)" in native
            and "defsetState_HP(self,h,p,*,rtol=1e-9,max_iterations=500)" in phase
            and "defsetState_SP(self,s,p,*,rtol=1e-9,max_iterations=500)" in phase
        ),
        "wrapper_failure_state_restoration": (
            native.count("savePartialState(state)") >= 4
            and native.count("restorePartialState(state)") >= 4
            and "original_state=self._state" in phase
            and "exceptException:self._state=original_state" in phase
            and "self._state=result" in phase
        ),
        "python_domain_is_explicitly_ideal_gas": "isinstance(phase,IdealGasMixture)" in python,
    }
    return {
        "verified": all(checks.values()),
        "verification_method": "normalized source-pattern checks for mapped algorithm branches; empirical parity is separately gate-checked",
        "checks": checks,
        "native_source_sha256": sha(native_path),
        "python_state_sha256": sha(python_state_path),
        "python_phase_sha256": sha(python_phase_path),
        "native_source_path": str(native_path.relative_to(ROOT.parent)),
        "python_state_path": str(python_state_path.relative_to(ROOT)),
        "python_phase_path": str(python_phase_path.relative_to(ROOT)),
    }


def source_jacobian_check(fallback, test_log):
    """Verify the TPX direct determinant update after removing the Python-only fallback."""
    native_path = ROOT.parent / "cantera-original" / "src" / "tpx" / "Sub.cpp"
    python_path = ROOT / "cantera_python" / "tpx.py"
    if not (native_path.is_file() and python_path.is_file()):
        return {"verified": False, "reason": "native or Python TPX source is missing"}
    native = native_path.read_text(encoding="utf-8")
    python = python_path.read_text(encoding="utf-8")
    try:
        method = python.split("    def _set_xy_source", 1)[1].split("\n    def ", 1)[0]
    except IndexError:
        method = ""
    checks = {
        "native_direct_determinant": "double det = dxdt*dydv - dydt*dxdv;" in native,
        "native_direct_temperature_division": "dt = ((X - x_here)*dydv - (Y - y_here)*dxdv)/det;" in native,
        "native_direct_volume_division": "dv = ((Y - y_here)*dxdt - (X - x_here)*dydt)/det;" in native,
        "python_source_translated_method_present": bool(method),
        "python_uses_ieee_division_for_both_updates": method.count("_ieee_divide(") == 2 and method.count(", det)") == 2,
        "python_gradient_regularization_absent": "abs(det)" not in method and "not math.isfinite(det)" not in method,
        "fallback_probe_says_branch_absent": fallback.get("python_only_fallback_present") is False,
        "ieee_edge_tests_present_in_fresh_log": bool(re.search(r"test_tpx_ieee_arithmetic\.py\s+\.{7}", test_log)),
    }
    return {
        "verified": all(checks.values()), "checks": checks,
        "disposition": "SOURCE_EQUIVALENT_DIRECT_DIVISION" if all(checks.values()) else "UNVERIFIED",
        "native_source_sha256": sha(native_path), "python_source_sha256": sha(python_path),
        "native_source_path": str(native_path.relative_to(ROOT.parent)),
        "python_source_path": str(python_path.relative_to(ROOT)),
    }


if __name__ == "__main__":
    raise SystemExit(main())
