"""Orchestrate end-to-end ReactorNet differential validation campaign."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

def to_wsl_path(win_path: Path) -> str:
    p = str(win_path.resolve()).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return f"/mnt/{drive}{p[2:]}"
    return p

def file_hash(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def compute_run_id(corpus_path: Path, commit: str, native_hash: str) -> str:
    h = hashlib.sha256()
    with open(corpus_path, "rb") as f:
        h.update(f.read())
    h.update(commit.encode("utf-8"))
    h.update(native_hash.encode("utf-8"))
    return f"rnet_diff_{h.hexdigest()[:16]}"

def rel_err(a, b, floor=1e-12):
    denom = max(abs(a), abs(b), floor)
    return abs(a - b) / denom

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    corpus_path = audit_dir / "corpus_reactor_network.jsonl"
    native_out = audit_dir / "reactor_net_native_oracle_results.jsonl"
    python_out = audit_dir / "reactor_net_python_port_results.jsonl"
    summary_path = audit_dir / "reactor_network_differential_summary.json"

    commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    native_lib_sha = "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744"
    run_id = compute_run_id(corpus_path, commit, native_lib_sha)
    print(f"Executing campaign with run_id: {run_id}")

    # 1. Run Native Oracle via WSL
    # Convert paths for WSL
    wsl_corpus = to_wsl_path(corpus_path)
    wsl_out = to_wsl_path(native_out)
    wsl_script = to_wsl_path(repo_root / "tools" / "run_native_reactor_net_oracle.py")

    print("Running native Cantera oracle in WSL...")
    cmd_native = [
        "wsl", "/home/nebula/cantera-oracle-env/bin/python3",
        wsl_script, wsl_corpus, wsl_out, run_id
    ]
    subprocess.check_call(cmd_native)
    print("Native oracle execution finished.")

    # 2. Run Python Port Runner
    print("Running Python port runner...")
    cmd_py = [
        sys.executable,
        str(repo_root / "tools" / "run_python_reactor_net_port.py"),
        str(corpus_path), str(python_out), run_id
    ]
    subprocess.check_call(cmd_py)
    print("Python port runner execution finished.")

    # 3. Read and compare results
    native_results = {}
    with open(native_out, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                native_results[row["case_id"]] = row

    python_results = {}
    with open(python_out, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                python_results[row["case_id"]] = row

    # Comparison metrics
    total_cases = len(native_results)
    passed_cases = 0
    failures = []
    temp_rel_errors = []
    press_rel_errors = []
    mass_rel_errors = []
    sens_rel_errors = []
    ign_delay_abs_errors = []
    ign_delay_rel_errors = []

    # Tolerances
    tol_T = 1e-3  # 0.1% max relative error for dynamic integration
    tol_P = 1e-3
    tol_mass = 1e-4
    tol_sens = 1e-2  # 1% for forward sensitivities

    case_comparisons = []

    for cid, nat in native_results.items():
        pyt = python_results.get(cid)
        if not pyt:
            failures.append({"case_id": cid, "reason": "Missing Python result"})
            continue

        cat = nat["category"]
        case_passed = True
        case_max_T_err = 0.0
        case_max_P_err = 0.0
        case_max_mass_err = 0.0

        if cat == "failure_semantics":
            # Compare failure caught
            if nat.get("failure_caught") == pyt.get("failure_caught"):
                passed_cases += 1
            else:
                failures.append({"case_id": cid, "reason": f"Failure caught mismatch: native={nat.get('failure_caught')}, py={pyt.get('failure_caught')}"})
            continue

        # Check ignition delay if present
        if "ignition_delay_s" in nat.get("metrics", {}) and "ignition_delay_s" in pyt.get("metrics", {}):
            nat_ign = nat["metrics"]["ignition_delay_s"]
            py_ign = pyt["metrics"]["ignition_delay_s"]
            if nat_ign is not None and py_ign is not None:
                abs_diff = abs(nat_ign - py_ign)
                rel_diff = abs_diff / max(nat_ign, 1e-12)
                ign_delay_abs_errors.append(abs_diff)
                ign_delay_rel_errors.append(rel_diff)
                if rel_diff > 0.05:  # within 5% for discrete ignition detection
                    case_passed = False
                    failures.append({"case_id": cid, "reason": f"Ignition delay mismatch: native={nat_ign}, py={py_ign}, rel_err={rel_diff}"})

        # Compare checkpoints
        nat_cps = nat.get("checkpoints", [])
        py_cps = pyt.get("checkpoints", [])

        if len(nat_cps) != len(py_cps):
            failures.append({"case_id": cid, "reason": f"Checkpoint count mismatch: native={len(nat_cps)}, py={len(py_cps)}"})
            continue

        for i, (n_cp, p_cp) in enumerate(zip(nat_cps, py_cps)):
            # Temperature
            if "temperature" in n_cp and "temperature" in p_cp:
                err = rel_err(n_cp["temperature"], p_cp["temperature"])
                temp_rel_errors.append(err)
                case_max_T_err = max(case_max_T_err, err)
                if err > tol_T:
                    case_passed = False
                    failures.append({"case_id": cid, "reason": f"T rel error {err} > {tol_T} at cp {i}"})

            # Pressure
            if "pressure" in n_cp and "pressure" in p_cp:
                err = rel_err(n_cp["pressure"], p_cp["pressure"])
                press_rel_errors.append(err)
                case_max_P_err = max(case_max_P_err, err)
                if err > tol_P:
                    case_passed = False
                    failures.append({"case_id": cid, "reason": f"P rel error {err} > {tol_P} at cp {i}"})

            # Mass
            if "mass" in n_cp and "mass" in p_cp:
                err = rel_err(n_cp["mass"], p_cp["mass"])
                mass_rel_errors.append(err)
                case_max_mass_err = max(case_max_mass_err, err)
                if err > tol_mass:
                    case_passed = False
                    failures.append({"case_id": cid, "reason": f"Mass rel error {err} > {tol_mass} at cp {i}"})

            # Sensitivities
            if "sens_rxn0_temp" in n_cp and "sens_rxn0_temp" in p_cp:
                s_nat = n_cp["sens_rxn0_temp"]
                s_py = p_cp["sens_rxn0_temp"]
                err = rel_err(s_nat, s_py, floor=1e-4)
                sens_rel_errors.append(err)
                if err > tol_sens:
                    case_passed = False
                    failures.append({"case_id": cid, "reason": f"Sens rel error {err} > {tol_sens} at cp {i}"})

        if case_passed:
            passed_cases += 1

        case_comparisons.append({
            "case_id": cid,
            "category": cat,
            "passed": case_passed,
            "max_T_rel_err": case_max_T_err,
            "max_P_rel_err": case_max_P_err,
            "max_mass_rel_err": case_max_mass_err,
        })

    summary = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_percent": (passed_cases / total_cases) * 100.0 if total_cases > 0 else 0.0,
        "metrics": {
            "temperature_relative_error": {
                "max": float(np.max(temp_rel_errors)) if temp_rel_errors else 0.0,
                "p50": float(np.percentile(temp_rel_errors, 50)) if temp_rel_errors else 0.0,
                "p95": float(np.percentile(temp_rel_errors, 95)) if temp_rel_errors else 0.0,
                "p99": float(np.percentile(temp_rel_errors, 99)) if temp_rel_errors else 0.0,
            },
            "pressure_relative_error": {
                "max": float(np.max(press_rel_errors)) if press_rel_errors else 0.0,
                "p50": float(np.percentile(press_rel_errors, 50)) if press_rel_errors else 0.0,
                "p95": float(np.percentile(press_rel_errors, 95)) if press_rel_errors else 0.0,
                "p99": float(np.percentile(press_rel_errors, 99)) if press_rel_errors else 0.0,
            },
            "mass_relative_error": {
                "max": float(np.max(mass_rel_errors)) if mass_rel_errors else 0.0,
                "p50": float(np.percentile(mass_rel_errors, 50)) if mass_rel_errors else 0.0,
                "p95": float(np.percentile(mass_rel_errors, 95)) if mass_rel_errors else 0.0,
                "p99": float(np.percentile(mass_rel_errors, 99)) if mass_rel_errors else 0.0,
            },
            "sensitivity_relative_error": {
                "max": float(np.max(sens_rel_errors)) if sens_rel_errors else 0.0,
                "p50": float(np.percentile(sens_rel_errors, 50)) if sens_rel_errors else 0.0,
            },
            "ignition_delay_difference": {
                "max_abs_s": float(np.max(ign_delay_abs_errors)) if ign_delay_abs_errors else 0.0,
                "max_rel": float(np.max(ign_delay_rel_errors)) if ign_delay_rel_errors else 0.0,
            }
        },
        "case_comparisons": case_comparisons,
        "failures": failures,
        "status": "PASS" if passed_cases == total_cases else "FAIL",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"REACTORNET CAMPAIGN SUMMARY: {summary['status']}")
    print(f"Passed: {passed_cases}/{total_cases} ({summary['pass_rate_percent']:.1f}%)")
    print(f"Max T rel error:    {summary['metrics']['temperature_relative_error']['max']:.4e}")
    print(f"Max P rel error:    {summary['metrics']['pressure_relative_error']['max']:.4e}")
    print(f"Max Mass rel error: {summary['metrics']['mass_relative_error']['max']:.4e}")
    if ign_delay_abs_errors:
        print(f"Max Ignition delay diff: {summary['metrics']['ignition_delay_difference']['max_abs_s']:.4e} s (rel {summary['metrics']['ignition_delay_difference']['max_rel']:.2%})")
    print(f"Summary written to: {summary_path}")
    print("=" * 60)

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)

if __name__ == "__main__":
    main()
