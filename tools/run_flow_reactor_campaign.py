"""End-to-end differential campaign runner for FlowReactor DAE subsystem."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys


def to_wsl_path(win_path: Path) -> str:
    resolved = win_path.resolve()
    drive = resolved.drive.replace(":", "").lower()
    rest = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{rest}"


def rel_err(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = audit_dir / "corpus_flow_reactor.jsonl"
    native_out = audit_dir / "native_flow_reactor_oracle_results.json"
    python_out = audit_dir / "python_flow_reactor_port_results.json"
    summary_out = audit_dir / "flow_reactor_differential_summary.json"

    run_id = f"flow_reactor_{int(datetime.now(timezone.utc).timestamp())}"

    # Step 1: Generate corpus
    print("Step 1: Generating FlowReactor DAE corpus...")
    from generate_flow_reactor_corpus import main as gen_corpus
    gen_corpus()

    # Step 2: Run native oracle in WSL
    print("Step 2: Running native Cantera FlowReactor oracle in WSL...")
    native_runner = repo_root / "tools" / "run_native_flow_reactor_oracle.py"
    wsl_script = to_wsl_path(native_runner)
    wsl_corpus = to_wsl_path(corpus_path)
    wsl_out = to_wsl_path(native_out)

    cmd_native = [
        "wsl", "/home/nebula/cantera-oracle-env/bin/python3",
        wsl_script, wsl_corpus, wsl_out, run_id,
    ]
    res_native = subprocess.run(cmd_native, capture_output=True, text=True)
    if res_native.returncode != 0:
        print("Native oracle execution failed:\n", res_native.stderr)
        sys.exit(1)
    print("Native oracle finished successfully.")

    # Step 3: Run Python port runner
    print("Step 3: Running Python FlowReactor port runner...")
    from run_python_flow_reactor_port import run_cases
    run_cases(corpus_path, python_out, run_id)
    print("Python port runner finished successfully.")

    # Step 4: Evaluate differential agreement
    print("Step 4: Evaluating differential agreement against native Cantera oracle...")
    with open(native_out, "r", encoding="utf-8") as f:
        native_results = {r["case_id"]: r for r in json.load(f)}

    with open(python_out, "r", encoding="utf-8") as f:
        python_results = {r["case_id"]: r for r in json.load(f)}

    failures = []
    total_cases = len(native_results)
    passed_cases = 0

    speed_rel_errors = []
    density_rel_errors = []
    temp_rel_errors = []
    pressure_rel_errors = []
    mass_fraction_max_errors = []
    case_comparisons = []

    tol_speed = 1e-3     # 0.1% max relative error
    tol_density = 1e-3   # 0.1%
    tol_temp = 1e-3      # 0.1%
    tol_pressure = 1e-3  # 0.1%
    tol_Y = 1e-3         # 1e-3 absolute difference for species mass fractions

    for cid, nat in native_results.items():
        pyt = python_results.get(cid)
        if not pyt:
            failures.append({"case_id": cid, "reason": "Missing Python result"})
            continue

        cat = nat["category"]
        case_passed = True
        case_max_speed_err = 0.0
        case_max_density_err = 0.0
        case_max_temp_err = 0.0
        case_max_press_err = 0.0
        case_max_Y_err = 0.0

        if cat == "flow_contracts":
            nat_m = nat.get("metrics", {})
            pyt_m = pyt.get("metrics", {})
            if nat_m.get("type") != pyt_m.get("type"):
                case_passed = False
                failures.append({"case_id": cid, "reason": f"Type mismatch: nat={nat_m.get('type')}, py={pyt_m.get('type')}"})
            if pyt_m.get("is_ode") is not False:
                case_passed = False
                failures.append({"case_id": cid, "reason": "isOde must return False"})
            if pyt_m.get("time_is_independent") is not False:
                case_passed = False
                failures.append({"case_id": cid, "reason": "timeIsIndependent must return False"})
            if not pyt_m.get("threw_get_state"):
                case_passed = False
                failures.append({"case_id": cid, "reason": "getState did not raise NotImplementedError"})
            if not pyt_m.get("threw_eval"):
                case_passed = False
                failures.append({"case_id": cid, "reason": "eval did not raise NotImplementedError"})
            if not pyt_m.get("threw_init_steady"):
                case_passed = False
                failures.append({"case_id": cid, "reason": "initializeSteady did not raise CanteraError"})

        elif cat == "flow_introspection":
            nat_m = nat.get("metrics", {})
            pyt_m = pyt.get("metrics", {})
            nat_idx = nat_m.get("component_indices", {})
            pyt_idx = pyt_m.get("component_indices", {})
            if nat_idx != pyt_idx:
                case_passed = False
                failures.append({"case_id": cid, "reason": f"Component index mismatch: {nat_idx} != {pyt_idx}"})
            nat_names = nat_m.get("component_names", {})
            pyt_names = pyt_m.get("component_names", {})
            if nat_names != pyt_names:
                case_passed = False
                failures.append({"case_id": cid, "reason": f"Component names mismatch: {nat_names} != {pyt_names}"})

        else:
            nat_cps = nat.get("checkpoints", [])
            py_cps = pyt.get("checkpoints", [])

            if len(nat_cps) != len(py_cps):
                case_passed = False
                failures.append({"case_id": cid, "reason": f"Checkpoint count mismatch: nat={len(nat_cps)}, py={len(py_cps)}"})
            else:
                for i, (n_cp, p_cp) in enumerate(zip(nat_cps, py_cps)):
                    # Speed
                    err_u = rel_err(n_cp["speed"], p_cp["speed"])
                    speed_rel_errors.append(err_u)
                    case_max_speed_err = max(case_max_speed_err, err_u)
                    if err_u > tol_speed:
                        case_passed = False
                        failures.append({"case_id": cid, "reason": f"Speed rel error {err_u:.2e} > {tol_speed} at cp {i}"})

                    # Density
                    err_rho = rel_err(n_cp["density"], p_cp["density"])
                    density_rel_errors.append(err_rho)
                    case_max_density_err = max(case_max_density_err, err_rho)
                    if err_rho > tol_density:
                        case_passed = False
                        failures.append({"case_id": cid, "reason": f"Density rel error {err_rho:.2e} > {tol_density} at cp {i}"})

                    # Temperature
                    err_T = rel_err(n_cp["temperature"], p_cp["temperature"])
                    temp_rel_errors.append(err_T)
                    case_max_temp_err = max(case_max_temp_err, err_T)
                    if err_T > tol_temp:
                        case_passed = False
                        failures.append({"case_id": cid, "reason": f"Temperature rel error {err_T:.2e} > {tol_temp} at cp {i}"})

                    # Pressure
                    err_P = rel_err(n_cp["pressure"], p_cp["pressure"])
                    pressure_rel_errors.append(err_P)
                    case_max_press_err = max(case_max_press_err, err_P)
                    if err_P > tol_pressure:
                        case_passed = False
                        failures.append({"case_id": cid, "reason": f"Pressure rel error {err_P:.2e} > {tol_pressure} at cp {i}"})

                    # Mass fractions
                    nY = n_cp["mass_fractions"]
                    pY = p_cp["mass_fractions"]
                    max_dY = max(abs(a - b) for a, b in zip(nY, pY))
                    mass_fraction_max_errors.append(max_dY)
                    case_max_Y_err = max(case_max_Y_err, max_dY)
                    if max_dY > tol_Y:
                        case_passed = False
                        failures.append({"case_id": cid, "reason": f"Species Y max error {max_dY:.2e} > {tol_Y} at cp {i}"})

        if case_passed:
            passed_cases += 1
            print(f"[PASS] {cid}: max_u_err={case_max_speed_err:.2e}, max_rho_err={case_max_density_err:.2e}, max_T_err={case_max_temp_err:.2e}, max_Y_err={case_max_Y_err:.2e}")
        else:
            print(f"[FAIL] {cid}")

        case_comparisons.append({
            "case_id": cid,
            "category": cat,
            "passed": case_passed,
            "max_speed_rel_err": case_max_speed_err,
            "max_density_rel_err": case_max_density_err,
            "max_temp_rel_err": case_max_temp_err,
            "max_pressure_rel_err": case_max_press_err,
            "max_mass_fraction_err": case_max_Y_err,
        })

    summary = {
        "schema_version": 1,
        "solver_family": "flow_reactor",
        "subsystem": "flow reactor DAE",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_pct": (passed_cases / total_cases * 100.0) if total_cases > 0 else 0.0,
        "speed_rel_error_mean": float(sum(speed_rel_errors) / max(len(speed_rel_errors), 1)),
        "speed_rel_error_max": float(max(speed_rel_errors)) if speed_rel_errors else 0.0,
        "density_rel_error_mean": float(sum(density_rel_errors) / max(len(density_rel_errors), 1)),
        "density_rel_error_max": float(max(density_rel_errors)) if density_rel_errors else 0.0,
        "temp_rel_error_mean": float(sum(temp_rel_errors) / max(len(temp_rel_errors), 1)),
        "temp_rel_error_max": float(max(temp_rel_errors)) if temp_rel_errors else 0.0,
        "pressure_rel_error_mean": float(sum(pressure_rel_errors) / max(len(pressure_rel_errors), 1)),
        "pressure_rel_error_max": float(max(pressure_rel_errors)) if pressure_rel_errors else 0.0,
        "mass_fraction_error_max": float(max(mass_fraction_max_errors)) if mass_fraction_max_errors else 0.0,
        "case_comparisons": case_comparisons,
        "failures": failures,
    }

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDifferential validation summary written to {summary_out}")
    print(f"Results: {passed_cases}/{total_cases} cases passed ({summary['pass_rate_pct']:.1f}%)")

    if passed_cases == total_cases:
        print("100% NUMERICAL PARITY CONFIRMED AGAINST PINNED NATIVE CANTERA ORACLE.")
    else:
        print("FAILURES DETECTED:")
        for fail in failures:
            print(f"  - {fail['case_id']}: {fail['reason']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
