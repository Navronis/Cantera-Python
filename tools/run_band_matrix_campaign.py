"""Execute BandMatrix differential validation campaign comparing Python port against native Cantera C++ oracle."""

from datetime import datetime, timezone
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
audit_dir = repo_root / "audit"
tools_dir = repo_root / "tools"


def to_wsl_path(win_path: Path) -> str:
    path_str = str(win_path.resolve()).replace("\\", "/")
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        return f"/mnt/{drive}{path_str[2:]}"
    return path_str


def main():
    corpus_path = audit_dir / "corpus_band_matrix.jsonl"
    native_out = audit_dir / "native_band_matrix_oracle_results.json"
    python_out = audit_dir / "python_band_matrix_port_results.json"
    native_bin = tools_dir / "run_native_band_matrix_oracle"

    run_id = f"band_matrix_diff_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # 1. Run native C++ oracle in WSL
    wsl_bin = to_wsl_path(native_bin)
    wsl_corpus = to_wsl_path(corpus_path)
    wsl_out = to_wsl_path(native_out)
    native_cmd = ["wsl", "-d", "Ubuntu", wsl_bin, wsl_corpus, wsl_out, run_id]
    print(f"Executing native oracle: {' '.join(native_cmd)}")
    subprocess.run(native_cmd, check=True)

    # 2. Run Python port runner
    print("Executing Python port runner...")
    sys.path.insert(0, str(tools_dir))
    import run_python_band_matrix_port
    run_python_band_matrix_port.run_cases(corpus_path, python_out, run_id)

    # 3. Load results
    with open(native_out, "r", encoding="utf-8") as f:
        native_results = {r["case_id"]: r for r in json.load(f)}
    with open(python_out, "r", encoding="utf-8") as f:
        python_results = {r["case_id"]: r for r in json.load(f)}

    # Also load corpus to reconstruct dense matrices for residual evaluation
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_cases = {c["case_id"]: c for c in [json.loads(l) for l in f if l.strip()]}

    total_cases = len(native_results)
    passed_cases = 0
    case_comparisons = []

    overall_max_sol_rel_err = 0.0
    overall_max_sol_abs_err = 0.0
    overall_max_residual = 0.0
    overall_max_mult_err = 0.0
    overall_max_rcond_err = 0.0

    for case_id, nat in native_results.items():
        py = python_results.get(case_id)
        if not py:
            print(f"FAILED: missing Python result for {case_id}")
            continue

        c_spec = corpus_cases[case_id]
        n = c_spec["n"]
        kl = c_spec["kl"]
        ku = c_spec["ku"]
        nrhs = c_spec["nrhs"]

        # Check dimensions
        assert nat["nRows"] == py["nRows"]
        assert nat["nColumns"] == py["nColumns"]
        assert nat["nSubDiagonals"] == py["nSubDiagonals"]
        assert nat["nSuperDiagonals"] == py["nSuperDiagonals"]
        assert nat["ldim"] == py["ldim"]

        status_match = (nat["status"] == py["status"])

        sol_rel_err = 0.0
        sol_abs_err = 0.0
        eqn_residual = 0.0
        mult_rel_err = 0.0
        left_mult_rel_err = 0.0
        rcond_rel_err = 0.0

        if n == 0:
            case_passed = status_match
        elif nat["status"] == "SINGULAR":
            case_passed = status_match
        else:
            # Check mult
            nat_mult = np.asarray(nat["mult_result"], dtype=float)
            py_mult = np.asarray(py["mult_result"], dtype=float)
            mult_abs = np.max(np.abs(nat_mult - py_mult))
            mult_denom = max(1.0, float(np.max(np.abs(nat_mult))))
            mult_rel_err = float(mult_abs / mult_denom)

            # Check leftMult
            nat_lmult = np.asarray(nat["leftMult_result"], dtype=float)
            py_lmult = np.asarray(py["leftMult_result"], dtype=float)
            lmult_abs = np.max(np.abs(nat_lmult - py_lmult))
            lmult_denom = max(1.0, float(np.max(np.abs(nat_lmult))))
            left_mult_rel_err = float(lmult_abs / lmult_denom)

            # Check checkRows and checkColumns
            assert nat["checkRows_iSmall"] == py["checkRows_iSmall"], f"checkRows row mismatch: {nat['checkRows_iSmall']} vs {py['checkRows_iSmall']}"
            assert nat["checkColumns_jSmall"] == py["checkColumns_jSmall"], f"checkColumns col mismatch: {nat['checkColumns_jSmall']} vs {py['checkColumns_jSmall']}"

            # Check rcond
            nat_rc = nat.get("rcond", 0.0)
            py_rc = py.get("rcond", 0.0)
            if nat_rc > 0.0:
                rcond_rel_err = abs(nat_rc - py_rc) / nat_rc

            # Check solution
            nat_sol = np.asarray(nat["solution"], dtype=float)
            py_sol = np.asarray(py["solution"], dtype=float)
            sol_abs_err = float(np.max(np.abs(nat_sol - py_sol)))
            denom = max(1.0, float(np.max(np.abs(nat_sol))))
            sol_rel_err = float(sol_abs_err / denom)

            # Reconstruct dense A and compute residual ||A*x - b||
            dense_A = np.zeros((n, n), dtype=float)
            for (i, j, v) in c_spec["entries"]:
                dense_A[i, j] = v

            rhs_flat = np.asarray([c_spec["rhs"][row][col] for col in range(nrhs) for row in range(n)], dtype=float)
            py_sol_mat = py_sol.reshape(nrhs, n).T
            rhs_mat = rhs_flat.reshape(nrhs, n).T
            res_mat = dense_A @ py_sol_mat - rhs_mat
            eqn_residual = float(np.max(np.abs(res_mat)))

            case_passed = (
                status_match and
                sol_rel_err < 1e-11 and
                mult_rel_err < 1e-12 and
                left_mult_rel_err < 1e-12 and
                rcond_rel_err < 1e-10
            )

        if case_passed:
            passed_cases += 1

        overall_max_sol_rel_err = max(overall_max_sol_rel_err, sol_rel_err)
        overall_max_sol_abs_err = max(overall_max_sol_abs_err, sol_abs_err)
        overall_max_residual = max(overall_max_residual, eqn_residual)
        overall_max_mult_err = max(overall_max_mult_err, max(mult_rel_err, left_mult_rel_err))
        overall_max_rcond_err = max(overall_max_rcond_err, rcond_rel_err)

        case_comparisons.append({
            "case_id": case_id,
            "category": nat["category"],
            "passed": case_passed,
            "sol_rel_error": sol_rel_err,
            "sol_abs_error": sol_abs_err,
            "equation_residual": eqn_residual,
            "mult_rel_error": mult_rel_err,
            "left_mult_rel_error": left_mult_rel_err,
            "rcond_rel_error": rcond_rel_err,
        })

        status_str = "PASS" if case_passed else "FAIL"
        print(f"[{status_str}] {case_id:<35} | sol_rel_err: {sol_rel_err:.2e} | res: {eqn_residual:.2e} | rcond_err: {rcond_rel_err:.2e}")

    pass_rate = 100.0 * passed_cases / total_cases if total_cases > 0 else 0.0
    print(f"\n=======================================================")
    print(f"BandMatrix Differential Campaign: {passed_cases}/{total_cases} passed ({pass_rate:.1f}%)")
    print(f"Overall Max Solution Relative Error: {overall_max_sol_rel_err:.3e}")
    print(f"Overall Max Solution Absolute Error: {overall_max_sol_abs_err:.3e}")
    print(f"Overall Max Equation Residual:        {overall_max_residual:.3e}")
    print(f"Overall Max Matrix-Vector Mult Error: {overall_max_mult_err:.3e}")
    print(f"Overall Max rcond Relative Error:     {overall_max_rcond_err:.3e}")
    print(f"=======================================================\n")

    summary = {
        "schema_version": 1,
        "solver_family": "band_matrix",
        "subsystem": "banded linear algebra",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_pct": pass_rate,
        "max_solution_relative_error": overall_max_sol_rel_err,
        "max_solution_absolute_error": overall_max_sol_abs_err,
        "max_equation_residual": overall_max_residual,
        "max_mult_relative_error": overall_max_mult_err,
        "max_rcond_relative_error": overall_max_rcond_err,
        "numerical_parity": (passed_cases == total_cases),
        "case_comparisons": case_comparisons,
    }

    summary_file = audit_dir / "band_matrix_differential_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_file}")


if __name__ == "__main__":
    main()
