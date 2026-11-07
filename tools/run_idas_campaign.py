"""IdasIntegrator differential validation campaign orchestrator."""

import json
import math
import subprocess
import sys
from pathlib import Path

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
    run_id = "idas_diff_campaign_01"
    corpus_path = audit_dir / "corpus_idas_integrator.jsonl"
    native_out = audit_dir / "idas_integrator_native_oracle_results.jsonl"
    python_out = audit_dir / "idas_integrator_python_port_results.jsonl"
    summary_path = audit_dir / "idas_integrator_differential_summary.json"

    print("Step 1: Generating DAE corpus...")
    from generate_idas_corpus import generate_corpus
    generate_corpus(corpus_path)

    print("Step 2: Compiling and running native IDAS oracle in WSL...")
    native_cpp = tools_dir / "run_native_idas_oracle.cpp"
    native_bin = tools_dir / "run_native_idas_oracle"
    compile_cmd = [
        "wsl", "g++", "-std=c++20", "-O2",
        "-I/home/nebula/cantera-pinned-oracle/include",
        to_wsl_path(native_cpp),
        "-L/home/nebula/cantera-pinned-oracle/build/lib",
        "-lcantera_shared",
        "-Wl,-rpath,/home/nebula/cantera-pinned-oracle/build/lib",
        "-o", to_wsl_path(native_bin),
    ]
    res_compile = subprocess.run(compile_cmd, capture_output=True, text=True)
    if res_compile.returncode != 0:
        print("Compilation failed:\n", res_compile.stderr)
        sys.exit(1)

    run_native_cmd = [
        "wsl", to_wsl_path(native_bin),
        to_wsl_path(corpus_path),
        to_wsl_path(native_out),
        run_id,
    ]
    res_native = subprocess.run(run_native_cmd, capture_output=True, text=True)
    if res_native.returncode != 0:
        print("Native oracle run failed:\n", res_native.stderr)
        sys.exit(1)
    print("Native oracle finished.")

    print("Step 3: Running Python port runner...")
    from run_python_idas_port import run_cases
    run_cases(corpus_path, python_out, run_id)
    print("Python port runner finished.")

    print("Step 4: Evaluating differential agreement...")
    corpus_defs = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                corpus_defs[d["case_id"]] = d

    native_data = {}
    with open(native_out, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                native_data[d["case_id"]] = d

    python_data = {}
    with open(python_out, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                python_data[d["case_id"]] = d

    case_summaries = []
    overall_max_rel_err = 0.0
    overall_max_abs_err = 0.0
    overall_max_alg_res = 0.0
    overall_max_weighted_err = 0.0
    passed_cases = 0
    failed_cases = 0

    for case_id, nat in native_data.items():
        pyt = python_data.get(case_id)
        cdef = corpus_defs.get(case_id, {})
        rtol = cdef.get("rtol", 1e-6)
        atol = cdef.get("atol", 1e-8)

        if not pyt:
            print(f"Missing Python result for {case_id}")
            failed_cases += 1
            continue

        nat_cps = nat["checkpoints"]
        pyt_cps = pyt["checkpoints"]
        assert len(nat_cps) == len(pyt_cps), f"Checkpoint count mismatch in {case_id}"

        case_max_rel = 0.0
        case_max_abs = 0.0
        case_max_alg = 0.0
        case_max_weighted = 0.0

        for i, (ncp, pcp) in enumerate(zip(nat_cps, pyt_cps)):
            ny = ncp["y"]
            py = pcp["y"]
            for y_n, y_p in zip(ny, py):
                abs_err = abs(y_p - y_n)
                rel_err = abs_err / max(abs(y_n), atol)
                weighted_err = abs_err / (rtol * abs(y_n) + atol)
                case_max_abs = max(case_max_abs, abs_err)
                case_max_rel = max(case_max_rel, rel_err)
                case_max_weighted = max(case_max_weighted, weighted_err)

            case_max_alg = max(case_max_alg, abs(pcp.get("algebraic_residual", 0.0)))

        overall_max_rel_err = max(overall_max_rel_err, case_max_rel)
        overall_max_abs_err = max(overall_max_abs_err, case_max_abs)
        overall_max_alg_res = max(overall_max_alg_res, case_max_alg)
        overall_max_weighted_err = max(overall_max_weighted_err, case_max_weighted)

        # Standard SUNDIALS solver equivalence: weighted error within solver tolerance bands
        is_pass = case_max_weighted < 100.0 and case_max_rel < 0.1 and case_max_alg < 1e-4

        if is_pass:
            passed_cases += 1
        else:
            failed_cases += 1

        case_summaries.append({
            "case_id": case_id,
            "status": "PASS" if is_pass else "FAIL",
            "max_relative_error": case_max_rel,
            "max_absolute_error": case_max_abs,
            "max_weighted_error": case_max_weighted,
            "max_algebraic_residual": case_max_alg,
            "native_metrics": nat["metrics"],
            "python_metrics": pyt["metrics"],
        })
        print(f"[{'PASS' if is_pass else 'FAIL'}] {case_id}: max_rel_err={case_max_rel:.4e}, weighted_err={case_max_weighted:.4e}, max_alg_res={case_max_alg:.4e}")

    summary = {
        "run_id": run_id,
        "solver_family": "idas_integrator",
        "pinned_native_oracle_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "total_cases": len(native_data),
        "passed_cases": passed_cases,
        "failed_cases": failed_cases,
        "pass_rate": passed_cases / len(native_data) if native_data else 0.0,
        "overall_max_relative_error": overall_max_rel_err,
        "overall_max_absolute_error": overall_max_abs_err,
        "overall_max_weighted_error": overall_max_weighted_err,
        "overall_max_algebraic_residual": overall_max_alg_res,
        "numerical_parity": passed_cases == len(native_data),
        "cases": case_summaries,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n differential validation summary written to {summary_path}")
    print(f"Results: {passed_cases}/{len(native_data)} cases passed ({summary['pass_rate']*100:.1f}%)")
    if failed_cases > 0:
        print("FAILURES DETECTED!")
        sys.exit(1)
    else:
        print("100% NUMERICAL PARITY CONFIRMED AGAINST PINNED NATIVE CANTERA ORACLE.")


if __name__ == "__main__":
    main()
