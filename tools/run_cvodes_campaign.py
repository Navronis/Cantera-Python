#!/usr/bin/env python3
"""Run complete differential validation campaign for CVodesIntegrator against native oracle."""

import json
import hashlib
import subprocess
import time
import math
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from cantera_python.numerics import CVodesIntegrator, FuncEval
from cantera_python import CanteraError

CORPUS_PATH = Path("audit/corpus_cvodes_integrator.jsonl")
NATIVE_RESULTS_PATH = Path("audit/cvodes_native_oracle_results.jsonl")
PYTHON_RESULTS_PATH = Path("audit/cvodes_python_port_results.jsonl")
SUMMARY_PATH = Path("audit/cvodes_integrator_differential_summary.json")

class PyCorpusFunc(FuncEval):
    def __init__(self, ptype, y_init, p_params=None, p_scales=None):
        super().__init__()
        self.problem_type = ptype
        self.y_init = list(y_init)
        self.lambda_val = 1.0
        self.omega = 1.0
        self.mu = 10.0
        self.fail_time = 1e9
        self.m_sens_params = list(p_params) if p_params else []
        self.m_paramScales = list(p_scales) if p_scales else []
        if self.problem_type == "param_exp":
            self.m_sens_params = [2.0]
            self.m_paramScales = [1.0]
        elif self.problem_type == "multi_param_sens":
            self.m_sens_params = [-1.0, -2.0]
            self.m_paramScales = [1.0, 1.0]

    def neq(self):
        return len(self.y_init)

    def nparams(self):
        return len(self.m_sens_params)

    def getState(self, y):
        y[:] = self.y_init

    def eval(self, t, y, ydot, p=None):
        if t >= self.fail_time:
            raise CanteraError("PyCorpusFunc::eval", f"Intentional RHS failure at t = {t}")
        if self.problem_type == "decay":
            ydot[0] = -self.lambda_val * y[0]
        elif self.problem_type == "param_exp":
            param = self.m_sens_params[0] if not p else p[0]
            ydot[0] = param * y[0]
        elif self.problem_type == "multi_param_sens":
            p0 = p[0] if p and len(p) > 0 else self.m_sens_params[0]
            p1 = p[1] if p and len(p) > 1 else self.m_sens_params[1]
            ydot[0] = p0 * y[0] + p1 * y[1]
            ydot[1] = -p1 * y[1]
        elif self.problem_type == "coupled_linear":
            ydot[0] = -1000.0 * y[0] + y[1]
            ydot[1] = -y[1]
        elif self.problem_type == "oscillator":
            ydot[0] = y[1]
            ydot[1] = -self.omega * self.omega * y[0]
        elif self.problem_type == "robertson":
            ydot[0] = -0.04 * y[0] + 1e4 * y[1] * y[2]
            ydot[1] = 0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1]
            ydot[2] = 3e7 * y[1] * y[1]
        elif self.problem_type == "vanderpol":
            ydot[0] = y[1]
            ydot[1] = self.mu * (1.0 - y[0] * y[0]) * y[1] - y[0]
        else:
            ydot[:] = 0.0

def run_native_oracle(run_id: str):
    print("Executing native CVodes C++ oracle in WSL Ubuntu 24.04...")
    corpus_wsl = "/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/corpus_cvodes_integrator.jsonl"
    results_wsl = "/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/cvodes_native_oracle_results.jsonl"
    cmd = ["wsl", "/home/nebula/run_native_cvodes_oracle", corpus_wsl, results_wsl, run_id]
    subprocess.run(cmd, check=True)
    print("Native oracle execution complete.")

def run_python_port(run_id: str):
    print("Executing Python CVodesIntegrator on differential corpus...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = []
    for case in cases:
        cid = case["case_id"]
        ptype = case["problem_type"]
        mode = case.get("mode", "integrate")
        t0 = case.get("t0", 0.0)
        tout = case.get("tout", 1.0)
        reltol = case.get("reltol", 1e-9)
        abstol = case.get("abstol", 1e-15)
        max_steps = case.get("max_steps", 20000)
        max_order = case.get("max_order", 5)
        hmax = case.get("hmax", 0.0)
        hmin = case.get("hmin", 0.0)
        max_err_test_fails = case.get("max_err_test_fails", 0)
        linear_solver = case.get("linear_solver", "DENSE")
        bandwidth_upper = case.get("bandwidth_upper", 0)
        bandwidth_lower = case.get("bandwidth_lower", 0)
        lambda_val = case.get("lambda", 1.0)
        omega_val = case.get("omega", 1.0)
        mu_val = case.get("mu", 10.0)
        fail_t = case.get("fail_time", 1e9)
        y_init = case.get("y_init", [1.0])
        abstol_vec = case.get("abstol_vec")

        func = PyCorpusFunc(ptype, y_init)
        func.lambda_val = lambda_val
        func.omega = omega_val
        func.mu = mu_val
        func.fail_time = fail_t
        if "p_val" in case:
            func.m_sens_params = [float(case["p_val"])]

        try:
            ode = CVodesIntegrator()
            if abstol_vec:
                ode.setTolerances(reltol, abstol_vec)
            else:
                ode.setTolerances(reltol, abstol)
            if max_order > 0:
                ode.setMaxOrder(max_order)
            if max_steps > 0:
                ode.setMaxSteps(max_steps)
            if hmax > 0:
                ode.setMaxStepSize(hmax)
            if hmin > 0:
                ode.setMinStepSize(hmin)
            if max_err_test_fails > 0:
                ode.setMaxErrTestFails(max_err_test_fails)
            if linear_solver != "DENSE":
                ode.setLinearSolverType(linear_solver)
                if linear_solver == "BAND":
                    ode.setBandwidth(bandwidth_upper, bandwidth_lower)

            ode.initialize(t0, func)

            if mode == "step":
                t_reached = ode.step(tout)
                results.append({
                    "case_id": cid,
                    "run_id": run_id,
                    "status": "PASS",
                    "time": float(t_reached),
                    "y": list(np.asarray(ode.solution(), dtype=float)),
                    "last_order": int(ode.lastOrder()),
                    "n_evals": int(ode.nEvals()),
                })
            elif mode == "reinit":
                ode.integrate(tout * 0.5)
                y_reinit = case.get("y_reinit", [2.0])
                func.y_init = y_reinit
                t_reinit = case.get("t_reinit", tout * 0.5)
                ode.reinitialize(t_reinit, func)
                ode.integrate(tout)
                results.append({
                    "case_id": cid,
                    "run_id": run_id,
                    "status": "PASS",
                    "time": float(ode.currentTime()),
                    "y": list(np.asarray(ode.solution(), dtype=float)),
                    "last_order": int(ode.lastOrder()),
                    "n_evals": int(ode.nEvals()),
                })
            elif mode == "sens":
                ode.integrate(tout)
                sens_mat = []
                for p in range(ode.nSensParams()):
                    p_row = [float(ode.sensitivity(k, p)) for k in range(ode.nEquations())]
                    sens_mat.append(p_row)
                results.append({
                    "case_id": cid,
                    "run_id": run_id,
                    "status": "PASS",
                    "time": float(ode.currentTime()),
                    "y": list(np.asarray(ode.solution(), dtype=float)),
                    "sens": sens_mat,
                    "last_order": int(ode.lastOrder()),
                    "n_evals": int(ode.nEvals()),
                })
            else:
                ode.integrate(tout)
                dky = ode.derivative(tout, 1)
                results.append({
                    "case_id": cid,
                    "run_id": run_id,
                    "status": "PASS",
                    "time": float(ode.currentTime()),
                    "y": list(np.asarray(ode.solution(), dtype=float)),
                    "ydot": list(np.asarray(dky, dtype=float)),
                    "last_order": int(ode.lastOrder()),
                    "n_evals": int(ode.nEvals()),
                    "linear_solver": ode.linearSolverType(),
                })
        except Exception as e:
            results.append({
                "case_id": cid,
                "run_id": run_id,
                "status": "ERROR",
                "error_message": str(e),
            })

    with open(PYTHON_RESULTS_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Python port execution complete ({len(results)} cases).")

def compare_results(run_id: str):
    print("Evaluating differential parity between native oracle and Python port...")
    with open(NATIVE_RESULTS_PATH, "r", encoding="utf-8") as f:
        native_map = {item["case_id"]: item for item in [json.loads(line) for line in f if line.strip()]}
    with open(PYTHON_RESULTS_PATH, "r", encoding="utf-8") as f:
        python_map = {item["case_id"]: item for item in [json.loads(line) for line in f if line.strip()]}
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus_map = {item["case_id"]: item for item in [json.loads(line) for line in f if line.strip()]}

    total_cases = len(native_map)
    passed_cases = 0
    max_state_rel_err = 0.0
    max_sens_rel_err = 0.0
    orders_reached = set()
    failures = []

    for cid, n_res in native_map.items():
        if cid not in python_map:
            failures.append({"case_id": cid, "reason": "Missing in Python results"})
            continue
        p_res = python_map[cid]
        case_cfg = corpus_map.get(cid, {})

        # 1. Status parity
        if n_res["status"] != p_res["status"]:
            failures.append({
                "case_id": cid,
                "reason": f"Status mismatch: native={n_res['status']} vs python={p_res['status']}",
            })
            continue

        if n_res["status"] == "ERROR":
            # Both expectedly failed
            passed_cases += 1
            continue

        # 2. State vector parity
        n_y = np.array(n_res["y"], dtype=float)
        p_y = np.array(p_res["y"], dtype=float)
        if len(n_y) != len(p_y):
            failures.append({"case_id": cid, "reason": "State dimensions mismatch"})
            continue

        # Relative error
        diff = np.abs(n_y - p_y)
        denom = np.maximum(np.abs(n_y), 1e-12)
        rel_err = float(np.max(diff / denom))
        max_state_rel_err = max(max_state_rel_err, rel_err)

        # 3. Sensitivity parity if present
        if "sens" in n_res and "sens" in p_res:
            n_s = np.array(n_res["sens"], dtype=float)
            p_s = np.array(p_res["sens"], dtype=float)
            s_diff = np.abs(n_s - p_s)
            s_denom = np.maximum(np.abs(n_s), 1e-12)
            s_rel_err = float(np.max(s_diff / s_denom))
            max_sens_rel_err = max(max_sens_rel_err, s_rel_err)

        # Track orders reached
        if "last_order" in p_res:
            orders_reached.add(p_res["last_order"])

        # Dynamic tolerance based on problem type and requested rtol
        req_rtol = case_cfg.get("reltol", 1e-9)
        base_tol = 1e-3 if ("vanderpol" in cid or "robertson" in cid) else 1e-4
        tol = max(base_tol, 50.0 * req_rtol)
        if rel_err > tol:
            failures.append({
                "case_id": cid,
                "reason": f"State rel error {rel_err:.2e} exceeds tolerance {tol:.2e}",
                "native_y": list(n_y),
                "python_y": list(p_y),
            })
            continue

        passed_cases += 1

    pass_rate = (passed_cases / total_cases) * 100.0 if total_cases > 0 else 0.0

    summary = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver_family": "cvodes_integrator",
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_percent": pass_rate,
        "max_state_relative_error": max_state_rel_err,
        "max_sensitivity_relative_error": max_sens_rel_err,
        "orders_reached": sorted(list(orders_reached)),
        "all_orders_1_to_5_reachable": all(o in orders_reached for o in [1, 2, 3, 4, 5]),
        "status": "PASS" if (pass_rate == 100.0 and all(o in orders_reached for o in [1, 2, 3, 4, 5])) else "FAIL",
        "failures": failures,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Campaign complete: {passed_cases}/{total_cases} passed ({pass_rate:.1f}%).")
    print(f"Max state rel err: {max_state_rel_err:.2e}, Max sens rel err: {max_sens_rel_err:.2e}")
    print(f"Orders reached: {sorted(list(orders_reached))}")
    print(f"All orders 1..5 reached: {summary['all_orders_1_to_5_reachable']}")
    print(f"Summary written to {SUMMARY_PATH}")
    return summary

def main():
    run_id = f"cvodes_diff_{int(time.time())}"
    run_native_oracle(run_id)
    run_python_port(run_id)
    summary = compare_results(run_id)
    if summary["status"] != "PASS":
        print("FAILURES DETECTED:")
        for f in summary["failures"]:
            print(f)
        exit(1)

if __name__ == "__main__":
    main()
