"""Python port runner for IdasIntegrator DAE differential verification."""

import json
import math
import sys
from pathlib import Path
import numpy as np

# Ensure cantera_python is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from cantera_python.numerics import IdasIntegrator, FuncEval


class PolynomialDAE(FuncEval):
    def __init__(self):
        super().__init__()
        self._y = np.array([1.0, 1.0], dtype=float)
        self._ydot = np.array([0.0, 0.0], dtype=float)

    def neq(self) -> int:
        return 2

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        y[:] = self._y
        ydot[:] = self._ydot

    def getConstraints(self, c: np.ndarray) -> None:
        c[0] = 1.0
        c[1] = 0.0

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray) -> None:
        res[0] = ydot[0] - (-y[0] + y[1])
        res[1] = y[0] ** 2 - y[1]

    def setState(self, y0: float, y1: float, yd0: float, yd1: float) -> None:
        self._y[0] = y0
        self._y[1] = y1
        self._ydot[0] = yd0
        self._ydot[1] = yd1


class TimeDepDAE(FuncEval):
    def __init__(self):
        super().__init__()
        self._y = np.array([1.0, 1.0], dtype=float)
        self._ydot = np.array([-1.0, 0.0], dtype=float)

    def neq(self) -> int:
        return 2

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        y[:] = self._y
        ydot[:] = self._ydot

    def getConstraints(self, c: np.ndarray) -> None:
        c[0] = 1.0
        c[1] = 0.0

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray) -> None:
        res[0] = ydot[0] - (-y[0] + math.sin(t))
        res[1] = y[1] - math.cos(t)


class RobertsonDAE(FuncEval):
    def __init__(self):
        super().__init__()
        self._y = np.array([1.0, 0.0, 0.0], dtype=float)
        self._ydot = np.array([-0.04, 0.04, 0.0], dtype=float)

    def neq(self) -> int:
        return 3

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        y[:] = self._y
        ydot[:] = self._ydot

    def getConstraints(self, c: np.ndarray) -> None:
        c[0] = 1.0
        c[1] = 1.0
        c[2] = 0.0

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray) -> None:
        res[0] = ydot[0] - (-0.04 * y[0] + 1e4 * y[1] * y[2])
        res[1] = ydot[1] - (0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2)
        res[2] = y[0] + y[1] + y[2] - 1.0


class Coupled5DAE(FuncEval):
    def __init__(self):
        super().__init__()
        self._y = np.array([1.0, 0.5, 0.2, 2.25, math.exp(-0.2)], dtype=float)
        self._ydot = np.array([1.75, math.exp(-0.2) - 0.5, 1.1, 0.0, 0.0], dtype=float)

    def neq(self) -> int:
        return 5

    def getStateDae(self, y: np.ndarray, ydot: np.ndarray) -> None:
        y[:] = self._y
        ydot[:] = self._ydot

    def getConstraints(self, c: np.ndarray) -> None:
        c[0] = 1.0
        c[1] = 1.0
        c[2] = 1.0
        c[3] = 0.0
        c[4] = 0.0

    def evalDae(self, t: float, y: np.ndarray, ydot: np.ndarray, res: np.ndarray) -> None:
        res[0] = ydot[0] - (-0.5 * y[0] + y[3])
        res[1] = ydot[1] - (-y[1] + y[4])
        res[2] = ydot[2] - (y[0] + y[1] - 2.0 * y[2])
        res[3] = y[3] - (y[0] + y[1]) ** 2
        res[4] = y[4] - math.exp(-y[2])


def run_cases(corpus_path: Path, output_path: Path, run_id: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(corpus_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            case_def = json.loads(line)
            case_id = case_def["case_id"]
            sys_type = case_def["system_type"]
            mode = case_def["mode"]
            rtol = case_def["rtol"]
            atol = case_def["atol"]
            max_order = case_def.get("max_order", 5)
            incl_alg = case_def.get("include_algebraic_in_error_test", True)
            sample_times = case_def["sample_times"]

            if sys_type in ("polynomial", "reinit_poly"):
                sys_obj = PolynomialDAE()
            elif sys_type == "timedep":
                sys_obj = TimeDepDAE()
            elif sys_type == "robertson":
                sys_obj = RobertsonDAE()
            elif sys_type == "coupled5":
                sys_obj = Coupled5DAE()
            else:
                raise ValueError(f"Unknown system type: {sys_type}")

            idas = IdasIntegrator()
            idas.setTolerances(rtol, atol)
            idas.setMaxOrder(max_order)
            idas.includeAlgebraicInErrorTest(incl_alg)
            idas.initialize(0.0, sys_obj)

            checkpoints = []
            for t_target in sample_times:
                if sys_type == "reinit_poly" and abs(t_target - 0.15) < 1e-9:
                    sys_obj.setState(0.5, 0.25, 0.0, 0.0)
                    idas.reinitialize(0.1, sys_obj)

                if mode == "step":
                    actual_time = idas.step(t_target)
                else:
                    actual_time = idas.integrate(t_target)

                sol = idas.solution()
                res = np.empty_like(sol)
                sys_obj.evalDae(actual_time, sol, np.zeros_like(sol), res)
                checkpoints.append({
                    "time": actual_time,
                    "y": sol.tolist(),
                    "algebraic_residual": float(res[-1]),
                })

            stats = idas.solverStats()
            result_row = {
                "case_id": case_id,
                "oracle": "python_port_idas",
                "run_id": run_id,
                "status": "PASS",
                "checkpoints": checkpoints,
                "metrics": {
                    "steps": stats["steps"],
                    "res_evals": stats["res_evals"],
                    "nonlinear_iters": stats["nonlinear_iters"],
                    "last_order": stats["last_order"],
                },
            }
            fout.write(json.dumps(result_row) + "\n")
    print("Python port IDAS execution completed successfully.")


if __name__ == "__main__":
    c_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audit/corpus_idas_integrator.jsonl")
    o_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("audit/idas_integrator_python_port_results.jsonl")
    rid = sys.argv[3] if len(sys.argv) > 3 else "test_run_01"
    run_cases(c_path, o_path, rid)
