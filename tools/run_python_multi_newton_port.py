"""Run Python MultiNewton and MultiJac port across the 20-case differential corpus.

Generates audit/python_multi_newton_port_results.json.
"""

import json
import math
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cantera_python.oned import MultiNewton, MultiJac
from cantera_python.numerics import SteadyStateSystem
from cantera_python.constants import CanteraError


class PyTestSteadySystem:
    def __init__(self, tc: dict):
        self.tc = tc
        self.m_size = int(tc["size"])
        self.m_bw = int(tc["bandwidth"])
        self.m_rdt = float(tc["rdt"])
        self.m_mask = list(tc.get("mask", [1] * self.m_size))
        if len(self.m_mask) < self.m_size:
            self.m_mask = [1] * self.m_size

        self.m_jac = MultiJac()
        self.m_jac.initialize(self.m_size)
        self.m_jac.setBandwidth(self.m_bw)

        self.m_current_x = np.array(tc.get("x0", [0.0] * self.m_size), dtype=float, copy=True)
        if len(self.m_current_x) < self.m_size:
            self.m_current_x = np.zeros(self.m_size, dtype=float)

    def size(self) -> int:
        return self.m_size

    def rdt(self) -> float:
        return self.m_rdt

    def linearSolver(self):
        return self.m_jac

    def factorizeJacobian(self):
        self.m_jac.updateTransient(self.m_rdt, self.m_mask)

    def eval(self, x, r, rdt=-1.0, count=1):
        n = self.m_size
        st = self.tc["system_type"]
        params = self.tc.get("params", {})
        diff = float(params.get("diff", 1.0))
        k = float(params.get("k", 1.0))
        s = params.get("s", [1.0] * n)

        if st == "linear":
            for i in range(n):
                xl = x[i - 1] if i > 0 else 0.0
                xr = x[i + 1] if i + 1 < n else 0.0
                d2x = xr - 2.0 * x[i] + xl
                r[i] = -diff * d2x + k * x[i] - s[i]
        elif st == "quadratic":
            for i in range(n):
                xl = x[i - 1] if i > 0 else 0.0
                xr = x[i + 1] if i + 1 < n else 0.0
                d2x = xr - 2.0 * x[i] + xl
                r[i] = -diff * d2x + k * x[i] * x[i] - s[i]
        elif st == "arrhenius":
            A = float(params.get("A", 1.0))
            Ta = float(params.get("Ta", 1.0))
            for i in range(n):
                xl = x[i - 1] if i > 0 else 0.0
                xr = x[i + 1] if i + 1 < n else 0.0
                d2x = xr - 2.0 * x[i] + xl
                T = max(float(x[i]), 1e-4)
                exp_t = math.exp(-Ta / T)
                r[i] = -diff * d2x + A * T * exp_t - s[i]
        elif st == "singular":
            zero_row = int(params.get("zero_row", -1))
            for i in range(n):
                if i == zero_row:
                    r[i] = 0.0
                else:
                    xl = x[i - 1] if i > 0 else 0.0
                    xr = x[i + 1] if i + 1 < n else 0.0
                    d2x = xr - 2.0 * x[i] + xl
                    r[i] = -diff * d2x + k * x[i] - 1.0
        elif st == "coupled3":
            M = n // 3
            inter_coupling = float(params.get("inter_coupling", 0.0))
            for j in range(M):
                iu = 3 * j
                iv = 3 * j + 1
                it = 3 * j + 2

                U = x[iu]
                V = x[iv]
                T = x[it]

                Ul = x[3 * (j - 1)] if j > 0 else 0.0
                Ur = x[3 * (j + 1)] if j + 1 < M else 0.0
                d2U = Ur - 2.0 * U + Ul
                r[iu] = -diff * d2U + 0.1 * V * U - s[iu]

                Vl = x[3 * (j - 1) + 1] if j > 0 else 0.0
                Vr = x[3 * (j + 1) + 1] if j + 1 < M else 0.0
                d2V = Vr - 2.0 * V + Vl
                r[iv] = -diff * d2V + 0.1 * V * V - s[iv]

                Tl = x[3 * (j - 1) + 2] if j > 0 else 300.0
                Tr = x[3 * (j + 1) + 2] if j + 1 < M else 1200.0
                d2T = Tr - 2.0 * T + Tl
                r[it] = -diff * d2T + inter_coupling * 0.001 * U * T - s[it]

    def evalJacobian(self, x0):
        self.m_current_x[:] = np.asarray(x0, dtype=float)
        self.m_jac.reset()
        n = self.m_size
        st = self.tc["system_type"]
        params = self.tc.get("params", {})
        diff = float(params.get("diff", 1.0))
        k = float(params.get("k", 1.0))

        if st == "linear":
            for i in range(n):
                self.m_jac.setValue(i, i, 2.0 * diff + k)
                if i > 0:
                    self.m_jac.setValue(i, i - 1, -diff)
                if i + 1 < n:
                    self.m_jac.setValue(i, i + 1, -diff)
        elif st == "quadratic":
            for i in range(n):
                self.m_jac.setValue(i, i, 2.0 * diff + 2.0 * k * x0[i])
                if i > 0:
                    self.m_jac.setValue(i, i - 1, -diff)
                if i + 1 < n:
                    self.m_jac.setValue(i, i + 1, -diff)
        elif st == "arrhenius":
            A = float(params.get("A", 1.0))
            Ta = float(params.get("Ta", 1.0))
            for i in range(n):
                T0 = max(float(x0[i]), 1e-4)
                exp_t0 = math.exp(-Ta / T0)
                d_reac = A * (1.0 + Ta / T0) * exp_t0
                self.m_jac.setValue(i, i, 2.0 * diff + d_reac)
                if i > 0:
                    self.m_jac.setValue(i, i - 1, -diff)
                if i + 1 < n:
                    self.m_jac.setValue(i, i + 1, -diff)
        elif st == "singular":
            zero_row = int(params.get("zero_row", -1))
            for i in range(n):
                if i == zero_row:
                    self.m_jac.setValue(i, i, 0.0)
                else:
                    self.m_jac.setValue(i, i, 2.0 * diff + k)
                    if i > 0:
                        self.m_jac.setValue(i, i - 1, -diff)
                    if i + 1 < n:
                        self.m_jac.setValue(i, i + 1, -diff)
        elif st == "coupled3":
            M = n // 3
            inter_coupling = float(params.get("inter_coupling", 0.0))
            for j in range(M):
                iu = 3 * j
                iv = 3 * j + 1
                it = 3 * j + 2
                U0 = x0[iu]
                V0 = x0[iv]
                T0 = x0[it]

                self.m_jac.setValue(iu, iu, 2.0 * diff + 0.1 * V0)
                self.m_jac.setValue(iu, iv, 0.1 * U0)

                self.m_jac.setValue(iv, iv, 2.0 * diff + 0.2 * V0)

                self.m_jac.setValue(it, it, 2.0 * diff + inter_coupling * 0.001 * U0)
                self.m_jac.setValue(it, iu, inter_coupling * 0.001 * T0)

                if j > 0:
                    self.m_jac.setValue(iu, 3 * (j - 1), -diff)
                    self.m_jac.setValue(iv, 3 * (j - 1) + 1, -diff)
                    self.m_jac.setValue(it, 3 * (j - 1) + 2, -diff)
                if j + 1 < M:
                    self.m_jac.setValue(iu, 3 * (j + 1), -diff)
                    self.m_jac.setValue(iv, 3 * (j + 1) + 1, -diff)
                    self.m_jac.setValue(it, 3 * (j + 1) + 2, -diff)

    def weightedNorm(self, step) -> float:
        total = 0.0
        atol = self.tc.get("atol", [1e-4] * self.m_size)
        rtol = self.tc.get("rtol", [1e-3] * self.m_size)
        for i in range(self.m_size):
            ewt = rtol[i] * abs(self.m_current_x[i]) + atol[i]
            val = step[i] / ewt
            total += val * val
        return float(math.sqrt(total / self.m_size))

    def upperBound(self, i: int) -> float:
        return float(self.tc["upper_bounds"][i])

    def lowerBound(self, i: int) -> float:
        return float(self.tc["lower_bounds"][i])

    def componentName(self, i: int) -> str:
        return f"var_{i}"


def main():
    audit_dir = REPO_ROOT / "audit"
    corpus_file = audit_dir / "corpus_multi_newton.jsonl"
    out_file = audit_dir / "python_multi_newton_port_results.json"

    results = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tc = json.loads(line)
            case_id = tc["case_id"]
            mode = tc["mode"]
            size = int(tc["size"])

            sys_obj = PyTestSteadySystem(tc)
            newt = MultiNewton(size)
            newt.resize(size)
            newt.setOptions(tc.get("max_jac_age", 5))

            entry = {"case_id": case_id, "mode": mode}

            if mode == "solve":
                x0 = np.array(tc.get("x0", [0.0] * size), dtype=float, copy=True)
                x1 = np.zeros(size, dtype=float)
                status = -999
                iters = 0

                sys_obj.evalJacobian(x0)
                try:
                    sys_obj.factorizeJacobian()
                except Exception as err:
                    if tc.get("rdt", 0.0) == 0.0:
                        status = -4

                if status != -4:
                    try:
                        status = newt.solve(x0, x1, sys_obj, 0)
                        iters = newt.lastIterations()
                    except Exception:
                        status = -99

                entry["status"] = status
                entry["last_iterations"] = iters
                entry["x1"] = list(x1)
            elif mode == "step":
                x0 = np.array(tc.get("x0", [0.0] * size), dtype=float, copy=True)
                step_vec = np.zeros(size, dtype=float)
                status = 0

                sys_obj.evalJacobian(x0)
                sys_obj.factorizeJacobian()

                try:
                    newt.step(x0, step_vec, sys_obj, 0)
                except Exception:
                    status = -1

                entry["status"] = status
                entry["step"] = list(step_vec)
            elif mode == "boundStep":
                x0 = np.array(tc.get("x0", [0.0] * size), dtype=float, copy=True)
                step0 = np.array(tc.get("step0", [0.0] * size), dtype=float, copy=True)
                fbound = newt.boundStep(x0, step0, sys_obj, 0)
                entry["fbound"] = fbound
            elif mode == "dampStep":
                x0 = np.array(tc.get("x0", [0.0] * size), dtype=float, copy=True)
                step0 = np.array(tc.get("step0", [0.0] * size), dtype=float, copy=True)
                x1 = np.zeros(size, dtype=float)
                step1 = np.zeros(size, dtype=float)
                s1_box = [1e30]

                sys_obj.evalJacobian(x0)
                sys_obj.factorizeJacobian()

                status = newt.dampStep(x0, step0, x1, step1, s1_box, sys_obj, 0, False)
                entry["status"] = status
                entry["s1"] = s1_box[0]
                entry["x1"] = list(x1)
            elif mode == "multijac_ops":
                jac = MultiJac()
                jac.initialize(size)
                jac.setBandwidth(int(tc["bandwidth"]))

                jac.setValue(0, 0, 4.0)
                jac.setValue(0, 1, -1.0)
                jac.setValue(1, 0, -1.0)
                jac.setValue(1, 1, 5.0)
                jac.setValue(1, 2, -2.0)
                jac.setValue(2, 1, -2.0)
                jac.setValue(2, 2, 6.0)
                jac.setValue(2, 3, -1.0)
                jac.setValue(3, 2, -1.0)
                jac.setValue(3, 3, 4.0)

                v11 = jac.value(1, 1)
                v12 = jac.value(1, 2)

                mask = [1, 1, 1, 0]
                jac.updateTransient(10.0, mask)

                rhs = np.array([2.0, 3.0, 1.0, 5.0], dtype=float)
                sol = np.zeros(size, dtype=float)
                jac.solve(rhs, sol)

                entry["type"] = jac.type()
                entry["value_1_1"] = v11
                entry["value_1_2"] = v12
                entry["info"] = jac.info()
                entry["sol"] = list(sol)

            results.append(entry)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"Successfully ran Python MultiNewton port across {len(results)} cases.")


if __name__ == "__main__":
    main()
