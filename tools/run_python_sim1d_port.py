"""Run Python Sim1D port runner on corpus_sim1d.jsonl.

Produces audit/python_sim1d_port_results.json.
"""

import json
import sys
from pathlib import Path
from typing import Any
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python.oned import Sim1D, Domain1D
from cantera_python.state import CanteraError


class PyTestSim1DDomain(Domain1D):
    def __init__(self, name: str, dtype: str, nv: int, np_pts: int, bw: int, grid: list[float]):
        super().__init__(nv=nv, points=np_pts, name=name)
        self.m_id = name
        self.name = name
        self.m_domain_type = dtype
        self._names = [f"{name}_var_{i}" for i in range(self.m_nv)]
        self.setBandwidth(bw)
        self.m_min = [-100.0 - 5.0 * i for i in range(self.m_nv)]
        self.m_max = [10000.0 + 10.0 * i for i in range(self.m_nv)]
        self.m_rtol_ss = [1e-4] * self.m_nv
        self.m_atol_ss = [1e-9] * self.m_nv
        if grid:
            self.setupGrid(grid)

    def nComponents(self) -> int:
        return self.m_nv

    def nPoints(self) -> int:
        return self.m_points

    def component_name(self, k: int) -> str:
        return self._names[int(k)]

    def componentName(self, k: int) -> str:
        return self._names[int(k)]

    def index(self, k: int, pt: int) -> int:
        return int(pt) * self.m_nv + int(k)

    def size(self) -> int:
        return self.m_points * self.m_nv

    def initialValue(self, n: int, j: int) -> float:
        return 1.0

    def eval(self, j: int | None, x: np.ndarray, r: np.ndarray, mask: np.ndarray, rdt: float = 0.0) -> None:
        j0 = 0
        j1 = self.nPoints() - 1
        if j is not None and j != -1:
            fp = self.firstPoint()
            lp = self.lastPoint()
            if j < fp or j > lp:
                if j + 1 < fp or (fp > 0 and j > lp + 1):
                    return
            if fp <= j <= lp:
                jloc = j - fp
                j0 = (jloc - 1) if jloc > 0 else 0
                j1 = min(self.nPoints() - 1, jloc + 1)
            elif j + 1 == fp:
                j0 = 0
                j1 = 0
            elif j == lp + 1:
                j0 = self.nPoints() - 1
                j1 = self.nPoints() - 1

        for pt in range(j0, j1 + 1):
            for k in range(self.m_nv):
                idx = self.index(k, pt)
                g_idx = self.loc() + idx
                val = x[g_idx]
                lapl = 0.0
                if 0 < pt < self.nPoints() - 1:
                    v_left = x[self.loc() + self.index(k, pt - 1)]
                    v_right = x[self.loc() + self.index(k, pt + 1)]
                    lapl = v_right - 2.0 * val + v_left
                elif pt == 0 and self.m_left:
                    v_left = x[self.m_left.loc() + self.m_left.index(k % self.m_left.nComponents(), self.m_left.nPoints() - 1)]
                    v_right = x[self.loc() + self.index(k, 1)] if self.nPoints() > 1 else val
                    lapl = v_right - 2.0 * val + v_left
                elif pt == self.nPoints() - 1 and self.m_right:
                    v_left = x[self.loc() + self.index(k, pt - 1)] if self.nPoints() > 1 else val
                    v_right = x[self.m_right.loc() + self.m_right.index(k % self.m_right.nComponents(), 0)]
                    lapl = v_right - 2.0 * val + v_left
                r[g_idx] = -0.5 * lapl + 0.1 * val * val - 0.05 * val
                if rdt > 0.0:
                    r[g_idx] += rdt * val
                    mask[g_idx] = 1


def main():
    corpus_path = repo_root / "audit" / "corpus_sim1d.jsonl"
    out_path = repo_root / "audit" / "python_sim1d_port_results.json"

    results = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["case_id"]

            try:
                doms = []
                for d_info in case.get("domains", []):
                    dom = PyTestSim1DDomain(
                        d_info["name"],
                        d_info.get("type", "flow"),
                        d_info["n_components"],
                        d_info["n_points"],
                        d_info.get("bandwidth", 1),
                        d_info.get("grid", []),
                    )
                    doms.append(dom)

                sim = Sim1D(doms)

                # Populate initial state by domain
                if "state_by_domain" in case:
                    sbd = case["state_by_domain"]
                    for d_idx, vals in enumerate(sbd):
                        comp_count = sim.domain(d_idx).nComponents()
                        pt_count = sim.domain(d_idx).nPoints()
                        for k, v in enumerate(vals):
                            if k < comp_count * pt_count:
                                pt = k // comp_count
                                c = k % comp_count
                                sim._setValue(d_idx, c, pt, float(v))

                # Refine criteria
                if "refine_criteria" in case:
                    for rc in case["refine_criteria"]:
                        d_id = rc.get("domain", 0)
                        ratio = float(rc.get("ratio", 10.0))
                        slope = float(rc.get("slope", 0.8))
                        curve = float(rc.get("curve", 0.8))
                        prune = float(rc.get("prune", -0.1))
                        sim.setRefineCriteria(d_id, ratio, slope, curve, prune)

                # Execute operations
                refine_result = 0
                queried_value = 0.0
                queried_criteria = []
                queried_max_points = 0
                queried_ts_regrid_max = 0
                adjoint_lambda = []
                jac_diag = []

                for op in case.get("operations", []):
                    op_name = op["op"]
                    if op_name == "set_value":
                        d = int(op.get("domain", 0))
                        c = int(op.get("comp", 0))
                        pt = int(op.get("point", 0))
                        val = float(op.get("val", 0.0))
                        sim._setValue(d, c, pt, val)
                    elif op_name == "get_value":
                        d = int(op.get("domain", 0))
                        c = int(op.get("comp", 0))
                        pt = int(op.get("point", 0))
                        queried_value = sim._value(d, c, pt)
                    elif op_name == "refine":
                        llevel = int(op.get("loglevel", 0))
                        refine_result = sim.refine(llevel)
                    elif op_name == "set_criteria":
                        d = int(op.get("domain", -1))
                        ratio = float(op.get("ratio", 10.0))
                        slope = float(op.get("slope", 0.8))
                        curve = float(op.get("curve", 0.8))
                        prune = float(op.get("prune", -0.1))
                        sim.setRefineCriteria(d, ratio, slope, curve, prune)
                    elif op_name == "get_criteria":
                        d = int(op.get("domain", 0))
                        queried_criteria = sim.getRefineCriteria(d)
                    elif op_name == "set_gridmin":
                        d = int(op.get("domain", -1))
                        gmin = float(op.get("gridmin", 1e-5))
                        sim.setGridMin(d, gmin)
                    elif op_name == "set_max_points":
                        d = int(op.get("domain", -1))
                        npts = int(op.get("npoints", 1000))
                        sim.setMaxGridPoints(d, npts)
                    elif op_name == "get_max_points":
                        d = int(op.get("domain", 0))
                        queried_max_points = sim.maxGridPoints(d)
                    elif op_name == "restore_steady":
                        sim.restoreSteadySolution()
                    elif op_name == "restore_timestepping":
                        sim.restoreTimeSteppingSolution()
                    elif op_name == "get_ts_regrid_max":
                        queried_ts_regrid_max = sim.timeStepRegridMax()
                    elif op_name == "set_ts_regrid_max":
                        count = int(op.get("count", 10))
                        sim.setTimeStepRegridMax(count)
                    elif op_name == "eval_ss_jacobian":
                        sim.evalSSJacobian()
                        mjac = sim.linearSolver()
                        if mjac:
                            jac_diag = [float(mjac.value(k, k)) for k in range(sim.size())]
                    elif op_name == "solve_adjoint":
                        b = case.get("adjoint_rhs", [])
                        adjoint_lambda = [0.0] * sim.size()
                        sim.solveAdjoint(b, adjoint_lambda)
                    elif op_name == "set_left_control":
                        temp = float(op.get("temp", 1000.0))
                        sim.setLeftControlPoint(temp)
                    elif op_name == "set_right_control":
                        temp = float(op.get("temp", 1000.0))
                        sim.setRightControlPoint(temp)

                domain_grids = [list(sim.domain(d).grid) for d in range(sim.nDomains())]
                domain_values = []
                for d in range(sim.nDomains()):
                    d_vals = []
                    nv = sim.domain(d).nComponents()
                    np_pts = sim.domain(d).nPoints()
                    for j in range(np_pts):
                        for c in range(nv):
                            d_vals.append(sim._value(d, c, j))
                    domain_values.append(d_vals)

                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": True,
                    "exception_thrown": False,
                    "error_message": "",
                    "n_domains": sim.nDomains(),
                    "points": sim.points(),
                    "size": sim.size(),
                    "bandwidth": sim.bandwidth(),
                    "refine_result": refine_result,
                    "queried_value": queried_value,
                    "queried_max_points": queried_max_points,
                    "queried_ts_regrid_max": queried_ts_regrid_max,
                    "queried_criteria": queried_criteria,
                    "domain_grids": domain_grids,
                    "domain_values": domain_values,
                    "jacobian_diag": jac_diag,
                    "adjoint_lambda": adjoint_lambda,
                })
            except (CanteraError, IndexError) as e:
                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": True,
                    "exception_thrown": True,
                    "error_message": str(e),
                    "n_domains": 0,
                    "points": 0,
                    "size": 0,
                    "bandwidth": 0,
                    "refine_result": 0,
                    "queried_value": 0.0,
                    "queried_max_points": 0,
                    "queried_ts_regrid_max": 0,
                    "queried_criteria": [],
                    "domain_grids": [],
                    "domain_values": [],
                    "jacobian_diag": [],
                    "adjoint_lambda": [],
                })
            except Exception as e:
                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": False,
                    "exception_thrown": True,
                    "error_message": f"{type(e).__name__}: {e}",
                    "n_domains": 0,
                    "points": 0,
                    "size": 0,
                    "bandwidth": 0,
                    "refine_result": 0,
                    "queried_value": 0.0,
                    "queried_max_points": 0,
                    "queried_ts_regrid_max": 0,
                    "queried_criteria": [],
                    "domain_grids": [],
                    "domain_values": [],
                    "jacobian_diag": [],
                    "adjoint_lambda": [],
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, indent=2)

    print(f"Python Sim1D Port finished processing {len(results)} cases into {out_path}")


if __name__ == "__main__":
    main()
