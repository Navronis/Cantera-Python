"""Run Python OneDim port runner on corpus_onedim.jsonl.

Produces audit/python_onedim_port_results.json.
"""

import json
import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python.oned import OneDim, Domain1D
from cantera_python.state import CanteraError


class PyTestOneDimDomain(Domain1D):
    def __init__(self, name: str, dtype: str, nv: int, np_pts: int, bw: int):
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

    def setTolerances(self, rtol: list[float], atol: list[float]) -> None:
        for i in range(self.m_nv):
            self.m_rtol_ss[i] = rtol[i % len(rtol)]
            self.m_atol_ss[i] = atol[i % len(atol)]

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
    corpus_path = repo_root / "audit" / "corpus_onedim.jsonl"
    out_path = repo_root / "audit" / "python_onedim_port_results.json"

    results = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            rdt = float(case.get("rdt", 0.0))
            eval_point = case.get("eval_point", -1)
            state = case.get("state", [])
            step = case.get("step", [])
            rtol = case.get("rtol", [1e-4])
            atol = case.get("atol", [1e-8])
            q_dom = case.get("query_domain_index", -1)
            q_comp = case.get("query_component_index", -1)

            try:
                oned = OneDim()
                for d_info in case.get("domains", []):
                    dom = PyTestOneDimDomain(
                        d_info["name"],
                        d_info.get("type", "flow"),
                        d_info["n_components"],
                        d_info["n_points"],
                        d_info.get("bandwidth", 1)
                    )
                    dom.setTolerances(rtol, atol)
                    oned.addDomain(dom)

                oned.init()
                oned.resize()

                x0 = list(state)
                if len(x0) < oned.size():
                    x0.extend([0.0] * (oned.size() - len(x0)))
                oned.setInitialGuess(x0)

                if q_dom >= 0:
                    oned.checkDomainIndex(q_dom)
                if q_comp >= 0:
                    oned.component(q_comp)

                wnorm = oned.weightedNorm(step)

                res = np.zeros(oned.size(), dtype=float)
                oned.eval(eval_point, np.array(x0, dtype=float), res, rdt=rdt, count=1)

                oned.evalJacobian(x0)
                mjac = oned.linearSolver()
                j_diag = []
                if mjac:
                    for k in range(oned.size()):
                        j_diag.append(float(mjac.value(k, k)))

                first_cname = oned.componentName(0)
                first_ub = oned.upperBound(0)
                first_lb = oned.lowerBound(0)

                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": True,
                    "exception_thrown": False,
                    "error_message": "",
                    "n_domains": oned.nDomains(),
                    "points": oned.points(),
                    "size": oned.size(),
                    "bandwidth": oned.bandwidth(),
                    "weighted_norm": wnorm,
                    "residual": res.tolist(),
                    "jacobian_diag": j_diag,
                    "first_component_name": first_cname,
                    "first_upper_bound": first_ub,
                    "first_lower_bound": first_lb,
                })
            except (IndexError, CanteraError) as e:
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
                    "weighted_norm": 0.0,
                    "residual": [],
                    "jacobian_diag": [],
                    "first_component_name": "",
                    "first_upper_bound": 0.0,
                    "first_lower_bound": 0.0,
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
                    "weighted_norm": 0.0,
                    "residual": [],
                    "jacobian_diag": [],
                    "first_component_name": "",
                    "first_upper_bound": 0.0,
                    "first_lower_bound": 0.0,
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, indent=2)

    print(f"Python OneDim Port finished processing {len(results)} cases into {out_path}")


if __name__ == "__main__":
    main()
