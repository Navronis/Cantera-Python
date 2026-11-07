"""Run Python Domain1D port runner against deterministic corpus in audit/corpus_domain1d.jsonl.

Outputs results to:
  - audit/python_domain1d_port_results.json
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cantera_python.oned import Domain1D
from cantera_python.state import CanteraError


class TestDomain1D(Domain1D):
    def initialValue(self, n: int, j: int) -> float:
        return 100.0 * (n + 1) + 1.0 * (j + 1)

    def eval(self, j: int | None, x: np.ndarray, r: np.ndarray, mask: np.ndarray, rdt: float = 0.0) -> None:
        pass


def run_cases():
    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = repo_root / "audit" / "corpus_domain1d.jsonl"
    output_path = repo_root / "audit" / "python_domain1d_port_results.json"

    results = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip() or line.strip().startswith("#"):
                continue
            case_data = json.loads(line)
            case_id = case_data["case_id"]

            out_rec = {
                "case_id": case_id,
                "index": idx,
                "success": True,
                "exception_thrown": False,
                "error_message": "",
            }

            try:
                nv = int(case_data.get("nv", 1))
                pts = int(case_data.get("points", 1))

                domain = TestDomain1D(nv, pts)
                shared_data = [0.0] * domain.size()
                domain.setData(shared_data)

                ops = case_data.get("operations", [])
                for op in ops:
                    op_name = op["op"]

                    if op_name == "query_basic":
                        out_rec["points"] = domain.nPoints()
                        out_rec["size"] = domain.size()
                        out_rec["bandwidth"] = domain.bandwidth()
                        out_rec["zmin"] = domain.zmin()
                        out_rec["zmax"] = domain.zmax()
                        out_rec["domain_type"] = domain.domainType()

                    elif op_name == "resize":
                        r_nv = int(op["nv"])
                        r_pts = int(op["points"])
                        domain.resize(r_nv, r_pts)
                        shared_data = [0.0] * domain.size()
                        domain.setData(shared_data)

                    elif op_name == "query_tolerances":
                        s_rtol, s_atol, t_rtol, t_atol = [], [], [], []
                        for n in range(domain.nComponents()):
                            s_rtol.append(domain.steady_rtol(n))
                            s_atol.append(domain.steady_atol(n))
                            t_rtol.append(domain.transient_rtol(n))
                            t_atol.append(domain.transient_atol(n))
                        out_rec["steady_rtol"] = s_rtol
                        out_rec["steady_atol"] = s_atol
                        out_rec["transient_rtol"] = t_rtol
                        out_rec["transient_atol"] = t_atol

                    elif op_name == "set_component_names":
                        names = op["names"]
                        for i in range(min(len(names), domain.nComponents())):
                            domain.setComponentName(i, names[i])

                    elif op_name == "query_component_names":
                        names = [domain.componentName(i) for i in range(domain.nComponents())]
                        out_rec["component_names"] = names

                    elif op_name == "query_component_indices":
                        query_names = op["names"]
                        indices = []
                        has_comps = []
                        for name in query_names:
                            indices.append(domain.componentIndex(name))
                            has_comps.append(domain.hasComponent(name))
                        out_rec["queried_indices"] = indices
                        out_rec["has_components"] = has_comps

                    elif op_name == "set_bounds":
                        n = int(op["n"])
                        lower = float(op["lower"])
                        upper = float(op["upper"])
                        domain.setBounds(n, lower, upper)

                    elif op_name == "query_bounds":
                        lower = [domain.lowerBound(n) for n in range(domain.nComponents())]
                        upper = [domain.upperBound(n) for n in range(domain.nComponents())]
                        out_rec["lower_bounds"] = lower
                        out_rec["upper_bounds"] = upper

                    elif op_name == "set_steady_tolerances":
                        rtol = float(op["rtol"])
                        atol = float(op["atol"])
                        n = int(op["n"])
                        domain.setSteadyTolerances(rtol, atol, n)

                    elif op_name == "set_transient_tolerances":
                        rtol = float(op["rtol"])
                        atol = float(op["atol"])
                        n = int(op["n"])
                        domain.setTransientTolerances(rtol, atol, n)

                    elif op_name == "init_time_integ":
                        dt = float(op["dt"])
                        x0 = [1.0] * domain.size()
                        domain.initTimeInteg(dt, x0)

                    elif op_name == "init_time_integ_with_state":
                        dt = float(op["dt"])
                        state = [float(x) for x in op["state"]]
                        domain.initTimeInteg(dt, state)

                    elif op_name == "set_steady_mode":
                        domain.setSteadyMode()

                    elif op_name == "query_mode_and_tolerances":
                        out_rec["steady_flag"] = domain.steady()
                        out_rec["transient_flag"] = domain.transient()
                        cur_rtol = [domain.rtol(n) for n in range(domain.nComponents())]
                        cur_atol = [domain.atol(n) for n in range(domain.nComponents())]
                        out_rec["current_rtol"] = cur_rtol
                        out_rec["current_atol"] = cur_atol

                    elif op_name == "query_prev_soln":
                        prev = []
                        for j in range(domain.nPoints()):
                            for n in range(domain.nComponents()):
                                prev.append(domain.prevSoln(n, j))
                        out_rec["prev_solution"] = prev

                    elif op_name == "setup_grid":
                        z = [float(x) for x in op["z"]]
                        domain.setupGrid(z)

                    elif op_name == "setup_uniform_grid":
                        unif_pts = int(op["points"])
                        length = float(op["length"])
                        start = float(op["start"])
                        domain.setupUniformGrid(unif_pts, length, start)

                    elif op_name == "query_grid":
                        z_pts = [domain.z(j) for j in range(domain.nPoints())]
                        out_rec["grid_points"] = z_pts
                        out_rec["zmin"] = domain.zmin()
                        out_rec["zmax"] = domain.zmax()

                    elif op_name == "reset_bad_values":
                        state = [float(x) for x in op["state"]]
                        domain.resetBadValues(state)
                        out_rec["clamped_state"] = state

                    elif op_name == "verify_topology_3":
                        d0 = TestDomain1D(1, 1)
                        d1 = TestDomain1D(3, 5)
                        d2 = TestDomain1D(1, 1)
                        d0.append(d1)
                        d1.append(d2)

                        out_rec["d0_loc"] = d0.loc()
                        out_rec["d0_firstPoint"] = d0.firstPoint()
                        out_rec["d0_lastPoint"] = d0.lastPoint()

                        out_rec["d1_loc"] = d1.loc()
                        out_rec["d1_firstPoint"] = d1.firstPoint()
                        out_rec["d1_lastPoint"] = d1.lastPoint()

                        out_rec["d2_loc"] = d2.loc()
                        out_rec["d2_firstPoint"] = d2.firstPoint()
                        out_rec["d2_lastPoint"] = d2.lastPoint()

                        out_rec["d1_global_idx_2_3"] = d1.loc() + d1.index(2, 3)
                        out_rec["has_d0_left"] = (d0.left() is not None)
                        out_rec["has_d0_right"] = (d0.right() is not None)
                        out_rec["has_d2_right"] = (d2.right() is not None)

                    elif op_name == "verify_topology_5":
                        mdoms = case_data.get("multi_domain", [])
                        dom_ptrs = [TestDomain1D(int(md["nv"]), int(md["points"])) for md in mdoms]
                        for k in range(len(dom_ptrs) - 1):
                            dom_ptrs[k].append(dom_ptrs[k + 1])

                        out_rec["topology5_locs"] = [dp.loc() for dp in dom_ptrs]
                        out_rec["topology5_firstPoints"] = [dp.firstPoint() for dp in dom_ptrs]
                        out_rec["topology5_lastPoints"] = [dp.lastPoint() for dp in dom_ptrs]

                    elif op_name == "set_refine_criteria":
                        ratio = float(op["ratio"])
                        slope = float(op["slope"])
                        curve = float(op["curve"])
                        prune = float(op["prune"])
                        domain.setRefineCriteria(ratio, slope, curve, prune)

                    elif op_name == "query_refine_criteria":
                        out_rec["refine_criteria"] = domain.getRefineCriteria()

                    elif op_name == "set_jacobian_mode":
                        domain.setJacobianMode(op["mode"])

                    elif op_name == "query_jacobian_mode":
                        out_rec["jacobian_mode"] = domain.jacobianMode()

                    elif op_name == "set_invalid_jacobian_mode":
                        caught = False
                        try:
                            domain.setJacobianMode(op["mode"])
                        except CanteraError:
                            caught = True
                        out_rec["caught_invalid_jacobian_mode"] = caught

                    elif op_name == "query_meta":
                        meta = domain.getMeta()
                        out_rec["meta"] = meta

                    elif op_name == "test_element_indexing_and_access":
                        x_arr = [0.0] * domain.size()
                        indices = []
                        for j in range(domain.nPoints()):
                            for n in range(domain.nComponents()):
                                idx = domain.index(n, j)
                                indices.append(idx)
                                x_arr[idx] = 10.0 * (n + 1) + 0.1 * (j + 1)

                        queried_values = []
                        for j in range(domain.nPoints()):
                            for n in range(domain.nComponents()):
                                queried_values.append(x_arr[domain.index(n, j)])

                        out_rec["element_indices"] = indices
                        out_rec["queried_values"] = queried_values

                    elif op_name == "test_initial_solution":
                        init_soln = [0.0] * domain.size()
                        domain._getInitialSoln(init_soln)
                        out_rec["initial_solution"] = init_soln

                    elif op_name == "set_flat_profile":
                        comp = int(op["component"])
                        val = float(op["value"])
                        for j in range(domain.nPoints()):
                            shared_data[domain.index(comp, j)] = val

                    elif op_name == "query_state":
                        out_rec["shared_state"] = list(shared_data)

            except CanteraError as err:
                out_rec["success"] = False
                out_rec["exception_thrown"] = True
                out_rec["error_message"] = str(err)
            except Exception as ex:
                out_rec["success"] = False
                out_rec["exception_thrown"] = True
                out_rec["error_message"] = str(ex)

            results.append(out_rec)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, indent=2)

    print(f"Executed {len(results)} Python Domain1D cases -> {output_path.relative_to(repo_root)}")


if __name__ == "__main__":
    run_cases()
