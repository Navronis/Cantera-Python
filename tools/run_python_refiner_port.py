"""Run Python Refiner port runner on corpus_refiner.jsonl.

Produces audit/python_refiner_port_results.json.
"""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python.oned import Refiner
from cantera_python.state import CanteraError


class MockDomain:
    def __init__(self, nv: int, np: int):
        self.n_components = nv
        self.n_points = np
        self._names = [f"component_{i}" for i in range(nv)]

    def nPoints(self) -> int:
        return self.n_points

    def nComponents(self) -> int:
        return self.n_components

    def componentName(self, n: int) -> str:
        return self._names[n]

    def component_name(self, n: int) -> str:
        return self._names[n]

    def index(self, n: int, j: int) -> int:
        return j * self.n_components + n

    def isFree(self) -> bool:
        return False

    def is_free(self) -> bool:
        return False

    def twoPointControlEnabled(self) -> bool:
        return False

    def two_point_control_enabled(self) -> bool:
        return False


def main():
    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = repo_root / "audit" / "corpus_refiner.jsonl"
    out_path = repo_root / "audit" / "python_refiner_port_results.json"

    results = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            n_points = case["n_points"]
            n_components = case["n_components"]
            n_call = case.get("n_call", n_points)
            z = case["z"]
            x = case["x"]
            criteria = case["criteria"]
            active = case.get("active", [True] * n_components)
            gridmin = case.get("gridmin", 1e-10)
            npmax = case.get("npmax", 1000)

            try:
                dom = MockDomain(n_components, n_points)
                ref = Refiner(dom)

                if len(criteria) >= 4:
                    ref.setCriteria(criteria[0], criteria[1], criteria[2], criteria[3])
                ref.setGridMin(gridmin)
                ref.setMaxPoints(npmax)

                for c_idx, act in enumerate(active):
                    ref.setActive(c_idx, act)

                n_new = ref.analyze(n_call, z, x)

                crit_out = ref.getCriteria()
                insert_pts = [j for j in range(n_call) if ref.newPointNeeded(j)]
                keep_pts = [ref.keepPoint(j) for j in range(n_call)]

                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": True,
                    "exception_thrown": False,
                    "error_message": "",
                    "n_new_points": n_new,
                    "max_ratio": ref.maxRatio(),
                    "max_delta": ref.maxDelta(),
                    "max_slope": ref.maxSlope(),
                    "prune": ref.prune(),
                    "criteria": crit_out,
                    "insert_points": insert_pts,
                    "keep_points": keep_pts,
                })
            except CanteraError as e:
                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": True,
                    "exception_thrown": True,
                    "error_message": str(e),
                    "n_new_points": -1,
                    "insert_points": [],
                    "keep_points": [],
                })
            except Exception as e:
                results.append({
                    "case_id": case_id,
                    "index": idx,
                    "success": False,
                    "exception_thrown": True,
                    "error_message": f"{type(e).__name__}: {e}",
                    "n_new_points": -1,
                    "insert_points": [],
                    "keep_points": [],
                })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"cases": results}, f, indent=2)

    print(f"Python Refiner Port finished processing {len(results)} cases into {out_path}")


if __name__ == "__main__":
    main()
