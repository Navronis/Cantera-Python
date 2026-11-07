"""Python port runner for BandMatrix differential campaign."""

import json
import sys
from pathlib import Path
import numpy as np

# Ensure outputs/cantera-python is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python.numerics import BandMatrix, NumericsError, CanteraError


def run_cases(corpus_path: Path, output_path: Path, run_id: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            category = case["category"]
            n = case["n"]
            kl = case["kl"]
            ku = case["ku"]
            nrhs = case["nrhs"]

            result_entry = {
                "case_id": case_id,
                "category": category,
                "run_id": run_id,
            }

            if n == 0:
                m = BandMatrix(0, 0, 0)
                result_entry.update({
                    "status": "PASS",
                    "nRows": m.nRows(),
                    "nColumns": m.nColumns(),
                    "nSubDiagonals": m.nSubDiagonals(),
                    "nSuperDiagonals": m.nSuperDiagonals(),
                    "ldim": m.ldim(),
                    "oneNorm": m.oneNorm(),
                    "solution": [],
                })
                results.append(result_entry)
                continue

            matrix = BandMatrix(n, kl, ku)
            for (i, j, val) in case["entries"]:
                matrix[i, j] = val

            result_entry["nRows"] = matrix.nRows()
            result_entry["nColumns"] = matrix.nColumns()
            result_entry["nSubDiagonals"] = matrix.nSubDiagonals()
            result_entry["nSuperDiagonals"] = matrix.nSuperDiagonals()
            result_entry["ldim"] = matrix.ldim()
            result_entry["factorAlgorithm"] = matrix.factorAlgorithm()

            one_norm = matrix.oneNorm()
            result_entry["oneNorm"] = one_norm

            i_small, val_row_small = matrix.checkRows()
            result_entry["checkRows_iSmall"] = i_small
            result_entry["checkRows_valSmall"] = val_row_small

            j_small, val_col_small = matrix.checkColumns()
            result_entry["checkColumns_jSmall"] = j_small
            result_entry["checkColumns_valSmall"] = val_col_small

            # RHS
            rhs_matrix = np.asarray(case["rhs"], dtype=float)
            if rhs_matrix.ndim == 2 and rhs_matrix.shape[1] > 0:
                b0 = rhs_matrix[:, 0]
            else:
                b0 = np.zeros(n)

            # Mult and leftMult
            mult_res = matrix.mult(b0)
            result_entry["mult_result"] = [float(x) for x in mult_res]

            left_mult_res = matrix.leftMult(b0)
            result_entry["leftMult_result"] = [float(x) for x in left_mult_res]

            # Flatten rhs in column-major order to match solve(rhs, nrhs, n)
            rhs_flat = [float(rhs_matrix[row, col]) for col in range(nrhs) for row in range(n)]

            try:
                matrix.factor()
                result_entry["status"] = "PASS"
                result_entry["factor_info"] = matrix.info()

                rc = matrix.rcond(one_norm)
                result_entry["rcond"] = rc

                # Solve
                sol = matrix.solve(rhs_matrix)
                # Flatten solution in column-major order to match C++
                if sol.ndim == 1:
                    sol_flat = [float(x) for x in sol]
                else:
                    sol_flat = [float(sol[row, col]) for col in range(nrhs) for row in range(n)]
                result_entry["solution"] = sol_flat
            except (NumericsError, CanteraError) as err:
                result_entry["status"] = "SINGULAR"
                result_entry["error_message"] = str(err)
                result_entry["factor_info"] = matrix.info()
                result_entry["solution"] = []

            results.append(result_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Python port runner completed -> {output_path}")


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_python_band_matrix_port.py <corpus_path> <output_path> <run_id>")
        sys.exit(1)
    run_cases(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])


if __name__ == "__main__":
    main()
