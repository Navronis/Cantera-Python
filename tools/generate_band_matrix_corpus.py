"""Generate deterministic test corpus for BandMatrix differential campaign."""

import json
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
audit_dir = repo_root / "audit"

def make_corpus():
    cases = []

    # 1. Tridiagonal n=8, symmetric diagonally dominant
    cases.append({
        "case_id": "band_tridiagonal_n8",
        "category": "tridiagonal",
        "n": 8, "kl": 1, "ku": 1, "nrhs": 1,
        "seed": 101, "diag_boost": 4.0, "force_pivot": False, "is_singular": False
    })

    # 2. Tridiagonal n=20, diffusion-like stencil
    cases.append({
        "case_id": "band_tridiagonal_n20",
        "category": "tridiagonal",
        "n": 20, "kl": 1, "ku": 1, "nrhs": 2,
        "seed": 102, "diag_boost": 3.0, "force_pivot": False, "is_singular": False
    })

    # 3. Pentadiagonal n=10, 4th order stencil
    cases.append({
        "case_id": "band_pentadiagonal_n10",
        "category": "pentadiagonal",
        "n": 10, "kl": 2, "ku": 2, "nrhs": 2,
        "seed": 103, "diag_boost": 5.0, "force_pivot": False, "is_singular": False
    })

    # 4. Asymmetric kl=1, ku=3, n=12 (advection upwind)
    cases.append({
        "case_id": "band_asymmetric_kl1_ku3_n12",
        "category": "asymmetric_bandwidth",
        "n": 12, "kl": 1, "ku": 3, "nrhs": 2,
        "seed": 104, "diag_boost": 4.0, "force_pivot": False, "is_singular": False
    })

    # 5. Asymmetric kl=3, ku=1, n=15 (downwind)
    cases.append({
        "case_id": "band_asymmetric_kl3_ku1_n15",
        "category": "asymmetric_bandwidth",
        "n": 15, "kl": 3, "ku": 1, "nrhs": 2,
        "seed": 105, "diag_boost": 4.0, "force_pivot": False, "is_singular": False
    })

    # 6. Upper triangular kl=0, ku=2, n=6
    cases.append({
        "case_id": "band_upper_triangular_kl0_ku2_n6",
        "category": "triangular",
        "n": 6, "kl": 0, "ku": 2, "nrhs": 1,
        "seed": 106, "diag_boost": 3.0, "force_pivot": False, "is_singular": False
    })

    # 7. Lower triangular kl=2, ku=0, n=6
    cases.append({
        "case_id": "band_lower_triangular_kl2_ku0_n6",
        "category": "triangular",
        "n": 6, "kl": 2, "ku": 0, "nrhs": 1,
        "seed": 107, "diag_boost": 3.0, "force_pivot": False, "is_singular": False
    })

    # 8. Zero diagonal at entry (0, 0) forcing row pivot
    cases.append({
        "case_id": "band_pivot_zero_diagonal_n6",
        "category": "pivoting",
        "n": 6, "kl": 1, "ku": 1, "nrhs": 2,
        "seed": 108, "diag_boost": 3.0, "force_pivot": True, "is_singular": False
    })

    # 9. Heavy row pivoting with larger subdiagonals
    cases.append({
        "case_id": "band_heavy_pivoting_n16",
        "category": "pivoting",
        "n": 16, "kl": 3, "ku": 2, "nrhs": 2,
        "seed": 109, "diag_boost": 0.5, "force_pivot": False, "is_singular": False
    })

    # 10. Ill-conditioned decaying band
    cases.append({
        "case_id": "band_ill_conditioned_n10",
        "category": "ill_conditioned",
        "n": 10, "kl": 2, "ku": 2, "nrhs": 2,
        "seed": 110, "diag_boost": 0.1, "force_pivot": False, "is_singular": False, "decay": True
    })

    # 11. Multi-RHS with 3 columns, n=12
    cases.append({
        "case_id": "band_multi_rhs_3cols_n12",
        "category": "multi_rhs",
        "n": 12, "kl": 2, "ku": 2, "nrhs": 3,
        "seed": 111, "diag_boost": 4.0, "force_pivot": False, "is_singular": False
    })

    # 12. Multi-RHS with 5 columns, n=25
    cases.append({
        "case_id": "band_multi_rhs_5cols_n25",
        "category": "multi_rhs",
        "n": 25, "kl": 3, "ku": 3, "nrhs": 5,
        "seed": 112, "diag_boost": 5.0, "force_pivot": False, "is_singular": False
    })

    # 13. Representative 1D combustion Jacobian band, n=100
    cases.append({
        "case_id": "band_combustion_jacobian_n100",
        "category": "large_flame_scale",
        "n": 100, "kl": 4, "ku": 4, "nrhs": 2,
        "seed": 113, "diag_boost": 6.0, "force_pivot": False, "is_singular": False
    })

    # 14. Matrix-vector multiplication and left multiplication, n=15
    cases.append({
        "case_id": "band_matrix_vector_mult_n15",
        "category": "matrix_vector_mult",
        "n": 15, "kl": 2, "ku": 3, "nrhs": 1,
        "seed": 114, "diag_boost": 3.0, "force_pivot": False, "is_singular": False
    })

    # 15. Check rows and check columns near-zero inspection, n=8
    cases.append({
        "case_id": "band_check_rows_columns_n8",
        "category": "introspection",
        "n": 8, "kl": 2, "ku": 2, "nrhs": 1,
        "seed": 115, "diag_boost": 4.0, "force_pivot": False, "is_singular": False, "small_row": 3
    })

    # 16. Norm and condition number (rcond), n=14
    cases.append({
        "case_id": "band_norm_and_rcond_n14",
        "category": "conditioning",
        "n": 14, "kl": 3, "ku": 1, "nrhs": 1,
        "seed": 116, "diag_boost": 3.5, "force_pivot": False, "is_singular": False
    })

    # 17. Repeated solves with reused factorization, n=10
    cases.append({
        "case_id": "band_repeated_solve_reused_lu_n10",
        "category": "factorization_reuse",
        "n": 10, "kl": 2, "ku": 2, "nrhs": 1,
        "repeated_rhs_count": 4,
        "seed": 117, "diag_boost": 4.0, "force_pivot": False, "is_singular": False
    })

    # 18. Modify matrix after factorization and re-solve, n=8
    cases.append({
        "case_id": "band_modify_and_refactor_n8",
        "category": "mutation_and_refactor",
        "n": 8, "kl": 1, "ku": 1, "nrhs": 1,
        "seed": 118, "diag_boost": 4.0, "force_pivot": False, "is_singular": False, "modify_diagonal": 5.0
    })

    # 19. Singular matrix detection, n=6 (column 2 all zeros)
    cases.append({
        "case_id": "band_singular_zero_column_n6",
        "category": "singular",
        "n": 6, "kl": 1, "ku": 1, "nrhs": 1,
        "seed": 119, "diag_boost": 3.0, "force_pivot": False, "is_singular": True, "zero_col": 2
    })

    # 20. Empty edge matrix n=0
    cases.append({
        "case_id": "band_zero_dimension_n0",
        "category": "edge_dimension",
        "n": 0, "kl": 0, "ku": 0, "nrhs": 1,
        "seed": 120, "diag_boost": 0.0, "force_pivot": False, "is_singular": False
    })

    # Populate explicit numerical matrix and rhs arrays for each case
    populated = []
    for spec in cases:
        n = spec["n"]
        kl = spec["kl"]
        ku = spec["ku"]
        nrhs = spec["nrhs"]
        seed = spec["seed"]
        rng = np.random.RandomState(seed)

        entries = []
        if n > 0:
            for i in range(n):
                for j in range(max(0, i - kl), min(n, i + ku + 1)):
                    if spec.get("decay", False):
                        val = 1.0 / (abs(i - j) + 1.0) * (0.8 ** abs(i - j))
                    else:
                        val = float(rng.uniform(-2.0, 2.0))
                    if i == j:
                        val += spec["diag_boost"]
                    entries.append((i, j, float(val)))

            if spec.get("force_pivot", False):
                # Set (0, 0) = 0.0 and (1, 0) = 2.0 to force pivot
                entries = [(i, j, (0.0 if (i == 0 and j == 0) else (2.0 if (i == 1 and j == 0) else v))) for (i, j, v) in entries]

            if spec.get("is_singular", False):
                zcol = spec.get("zero_col", 1)
                entries = [(i, j, (0.0 if j == zcol else v)) for (i, j, v) in entries]

            if "small_row" in spec:
                srow = spec["small_row"]
                entries = [(i, j, (v * 1e-12 if i == srow else v)) for (i, j, v) in entries]

            # RHS vectors
            rhs_matrix = [[float(rng.uniform(-5.0, 5.0)) for _ in range(nrhs)] for _ in range(n)]
        else:
            rhs_matrix = []

        item = {
            "case_id": spec["case_id"],
            "category": spec["category"],
            "n": n,
            "kl": kl,
            "ku": ku,
            "nrhs": nrhs,
            "is_singular": spec.get("is_singular", False),
            "entries": entries,
            "rhs": rhs_matrix,
            "repeated_rhs_count": spec.get("repeated_rhs_count", 0),
            "modify_diagonal": spec.get("modify_diagonal", None),
        }
        populated.append(item)

    corpus_file = audit_dir / "corpus_band_matrix.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for item in populated:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {corpus_file} with {len(populated)} deterministic BandMatrix test cases.")


if __name__ == "__main__":
    make_corpus()
