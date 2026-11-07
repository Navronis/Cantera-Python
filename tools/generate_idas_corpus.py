"""Corpus generator for IdasIntegrator DAE differential verification."""

import json
from pathlib import Path


def generate_corpus(dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "case_id": "idas_poly_tol_standard",
            "system_type": "polynomial",
            "mode": "integrate",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.01, 0.05, 0.1, 0.2, 0.5],
        },
        {
            "case_id": "idas_poly_tol_tight",
            "system_type": "polynomial",
            "mode": "integrate",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.05, 0.1, 0.5, 1.0],
        },
        {
            "case_id": "idas_time_dep_standard",
            "system_type": "timedep",
            "mode": "integrate",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.1, 0.5, 1.0, 2.0],
        },
        {
            "case_id": "idas_time_dep_tight",
            "system_type": "timedep",
            "mode": "integrate",
            "rtol": 1e-8,
            "atol": 1e-10,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.2, 0.8, 1.5, 3.0],
        },
        {
            "case_id": "idas_robertson_stiff_short",
            "system_type": "robertson",
            "mode": "integrate",
            "rtol": 1e-4,
            "atol": 1e-6,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
        },
        {
            "case_id": "idas_robertson_stiff_long",
            "system_type": "robertson",
            "mode": "integrate",
            "rtol": 1e-5,
            "atol": 1e-7,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [1e-2, 1.0, 10.0, 100.0],
        },
        {
            "case_id": "idas_robertson_tight",
            "system_type": "robertson",
            "mode": "integrate",
            "rtol": 1e-7,
            "atol": 1e-9,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [1e-3, 0.1, 1.0, 10.0],
        },
        {
            "case_id": "idas_coupled_5var",
            "system_type": "coupled5",
            "mode": "integrate",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.05, 0.1, 0.2, 0.5],
        },
        {
            "case_id": "idas_step_mode",
            "system_type": "polynomial",
            "mode": "step",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_order": 4,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.02, 0.05, 0.1],
        },
        {
            "case_id": "idas_suppress_alg_false",
            "system_type": "robertson",
            "mode": "integrate",
            "rtol": 1e-5,
            "atol": 1e-7,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [1e-3, 0.1, 1.0],
        },
        {
            "case_id": "idas_suppress_alg_true",
            "system_type": "robertson",
            "mode": "integrate",
            "rtol": 1e-5,
            "atol": 1e-7,
            "max_order": 5,
            "include_algebraic_in_error_test": False,
            "sample_times": [1e-3, 0.1, 1.0],
        },
        {
            "case_id": "idas_reinit",
            "system_type": "reinit_poly",
            "mode": "integrate",
            "rtol": 1e-6,
            "atol": 1e-8,
            "max_order": 5,
            "include_algebraic_in_error_test": True,
            "sample_times": [0.05, 0.1, 0.15, 0.2],
        },
    ]

    with open(dest_path, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"Generated {len(cases)} IDAS DAE cases into {dest_path}")


if __name__ == "__main__":
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("audit/corpus_idas_integrator.jsonl")
    generate_corpus(out)
