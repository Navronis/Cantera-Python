#!/usr/bin/env python3
"""Generate comprehensive differential corpus for CVodesIntegrator parity validation."""

import json
from pathlib import Path

def generate_corpus():
    corpus = []

    # 1. Scalar exponential decay across lambdas
    for lam in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        corpus.append({
            "case_id": f"decay_lam_{lam}",
            "problem_type": "decay",
            "mode": "integrate",
            "t0": 0.0,
            "tout": 1.0,
            "lambda": float(lam),
            "y_init": [1.0],
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 2. Parameterized exponential and sensitivities
    for p_val in [0.5, 1.0, 2.0, -1.0]:
        corpus.append({
            "case_id": f"sens_param_exp_p_{p_val}",
            "problem_type": "param_exp",
            "mode": "sens",
            "t0": 0.0,
            "tout": 1.0,
            "y_init": [1.0],
            "p_val": float(p_val),
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 3. Multi-parameter sensitivities
    corpus.append({
        "case_id": "sens_multi_param",
        "problem_type": "multi_param_sens",
        "mode": "sens",
        "t0": 0.0,
        "tout": 1.0,
        "y_init": [1.0, 2.0],
        "reltol": 1e-9,
        "abstol": 1e-15,
    })

    # 4. Coupled multiscale linear system
    corpus.append({
        "case_id": "coupled_linear_multiscale",
        "problem_type": "coupled_linear",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 1.0,
        "y_init": [1.0, 1.0],
        "reltol": 1e-9,
        "abstol": 1e-15,
    })

    # 5. Driven / harmonic oscillator
    for omega in [0.5, 1.0, 2.0, 5.0]:
        corpus.append({
            "case_id": f"oscillator_w_{omega}",
            "problem_type": "oscillator",
            "mode": "integrate",
            "t0": 0.0,
            "tout": 2.0,
            "omega": float(omega),
            "y_init": [1.0, 0.0],
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 6. Classic stiff Robertson kinetics across multiple time horizons
    for tout in [1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]:
        corpus.append({
            "case_id": f"robertson_t_{tout}",
            "problem_type": "robertson",
            "mode": "integrate",
            "t0": 0.0,
            "tout": float(tout),
            "y_init": [1.0, 0.0, 0.0],
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 7. Stiff van der Pol oscillator
    for tout in [0.5, 1.0, 2.0]:
        corpus.append({
            "case_id": f"vanderpol_mu10_t_{tout}",
            "problem_type": "vanderpol",
            "mode": "integrate",
            "t0": 0.0,
            "tout": float(tout),
            "mu": 10.0,
            "y_init": [2.0, 0.0],
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 8. Single-step advancement (step vs integrate)
    for tout in [0.1, 1.0, 5.0]:
        corpus.append({
            "case_id": f"step_advancement_t_{tout}",
            "problem_type": "decay",
            "mode": "step",
            "t0": 0.0,
            "tout": float(tout),
            "lambda": 2.0,
            "y_init": [1.0],
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 9. Reinitialization after solve
    corpus.append({
        "case_id": "reinit_state_reset",
        "problem_type": "decay",
        "mode": "reinit",
        "t0": 0.0,
        "tout": 2.0,
        "t_reinit": 1.0,
        "lambda": 1.0,
        "y_init": [1.0],
        "y_reinit": [5.0],
        "reltol": 1e-9,
        "abstol": 1e-15,
    })

    # 10. Tolerance sweeps
    for rtol in [1e-4, 1e-6, 1e-8, 1e-10]:
        corpus.append({
            "case_id": f"tol_sweep_rtol_{rtol}",
            "problem_type": "decay",
            "mode": "integrate",
            "t0": 0.0,
            "tout": 1.0,
            "lambda": 1.0,
            "y_init": [1.0],
            "reltol": float(rtol),
            "abstol": 1e-15,
        })

    # 11. Vector absolute tolerances
    corpus.append({
        "case_id": "vector_abstol_multiscale",
        "problem_type": "coupled_linear",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 1.0,
        "y_init": [1.0, 1.0],
        "reltol": 1e-8,
        "abstol_vec": [1e-12, 1e-6],
    })

    # 12. Maximum step size configuration
    corpus.append({
        "case_id": "max_step_size_limit",
        "problem_type": "decay",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 1.0,
        "lambda": 1.0,
        "y_init": [1.0],
        "hmax": 0.05,
        "reltol": 1e-9,
        "abstol": 1e-15,
    })

    # 13. Max steps failure and error propagation
    corpus.append({
        "case_id": "step_limit_failure",
        "problem_type": "decay",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 10.0,
        "lambda": 1.0,
        "y_init": [1.0],
        "max_steps": 3,
        "reltol": 1e-9,
        "abstol": 1e-15,
    })

    # 14. Max order configuration (Orders 1, 2, 3, 4, 5)
    for ord_val in [1, 2, 3, 4, 5]:
        t_target = 0.02 if ord_val == 1 else 1.0
        corpus.append({
            "case_id": f"max_order_setting_{ord_val}",
            "problem_type": "decay",
            "mode": "integrate",
            "t0": 0.0,
            "tout": t_target,
            "lambda": 1.0,
            "y_init": [1.0],
            "max_order": int(ord_val),
            "reltol": 1e-9,
            "abstol": 1e-15,
        })

    # 15. Linear solver types: DENSE and BAND
    corpus.append({
        "case_id": "linear_solver_dense",
        "problem_type": "decay",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 1.0,
        "lambda": 1.0,
        "y_init": [1.0],
        "linear_solver": "DENSE",
    })
    corpus.append({
        "case_id": "linear_solver_band",
        "problem_type": "coupled_linear",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 1.0,
        "y_init": [1.0, 1.0],
        "linear_solver": "BAND",
        "bandwidth_upper": 1,
        "bandwidth_lower": 1,
    })

    # 16. Intentional RHS failure propagation
    corpus.append({
        "case_id": "rhs_failure_propagation",
        "problem_type": "decay",
        "mode": "integrate",
        "t0": 0.0,
        "tout": 2.0,
        "fail_time": 0.5,
        "lambda": 1.0,
        "y_init": [1.0],
    })

    out_path = Path("audit/corpus_cvodes_integrator.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item) + "\n")

    print(f"Generated {len(corpus)} cases in {out_path}")

if __name__ == "__main__":
    generate_corpus()
