"""Generate deterministic differential corpus for MultiNewton and MultiJac.

Produces audit/corpus_multi_newton.jsonl containing 20 mathematically and
physically diverse 1D nonlinear boundary value and steady-state systems.
"""

import json
import math
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = audit_dir / "corpus_multi_newton.jsonl"

    cases = []

    # 1. Linear 1D reaction-diffusion system (N=5, tridiagonal)
    # -d2x/dz2 + k*x = s, exact analytic/linear solution
    n1 = 5
    cases.append({
        "case_id": "mn_01_linear_tridiagonal",
        "description": "Linear 1D reaction-diffusion system (tridiagonal, analytic convergence in 1 Newton step)",
        "mode": "solve",
        "size": n1,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"diff": 1.0, "k": 2.0, "s": [1.0, 2.0, 1.5, 3.0, 0.5]},
        "lower_bounds": [-100.0] * n1,
        "upper_bounds": [100.0] * n1,
        "atol": [1e-4] * n1,
        "rtol": [1e-3] * n1,
        "rdt": 0.0,
        "mask": [1] * n1,
        "x0": [0.0] * n1,
        "max_jac_age": 5
    })

    # 2. Mildly nonlinear 1D system (N=6, quadratic source)
    # -d2x/dz2 + k*x^2 = s
    n2 = 6
    cases.append({
        "case_id": "mn_02_mildly_nonlinear_quadratic",
        "description": "Mildly nonlinear 1D diffusion with quadratic reaction rate (3-5 Newton iterations)",
        "mode": "solve",
        "size": n2,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.5, "k": 1.5, "s": [0.8, 1.2, 1.5, 1.4, 1.1, 0.9]},
        "lower_bounds": [0.0] * n2,
        "upper_bounds": [10.0] * n2,
        "atol": [1e-5] * n2,
        "rtol": [1e-4] * n2,
        "rdt": 0.0,
        "mask": [1] * n2,
        "x0": [0.5] * n2,
        "max_jac_age": 5
    })

    # 3. Strongly nonlinear Arrhenius flame reaction (N=8)
    # -d2T/dz2 + A * T * exp(-Ta / T) = s
    n3 = 8
    cases.append({
        "case_id": "mn_03_strong_arrhenius_damping",
        "description": "Strongly nonlinear Arrhenius flame temperature equation requiring Kee et al. damping",
        "mode": "solve",
        "size": n3,
        "bandwidth": 1,
        "system_type": "arrhenius",
        "params": {"diff": 0.2, "A": 50.0, "Ta": 5.0, "s": [1.0] * n3},
        "lower_bounds": [0.1] * n3,
        "upper_bounds": [20.0] * n3,
        "atol": [1e-4] * n3,
        "rtol": [1e-3] * n3,
        "rdt": 0.0,
        "mask": [1] * n3,
        "x0": [1.0] * n3,
        "max_jac_age": 5
    })

    # 4. Lower bound enforcement (N=5)
    # x0 + step would violate x >= 0, tests boundStep truncation
    n4 = 5
    cases.append({
        "case_id": "mn_04_lower_bound_truncation",
        "description": "Lower bound enforcement: Newton step would produce negative values, truncated by boundStep",
        "mode": "solve",
        "size": n4,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 1.0, "k": 5.0, "s": [0.1, 0.05, 0.02, 0.05, 0.1]},
        "lower_bounds": [0.01] * n4,
        "upper_bounds": [10.0] * n4,
        "atol": [1e-4] * n4,
        "rtol": [1e-3] * n4,
        "rdt": 0.0,
        "mask": [1] * n4,
        "x0": [0.05, 0.03, 0.02, 0.03, 0.05],
        "max_jac_age": 5
    })

    # 5. Upper bound enforcement (N=5)
    # x0 + step would exceed x <= 1.0 (e.g. mass fraction sum)
    n5 = 5
    cases.append({
        "case_id": "mn_05_upper_bound_truncation",
        "description": "Upper bound enforcement: step would exceed 1.0, fbound reduces step",
        "mode": "solve",
        "size": n5,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.5, "k": -2.0, "s": [1.5, 2.0, 2.5, 2.0, 1.5]},
        "lower_bounds": [0.0] * n5,
        "upper_bounds": [1.0] * n5,
        "atol": [1e-4] * n5,
        "rtol": [1e-3] * n5,
        "rdt": 0.0,
        "mask": [1] * n5,
        "x0": [0.85, 0.90, 0.92, 0.90, 0.85],
        "max_jac_age": 5
    })

    # 6. Solution at boundary failure (fbound < 1e-10 -> status = -3)
    n6 = 4
    cases.append({
        "case_id": "mn_06_boundary_failure_minus3",
        "description": "State x0 is right at boundary (1e-11) with outward step, triggers fbound < 1e-10 status -3",
        "mode": "dampStep",
        "size": n6,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"diff": 1.0, "k": 1.0, "s": [1.0] * n6},
        "lower_bounds": [0.0] * n6,
        "upper_bounds": [10.0] * n6,
        "atol": [1e-4] * n6,
        "rtol": [1e-3] * n6,
        "rdt": 0.0,
        "mask": [1] * n6,
        "x0": [1e-11, 0.5, 0.5, 0.5],
        "step0": [-0.1, 0.0, 0.0, 0.0],
        "max_jac_age": 5
    })

    # 7. Adaptive Jacobian aging (maxAge = 2)
    # Triggers forceNewJac when jac->age() > maxAge
    n7 = 6
    cases.append({
        "case_id": "mn_07_jacobian_aging_recompute",
        "description": "Jacobian aging with maxAge=2 requiring periodic Jacobian re-evaluations",
        "mode": "solve",
        "size": n7,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.2, "k": 3.0, "s": [1.0, 2.0, 3.0, 2.5, 1.8, 1.0]},
        "lower_bounds": [0.0] * n7,
        "upper_bounds": [50.0] * n7,
        "atol": [1e-5] * n7,
        "rtol": [1e-4] * n7,
        "rdt": 0.0,
        "mask": [1] * n7,
        "x0": [0.1] * n7,
        "max_jac_age": 2
    })

    # 8. Damping recovery from poor initial guess
    n8 = 7
    cases.append({
        "case_id": "mn_08_damping_recovery_poor_guess",
        "description": "Highly non-ideal initial guess far from equilibrium requiring progressive step damping",
        "mode": "solve",
        "size": n8,
        "bandwidth": 1,
        "system_type": "arrhenius",
        "params": {"diff": 0.1, "A": 20.0, "Ta": 4.0, "s": [0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]},
        "lower_bounds": [0.2] * n8,
        "upper_bounds": [15.0] * n8,
        "atol": [1e-4] * n8,
        "rtol": [1e-3] * n8,
        "rdt": 0.0,
        "mask": [1] * n8,
        "x0": [5.0] * n8,
        "max_jac_age": 5
    })

    # 9. Singular steady Jacobian handling (status = -4)
    # Matrix with zero diagonal element and rdt=0.0
    n9 = 4
    cases.append({
        "case_id": "mn_09_singular_jacobian_minus4",
        "description": "Steady-state problem with singular Jacobian (zero row) returning status -4",
        "mode": "solve",
        "size": n9,
        "bandwidth": 1,
        "system_type": "singular",
        "params": {"zero_row": 2},
        "lower_bounds": [-100.0] * n9,
        "upper_bounds": [100.0] * n9,
        "atol": [1e-4] * n9,
        "rtol": [1e-3] * n9,
        "rdt": 0.0,
        "mask": [1] * n9,
        "x0": [1.0, 1.0, 1.0, 1.0],
        "max_jac_age": 5
    })

    # 10. Pseudo-transient time stepping mode (rdt = 100.0)
    # Tests M = J - mask * rdt stabilization
    n10 = 6
    cases.append({
        "case_id": "mn_10_pseudo_transient_rdt",
        "description": "Pseudo-transient time step with rdt=100.0 adding diagonal damping terms M = J - mask*rdt",
        "mode": "solve",
        "size": n10,
        "bandwidth": 1,
        "system_type": "arrhenius",
        "params": {"diff": 0.2, "A": 30.0, "Ta": 4.0, "s": [1.0] * n10},
        "lower_bounds": [0.1] * n10,
        "upper_bounds": [20.0] * n10,
        "atol": [1e-4] * n10,
        "rtol": [1e-3] * n10,
        "rdt": 100.0,
        "mask": [1, 1, 1, 1, 1, 0],  # Last component algebraic
        "x0": [0.8] * n10,
        "max_jac_age": 5
    })

    # 11. Multi-variable coupled system (N=9, 3 points x 3 variables: U, V, T; bandwidth=3)
    n11 = 9
    cases.append({
        "case_id": "mn_11_coupled_block_banded",
        "description": "Coupled 3-variable 1D flow system (U, V, T) with block-banded Jacobian (bw=3)",
        "mode": "solve",
        "size": n11,
        "bandwidth": 3,
        "system_type": "coupled3",
        "params": {"diff": 0.5, "inter_coupling": 0.3, "s": [1.0, 0.5, 300.0] * 3},
        "lower_bounds": [-10.0, -10.0, 200.0] * 3,
        "upper_bounds": [50.0, 50.0, 2500.0] * 3,
        "atol": [1e-3, 1e-3, 1.0] * 3,
        "rtol": [1e-3, 1e-3, 1e-3] * 3,
        "rdt": 0.0,
        "mask": [1] * n11,
        "x0": [0.0, 0.0, 300.0, 1.0, 0.2, 500.0, 2.0, 0.5, 1200.0],
        "max_jac_age": 5
    })

    # 12. Direct MultiNewton.step undamped test
    n12 = 5
    cases.append({
        "case_id": "mn_12_undamped_step_eval",
        "description": "Direct evaluation of MultiNewton.step delta_x = -J^{-1} F(x) without damping",
        "mode": "step",
        "size": n12,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"diff": 1.0, "k": 3.0, "s": [2.0, 4.0, 5.0, 3.0, 1.0]},
        "lower_bounds": [-100.0] * n12,
        "upper_bounds": [100.0] * n12,
        "atol": [1e-4] * n12,
        "rtol": [1e-3] * n12,
        "rdt": 0.0,
        "mask": [1] * n12,
        "x0": [0.5, 1.0, 1.5, 1.0, 0.5],
        "max_jac_age": 5
    })

    # 13. Direct MultiNewton.boundStep verification
    n13 = 5
    cases.append({
        "case_id": "mn_13_bound_step_eval",
        "description": "Direct evaluation of MultiNewton.boundStep with various overshoot components",
        "mode": "boundStep",
        "size": n13,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"diff": 1.0, "k": 1.0, "s": [1.0] * n13},
        "lower_bounds": [0.0, 0.0, -1.0, 0.5, 0.0],
        "upper_bounds": [1.0, 2.0, 1.0, 5.0, 0.8],
        "atol": [1e-4] * n13,
        "rtol": [1e-3] * n13,
        "rdt": 0.0,
        "mask": [1] * n13,
        "x0": [0.5, 1.0, 0.0, 2.0, 0.7],
        "step0": [0.8, -1.5, 0.5, 4.0, 0.3],  # multiple components exceed bounds
        "max_jac_age": 5
    })

    # 14. Direct MultiNewton.dampStep iteration verification
    n14 = 5
    cases.append({
        "case_id": "mn_14_damp_step_eval",
        "description": "Direct evaluation of MultiNewton.dampStep testing exact alpha_k and norm progression",
        "mode": "dampStep",
        "size": n14,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.5, "k": 2.0, "s": [1.0] * n14},
        "lower_bounds": [0.0] * n14,
        "upper_bounds": [10.0] * n14,
        "atol": [1e-4] * n14,
        "rtol": [1e-3] * n14,
        "rdt": 0.0,
        "mask": [1] * n14,
        "x0": [1.0] * n14,
        "step0": [-0.4, 0.5, -0.6, 0.4, -0.3],
        "max_jac_age": 5
    })

    # 15. MultiJac standalone operations
    # initialize, setValue, value, setBandwidth, updateTransient, factorize, solve, reset, info
    n15 = 4
    cases.append({
        "case_id": "mn_15_multijac_operations",
        "description": "MultiJac lifecycle: initialization, banded fill, transient update, factorize, solve, reset",
        "mode": "multijac_ops",
        "size": n15,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"values": [[0, 0, 4.0], [0, 1, -1.0], [1, 0, -1.0], [1, 1, 5.0], [1, 2, -2.0],
                              [2, 1, -2.0], [2, 2, 6.0], [2, 3, -1.0], [3, 2, -1.0], [3, 3, 4.0]],
                   "rdt": 10.0, "mask": [1, 1, 1, 0], "rhs": [2.0, 3.0, 1.0, 5.0]},
        "lower_bounds": [-100.0] * n15,
        "upper_bounds": [100.0] * n15,
        "atol": [1e-4] * n15,
        "rtol": [1e-3] * n15,
        "rdt": 10.0,
        "mask": [1, 1, 1, 0],
        "x0": [0.0] * n15,
        "max_jac_age": 5
    })

    # 16. Larger flame grid (N=20, localized heat release zone)
    n16 = 20
    s16 = [0.1] * n16
    for i in range(8, 12):
        s16[i] = 10.0  # Flame flame front source peak
    cases.append({
        "case_id": "mn_16_flame_grid_20pts",
        "description": "1D flame grid (N=20) with localized heat release peak and steep gradients",
        "mode": "solve",
        "size": n16,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.1, "k": 1.0, "s": s16},
        "lower_bounds": [0.0] * n16,
        "upper_bounds": [100.0] * n16,
        "atol": [1e-4] * n16,
        "rtol": [1e-3] * n16,
        "rdt": 0.0,
        "mask": [1] * n16,
        "x0": [0.5] * n16,
        "max_jac_age": 5
    })

    # 17. Ill-conditioned boundary value system (small diffusion eps=1e-3)
    n17 = 6
    cases.append({
        "case_id": "mn_17_ill_conditioned_boundary_layer",
        "description": "Singularly perturbed boundary layer problem with small diffusion (eps=1e-3)",
        "mode": "solve",
        "size": n17,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"diff": 0.001, "k": 10.0, "s": [1.0, 1.0, 1.0, 1.0, 1.0, 50.0]},
        "lower_bounds": [-10.0] * n17,
        "upper_bounds": [100.0] * n17,
        "atol": [1e-6] * n17,
        "rtol": [1e-5] * n17,
        "rdt": 0.0,
        "mask": [1] * n17,
        "x0": [0.0] * n17,
        "max_jac_age": 5
    })

    # 18. Custom component tolerances (varying atol/rtol across species & temperature)
    n18 = 6
    cases.append({
        "case_id": "mn_18_variable_tolerance_norm",
        "description": "Multi-variable system with widely differing atol and rtol scaling per component",
        "mode": "solve",
        "size": n18,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.5, "k": 1.0, "s": [100.0, 1e-4, 150.0, 2e-4, 200.0, 1e-3]},
        "lower_bounds": [0.0] * n18,
        "upper_bounds": [5000.0] * n18,
        "atol": [1.0, 1e-6, 1.0, 1e-6, 1.0, 1e-6],
        "rtol": [1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 1e-4],
        "rdt": 0.0,
        "mask": [1] * n18,
        "x0": [50.0, 1e-5, 80.0, 1e-5, 120.0, 1e-4],
        "max_jac_age": 5
    })

    # 19. Boundary asymptotic approach without overshoot
    n19 = 5
    cases.append({
        "case_id": "mn_19_asymptotic_boundary_approach",
        "description": "Nonlinear system converging close to upper boundary (0.999) without overshoot",
        "mode": "solve",
        "size": n19,
        "bandwidth": 1,
        "system_type": "quadratic",
        "params": {"diff": 0.5, "k": 0.5, "s": [0.95, 0.98, 0.99, 0.98, 0.95]},
        "lower_bounds": [0.0] * n19,
        "upper_bounds": [1.0] * n19,
        "atol": [1e-4] * n19,
        "rtol": [1e-3] * n19,
        "rdt": 0.0,
        "mask": [1] * n19,
        "x0": [0.2] * n19,
        "max_jac_age": 5
    })

    # 20. Already converged initial guess (0 iterations needed)
    n20 = 4
    # Solve linear system exactly beforehand:
    # A = [[3, -1, 0, 0], [-1, 3, -1, 0], [0, -1, 3, -1], [0, 0, -1, 3]]
    # rhs = [1, 1, 1, 1] -> sol = [0.6, 0.8, 0.8, 0.6]
    cases.append({
        "case_id": "mn_20_already_converged_guess",
        "description": "Initial guess is already converged solution: single step norm < 1.0",
        "mode": "solve",
        "size": n20,
        "bandwidth": 1,
        "system_type": "linear",
        "params": {"diff": 1.0, "k": 1.0, "s": [1.0, 1.0, 1.0, 1.0]},
        "lower_bounds": [-10.0] * n20,
        "upper_bounds": [10.0] * n20,
        "atol": [1e-4] * n20,
        "rtol": [1e-3] * n20,
        "rdt": 0.0,
        "mask": [1] * n20,
        "x0": [0.6, 0.8, 0.8, 0.6],  # exact solution
        "max_jac_age": 5
    })

    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} cases in {corpus_path}")


if __name__ == "__main__":
    main()
