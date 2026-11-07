"""Generate deterministic differential test corpus for Sim1D (1D solve orchestration).

Produces audit/corpus_sim1d.jsonl.
"""

import json
import math
from pathlib import Path


def make_tanh_profile(n_pts, z_min=0.0, z_max=1.0, z_mid=0.5, scale=20.0, v_low=1.0, v_high=10.0):
    z = [z_min + (z_max - z_min) * i / max(n_pts - 1, 1) for i in range(n_pts)]
    val = [v_low + (v_high - v_low) / (1.0 + math.exp(-scale * (zj - z_mid))) for zj in z]
    return z, val


def make_gaussian_profile(n_pts, z_min=0.0, z_max=1.0, z_center=0.5, width=0.05, amplitude=15.0, base=2.0):
    z = [z_min + (z_max - z_min) * i / max(n_pts - 1, 1) for i in range(n_pts)]
    val = [base + amplitude * math.exp(-((zj - z_center) / width) ** 2) for zj in z]
    return z, val


def main():
    repo_root = Path(__file__).resolve().parent.parent
    corpus_file = repo_root / "audit" / "corpus_sim1d.jsonl"

    cases = []

    # Case 1: Single domain value and grid access
    cases.append({
        "case_id": "sim1d_01_single_domain_values_and_grid",
        "description": "Single domain basic grid, values access and bandwidth",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 2, "grid": [0.0, 0.25, 0.5, 0.75, 1.0]}
        ],
        "state_by_domain": [
            [10.0, 20.0, 11.0, 21.0, 12.0, 22.0, 13.0, 23.0, 14.0, 24.0]
        ],
        "operations": [
            {"op": "check_structure"},
            {"op": "set_value", "domain": 0, "comp": 1, "point": 2, "val": 99.5},
            {"op": "get_value", "domain": 0, "comp": 1, "point": 2},
        ],
    })

    # Case 2: Refine with steep tanh slope
    z2, v2 = make_tanh_profile(9, scale=15.0)
    cases.append({
        "case_id": "sim1d_02_refine_slope_steep_tanh",
        "description": "Refinement triggered by steep slope in tanh profile",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 9, "bandwidth": 1, "grid": z2}
        ],
        "state_by_domain": [v2],
        "refine_criteria": [
            {"domain": 0, "ratio": 10.0, "slope": 0.25, "curve": 0.5, "prune": 0.0}
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 3: Refine with Gaussian curvature
    z3, v3 = make_gaussian_profile(11, width=0.08)
    cases.append({
        "case_id": "sim1d_03_refine_curvature_gaussian_peak",
        "description": "Refinement triggered by high curvature at Gaussian peak",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 11, "bandwidth": 1, "grid": z3}
        ],
        "state_by_domain": [v3],
        "refine_criteria": [
            {"domain": 0, "ratio": 10.0, "slope": 0.8, "curve": 0.2, "prune": 0.0}
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 4: Three domains (inlet, flow, outlet) refine
    z4, v4 = make_tanh_profile(7, scale=12.0)
    cases.append({
        "case_id": "sim1d_04_refine_three_domains_inlet_flow_outlet",
        "description": "3-domain layout with steep flow domain; inlet and outlet preserved",
        "domains": [
            {"name": "inlet", "type": "inlet", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [0.0]},
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 7, "bandwidth": 1, "grid": z4},
            {"name": "outlet", "type": "outlet", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [1.0]},
        ],
        "state_by_domain": [
            [1.0],
            v4,
            [10.0],
        ],
        "refine_criteria": [
            {"domain": 1, "ratio": 10.0, "slope": 0.3, "curve": 0.3, "prune": 0.0}
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 5: Pruning flat regions
    z5 = [0.1 * i for i in range(11)]
    v5 = [5.0] * 11  # completely flat
    cases.append({
        "case_id": "sim1d_05_refine_prune_discard_flat",
        "description": "Pruning flat profile with positive prune threshold",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 11, "bandwidth": 1, "grid": z5}
        ],
        "state_by_domain": [v5],
        "refine_criteria": [
            {"domain": 0, "ratio": 10.0, "slope": 0.8, "curve": 0.8, "prune": 0.5}
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 6: Set refine criteria per domain and all
    cases.append({
        "case_id": "sim1d_06_set_refine_criteria_multi_domain",
        "description": "Set refine criteria for all domains and per-domain overrides",
        "domains": [
            {"name": "inlet", "type": "inlet", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [0.0]},
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 2, "grid": [0.0, 0.25, 0.5, 0.75, 1.0]},
            {"name": "outlet", "type": "outlet", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [1.0]},
        ],
        "state_by_domain": [
            [1.0],
            [1.0, 2.0, 1.1, 2.1, 1.2, 2.2, 1.3, 2.3, 1.4, 2.4],
            [1.5],
        ],
        "operations": [
            {"op": "set_criteria", "domain": -1, "ratio": 8.0, "slope": 0.4, "curve": 0.4, "prune": 0.05},
            {"op": "set_criteria", "domain": 1, "ratio": 4.0, "slope": 0.2, "curve": 0.2, "prune": 0.02},
            {"op": "get_criteria", "domain": 0},
            {"op": "get_criteria", "domain": 1},
        ],
    })

    # Case 7: Gridmin clamp suppresses insertions
    z7, v7 = make_tanh_profile(7, scale=30.0)
    cases.append({
        "case_id": "sim1d_07_gridmin_clamp_suppression",
        "description": "gridmin larger than spacing suppresses insertions despite steep slope",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 7, "bandwidth": 1, "grid": z7}
        ],
        "state_by_domain": [v7],
        "refine_criteria": [
            {"domain": 0, "ratio": 10.0, "slope": 0.1, "curve": 0.1, "prune": 0.0}
        ],
        "operations": [
            {"op": "set_gridmin", "domain": 0, "gridmin": 0.5},
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 8: Max grid points limit
    cases.append({
        "case_id": "sim1d_08_max_grid_points_limit",
        "description": "Query and set max grid points limit",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 5, "bandwidth": 1, "grid": [0.0, 0.25, 0.5, 0.75, 1.0]}
        ],
        "state_by_domain": [
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ],
        "operations": [
            {"op": "set_max_points", "domain": 0, "npoints": 15},
            {"op": "get_max_points", "domain": 0},
            {"op": "set_max_points", "domain": -1, "npoints": 20},
            {"op": "get_max_points", "domain": 0},
        ],
    })

    # Case 9: Restore steady solution
    z9, v9 = make_tanh_profile(7, scale=20.0)
    cases.append({
        "case_id": "sim1d_09_restore_steady_solution",
        "description": "Refine updates m_xlast_ss and m_grid_last_ss, then restoreSteadySolution resets state",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 7, "bandwidth": 1, "grid": z9}
        ],
        "state_by_domain": [v9],
        "refine_criteria": [
            {"domain": 0, "ratio": 10.0, "slope": 0.3, "curve": 0.3, "prune": 0.0}
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
            {"op": "restore_steady"},
        ],
    })

    # Case 10: Error on restore steady solution without prior solution
    cases.append({
        "case_id": "sim1d_10_restore_steady_solution_error_no_prior",
        "description": "Calling restoreSteadySolution without prior solution throws CanteraError",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 5, "bandwidth": 1, "grid": [0.0, 0.25, 0.5, 0.75, 1.0]}
        ],
        "state_by_domain": [
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ],
        "operations": [
            {"op": "restore_steady"},
        ],
    })

    # Case 11: Error on restore time stepping solution without prior solution
    cases.append({
        "case_id": "sim1d_11_restore_time_stepping_solution_error_no_prior",
        "description": "Calling restoreTimeSteppingSolution without prior time steps throws CanteraError",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 5, "bandwidth": 1, "grid": [0.0, 0.25, 0.5, 0.75, 1.0]}
        ],
        "state_by_domain": [
            [1.0, 2.0, 3.0, 4.0, 5.0]
        ],
        "operations": [
            {"op": "restore_timestepping"},
        ],
    })

    # Case 12: Time step regrid max configuration
    cases.append({
        "case_id": "sim1d_12_time_step_regrid_max",
        "description": "Query and modify timeStepRegridMax setting",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 3, "bandwidth": 1, "grid": [0.0, 0.5, 1.0]}
        ],
        "state_by_domain": [
            [1.0, 2.0, 3.0]
        ],
        "operations": [
            {"op": "get_ts_regrid_max"},
            {"op": "set_ts_regrid_max", "count": 25},
            {"op": "get_ts_regrid_max"},
        ],
    })

    # Case 13: Eval steady-state Jacobian
    cases.append({
        "case_id": "sim1d_13_eval_ss_jacobian",
        "description": "Assemble steady-state Jacobian across multi-domain system via evalSSJacobian",
        "domains": [
            {"name": "inlet", "type": "inlet", "n_components": 2, "n_points": 1, "bandwidth": 1, "grid": [0.0]},
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 2, "grid": [0.0, 0.33, 0.67, 1.0]},
            {"name": "outlet", "type": "outlet", "n_components": 2, "n_points": 1, "bandwidth": 1, "grid": [1.0]},
        ],
        "state_by_domain": [
            [1.0, 2.0],
            [1.1, 2.1, 1.2, 2.2, 1.3, 2.3, 1.4, 2.4],
            [1.5, 2.5],
        ],
        "operations": [
            {"op": "eval_ss_jacobian"},
        ],
    })

    # Case 14: Solve adjoint system
    sz14 = 2 + 8 + 2
    b14 = [float(k + 1) for k in range(sz14)]
    cases.append({
        "case_id": "sim1d_14_solve_adjoint",
        "description": "Solve adjoint linear system J^T lambda = b via solveAdjoint",
        "domains": [
            {"name": "inlet", "type": "inlet", "n_components": 2, "n_points": 1, "bandwidth": 1, "grid": [0.0]},
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 2, "grid": [0.0, 0.33, 0.67, 1.0]},
            {"name": "outlet", "type": "outlet", "n_components": 2, "n_points": 1, "bandwidth": 1, "grid": [1.0]},
        ],
        "state_by_domain": [
            [1.0, 2.0],
            [1.1, 2.1, 1.2, 2.2, 1.3, 2.3, 1.4, 2.4],
            [1.5, 2.5],
        ],
        "adjoint_rhs": b14,
        "operations": [
            {"op": "solve_adjoint"},
        ],
    })

    # Case 15: 5-domain multistage layout refine
    z_s1, v_s1 = make_tanh_profile(5, z_min=0.0, z_max=0.5, scale=20.0)
    z_s2, v_s2 = make_gaussian_profile(5, z_min=0.5, z_max=1.0, width=0.1)
    cases.append({
        "case_id": "sim1d_15_multidomain_5_domains_refine",
        "description": "5-domain layout with independent refinement in two separate stages",
        "domains": [
            {"name": "inlet", "type": "inlet", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [0.0]},
            {"name": "stage1", "type": "flow", "n_components": 1, "n_points": 5, "bandwidth": 1, "grid": z_s1},
            {"name": "mid", "type": "connector", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [0.5]},
            {"name": "stage2", "type": "flow", "n_components": 1, "n_points": 5, "bandwidth": 1, "grid": z_s2},
            {"name": "outlet", "type": "outlet", "n_components": 1, "n_points": 1, "bandwidth": 1, "grid": [1.0]},
        ],
        "state_by_domain": [
            [1.0],
            v_s1,
            [5.0],
            v_s2,
            [10.0],
        ],
        "refine_criteria": [
            {"domain": 1, "ratio": 10.0, "slope": 0.3, "curve": 0.8, "prune": 0.0},
            {"domain": 3, "ratio": 10.0, "slope": 0.8, "curve": 0.3, "prune": 0.0},
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 16: Multicomponent profile refine
    z16 = [0.1 * i for i in range(7)]
    v16_T = [300.0 + 1500.0 / (1.0 + math.exp(-25.0 * (zj - 0.4))) for zj in z16]
    v16_u = [1.0 + 3.0 / (1.0 + math.exp(-20.0 * (zj - 0.4))) for zj in z16]
    v16_ch4 = [0.05 * math.exp(-((zj - 0.35) / 0.08) ** 2) for zj in z16]
    # Interleave 3 components
    v16_interleaved = []
    for j in range(7):
        v16_interleaved.extend([v16_T[j], v16_u[j], v16_ch4[j]])
    cases.append({
        "case_id": "sim1d_16_multicomponent_profile_refine",
        "description": "Refine across multiple active variables with different physical scales",
        "domains": [
            {"name": "flame", "type": "flow", "n_components": 3, "n_points": 7, "bandwidth": 3, "grid": z16}
        ],
        "state_by_domain": [v16_interleaved],
        "refine_criteria": [
            {"domain": 0, "ratio": 10.0, "slope": 0.4, "curve": 0.4, "prune": 0.0}
        ],
        "operations": [
            {"op": "refine", "loglevel": 0},
        ],
    })

    # Case 17: setLeftControlPoint exception on non-axisymmetric
    cases.append({
        "case_id": "sim1d_17_set_left_control_point_no_axisymmetric_error",
        "description": "setLeftControlPoint raises CanteraError when two-point control domain missing",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 3, "bandwidth": 1, "grid": [0.0, 0.5, 1.0]}
        ],
        "state_by_domain": [
            [300.0, 1000.0, 2000.0]
        ],
        "operations": [
            {"op": "set_left_control", "temp": 1200.0},
        ],
    })

    # Case 18: setRightControlPoint exception on non-axisymmetric
    cases.append({
        "case_id": "sim1d_18_set_right_control_point_no_axisymmetric_error",
        "description": "setRightControlPoint raises CanteraError when two-point control domain missing",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 3, "bandwidth": 1, "grid": [0.0, 0.5, 1.0]}
        ],
        "state_by_domain": [
            [300.0, 1000.0, 2000.0]
        ],
        "operations": [
            {"op": "set_right_control", "temp": 1200.0},
        ],
    })

    # Case 19: Error on _value index out of bounds
    cases.append({
        "case_id": "sim1d_19_error_value_index_out_of_bounds",
        "description": "_value with invalid localPoint raises CanteraError",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 3, "bandwidth": 1, "grid": [0.0, 0.5, 1.0]}
        ],
        "state_by_domain": [
            [1.0, 2.0, 3.0]
        ],
        "operations": [
            {"op": "get_value", "domain": 0, "comp": 0, "point": 999},
        ],
    })

    # Case 20: Error on _setValue index out of bounds
    cases.append({
        "case_id": "sim1d_20_error_set_value_index_out_of_bounds",
        "description": "_setValue with invalid localPoint raises CanteraError",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 1, "n_points": 3, "bandwidth": 1, "grid": [0.0, 0.5, 1.0]}
        ],
        "state_by_domain": [
            [1.0, 2.0, 3.0]
        ],
        "operations": [
            {"op": "set_value", "domain": 0, "comp": 0, "point": 999, "val": 42.0},
        ],
    })

    with open(corpus_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} deterministic Sim1D cases into {corpus_file}")


if __name__ == "__main__":
    main()
