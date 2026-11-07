"""Generate deterministic differential corpus for OneDim (1D multi-domain container & residual).

Produces audit/corpus_onedim.jsonl containing 20 mathematically and physically
representative test cases.
"""

import json
import math
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = audit_dir / "corpus_onedim.jsonl"

    cases = []

    # 1. Single domain, 5 points, 2 components (linear system)
    cases.append({
        "case_id": "oned_01_single_domain_linear",
        "description": "Single domain with 5 grid points and 2 components (linear reaction-diffusion)",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 3}
        ],
        "state": [100.0, 0.2] * 5,
        "step": [1.0, 0.01] * 5,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-9, 1e-9]
    })

    # 2. Single domain, 8 points, 3 components (quadratic system)
    cases.append({
        "case_id": "oned_02_single_domain_quadratic",
        "description": "Single domain with 8 grid points and 3 components (quadratic reaction-diffusion)",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 3, "n_points": 8, "bandwidth": 5}
        ],
        "state": [300.0, 0.5, 0.02] * 8,
        "step": [2.5, 0.05, 0.001] * 8,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4, 1e-4],
        "atol": [1e-8, 1e-8, 1e-8]
    })

    # 3. Three domains: connector (1 pt, 2 comp) + bulk (6 pts, 2 comp) + connector (1 pt, 2 comp)
    cases.append({
        "case_id": "oned_03_three_domains_symmetric",
        "description": "Three domains (inlet connector, bulk flow, outlet connector) with 2 components",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "flame", "type": "flow", "n_components": 2, "n_points": 6, "bandwidth": 3},
            {"name": "outlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1}
        ],
        "state": [300.0, 0.1] + [x for i in range(6) for x in (500.0 + 200.0 * i, 0.1 + 0.05 * i)] + [1700.0, 0.4],
        "step": [0.5, 0.01] * 8,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 4. Three domains with asymmetric component counts (1 comp, 4 comp, 1 comp)
    cases.append({
        "case_id": "oned_04_three_domains_asymmetric_vars",
        "description": "Three domains with asymmetric component counts: inlet (1), flame (4), outlet (1)",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 1, "n_points": 1, "bandwidth": 0},
            {"name": "flame", "type": "flow", "n_components": 4, "n_points": 5, "bandwidth": 7},
            {"name": "outlet", "type": "connector", "n_components": 1, "n_points": 1, "bandwidth": 0}
        ],
        "state": [300.0] + [300.0, 1.0, 0.2, 0.8] * 5 + [1500.0],
        "step": [1.0] + [5.0, 0.02, 0.005, 0.01] * 5 + [10.0],
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4, 1e-4, 1e-4],
        "atol": [1e-9, 1e-9, 1e-9, 1e-9]
    })

    # 5. Five domains: Inlet + Flow1 + Interface + Flow2 + Outlet
    cases.append({
        "case_id": "oned_05_five_domains_multistage",
        "description": "Five domains representing a catalytic surface dividing two reacting flow regions",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "flow1", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 3},
            {"name": "surface", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "flow2", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 3},
            {"name": "outlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1}
        ],
        "state": [300.0, 0.2] * 11,
        "step": [0.1, 0.001] * 11,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 6. weightedNorm with uniform step, zero state vector
    cases.append({
        "case_id": "oned_06_weighted_norm_zero_state",
        "description": "weightedNorm test: step vector evaluated on zero state, reducing error weight to atol",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 3}
        ],
        "state": [0.0, 0.0] * 5,
        "step": [1e-9, 2e-9] * 5,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-9, 1e-9]
    })

    # 7. weightedNorm with non-zero state and large disparate magnitudes
    cases.append({
        "case_id": "oned_07_weighted_norm_disparate_scales",
        "description": "weightedNorm test with flame-like disparate scales: T ~ 2000 K, Y ~ 1e-5",
        "domains": [
            {"name": "flame", "type": "flow", "n_components": 2, "n_points": 6, "bandwidth": 3}
        ],
        "state": [1800.0, 1.5e-5] * 6,
        "step": [1.8, 1.5e-7] * 6,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-3, 1e-3],
        "atol": [1e-5, 1e-9]
    })

    # 8. Multi-domain residual eval at steady-state (j=-1, rdt=0.0)
    cases.append({
        "case_id": "oned_08_eval_steady_full_grid",
        "description": "Full grid residual evaluation at steady-state (j=-1, rdt=0.0)",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 3},
            {"name": "outlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1}
        ],
        "state": [300.0, 1.0] + [400.0, 0.8, 600.0, 0.6, 900.0, 0.4, 1200.0, 0.2, 1500.0, 0.1] + [1500.0, 0.1],
        "step": [0.0] * 14,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 9. Multi-domain residual eval with transient pseudo-time stepping (rdt = 50.0)
    cases.append({
        "case_id": "oned_09_eval_transient_pseudo_time",
        "description": "Transient residual evaluation with pseudo-time step (rdt=50.0)",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 3}
        ],
        "state": [500.0, 0.3] * 5,
        "step": [0.0] * 10,
        "rdt": 50.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 10. Point-local residual eval (j = 2, only updating stencil neighbors)
    cases.append({
        "case_id": "oned_10_eval_point_local",
        "description": "Point-local residual evaluation at global point j=2",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 6, "bandwidth": 3}
        ],
        "state": [300.0, 0.1, 400.0, 0.2, 500.0, 0.3, 600.0, 0.4, 700.0, 0.5, 800.0, 0.6],
        "step": [0.0] * 12,
        "rdt": 0.0,
        "eval_point": 2,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 11. Bandwidth computation across multi-domain layout
    cases.append({
        "case_id": "oned_11_bandwidth_layout",
        "description": "Bandwidth calculation checking local domain bandwidth and inter-domain coupling",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 3, "n_points": 1, "bandwidth": 2},
            {"name": "flow", "type": "flow", "n_components": 3, "n_points": 5, "bandwidth": 5},
            {"name": "outlet", "type": "connector", "n_components": 3, "n_points": 1, "bandwidth": 2}
        ],
        "state": [1.0] * 21,
        "step": [0.1] * 21,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4] * 3,
        "atol": [1e-8] * 3
    })

    # 12. Finite-difference Jacobian evalJacobian for single domain
    cases.append({
        "case_id": "oned_12_eval_jacobian_single_domain",
        "description": "Finite-difference Jacobian assembly for single 5-point 2-component domain",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 3}
        ],
        "state": [300.0, 0.5, 450.0, 0.4, 600.0, 0.3, 800.0, 0.2, 1000.0, 0.1],
        "step": [0.0] * 10,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 13. Finite-difference Jacobian evalJacobian for three-domain assembly
    cases.append({
        "case_id": "oned_13_eval_jacobian_three_domains",
        "description": "Finite-difference Jacobian assembly across 3 domains with interface couplings",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 3},
            {"name": "outlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1}
        ],
        "state": [300.0, 0.2] + [400.0, 0.2, 600.0, 0.15, 800.0, 0.1, 1000.0, 0.05] + [1000.0, 0.05],
        "step": [0.0] * 12,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 14. Component queries: component, componentName, componentTableLabel
    cases.append({
        "case_id": "oned_14_component_queries",
        "description": "Querying component tuples (domain, pt, name) and table formatting",
        "domains": [
            {"name": "left_bnd", "type": "connector", "n_components": 1, "n_points": 1, "bandwidth": 0},
            {"name": "bulk_gas", "type": "flow", "n_components": 2, "n_points": 3, "bandwidth": 3},
            {"name": "right_bnd", "type": "connector", "n_components": 1, "n_points": 1, "bandwidth": 0}
        ],
        "state": [10.0, 100.0, 0.5, 200.0, 0.4, 300.0, 0.3, 20.0],
        "step": [0.0] * 8,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 15. Bounds queries: upperBound, lowerBound across domains
    cases.append({
        "case_id": "oned_15_bounds_queries",
        "description": "Upper and lower variable bounds delegation across multi-domain layout",
        "domains": [
            {"name": "inlet", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "flame", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 3}
        ],
        "state": [300.0, 0.1] * 5,
        "step": [0.0] * 10,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 16. Mode transitions: initTimeInteg and setSteadyMode
    cases.append({
        "case_id": "oned_16_time_integration_modes",
        "description": "Switching between time integration mode (dt=0.01) and steady mode (rdt=0)",
        "domains": [
            {"name": "flow", "type": "flow", "n_components": 2, "n_points": 4, "bandwidth": 3}
        ],
        "state": [400.0, 0.5] * 4,
        "step": [0.0] * 8,
        "rdt": 100.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 17. pointDomain mapping across 4 domains
    cases.append({
        "case_id": "oned_17_point_domain_mapping",
        "description": "Locating domain ownership for arbitrary global state vector indices",
        "domains": [
            {"name": "d0", "type": "connector", "n_components": 1, "n_points": 1, "bandwidth": 0},
            {"name": "d1", "type": "flow", "n_components": 3, "n_points": 4, "bandwidth": 5},
            {"name": "d2", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "d3", "type": "flow", "n_components": 2, "n_points": 3, "bandwidth": 3}
        ],
        "state": [1.0] * (1*1 + 3*4 + 2*1 + 2*3),  # 1 + 12 + 2 + 6 = 21
        "step": [0.0] * 21,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4] * 3,
        "atol": [1e-8] * 3
    })

    # 18. Domain index lookup by string name
    cases.append({
        "case_id": "oned_18_domain_index_lookup",
        "description": "Domain index query by id string name and bounds checking",
        "domains": [
            {"name": "inlet_zone", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1},
            {"name": "reaction_zone", "type": "flow", "n_components": 2, "n_points": 5, "bandwidth": 3},
            {"name": "exhaust_zone", "type": "connector", "n_components": 2, "n_points": 1, "bandwidth": 1}
        ],
        "state": [1.0] * 14,
        "step": [0.0] * 14,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8]
    })

    # 19. Out-of-bounds domain index exception
    cases.append({
        "case_id": "oned_19_domain_index_out_of_bounds_exception",
        "description": "Check domain index out of bounds throwing IndexError",
        "domains": [
            {"name": "d0", "type": "flow", "n_components": 1, "n_points": 3, "bandwidth": 1}
        ],
        "state": [1.0, 2.0, 3.0],
        "step": [0.0, 0.0, 0.0],
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4],
        "atol": [1e-8],
        "query_domain_index": 5,
        "expected_exception": "IndexError"
    })

    # 20. Out-of-bounds component index exception
    cases.append({
        "case_id": "oned_20_component_index_out_of_bounds_exception",
        "description": "Check component index out of bounds throwing IndexError",
        "domains": [
            {"name": "d0", "type": "flow", "n_components": 2, "n_points": 3, "bandwidth": 3}
        ],
        "state": [1.0] * 6,
        "step": [0.0] * 6,
        "rdt": 0.0,
        "eval_point": -1,
        "rtol": [1e-4, 1e-4],
        "atol": [1e-8, 1e-8],
        "query_component_index": 100,
        "expected_exception": "IndexError"
    })

    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} cases in {corpus_path}")


if __name__ == "__main__":
    main()
