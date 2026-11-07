"""Generate deterministic differential validation corpus for Domain1D.

Generates:
  - audit/corpus_domain1d.jsonl
"""

import json
from pathlib import Path


def generate_corpus():
    repo_root = Path(__file__).resolve().parent.parent
    corpus_path = repo_root / "audit" / "corpus_domain1d.jsonl"

    cases = [
        {
            "case_id": "domain1d_01_default_construction",
            "description": "Default constructor (nv=1, points=1, time=0.0)",
            "nv": 1,
            "points": 1,
            "operations": [
                {"op": "query_basic"},
            ],
        },
        {
            "case_id": "domain1d_02_resize_and_defaults",
            "description": "Resize to nv=3, points=8, check default tolerances and bounds",
            "nv": 1,
            "points": 1,
            "operations": [
                {"op": "resize", "nv": 3, "points": 8},
                {"op": "query_basic"},
                {"op": "query_tolerances"},
            ],
        },
        {
            "case_id": "domain1d_03_component_names_and_indices",
            "description": "Set component names, query componentName, componentIndex, hasComponent",
            "nv": 3,
            "points": 5,
            "operations": [
                {"op": "set_component_names", "names": ["velocity", "temperature", "species_H2"]},
                {"op": "query_component_names"},
                {"op": "query_component_indices", "names": ["velocity", "temperature", "species_H2"]},
            ],
        },
        {
            "case_id": "domain1d_04_set_and_query_bounds",
            "description": "Set lower and upper bounds on components and query them",
            "nv": 3,
            "points": 5,
            "operations": [
                {"op": "set_bounds", "n": 0, "lower": -100.0, "upper": 200.0},
                {"op": "set_bounds", "n": 1, "lower": 250.0, "upper": 3500.0},
                {"op": "set_bounds", "n": 2, "lower": 0.0, "upper": 1.0},
                {"op": "query_bounds"},
            ],
        },
        {
            "case_id": "domain1d_05_steady_tolerances_all_and_per_comp",
            "description": "Set steady tolerances globally and override specific component",
            "nv": 4,
            "points": 6,
            "operations": [
                {"op": "set_steady_tolerances", "rtol": 1e-5, "atol": 1e-10, "n": -1},
                {"op": "set_steady_tolerances", "rtol": 1e-7, "atol": 1e-12, "n": 2},
                {"op": "query_tolerances"},
            ],
        },
        {
            "case_id": "domain1d_06_transient_tolerances_all_and_per_comp",
            "description": "Set transient tolerances globally and override specific component",
            "nv": 4,
            "points": 6,
            "operations": [
                {"op": "set_transient_tolerances", "rtol": 2e-4, "atol": 3e-10, "n": -1},
                {"op": "set_transient_tolerances", "rtol": 5e-6, "atol": 7e-12, "n": 1},
                {"op": "query_tolerances"},
            ],
        },
        {
            "case_id": "domain1d_07_mode_switching_tolerances",
            "description": "Switch between steady and transient mode and verify rtol/atol getters",
            "nv": 2,
            "points": 4,
            "operations": [
                {"op": "set_steady_tolerances", "rtol": 1e-3, "atol": 1e-8, "n": -1},
                {"op": "set_transient_tolerances", "rtol": 2e-5, "atol": 3e-11, "n": -1},
                {"op": "query_tolerances"},
                {"op": "init_time_integ", "dt": 0.005},
                {"op": "query_mode_and_tolerances"},
                {"op": "set_steady_mode"},
                {"op": "query_mode_and_tolerances"},
            ],
        },
        {
            "case_id": "domain1d_08_prev_solution_caching",
            "description": "initTimeInteg caches previous solution state vector for prevSoln query",
            "nv": 2,
            "points": 3,
            "operations": [
                {"op": "init_time_integ_with_state", "dt": 0.01, "state": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]},
                {"op": "query_prev_soln"},
            ],
        },
        {
            "case_id": "domain1d_09_setup_custom_grid",
            "description": "setupGrid with non-uniform grid coordinates and query z, zmin, zmax, grid",
            "nv": 2,
            "points": 1,
            "operations": [
                {"op": "setup_grid", "z": [0.0, 0.05, 0.12, 0.25, 0.45, 0.70, 1.0]},
                {"op": "query_grid"},
            ],
        },
        {
            "case_id": "domain1d_10_setup_uniform_grid",
            "description": "setupUniformGrid with points, length, and start offset",
            "nv": 1,
            "points": 1,
            "operations": [
                {"op": "setup_uniform_grid", "points": 11, "length": 0.05, "start": 0.01},
                {"op": "query_grid"},
            ],
        },
        {
            "case_id": "domain1d_11_reset_bad_values_clamping",
            "description": "resetBadValues clamps state components violating min and max bounds",
            "nv": 2,
            "points": 3,
            "operations": [
                {"op": "set_bounds", "n": 0, "lower": 0.0, "upper": 10.0},
                {"op": "set_bounds", "n": 1, "lower": -5.0, "upper": 5.0},
                {"op": "reset_bad_values", "state": [-2.0, 15.0, 5.0, -10.0, 12.0, 0.0]},
            ],
        },
        {
            "case_id": "domain1d_12_topology_three_domains",
            "description": "Assemble 3 domains (d0, d1, d2) via append, verify left, right, locate, loc, firstPoint, lastPoint",
            "nv": 2,
            "points": 4,
            "multi_domain": [
                {"name": "inlet", "nv": 1, "points": 1},
                {"name": "flow", "nv": 3, "points": 5},
                {"name": "outlet", "nv": 1, "points": 1},
            ],
            "operations": [
                {"op": "verify_topology_3"},
            ],
        },
        {
            "case_id": "domain1d_13_topology_five_domains",
            "description": "Assemble 5 domains and verify cascading locate() offsets and globalComponentIndex",
            "nv": 1,
            "points": 1,
            "multi_domain": [
                {"name": "d0", "nv": 2, "points": 1},
                {"name": "d1", "nv": 4, "points": 4},
                {"name": "d2", "nv": 1, "points": 1},
                {"name": "d3", "nv": 3, "points": 6},
                {"name": "d4", "nv": 2, "points": 1},
            ],
            "operations": [
                {"op": "verify_topology_5"},
            ],
        },
        {
            "case_id": "domain1d_14_refiner_criteria_delegation",
            "description": "setRefineCriteria and getRefineCriteria delegation to attached Refiner",
            "nv": 2,
            "points": 5,
            "operations": [
                {"op": "set_refine_criteria", "ratio": 12.0, "slope": 0.35, "curve": 0.45, "prune": 0.02},
                {"op": "query_refine_criteria"},
            ],
        },
        {
            "case_id": "domain1d_15_jacobian_mode_configuration",
            "description": "setJacobianMode and query jacobianMode across valid modes and invalid error",
            "nv": 1,
            "points": 3,
            "operations": [
                {"op": "set_jacobian_mode", "mode": "finite-difference"},
                {"op": "query_jacobian_mode"},
                {"op": "set_jacobian_mode", "mode": "analytic"},
                {"op": "query_jacobian_mode"},
                {"op": "set_jacobian_mode", "mode": "auto"},
                {"op": "query_jacobian_mode"},
                {"op": "set_invalid_jacobian_mode", "mode": "invalid_mode"},
            ],
        },
        {
            "case_id": "domain1d_16_metadata_uniform_tolerances",
            "description": "getMeta and setMeta with uniform scalar tolerances",
            "nv": 3,
            "points": 5,
            "operations": [
                {"op": "set_steady_tolerances", "rtol": 1e-4, "atol": 1e-8, "n": -1},
                {"op": "set_transient_tolerances", "rtol": 2e-4, "atol": 2e-9, "n": -1},
                {"op": "query_meta"},
            ],
        },
        {
            "case_id": "domain1d_17_metadata_nonuniform_tolerances",
            "description": "getMeta and setMeta with component-specific non-uniform tolerances",
            "nv": 3,
            "points": 4,
            "operations": [
                {"op": "set_component_names", "names": ["u", "V", "T"]},
                {"op": "set_steady_tolerances", "rtol": 1e-4, "atol": 1e-8, "n": 0},
                {"op": "set_steady_tolerances", "rtol": 2e-4, "atol": 3e-8, "n": 1},
                {"op": "set_steady_tolerances", "rtol": 5e-5, "atol": 1e-7, "n": 2},
                {"op": "query_meta"},
            ],
        },
        {
            "case_id": "domain1d_18_element_indexing_and_access",
            "description": "index(n, j), value(x, n, j), and setValue(n, j, v, x)",
            "nv": 3,
            "points": 4,
            "operations": [
                {"op": "test_element_indexing_and_access"},
            ],
        },
        {
            "case_id": "domain1d_19_initial_solution_loading",
            "description": "_getInitialSoln fills state vector using initialValue hooks",
            "nv": 2,
            "points": 4,
            "operations": [
                {"op": "test_initial_solution"},
            ],
        },
        {
            "case_id": "domain1d_20_set_flat_profile",
            "description": "setFlatProfile sets constant component value across all grid points",
            "nv": 3,
            "points": 5,
            "operations": [
                {"op": "set_flat_profile", "component": 1, "value": 42.75},
                {"op": "query_state"},
            ],
        },
    ]

    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} Domain1D differential test cases in {corpus_path.relative_to(repo_root)}")


if __name__ == "__main__":
    generate_corpus()
