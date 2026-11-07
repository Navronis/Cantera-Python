"""Generate deterministic differential corpus for Refiner (1D adaptive grid refinement).

Produces audit/corpus_refiner.jsonl containing 20 mathematically and physically
diverse 1D grid refinement test cases.
"""

import json
import math
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = audit_dir / "corpus_refiner.jsonl"

    cases = []

    # 1. Uniform grid with constant profile (N=10, nv=1)
    n1 = 10
    z1 = [i / (n1 - 1) for i in range(n1)]
    x1 = [1.0] * n1
    cases.append({
        "case_id": "ref_01_uniform_grid_constant",
        "description": "Uniform grid with constant scalar profile; zero insertions, all points kept",
        "n_points": n1,
        "n_components": 1,
        "z": z1,
        "x": x1,
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 2. Uniform grid with linear profile (N=10, nv=1)
    n2 = 10
    z2 = [i / (n2 - 1) for i in range(n2)]
    x2 = [2.0 * z for z in z2]
    cases.append({
        "case_id": "ref_02_uniform_grid_linear",
        "description": "Uniform grid with linear scalar profile; constant slope, zero curvature",
        "n_points": n2,
        "n_components": 1,
        "z": z2,
        "x": x2,
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 3. Steep tanh gradient triggering slope criterion (N=12, nv=1)
    n3 = 12
    z3 = [i / (n3 - 1) for i in range(n3)]
    x3 = [math.tanh(25.0 * (z - 0.5)) for z in z3]
    cases.append({
        "case_id": "ref_03_steep_gradient_slope",
        "description": "Steep tanh gradient profile exceeding slope criterion (m_slope=0.4)",
        "n_points": n3,
        "n_components": 1,
        "z": z3,
        "x": x3,
        "criteria": [10.0, 0.4, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 4. Sharp Gaussian peak triggering curvature criterion (N=15, nv=1)
    n4 = 15
    z4 = [i / (n4 - 1) for i in range(n4)]
    x4 = [math.exp(-120.0 * (z - 0.5) ** 2) for z in z4]
    cases.append({
        "case_id": "ref_04_sharp_gaussian_peak_curvature",
        "description": "Sharp localized Gaussian peak exceeding curvature criterion (m_curve=0.4)",
        "n_points": n4,
        "n_components": 1,
        "z": z4,
        "x": x4,
        "criteria": [10.0, 0.8, 0.4, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 5. Non-uniform grid ratio left (dz[j] > ratio * dz[j-1])
    # z = [0.0, 0.01, 0.02, 0.3, 0.6, 1.0] -> dz = [0.01, 0.01, 0.28, 0.3, 0.4]
    # dz[2] / dz[1] = 0.28 / 0.01 = 28.0 > 10.0
    z5 = [0.0, 0.01, 0.02, 0.3, 0.6, 1.0]
    n5 = len(z5)
    x5 = [0.5] * n5
    cases.append({
        "case_id": "ref_05_grid_ratio_left",
        "description": "Grid ratio test: dz[2] > 10.0 * dz[1], inserts at point 2 and protects points 1, 2, 3, 4",
        "n_points": n5,
        "n_components": 1,
        "z": z5,
        "x": x5,
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 6. Non-uniform grid ratio right (dz[j-1] > ratio * dz[j])
    # z = [0.0, 0.4, 0.7, 0.71, 0.72, 1.0] -> dz = [0.4, 0.3, 0.01, 0.01, 0.28]
    # dz[1] / dz[2] = 0.3 / 0.01 = 30.0 > 10.0
    z6 = [0.0, 0.4, 0.7, 0.71, 0.72, 1.0]
    n6 = len(z6)
    x6 = [0.5] * n6
    cases.append({
        "case_id": "ref_06_grid_ratio_right",
        "description": "Grid ratio test: dz[1] > 10.0 * dz[2], inserts at point 1 and protects points 0, 1, 2, 3",
        "n_points": n6,
        "n_components": 1,
        "z": z6,
        "x": x6,
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 7. Pruning in flat regions with positive prune threshold (N=20, prune=0.05)
    n7 = 20
    z7 = [i / (n7 - 1) for i in range(n7)]
    x7 = [math.exp(-200.0 * (z - 0.5) ** 2) for z in z7]
    cases.append({
        "case_id": "ref_07_prune_flat_regions",
        "description": "Pruning flat regions outside localized reaction zone (m_prune=0.05)",
        "n_points": n7,
        "n_components": 1,
        "z": z7,
        "x": x7,
        "criteria": [10.0, 0.8, 0.8, 0.05],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 8. Adjacent pruning protection (prevent removing consecutive points in single pass)
    n8 = 15
    z8 = [i / (n8 - 1) for i in range(n8)]
    x8 = [1.0] * n8  # completely flat
    cases.append({
        "case_id": "ref_08_adjacent_prune_protection",
        "description": "Adjacent pruning protection: ensures adjacent candidate removal points alternate with KEEP",
        "n_points": n8,
        "n_components": 1,
        "z": z8,
        "x": x8,
        "criteria": [10.0, 0.8, 0.8, 0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 9. Pruning ratio guard left (z[j+1] - z[j-1] > ratio * dz[j-2])
    # Protects point j from deletion if removing it would make left ratio too large
    z9 = [0.0, 0.01, 0.03, 0.25, 0.26, 0.5]
    n9 = len(z9)
    x9 = [0.0] * n9
    cases.append({
        "case_id": "ref_09_prune_ratio_guard_left",
        "description": "Pruning ratio guard left: point protected from deletion to avoid large interval ratio",
        "n_points": n9,
        "n_components": 1,
        "z": z9,
        "x": x9,
        "criteria": [5.0, 0.8, 0.8, 0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 10. Pruning ratio guard right (z[j+1] - z[j-1] > ratio * dz[j+1])
    z10 = [0.0, 0.24, 0.25, 0.47, 0.49, 0.5]
    n10 = len(z10)
    x10 = [0.0] * n10
    cases.append({
        "case_id": "ref_10_prune_ratio_guard_right",
        "description": "Pruning ratio guard right: point protected from deletion to avoid large right ratio",
        "n_points": n10,
        "n_components": 1,
        "z": z10,
        "x": x10,
        "criteria": [5.0, 0.8, 0.8, 0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 11. Minimum grid spacing clamp (dz < 2 * gridmin -> no insertion)
    z11 = [0.0, 0.1, 0.1 + 1e-11, 0.2, 0.3]
    n11 = len(z11)
    x11 = [0.0, 0.0, 1.0, 1.0, 1.0]  # huge step gradient at tiny interval
    cases.append({
        "case_id": "ref_11_gridmin_spacing_clamp",
        "description": "Minimum grid spacing clamp: interval dz < 2*gridmin suppresses point insertion",
        "n_points": n11,
        "n_components": 1,
        "z": z11,
        "x": x11,
        "criteria": [10.0, 0.2, 0.2, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 12. Multicomponent flame profile (T, u, Y_CH4, Y_O2; N=15, nv=4)
    n12 = 15
    nv12 = 4
    z12 = [i / (n12 - 1) for i in range(n12)]
    x12 = []
    for z in z12:
        T = 300.0 + 1700.0 * 0.5 * (1.0 + math.tanh(20.0 * (z - 0.4)))
        u = 0.5 + 2.0 * 0.5 * (1.0 + math.tanh(20.0 * (z - 0.4)))
        ych4 = 0.05 * 0.5 * (1.0 - math.tanh(20.0 * (z - 0.4)))
        yo2 = 0.20 * 0.5 * (1.0 - math.tanh(20.0 * (z - 0.4)))
        x12.extend([T, u, ych4, yo2])
    cases.append({
        "case_id": "ref_12_multicomponent_flame",
        "description": "Multicomponent flame profile (4 variables: T, u, Y_CH4, Y_O2) across 15 points",
        "n_points": n12,
        "n_components": nv12,
        "z": z12,
        "x": x12,
        "criteria": [10.0, 0.5, 0.5, -0.1],
        "active": [True, True, True, True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 13. Multicomponent with inactive component (setActive(2, False))
    cases.append({
        "case_id": "ref_13_active_deactive_components",
        "description": "Multicomponent flame profile with component 2 (Y_CH4) deactivated via setActive",
        "n_points": n12,
        "n_components": nv12,
        "z": z12,
        "x": x12,
        "criteria": [10.0, 0.5, 0.5, -0.1],
        "active": [True, True, False, True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 14. Small fluctuations filtered by min_range (range < min_range * magnitude)
    n14 = 10
    z14 = [i / (n14 - 1) for i in range(n14)]
    x14 = [1.0 + 1e-5 * math.sin(2.0 * math.pi * z) for z in z14]
    cases.append({
        "case_id": "ref_14_small_fluctuation_min_range",
        "description": "Small numerical fluctuations filtered out by m_min_range without refinement",
        "n_points": n14,
        "n_components": 1,
        "z": z14,
        "x": x14,
        "criteria": [10.0, 0.2, 0.2, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 15. Custom criteria (ratio=4.0, slope=0.25, curve=0.25, prune=0.02)
    n15 = 15
    z15 = [i / (n15 - 1) for i in range(n15)]
    x15 = [math.sin(3.0 * math.pi * z) for z in z15]
    cases.append({
        "case_id": "ref_15_set_criteria_custom",
        "description": "Multi-wave profile analyzed with custom stringent criteria (slope=0.25, curve=0.25)",
        "n_points": n15,
        "n_components": 1,
        "z": z15,
        "x": x15,
        "criteria": [4.0, 0.25, 0.25, 0.02],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 16. Strict boundary preservation (j=0 and j=N-1 are always KEEP)
    n16 = 8
    z16 = [i / (n16 - 1) for i in range(n16)]
    x16 = [0.0] * n16
    cases.append({
        "case_id": "ref_16_boundary_points_kept",
        "description": "Boundary points j=0 and j=N-1 are strictly preserved as KEEP regardless of prune threshold",
        "n_points": n16,
        "n_components": 1,
        "z": z16,
        "x": x16,
        "criteria": [10.0, 0.8, 0.8, 0.5],  # high prune threshold
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 17. Max points limit exception (n >= npmax)
    cases.append({
        "case_id": "ref_17_max_points_exception",
        "description": "Grid size reaches max allowable points (npmax=10), throwing CanteraError",
        "n_points": 10,
        "n_components": 1,
        "z": [i / 9 for i in range(10)],
        "x": [1.0] * 10,
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 10,  # n >= npmax triggers exception
        "expected_exception": "Refiner::analyze: max number of grid points reached (10)."
    })

    # 18. Mismatched grid size exception (n != domain.nPoints())
    cases.append({
        "case_id": "ref_18_mismatched_grid_size_exception",
        "description": "Call analyze with n=8 on a domain of size 10, throwing CanteraError",
        "n_points": 10,
        "n_components": 1,
        "n_call": 8,  # mismatch with domain size
        "z": [i / 7 for i in range(8)],
        "x": [1.0] * 8,
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": "Refiner::analyze: number of grid points provided does not match domain size."
    })

    # 19. Single point domain (n=1 returns 0 without error)
    cases.append({
        "case_id": "ref_19_single_point_domain",
        "description": "Single-point domain (n=1) returns 0 new points without error",
        "n_points": 1,
        "n_components": 1,
        "z": [0.0],
        "x": [1.0],
        "criteria": [10.0, 0.8, 0.8, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    # 20. Two separate localized reaction zone peaks (N=25, nv=1)
    n20 = 25
    z20 = [i / (n20 - 1) for i in range(n20)]
    x20 = [math.exp(-200.0 * (z - 0.25) ** 2) + math.exp(-200.0 * (z - 0.75) ** 2) for z in z20]
    cases.append({
        "case_id": "ref_20_two_peaks_localized",
        "description": "Two separate localized Gaussian peaks at z=0.25 and z=0.75 creating disjoint insertion clusters",
        "n_points": n20,
        "n_components": 1,
        "z": z20,
        "x": x20,
        "criteria": [10.0, 0.5, 0.5, -0.1],
        "active": [True],
        "gridmin": 1e-10,
        "npmax": 1000,
        "expected_exception": None
    })

    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} cases in {corpus_path}")


if __name__ == "__main__":
    main()
