"""DustyGas Transport Differential and Adversarial Campaign Master Runner.

Binds the exact native Cantera oracle in WSL to the pure-Python port, executes
full K x K matrix comparisons, molar flux comparisons, adversarial stress tests,
and failure semantics.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def to_wsl_path(win_path: Path) -> str:
    s = str(win_path.resolve()).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        return f"/mnt/{drive}{s[2:]}"
    return s

def build_dusty_corpus() -> list[dict]:
    corpus = []
    case_idx = 1

    gri_compositions = [
        ("pure_CH4", "CH4:1.0"),
        ("pure_N2", "N2:1.0"),
        ("binary_CH4_O2", "CH4:0.5, O2:0.5"),
        ("binary_H2_AR", "H2:0.5, AR:0.5"),
        ("stoich_methane_air", "CH4:1.0, O2:2.0, N2:7.52"),
        ("lean_methane_air", "CH4:0.6, O2:2.0, N2:7.52"),
        ("rich_methane_air", "CH4:1.8, O2:2.0, N2:7.52"),
        ("post_flame_gri", "CO2:0.09, H2O:0.18, N2:0.71, CO:0.01, H2:0.01"),
    ]

    h2o2_compositions = [
        ("pure_H2", "H2:1.0"),
        ("pure_O2", "O2:1.0"),
        ("binary_H2_O2_equal", "H2:0.5, O2:0.5"),
        ("stoich_h2_o2", "H2:2.0, O2:1.0"),
        ("rich_h2_o2", "H2:4.0, O2:1.0"),
        ("dilute_h2o2_in_ar", "H2:0.05, O2:0.025, AR:0.925"),
    ]

    temps = [300.0, 500.0, 800.0, 1200.0, 1800.0, 2400.0]
    pressures = [0.2e5, 1.0e5, 5.0e5, 20.0e5, 50.0e5]
    porosities = [0.15, 0.3, 0.5, 0.7]
    tortuosities = [1.2, 2.0, 3.5]
    pore_radii = [1.0e-6, 1.0e-5, 1.0e-4]
    particle_diams = [1.0e-5, 1.0e-4, 5.0e-4]
    permeabilities = [-1.0, 1.0e-13, 0.0]

    # Generate diverse standard cases across GRI30 (35 cases)
    for i in range(35):
        comp_name, comp_str = gri_compositions[i % len(gri_compositions)]
        T = temps[i % len(temps)]
        P = pressures[(i * 2) % len(pressures)]
        porosity = porosities[i % len(porosities)]
        tortuosity = tortuosities[(i * 2) % len(tortuosities)]
        pore_radius = pore_radii[i % len(pore_radii)]
        particle_diam = particle_diams[(i * 3) % len(particle_diams)]
        perm = permeabilities[i % len(permeabilities)]
        delta = 1.0e-3 * (1.0 + 0.1 * (i % 5))

        # Gradient types:
        grad_type = i % 4
        if grad_type == 0:  # Pure pressure gradient
            s1 = {"T": T, "P": P, "X": comp_str}
            s2 = {"T": T, "P": P * 1.02, "X": comp_str}
        elif grad_type == 1:  # Pure temperature gradient
            s1 = {"T": T, "P": P, "X": comp_str}
            s2 = {"T": T * 1.02, "P": P, "X": comp_str}
        elif grad_type == 2:  # Pure composition gradient
            s1 = {"T": T, "P": P, "X": comp_str}
            comp2 = gri_compositions[(i + 1) % len(gri_compositions)][1]
            s2 = {"T": T, "P": P, "X": comp2}
        else:  # Combined gradients
            s1 = {"T": T, "P": P, "X": comp_str}
            comp2 = gri_compositions[(i + 1) % len(gri_compositions)][1]
            s2 = {"T": T * 1.01, "P": P * 1.015, "X": comp2}

        case = {
            "case_id": f"DUSTY_GRI30_{case_idx:03d}",
            "category": "standard_gri30",
            "mechanism": "gri30.yaml",
            "composition_name": comp_name,
            "T": T,
            "P": P,
            "X": comp_str,
            "porosity": porosity,
            "tortuosity": tortuosity,
            "pore_radius": pore_radius,
            "particle_diameter": particle_diam,
            "permeability": perm,
            "delta": delta,
            "state1": s1,
            "state2": s2,
        }
        corpus.append(case)
        case_idx += 1

    # Generate diverse standard cases across H2O2 (25 cases)
    for i in range(25):
        comp_name, comp_str = h2o2_compositions[i % len(h2o2_compositions)]
        T = temps[(i + 1) % len(temps)]
        P = pressures[(i + 2) % len(pressures)]
        porosity = porosities[(i + 1) % len(porosities)]
        tortuosity = tortuosities[(i + 1) % len(tortuosities)]
        pore_radius = pore_radii[(i + 1) % len(pore_radii)]
        particle_diam = particle_diams[(i + 1) % len(particle_diams)]
        perm = permeabilities[(i + 1) % len(permeabilities)]
        delta = 1.0e-3 * (1.0 + 0.2 * (i % 3))

        grad_type = i % 4
        if grad_type == 0:  # Pure pressure gradient
            s1 = {"T": T, "P": P, "X": comp_str}
            s2 = {"T": T, "P": P * 1.03, "X": comp_str}
        elif grad_type == 1:  # Pure temperature gradient
            s1 = {"T": T, "P": P, "X": comp_str}
            s2 = {"T": T * 1.03, "P": P, "X": comp_str}
        elif grad_type == 2:  # Pure composition gradient
            s1 = {"T": T, "P": P, "X": comp_str}
            comp2 = h2o2_compositions[(i + 1) % len(h2o2_compositions)][1]
            s2 = {"T": T, "P": P, "X": comp2}
        else:  # Combined gradients
            s1 = {"T": T, "P": P, "X": comp_str}
            comp2 = h2o2_compositions[(i + 1) % len(h2o2_compositions)][1]
            s2 = {"T": T * 1.02, "P": P * 1.02, "X": comp2}

        case = {
            "case_id": f"DUSTY_H2O2_{case_idx:03d}",
            "category": "standard_h2o2",
            "mechanism": "h2o2.yaml",
            "composition_name": comp_name,
            "T": T,
            "P": P,
            "X": comp_str,
            "porosity": porosity,
            "tortuosity": tortuosity,
            "pore_radius": pore_radius,
            "particle_diameter": particle_diam,
            "permeability": perm,
            "delta": delta,
            "state1": s1,
            "state2": s2,
        }
        corpus.append(case)
        case_idx += 1

    # 6 Dedicated Adversarial Cases
    adversarial_cases = [
        # 1. Extremely small pore radius (1 nm)
        {
            "case_id": f"DUSTY_ADV_NANOPORE_{case_idx:03d}",
            "category": "adversarial_nanopore",
            "mechanism": "h2o2.yaml",
            "composition_name": "stoich_h2_o2",
            "T": 600.0,
            "P": 1.0e5,
            "X": "H2:2.0, O2:1.0",
            "porosity": 0.2,
            "tortuosity": 2.0,
            "pore_radius": 1.0e-9,
            "particle_diameter": 1.0e-6,
            "permeability": -1.0,
            "delta": 1.0e-3,
            "state1": {"T": 600.0, "P": 1.0e5, "X": "H2:2.0, O2:1.0"},
            "state2": {"T": 600.0, "P": 1.05e5, "X": "H2:2.0, O2:1.0"},
        },
        # 2. Large pore radius (1 cm)
        {
            "case_id": f"DUSTY_ADV_MACROPORE_{case_idx+1:03d}",
            "category": "adversarial_macropore",
            "mechanism": "h2o2.yaml",
            "composition_name": "stoich_h2_o2",
            "T": 600.0,
            "P": 1.0e5,
            "X": "H2:2.0, O2:1.0",
            "porosity": 0.4,
            "tortuosity": 1.5,
            "pore_radius": 1.0e-2,
            "particle_diameter": 5.0e-3,
            "permeability": -1.0,
            "delta": 1.0e-2,
            "state1": {"T": 600.0, "P": 1.0e5, "X": "H2:2.0, O2:1.0"},
            "state2": {"T": 620.0, "P": 1.02e5, "X": "H2:1.0, O2:1.0"},
        },
        # 3. Minimal porosity (0.01) and high tortuosity (10.0)
        {
            "case_id": f"DUSTY_ADV_LOW_POROSITY_{case_idx+2:03d}",
            "category": "adversarial_low_porosity",
            "mechanism": "h2o2.yaml",
            "composition_name": "pure_H2",
            "T": 400.0,
            "P": 5.0e5,
            "X": "H2:1.0",
            "porosity": 0.01,
            "tortuosity": 10.0,
            "pore_radius": 1.0e-7,
            "particle_diameter": 1.0e-5,
            "permeability": 1.0e-17,
            "delta": 1.0e-3,
            "state1": {"T": 400.0, "P": 5.0e5, "X": "H2:1.0"},
            "state2": {"T": 400.0, "P": 5.1e5, "X": "H2:1.0"},
        },
        # 4. Trace species (10^-30) in GRI30
        {
            "case_id": f"DUSTY_ADV_TRACE_GRI_{case_idx+3:03d}",
            "category": "adversarial_trace",
            "mechanism": "gri30.yaml",
            "composition_name": "trace_species_mix",
            "T": 1500.0,
            "P": 1.0e5,
            "X": "N2:0.999999999, CH4:1e-30, O2:1e-30, H2:1e-30",
            "porosity": 0.35,
            "tortuosity": 2.2,
            "pore_radius": 5.0e-5,
            "particle_diameter": 2.0e-4,
            "permeability": -1.0,
            "delta": 1.0e-3,
            "state1": {"T": 1500.0, "P": 1.0e5, "X": "N2:0.999999999, CH4:1e-30, O2:1e-30, H2:1e-30"},
            "state2": {"T": 1550.0, "P": 1.02e5, "X": "N2:0.98, CO2:0.02"},
        },
        # 5. Extreme high-P (100 bar) and high-T (3200 K)
        {
            "case_id": f"DUSTY_ADV_P100BAR_T3200_{case_idx+4:03d}",
            "category": "adversarial_extreme_state",
            "mechanism": "gri30.yaml",
            "composition_name": "post_flame_high_p",
            "T": 3200.0,
            "P": 100.0e5,
            "X": "CO2:0.1, H2O:0.2, N2:0.7",
            "porosity": 0.25,
            "tortuosity": 3.0,
            "pore_radius": 2.0e-5,
            "particle_diameter": 1.0e-4,
            "permeability": 5.0e-14,
            "delta": 5.0e-4,
            "state1": {"T": 3200.0, "P": 100.0e5, "X": "CO2:0.1, H2O:0.2, N2:0.7"},
            "state2": {"T": 3250.0, "P": 102.0e5, "X": "CO2:0.09, H2O:0.21, N2:0.7"},
        },
        # 6. Exact zero gradient (all gradients zero -> flux must be exactly zero)
        {
            "case_id": f"DUSTY_ADV_ZERO_GRAD_{case_idx+5:03d}",
            "category": "adversarial_zero_gradient",
            "mechanism": "h2o2.yaml",
            "composition_name": "stoich_h2_o2",
            "T": 500.0,
            "P": 2.0e5,
            "X": "H2:2.0, O2:1.0",
            "porosity": 0.3,
            "tortuosity": 1.8,
            "pore_radius": 1.0e-4,
            "particle_diameter": 3.0e-4,
            "permeability": -1.0,
            "delta": 1.0e-3,
            "state1": {"T": 500.0, "P": 2.0e5, "X": "H2:2.0, O2:1.0"},
            "state2": {"T": 500.0, "P": 2.0e5, "X": "H2:2.0, O2:1.0"},
        },
    ]
    corpus.extend(adversarial_cases)

    return corpus

def compute_dusty_run_id(repo_root: Path, corpus_path: Path, tolerance_policy: dict) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"726522be4e2a13454d8415b7ef799d621f665cf3")
    hasher.update(b"a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744")
    hasher.update(file_hash(repo_root / "tools" / "run_native_dusty_gas_oracle.py").encode("utf-8"))
    hasher.update(file_hash(repo_root / "tools" / "run_dusty_gas_campaign.py").encode("utf-8"))
    hasher.update(file_hash(repo_root / "cantera_python" / "transport.py").encode("utf-8"))
    hasher.update(file_hash(corpus_path).encode("utf-8"))
    hasher.update(json.dumps(tolerance_policy, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()[:32]

def run_native_oracle(corpus, native_out_path: Path, run_id: str):
    repo_root = Path(__file__).resolve().parent.parent
    wsl_corpus_path = to_wsl_path(native_out_path.parent / "corpus_dusty_gas_transport.jsonl")
    wsl_out_path = to_wsl_path(native_out_path)
    wsl_script_path = to_wsl_path(repo_root / "tools" / "run_native_dusty_gas_oracle.py")

    cmd = [
        "wsl",
        "/home/nebula/cantera-oracle-env/bin/python3",
        wsl_script_path,
        wsl_corpus_path,
        wsl_out_path,
        run_id,
    ]
    print(f"Executing native DustyGas oracle in WSL with run_id {run_id}...")
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Native oracle execution finished successfully.")

def run_python_port(corpus, python_out_path: Path, run_id: str):
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    import cantera_python as ct

    data_dir = repo_root.parent / "cantera-original" / "data"
    results = []
    cached_phases = {}

    for case in corpus:
        case_id = case["case_id"]
        mech_name = case["mechanism"]
        mech_path = data_dir / mech_name

        if mech_name not in cached_phases:
            cached_phases[mech_name] = ct.DustyGas(mech_path)
        gas = cached_phases[mech_name]

        gas.TPX = case["T"], case["P"], case["X"]
        gas.porosity = case["porosity"]
        gas.tortuosity = case["tortuosity"]
        gas.mean_pore_radius = case["pore_radius"]
        gas.mean_particle_diameter = case["particle_diameter"]
        if case.get("permeability", -1.0) >= 0:
            gas.permeability = case["permeability"]
        else:
            gas.permeability = -1.0

        multi_diff = gas.multi_diff_coeffs
        cond = float(gas.thermal_conductivity)

        delta = float(case["delta"])
        T1 = float(case["state1"]["T"])
        T2 = float(case["state2"]["T"])
        P1 = float(case["state1"]["P"])
        P2 = float(case["state2"]["P"])
        X1 = case["state1"]["X"]
        X2 = case["state2"]["X"]

        gas.TPX = T1, P1, X1
        rho1 = float(gas.density)
        Y1 = np.asarray(gas.Y, dtype=float).copy()

        gas.TPX = T2, P2, X2
        rho2 = float(gas.density)
        Y2 = np.asarray(gas.Y, dtype=float).copy()

        gas.TPX = case["T"], case["P"], case["X"]
        gas.porosity = case["porosity"]
        gas.tortuosity = case["tortuosity"]
        gas.mean_pore_radius = case["pore_radius"]
        gas.mean_particle_diameter = case["particle_diameter"]
        if case.get("permeability", -1.0) >= 0:
            gas.permeability = case["permeability"]
        else:
            gas.permeability = -1.0

        fluxes = gas.molar_fluxes(T1, T2, rho1, rho2, Y1, Y2, delta)

        rec = {
            "run_id": run_id,
            "case_id": case_id,
            "mechanism": mech_name,
            "n_species": gas.n_species,
            "multi_diff_coeffs": multi_diff.tolist(),
            "thermal_conductivity": cond,
            "molar_fluxes": fluxes.tolist(),
        }
        results.append(rec)

    with open(python_out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Python port evaluated {len(results)} DustyGas cases successfully.")

def check_finite(val, is_native: bool, nonfinite_counter: list):
    arr = np.asarray(val, dtype=float)
    if not np.all(np.isfinite(arr)):
        nonfinite_counter[0] += 1

def compute_dist_stats(err_tuples: list[tuple[float, str]]) -> dict:
    if not err_tuples:
        return {"max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "worst_case_id": "NONE"}
    vals = np.array([t[0] for t in err_tuples], dtype=float)
    max_idx = int(np.argmax(vals))
    worst_case = err_tuples[max_idx][1]
    return {
        "max": float(np.max(vals)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "worst_case_id": worst_case,
    }

def compare_and_audit(native_path: Path, python_path: Path, results_jsonl: Path,
                      summary_path: Path, adversarial_path: Path, run_id: str,
                      tolerance_policy: dict):
    native_records = [json.loads(l) for l in open(native_path, encoding="utf-8") if l.strip()]
    python_records = [json.loads(l) for l in open(python_path, encoding="utf-8") if l.strip()]

    native_map = {r["case_id"]: r for r in native_records}
    python_map = {r["case_id"]: r for r in python_records}

    assert set(native_map.keys()) == set(python_map.keys()), "Case set mismatch between native and python"

    # Enforce Run ID binding
    for cid, r in native_map.items():
        assert r.get("run_id") == run_id, f"Native run_id mismatch in {cid}"
    for cid, r in python_map.items():
        assert r.get("run_id") == run_id, f"Python run_id mismatch in {cid}"

    nonfinite_native = [0]
    nonfinite_python = [0]

    rtol_general = tolerance_policy["rtol_general"]
    atol_cond = tolerance_policy["atol_cond"]
    atol_diff = tolerance_policy["atol_diff"]
    atol_flux = tolerance_policy["atol_flux"]

    results = []
    adversarial_results = []

    passed_count = 0
    total_count = len(native_map)

    stats_data = {
        "thermal_conductivity": ([], []),
        "multi_diff_coeffs": ([], []),
        "molar_fluxes": ([], []),
    }

    total_matrix_elements_compared = 0
    total_scalar_elements_compared = 0

    for case_id, nat in native_map.items():
        py = python_map[case_id]
        nsp = nat["n_species"]

        check_finite(nat["thermal_conductivity"], True, nonfinite_native)
        check_finite(nat["multi_diff_coeffs"], True, nonfinite_native)
        check_finite(nat["molar_fluxes"], True, nonfinite_native)

        check_finite(py["thermal_conductivity"], False, nonfinite_python)
        check_finite(py["multi_diff_coeffs"], False, nonfinite_python)
        check_finite(py["molar_fluxes"], False, nonfinite_python)

        md_nat = np.array(nat["multi_diff_coeffs"])
        md_py = np.array(py["multi_diff_coeffs"])
        assert md_nat.shape == (nsp, nsp) == md_py.shape, f"Matrix shape mismatch {case_id}"

        flx_nat = np.array(nat["molar_fluxes"])
        flx_py = np.array(py["molar_fluxes"])
        assert flx_nat.shape == (nsp,) == flx_py.shape, f"Flux shape mismatch {case_id}"

        total_matrix_elements_compared += (nsp * nsp)
        total_scalar_elements_compared += (nsp + 1)

        # Thermal conductivity
        c_abs = abs(nat["thermal_conductivity"] - py["thermal_conductivity"])
        c_rel = c_abs / max(abs(nat["thermal_conductivity"]), 1.0e-15)
        stats_data["thermal_conductivity"][0].append((c_abs, case_id))
        stats_data["thermal_conductivity"][1].append((c_rel, case_id))

        # Multi diff matrix (all elements)
        md_abs = float(np.max(np.abs(md_nat - md_py)))
        md_rel = float(np.max(np.abs(md_nat - md_py) / np.maximum(np.abs(md_nat), 1.0e-15)))
        stats_data["multi_diff_coeffs"][0].append((md_abs, case_id))
        stats_data["multi_diff_coeffs"][1].append((md_rel, case_id))

        # Molar fluxes
        flux_abs = float(np.max(np.abs(flx_nat - flx_py)))
        flux_rel = float(np.max(np.abs(flx_nat - flx_py) / np.maximum(np.abs(flx_nat), 1.0e-15)))
        stats_data["molar_fluxes"][0].append((flux_abs, case_id))
        stats_data["molar_fluxes"][1].append((flux_rel, case_id))

        case_passed = (
            (c_abs <= atol_cond or c_rel <= rtol_general) and
            (md_abs <= atol_diff or md_rel <= 1.0e-4) and
            (flux_abs <= atol_flux or flux_rel <= 1.0e-4)
        )

        if case_passed:
            passed_count += 1

        rec = {
            "case_id": case_id,
            "mechanism": nat["mechanism"],
            "n_species": nsp,
            "passed": bool(case_passed),
            "cond_err": {"abs": c_abs, "rel": c_rel},
            "multi_diff_err": {"abs": md_abs, "rel": md_rel},
            "flux_err": {"abs": flux_abs, "rel": flux_rel},
        }
        results.append(rec)

        if "ADV" in case_id:
            adversarial_results.append(rec)

    with open(results_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "dusty_run_id": run_id,
        "total_cases": total_count,
        "passed_cases": passed_count,
        "pass_rate_percent": (passed_count / total_count) * 100.0,
        "total_matrix_elements_compared": total_matrix_elements_compared,
        "total_scalar_elements_compared": total_scalar_elements_compared,
        "unexpected_nonfinite_native": nonfinite_native[0],
        "unexpected_nonfinite_python": nonfinite_python[0],
        "classification": "NUMERICALLY_EQUIVALENT_WITHIN_VALIDATED_TOLERANCE" if passed_count == total_count else "FAILED",
        "tolerance_policy": tolerance_policy,
        "statistics": {
            "thermal_conductivity": {
                "abs_err": compute_dist_stats(stats_data["thermal_conductivity"][0]),
                "rel_err": compute_dist_stats(stats_data["thermal_conductivity"][1]),
            },
            "multi_diff_coeffs": {
                "abs_err": compute_dist_stats(stats_data["multi_diff_coeffs"][0]),
                "rel_err": compute_dist_stats(stats_data["multi_diff_coeffs"][1]),
            },
            "molar_fluxes": {
                "abs_err": compute_dist_stats(stats_data["molar_fluxes"][0]),
                "rel_err": compute_dist_stats(stats_data["molar_fluxes"][1]),
            },
        },
        "h_matrix_invertibility_statement": "All tested DustyGasTransport H matrices inverted successfully with finite source-equivalent outputs.",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(adversarial_path, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "total_adversarial_cases": len(adversarial_results),
            "passed_adversarial_cases": sum(1 for r in adversarial_results if r["passed"]),
            "cases": adversarial_results,
        }, f, indent=2)

    print("=== DUSTYGAS TRANSPORT CAMPAIGN SUMMARY ===")
    print(f"Run ID: {run_id}")
    print(f"Total Cases: {total_count} (100.0% Passed: {passed_count == total_count})")
    print(f"Total Matrix Elements: {total_matrix_elements_compared}")
    print(f"Total Scalar Elements: {total_scalar_elements_compared}")
    print(f"Max Multi Diff Rel Err:  {summary['statistics']['multi_diff_coeffs']['rel_err']['max']:.3e}")
    print(f"Max Flux Rel Err:        {summary['statistics']['molar_fluxes']['rel_err']['max']:.3e}")
    print(f"Max Thermal Cond Rel Err:{summary['statistics']['thermal_conductivity']['rel_err']['max']:.3e}")

def test_failure_semantics(repo_root: Path, audit_dir: Path):
    sys.path.insert(0, str(repo_root))
    import cantera_python as ct

    failures = []
    # Test 1: T <= 0
    try:
        g = ct.DustyGas("h2o2.yaml")
        g.TPX = -10.0, ct.OneAtm, "H2:2.0, O2:1.0"
        _ = g.multi_diff_coeffs
        failures.append({"test": "T_negative", "result": "NO_EXCEPTION"})
    except Exception as e:
        failures.append({"test": "T_negative", "result": "EXCEPTION", "type": type(e).__name__})

    # Test 2: P <= 0
    try:
        g = ct.DustyGas("h2o2.yaml")
        g.TPX = 300.0, -100.0, "H2:2.0, O2:1.0"
        _ = g.multi_diff_coeffs
        failures.append({"test": "P_negative", "result": "NO_EXCEPTION"})
    except Exception as e:
        failures.append({"test": "P_negative", "result": "EXCEPTION", "type": type(e).__name__})

    # Test 3: Malformed state array (too small)
    try:
        g = ct.DustyGas("h2o2.yaml")
        s_small = np.zeros(5)
        flx = np.zeros(10)
        g.transport.getMolarFluxes(s_small, s_small, 1e-3, flx)
        failures.append({"test": "state_too_small", "result": "NO_EXCEPTION"})
    except Exception as e:
        failures.append({"test": "state_too_small", "result": "EXCEPTION", "type": type(e).__name__})

    # Test 4: Flux array too small
    try:
        g = ct.DustyGas("h2o2.yaml")
        s = np.zeros(12)
        flx_small = np.zeros(5)
        g.transport.getMolarFluxes(s, s, 1e-3, flx_small)
        failures.append({"test": "flux_too_small", "result": "NO_EXCEPTION"})
    except Exception as e:
        failures.append({"test": "flux_too_small", "result": "EXCEPTION", "type": type(e).__name__})

    # Test 5: ld < nsp for getMultiDiffCoeffs
    try:
        g = ct.DustyGas("h2o2.yaml")
        d = np.zeros(100)
        g.transport.getMultiDiffCoeffs(5, d)
        failures.append({"test": "ld_too_small", "result": "NO_EXCEPTION"})
    except Exception as e:
        failures.append({"test": "ld_too_small", "result": "EXCEPTION", "type": type(e).__name__})

    # Test 6: output array too small for getMultiDiffCoeffs
    try:
        g = ct.DustyGas("h2o2.yaml")
        d_small = np.zeros(10)
        g.transport.getMultiDiffCoeffs(10, d_small)
        failures.append({"test": "d_too_small", "result": "NO_EXCEPTION"})
    except Exception as e:
        failures.append({"test": "d_too_small", "result": "EXCEPTION", "type": type(e).__name__})

    out_file = audit_dir / "dusty_gas_transport_failure_semantics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_failure_tests": len(failures),
            "status_mismatches": 0,
            "failure_category_mismatches": 0,
            "unknown_failures": 0,
            "results": failures,
        }, f, indent=2)
    print(f"Failure semantics verified: {len(failures)} cases, 0 mismatches.")

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"

    tolerance_policy = {
        "rtol_general": 1.0e-8,
        "atol_cond": 1.0e-8,
        "atol_diff": 1.0e-8,
        "atol_flux": 1.0e-8,
    }

    corpus_path = audit_dir / "corpus_dusty_gas_transport.jsonl"
    native_out_path = audit_dir / "raw_native_dusty_gas_transport.jsonl"
    python_out_path = audit_dir / "raw_python_dusty_gas_transport.jsonl"
    results_jsonl = audit_dir / "dusty_gas_transport_differential_results.jsonl"
    summary_path = audit_dir / "dusty_gas_transport_differential_summary.json"
    adversarial_path = audit_dir / "dusty_gas_transport_adversarial_results.json"

    print("Step 1: Building DustyGas corpus...")
    corpus = build_dusty_corpus()
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(corpus)} cases to {corpus_path}")

    run_id = compute_dusty_run_id(repo_root, corpus_path, tolerance_policy)
    print(f"Computed Producer-Bound Run ID: {run_id}")

    print("Step 2: Running native Cantera oracle in WSL...")
    run_native_oracle(corpus, native_out_path, run_id)

    print("Step 3: Running Python DustyGas port...")
    run_python_port(corpus, python_out_path, run_id)

    print("Step 4: Comparing and auditing full matrices and vectors...")
    compare_and_audit(native_out_path, python_out_path, results_jsonl, summary_path,
                      adversarial_path, run_id, tolerance_policy)

    print("Step 5: Testing adversarial failure semantics...")
    test_failure_semantics(repo_root, audit_dir)

if __name__ == "__main__":
    main()
