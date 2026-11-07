"""Deterministic differential campaign for Mixture Transport against pinned native Cantera oracle (726522b)."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

def to_wsl_path(path: Path) -> str:
    s = str(path.resolve()).replace("\\", "/")
    if len(s) > 2 and s[1] == ":":
        drive = s[0].lower()
        return f"/mnt/{drive}{s[2:]}"
    return s


def build_test_corpus():
    """Build a comprehensive, deterministic corpus of gas transport states."""
    corpus = []
    case_idx = 0

    # 1. GRI30 (53 species)
    # Temperatures: 300, 500, 800, 1000, 1500, 2000, 2500, 3000 K
    # Pressures: 0.1 bar, 1 atm, 5 bar, 20 bar, 50 bar, 100 bar
    gri_compositions = [
        ("pure_CH4", "CH4:1.0"),
        ("pure_H2", "H2:1.0"),
        ("pure_O2", "O2:1.0"),
        ("pure_N2", "N2:1.0"),
        ("pure_AR", "AR:1.0"),
        ("pure_CO2", "CO2:1.0"),
        ("pure_H2O", "H2O:1.0"),
        ("binary_H2_O2_equal", "H2:0.5, O2:0.5"),
        ("binary_CH4_O2_equal", "CH4:0.5, O2:0.5"),
        ("binary_H2_AR_extreme_mass", "H2:0.5, AR:0.5"),
        ("binary_H2_C3H8_extreme_mass", "H2:0.9, C3H8:0.1"),
        ("stoich_methane_air", "CH4:1.0, O2:2.0, N2:7.52"),
        ("lean_methane_air_phi05", "CH4:0.5, O2:2.0, N2:7.52"),
        ("rich_methane_air_phi20", "CH4:2.0, O2:2.0, N2:7.52"),
        ("stoich_syngas_air", "H2:1.0, CO:1.0, O2:1.0, N2:3.76"),
        ("oxy_methane", "CH4:1.0, O2:2.0"),
        ("trace_radicals_in_N2", "N2:0.98, H:0.005, OH:0.005, O:0.005, CH3:0.005"),
        ("trace_heavy_in_H2", "H2:0.999, C3H8:0.001"),
        ("post_flame_exhaust", "CO2:0.09, H2O:0.18, N2:0.71, O2:0.02"),
    ]

    gri_conditions = [
        (300.0, 101325.0),
        (500.0, 1.0e4),
        (800.0, 5.0e5),
        (1000.0, 101325.0),
        (1200.0, 2.0e6),
        (1500.0, 101325.0),
        (1800.0, 5.0e6),
        (2200.0, 1.0e7),
        (2800.0, 101325.0),
        (3200.0, 2.0e6),
    ]

    for comp_name, comp_str in gri_compositions:
        for T, P in gri_conditions[:4]:  # 4 states per composition
            case_idx += 1
            corpus.append({
                "case_id": f"MIXTRAN_GRI30_{case_idx:03d}",
                "mechanism": "gri30.yaml",
                "composition_name": comp_name,
                "T": T,
                "P": P,
                "X": comp_str,
            })

    # Additional high-stress GRI30 states
    for T, P in [(300.0, 1.0e7), (3000.0, 1.0e4), (3200.0, 1.0e7)]:
        for comp_name, comp_str in [
            ("stoich_methane_air", "CH4:1.0, O2:2.0, N2:7.52"),
            ("binary_H2_AR_extreme_mass", "H2:0.5, AR:0.5"),
            ("trace_heavy_in_H2", "H2:0.999, C3H8:0.001"),
        ]:
            case_idx += 1
            corpus.append({
                "case_id": f"MIXTRAN_GRI30_STRESS_{case_idx:03d}",
                "mechanism": "gri30.yaml",
                "composition_name": comp_name,
                "T": T,
                "P": P,
                "X": comp_str,
            })

    # 2. H2O2 (9/10 species)
    h2o2_compositions = [
        ("pure_H2", "H2:1.0"),
        ("pure_O2", "O2:1.0"),
        ("pure_H2O", "H2O:1.0"),
        ("pure_H2O2", "H2O2:1.0"),
        ("binary_H2_O2_stoich", "H2:2.0, O2:1.0"),
        ("binary_H2_O2_rich", "H2:4.0, O2:1.0"),
        ("binary_H2_O2_lean", "H2:1.0, O2:1.0"),
        ("dilute_radicals", "AR:0.95, H:0.01, OH:0.02, HO2:0.01, O:0.01"),
        ("steam_hydrogen", "H2O:0.8, H2:0.2"),
    ]

    h2o2_conditions = [
        (300.0, 101325.0),
        (600.0, 5.0e5),
        (1200.0, 101325.0),
        (1800.0, 2.0e6),
        (2500.0, 5.0e6),
    ]

    for comp_name, comp_str in h2o2_compositions:
        for T, P in h2o2_conditions:
            case_idx += 1
            corpus.append({
                "case_id": f"MIXTRAN_H2O2_{case_idx:03d}",
                "mechanism": "h2o2.yaml",
                "composition_name": comp_name,
                "T": T,
                "P": P,
                "X": comp_str,
            })

    return corpus


def run_native_oracle(corpus, output_path: Path):
    """Run native Cantera oracle in WSL."""
    repo_root = Path(__file__).resolve().parent.parent
    wsl_corpus_path = to_wsl_path(output_path.parent / "corpus_mixture_transport.jsonl")
    wsl_out_path = to_wsl_path(output_path)
    wsl_script_path = to_wsl_path(repo_root / "tools" / "run_native_mixture_transport_oracle.py")

    cmd = [
        "wsl",
        "/home/nebula/cantera-oracle-env/bin/python3",
        wsl_script_path,
        wsl_corpus_path,
        wsl_out_path,
    ]
    print(f"Running native oracle in WSL: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("Native oracle finished successfully.")


def run_python_port(corpus, output_path: Path):
    """Run Python port on corpus."""
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    import cantera_python as ct

    data_dir = repo_root.parent / "cantera-original" / "data"
    results = []

    cached_phases = {}

    for case in corpus:
        mech = case["mechanism"]
        if mech not in cached_phases:
            mech_path = data_dir / mech
            phase = ct.Solution(str(mech_path))
            phase.transport_model = "mixture-averaged"
            cached_phases[mech] = phase
        else:
            phase = cached_phases[mech]

        phase.TPX = case["T"], case["P"], case["X"]

        visc = float(phase.viscosity)
        cond = float(phase.thermal_conductivity)
        sp_visc = [float(v) for v in phase.species_viscosities]
        bdiff = phase.binary_diff_coeffs.tolist()
        mdiff = [float(v) for v in phase.mix_diff_coeffs]
        mdiff_mole = [float(v) for v in phase.mix_diff_coeffs_mole]
        mdiff_mass = [float(v) for v in phase.mix_diff_coeffs_mass]

        results.append({
            "case_id": case["case_id"],
            "mechanism": case["mechanism"],
            "T": case["T"],
            "P": case["P"],
            "viscosity": visc,
            "thermal_conductivity": cond,
            "species_viscosities": sp_visc,
            "binary_diff_coeffs": bdiff,
            "mix_diff_coeffs": mdiff,
            "mix_diff_coeffs_mole": mdiff_mole,
            "mix_diff_coeffs_mass": mdiff_mass,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Python port evaluated {len(results)} cases.")


def compare_results(native_path: Path, python_path: Path, output_jsonl: Path, summary_path: Path):
    """Compare native vs python outputs and compute error distributions."""
    with open(native_path, "r", encoding="utf-8") as f:
        native_map = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    with open(python_path, "r", encoding="utf-8") as f:
        python_map = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    # Tolerances
    rtol = 1.0e-9
    atol_visc = 1.0e-10
    atol_cond = 1.0e-8
    atol_diff = 1.0e-10

    results = []
    
    visc_abs_errs = []
    visc_rel_errs = []
    cond_abs_errs = []
    cond_rel_errs = []
    spvisc_abs_errs = []
    spvisc_rel_errs = []
    bdiff_abs_errs = []
    bdiff_rel_errs = []
    mdiff_abs_errs = []
    mdiff_rel_errs = []
    mdiff_mole_abs_errs = []
    mdiff_mole_rel_errs = []
    mdiff_mass_abs_errs = []
    mdiff_mass_rel_errs = []

    passed_count = 0
    total_count = len(native_map)

    for case_id, nat in native_map.items():
        py = python_map[case_id]

        # Viscosity
        v_nat = nat["viscosity"]
        v_py = py["viscosity"]
        v_abs = abs(v_nat - v_py)
        v_rel = v_abs / max(abs(v_nat), 1.0e-15)
        visc_abs_errs.append((v_abs, case_id))
        visc_rel_errs.append((v_rel, case_id))

        # Thermal conductivity
        c_nat = nat["thermal_conductivity"]
        c_py = py["thermal_conductivity"]
        c_abs = abs(c_nat - c_py)
        c_rel = c_abs / max(abs(c_nat), 1.0e-15)
        cond_abs_errs.append((c_abs, case_id))
        cond_rel_errs.append((c_rel, case_id))

        # Species viscosities
        sv_nat = np.array(nat["species_viscosities"])
        sv_py = np.array(py["species_viscosities"])
        sv_abs = np.max(np.abs(sv_nat - sv_py))
        sv_rel = np.max(np.abs(sv_nat - sv_py) / np.maximum(np.abs(sv_nat), 1.0e-15))
        spvisc_abs_errs.append((float(sv_abs), case_id))
        spvisc_rel_errs.append((float(sv_rel), case_id))

        # Binary diff coeffs
        bd_nat = np.array(nat["binary_diff_coeffs"])
        bd_py = np.array(py["binary_diff_coeffs"])
        bd_abs = np.max(np.abs(bd_nat - bd_py))
        bd_rel = np.max(np.abs(bd_nat - bd_py) / np.maximum(np.abs(bd_nat), 1.0e-15))
        bdiff_abs_errs.append((float(bd_abs), case_id))
        bdiff_rel_errs.append((float(bd_rel), case_id))

        # Mix diff coeffs
        md_nat = np.array(nat["mix_diff_coeffs"])
        md_py = np.array(py["mix_diff_coeffs"])
        md_abs = np.max(np.abs(md_nat - md_py))
        md_rel = np.max(np.abs(md_nat - md_py) / np.maximum(np.abs(md_nat), 1.0e-15))
        mdiff_abs_errs.append((float(md_abs), case_id))
        mdiff_rel_errs.append((float(md_rel), case_id))

        # Mix diff mole
        mdm_nat = np.array(nat["mix_diff_coeffs_mole"])
        mdm_py = np.array(py["mix_diff_coeffs_mole"])
        mdm_abs = np.max(np.abs(mdm_nat - mdm_py))
        mdm_rel = np.max(np.abs(mdm_nat - mdm_py) / np.maximum(np.abs(mdm_nat), 1.0e-15))
        mdiff_mole_abs_errs.append((float(mdm_abs), case_id))
        mdiff_mole_rel_errs.append((float(mdm_rel), case_id))

        # Mix diff mass
        mds_nat = np.array(nat["mix_diff_coeffs_mass"])
        mds_py = np.array(py["mix_diff_coeffs_mass"])
        mds_abs = np.max(np.abs(mds_nat - mds_py))
        mds_rel = np.max(np.abs(mds_nat - mds_py) / np.maximum(np.abs(mds_nat), 1.0e-15))
        mdiff_mass_abs_errs.append((float(mds_abs), case_id))
        mdiff_mass_rel_errs.append((float(mds_rel), case_id))

        # Check all pass criteria
        case_passed = (
            (v_abs <= atol_visc or v_rel <= rtol) and
            (c_abs <= atol_cond or c_rel <= rtol) and
            (sv_abs <= atol_visc or sv_rel <= rtol) and
            (bd_abs <= atol_diff or bd_rel <= rtol) and
            (md_abs <= atol_diff or md_rel <= rtol) and
            (mdm_abs <= atol_diff or mdm_rel <= rtol) and
            (mds_abs <= atol_diff or mds_rel <= rtol)
        )
        if case_passed:
            passed_count += 1

        record = {
            "case_id": case_id,
            "mechanism": nat["mechanism"],
            "T": nat["T"],
            "P": nat["P"],
            "passed": bool(case_passed),
            "viscosity_error": {"abs": v_abs, "rel": v_rel},
            "thermal_conductivity_error": {"abs": c_abs, "rel": c_rel},
            "species_viscosities_error": {"abs": float(sv_abs), "rel": float(sv_rel)},
            "binary_diff_error": {"abs": float(bd_abs), "rel": float(bd_rel)},
            "mix_diff_error": {"abs": float(md_abs), "rel": float(md_rel)},
            "mix_diff_mole_error": {"abs": float(mdm_abs), "rel": float(mdm_rel)},
            "mix_diff_mass_error": {"abs": float(mds_abs), "rel": float(mds_rel)},
        }
        results.append(record)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def calc_stats(arr):
        vals = [x[0] for x in arr]
        worst_case = max(arr, key=lambda x: x[0])[1]
        return {
            "max": float(np.max(vals)),
            "p50": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "worst_case_id": worst_case,
        }

    summary = {
        "total_cases": total_count,
        "passed_cases": passed_count,
        "pass_rate_percent": (passed_count / total_count) * 100.0,
        "classification": "NUMERICALLY_EQUIVALENT_WITHIN_VALIDATED_TOLERANCE" if passed_count == total_count else "FAILED",
        "tolerances": {
            "relative_tolerance": rtol,
            "absolute_tolerance_viscosity": atol_visc,
            "absolute_tolerance_thermal_conductivity": atol_cond,
            "absolute_tolerance_diffusion": atol_diff,
        },
        "statistics": {
            "viscosity": {
                "absolute_error": calc_stats(visc_abs_errs),
                "relative_error": calc_stats(visc_rel_errs),
            },
            "thermal_conductivity": {
                "absolute_error": calc_stats(cond_abs_errs),
                "relative_error": calc_stats(cond_rel_errs),
            },
            "species_viscosities": {
                "absolute_error": calc_stats(spvisc_abs_errs),
                "relative_error": calc_stats(spvisc_rel_errs),
            },
            "binary_diffusion_coefficients": {
                "absolute_error": calc_stats(bdiff_abs_errs),
                "relative_error": calc_stats(bdiff_rel_errs),
            },
            "mixture_diffusion_coefficients": {
                "absolute_error": calc_stats(mdiff_abs_errs),
                "relative_error": calc_stats(mdiff_rel_errs),
            },
            "mixture_diffusion_mole": {
                "absolute_error": calc_stats(mdiff_mole_abs_errs),
                "relative_error": calc_stats(mdiff_mole_rel_errs),
            },
            "mixture_diffusion_mass": {
                "absolute_error": calc_stats(mdiff_mass_abs_errs),
                "relative_error": calc_stats(mdiff_mass_rel_errs),
            },
        }
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"=== DIFFERENTIAL VALIDATION COMPLETE ===")
    print(f"Total cases: {total_count}")
    print(f"Passed: {passed_count} ({summary['pass_rate_percent']:.2f}%)")
    print(f"Max Viscosity Rel Err: {summary['statistics']['viscosity']['relative_error']['max']:.3e}")
    print(f"Max Thermal Cond Rel Err: {summary['statistics']['thermal_conductivity']['relative_error']['max']:.3e}")
    print(f"Max Binary Diff Rel Err: {summary['statistics']['binary_diffusion_coefficients']['relative_error']['max']:.3e}")
    print(f"Max Mix Diff Rel Err: {summary['statistics']['mixture_diffusion_coefficients']['relative_error']['max']:.3e}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = audit_dir / "corpus_mixture_transport.jsonl"
    native_out_path = audit_dir / "raw_native_mixture_transport.jsonl"
    python_out_path = audit_dir / "raw_python_mixture_transport.jsonl"
    results_jsonl = audit_dir / "mixture_transport_differential_results.jsonl"
    summary_path = audit_dir / "mixture_transport_differential_summary.json"

    print("Step 1: Generating mixture transport test corpus...")
    corpus = build_test_corpus()
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(corpus)} test cases to {corpus_path}")

    print("Step 2: Running native Cantera oracle in WSL...")
    run_native_oracle(corpus, native_out_path)

    print("Step 3: Running Python port runner...")
    run_python_port(corpus, python_out_path)

    print("Step 4: Comparing native vs Python outputs...")
    compare_results(native_out_path, python_out_path, results_jsonl, summary_path)

if __name__ == "__main__":
    main()
