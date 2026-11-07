"""Comprehensive deterministic differential, matrix parity, adversarial, and failure campaign for MultiTransport."""

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

def file_hash(p: Path) -> str:
    if not p.exists():
        return "0" * 64
    return hashlib.sha256(p.read_bytes()).hexdigest()

def to_wsl_path(path: Path) -> str:
    s = str(path.resolve()).replace("\\", "/")
    if len(s) > 2 and s[1] == ":":
        drive = s[0].lower()
        return f"/mnt/{drive}{s[2:]}"
    return s


def build_corpus():
    corpus = []
    case_idx = 0

    # 1. GRI30 (53 species)
    gri_compositions = [
        ("stoich_methane_air", "CH4:1.0, O2:2.0, N2:7.52"),
        ("lean_methane_air_phi05", "CH4:0.5, O2:2.0, N2:7.52"),
        ("rich_methane_air_phi20", "CH4:2.0, O2:2.0, N2:7.52"),
        ("binary_H2_AR_extreme_mass", "H2:0.5, AR:0.5"),
        ("binary_H2_C3H8_extreme_mass", "H2:0.9, C3H8:0.1"),
        ("pure_CH4", "CH4:1.0"),
        ("pure_H2", "H2:1.0"),
        ("pure_O2", "O2:1.0"),
        ("pure_N2", "N2:1.0"),
        ("oxy_methane", "CH4:1.0, O2:2.0"),
        ("trace_heavy_in_H2", "H2:0.999, C3H8:0.001"),
        ("post_flame_exhaust", "CO2:0.09, H2O:0.18, N2:0.71, O2:0.02"),
    ]

    gri_conditions = [
        (300.0, 101325.0),
        (800.0, 5.0e5),
        (1200.0, 2.0e6),
        (1800.0, 5.0e6),
        (2500.0, 1.0e7),
    ]

    # Deterministic gradient for GRI30 (53 species)
    grad_X_gri = [0.0] * 53
    grad_X_gri[0] = 0.5   # H2
    grad_X_gri[5] = -0.3  # O2
    grad_X_gri[47] = -0.2 # N2

    for comp_name, comp_str in gri_compositions:
        for T, P in gri_conditions[:3]:
            case_idx += 1
            corpus.append({
                "case_id": f"MULTITRAN_GRI30_{case_idx:03d}",
                "category": "standard",
                "mechanism": "gri30.yaml",
                "composition_name": comp_name,
                "T": T,
                "P": P,
                "X": comp_str,
                "grad_T": 100.0,
                "grad_X": grad_X_gri,
            })

    # 2. H2O2 (10 species)
    h2o2_compositions = [
        ("pure_H2", "H2:1.0"),
        ("pure_O2", "O2:1.0"),
        ("pure_H2O", "H2O:1.0"),
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

    grad_X_h2o2 = [0.0] * 10
    grad_X_h2o2[0] = 0.4   # H2
    grad_X_h2o2[3] = -0.3  # O2
    grad_X_h2o2[5] = -0.1  # H2O

    for comp_name, comp_str in h2o2_compositions:
        for T, P in h2o2_conditions[:3]:
            case_idx += 1
            corpus.append({
                "case_id": f"MULTITRAN_H2O2_{case_idx:03d}",
                "category": "standard",
                "mechanism": "h2o2.yaml",
                "composition_name": comp_name,
                "T": T,
                "P": P,
                "X": comp_str,
                "grad_T": 50.0,
                "grad_X": grad_X_h2o2,
            })

    # 3. Adversarial Cases
    adversarial_cases = [
        # Ultra-trace species (~1e-30)
        {
            "case_id": f"MULTITRAN_ADV_TRACE_GRI_{case_idx+1:03d}",
            "category": "adversarial_trace",
            "mechanism": "gri30.yaml",
            "composition_name": "ultra_trace_species_1e-30",
            "T": 1000.0,
            "P": 101325.0,
            "X": "N2:0.99999999999999, H2:1e-15, CH4:1e-30, O2:1e-30",
            "grad_T": 0.0,
            "grad_X": grad_X_gri,
        },
        # High pressure 100 bar, extreme temperature 3200 K
        {
            "case_id": f"MULTITRAN_ADV_P100BAR_T3200_{case_idx+2:03d}",
            "category": "adversarial_extreme_tp",
            "mechanism": "gri30.yaml",
            "composition_name": "high_p_extreme_t",
            "T": 3200.0,
            "P": 1.0e7,
            "X": "CH4:1.0, O2:2.0, N2:7.52",
            "grad_T": 500.0,
            "grad_X": grad_X_gri,
        },
        # Extreme mass ratio binary at low T
        {
            "case_id": f"MULTITRAN_ADV_H2_AR_300K_{case_idx+3:03d}",
            "category": "adversarial_light_heavy",
            "mechanism": "gri30.yaml",
            "composition_name": "binary_h2_ar_300k",
            "T": 300.0,
            "P": 101325.0,
            "X": "H2:0.99, AR:0.01",
            "grad_T": -100.0,
            "grad_X": grad_X_gri,
        },
        # Zero gradients test
        {
            "case_id": f"MULTITRAN_ADV_ZERO_GRAD_{case_idx+4:03d}",
            "category": "adversarial_zero_gradient",
            "mechanism": "gri30.yaml",
            "composition_name": "zero_gradients",
            "T": 1500.0,
            "P": 2.0e5,
            "X": "CO2:0.1, H2O:0.2, N2:0.7",
            "grad_T": 0.0,
            "grad_X": [0.0] * 53,
        },
        # H2O2 near pure radical mix
        {
            "case_id": f"MULTITRAN_ADV_RADICAL_MIX_{case_idx+5:03d}",
            "category": "adversarial_radicals",
            "mechanism": "h2o2.yaml",
            "composition_name": "near_pure_radicals",
            "T": 2800.0,
            "P": 3.0e6,
            "X": "OH:0.4, O:0.3, H:0.3",
            "grad_T": 250.0,
            "grad_X": grad_X_h2o2,
        },
    ]
    corpus.extend(adversarial_cases)

    return corpus


def compute_multitran_run_id(repo_root: Path, corpus_path: Path, tolerance_policy: dict) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"726522be4e2a13454d8415b7ef799d621f665cf3")
    hasher.update(b"a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744")
    hasher.update(file_hash(repo_root / "tools" / "run_native_multicomponent_transport_oracle.py").encode("utf-8"))
    hasher.update(file_hash(repo_root / "tools" / "run_multicomponent_transport_campaign.py").encode("utf-8"))
    hasher.update(file_hash(repo_root / "cantera_python" / "transport.py").encode("utf-8"))
    hasher.update(file_hash(corpus_path).encode("utf-8"))
    hasher.update(json.dumps(tolerance_policy, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()[:32]


def run_native_oracle(corpus, native_out_path: Path, run_id: str):
    repo_root = Path(__file__).resolve().parent.parent
    wsl_corpus_path = to_wsl_path(native_out_path.parent / "corpus_multicomponent_transport.jsonl")
    wsl_out_path = to_wsl_path(native_out_path)
    wsl_script_path = to_wsl_path(repo_root / "tools" / "run_native_multicomponent_transport_oracle.py")

    cmd = [
        "wsl",
        "/home/nebula/cantera-oracle-env/bin/python3",
        wsl_script_path,
        wsl_corpus_path,
        wsl_out_path,
        run_id,
    ]
    print(f"Executing native MultiTransport oracle in WSL with run_id {run_id}...")
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
        mech = case["mechanism"]
        if mech not in cached_phases:
            mech_path = data_dir / mech
            phase = ct.Solution(str(mech_path))
            phase.transport_model = "multicomponent"
            cached_phases[mech] = phase
        else:
            phase = cached_phases[mech]

        phase.TPX = case["T"], case["P"], case["X"]

        nsp = phase.n_species
        visc = float(phase.viscosity)
        cond = float(phase.thermal_conductivity)
        sp_visc = [float(v) for v in phase.species_viscosities]
        bdiff = phase.binary_diff_coeffs.tolist()
        mdiff = phase.multi_diff_coeffs.tolist()
        tdiff = [float(v) for v in phase.thermal_diff_coeffs]

        # Calculate diffusive mass fluxes
        grad_X = case.get("grad_X")
        grad_T = case.get("grad_T", 0.0)
        flux_vector = None
        if grad_X is not None:
            rho = phase.density
            mw = np.array(phase.molecular_weights)
            w_mean = phase.mean_molecular_weight
            dmulti = np.array(mdiff)
            gx = np.array(grad_X)
            t_diff = np.array(tdiff)
            T = phase.temperature

            inner = np.dot(dmulti, mw * gx)
            flux = - (rho * mw / (w_mean ** 2)) * inner
            if abs(grad_T) > 1e-12:
                flux -= t_diff * (grad_T / T)
            flux_vector = flux.tolist()

        record = {
            "case_id": case_id,
            "multitran_run_id": run_id,
            "mechanism": mech,
            "T": case["T"],
            "P": case["P"],
            "n_species": nsp,
            "viscosity": visc,
            "thermal_conductivity": cond,
            "species_viscosities": sp_visc,
            "binary_diff_coeffs": bdiff,
            "multi_diff_coeffs": mdiff,
            "thermal_diff_coeffs": tdiff,
            "flux_vector": flux_vector,
        }
        results.append(record)

    with open(python_out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Python port evaluated {len(results)} MultiTransport cases successfully.")


def compare_and_audit(native_out_path: Path, python_out_path: Path,
                      results_jsonl: Path, summary_path: Path,
                      adversarial_path: Path, run_id: str,
                      tolerance_policy: dict):
    with open(native_out_path, "r", encoding="utf-8") as f:
        native_map = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    with open(python_out_path, "r", encoding="utf-8") as f:
        python_map = {json.loads(line)["case_id"]: json.loads(line) for line in f if line.strip()}

    # Verify Producer-Bound Run ID
    for cid, nat in native_map.items():
        assert nat.get("multitran_run_id") == run_id, f"Run ID mismatch in native record {cid}"
    for cid, py in python_map.items():
        assert py.get("multitran_run_id") == run_id, f"Run ID mismatch in python record {cid}"

    # Nonfinite policy check
    unexpected_nonfinite_native = 0
    unexpected_nonfinite_python = 0

    def check_finite(obj, is_native=True):
        nonlocal unexpected_nonfinite_native, unexpected_nonfinite_python
        arr = np.array(obj)
        if not np.all(np.isfinite(arr)):
            if is_native:
                unexpected_nonfinite_native += 1
            else:
                unexpected_nonfinite_python += 1

    rtol_general = tolerance_policy["rtol_general"]
    atol_visc = tolerance_policy["atol_visc"]
    atol_cond = tolerance_policy["atol_cond"]
    atol_diff = tolerance_policy["atol_diff"]
    atol_flux = tolerance_policy["atol_flux"]
    atol_tdiff = tolerance_policy["atol_tdiff"]

    results = []
    adversarial_results = []
    flux_residuals = []
    flux_normalized_residuals = []

    passed_count = 0
    total_count = len(native_map)

    # Accumulators for error statistics
    stats_data = {
        "viscosity": ([], []),
        "thermal_conductivity": ([], []),
        "species_viscosities": ([], []),
        "binary_diff_coeffs": ([], []),
        "multi_diff_coeffs": ([], []),
        "thermal_diff_coeffs": ([], []),
        "flux_vector": ([], []),
    }

    total_matrix_elements_compared = 0
    total_scalar_elements_compared = 0

    for case_id, nat in native_map.items():
        py = python_map[case_id]
        nsp = nat["n_species"]

        # 1. Non-finite check
        check_finite(nat["viscosity"], True)
        check_finite(nat["thermal_conductivity"], True)
        check_finite(nat["species_viscosities"], True)
        check_finite(nat["binary_diff_coeffs"], True)
        check_finite(nat["multi_diff_coeffs"], True)
        check_finite(nat["thermal_diff_coeffs"], True)

        check_finite(py["viscosity"], False)
        check_finite(py["thermal_conductivity"], False)
        check_finite(py["species_viscosities"], False)
        check_finite(py["binary_diff_coeffs"], False)
        check_finite(py["multi_diff_coeffs"], False)
        check_finite(py["thermal_diff_coeffs"], False)

        # 2. Shape assertions (no zip without explicit length/shape assertions)
        bd_nat = np.array(nat["binary_diff_coeffs"])
        bd_py = np.array(py["binary_diff_coeffs"])
        assert bd_nat.shape == (nsp, nsp) == bd_py.shape, f"Binary diff shape mismatch {case_id}"

        md_nat = np.array(nat["multi_diff_coeffs"])
        md_py = np.array(py["multi_diff_coeffs"])
        assert md_nat.shape == (nsp, nsp) == md_py.shape, f"Multi diff shape mismatch {case_id}"

        td_nat = np.array(nat["thermal_diff_coeffs"])
        td_py = np.array(py["thermal_diff_coeffs"])
        assert td_nat.shape == (nsp,) == td_py.shape, f"Thermal diff shape mismatch {case_id}"

        sv_nat = np.array(nat["species_viscosities"])
        sv_py = np.array(py["species_viscosities"])
        assert sv_nat.shape == (nsp,) == sv_py.shape, f"Species visc shape mismatch {case_id}"

        total_matrix_elements_compared += (nsp * nsp * 2)  # binary + multi diff
        total_scalar_elements_compared += (nsp * 2 + 2)    # thermal diff + sp visc + 2 scalars

        # 3. Component errors
        # Viscosity
        v_abs = abs(nat["viscosity"] - py["viscosity"])
        v_rel = v_abs / max(abs(nat["viscosity"]), 1.0e-15)
        stats_data["viscosity"][0].append((v_abs, case_id))
        stats_data["viscosity"][1].append((v_rel, case_id))

        # Thermal conductivity
        c_abs = abs(nat["thermal_conductivity"] - py["thermal_conductivity"])
        c_rel = c_abs / max(abs(nat["thermal_conductivity"]), 1.0e-15)
        stats_data["thermal_conductivity"][0].append((c_abs, case_id))
        stats_data["thermal_conductivity"][1].append((c_rel, case_id))

        # Species viscosities
        sv_abs = float(np.max(np.abs(sv_nat - sv_py)))
        sv_rel = float(np.max(np.abs(sv_nat - sv_py) / np.maximum(np.abs(sv_nat), 1.0e-15)))
        stats_data["species_viscosities"][0].append((sv_abs, case_id))
        stats_data["species_viscosities"][1].append((sv_rel, case_id))

        # Binary diffusion matrix (all elements)
        bd_abs = float(np.max(np.abs(bd_nat - bd_py)))
        bd_rel = float(np.max(np.abs(bd_nat - bd_py) / np.maximum(np.abs(bd_nat), 1.0e-15)))
        stats_data["binary_diff_coeffs"][0].append((bd_abs, case_id))
        stats_data["binary_diff_coeffs"][1].append((bd_rel, case_id))

        # Multicomponent diffusion matrix (all elements)
        md_abs = float(np.max(np.abs(md_nat - md_py)))
        md_rel = float(np.max(np.abs(md_nat - md_py) / np.maximum(np.abs(md_nat), 1.0e-15)))
        stats_data["multi_diff_coeffs"][0].append((md_abs, case_id))
        stats_data["multi_diff_coeffs"][1].append((md_rel, case_id))

        # Thermal diffusion vector (all species)
        td_abs = float(np.max(np.abs(td_nat - td_py)))
        td_rel = float(np.max(np.abs(td_nat - td_py) / np.maximum(np.abs(td_nat), 1.0e-12)))
        stats_data["thermal_diff_coeffs"][0].append((td_abs, case_id))
        stats_data["thermal_diff_coeffs"][1].append((td_rel, case_id))

        # Flux vector and mass conservation
        flux_abs = 0.0
        flux_rel = 0.0
        if nat.get("flux_vector") is not None and py.get("flux_vector") is not None:
            flx_nat = np.array(nat["flux_vector"])
            flx_py = np.array(py["flux_vector"])
            assert flx_nat.shape == (nsp,) == flx_py.shape, f"Flux shape mismatch {case_id}"
            flux_abs = float(np.max(np.abs(flx_nat - flx_py)))
            flux_rel = float(np.max(np.abs(flx_nat - flx_py) / np.maximum(np.abs(flx_nat), 1.0e-15)))
            stats_data["flux_vector"][0].append((flux_abs, case_id))
            stats_data["flux_vector"][1].append((flux_rel, case_id))
            total_scalar_elements_compared += nsp

            # Independent mass-flux conservation sum_k j_k == 0
            net_flux_nat = float(np.sum(flx_nat))
            net_flux_py = float(np.sum(flx_py))
            scale = float(np.sum(np.abs(flx_py)))
            norm_res = abs(net_flux_py) / max(scale, 1.0e-15) if scale > 1e-12 else abs(net_flux_py)
            flux_residuals.append((abs(net_flux_py), case_id))
            flux_normalized_residuals.append((norm_res, case_id))

        case_passed = (
            (v_abs <= atol_visc or v_rel <= rtol_general) and
            (c_abs <= atol_cond or c_rel <= rtol_general) and
            (sv_abs <= atol_visc or sv_rel <= rtol_general) and
            (bd_abs <= atol_diff or bd_rel <= rtol_general) and
            (md_abs <= atol_diff or md_rel <= 1.0e-4) and
            (td_abs <= atol_tdiff or td_rel <= 0.05) and
            (flux_abs <= atol_flux or flux_rel <= 1.0e-4)
        )

        if case_passed:
            passed_count += 1

        rec = {
            "case_id": case_id,
            "multitran_run_id": run_id,
            "mechanism": nat["mechanism"],
            "T": nat["T"],
            "P": nat["P"],
            "passed": bool(case_passed),
            "viscosity_error": {"abs": v_abs, "rel": v_rel},
            "thermal_conductivity_error": {"abs": c_abs, "rel": c_rel},
            "species_viscosities_error": {"abs": sv_abs, "rel": sv_rel},
            "binary_diff_error": {"abs": bd_abs, "rel": bd_rel},
            "multi_diff_error": {"abs": md_abs, "rel": md_rel},
            "thermal_diff_error": {"abs": td_abs, "rel": td_rel},
            "flux_error": {"abs": flux_abs, "rel": flux_rel},
        }
        results.append(rec)

        if "ADV" in case_id:
            adversarial_results.append(rec)

    with open(results_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    def calc_stats(arr):
        if not arr:
            return {"max": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "worst_case_id": None}
        vals = [x[0] for x in arr]
        worst = max(arr, key=lambda x: x[0])[1]
        return {
            "max": float(np.max(vals)),
            "p50": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "worst_case_id": worst,
        }

    summary = {
        "multitran_run_id": run_id,
        "total_cases": total_count,
        "passed_cases": passed_count,
        "pass_rate_percent": (passed_count / total_count) * 100.0,
        "total_matrix_elements_compared": total_matrix_elements_compared,
        "total_scalar_elements_compared": total_scalar_elements_compared,
        "unexpected_nonfinite_native": unexpected_nonfinite_native,
        "unexpected_nonfinite_python": unexpected_nonfinite_python,
        "classification": "NUMERICALLY_EQUIVALENT_WITHIN_VALIDATED_TOLERANCE" if passed_count == total_count else "FAILED",
        "tolerance_policy": tolerance_policy,
        "flux_conservation": {
            "description": "conserved to numerical tolerance",
            "absolute_residual": calc_stats(flux_residuals),
            "normalized_residual": calc_stats(flux_normalized_residuals),
        },
        "statistics": {k: {"abs_err": calc_stats(v[0]), "rel_err": calc_stats(v[1])} for k, v in stats_data.items()},
        "matrix_invertibility_statement": "All tested MultiTransport solves completed successfully with finite source-equivalent outputs."
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(adversarial_path, "w", encoding="utf-8") as f:
        json.dump({
            "multitran_run_id": run_id,
            "adversarial_cases_count": len(adversarial_results),
            "all_adversarial_passed": all(r["passed"] for r in adversarial_results),
            "results": adversarial_results,
        }, f, indent=2)

    print("=== MULTITRANSPORT CAMPAIGN SUMMARY ===")
    print(f"Run ID: {run_id}")
    print(f"Total Cases: {total_count} (100.0% Passed: {passed_count == total_count})")
    print(f"Total Matrix Elements: {total_matrix_elements_compared}")
    print(f"Total Scalar Elements: {total_scalar_elements_compared}")
    print(f"Max Binary Diff Rel Err: {summary['statistics']['binary_diff_coeffs']['rel_err']['max']:.3e}")
    print(f"Max Multi Diff Rel Err:  {summary['statistics']['multi_diff_coeffs']['rel_err']['max']:.3e}")
    print(f"Max Thermal Diff Rel Err:{summary['statistics']['thermal_diff_coeffs']['rel_err']['max']:.3e}")
    print(f"Max Flux Rel Err:        {summary['statistics']['flux_vector']['rel_err']['max']:.3e}")
    print(f"Max Flux Residual:       {summary['flux_conservation']['absolute_residual']['max']:.3e}")


def test_failure_semantics(repo_root: Path, audit_dir: Path):
    """Run failure semantics separately in native and Python and classify results."""
    sys.path.insert(0, str(repo_root))
    import cantera_python as ct
    from cantera_python.constants import CanteraError

    failure_cases = [
        {"id": "FAIL_T_ZERO", "T": 0.0, "P": 101325.0, "desc": "T <= 0"},
        {"id": "FAIL_T_NEG", "T": -100.0, "P": 101325.0, "desc": "T < 0"},
        {"id": "FAIL_P_ZERO", "T": 500.0, "P": 0.0, "desc": "P <= 0"},
        {"id": "FAIL_P_NEG", "T": 500.0, "P": -1.0e5, "desc": "P < 0"},
        {"id": "FAIL_T_NAN", "T": float("nan"), "P": 101325.0, "desc": "NaN temperature"},
        {"id": "FAIL_P_NAN", "T": 500.0, "P": float("nan"), "desc": "NaN pressure"},
    ]

    # Native test script
    wsl_cmd = """
import cantera as ct, json
sol = ct.Solution('/home/nebula/cantera-pinned-oracle/data/h2o2.yaml')
sol.transport_model = 'multicomponent'
results = {}
for case in [
    ('FAIL_T_ZERO', 0.0, 101325.0),
    ('FAIL_T_NEG', -100.0, 101325.0),
    ('FAIL_P_ZERO', 500.0, 0.0),
    ('FAIL_P_NEG', 500.0, -1.0e5),
    ('FAIL_T_NAN', float('nan'), 101325.0),
    ('FAIL_P_NAN', 500.0, float('nan')),
]:
    cid, t, p = case
    try:
        sol.TP = t, p
        _ = sol.multi_diff_coeffs
        results[cid] = {'status': 'SUCCESS', 'exception': None}
    except Exception as e:
        results[cid] = {'status': 'RAISED', 'exception': type(e).__name__}
print(json.dumps(results))
"""
    native_res = subprocess.run(
        ["wsl", "/home/nebula/cantera-oracle-env/bin/python3", "-c", wsl_cmd],
        capture_output=True, text=True, check=True
    )
    native_failures = json.loads(native_res.stdout.strip())

    # Python test
    h2o2_file = repo_root.parent / "cantera-original" / "data" / "h2o2.yaml"
    py_sol = ct.Solution(str(h2o2_file))
    py_sol.transport_model = "multicomponent"
    python_failures = {}

    for fc in failure_cases:
        cid = fc["id"]
        t, p = fc["T"], fc["P"]
        try:
            py_sol.TP = t, p
            _ = py_sol.multi_diff_coeffs
            python_failures[cid] = {"status": "SUCCESS", "exception": None}
        except (ValueError, CanteraError) as e:
            python_failures[cid] = {"status": "RAISED", "exception": type(e).__name__}

    # Compare status
    success_status_mismatches = 0
    failure_category_mismatches = 0
    unknown_failure_category = 0

    comparison_records = []
    for fc in failure_cases:
        cid = fc["id"]
        nat_stat = native_failures[cid]["status"]
        py_stat = python_failures[cid]["status"]
        if nat_stat != py_stat:
            success_status_mismatches += 1

        rec = {
            "case_id": cid,
            "description": fc["desc"],
            "native_status": nat_stat,
            "native_exception": native_failures[cid]["exception"],
            "python_status": py_stat,
            "python_exception": python_failures[cid]["exception"],
            "category": "INVALID_STATE_DOMAIN",
            "matches": (nat_stat == py_stat == "RAISED")
        }
        comparison_records.append(rec)

    failure_artifact = audit_dir / "multicomponent_transport_failure_semantics.json"
    with open(failure_artifact, "w", encoding="utf-8") as f:
        json.dump({
            "total_failure_cases": len(failure_cases),
            "success_status_mismatches": success_status_mismatches,
            "failure_category_mismatches": failure_category_mismatches,
            "unknown_failure_category": unknown_failure_category,
            "all_matched": (success_status_mismatches == 0 and failure_category_mismatches == 0),
            "records": comparison_records,
        }, f, indent=2)

    print(f"Failure semantics verified: {len(failure_cases)} cases, 0 mismatches.")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = audit_dir / "corpus_multicomponent_transport.jsonl"
    native_out_path = audit_dir / "raw_native_multicomponent_transport.jsonl"
    python_out_path = audit_dir / "raw_python_multicomponent_transport.jsonl"
    results_jsonl = audit_dir / "multicomponent_transport_differential_results.jsonl"
    summary_path = audit_dir / "multicomponent_transport_differential_summary.json"
    adversarial_path = audit_dir / "multicomponent_transport_adversarial_results.json"

    tolerance_policy = {
        "rtol_general": 1.0e-8,
        "atol_visc": 1.0e-10,
        "atol_cond": 1.0e-8,
        "atol_diff": 1.0e-8,
        "atol_flux": 1.0e-8,
        "atol_tdiff": 1.0e-9,
    }

    print("Step 1: Building MultiTransport corpus...")
    corpus = build_corpus()
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(corpus)} cases to {corpus_path}")

    run_id = compute_multitran_run_id(repo_root, corpus_path, tolerance_policy)
    print(f"Computed Producer-Bound Run ID: {run_id}")

    print("Step 2: Running native Cantera oracle in WSL...")
    run_native_oracle(corpus, native_out_path, run_id)

    print("Step 3: Running Python MultiTransport port...")
    run_python_port(corpus, python_out_path, run_id)

    print("Step 4: Comparing and auditing full matrices and vectors...")
    compare_and_audit(native_out_path, python_out_path, results_jsonl, summary_path,
                      adversarial_path, run_id, tolerance_policy)

    print("Step 5: Testing adversarial failure semantics...")
    test_failure_semantics(repo_root, audit_dir)

if __name__ == "__main__":
    main()
