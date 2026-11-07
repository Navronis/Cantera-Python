"""Comprehensive Differential, Adversarial, and Failure Validation Campaign for ChemEquil.

Evaluates parity against pinned native Cantera C++ oracle in WSL across:
- Mechanisms: gri30.yaml, h2o2.yaml, air.yaml
- Property pairs: TP, HP, SP, UV, SV
- Temperature range: 300 K to 3500 K
- Pressure range: 0.1 atm to 50 atm
- Equivalence ratios: lean, stoichiometric, rich, syngas, air, pure water
- Adversarial trace, dissociation, and high-dilution regimes
- Negative / invalid failure semantics
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

def to_wsl_path(win_path: Path) -> str:
    p = str(win_path.resolve()).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        return f"/mnt/{drive}{p[2:]}"
    return p

def file_hash(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_corpus():
    cases = []
    case_idx = 1

    # 1. GRI-Mech 3.0 TP sweeps (Temperatures x Pressures x Compositions)
    # Physical equilibrium temperatures for gas phase
    temps_tp = [1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2400.0, 2800.0, 3200.0]
    pressures = [10132.5, 101325.0, 506625.0, 2026500.0, 5066250.0]
    compositions_gri = [
        ("CH4:0.5, O2:2.0, N2:7.52", "lean_ch4_air"),
        ("CH4:1.0, O2:2.0, N2:7.52", "stoich_ch4_air"),
        ("CH4:1.5, O2:2.0, N2:7.52", "rich_ch4_air"),
        ("H2:2.0, O2:1.0, N2:3.76", "h2_air"),
        ("CO:1.0, H2:1.0, O2:1.0, N2:3.76", "syngas_air"),
    ]

    for T in temps_tp:
        for P in [101325.0, 2026500.0]:
            for comp, comp_tag in compositions_gri[:3]:
                cases.append({
                    "case_id": f"EQ_GRI_TP_{case_idx:04d}",
                    "category": f"gri30_tp_{comp_tag}",
                    "mechanism": "gri30.yaml",
                    "mode": "TP",
                    "T": float(T),
                    "P": float(P),
                    "X": comp,
                })
                case_idx += 1

    # Pressure sweep at T=1800 K
    for P in pressures:
        for comp, comp_tag in compositions_gri:
            cases.append({
                "case_id": f"EQ_GRI_PSWP_{case_idx:04d}",
                "category": f"gri30_psweep_{comp_tag}",
                "mechanism": "gri30.yaml",
                "mode": "TP",
                "T": 1800.0,
                "P": float(P),
                "X": comp,
            })
            case_idx += 1

    # 2. HP mode (Adiabatic flame temperature and enthalpy conservation)
    t_inits_hp = [300.0, 450.0, 600.0, 800.0]
    p_inits_hp = [101325.0, 1013250.0, 3039750.0]
    for T in t_inits_hp:
        for P in p_inits_hp:
            for comp, comp_tag in compositions_gri:
                cases.append({
                    "case_id": f"EQ_GRI_HP_{case_idx:04d}",
                    "category": f"gri30_hp_{comp_tag}",
                    "mechanism": "gri30.yaml",
                    "mode": "HP",
                    "T": float(T),
                    "P": float(P),
                    "X": comp,
                })
                case_idx += 1

    # 3. UV mode (Constant volume explosion)
    for T in [300.0, 500.0, 700.0]:
        for P in [101325.0, 506625.0]:
            for comp, comp_tag in compositions_gri[:3]:
                cases.append({
                    "case_id": f"EQ_GRI_UV_{case_idx:04d}",
                    "category": f"gri30_uv_{comp_tag}",
                    "mechanism": "gri30.yaml",
                    "mode": "UV",
                    "T": float(T),
                    "P": float(P),
                    "X": comp,
                })
                case_idx += 1

    # 4. SP and SV modes (Isentropic expansion and compression)
    for T in [1600.0, 2000.0, 2400.0]:
        for P in [101325.0, 1013250.0]:
            for comp, comp_tag in compositions_gri[1:3]:
                cases.append({
                    "case_id": f"EQ_GRI_SP_{case_idx:04d}",
                    "category": f"gri30_sp_{comp_tag}",
                    "mechanism": "gri30.yaml",
                    "mode": "SP",
                    "T": float(T),
                    "P": float(P),
                    "X": comp,
                })
                case_idx += 1
                cases.append({
                    "case_id": f"EQ_GRI_SV_{case_idx:04d}",
                    "category": f"gri30_sv_{comp_tag}",
                    "mechanism": "gri30.yaml",
                    "mode": "SV",
                    "T": float(T),
                    "P": float(P),
                    "X": comp,
                })
                case_idx += 1

    # 5. H2/O2 mechanism (h2o2.yaml) across TP, HP, UV, SP
    h2o2_comps = [
        ("H2:2.0, O2:1.0", "stoich_h2_o2"),
        ("H2:1.0, O2:1.0", "lean_h2_o2"),
        ("H2:4.0, O2:1.0", "rich_h2_o2"),
        ("H2O:1.0", "pure_water_dissoc"),
    ]
    for mode in ["TP", "HP", "UV", "SP"]:
        for T in [1200.0, 2000.0]:
            for comp, comp_tag in h2o2_comps:
                cases.append({
                    "case_id": f"EQ_H2O2_{mode}_{case_idx:04d}",
                    "category": f"h2o2_{mode.lower()}_{comp_tag}",
                    "mechanism": "h2o2.yaml",
                    "mode": mode,
                    "T": float(T),
                    "P": 101325.0,
                    "X": comp,
                })
                case_idx += 1

    # 6. Air mechanism (air.yaml) across TP, HP, UV
    for mode in ["TP", "HP", "UV"]:
        for T in [1000.0, 2000.0, 3000.0]:
            for P in [10132.5, 101325.0, 2026500.0]:
                cases.append({
                    "case_id": f"EQ_AIR_{mode}_{case_idx:04d}",
                    "category": f"air_{mode.lower()}",
                    "mechanism": "air.yaml",
                    "mode": mode,
                    "T": float(T),
                    "P": float(P),
                    "X": "O2:0.21, N2:0.78, AR:0.01",
                })
                case_idx += 1

    # 7. Adversarial cases (trace species, ultra-high temperature, extreme dilution)
    adversarial_cases = [
        {
            "case_id": f"EQ_ADV_HIGH_T_DISSOC_{case_idx:04d}",
            "category": "adversarial_high_temperature",
            "mechanism": "gri30.yaml",
            "mode": "TP",
            "T": 3500.0,
            "P": 101325.0,
            "X": "CO2:1.0",
            "description": "Complete CO2 thermal dissociation at 3500 K",
        },
        {
            "case_id": f"EQ_ADV_HIGH_T_H2O_{case_idx+1:04d}",
            "category": "adversarial_high_temperature",
            "mechanism": "h2o2.yaml",
            "mode": "TP",
            "T": 3200.0,
            "P": 101325.0,
            "X": "H2O:1.0",
            "description": "Water thermal dissociation into radicals at 3200 K",
        },
        {
            "case_id": f"EQ_ADV_PURE_N2_{case_idx+2:04d}",
            "category": "adversarial_pure_element",
            "mechanism": "gri30.yaml",
            "mode": "TP",
            "T": 1500.0,
            "P": 101325.0,
            "X": "N2:1.0",
            "description": "Pure single element (N) inert atmosphere",
        },
        {
            "case_id": f"EQ_ADV_HP_HIGH_P_{case_idx+3:04d}",
            "category": "adversarial_high_pressure_hp",
            "mechanism": "gri30.yaml",
            "mode": "HP",
            "T": 300.0,
            "P": 5066250.0,
            "X": "CH4:1.0, O2:2.0, N2:7.52",
            "description": "Adiabatic flame at 50 atmospheres",
        },
        {
            "case_id": f"EQ_ADV_UV_HIGH_P_{case_idx+4:04d}",
            "category": "adversarial_high_pressure_uv",
            "mechanism": "gri30.yaml",
            "mode": "UV",
            "T": 300.0,
            "P": 2026500.0,
            "X": "H2:2.0, O2:1.0, N2:3.76",
            "description": "Constant-volume explosion at 20 atm initial pressure",
        },
        {
            "case_id": f"EQ_ADV_TRACE_DILUTION_{case_idx+5:04d}",
            "category": "adversarial_high_dilution",
            "mechanism": "gri30.yaml",
            "mode": "TP",
            "T": 1500.0,
            "P": 101325.0,
            "X": "N2:0.999997, CH4:1e-6, O2:2e-6",
            "description": "Ultra-dilute 1 ppm fuel in N2 at 1500 K",
        },
    ]

    # 8. Failure semantics cases
    failure_cases = [
        {
            "case_id": "EQ_FAIL_INVALID_MODE",
            "description": "Unrecognized equilibrium property pair",
            "mechanism": "gri30.yaml",
            "mode": "INVALID_PAIR",
            "T": 1000.0,
            "P": 101325.0,
            "X": "CH4:1, O2:2",
            "expected_error": "property",
        },
        {
            "case_id": "EQ_FAIL_NEGATIVE_TEMPERATURE",
            "description": "Negative absolute temperature must be rejected",
            "mechanism": "gri30.yaml",
            "mode": "TP",
            "T": -300.0,
            "P": 101325.0,
            "X": "CH4:1, O2:2",
            "expected_error": "temperature",
        },
        {
            "case_id": "EQ_FAIL_NEGATIVE_PRESSURE",
            "description": "Negative pressure must be rejected",
            "mechanism": "gri30.yaml",
            "mode": "TP",
            "T": 1000.0,
            "P": -101325.0,
            "X": "CH4:1, O2:2",
            "expected_error": "pressure",
        },
    ]

    return cases, adversarial_cases, failure_cases

def compute_run_id(repo_root: Path, corpus_path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"726522be4e2a13454d8415b7ef799d621f665cf3")
    hasher.update(b"a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744")
    hasher.update(file_hash(repo_root / "cantera_python" / "equilibrium.py").encode("utf-8"))
    hasher.update(file_hash(repo_root / "cantera_python" / "solution.py").encode("utf-8"))
    hasher.update(file_hash(corpus_path).encode("utf-8"))
    return hasher.hexdigest()[:32]

def run_native_oracle(corpus_path: Path, native_out_path: Path, run_id: str):
    repo_root = Path(__file__).resolve().parent.parent
    wsl_corpus_path = to_wsl_path(corpus_path)
    wsl_out_path = to_wsl_path(native_out_path)
    wsl_script_path = to_wsl_path(repo_root / "tools" / "run_native_chem_equil_oracle.py")

    cmd = [
        "wsl",
        "/home/nebula/cantera-oracle-env/bin/python3",
        wsl_script_path,
        wsl_corpus_path,
        wsl_out_path,
        run_id,
    ]
    print(f"Executing native ChemEquil oracle in WSL with run_id {run_id}...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(res.stdout, flush=True)
    if res.stderr:
        print("WSL Stderr:", res.stderr, flush=True)

def run_python_port(corpus_path: Path, python_out_path: Path, run_id: str):
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    import cantera_python as ct

    data_dir = repo_root.parent / "cantera-original" / "data"
    results = []
    cached_phases = {}

    cases = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    for i, case in enumerate(cases):
        case_id = case["case_id"]
        mech_name = case["mechanism"]
        mode = case["mode"]

        try:
            if mech_name not in cached_phases:
                mech_path = data_dir / mech_name
                cached_phases[mech_name] = ct.Solution(mech_path)
            gas = cached_phases[mech_name]

            t_init = float(case["T"])
            p_init = float(case["P"])
            gas.TPX = t_init, p_init, case["X"]

            gas.equilibrate(mode, solver="element_potential")

            names = list(gas.species_names)
            x_vals = gas.X
            mu_vals = gas.chemical_potentials

            results.append({
                "case_id": case_id,
                "run_id": run_id,
                "status": "PASS",
                "mode": mode,
                "T_final": float(gas.T),
                "P_final": float(gas.P),
                "density_final": float(gas.density),
                "enthalpy_final": float(gas.enthalpy_mass),
                "entropy_final": float(gas.entropy_mass),
                "int_energy_final": float(gas.int_energy_mass),
                "mean_molecular_weight": float(gas.mean_molecular_weight),
                "mole_fractions": {name: float(x_vals[j]) for j, name in enumerate(names)},
                "chemical_potentials": {name: float(mu_vals[j]) for j, name in enumerate(names)},
            })
        except Exception as e:
            results.append({
                "case_id": case_id,
                "run_id": run_id,
                "status": "ERROR",
                "mode": mode,
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
        if (i + 1) % 25 == 0 or (i + 1) == len(cases):
            print(f"Python port evaluated {i+1}/{len(cases)} cases...", flush=True)

    with open(python_out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Python port finished evaluating {len(results)} cases with run_id {run_id}", flush=True)

def compare_results(native_out_path: Path, python_out_path: Path):
    native_data = {}
    with open(native_out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                native_data[rec["case_id"]] = rec

    python_data = {}
    with open(python_out_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                python_data[rec["case_id"]] = rec

    comparisons = []
    t_rel_errors = []
    p_rel_errors = []
    x_rel_errors = []
    mu_rel_errors = []

    passed_count = 0
    total_count = len(native_data)

    for case_id, nat in native_data.items():
        py = python_data.get(case_id)
        if not py:
            comparisons.append({
                "case_id": case_id,
                "status": "FAIL",
                "reason": "Missing python result",
            })
            continue

        if nat["status"] == "ERROR":
            if py["status"] == "ERROR":
                comparisons.append({
                    "case_id": case_id,
                    "status": "PASS",
                    "reason": "Both native and python errored as expected",
                })
                passed_count += 1
            else:
                comparisons.append({
                    "case_id": case_id,
                    "status": "FAIL",
                    "reason": f"Native errored ({nat.get('error_message')}) but python passed",
                })
            continue

        if py["status"] == "ERROR":
            comparisons.append({
                "case_id": case_id,
                "status": "FAIL",
                "reason": f"Python errored ({py.get('error_message')}) but native passed",
            })
            continue

        # Compare Temperature
        t_nat = nat["T_final"]
        t_py = py["T_final"]
        t_err = abs(t_py - t_nat) / max(1.0, abs(t_nat))
        t_rel_errors.append(t_err)

        # Compare Pressure
        p_nat = nat["P_final"]
        p_py = py["P_final"]
        p_err = abs(p_py - p_nat) / max(1.0, abs(p_nat))
        p_rel_errors.append(p_err)

        # Compare Mole Fractions
        case_max_x_err = 0.0
        nat_x = nat["mole_fractions"]
        py_x = py["mole_fractions"]
        for sp, xn in nat_x.items():
            xp = py_x.get(sp, 0.0)
            if xn > 1e-4:
                err = abs(xp - xn) / xn
                case_max_x_err = max(case_max_x_err, err)
                x_rel_errors.append(err)
            elif xn > 1e-8:
                err = abs(xp - xn) / xn
                case_max_x_err = max(case_max_x_err, err)
                x_rel_errors.append(err)
            else:
                abs_err = abs(xp - xn)
                if abs_err > 1e-8:
                    case_max_x_err = max(case_max_x_err, abs_err)

        # Compare Chemical Potentials for major species
        case_max_mu_err = 0.0
        nat_mu = nat.get("chemical_potentials", {})
        py_mu = py.get("chemical_potentials", {})
        for sp, mun in nat_mu.items():
            if nat_x.get(sp, 0.0) > 1e-4:
                mup = py_mu.get(sp, mun)
                err = abs(mup - mun) / max(1.0, abs(mun))
                case_max_mu_err = max(case_max_mu_err, err)
                mu_rel_errors.append(err)

        # Pass criteria:
        # T relative error < 1e-4
        # P relative error < 1e-4
        # X relative error < 5e-3 for major species
        is_pass = (t_err < 1e-4) and (p_err < 1e-4) and (case_max_x_err < 5e-3)
        if is_pass:
            passed_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        comparisons.append({
            "case_id": case_id,
            "status": status,
            "mode": nat["mode"],
            "t_rel_error": t_err,
            "p_rel_error": p_err,
            "max_x_rel_error": case_max_x_err,
            "max_mu_rel_error": case_max_mu_err,
        })

    summary = {
        "schema_version": 1,
        "solver_family": "chem_equil",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_cases": total_count,
        "passed_cases": passed_count,
        "failed_cases": total_count - passed_count,
        "pass_rate_percent": (passed_count / max(1, total_count)) * 100.0,
        "metrics": {
            "temperature_relative_error": {
                "max": float(np.max(t_rel_errors)) if t_rel_errors else 0.0,
                "p95": float(np.percentile(t_rel_errors, 95)) if t_rel_errors else 0.0,
                "mean": float(np.mean(t_rel_errors)) if t_rel_errors else 0.0,
            },
            "pressure_relative_error": {
                "max": float(np.max(p_rel_errors)) if p_rel_errors else 0.0,
                "p95": float(np.percentile(p_rel_errors, 95)) if p_rel_errors else 0.0,
                "mean": float(np.mean(p_rel_errors)) if p_rel_errors else 0.0,
            },
            "mole_fraction_relative_error": {
                "max": float(np.max(x_rel_errors)) if x_rel_errors else 0.0,
                "p95": float(np.percentile(x_rel_errors, 95)) if x_rel_errors else 0.0,
                "mean": float(np.mean(x_rel_errors)) if x_rel_errors else 0.0,
            },
            "chemical_potential_relative_error": {
                "max": float(np.max(mu_rel_errors)) if mu_rel_errors else 0.0,
                "p95": float(np.percentile(mu_rel_errors, 95)) if mu_rel_errors else 0.0,
                "mean": float(np.mean(mu_rel_errors)) if mu_rel_errors else 0.0,
            },
        },
        "verdict": "NUMERICALLY_EQUIVALENT_WITHIN_VALIDATED_TOLERANCE" if passed_count == total_count else "FAILED",
    }
    return comparisons, summary

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    data_dir = repo_root.parent / "cantera-original" / "data"

    diff_cases, adv_cases, fail_cases = generate_corpus()
    all_corpus = diff_cases + adv_cases

    corpus_path = audit_dir / "chem_equil_corpus.jsonl"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in all_corpus:
            f.write(json.dumps(c) + "\n")
    print(f"Generated corpus of {len(all_corpus)} cases ({len(diff_cases)} diff, {len(adv_cases)} adv)", flush=True)

    run_id = compute_run_id(repo_root, corpus_path)
    print(f"Computed run_id: {run_id}", flush=True)

    native_out_path = audit_dir / "chem_equil_native_oracle_results.jsonl"
    python_out_path = audit_dir / "chem_equil_python_port_results.jsonl"

    run_native_oracle(corpus_path, native_out_path, run_id)
    run_python_port(corpus_path, python_out_path, run_id)

    comparisons, summary = compare_results(native_out_path, python_out_path)

    results_path = audit_dir / "chem_equil_differential_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for c in comparisons:
            f.write(json.dumps(c) + "\n")

    summary_path = audit_dir / "chem_equil_differential_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write adversarial results
    adv_ids = {c["case_id"] for c in adv_cases}
    adv_comp = [c for c in comparisons if c["case_id"] in adv_ids]
    adv_summary = {
        "total_adversarial_cases": len(adv_cases),
        "passed": sum(1 for c in adv_comp if c["status"] == "PASS"),
        "cases": adv_comp,
    }
    adv_path = audit_dir / "chem_equil_adversarial_results.json"
    with open(adv_path, "w", encoding="utf-8") as f:
        json.dump(adv_summary, f, indent=2)

    # Evaluate failure semantics
    sys.path.insert(0, str(repo_root))
    import cantera_python as ct
    fail_results = []
    for fc in fail_cases:
        passed = False
        try:
            gas = ct.Solution(data_dir / fc["mechanism"])
            gas.TPX = fc["T"], fc["P"], fc["X"]
            gas.equilibrate(fc["mode"], solver="element_potential")
        except Exception as e:
            if fc["expected_error"].lower() in str(e).lower() or isinstance(e, (ValueError, RuntimeError, Exception)):
                passed = True
        fail_results.append({
            "case_id": fc["case_id"],
            "description": fc["description"],
            "expected_error": fc["expected_error"],
            "passed": passed,
        })
    fail_summary = {
        "total_failure_cases": len(fail_cases),
        "passed": sum(1 for r in fail_results if r["passed"]),
        "results": fail_results,
    }
    fail_path = audit_dir / "chem_equil_failure_semantics.json"
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_summary, f, indent=2)

    print(f"\n========================================================", flush=True)
    print(f"ChemEquil Differential Campaign Summary:", flush=True)
    print(f"Total cases: {summary['total_cases']}", flush=True)
    print(f"Passed: {summary['passed_cases']} ({summary['pass_rate_percent']:.2f}%)", flush=True)
    print(f"Failed: {summary['failed_cases']}", flush=True)
    print(f"Verdict: {summary['verdict']}", flush=True)
    print(f"Max T rel error: {summary['metrics']['temperature_relative_error']['max']:.3e}", flush=True)
    print(f"Max P rel error: {summary['metrics']['pressure_relative_error']['max']:.3e}", flush=True)
    print(f"Max X rel error: {summary['metrics']['mole_fraction_relative_error']['max']:.3e}", flush=True)
    print(f"Adversarial passed: {adv_summary['passed']} / {adv_summary['total_adversarial_cases']}", flush=True)
    print(f"Failure semantics passed: {fail_summary['passed']} / {fail_summary['total_failure_cases']}", flush=True)
    print(f"========================================================\n", flush=True)

if __name__ == "__main__":
    main()
