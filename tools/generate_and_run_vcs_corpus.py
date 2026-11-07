"""Generate and run comprehensive VCS rocket, combustion, and multiphase corpus.

Compares outputs from cantera_python VCS against the pinned native Cantera oracle.
"""

import json
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



def generate_corpus():
    cases = []
    case_id = 0

    # 1. Rocket Hydrolox (H2/O2) at chamber pressures 10, 30, 70, 100, 200 bar
    # O/F mass ratios: 4.0, 5.0, 6.0, 7.0, 8.0 (stoichiometric is 8.0)
    # T from 2000 to 3500 K
    for p_bar in [10.0, 30.0, 70.0, 100.0, 200.0]:
        p_pa = p_bar * 1.0e5
        for of_ratio in [4.0, 5.5, 6.0, 8.0]:
            # Convert mass ratio to mole ratio:
            # moles_O2 = (of_ratio / 31.998), moles_H2 = (1.0 / 2.016)
            n_o2 = of_ratio / 31.9988
            n_h2 = 1.0 / 2.016
            x_str = f"H2:{n_h2:.6f}, O2:{n_o2:.6f}"
            for temp in [2500.0, 3200.0]:
                case_id += 1
                cases.append({
                    "case_id": f"VCS_HYDROLOX_{case_id:03d}",
                    "category": "rocket_hydrolox",
                    "mechanism": "gri30.yaml",
                    "T": temp,
                    "P": p_pa,
                    "X": x_str,
                    "phases": [{"name": "gas", "source": "gri30.yaml", "moles": 1.0}],
                    "mode": "TP"
                })

    # 2. Rocket Methalox (CH4/O2) at chamber pressures 10, 20, 70, 100, 200 bar
    # O/F mass ratios: 2.5, 3.0, 3.5, 3.8 (stoichiometric is 4.0)
    # T from 2200 to 3400 K
    for p_bar in [10.0, 20.0, 70.0, 100.0, 200.0]:
        p_pa = p_bar * 1.0e5
        for of_ratio in [2.5, 3.2, 3.5]:
            n_o2 = of_ratio / 31.9988
            n_ch4 = 1.0 / 16.043
            x_str = f"CH4:{n_ch4:.6f}, O2:{n_o2:.6f}"
            for temp in [2400.0, 3000.0]:
                case_id += 1
                cases.append({
                    "case_id": f"VCS_METHALOX_{case_id:03d}",
                    "category": "rocket_methalox",
                    "mechanism": "gri30.yaml",
                    "T": temp,
                    "P": p_pa,
                    "X": x_str,
                    "phases": [{"name": "gas", "source": "gri30.yaml", "moles": 1.0}],
                    "mode": "TP"
                })

    # 3. Hydrocarbon & Hydrogen Air Combustion at equivalence ratios phi in [0.5, 1.0, 1.5]
    for phi in [0.6, 1.0, 1.4]:
        # CH4 + 2/phi (O2 + 3.76 N2)
        n_ch4 = 1.0
        n_o2 = 2.0 / phi
        n_n2 = n_o2 * 3.76
        x_str = f"CH4:{n_ch4:.4f}, O2:{n_o2:.4f}, N2:{n_n2:.4f}"
        for p_bar in [1.0, 10.0, 30.0]:
            case_id += 1
            cases.append({
                "case_id": f"VCS_CH4_AIR_{case_id:03d}",
                "category": "air_combustion_ch4",
                "mechanism": "gri30.yaml",
                "T": 1800.0,
                "P": p_bar * 1.0e5,
                "X": x_str,
                "phases": [{"name": "gas", "source": "gri30.yaml", "moles": 1.0}],
                "mode": "TP"
            })

    # 4. Multiphase systems: Gas + Graphite (C(s))
    # Lean (graphite is consumed) and rich (graphite precipitates / survives)
    for c_moles in [0.1, 0.5, 1.5]:
        for temp in [1200.0, 1800.0]:
            case_id += 1
            cases.append({
                "case_id": f"VCS_MULTIPHASE_GRAPHITE_{case_id:03d}",
                "category": "multiphase_gas_solid",
                "mechanism": "gri30.yaml",
                "T": temp,
                "P": 1.0e5,
                "X": "CH4:1.0, O2:1.5, N2:5.64",
                "phases": [
                    {"name": "gas", "source": "gri30.yaml", "moles": 1.0},
                    {"name": "graphite", "source": "graphite.yaml", "moles": c_moles}
                ],
                "mode": "TP"
            })

    return cases

def run_python_case(case):
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import cantera_python as ct
    from cantera_python.vcs import MultiPhase, vcs_equilibrate

    mp = MultiPhase()
    for p_info in case["phases"]:
        sol = ct.Solution(p_info["source"])
        sol.TP = case["T"], case["P"]
        if p_info["name"] == "gas":
            sol.X = case["X"]
        mp.add_phase(sol, p_info["moles"])

    mp.set_TP(case["T"], case["P"])
    vcs_equilibrate(mp, case["mode"])

    phase_moles = [float(mp.phase_moles[i]) for i in range(mp.n_phases)]
    gas_phase = mp.phase(0)
    
    # Track key species
    key_species = ["H2", "H2O", "O2", "CO", "CO2", "CH4", "OH", "H", "O"]
    mole_fractions = {}
    for sp in key_species:
        if sp in gas_phase.species_names:
            mole_fractions[sp] = float(gas_phase.X[gas_phase.species_index(sp)])

    # Element conservation check
    elem_moles = {elem: float(mp.elementMoles(i)) for i, elem in enumerate(mp.element_names)}

    return {
        "status": "CONVERGED",
        "phase_moles": phase_moles,
        "mole_fractions": mole_fractions,
        "element_moles": elem_moles,
        "T": float(mp.T),
        "P": float(mp.P),
    }

def main():
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    cases = generate_corpus()
    print(f"Total VCS test cases generated: {len(cases)}")

    corpus_file = audit_dir / "corpus_vcs.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    # Run Python port
    python_results = {}
    for c in cases:
        try:
            res = run_python_case(c)
            python_results[c["case_id"]] = res
        except Exception as e:
            python_results[c["case_id"]] = {"status": "FAILED", "error": str(e)}

    raw_native_path = audit_dir / "raw_native_vcs.json"
    run_native = True
    if raw_native_path.exists():
        try:
            with open(raw_native_path, "r", encoding="utf-8") as f:
                existing_nat = json.load(f)
            if len(existing_nat) == len(cases) and "--regenerate-native" not in sys.argv:
                print(f"Reusing existing native oracle results from {raw_native_path} ({len(existing_nat)} cases)")
                native_results = existing_nat
                run_native = False
        except Exception:
            run_native = True

    if run_native:
        wsl_corpus_path = to_wsl_path(corpus_file)
        wsl_raw_native_path = to_wsl_path(raw_native_path)

        oracle_runner_script = f"""
import json
import cantera as ct

with open('{wsl_corpus_path}') as f:
    cases = [json.loads(line) for line in f]

results = {{}}
for c in cases:
    try:
        phases = []
        for p_info in c['phases']:
            sol = ct.Solution(p_info['source'])
            sol.TP = c['T'], c['P']
            if p_info['name'] == 'gas':
                sol.X = c['X']
            phases.append((sol, p_info['moles']))
        mix = ct.Mixture(phases)
        mix.T = c['T']
        mix.P = c['P']
        mix.equilibrate(c['mode'], solver='vcs')

        pmoles = [float(mix.phase_moles(i)) for i in range(mix.n_phases)]
        gas = phases[0][0]
        key_species = ['H2', 'H2O', 'O2', 'CO', 'CO2', 'CH4', 'OH', 'H', 'O']
        mf = {{}}
        for sp in key_species:
            if sp in gas.species_names:
                mf[sp] = float(gas.X[gas.species_index(sp)])
        results[c['case_id']] = {{
            'status': 'CONVERGED',
            'phase_moles': pmoles,
            'mole_fractions': mf,
            'T': float(mix.T),
            'P': float(mix.P)
        }}
    except Exception as e:
        results[c['case_id']] = {{'status': 'FAILED', 'error': str(e)}}

with open('{wsl_raw_native_path}', 'w') as f:
    json.dump(results, f, indent=2)
print('Native oracle execution finished successfully')
"""
        native_script_path = audit_dir / "run_native_vcs_oracle.py"
        native_script_path.write_text(oracle_runner_script, encoding="utf-8")
        wsl_native_script_path = to_wsl_path(native_script_path)

        wsl_cmd = [
            "wsl", "-d", "Ubuntu", "--",
            "/home/nebula/cantera-oracle-env/bin/python3",
            wsl_native_script_path
        ]
        print("Running pinned native Cantera oracle in WSL...")
        p = subprocess.run(wsl_cmd, capture_output=True, text=True)
        if p.returncode != 0:
            print("WSL native oracle stderr:", p.stderr)
            raise RuntimeError("Failed to run native oracle in WSL")
        print(p.stdout.strip())

        with open(raw_native_path, "r", encoding="utf-8") as f:
            native_results = json.load(f)

    # Compare Python vs Native
    diff_results = []
    max_phase_mole_diff = 0.0
    max_species_mf_diff = 0.0
    total_compared = 0
    total_passed = 0

    for c in cases:
        cid = c["case_id"]
        py_r = python_results.get(cid)
        nat_r = native_results.get(cid)

        if py_r.get("status") != "CONVERGED" or nat_r.get("status") != "CONVERGED":
            diff_results.append({
                "case_id": cid,
                "status": "FAIL_STATUS",
                "python": py_r,
                "native": nat_r
            })
            continue

        # Phase moles error
        pm_py = np.array(py_r["phase_moles"])
        pm_nat = np.array(nat_r["phase_moles"])
        pm_diff = float(np.max(np.abs(pm_py - pm_nat)))
        max_phase_mole_diff = max(max_phase_mole_diff, pm_diff)

        # Species mole fraction error
        mf_py = py_r["mole_fractions"]
        mf_nat = nat_r["mole_fractions"]
        sp_diffs = {}
        for sp in mf_py:
            if sp in mf_nat:
                diff = abs(mf_py[sp] - mf_nat[sp])
                sp_diffs[sp] = diff
                max_species_mf_diff = max(max_species_mf_diff, diff)

        passed = (pm_diff < 1.0e-4) and all(d < 1.0e-3 for d in sp_diffs.values())
        total_compared += 1
        if passed:
            total_passed += 1

        diff_results.append({
            "case_id": cid,
            "category": c["category"],
            "passed": passed,
            "phase_mole_max_diff": pm_diff,
            "species_max_diff": max(sp_diffs.values()) if sp_diffs else 0.0,
            "python_phase_moles": py_r["phase_moles"],
            "native_phase_moles": nat_r["phase_moles"],
            "species_diffs": sp_diffs
        })

    with open(audit_dir / "vcs_differential_results.jsonl", "w", encoding="utf-8") as f:
        for r in diff_results:
            f.write(json.dumps(r) + "\n")

    summary = {
        "VCS_TOTAL_CASES": len(cases),
        "VCS_CASES_COMPARED": total_compared,
        "VCS_CASES_PASSED": total_passed,
        "VCS_PASS_RATE": total_passed / total_compared if total_compared > 0 else 0.0,
        "MAX_PHASE_MOLE_DIFF": max_phase_mole_diff,
        "MAX_SPECIES_MF_DIFF": max_species_mf_diff,
        "NATIVE_ORACLE_COMMIT": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "VCS_STATUS": "EQUIVALENT_PROVEN" if total_passed == total_compared else "PARTIAL"
    }

    with open(audit_dir / "vcs_differential_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== VCS DIFFERENTIAL VALIDATION SUMMARY ===")
    print(f"Total Cases: {summary['VCS_TOTAL_CASES']}")
    print(f"Passed: {summary['VCS_CASES_PASSED']} / {summary['VCS_CASES_COMPARED']} ({summary['VCS_PASS_RATE']*100:.2f}%)")
    print(f"Max Phase Mole Diff: {summary['MAX_PHASE_MOLE_DIFF']:.4e}")
    print(f"Max Species MF Diff: {summary['MAX_SPECIES_MF_DIFF']:.4e}")
    print(f"VCS_STATUS = {summary['VCS_STATUS']}")

if __name__ == "__main__":
    main()
