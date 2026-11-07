"""Differential comparison of Python Phase implementation against native Cantera Family 1 Oracle."""
import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from cantera_python.phases import Phase
from cantera_python.solution import Solution

def main():
    oracle_path = "audit/native_thermo_family1_oracle_results.json"
    with open(oracle_path, "r", encoding="utf-8") as f:
        nat_data = json.load(f)

    print("Comparing Python Phase implementation against Native Oracle...")
    summary = {
        "campaign": "thermo_family1_phase_differential",
        "pinned_commit": nat_data["pinned_commit"],
        "cases": []
    }

    all_passed = True
    for nat_phase in nat_data["phases"]:
        fname = nat_phase["file"]
        sol = Solution(fname)
        ph = sol.thermo

        passed = True
        errs = []

        # Compare basic properties
        if sol.n_species != nat_phase["nSpecies"]:
            passed = False
            errs.append(f"nSpecies mismatch: py={sol.n_species} nat={nat_phase['nSpecies']}")

        # Compare element names
        el_names_py = list(sol.element_names)
        if el_names_py != nat_phase["elementNames"]:
            passed = False
            errs.append(f"elementNames mismatch: py={el_names_py} nat={nat_phase['elementNames']}")

        # Compare atomic weights
        at_weights_py = [sol.atomic_weight(m) for m in range(len(el_names_py))]
        at_weights_nat = nat_phase["atomicWeights"]
        max_at_err = np.max(np.abs(np.array(at_weights_py) - np.array(at_weights_nat)))
        if max_at_err > 1e-4:
            passed = False
            errs.append(f"atomicWeights max err={max_at_err:.3e}")

        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        summary["cases"].append({
            "mechanism": fname,
            "status": status,
            "errors": errs
        })
        print(f"  {fname:15s}: {status}")

    summary_path = "audit/thermo_family1_differential_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Differential summary written to {summary_path}")
    assert all_passed, "Differential comparison failed"

if __name__ == "__main__":
    main()
