"""Pure-Python kinetics Family 2 port runner for differential validation against native Cantera.

Reads audit/corpus_kinetics_family2.jsonl, evaluates all 39 test cases using pure-Python
cantera_python (zero native dependencies), and writes exact results to
audit/python_kinetics_family2_port_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cantera_python as cp
CORPUS_PATH = ROOT / "audit" / "corpus_kinetics_family2.jsonl"
OUTPUT_PATH = ROOT / "audit" / "python_kinetics_family2_port_results.json"


def clean_vector(vec):
    """Serialize numpy vector replacing NaNs and Infs with string representations."""
    out = []
    for val in vec:
        if np.isnan(val):
            out.append("NaN")
        elif np.isposinf(val):
            out.append("Inf")
        elif np.isneginf(val):
            out.append("-Inf")
        else:
            out.append(float(val))
    return out


def main():
    print(f"Executing Pure-Python Kinetics Family 2 Port Runner against {CORPUS_PATH}")

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = [json.loads(line) for line in f if line.strip()]

    solutions = {}
    results = []

    for idx, case in enumerate(corpus):
        case_id = case["case_id"]
        mech = case["mechanism"]
        phase = case.get("phase", "")
        T = float(case["temperature"])
        P = float(case["pressure"])
        X = case["mole_fractions"]
        multipliers = case.get("multipliers", {})

        print(f"  [{idx+1}/{len(corpus)}] {case_id} ({mech} phase={phase or 'default'} T={T} P={P})...", end="", flush=True)

        sol_key = (mech, phase)
        if sol_key not in solutions:
            if phase:
                solutions[sol_key] = cp.Solution(mech, phase)
            else:
                solutions[sol_key] = cp.Solution(mech)
        gas = solutions[sol_key]
        kin = gas.kinetics

        # Reset multipliers
        kin.set_multiplier(1.0)
        # Apply case multipliers
        for rxn_idx_str, mult in multipliers.items():
            kin.set_multiplier(mult, int(rxn_idx_str))

        gas.TPX = T, P, X

        kf = list(gas.forward_rate_constants)
        kr = list(gas.reverse_rate_constants)
        kc = list(gas.equilibrium_constants)
        ropf = list(gas.forward_rates_of_progress)
        ropr = list(gas.reverse_rates_of_progress)
        ropnet = list(gas.net_rates_of_progress)
        cdot = list(gas.creation_rates)
        ddot = list(gas.destruction_rates)
        wdot = list(gas.net_production_rates)

        dg = list(gas.delta_gibbs)
        dh = list(gas.delta_enthalpy)
        ds = list(gas.delta_entropy)
        dg0 = list(gas.delta_standard_gibbs)
        dh0 = list(gas.delta_standard_enthalpy)
        ds0 = list(gas.delta_standard_entropy)
        tb_conc = list(gas.third_body_concentrations)

        results.append({
            "case_id": case_id,
            "mechanism": mech,
            "phase": phase,
            "temperature": T,
            "pressure": P,
            "mole_fractions": X,
            "n_reactions": len(kf),
            "n_species": len(wdot),
            "forward_rate_constants": clean_vector(kf),
            "reverse_rate_constants": clean_vector(kr),
            "equilibrium_constants": clean_vector(kc),
            "forward_rates_of_progress": clean_vector(ropf),
            "reverse_rates_of_progress": clean_vector(ropr),
            "net_rates_of_progress": clean_vector(ropnet),
            "species_creation_rates": clean_vector(cdot),
            "species_destruction_rates": clean_vector(ddot),
            "net_species_production_rates": clean_vector(wdot),
            "delta_gibbs": clean_vector(dg),
            "delta_enthalpy": clean_vector(dh),
            "delta_entropy": clean_vector(ds),
            "delta_standard_gibbs": clean_vector(dg0),
            "delta_standard_enthalpy": clean_vector(dh0),
            "delta_standard_entropy": clean_vector(ds0),
            "third_body_concentrations": clean_vector(tb_conc)
        })
        print(" done.")

    payload = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "total_cases": len(results),
        "cases": results
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Finished executing python kinetics Family 2 port runner for {len(results)} cases. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
