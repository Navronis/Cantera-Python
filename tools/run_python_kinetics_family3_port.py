#!/usr/bin/env python3
"""Execute the Pure-Python Kinetics Family 3 Port against audit/corpus_kinetics_family3.jsonl,
writing structured numerical results to audit/python_kinetics_family3_port_results.json.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add python port to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cantera_python.surface import Interface

def resolve_mech(mech_name):
    orig_root = ROOT.parent / "cantera-original"
    p1 = orig_root / "data" / mech_name
    if p1.exists(): return str(p1)
    p2 = orig_root / "test" / "data" / mech_name
    if p2.exists(): return str(p2)
    raise FileNotFoundError(f"Cannot resolve mechanism {mech_name}")

def to_list(arr):
    out = []
    for x in arr:
        v = float(x)
        if np.isnan(v):
            out.append("NaN")
        elif np.isinf(v):
            out.append("Inf" if v > 0 else "-Inf")
        else:
            out.append(v)
    return out

def main():
    corpus_file = ROOT / "audit" / "corpus_kinetics_family3.jsonl"
    out_file = ROOT / "audit" / "python_kinetics_family3_port_results.json"

    print(f"Executing Python Kinetics Family 3 Port against {corpus_file}")

    with open(corpus_file, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    cached_interfaces = {}

    results = []
    for idx, c in enumerate(cases, 1):
        cid = c["case_id"]
        mech = c["mechanism"]
        phase = c["phase"]
        T = float(c["temperature"])
        P = float(c["pressure"])
        gas_phase = c.get("gas_phase", "gas")
        gas_X = c.get("gas_mole_fractions", "")
        covs = c.get("coverages", "")
        phi = float(c.get("electric_potential", 0.0))
        mults = {int(k): float(v) for k, v in c.get("multipliers", {}).items()}

        print(f"  [{idx}/{len(cases)}] {cid} ({mech} phase={phase} T={T} P={P})...", end="", flush=True)

        cache_key = (mech, phase)
        if cache_key not in cached_interfaces:
            mech_path = resolve_mech(mech)
            cached_interfaces[cache_key] = Interface(mech_path, phase)
        surf = cached_interfaces[cache_key]
        kin = surf.kinetics

        # Reset multipliers to 1.0
        for i in range(kin.n_reactions):
            kin.setMultiplier(i, 1.0)
        # Apply custom multipliers
        for r_idx, val in mults.items():
            if r_idx < kin.n_reactions:
                kin.setMultiplier(r_idx, val)

        # Set gas phase state
        if gas_phase and hasattr(surf, 'adjacent') and gas_phase in surf.adjacent:
            gas = surf.adjacent[gas_phase]
            if gas_X:
                gas.TPX = T, P, gas_X
            else:
                gas.TP = T, P

        # Set surface phase state
        surf.TP = T, P
        if covs:
            surf.coverages = covs
        surf.electric_potential = phi

        nRxn = kin.n_reactions
        nTotalSp = kin.n_total_species
        nSurfSp = surf.n_species

        res = {
            "case_id": cid,
            "mechanism": mech,
            "phase": phase,
            "temperature": T,
            "pressure": P,
            "gas_mole_fractions": gas_X,
            "coverages": covs,
            "electric_potential": phi,
            "n_reactions": nRxn,
            "n_total_species": nTotalSp,
            "n_surface_species": nSurfSp,
            "forward_rate_constants": to_list(kin.forward_rate_constants),
            "reverse_rate_constants": to_list(kin.reverse_rate_constants),
            "equilibrium_constants": to_list(kin.equilibrium_constants),
            "forward_rates_of_progress": to_list(kin.forward_rates_of_progress),
            "reverse_rates_of_progress": to_list(kin.reverse_rates_of_progress),
            "net_rates_of_progress": to_list(kin.net_rates_of_progress),
            "species_creation_rates": to_list(kin.creation_rates),
            "species_destruction_rates": to_list(kin.destruction_rates),
            "net_species_production_rates": to_list(kin.net_production_rates),
            "delta_gibbs": to_list(kin.delta_gibbs),
            "delta_enthalpy": to_list(kin.delta_enthalpy),
            "delta_entropy": to_list(kin.delta_entropy),
            "delta_standard_gibbs": to_list(kin.delta_standard_gibbs),
            "delta_standard_enthalpy": to_list(kin.delta_standard_enthalpy),
            "delta_standard_entropy": to_list(kin.delta_standard_entropy),
            "surface_concentrations": to_list(surf.concentrations),
            "surface_coverages": to_list(surf.coverages),
            "interface_current": float(kin.interfaceCurrent(0))
        }
        results.append(res)
        print(" done.")

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"total_cases": len(results), "cases": results}, f, indent=2)

    print(f"Finished executing pure-Python Family 3 port for {len(results)} cases -> {out_file}")

if __name__ == "__main__":
    main()
