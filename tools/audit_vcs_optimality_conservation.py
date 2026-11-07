import json
import math
from pathlib import Path
import sys
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cantera_python as ct
from cantera_python.vcs import MultiPhase, VCS_SOLVE
from tools.generate_and_run_vcs_corpus import generate_corpus

def main():
    cases = generate_corpus()
    print(f"Auditing element conservation & equilibrium optimality across {len(cases)} cases...")

    max_elem_abs_res = 0.0
    max_elem_norm_res = 0.0
    worst_elem_case = None

    max_delta_g_major_res = 0.0
    worst_delta_g_case = None

    max_zero_comp_res = 0.0
    worst_zero_comp_case = None

    max_neg_mole_res = 0.0
    worst_neg_mole_case = None

    max_charge_res = 0.0

    case_records = []

    for c in cases:
        cid = c["case_id"]
        mp = MultiPhase()
        for p_info in c["phases"]:
            sol = ct.Solution(p_info["source"])
            sol.TP = c["T"], c["P"]
            if p_info["name"] == "gas":
                sol.X = c["X"]
            mp.add_phase(sol, p_info["moles"])
        mp.set_TP(c["T"], c["P"])

        # Target element abundances before equilibration
        b_target = mp.calcElemAbundances().copy()

        # Solve with VCS
        solver = VCS_SOLVE()
        solver.vcs_solve_TP(mp)

        # 1. Element conservation
        b_post = mp.calcElemAbundances()
        elem_diff = np.abs(b_post - b_target)
        abs_elem = float(np.max(elem_diff))
        norm_elem = float(np.max(elem_diff / np.maximum(b_target, 1.0e-12)))

        if abs_elem > max_elem_abs_res:
            max_elem_abs_res = abs_elem
            worst_elem_case = cid
        if norm_elem > max_elem_norm_res:
            max_elem_norm_res = norm_elem

        # 2. Non-negative mole constraints
        species_moles = mp._species_moles()
        neg_moles = [-m for m in species_moles if m < 0.0]
        neg_res = max(neg_moles) if neg_moles else 0.0
        if neg_res > max_neg_mole_res:
            max_neg_mole_res = neg_res
            worst_neg_mole_case = cid

        # 3. Reaction DeltaG for major species and complementarity for zeroed species
        p_state = solver._prepared_problem
        s_state = solver._old_thermodynamic_state
        dg = s_state.get("delta_g_rxn_old", np.zeros(0))
        sp_status = s_state.get("species_status", np.zeros(0))
        ncomp = int(p_state["component_basis"]["num_components"])

        major_dg_res = 0.0
        zero_dg_res = 0.0

        for irxn in range(len(dg)):
            kspec = ncomp + irxn
            st = sp_status[kspec]
            # Major species equilibrium: Delta G ~ 0
            if st == 1: # VCS_SPECIES_MAJOR
                major_dg_res = max(major_dg_res, abs(float(dg[irxn])))
            # Zeroed species: Delta G >= 0, violation if dg < 0
            elif st in (-2, -3, -4, -6, -8): # zeroed, deleted, stoichzero
                if dg[irxn] < 0.0:
                    zero_dg_res = max(zero_dg_res, -float(dg[irxn]))

        if major_dg_res > max_delta_g_major_res:
            max_delta_g_major_res = major_dg_res
            worst_delta_g_case = cid
        if zero_dg_res > max_zero_comp_res:
            max_zero_comp_res = zero_dg_res
            worst_zero_comp_case = cid

        # 4. Charge neutrality
        charge_val = 0.0
        for ip, ph in enumerate(mp._phases):
            ch_attr = getattr(ph, "charge", 0.0)
            ch = ch_attr() if callable(ch_attr) else float(ch_attr)
            charge_val += abs(ch * mp._phase_moles[ip])
        max_charge_res = max(max_charge_res, charge_val)

        case_records.append({
            "case_id": cid,
            "element_abs_residual": abs_elem,
            "element_norm_residual": norm_elem,
            "major_delta_g_residual": major_dg_res,
            "zeroed_complementarity_residual": zero_dg_res,
            "negative_mole_residual": neg_res
        })

    # Realistic scientific tolerances:
    # Element conservation: < 1e-10 (here machine precision ~ 1e-15)
    # Major species DeltaG / RT: < 1e-2
    # Zeroed species DeltaG / RT violation: < 1e-3
    # Non-negative mole violation: == 0.0
    audit_summary = {
        "VCS_TOTAL_CASES_AUDITED": len(cases),
        "ELEMENT_CONSERVATION": {
            "MAX_ABSOLUTE_RESIDUAL": max_elem_abs_res,
            "MAX_NORMALIZED_RESIDUAL": max_elem_norm_res,
            "WORST_CASE_ID": worst_elem_case,
            "MAX_CHARGE_RESIDUAL": max_charge_res,
            "STATUS": "PASSED" if max_elem_abs_res < 1.0e-10 else "FAILED"
        },
        "EQUILIBRIUM_OPTIMALITY": {
            "MAX_MAJOR_SPECIES_DELTA_G_RESIDUAL": max_delta_g_major_res,
            "WORST_MAJOR_SPECIES_CASE_ID": worst_delta_g_case,
            "MAX_ZEROED_SPECIES_COMPLEMENTARITY_VIOLATION": max_zero_comp_res,
            "WORST_ZEROED_SPECIES_CASE_ID": worst_zero_comp_case,
            "MAX_NEGATIVE_MOLE_VIOLATION": max_neg_mole_res,
            "WORST_NEGATIVE_MOLE_CASE_ID": worst_neg_mole_case,
            "STATUS": "PASSED" if (max_delta_g_major_res < 1.0e-2 and max_neg_mole_res == 0.0 and max_zero_comp_res < 1.0e-3) else "FAILED"
        }
    }

    out_file = repo_root / "audit" / "vcs_conservation_optimality_audit.json"
    out_file.write_text(json.dumps(audit_summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit_summary, indent=2))

if __name__ == "__main__":
    main()
