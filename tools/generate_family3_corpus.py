#!/usr/bin/env python3
"""Generate comprehensive differential corpus for Kinetics Family 3:
audit/corpus_kinetics_family3.jsonl
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    cases = []
    case_idx = 1

    def add_case(mech, phase, T, P, gas_phase, gas_X, surf_cov, desc, phi=0.0, multipliers=None):
        nonlocal case_idx
        cid = f"fam3_{case_idx:03d}_{Path(mech).stem}_{phase}"
        cases.append({
            "case_id": cid,
            "mechanism": mech,
            "phase": phase,
            "temperature": float(T),
            "pressure": float(P),
            "gas_phase": gas_phase,
            "gas_mole_fractions": gas_X,
            "coverages": surf_cov,
            "electric_potential": float(phi),
            "multipliers": multipliers or {},
            "description": desc
        })
        case_idx += 1

    OneAtm = 101325.0

    # 1. ptcombust.yaml (Pt_surf) - 24 reactions, Langmuir-Hinshelwood + Sticking
    # Gas species: H2, H, O, O2, OH, H2O, HO2, H2O2, C, CH, CH2, CH2(S), CH3, CH4,
    #              CO, CO2, HCO, CH2O, CH2OH, CH3O, CH3OH, C2H, C2H2, C2H3, C2H4,
    #              C2H5, C2H6, HCCO, CH2CO, HCCOH, N2, AR
    # Surface species: PT(S), H(S), H2O(S), OH(S), CO(S), CO2(S), C(S), O(S)
    gas_lean = "CH4:0.05, O2:0.20, AR:0.75"
    gas_rich = "CH4:0.20, O2:0.15, AR:0.65"
    gas_stoich = "CH4:0.095, O2:0.19, AR:0.715"
    gas_syngas = "H2:0.3, CO:0.2, O2:0.15, AR:0.35"

    add_case("ptcombust.yaml", "Pt_surf", 300.0, OneAtm, "gas", gas_lean,
             "PT(S):1.0", "Pt_surf low T, 1 atm, bare surface lean methane")
    add_case("ptcombust.yaml", "Pt_surf", 600.0, 0.1 * OneAtm, "gas", gas_rich,
             "PT(S):0.5, H(S):0.2, O(S):0.2, OH(S):0.1", "Pt_surf 600 K, 0.1 atm, mixed coverages rich")
    add_case("ptcombust.yaml", "Pt_surf", 1000.0, OneAtm, "gas", gas_stoich,
             "PT(S):0.6, C(S):0.1, CO(S):0.1, CO2(S):0.1, H2O(S):0.1", "Pt_surf 1000 K, 1 atm, carbonaceous coverages")
    add_case("ptcombust.yaml", "Pt_surf", 1400.0, 5.0 * OneAtm, "gas", gas_stoich,
             "PT(S):0.4, O(S):0.3, H(S):0.2, OH(S):0.1", "Pt_surf 1400 K, 5 atm, high T flame")
    add_case("ptcombust.yaml", "Pt_surf", 1800.0, 20.0 * OneAtm, "gas", gas_lean,
             "PT(S):0.7, O(S):0.3", "Pt_surf 1800 K, 20 atm, high pressure lean")
    add_case("ptcombust.yaml", "Pt_surf", 800.0, OneAtm, "gas", gas_stoich,
             "PT(S):0.5, H(S):0.3, O(S):0.2", "Pt_surf 800 K, 1 atm, multiplier perturbations",
             multipliers={0: 1.5, 3: 0.7, 10: 2.0})
    add_case("ptcombust.yaml", "Pt_surf", 500.0, 0.01 * OneAtm, "gas", gas_lean,
             "PT(S):0.8, H(S):0.1, O(S):0.1", "Pt_surf 500 K, 0.01 atm, low P sticking regime")
    add_case("ptcombust.yaml", "Pt_surf", 1200.0, 10.0 * OneAtm, "gas", gas_syngas,
             "PT(S):0.5, CO(S):0.3, H(S):0.2", "Pt_surf 1200 K, 10 atm, syngas mixture")
    add_case("ptcombust.yaml", "Pt_surf", 2200.0, 50.0 * OneAtm, "gas", gas_stoich,
             "PT(S):0.9, O(S):0.1", "Pt_surf 2200 K, 50 atm, extreme temperature")

    # 2. ptcombust-motzwise.yaml (Pt_surf) - Motz-Wise sticking corrections
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 400.0, OneAtm, "gas", gas_lean,
             "PT(S):0.9, H(S):0.1", "Motz-Wise 400 K, 1 atm")
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 800.0, 0.5 * OneAtm, "gas", gas_stoich,
             "PT(S):0.6, H(S):0.2, O(S):0.2", "Motz-Wise 800 K, 0.5 atm")
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 1200.0, 2.0 * OneAtm, "gas", gas_rich,
             "PT(S):0.5, C(S):0.2, H(S):0.3", "Motz-Wise 1200 K, 2 atm")
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 1600.0, 10.0 * OneAtm, "gas", gas_stoich,
             "PT(S):0.7, O(S):0.2, OH(S):0.1", "Motz-Wise 1600 K, 10 atm")
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 1000.0, OneAtm, "gas", gas_syngas,
             "PT(S):0.6, CO(S):0.2, H(S):0.2", "Motz-Wise 1000 K, multipliers",
             multipliers={2: 0.5, 7: 1.8})
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 600.0, 0.1 * OneAtm, "gas", gas_lean,
             "PT(S):0.8, O(S):0.2", "Motz-Wise 600 K, 0.1 atm low P")
    add_case("ptcombust-motzwise.yaml", "Pt_surf", 2000.0, 30.0 * OneAtm, "gas", gas_stoich,
             "PT(S):0.95, O(S):0.05", "Motz-Wise 2000 K, 30 atm extreme")

    # 3. sticking_coeff_check.yaml (Pt_surf) - Sticking reactions
    # Gas species: O2, OH
    # Surface species: PT(S), OH(S), O(S)
    stk_gas = "O2:0.9, OH:0.1"
    add_case("sticking_coeff_check.yaml", "Pt_surf", 300.0, OneAtm, "gas", stk_gas,
             "PT(S):1.0", "Sticking check 300 K, bare surface")
    add_case("sticking_coeff_check.yaml", "Pt_surf", 700.0, OneAtm, "gas", stk_gas,
             "PT(S):0.7, O(S):0.2, OH(S):0.1", "Sticking check 700 K, 1 atm")
    add_case("sticking_coeff_check.yaml", "Pt_surf", 1100.0, 5.0 * OneAtm, "gas", stk_gas,
             "PT(S):0.5, O(S):0.3, OH(S):0.2", "Sticking check 1100 K, 5 atm")
    add_case("sticking_coeff_check.yaml", "Pt_surf", 1500.0, 0.2 * OneAtm, "gas", stk_gas,
             "PT(S):0.8, OH(S):0.2", "Sticking check 1500 K, 0.2 atm")
    add_case("sticking_coeff_check.yaml", "Pt_surf", 2000.0, 20.0 * OneAtm, "gas", stk_gas,
             "PT(S):0.9, O(S):0.1", "Sticking check 2000 K, 20 atm")
    add_case("sticking_coeff_check.yaml", "Pt_surf", 900.0, OneAtm, "gas", stk_gas,
             "PT(S):0.6, O(S):0.2, OH(S):0.2", "Sticking check 900 K, multiplier",
             multipliers={0: 1.5})

    # 4. blowers-masel.yaml (Pt_surf) - Surface Blowers-Masel reactions
    # Gas species: H2, H, O, O2, OH, H2O, CH4, AR, N2
    # Surface species: PT(S), H(S), H2O(S), OH(S), O(S)
    bm_gas = "H2:0.4, O2:0.2, H2O:0.1, AR:0.3"
    add_case("blowers-masel.yaml", "Pt_surf", 400.0, OneAtm, "gas", bm_gas,
             "PT(S):0.8, H(S):0.2", "Blowers-Masel surface 400 K, 1 atm")
    add_case("blowers-masel.yaml", "Pt_surf", 700.0, 0.5 * OneAtm, "gas", bm_gas,
             "PT(S):0.5, H(S):0.3, O(S):0.2", "Blowers-Masel surface 700 K, 0.5 atm")
    add_case("blowers-masel.yaml", "Pt_surf", 1000.0, OneAtm, "gas", bm_gas,
             "PT(S):0.4, H(S):0.2, O(S):0.2, OH(S):0.1, H2O(S):0.1", "Blowers-Masel surface 1000 K, 1 atm")
    add_case("blowers-masel.yaml", "Pt_surf", 1300.0, 5.0 * OneAtm, "gas", bm_gas,
             "PT(S):0.6, O(S):0.3, H(S):0.1", "Blowers-Masel surface 1300 K, 5 atm")
    add_case("blowers-masel.yaml", "Pt_surf", 1800.0, 15.0 * OneAtm, "gas", bm_gas,
             "PT(S):0.7, O(S):0.3", "Blowers-Masel surface 1800 K, 15 atm")
    add_case("blowers-masel.yaml", "Pt_surf", 2200.0, 40.0 * OneAtm, "gas", bm_gas,
             "PT(S):0.85, O(S):0.15", "Blowers-Masel surface 2200 K, 40 atm")
    add_case("blowers-masel.yaml", "Pt_surf", 1100.0, OneAtm, "gas", bm_gas,
             "PT(S):0.5, H(S):0.3, O(S):0.2", "Blowers-Masel surface 1100 K, multipliers",
             multipliers={0: 0.8, 5: 1.5})
    add_case("blowers-masel.yaml", "Pt_surf", 500.0, 0.05 * OneAtm, "gas", bm_gas,
             "PT(S):0.9, H(S):0.1", "Blowers-Masel surface 500 K, 0.05 atm")

    # 5. sofc.yaml (metal_surface) - SOFC electrochemical surface kinetics
    # Gas species: H2, H2O, N2, O2
    # Surface species: (m), H(m), O(m), OH(m), H2O(m)
    sofc_gas1 = "H2:0.95, H2O:0.05"
    sofc_gas2 = "H2:0.50, H2O:0.40, N2:0.10"
    sofc_gas3 = "O2:0.21, N2:0.79"

    add_case("sofc.yaml", "metal_surface", 800.0, OneAtm, "gas", sofc_gas1,
             "(m):0.6, H(m):0.4", "SOFC metal_surface 800 K, phi=0.0 V", phi=0.0)
    add_case("sofc.yaml", "metal_surface", 1073.15, OneAtm, "gas", sofc_gas1,
             "(m):0.5, H(m):0.5", "SOFC metal_surface 1073.15 K, baseline phi=0.0 V", phi=0.0)
    add_case("sofc.yaml", "metal_surface", 1073.15, OneAtm, "gas", sofc_gas1,
             "(m):0.5, H(m):0.5", "SOFC metal_surface 1073.15 K, anodic phi=+0.5 V", phi=0.5)
    add_case("sofc.yaml", "metal_surface", 1073.15, OneAtm, "gas", sofc_gas1,
             "(m):0.5, H(m):0.5", "SOFC metal_surface 1073.15 K, cathodic phi=-0.5 V", phi=-0.5)
    add_case("sofc.yaml", "metal_surface", 1200.0, 3.0 * OneAtm, "gas", sofc_gas2,
             "(m):0.4, H(m):0.3, OH(m):0.2, H2O(m):0.1", "SOFC metal_surface 1200 K, phi=+0.8 V", phi=0.8)
    add_case("sofc.yaml", "metal_surface", 900.0, 0.2 * OneAtm, "gas", sofc_gas1,
             "(m):0.7, H(m):0.3", "SOFC metal_surface 900 K, phi=-0.3 V", phi=-0.3)
    add_case("sofc.yaml", "metal_surface", 1400.0, 10.0 * OneAtm, "gas", sofc_gas2,
             "(m):0.5, O(m):0.2, OH(m):0.2, H2O(m):0.1", "SOFC metal_surface 1400 K, phi=+0.2 V", phi=0.2)
    add_case("sofc.yaml", "metal_surface", 1000.0, OneAtm, "gas", sofc_gas1,
             "(m):0.5, H(m):0.5", "SOFC metal_surface 1000 K, multiplier",
             phi=0.0, multipliers={1: 2.0})
    add_case("sofc.yaml", "metal_surface", 1073.15, OneAtm, "gas", sofc_gas1,
             "(m):0.5, H(m):0.5", "SOFC metal_surface 1073.15 K, high potential phi=+1.2 V", phi=1.2)
    add_case("sofc.yaml", "metal_surface", 600.0, OneAtm, "gas", sofc_gas1,
             "(m):0.8, H(m):0.2", "SOFC metal_surface 600 K, startup phi=0.0 V", phi=0.0)

    out_file = ROOT / "audit" / "corpus_kinetics_family3.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} cases for Family 3 differential corpus: {out_file}")

if __name__ == "__main__":
    main()
