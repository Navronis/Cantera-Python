#!/usr/bin/env python3
"""Generate comprehensive differential corpus for Family 2:
audit/corpus_kinetics_family2.jsonl
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    cases = []
    case_idx = 1

    def add_case(mech, phase, T, P, X, desc, multipliers=None):
        nonlocal case_idx
        cid = f"fam2_{case_idx:03d}_{Path(mech).stem}"
        cases.append({
            "case_id": cid,
            "mechanism": mech,
            "phase": phase or "",
            "temperature": float(T),
            "pressure": float(P),
            "mole_fractions": X,
            "multipliers": multipliers or {},
            "description": desc
        })
        case_idx += 1

    OneAtm = 101325.0

    # 1. SRI Falloff (sri-falloff.yaml)
    # Species: H, R1A, R1B, P1, R2, P2A, P2B, R3
    sri_X = "H:0.1, R1A:0.2, R1B:0.2, P1:0.1, R2:0.1, P2A:0.1, P2B:0.1, R3:0.1"
    add_case("sri-falloff.yaml", "", 300.0, 100.0, sri_X, "SRI falloff low T, low P asymptote")
    add_case("sri-falloff.yaml", "", 700.0, OneAtm, sri_X, "SRI falloff intermediate T, 1 atm")
    add_case("sri-falloff.yaml", "", 1200.0, 10.0 * OneAtm, sri_X, "SRI falloff high T, 10 atm")
    add_case("sri-falloff.yaml", "", 2200.0, 50.0 * OneAtm, sri_X, "SRI falloff very high T, high P limit")
    add_case("sri-falloff.yaml", "", 1000.0, 0.05 * OneAtm, sri_X, "SRI falloff intermediate T, low P", multipliers={0: 2.5})

    # 2. Tsang Falloff (tsang-falloff.yaml)
    # Species: H, NO, OH, CN, HCN, CO2, H2O, N2, N2O, HONO
    tsang_X = "H:0.05, NO:0.1, OH:0.1, CN:0.05, HCN:0.1, CO2:0.1, H2O:0.1, N2:0.3, N2O:0.05, HONO:0.05"
    add_case("tsang-falloff.yaml", "", 350.0, 500.0, tsang_X, "Tsang falloff low T, low P")
    add_case("tsang-falloff.yaml", "", 800.0, OneAtm, tsang_X, "Tsang falloff intermediate T, 1 atm")
    add_case("tsang-falloff.yaml", "", 1400.0, 5.0 * OneAtm, tsang_X, "Tsang falloff moderate T, 5 atm")
    add_case("tsang-falloff.yaml", "", 2400.0, 25.0 * OneAtm, tsang_X, "Tsang falloff high T, 25 atm")

    # 3. Chemically Activated Falloff (chemically-activated-reaction.yaml)
    # Species: ch3, oh, ch2o, h2, n2
    chem_X = "ch3:0.2, oh:0.2, ch2o:0.1, h2:0.1, n2:0.4"
    add_case("chemically-activated-reaction.yaml", "", 300.0, 100.0, chem_X, "Chemically activated low T, low P limit")
    add_case("chemically-activated-reaction.yaml", "", 800.0, 0.5 * OneAtm, chem_X, "Chemically activated intermediate T, 0.5 atm")
    add_case("chemically-activated-reaction.yaml", "", 1400.0, OneAtm, chem_X, "Chemically activated intermediate T, 1 atm")
    add_case("chemically-activated-reaction.yaml", "", 2000.0, 10.0 * OneAtm, chem_X, "Chemically activated high T, 10 atm")
    add_case("chemically-activated-reaction.yaml", "", 2800.0, 100.0 * OneAtm, chem_X, "Chemically activated high T, high P limit")

    # 4. P-log and Chebyshev (pdep-test.yaml)
    # Species: R1A, R1B, P1, H, R2, P2A, P2B, R3, P3A, P3B, R4, P4, R5, P5A, P5B, R6, P6A, P6B, R7, P7A, P7B
    pdep_X = "R1A:0.05, R1B:0.05, P1:0.05, H:0.05, R2:0.05, P2A:0.05, P2B:0.05, R3:0.05, P3A:0.05, P3B:0.05, R4:0.05, P4:0.05, R5:0.05, P5A:0.05, P5B:0.05, R6:0.05, P6A:0.05, P6B:0.05, R7:0.05, P7A:0.05, P7B:0.05"
    add_case("pdep-test.yaml", "", 400.0, 1000.0, pdep_X, "P-log/Chebyshev low T, P < knot 1")
    add_case("pdep-test.yaml", "", 600.0, 0.01 * OneAtm, pdep_X, "P-log/Chebyshev exact knot 1 pressure")
    add_case("pdep-test.yaml", "", 1000.0, 0.1 * OneAtm, pdep_X, "P-log/Chebyshev between knots 1 and 2")
    add_case("pdep-test.yaml", "", 1200.0, OneAtm, pdep_X, "P-log/Chebyshev exact 1 atm knot")
    add_case("pdep-test.yaml", "", 1500.0, 5.0 * OneAtm, pdep_X, "P-log/Chebyshev between knots 2 and 3")
    add_case("pdep-test.yaml", "", 1800.0, 10.0 * OneAtm, pdep_X, "P-log/Chebyshev exact 10 atm knot")
    add_case("pdep-test.yaml", "", 2200.0, 100.0 * OneAtm, pdep_X, "P-log/Chebyshev exact 100 atm knot")
    add_case("pdep-test.yaml", "", 2600.0, 200.0 * OneAtm, pdep_X, "P-log/Chebyshev P > knot N extrapolation")
    add_case("pdep-test.yaml", "", 1100.0, 2.5 * OneAtm, pdep_X, "P-log/Chebyshev with multiplier perturbation", multipliers={0: 1.5, 4: 0.8})

    # 5. Blowers-Masel (blowers-masel.yaml, gas phase)
    # Species: H2, H, O, O2, OH, H2O, CH4, AR, N2
    bm_X = "O:0.05, H2:0.4, O2:0.1, OH:0.05, H2O:0.1, CH4:0.05, AR:0.1, N2:0.15"
    add_case("blowers-masel.yaml", "gas", 300.0, OneAtm, bm_X, "Blowers-Masel low T, 1 atm")
    add_case("blowers-masel.yaml", "gas", 600.0, 0.1 * OneAtm, bm_X, "Blowers-Masel moderate T, low P")
    add_case("blowers-masel.yaml", "gas", 1100.0, 0.1 * OneAtm, bm_X, "Blowers-Masel intermediate T (tested in C++ unit test)")
    add_case("blowers-masel.yaml", "gas", 1600.0, OneAtm, bm_X, "Blowers-Masel high T, 1 atm")
    add_case("blowers-masel.yaml", "gas", 2500.0, 10.0 * OneAtm, bm_X, "Blowers-Masel very high T, 10 atm")

    # 6. Linear Burke (linearBurke-test.yaml)
    # Species: H, OH, H2O, O2, HO2, H2O2
    lb_X1 = "H:0.1, OH:0.1, H2O:0.3, O2:0.2, HO2:0.1, H2O2:0.2"
    lb_X2 = "H:0.01, OH:0.01, H2O:0.8, O2:0.1, HO2:0.04, H2O2:0.04"
    add_case("linearBurke-test.yaml", "", 500.0, 0.1 * OneAtm, lb_X1, "Linear Burke low T, 0.1 atm, mixed colliders")
    add_case("linearBurke-test.yaml", "", 1000.0, OneAtm, lb_X1, "Linear Burke intermediate T, 1 atm")
    add_case("linearBurke-test.yaml", "", 1500.0, 5.0 * OneAtm, lb_X2, "Linear Burke high T, 5 atm, steam-dominated")
    add_case("linearBurke-test.yaml", "", 2200.0, 20.0 * OneAtm, lb_X1, "Linear Burke high T, 20 atm")

    # 7. Lindemann & Troe Falloff in Comprehensive Mechanism (gri30.yaml)
    # Lean methane, stoichiometric, rich, syngas, extreme pressures
    add_case("gri30.yaml", "", 300.0, 50.0, "CH4:0.095, O2:0.19, N2:0.715", "GRI30 Lindemann/Troe extreme low P (50 Pa)")
    add_case("gri30.yaml", "", 600.0, 0.1 * OneAtm, "CH4:0.05, O2:0.2, N2:0.75", "GRI30 Lindemann/Troe lean 0.1 atm")
    add_case("gri30.yaml", "", 1000.0, OneAtm, "CH4:0.095, O2:0.19, N2:0.715", "GRI30 Lindemann/Troe stoich 1 atm")
    add_case("gri30.yaml", "", 1400.0, 5.0 * OneAtm, "CH4:0.15, O2:0.15, N2:0.70", "GRI30 Lindemann/Troe rich 5 atm")
    add_case("gri30.yaml", "", 1800.0, 20.0 * OneAtm, "H2:0.3, CO:0.2, O2:0.15, N2:0.35", "GRI30 Lindemann/Troe syngas 20 atm")
    add_case("gri30.yaml", "", 2400.0, 60.0 * OneAtm, "CH4:0.095, O2:0.19, N2:0.715", "GRI30 Lindemann/Troe high P flame condition")
    add_case("gri30.yaml", "", 3200.0, 100.0 * OneAtm, "CH4:0.095, O2:0.19, N2:0.715", "GRI30 Lindemann/Troe extreme 100 atm limit")

    out_file = ROOT / "audit" / "corpus_kinetics_family2.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} Family 2 differential cases -> {out_file}")

if __name__ == "__main__":
    main()
