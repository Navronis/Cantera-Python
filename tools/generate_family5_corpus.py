"""Generate the deterministic differential validation corpus for Family 5:
Plasma, Vibrational Relaxation, MultiRate, Delegators, and Kinetics Derivatives.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "audit" / "corpus_kinetics_family5.jsonl"


def generate_corpus():
    cases = []

    # Category 1: Vibrational Relaxation (10 cases)
    # Testing constant, Starikovskiy, Castela, and multi-state-resolved vibrational relaxation rates
    temps = [300.0, 500.0, 800.0, 1000.0, 1200.0, 1500.0, 1800.0, 2200.0, 2600.0, 3000.0]
    pressures = [101325.0, 50662.5, 202650.0, 101325.0, 1013250.0, 101325.0, 506625.0, 101325.0, 2026500.0, 101325.0]
    for idx, (T, P) in enumerate(zip(temps, pressures), 1):
        cases.append({
            "case_id": f"fam5_vib_{idx:03d}_T{int(T)}_P{int(P)}",
            "category": "vibrational_relaxation",
            "mechanism": "vibrational-relaxation.yaml",
            "temperature": T,
            "pressure": P,
            "mole_fractions": "N2:0.7, O2:0.2, N2(v1):0.1",
        })

    # Category 2: Two-Temperature Plasma (10 cases)
    # Testing TwoTempPlasmaRate with varied T (gas temp) and Te (electron temp)
    plasma_t = [300.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 1800.0, 2000.0, 2500.0]
    plasma_te = [11604.5, 23209.0, 34813.5, 46418.0, 58022.5, 11604.5, 23209.0, 34813.5, 46418.0, 58022.5] # 1eV to 5eV
    for idx, (T, Te) in enumerate(zip(plasma_t, plasma_te), 1):
        cases.append({
            "case_id": f"fam5_plasma2t_{idx:03d}_T{int(T)}_Te{int(Te)}",
            "category": "two_temp_plasma",
            "mechanism": "oxygen-plasma.yaml",
            "temperature": T,
            "electron_temperature": Te,
            "pressure": 101325.0,
            "mole_fractions": "O2:0.9, E:0.05, O2-:0.05",
        })

    # Category 3: Air Plasma & Electron Collision Plasma (5 cases)
    for idx, T in enumerate([300.0, 500.0, 1000.0, 1500.0, 2000.0], 1):
        cases.append({
            "case_id": f"fam5_airplasma_{idx:03d}_T{int(T)}",
            "category": "air_plasma",
            "mechanism": "air-plasma.yaml",
            "temperature": T,
            "electron_temperature": 23209.0,
            "pressure": 101325.0,
            "mole_fractions": "N2:0.78, O2:0.21, Electron:0.01",
        })

    # Category 4: CustomFunc1Rate & Delegators (5 cases)
    custom_temps = [300.0, 500.0, 1000.0, 1500.0, 2000.0]
    for idx, T in enumerate(custom_temps, 1):
        cases.append({
            "case_id": f"fam5_custom_delegator_{idx:03d}_T{int(T)}",
            "category": "custom_delegator",
            "temperature": T,
            "pressure": 101325.0,
        })

    # Category 5: MultiRate container operations (5 cases)
    multirate_types = ["Arrhenius", "Blowers-Masel", "two-temperature-plasma", "constant-vibrational-relaxation", "custom-rate-function"]
    for idx, rtype in enumerate(multirate_types, 1):
        cases.append({
            "case_id": f"fam5_multirate_{idx:03d}_{rtype.replace('-', '_')}",
            "category": "multirate_ops",
            "rate_type": rtype,
            "temperature": 1000.0,
        })

    # Category 6: Kinetics derivative evaluators (10 cases)
    # getCreationRates_ddT, getDestructionRates_ddT, getNetProductionRates_ddT, ddP, ddC
    mechs = ["h2o2.yaml", "gri30.yaml", "vibrational-relaxation.yaml", "h2o2.yaml", "gri30.yaml",
             "h2o2.yaml", "gri30.yaml", "vibrational-relaxation.yaml", "h2o2.yaml", "gri30.yaml"]
    deriv_temps = [500.0, 800.0, 1000.0, 1200.0, 1500.0, 1800.0, 2000.0, 2200.0, 2500.0, 3000.0]
    deriv_press = [101325.0, 202650.0, 101325.0, 506625.0, 1013250.0, 101325.0, 202650.0, 101325.0, 506625.0, 101325.0]
    for idx, (mech, T, P) in enumerate(zip(mechs, deriv_temps, deriv_press), 1):
        cases.append({
            "case_id": f"fam5_deriv_{idx:03d}_{mech.replace('.yaml', '')}_T{int(T)}",
            "category": "kinetics_derivatives",
            "mechanism": mech,
            "temperature": T,
            "pressure": P,
            "mole_fractions": "H2:0.5, O2:0.5" if "h2o2" in mech else ("CH4:0.1, O2:0.2, N2:0.7" if "gri30" in mech else "N2:0.7, O2:0.2, N2(v1):0.1"),
        })

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} test cases in {CORPUS_PATH}")


if __name__ == "__main__":
    generate_corpus()
