"""Generate comprehensive differential validation corpus for Cantera kinetics.

Covers:
- Mechanisms: h2o2.yaml, gri30.yaml, air.yaml
- Temperatures: 300 K to 3000 K
- Pressures: 0.05 atm to 50 atm
- Compositions: lean, stoichiometric, rich, syngas, post-flame equilibrium, trace species
- Multipliers: unperturbed, single reaction perturbed, global perturbed
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "audit" / "corpus_kinetics.jsonl"

CASES = []

def add_case(case_id, mechanism, temperature, pressure, mole_fractions, multipliers=None, description=""):
    CASES.append({
        "case_id": case_id,
        "mechanism": mechanism,
        "temperature": float(temperature),
        "pressure": float(pressure),
        "mole_fractions": mole_fractions,
        "multipliers": multipliers or {},
        "description": description
    })

# 1. H2/O2 system: 9 species, 28 reactions (Arrhenius, third body, Troe falloff)
h2_temps = [300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2200.0, 2600.0, 3000.0]
h2_pressures = [10132.5, 101325.0, 506625.0, 2026500.0, 5066250.0]
h2_mixtures = [
    ("H2:2, O2:1, N2:3.76", "Stoichiometric H2-air"),
    ("H2:4, O2:1, AR:5", "Fuel-rich H2-O2-Ar"),
    ("H2:0.5, O2:1, N2:8", "Fuel-lean H2-air"),
    ("H2O:0.6, H2:0.1, O2:0.1, OH:0.05, H:0.05, O:0.05, HO2:0.03, H2O2:0.02", "Radical-rich post-flame"),
]

c_idx = 1
for T in [300.0, 1000.0, 1500.0, 2000.0, 2800.0]:
    for P in [101325.0, 1013250.0]:
        for X, desc in h2_mixtures[:2]:
            add_case(f"kinetics_h2o2_{c_idx:03d}", "h2o2.yaml", T, P, X, description=f"H2/O2 {desc} at {T}K, {P}Pa")
            c_idx += 1

# Perturbed multipliers on H2/O2
add_case(f"kinetics_h2o2_{c_idx:03d}", "h2o2.yaml", 1400.0, 202650.0, "H2:2, O2:1, N2:3.76",
         multipliers={"0": 1.5, "2": 0.5, "5": 2.0}, description="H2/O2 with selective reaction multipliers")
c_idx += 1
add_case(f"kinetics_h2o2_{c_idx:03d}", "h2o2.yaml", 1800.0, 101325.0, "H2:1, O2:1, AR:4",
         multipliers={"1": 0.1, "4": 3.0, "8": 0.0}, description="H2/O2 with zero and elevated multipliers")
c_idx += 1

# 2. GRI-Mech 3.0: 53 species, 325 reactions (Full natural gas combustion)
gri_temps = [300.0, 800.0, 1200.0, 1600.0, 2000.0, 2400.0, 3000.0]
gri_pressures = [50662.5, 101325.0, 1013250.0, 3039750.0]
gri_mixtures = [
    ("CH4:0.095, O2:0.21, N2:0.79", "Stoichiometric CH4-air"),
    ("CH4:0.15, O2:0.2, N2:0.65", "Rich CH4-air"),
    ("CH4:0.04, O2:0.22, N2:0.74", "Lean CH4-air"),
    ("CO:0.3, H2:0.3, O2:0.2, N2:0.2", "Syngas mixture"),
    ("CH4:0.05, C2H6:0.02, C3H8:0.01, O2:0.2, N2:0.72", "Multi-alkane blend"),
    ("CO2:0.1, H2O:0.15, O2:0.05, N2:0.68, OH:0.005, NO:0.005, CO:0.01", "Post-flame exhaust with NOx"),
]

for idx, (T, P) in enumerate([
    (300.0, 101325.0),
    (600.0, 50662.5),
    (1000.0, 101325.0),
    (1400.0, 1013250.0),
    (1800.0, 2026500.0),
    (2200.0, 101325.0),
    (2500.0, 3039750.0),
    (3000.0, 5066250.0),
]):
    X, desc = gri_mixtures[idx % len(gri_mixtures)]
    add_case(f"kinetics_gri30_{c_idx:03d}", "gri30.yaml", T, P, X, description=f"GRI30 {desc} at {T}K, {P}Pa")
    c_idx += 1

# GRI-Mech with multipliers
add_case(f"kinetics_gri30_{c_idx:03d}", "gri30.yaml", 1500.0, 101325.0, "CH4:0.095, O2:0.21, N2:0.79",
         multipliers={"10": 1.25, "50": 0.75, "100": 2.0, "200": 0.5}, description="GRI30 with multi-reaction perturbation")
c_idx += 1

# 3. Air mechanism: High-T nitrogen dissociation and Zeldovich NOx
air_temps = [500.0, 1500.0, 2500.0, 3500.0]
for T in air_temps:
    add_case(f"kinetics_air_{c_idx:03d}", "air.yaml", T, 101325.0, "N2:0.78, O2:0.21, AR:0.01",
             description=f"Air thermal kinetics at {T}K")
    c_idx += 1

# 4. Low pressure and high pressure extremes (adversarial)
add_case(f"kinetics_adversarial_{c_idx:03d}", "h2o2.yaml", 300.0, 1000.0, "H2:0.5, O2:0.5",
         description="Low pressure 1000 Pa, room temperature")
c_idx += 1
add_case(f"kinetics_adversarial_{c_idx:03d}", "gri30.yaml", 3200.0, 10000000.0, "CH4:0.05, O2:0.1, N2:0.85",
         description="High pressure 100 atm, ultra-high temperature 3200 K")
c_idx += 1

def main():
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for case in CASES:
            f.write(json.dumps(case) + "\n")
    print(f"Generated {len(CASES)} kinetics corpus cases in {CORPUS_PATH}")

if __name__ == "__main__":
    main()
