import json
import random
import math
from pathlib import Path

random.seed(42)

def generate_adversarial_corpus(output_path="audit/corpus_wave4_adversarial_diff.jsonl"):
    cases = []
    case_id = 0

    # Species lists
    gri30_species = [
        "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "C", "CH", "CH2", "CH2(S)",
        "CH3", "CH4", "CO", "CO2", "HCO", "CH2O", "CH2OH", "CH3O", "CH3OH", "C2H", "C2H2",
        "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH", "N", "NH", "NH2", "NH3",
        "NNH", "NO", "NO2", "N2O", "HNO", "CN", "HCN", "H2CN", "HCNN", "HCNO", "HOCN", "HNCO",
        "NCO", "N2", "AR", "C3H7", "C3H8", "CH2CHO", "CH3CHO"
    ]
    h2o2_species = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"]

    # 1. Deep Trace Species (500 cases)
    for _ in range(500):
        mech = "gri30.yaml" if random.random() < 0.7 else "h2o2.yaml"
        sp_list = gri30_species if mech == "gri30.yaml" else h2o2_species
        nsp = len(sp_list)
        
        # Pick 1 or 2 carrier species
        carrier = random.choice(["N2", "AR", "O2"])
        X = {sp: 0.0 for sp in sp_list}
        X[carrier] = 1.0 - 1e-15
        
        # Add deep trace species with 1e-25 or 0.0
        for sp in sp_list:
            if sp != carrier and random.random() < 0.4:
                X[sp] = random.choice([0.0, 1e-25, 1e-22, 1e-18, 1e-15])
        
        # Normalize
        total = sum(X.values())
        X = {k: v / total for k, v in X.items()}

        T = random.uniform(300.0, 2500.0)
        P = random.uniform(0.5, 30.0) * 101325.0
        grad_T = random.uniform(-1e4, 1e4)
        grad_X = {sp: random.uniform(-10.0, 10.0) if X[sp] > 1e-10 else 0.0 for sp in sp_list}

        cases.append({
            "case_id": case_id,
            "category": "trace_species",
            "mech": mech,
            "T": T,
            "P": P,
            "X": X,
            "grad_T": grad_T,
            "grad_X": grad_X
        })
        case_id += 1

    # 2. Pure Species Limits (500 cases)
    for _ in range(500):
        mech = "gri30.yaml" if random.random() < 0.7 else "h2o2.yaml"
        sp_list = gri30_species if mech == "gri30.yaml" else h2o2_species
        pure_sp = random.choice(sp_list)
        X = {sp: 1.0 if sp == pure_sp else 0.0 for sp in sp_list}

        T = random.uniform(300.0, 3000.0)
        P = random.uniform(0.1, 50.0) * 101325.0
        grad_T = random.uniform(-5e3, 5e3)
        grad_X = {sp: 0.0 for sp in sp_list}

        cases.append({
            "case_id": case_id,
            "category": "pure_species_limit",
            "mech": mech,
            "T": T,
            "P": P,
            "X": X,
            "grad_T": grad_T,
            "grad_X": grad_X
        })
        case_id += 1

    # 3. Extreme MW Ratios (500 cases)
    for _ in range(500):
        mech = "gri30.yaml"
        sp_list = gri30_species
        light = "H2" # MW ~ 2
        heavy = random.choice(["C3H8", "AR", "CO2"]) # MW 40-44
        ratio = 10.0 ** random.uniform(-6, 6)
        x_light = ratio / (1.0 + ratio)
        x_heavy = 1.0 - x_light
        X = {sp: 0.0 for sp in sp_list}
        X[light] = x_light
        X[heavy] = x_heavy

        T = random.uniform(300.0, 2500.0)
        P = random.uniform(0.1, 40.0) * 101325.0
        grad_T = random.uniform(-1e4, 1e4)
        grad_X = {sp: 0.0 for sp in sp_list}
        grad_X[light] = random.uniform(-5.0, 5.0)
        grad_X[heavy] = -grad_X[light]

        cases.append({
            "case_id": case_id,
            "category": "extreme_mw_ratio",
            "mech": mech,
            "T": T,
            "P": P,
            "X": X,
            "grad_T": grad_T,
            "grad_X": grad_X
        })
        case_id += 1

    # 4. Temperature Extremes (500 cases)
    for _ in range(500):
        mech = "gri30.yaml" if random.random() < 0.5 else "h2o2.yaml"
        sp_list = gri30_species if mech == "gri30.yaml" else h2o2_species
        T = random.choice([
            random.uniform(200.0, 300.0),
            random.uniform(3000.0, 4500.0)
        ])
        P = random.uniform(0.1, 50.0) * 101325.0
        
        # Typical mixture
        X = {sp: random.uniform(0.01, 1.0) if sp in ["N2", "O2", "H2", "CH4", "CO2", "H2O"] else 0.0 for sp in sp_list}
        tot = sum(X.values())
        X = {k: v / tot for k, v in X.items()}

        grad_T = random.uniform(-2e4, 2e4)
        grad_X = {sp: random.uniform(-2.0, 2.0) if X[sp] > 0.01 else 0.0 for sp in sp_list}

        cases.append({
            "case_id": case_id,
            "category": "temperature_extreme",
            "mech": mech,
            "T": T,
            "P": P,
            "X": X,
            "grad_T": grad_T,
            "grad_X": grad_X
        })
        case_id += 1

    # 5. Pressure Extremes (500 cases)
    for _ in range(500):
        mech = "gri30.yaml" if random.random() < 0.5 else "h2o2.yaml"
        sp_list = gri30_species if mech == "gri30.yaml" else h2o2_species
        # 1e-3 atm to 0.05 atm OR 20 atm to 100 atm
        P_atm = random.choice([
            random.uniform(1e-3, 0.05),
            random.uniform(20.0, 100.0)
        ])
        P = P_atm * 101325.0
        T = random.uniform(300.0, 2500.0)
        
        X = {sp: random.uniform(0.01, 1.0) if sp in ["N2", "O2", "H2", "CH4", "AR"] else 0.0 for sp in sp_list}
        tot = sum(X.values())
        X = {k: v / tot for k, v in X.items()}

        grad_T = random.uniform(-5e3, 5e3)
        grad_X = {sp: random.uniform(-2.0, 2.0) if X[sp] > 0.01 else 0.0 for sp in sp_list}

        cases.append({
            "case_id": case_id,
            "category": "pressure_extreme",
            "mech": mech,
            "T": T,
            "P": P,
            "X": X,
            "grad_T": grad_T,
            "grad_X": grad_X
        })
        case_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} adversarial differential cases in {output_path}!")

if __name__ == "__main__":
    generate_adversarial_corpus()
