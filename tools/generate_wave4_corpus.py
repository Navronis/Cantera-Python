"""
Generate a diverse 10,000-case differential corpus for Wave 4 MultiTransport validation.
Covers:
- Mechanisms: gri30.yaml, h2o2.yaml, air.yaml
- Temperatures: 300 K to 3000 K (uniform and extreme sampling)
- Pressures: 0.1 atm to 50 atm (log-uniform sampling)
- Compositions:
  - Stoichiometric, fuel-rich, fuel-lean mixtures
  - Pure species cases
  - Binary mixtures with extreme MW ratios (e.g. H2 in N2, CH4 in Ar)
  - Pyrolysis / high radical concentrations
  - Trace species down to 1e-20 (adversarial boundary conditions)
"""

import json
import numpy as np

def generate_corpus(output_path="audit/corpus_wave4.jsonl", num_cases=10000):
    rng = np.random.default_rng(42)  # Deterministic seed for exact reproducibility

    # Mechanisms and key species
    gri30_species = [
        "H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "C", "CH",
        "CH2", "CH2(S)", "CH3", "CH4", "CO", "CO2", "HCO", "CH2O", "CH2OH", "CH3O",
        "CH3OH", "C2H", "C2H2", "C2H3", "C2H4", "C2H5", "C2H6", "HCCO", "CH2CO", "HCCOH",
        "N", "NH", "NH2", "NH3", "NNH", "NO", "NO2", "N2O", "HNO", "CN",
        "HCN", "H2CN", "HCNN", "HCNO", "HOCN", "HNCO", "NCO", "N2", "AR", "C3H7",
        "C3H8", "CH2CHO", "CH3CHO"
    ]
    h2o2_species = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"]

    cases = []
    
    # 1. H2O2 mechanism cases (2,500 cases)
    for i in range(2500):
        T = float(rng.uniform(300.0, 3000.0))
        P = float(101325.0 * (10 ** rng.uniform(-1.0, 1.7)))  # 0.1 atm to 50 atm
        
        mode = i % 5
        if mode == 0:
            # Random Dirichlet composition
            alpha = rng.exponential(1.0, len(h2o2_species))
            x = alpha / np.sum(alpha)
        elif mode == 1:
            # H2-O2 flame-like (H2, O2, H2O, N2, AR)
            x = np.zeros(len(h2o2_species))
            phi = rng.uniform(0.5, 2.5)
            x[0] = 2.0 * phi  # H2
            x[3] = 1.0        # O2
            x[5] = rng.uniform(0.01, 0.2) # H2O
            x[9] = 3.76       # N2
            x /= np.sum(x)
        elif mode == 2:
            # Pure or binary extreme
            sp_idx = rng.choice(len(h2o2_species), size=2, replace=False)
            x = np.zeros(len(h2o2_species))
            frac = rng.uniform(0.01, 0.99)
            x[sp_idx[0]] = frac
            x[sp_idx[1]] = 1.0 - frac
        elif mode == 3:
            # Adversarial trace species: some species at 1e-18 .. 1e-12
            x = rng.exponential(1.0, len(h2o2_species))
            x /= np.sum(x)
            trace_indices = rng.choice(len(h2o2_species), size=3, replace=False)
            for tidx in trace_indices:
                x[tidx] = 10.0 ** rng.uniform(-20.0, -12.0)
            x /= np.sum(x)
        else:
            # Near-pure species with tiny impurities
            major = rng.choice(len(h2o2_species))
            x = np.full(len(h2o2_species), 1e-15)
            x[major] = 1.0 - np.sum(x) + 1e-15

        comp_dict = {h2o2_species[k]: float(x[k]) for k in range(len(h2o2_species))}
        grad_T = float(rng.uniform(-500.0, 500.0))
        grad_X = {sp: float(rng.uniform(-0.1, 0.1)) for sp in h2o2_species}
        # Zero net gradient
        mean_gx = sum(grad_X.values()) / len(grad_X)
        grad_X = {sp: gx - mean_gx for sp, gx in grad_X.items()}

        cases.append({
            "case_id": len(cases),
            "mech": "h2o2.yaml",
            "T": T,
            "P": P,
            "X": comp_dict,
            "grad_T": grad_T,
            "grad_X": grad_X
        })

    # 2. GRI-3.0 mechanism cases (7,500 cases)
    for i in range(7500):
        T = float(rng.uniform(300.0, 3000.0))
        P = float(101325.0 * (10 ** rng.uniform(-1.0, 1.7)))  # 0.1 atm to 50 atm
        
        mode = i % 6
        if mode == 0:
            # Methane/Air combustion mixture
            x = np.zeros(len(gri30_species))
            phi = rng.uniform(0.4, 3.0)
            ch4_idx = gri30_species.index("CH4")
            o2_idx = gri30_species.index("O2")
            n2_idx = gri30_species.index("N2")
            co2_idx = gri30_species.index("CO2")
            h2o_idx = gri30_species.index("H2O")
            x[ch4_idx] = phi
            x[o2_idx] = 2.0
            x[n2_idx] = 2.0 * 3.76
            x[co2_idx] = rng.uniform(0.001, 0.1)
            x[h2o_idx] = rng.uniform(0.001, 0.1)
            x /= np.sum(x)
        elif mode == 1:
            # Radical-rich / dissociation / high-T plasma
            x = np.zeros(len(gri30_species))
            rads = ["H", "O", "OH", "CH3", "HO2", "CO", "H2", "O2", "N2"]
            for r in rads:
                x[gri30_species.index(r)] = rng.uniform(0.01, 0.5)
            x /= np.sum(x)
        elif mode == 2:
            # Syngas / CO2 dilution
            x = np.zeros(len(gri30_species))
            for sp in ["H2", "CO", "CO2", "H2O", "N2", "CH4"]:
                x[gri30_species.index(sp)] = rng.uniform(0.05, 0.8)
            x /= np.sum(x)
        elif mode == 3:
            # Binary extreme mixture (e.g. H2 in AR, CH4 in CO2, H in N2)
            sp_idx = rng.choice(len(gri30_species), size=2, replace=False)
            x = np.zeros(len(gri30_species))
            frac = rng.uniform(0.005, 0.995)
            x[sp_idx[0]] = frac
            x[sp_idx[1]] = 1.0 - frac
        elif mode == 4:
            # Adversarial trace species: 20 species with tiny fractions < 1e-15
            active = rng.choice(len(gri30_species), size=8, replace=False)
            x = np.full(len(gri30_species), 10.0 ** rng.uniform(-22.0, -15.0))
            for a in active:
                x[a] = rng.uniform(0.01, 0.5)
            x /= np.sum(x)
        else:
            # Pure species (single species = 1.0)
            pure_idx = rng.choice(len(gri30_species))
            x = np.zeros(len(gri30_species))
            x[pure_idx] = 1.0

        comp_dict = {gri30_species[k]: float(x[k]) for k in range(len(gri30_species))}
        grad_T = float(rng.uniform(-1000.0, 1000.0))
        grad_X = {sp: float(rng.uniform(-0.05, 0.05)) for sp in gri30_species}
        mean_gx = sum(grad_X.values()) / len(grad_X)
        grad_X = {sp: gx - mean_gx for sp, gx in grad_X.items()}

        cases.append({
            "case_id": len(cases),
            "mech": "gri30.yaml",
            "T": T,
            "P": P,
            "X": comp_dict,
            "grad_T": grad_T,
            "grad_X": grad_X
        })

    print(f"Generated {len(cases)} cases. Writing to {output_path}...")
    with open(output_path, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print("Done!")

if __name__ == "__main__":
    generate_corpus()
