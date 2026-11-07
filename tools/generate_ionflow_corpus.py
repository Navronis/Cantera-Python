"""Generate deterministic differential validation corpus for IonFlow."""

import json
from pathlib import Path
import numpy as np

import cantera_python as ct

corpus_path = Path(__file__).resolve().parent.parent / "audit" / "corpus_ionflow.jsonl"


def generate():
    cases = []

    # -------------------------------------------------------------
    # Case 1-3: Neutral limit (h2o2.yaml)
    # -------------------------------------------------------------
    gas_neut = ct.Solution("h2o2.yaml")
    neut_species = list(gas_neut.species_names)

    # Case 1: Neutral free-flow
    grid1 = [0.0, 0.005, 0.012, 0.025, 0.04, 0.06]
    profile1 = []
    for j, z in enumerate(grid1):
        frac = j / (len(grid1) - 1)
        T = 300.0 + 1500.0 * frac
        u = 0.5 + 1.5 * frac
        Y = {
            "H2": 0.05 * (1.0 - frac) + 1e-6,
            "O2": 0.2 * (1.0 - frac) + 1e-6,
            "H2O": 0.25 * frac + 1e-6,
            "OH": 0.005 * np.sin(np.pi * frac) + 1e-6,
            "H": 0.001 * np.sin(np.pi * frac) + 1e-6,
            "O": 0.001 * np.sin(np.pi * frac) + 1e-6,
            "HO2": 1e-4 * (1.0 - frac) + 1e-6,
            "H2O2": 1e-4 * (1.0 - frac) + 1e-6,
        }
        sum_Y = sum(Y.values())
        Y["AR"] = 1.0 - sum_Y
        profile1.append({
            "j": j,
            "z": z,
            "u": u,
            "V": 0.0,
            "T": T,
            "Lambda": 0.0,
            "eField": 0.0,
            "Uo": 0.0,
            "Y": Y,
        })
    cases.append({
        "case_id": "ionflow_case_01_neut_free",
        "mechanism": "h2o2.yaml",
        "domain_type": "free-flow",
        "grid": grid1,
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": False,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": profile1,
    })

    # Case 2: Neutral unstrained-flow
    cases.append({
        "case_id": "ionflow_case_02_neut_unstrained",
        "mechanism": "h2o2.yaml",
        "domain_type": "unstrained-flow",
        "grid": grid1,
        "pressure": 202650.0,
        "energy_enabled": True,
        "electric_field_enabled": False,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": profile1,
    })

    # Case 3: Neutral axisymmetric-flow with Lambda
    profile3 = [dict(p) for p in profile1]
    for p in profile3:
        p["V"] = 2.0 * p["u"]
        p["Lambda"] = 15.0
    cases.append({
        "case_id": "ionflow_case_03_neut_axisymm",
        "mechanism": "h2o2.yaml",
        "domain_type": "axisymmetric-flow",
        "grid": grid1,
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": False,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": profile3,
    })

    # -------------------------------------------------------------
    # Cases 4-12: Ionized gas (ch4_ion.yaml)
    # -------------------------------------------------------------
    gas_ion = ct.Solution("ch4_ion.yaml")
    ion_species = list(gas_ion.species_names)

    grids = [
        [0.0, 0.004, 0.008, 0.012, 0.016, 0.02], # 6 pts
        [0.0, 0.005, 0.015, 0.03, 0.05],         # 5 pts
        [0.0, 0.002, 0.005, 0.009, 0.014, 0.02, 0.027, 0.035], # 8 pts
    ]

    def make_ion_profile(grid, efield_max=100.0, with_V=False, lam=0.0):
        prof = []
        n_pts = len(grid)
        for j, z in enumerate(grid):
            frac = j / (n_pts - 1)
            T = 300.0 + 1700.0 * frac
            u = 0.4 + 1.2 * frac
            V = 1.5 * u if with_V else 0.0
            eField = efield_max * np.sin(np.pi * frac)
            
            # Species mass fractions
            Y = {sp: 1e-8 for sp in ion_species}
            # Neutral reactants and products
            Y["CH4"] = 0.08 * (1.0 - frac) + 1e-6
            Y["O2"] = 0.18 * (1.0 - frac) + 1e-6
            Y["H2O"] = 0.15 * frac + 1e-6
            Y["CO2"] = 0.10 * frac + 1e-6
            Y["CO"] = 0.02 * np.sin(np.pi * frac) + 1e-6
            Y["H2"] = 0.01 * np.sin(np.pi * frac) + 1e-6
            Y["OH"] = 0.003 * np.sin(np.pi * frac) + 1e-6
            Y["H"] = 0.0005 * np.sin(np.pi * frac) + 1e-6
            Y["O"] = 0.0005 * np.sin(np.pi * frac) + 1e-6

            # Charged species
            Y["H3O+"] = 5.0e-6 * np.sin(np.pi * frac) + 1e-10
            Y["HCO+"] = 1.0e-6 * np.sin(np.pi * frac) + 1e-10
            Y["O2-"] = 1.0e-7 * np.sin(np.pi * frac) + 1e-10
            Y["E"] = 5.0e-8 * np.sin(np.pi * frac) + 1e-14

            sum_Y = sum(Y.values())
            Y["N2"] = max(0.0, 1.0 - sum_Y)

            prof.append({
                "j": j,
                "z": z,
                "u": u,
                "V": V,
                "T": T,
                "Lambda": lam,
                "eField": eField,
                "Uo": 0.0,
                "Y": Y,
            })
        return prof

    # Case 4: ch4_ion free-flow, electric field DISABLED (frozenIonMethod)
    cases.append({
        "case_id": "ionflow_case_04_ch4_frozen_free",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "free-flow",
        "grid": grids[0],
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": False,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[0], efield_max=0.0),
    })

    # Case 5: ch4_ion free-flow, electric field ENABLED (electricFieldMethod)
    cases.append({
        "case_id": "ionflow_case_05_ch4_efield_free",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "free-flow",
        "grid": grids[0],
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[0], efield_max=250.0),
    })

    # Case 6: ch4_ion axisymmetric-flow, electric field ENABLED
    cases.append({
        "case_id": "ionflow_case_06_ch4_efield_axisymm",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "axisymmetric-flow",
        "grid": grids[0],
        "pressure": 151987.5,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[0], efield_max=300.0, with_V=True, lam=20.0),
    })

    # Case 7: ch4_ion unstrained-flow, electric field ENABLED
    cases.append({
        "case_id": "ionflow_case_07_ch4_efield_unstrained",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "unstrained-flow",
        "grid": grids[1],
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[1], efield_max=150.0),
    })

    # Case 8: ch4_ion free-flow, transient evaluation (rdt > 0)
    cases.append({
        "case_id": "ionflow_case_08_ch4_efield_transient",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "free-flow",
        "grid": grids[0],
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": False,
        "rdt": 250.0,
        "state_profile": make_ion_profile(grids[0], efield_max=200.0),
    })

    # Case 9: ch4_ion free-flow, electron transport profile fitting enabled
    cases.append({
        "case_id": "ionflow_case_09_ch4_electron_transport_fitting",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "free-flow",
        "grid": grids[0],
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": True,
        "tfix": [300.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        "diff_e": [0.0001, 0.00025, 0.0007, 0.0013, 0.0020, 0.0028],
        "mobi_e": [0.4, 0.38, 0.35, 0.32, 0.30, 0.28],
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[0], efield_max=200.0),
    })

    # Case 10: ch4_ion free-flow, 8 points, high electric field, higher pressure
    cases.append({
        "case_id": "ionflow_case_10_ch4_high_efield_8pts",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "free-flow",
        "grid": grids[2],
        "pressure": 202650.0,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[2], efield_max=800.0),
    })

    # Case 11: ch4_ion axisymmetric, electron transport fitting + transient
    cases.append({
        "case_id": "ionflow_case_11_ch4_electron_trans_axisymm_transient",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "axisymmetric-flow",
        "grid": grids[0],
        "pressure": 101325.0,
        "energy_enabled": True,
        "electric_field_enabled": True,
        "electron_transport": True,
        "tfix": [300.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0],
        "diff_e": [0.0001, 0.00025, 0.0007, 0.0013, 0.0020, 0.0028],
        "mobi_e": [0.4, 0.38, 0.35, 0.32, 0.30, 0.28],
        "rdt": 100.0,
        "state_profile": make_ion_profile(grids[0], efield_max=350.0, with_V=True, lam=15.0),
    })

    # Case 12: ch4_ion fixed temperature (energy disabled)
    cases.append({
        "case_id": "ionflow_case_12_ch4_fixed_temperature",
        "mechanism": "ch4_ion.yaml",
        "domain_type": "free-flow",
        "grid": grids[0],
        "pressure": 101325.0,
        "energy_enabled": False,
        "electric_field_enabled": True,
        "electron_transport": False,
        "rdt": 0.0,
        "state_profile": make_ion_profile(grids[0], efield_max=200.0),
    })

    with open(corpus_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} cases in {corpus_path}")


if __name__ == "__main__":
    generate()
