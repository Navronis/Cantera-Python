"""Generate the deterministic differential validation corpus for Family 4:
Reaction Topology, ReactionPath Analysis, and Kinetics/ReactionRate Factories.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "audit" / "corpus_kinetics_family4.jsonl"


def generate_corpus():
    cases = []

    # 1. ReactionPathDiagram cases (20 cases)
    # -------------------------------------------------------------
    # h2o2.yaml cases
    cases.append({
        "case_id": "fam4_001_h2o2_H_default",
        "category": "reaction_path_diagram",
        "mechanism": "h2o2.yaml",
        "element": "H",
        "temperature": 1100.0,
        "pressure": 1013250.0,
        "mole_fractions": "H:0.1, HO2:0.1, AR:6.0",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_002_h2o2_O_default",
        "category": "reaction_path_diagram",
        "mechanism": "h2o2.yaml",
        "element": "O",
        "temperature": 1200.0,
        "pressure": 101325.0,
        "mole_fractions": "H2:0.2, O2:0.1, AR:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })

    # gri30.yaml cases
    cases.append({
        "case_id": "fam4_003_gri30_C_net",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_004_gri30_H_net",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "H",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_005_gri30_O_net",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "O",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_006_gri30_N_net",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "N",
        "temperature": 1800.0,
        "pressure": 506625.0,
        "mole_fractions": "CH4:0.05, O2:0.1, N2:0.85",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_007_gri30_C_oneway",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "OneWayFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_008_gri30_C_unscaled",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": 1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_009_gri30_C_oneway_unscaled",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "OneWayFlow",
        "scale": 1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_010_gri30_C_high_threshold",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.05,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_011_gri30_C_included",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": ["CH4", "CO", "CO2"],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_012_gri30_C_excluded",
        "category": "reaction_path_diagram",
        "mechanism": "gri30.yaml",
        "element": "C",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": ["CH2O", "CH3OH"],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_013_h2o2_H_display_only",
        "category": "reaction_path_diagram",
        "mechanism": "h2o2.yaml",
        "element": "H",
        "temperature": 1100.0,
        "pressure": 1013250.0,
        "mole_fractions": "H:0.1, HO2:0.1, AR:6.0",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": 1  # species index for H
    })

    # Surface & diverse mechanisms
    cases.append({
        "case_id": "fam4_014_ptcombust_C",
        "category": "reaction_path_diagram",
        "mechanism": "ptcombust.yaml",
        "phase": "gas",
        "element": "C",
        "temperature": 1000.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, AR:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_015_ptcombust_H",
        "category": "reaction_path_diagram",
        "mechanism": "ptcombust.yaml",
        "phase": "gas",
        "element": "H",
        "temperature": 1000.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, AR:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_016_ptcombust_O",
        "category": "reaction_path_diagram",
        "mechanism": "ptcombust.yaml",
        "phase": "gas",
        "element": "O",
        "temperature": 1000.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, AR:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_017_gri30_ion_H",
        "category": "reaction_path_diagram",
        "mechanism": "gri30_ion.yaml",
        "element": "H",
        "temperature": 1500.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_018_nDodecane_C",
        "category": "reaction_path_diagram",
        "mechanism": "nDodecane_Reitz.yaml",
        "element": "C",
        "temperature": 1200.0,
        "pressure": 101325.0,
        "mole_fractions": "c12h26:0.05, o2:0.2, n2:0.75",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_019_blowers_masel_H",
        "category": "reaction_path_diagram",
        "mechanism": "blowers-masel.yaml",
        "phase": "gas",
        "element": "H",
        "temperature": 1200.0,
        "pressure": 101325.0,
        "mole_fractions": "H2:0.5, O2:0.2, H2O:0.3",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })
    cases.append({
        "case_id": "fam4_020_blowers_masel_C",
        "category": "reaction_path_diagram",
        "mechanism": "blowers-masel.yaml",
        "phase": "gas",
        "element": "C",
        "temperature": 1200.0,
        "pressure": 101325.0,
        "mole_fractions": "CH4:0.1, O2:0.2, N2:0.7",
        "flow_type": "NetFlow",
        "scale": -1.0,
        "threshold": 0.005,
        "included": [],
        "excluded": [],
        "display_only": -1
    })

    # 2. Reaction inspection cases (11 cases)
    # -------------------------------------------------------------
    cases.append({
        "case_id": "fam4_021_rxn_gri30_elementary",
        "category": "reaction_inspection",
        "mechanism": "gri30.yaml",
        "reaction_index": 0
    })
    cases.append({
        "case_id": "fam4_022_rxn_gri30_threebody",
        "category": "reaction_inspection",
        "mechanism": "gri30.yaml",
        "reaction_index": 3
    })
    cases.append({
        "case_id": "fam4_023_rxn_gri30_troe",
        "category": "reaction_inspection",
        "mechanism": "gri30.yaml",
        "reaction_index": 48
    })
    cases.append({
        "case_id": "fam4_024_rxn_gri30_lindemann",
        "category": "reaction_inspection",
        "mechanism": "gri30.yaml",
        "reaction_index": 52
    })
    cases.append({
        "case_id": "fam4_025_rxn_h2o2_elementary",
        "category": "reaction_inspection",
        "mechanism": "h2o2.yaml",
        "reaction_index": 2
    })
    cases.append({
        "case_id": "fam4_026_rxn_h2o2_threebody",
        "category": "reaction_inspection",
        "mechanism": "h2o2.yaml",
        "reaction_index": 5
    })
    cases.append({
        "case_id": "fam4_027_rxn_ptcombust_surf",
        "category": "reaction_inspection",
        "mechanism": "ptcombust.yaml",
        "phase": "Pt_surf",
        "reaction_index": 0
    })
    cases.append({
        "case_id": "fam4_028_rxn_ptcombust_sticking",
        "category": "reaction_inspection",
        "mechanism": "ptcombust.yaml",
        "phase": "Pt_surf",
        "reaction_index": 2
    })
    cases.append({
        "case_id": "fam4_029_rxn_blowers_masel_surf",
        "category": "reaction_inspection",
        "mechanism": "blowers-masel.yaml",
        "phase": "Pt_surf",
        "reaction_index": 5
    })
    cases.append({
        "case_id": "fam4_030_rxn_sofc_electrochem",
        "category": "reaction_inspection",
        "mechanism": "sofc.yaml",
        "phase": "metal_surface",
        "reaction_index": 0
    })
    cases.append({
        "case_id": "fam4_031_rxn_sofc_oxide",
        "category": "reaction_inspection",
        "mechanism": "sofc.yaml",
        "phase": "oxide_surface",
        "reaction_index": 0
    })

    # 3. Group algebra cases (7 cases)
    # -------------------------------------------------------------
    cases.append({
        "case_id": "fam4_032_group_single_atom",
        "category": "group_algebra",
        "comp1": [1, 0, 0],
        "elements": ["H", "O", "C"]
    })
    cases.append({
        "case_id": "fam4_033_group_multi_atom",
        "category": "group_algebra",
        "comp1": [2, 1, 0],
        "elements": ["H", "O", "C"]
    })
    cases.append({
        "case_id": "fam4_034_group_sum",
        "category": "group_algebra",
        "comp1": [1, 1, 0],
        "comp2": [0, 1, 1],
        "elements": ["H", "O", "C"]
    })
    cases.append({
        "case_id": "fam4_035_group_diff",
        "category": "group_algebra",
        "comp1": [2, 2, 0],
        "comp2": [1, 1, 0],
        "elements": ["H", "O", "C"]
    })
    cases.append({
        "case_id": "fam4_036_group_negative",
        "category": "group_algebra",
        "comp1": [-1, -2, 0],
        "elements": ["H", "O", "C"]
    })
    cases.append({
        "case_id": "fam4_037_group_mixed_sign",
        "category": "group_algebra",
        "comp1": [1, -1, 0],
        "elements": ["H", "O", "C"]
    })
    cases.append({
        "case_id": "fam4_038_group_empty",
        "category": "group_algebra",
        "comp1": [],
        "elements": []
    })

    # 4. Factories cases (2 cases)
    # -------------------------------------------------------------
    cases.append({
        "case_id": "fam4_039_kinetics_factory",
        "category": "kinetics_factory",
        "models": ["gas", "Interface", "Edge"]
    })
    cases.append({
        "case_id": "fam4_040_rate_factory",
        "category": "rate_factory",
        "rate_types": [
            "Arrhenius", "elementary", "three-body", "Troe", "Lindemann",
            "falloff", "Blowers-Masel", "interface-Arrhenius", "sticking-Arrhenius"
        ]
    })

    # Write corpus
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, sort_keys=True) + "\n")

    print(f"Generated {len(cases)} cases in {CORPUS_PATH}")


if __name__ == "__main__":
    generate_corpus()
