"""Generate deterministic differential test corpus for FlowReactor DAE subsystem."""

import json
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = audit_dir / "corpus_flow_reactor.jsonl"

    cases = [
        # 1. Inert gas flow
        {
            "case_id": "flow_inert_argon_300k",
            "category": "flow_hydrodynamics",
            "description": "Inert plug flow reactor with Argon at 300K, 1 atm",
            "mechanism": "h2o2.yaml",
            "T": 300.0,
            "P": 101325.0,
            "X": "AR:1.0",
            "mass_flow_rate": 0.5,
            "area": 0.01,
            "sample_distances": [1e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 2. Inert mixture at elevated temperature and pressure
        {
            "case_id": "flow_inert_n2_o2_800k",
            "category": "flow_hydrodynamics",
            "description": "Inert plug flow with N2/O2 at 800K, 2 atm",
            "mechanism": "h2o2.yaml",
            "T": 800.0,
            "P": 202650.0,
            "X": "N2:0.79, O2:0.21",
            "mass_flow_rate": 0.2,
            "area": 0.005,
            "sample_distances": [1e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 3. Reacting H2/air at 1000K, standard pressure
        {
            "case_id": "flow_h2_air_1000k_standard",
            "category": "flow_reacting",
            "description": "Reacting H2/air stoichiometric plug flow at 1000K, 1 atm",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:3.76",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 4. Reacting H2/air at 1200K, elevated temperature
        {
            "case_id": "flow_h2_air_1200k_fast",
            "category": "flow_reacting",
            "description": "Fast reacting H2/air plug flow at 1200K, 1 atm",
            "mechanism": "h2o2.yaml",
            "T": 1200.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:3.76",
            "mass_flow_rate": 0.2,
            "area": 0.005,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 5. Dilute H2/O2 in Argon
        {
            "case_id": "flow_h2_o2_ar_dilute",
            "category": "flow_reacting",
            "description": "Dilute H2/O2 in Argon bath at 1100K, 1 atm",
            "mechanism": "h2o2.yaml",
            "T": 1100.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:10",
            "mass_flow_rate": 0.15,
            "area": 0.01,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 6. Reacting CH4/air combustion
        {
            "case_id": "flow_ch4_air_1200k",
            "category": "flow_reacting",
            "description": "Reacting CH4/air plug flow at 1200K, 1 atm",
            "mechanism": "gri30.yaml",
            "T": 1200.0,
            "P": 101325.0,
            "X": "CH4:1, O2:2, N2:7.52",
            "mass_flow_rate": 0.1,
            "area": 0.01,
            "sample_distances": [1e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 7. High pressure flow at 10 atm
        {
            "case_id": "flow_high_pressure_10atm",
            "category": "flow_extreme_pressure",
            "description": "High pressure reacting plug flow at 10 atm, 1000K",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 1013250.0,
            "X": "H2:2, O2:1, AR:3.76",
            "mass_flow_rate": 0.5,
            "area": 0.01,
            "sample_distances": [1e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 8. Low pressure flow at 0.5 atm
        {
            "case_id": "flow_low_pressure_05atm",
            "category": "flow_extreme_pressure",
            "description": "Low pressure reacting plug flow at 0.5 atm, 900K",
            "mechanism": "h2o2.yaml",
            "T": 900.0,
            "P": 50662.5,
            "X": "H2:2, O2:1, AR:3.76",
            "mass_flow_rate": 0.05,
            "area": 0.005,
            "sample_distances": [1e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 9. Area modification test
        {
            "case_id": "flow_area_change_preserves_mdot",
            "category": "flow_geometry",
            "description": "Flow reactor cross-sectional area expansion preserving mass flow rate",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.01,
            "new_area": 0.02,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 10. Fuel-rich stoichiometry (phi=2.0)
        {
            "case_id": "flow_fuel_rich_phi2",
            "category": "flow_stoichiometry",
            "description": "Fuel-rich H2/O2 mixture (phi=2.0) at 1000K, 1 atm",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:4, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 11. Fuel-lean stoichiometry (phi=0.5)
        {
            "case_id": "flow_fuel_lean_phi05",
            "category": "flow_stoichiometry",
            "description": "Fuel-lean H2/O2 mixture (phi=0.5) at 1000K, 1 atm",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:1, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 12. Energy disabled (isothermal plug flow)
        {
            "case_id": "flow_energy_disabled_isothermal",
            "category": "flow_switches",
            "description": "Isothermal plug flow reactor with energy equation disabled",
            "mechanism": "h2o2.yaml",
            "T": 1100.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "energy_enabled": False,
            "chemistry_enabled": True,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 13. Chemistry disabled (chemically frozen hydrodynamic flow)
        {
            "case_id": "flow_chemistry_disabled_frozen",
            "category": "flow_switches",
            "description": "Chemically frozen plug flow reactor with kinetics disabled",
            "mechanism": "h2o2.yaml",
            "T": 1100.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "energy_enabled": True,
            "chemistry_enabled": False,
            "sample_distances": [2e-6, 5e-6, 1e-5],
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 14. Step mode verification
        {
            "case_id": "flow_step_mode",
            "category": "flow_stepping",
            "description": "Flow reactor step mode advancing internal BDF steps",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "n_steps": 5,
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 15. Component introspection verification
        {
            "case_id": "flow_component_introspection",
            "category": "flow_introspection",
            "description": "Component indexing and naming introspection for all state variables",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "rtol": 1e-4,
            "atol": 1e-10,
        },
        # 16. Source contracts and error semantics
        {
            "case_id": "flow_source_contracts",
            "category": "flow_contracts",
            "description": "FlowReactor source contracts: type, isOde, timeIsIndependent, NotImplementedError on getState/eval",
            "mechanism": "h2o2.yaml",
            "T": 1000.0,
            "P": 101325.0,
            "X": "H2:2, O2:1, AR:4",
            "mass_flow_rate": 0.1,
            "area": 0.005,
            "rtol": 1e-4,
            "atol": 1e-10,
        },
    ]

    with open(corpus_file, "w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    print(f"Generated {len(cases)} FlowReactor cases into {corpus_file}")


if __name__ == "__main__":
    main()
