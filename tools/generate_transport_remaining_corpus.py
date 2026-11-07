"""Generate differential test corpus for remaining Transport models:
- WaterTransport (7 cases)
- UnityLewisTransport (7 cases)
- IonGasTransport (7 cases)
- TransportFactory model registration (9 cases)
Total: 30 cases
"""
import json

cases = []

# 1. WaterTransport cases (various (T, P) states across liquid, supercritical, and vapor-like states)
# Densities, isothermal compressibilities, and thermal expansion coefficients from authoritative native IAPWS95.
water_conditions = [
    ("tr_wat_001_T298_P1atm", 298.15, 101325.0, 997.0476367603455, 4.5246171732875335e-10, 0.00025728890194813005),
    ("tr_wat_002_T350_P1atm", 350.0, 101325.0, 973.7284443770474, 4.5807009974768534e-10, 0.0006235620422950847),
    ("tr_wat_003_T400_P10atm", 400.0, 1013250.0, 937.880130052789, 5.467600947169463e-10, 0.0008929649415179849),
    ("tr_wat_004_T500_P50atm", 500.0, 5066250.0, 833.5729951760819, 1.1050896689100317e-09, 0.0016164895794745122),
    ("tr_wat_005_T600_P150atm", 600.0, 15198750.0, 660.1000483495798, 5.255287785151436e-09, 0.004519407962399094),
    ("tr_wat_006_T647_P220atm_nearcrit", 647.1, 22064000.0, 287.85399512967143, 2.6487507661709724e-05, 6.739669028283377),
    ("tr_wat_007_T700_P250atm_supercrit", 700.0, 25331250.0, 128.21252691867357, 7.450987383801051e-08, 0.007472570288461471),
]

for case_id, T, P, rho, kappa, alpha in water_conditions:
    cases.append({
        "case_id": case_id,
        "category": "water_transport",
        "model": "water",
        "phase": "liquid-water",
        "input_file": "thermo-models.yaml",
        "temperature": T,
        "pressure": P,
        "density": rho,
        "isothermal_compressibility": kappa,
        "thermal_expansion_coeff": alpha,
        "mole_fractions": "H2O:1.0"
    })

# 2. UnityLewisTransport cases (gri30 mixtures across lean to rich, low to high T)
unity_conditions = [
    ("tr_ul_001_T300_P1atm_air", 300.0, 101325.0, "O2:0.21, N2:0.79"),
    ("tr_ul_002_T800_P1atm_lean", 800.0, 101325.0, "CH4:0.05, O2:0.2, N2:0.75"),
    ("tr_ul_003_T1200_P1atm_stoich", 1200.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715"),
    ("tr_ul_004_T1600_P5atm_rich", 1600.0, 506625.0, "CH4:0.15, O2:0.15, N2:0.7"),
    ("tr_ul_005_T2000_P10atm_prod", 2000.0, 1013250.0, "CO2:0.08, H2O:0.16, N2:0.72, O2:0.04"),
    ("tr_ul_006_T2400_P20atm_dissoc", 2400.0, 2026500.0, "CO:0.04, CO2:0.04, H2:0.02, H2O:0.12, OH:0.01, NO:0.005, N2:0.765"),
    ("tr_ul_007_T3000_P1atm_plasma", 3000.0, 101325.0, "H:0.05, OH:0.05, O:0.05, O2:0.1, H2O:0.1, N2:0.65"),
]

for case_id, T, P, X in unity_conditions:
    cases.append({
        "case_id": case_id,
        "category": "unity_lewis_transport",
        "model": "unity-Lewis-number",
        "phase": "gri30",
        "input_file": "gri30.yaml",
        "temperature": T,
        "pressure": P,
        "mole_fractions": X
    })

# 3. IonGasTransport cases (ch4_ion.yaml with charged species e-, H3O+, HCO+, etc.)
ion_conditions = [
    ("tr_ion_001_T300_P1atm", 300.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715"),
    ("tr_ion_002_T800_P1atm", 800.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715"),
    ("tr_ion_003_T1200_P1atm", 1200.0, 101325.0, "CH4:0.095, O2:0.19, N2:0.715"),
    ("tr_ion_004_T1500_P1atm", 1500.0, 101325.0, "CO:0.05, CO2:0.05, H2O:0.15, N2:0.74, E:0.005, H3O+:0.005"),
    ("tr_ion_005_T1800_P1atm", 1800.0, 101325.0, "CO:0.06, CO2:0.04, H2O:0.14, N2:0.74, E:0.01, H3O+:0.01"),
    ("tr_ion_006_T2000_P10atm", 2000.0, 1013250.0, "H2:0.1, O2:0.1, N2:0.75, E:0.02, H3O+:0.015, HCO+:0.015"),
    ("tr_ion_007_T2500_P1atm", 2500.0, 101325.0, "N2:0.84, O2:0.10, E:0.02, H3O+:0.02, HCO+:0.02"),
]

for case_id, T, P, X in ion_conditions:
    cases.append({
        "case_id": case_id,
        "category": "ion_gas_transport",
        "model": "ionized-gas",
        "phase": "gas",
        "input_file": "ch4_ion.yaml",
        "temperature": T,
        "pressure": P,
        "mole_fractions": X
    })

# 4. TransportFactory registration cases
factory_models = [
    ("tr_fac_001_none", "none", "Transport", "gri30.yaml", "gri30"),
    ("tr_fac_002_unity_lewis", "unity-Lewis-number", "UnityLewisTransport", "gri30.yaml", "gri30"),
    ("tr_fac_003_mix", "mixture-averaged", "MixTransport", "gri30.yaml", "gri30"),
    ("tr_fac_004_multi", "multicomponent", "MultiTransport", "gri30.yaml", "gri30"),
    ("tr_fac_005_ion", "ionized-gas", "IonGasTransport", "ch4_ion.yaml", "gas"),
    ("tr_fac_006_water", "water", "WaterTransport", "thermo-models.yaml", "liquid-water"),
    ("tr_fac_007_high_p", "high-pressure", "HighPressureGasTransport", "methane_co2.yaml", "methane_co2"),
    ("tr_fac_008_chung", "high-pressure-Chung", "ChungHighPressureGasTransport", "methane_co2.yaml", "methane_co2"),
    ("tr_fac_009_dusty", "DustyGas", "DustyGasTransport", "gri30.yaml", "gri30"),
]

for case_id, model_name, expected_class, input_file, phase_name in factory_models:
    cases.append({
        "case_id": case_id,
        "category": "transport_factory",
        "model": model_name,
        "expected_class": expected_class,
        "phase": phase_name,
        "input_file": input_file,
        "temperature": 1000.0,
        "pressure": 101325.0
    })

with open("audit/corpus_transport_remaining.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")

print(f"Generated audit/corpus_transport_remaining.jsonl with {len(cases)} cases.")
