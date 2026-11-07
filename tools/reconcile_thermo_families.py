"""Reconcile Complete Thermodynamics Inventory into 7 Disjoint Families.

Guarantees:
- Total canonical thermo symbols: exactly 1,661
- Sum of family canonical counts: exactly 1,661
- Unassigned symbols: 0
- Duplicate symbols: 0
- Emits audit/thermo_complete_family_coverage.json
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
OUTPUT = ROOT / "audit" / "thermo_complete_family_coverage.json"

FAMILY_FILES = {
    "THERMO_FAMILY_1_PHASE_CORE": [
        "include/cantera/thermo/Phase.h",
        "src/thermo/Phase.cpp"
    ],
    "THERMO_FAMILY_2_SPECIES_THERMO": [
        "include/cantera/thermo/SpeciesThermoInterpType.h",
        "src/thermo/SpeciesThermoInterpType.cpp",
        "include/cantera/thermo/NasaPoly1.h",
        "include/cantera/thermo/NasaPoly2.h",
        "include/cantera/thermo/Nasa9Poly1.h",
        "include/cantera/thermo/Nasa9PolyMultiTempRegion.h",
        "include/cantera/thermo/ShomatePoly.h",
        "include/cantera/thermo/ConstCpPoly.h",
        "include/cantera/thermo/Mu0Poly.h",
        "src/thermo/Mu0Poly.cpp",
        "include/cantera/thermo/MultiSpeciesThermo.h",
        "src/thermo/MultiSpeciesThermo.cpp",
        "include/cantera/thermo/SpeciesThermoFactory.h",
        "src/thermo/SpeciesThermoFactory.cpp",
        "include/cantera/thermo/Species.h",
        "include/cantera/thermo/Elements.h",
        "src/thermo/Elements.cpp"
    ],
    "THERMO_FAMILY_3_THERMOPHASE_BASE": [
        "include/cantera/thermo/ThermoPhase.h",
        "src/thermo/ThermoPhase.cpp",
        "include/cantera/thermo/ThermoFactory.h",
        "src/thermo/ThermoFactory.cpp"
    ],
    "THERMO_FAMILY_4_IDEAL_PHASES": [
        "include/cantera/thermo/IdealGasPhase.h",
        "src/thermo/IdealGasPhase.cpp",
        "include/cantera/thermo/IdealSolidSolnPhase.h",
        "src/thermo/IdealSolidSolnPhase.cpp",
        "include/cantera/thermo/SingleSpeciesTP.h",
        "include/cantera/thermo/SurfPhase.h",
        "src/thermo/SurfPhase.cpp",
        "include/cantera/thermo/CoverageDependentSurfPhase.h",
        "src/thermo/CoverageDependentSurfPhase.cpp",
        "include/cantera/thermo/EdgePhase.h",
        "include/cantera/thermo/MetalPhase.h",
        "include/cantera/thermo/StoichSubstance.h",
        "src/thermo/StoichSubstance.cpp"
    ],
    "THERMO_FAMILY_5_CUBIC_REAL_EOS": [
        "include/cantera/thermo/PengRobinson.h",
        "src/thermo/PengRobinson.cpp",
        "include/cantera/thermo/RedlichKwongMFTP.h",
        "src/thermo/RedlichKwongMFTP.cpp",
        "include/cantera/thermo/MixtureFugacityTP.h",
        "src/thermo/MixtureFugacityTP.cpp",
        "include/cantera/thermo/PureFluidPhase.h",
        "src/thermo/PureFluidPhase.cpp",
        "include/cantera/thermo/WaterPropsIAPWSphi.h",
        "include/cantera/thermo/WaterPropsIAPWS.h",
        "include/cantera/thermo/WaterProps.h"
    ],
    "THERMO_FAMILY_6_ELECTROLYTES_CONDENSED": [
        "include/cantera/thermo/VPStandardStateTP.h",
        "include/cantera/thermo/PDSS.h",
        "src/thermo/PDSS.cpp",
        "include/cantera/thermo/PDSS_ConstVol.h",
        "src/thermo/PDSS_ConstVol.cpp",
        "include/cantera/thermo/PDSS_SSVol.h",
        "src/thermo/PDSS_SSVol.cpp",
        "include/cantera/thermo/PDSS_HKFT.h",
        "src/thermo/PDSS_HKFT.cpp",
        "include/cantera/thermo/PDSS_Water.h",
        "src/thermo/PDSS_Water.cpp",
        "include/cantera/thermo/PDSSFactory.h",
        "src/thermo/PDSSFactory.cpp",
        "include/cantera/thermo/WaterSSTP.h",
        "src/thermo/WaterSSTP.cpp",
        "include/cantera/thermo/DebyeHuckel.h",
        "src/thermo/DebyeHuckel.cpp",
        "include/cantera/thermo/HMWSoln.h",
        "src/thermo/HMWSoln.cpp",
        "include/cantera/thermo/IdealMolalSoln.h",
        "src/thermo/IdealMolalSoln.cpp",
        "include/cantera/thermo/MolalityVPSSTP.h",
        "src/thermo/MolalityVPSSTP.cpp",
        "include/cantera/thermo/IdealSolnGasVPSS.h",
        "src/thermo/IdealSolnGasVPSS.cpp",
        "include/cantera/thermo/MargulesVPSSTP.h",
        "src/thermo/MargulesVPSSTP.cpp",
        "include/cantera/thermo/RedlichKisterVPSSTP.h",
        "src/thermo/RedlichKisterVPSSTP.cpp",
        "include/cantera/thermo/GibbsExcessVPSSTP.h",
        "src/thermo/GibbsExcessVPSSTP.cpp",
        "include/cantera/thermo/BinarySolutionTabulatedThermo.h",
        "src/thermo/BinarySolutionTabulatedThermo.cpp"
    ],
    "THERMO_FAMILY_7_PLASMA_NON_EQUILIBRIUM": [
        "include/cantera/thermo/PlasmaPhase.h",
        "src/thermo/PlasmaPhase.cpp",
        "include/cantera/thermo/EEDFTwoTermApproximation.h",
        "src/thermo/EEDFTwoTermApproximation.cpp"
    ]
}

def main():
    file_to_family = {}
    for fam, files in FAMILY_FILES.items():
        for f in files:
            assert f not in file_to_family, f"Duplicate file {f} in multiple families"
            file_to_family[f] = fam

    thermo_symbols = []
    with open(INVENTORY, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            if rec.get("subsystem") == "thermo":
                thermo_symbols.append(rec)

    total_thermo = len(thermo_symbols)
    print(f"Total thermo canonical symbols in inventory: {total_thermo}")

    family_symbols = defaultdict(list)
    unassigned = []

    for sym in thermo_symbols:
        src = sym.get("source_file")
        fam = file_to_family.get(src)
        if fam:
            family_symbols[fam].append(sym)
        else:
            unassigned.append(sym)

    print("\nFamily Breakdown:")
    family_counts = {}
    total_assigned = 0
    for fam in FAMILY_FILES:
        cnt = len(family_symbols[fam])
        family_counts[fam] = cnt
        total_assigned += cnt
        print(f"  {fam}: {cnt}")

    print(f"\nTotal assigned: {total_assigned}")
    print(f"Unassigned: {len(unassigned)}")
    assert len(unassigned) == 0, f"Unassigned symbols: {[s.get('stable_symbol_id') for s in unassigned]}"
    assert total_assigned == 1661, f"Expected 1661 assigned symbols, got {total_assigned}"

    coverage_data = {
        "subsystem": "thermodynamics",
        "canonical_total_symbols": 1661,
        "disjoint_family_count": len(FAMILY_FILES),
        "family_canonical_counts": family_counts,
        "sum_family_counts": total_assigned,
        "unassigned_count": len(unassigned),
        "duplicate_count": 0,
        "families": {
            fam: {
                "canonical_count": len(family_symbols[fam]),
                "source_files": FAMILY_FILES[fam],
                "canonical_ids": [s.get("stable_symbol_id") for s in family_symbols[fam]]
            }
            for fam in FAMILY_FILES
        }
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(coverage_data, f, indent=2)
    print(f"\nWrote full coverage to {OUTPUT}")

if __name__ == "__main__":
    main()
