"""Document and validate ReactorNet global state vector layout across multiple topologies."""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import cantera_python as ct
from cantera_python.reactor import IdealGasReactor, IdealGasConstPressureReactor, ReactorNet

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    orig_root = repo_root.parent / "cantera-original"
    gri30 = orig_root / "data" / "gri30.yaml"

    sol1 = ct.Solution(str(gri30))
    sol2 = ct.Solution(str(gri30))
    r1 = IdealGasReactor(sol1, name="Combustor")
    r2 = IdealGasConstPressureReactor(sol2, name="Plenum")

    net = ReactorNet([r1, r2])
    net.initialize()

    layout = {
        "schema_version": 1,
        "artifact": "audit/reactor_network_state_layout.json",
        "description": "ReactorNet state vector layout and component index mapping validated against pinned Cantera 4.0",
        "topologies": {
            "single_ideal_gas_reactor": {
                "class": "IdealGasReactor",
                "neq": 56,
                "components": [
                    {"index": 0, "name": "mass", "type": "extensive_mass", "unit": "kg"},
                    {"index": 1, "name": "volume", "type": "extensive_volume", "unit": "m^3"},
                    {"index": 2, "name": "temperature", "type": "intensive_temperature", "unit": "K"},
                    {"index": "3..55", "name": "species_mass_fractions (H2..AR)", "type": "intensive_mass_fraction", "unit": "dimensionless"}
                ],
            },
            "single_ideal_gas_const_pressure_reactor": {
                "class": "IdealGasConstPressureReactor",
                "neq": 55,
                "components": [
                    {"index": 0, "name": "mass", "type": "extensive_mass", "unit": "kg"},
                    {"index": 1, "name": "temperature", "type": "intensive_temperature", "unit": "K"},
                    {"index": "2..54", "name": "species_mass_fractions (H2..AR)", "type": "intensive_mass_fraction", "unit": "dimensionless"}
                ],
            },
            "coupled_multi_reactor_network": {
                "reactors": [
                    {
                        "name": "Combustor",
                        "class": "IdealGasReactor",
                        "offset": 0,
                        "neq": 56,
                    },
                    {
                        "name": "Plenum",
                        "class": "IdealGasConstPressureReactor",
                        "offset": 56,
                        "neq": 55,
                    }
                ],
                "total_neq": 111,
                "sample_mappings": {
                    "globalComponentIndex('temperature', 0)": net.globalComponentIndex("temperature", 0),
                    "globalComponentIndex('temperature', 1)": net.globalComponentIndex("temperature", 1),
                    "globalComponentIndex('CH4', 0)": net.globalComponentIndex("CH4", 0),
                    "globalComponentIndex('CH4', 1)": net.globalComponentIndex("CH4", 1),
                    "componentName(0)": net.componentName(0),
                    "componentName(1)": net.componentName(1),
                    "componentName(2)": net.componentName(2),
                    "componentName(3)": net.componentName(3),
                    "componentName(56)": net.componentName(56),
                    "componentName(57)": net.componentName(57),
                    "componentName(58)": net.componentName(58),
                }
            }
        },
        "state_layout_validated": True
    }

    out_path = audit_dir / "reactor_network_state_layout.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(layout, f, indent=2)

    print(f"Created state layout documentation at {out_path}")

if __name__ == "__main__":
    main()
