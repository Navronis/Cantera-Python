import json

open_transport_indices = [
    3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517,
    3580, 3581, 3582, 3583, 3584, 3585, 3586, 3587, 3589, 3590,
    3591, 3592, 3593, 3594, 5237, 5238, 5239, 5240, 5241, 5278,
    5279, 5281, 5283
]

inventory_rows = {}
with open('provenance/canonical_5784_symbol_inventory.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        if row.get('canonical_index') in open_transport_indices:
            inventory_rows[row['canonical_index']] = row

print(f"Loaded {len(inventory_rows)} open transport rows from inventory.")

# 1. Starting Snapshot
snapshot = {
    "subsystem": "transport",
    "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
    "total_transport_symbols": 229,
    "starting_open_symbols": len(inventory_rows),
    "starting_status_breakdown": {
        "NOT_PORTED": sum(1 for r in inventory_rows.values() if r.get('current_status') == 'NOT_PORTED'),
        "PARTIAL": sum(1 for r in inventory_rows.values() if r.get('current_status') == 'PARTIAL')
    },
    "symbols": list(inventory_rows.values())
}

with open('audit/transport_remaining_starting_snapshot.json', 'w', encoding='utf-8') as f:
    json.dump(snapshot, f, indent=2)
print("Wrote audit/transport_remaining_starting_snapshot.json")

# 2. Closure Symbol Set
closure_set = {
    "subsystem": "transport",
    "closure_name": "TRANSPORT_FINAL_REMAINING_CLOSURE",
    "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
    "symbol_count": len(inventory_rows),
    "classes": {
        "IonGasTransport": 15,
        "TransportFactory": 7,
        "UnityLewisTransport": 5,
        "WaterTransport": 6
    },
    "symbols": [
        {
            "canonical_index": r["canonical_index"],
            "stable_symbol_id": r["stable_symbol_id"],
            "class": r.get("class"),
            "signature": r.get("signature"),
            "native_file": r.get("source_file"),
            "python_target": (
                f"cantera_python.transport.{r.get('class')}.{r.get('function_or_method')}"
                if r.get('class') else f"cantera_python.transport.{r.get('function_or_method')}"
            ),
            "scientific_category": (
                "ion_charged_species_transport" if "Ion" in (r.get("class") or "") or "Ion" in r.get("source_file", "")
                else "water_liquid_transport" if "Water" in (r.get("class") or "") or "Water" in r.get("source_file", "")
                else "unity_lewis_transport" if "Unity" in (r.get("class") or "")
                else "transport_factory"
            )
        }
        for r in sorted(inventory_rows.values(), key=lambda x: x["canonical_index"])
    ]
}

with open('audit/transport_remaining_closure_symbol_set.json', 'w', encoding='utf-8') as f:
    json.dump(closure_set, f, indent=2)
print("Wrote audit/transport_remaining_closure_symbol_set.json")

# 3. Runtime Call Graph
call_graph = {
    "subsystem": "transport",
    "closure_name": "TRANSPORT_FINAL_REMAINING_CLOSURE",
    "call_chains": {
        "IonGasTransport": {
            "init": ["setupCollisionParameters", "setupN64", "setupCollisionIntegral"],
            "viscosity": ["update_T", "update_C", "updateViscosity_T", "multiply"],
            "thermalConductivity": ["update_T", "update_C", "updateCond_T"],
            "getMobilities": ["update_T", "update_C", "updateDiff_T"],
            "getMixDiffCoeffs": ["update_T", "update_C", "updateDiff_T"],
            "electricalConductivity": ["getMobilities"],
            "fitDiffCoeffs": ["GasTransport::fitDiffCoeffs", "omega11_n64", "polyfit"],
            "setupN64": ["m_gamma.resize", "Tang_C6_combination_rule"]
        },
        "UnityLewisTransport": {
            "getMixDiffCoeffs": ["thermalConductivity", "thermo.density", "thermo.cp_mass"],
            "getMixDiffCoeffsMass": ["thermalConductivity", "thermo.density", "thermo.cp_mass"],
            "getThermalDiffCoeffs": ["zeros"],
            "getMixDiffCoeffsMole": ["NotImplementedError"]
        },
        "WaterTransport": {
            "init": ["thermo_assignment"],
            "viscosity": ["thermo.temperature", "thermo.density", "thermo.isothermalCompressibility", "SengersWatson1986"],
            "bulkViscosity": ["return 0.0"],
            "thermalConductivity": ["thermo.temperature", "thermo.density", "thermo.isothermalCompressibility", "thermo.thermalExpansionCoeff", "SengersWatson1986"]
        },
        "TransportFactory": {
            "factory": ["singleton_accessor"],
            "deleteFactory": ["singleton_destructor"],
            "newTransport_model": ["TransportFactory.create", "tr.init"],
            "newTransport_thermo": ["phase.input", "newTransport_model"]
        }
    }
}

with open('audit/transport_remaining_runtime_call_graph.json', 'w', encoding='utf-8') as f:
    json.dump(call_graph, f, indent=2)
print("Wrote audit/transport_remaining_runtime_call_graph.json")
