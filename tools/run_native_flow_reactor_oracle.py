"""Native Cantera FlowReactor differential oracle runner for WSL."""

import json
import os
import sys
from pathlib import Path


def get_mechanism_path(mech_name):
    candidates = [
        Path(f"/home/nebula/cantera-pinned-oracle/data/{mech_name}"),
        Path(f"/home/nebula/cantera-pinned-oracle/test/data/{mech_name}"),
        Path(f"./data/{mech_name}"),
        Path(f"../cantera-original/data/{mech_name}"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return mech_name


def main():
    if len(sys.argv) < 4:
        print("Usage: run_native_flow_reactor_oracle.py <corpus_path> <output_path> <run_id>")
        sys.exit(1)

    import cantera as ct
    assert ct.__git_commit__.startswith("726522be"), f"Unexpected commit: {ct.__git_commit__}"

    corpus_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    run_id = sys.argv[3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            case_id = case["case_id"]
            cat = case["category"]
            mech_path = get_mechanism_path(case.get("mechanism", "h2o2.yaml"))

            result_entry = {
                "case_id": case_id,
                "category": cat,
                "run_id": run_id,
                "oracle": "native_cantera_726522be",
                "status": "PASS",
                "checkpoints": [],
                "metrics": {},
            }

            try:
                gas = ct.Solution(mech_path)
                gas.TPX = case["T"], case["P"], case["X"]
                r = ct.FlowReactor(gas)
                r.area = case["area"]
                r.mass_flow_rate = case["mass_flow_rate"]

                if case.get("new_area"):
                    r.area = case["new_area"]

                if "energy_enabled" in case:
                    r.energy_enabled = case["energy_enabled"]

                if "chemistry_enabled" in case:
                    r.chemistry_enabled = case["chemistry_enabled"]

                net = ct.ReactorNet([r])
                net.rtol = case["rtol"]
                net.atol = case["atol"]

                if cat in ("flow_hydrodynamics", "flow_reacting", "flow_extreme_pressure",
                           "flow_geometry", "flow_stoichiometry", "flow_switches"):
                    for z_samp in case["sample_distances"]:
                        net.advance(z_samp)
                        cp = {
                            "distance": float(net.distance),
                            "density": float(gas.density),
                            "speed": float(r.speed),
                            "pressure": float(gas.P),
                            "temperature": float(gas.T),
                            "mass_fractions": list(gas.Y),
                            "mass_flow_rate": float(r.speed * gas.density * r.area),
                        }
                        result_entry["checkpoints"].append(cp)

                elif cat == "flow_stepping":
                    n_steps = case.get("n_steps", 5)
                    for step_idx in range(n_steps):
                        net.step()
                        cp = {
                            "step_index": step_idx,
                            "distance": float(net.distance),
                            "density": float(gas.density),
                            "speed": float(r.speed),
                            "pressure": float(gas.P),
                            "temperature": float(gas.T),
                            "mass_fractions": list(gas.Y),
                        }
                        result_entry["checkpoints"].append(cp)

                elif cat == "flow_introspection":
                    components = ["density", "speed", "pressure", "temperature"] + list(gas.species_names)
                    idx_map = {name: int(r.component_index(name)) for name in components}
                    name_map = {k: str(r.component_name(k)) for k in range(r.n_vars)}
                    result_entry["metrics"]["component_indices"] = idx_map
                    result_entry["metrics"]["component_names"] = name_map
                    result_entry["metrics"]["n_vars"] = int(r.n_vars)

                elif cat == "flow_contracts":
                    result_entry["metrics"]["type"] = str(r.type)
                    # Verify getState raises NotImplementedError
                    threw_get_state = False
                    try:
                        r.get_state()
                    except NotImplementedError:
                        threw_get_state = True
                    result_entry["metrics"]["threw_get_state"] = threw_get_state

            except Exception as exc:
                result_entry["status"] = "ERROR"
                result_entry["error"] = str(exc)

            results.append(result_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Native FlowReactor oracle processed {len(results)} cases -> {output_path}")


if __name__ == "__main__":
    main()
