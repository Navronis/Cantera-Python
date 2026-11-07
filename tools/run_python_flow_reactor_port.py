"""Python port runner for FlowReactor differential campaign."""

import json
import os
import sys
from pathlib import Path

# Ensure outputs/cantera-python is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import cantera_python as ct


def run_cases(corpus_path: Path, output_path: Path, run_id: str):
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
            mech_name = case.get("mechanism", "h2o2.yaml")

            result_entry = {
                "case_id": case_id,
                "category": cat,
                "run_id": run_id,
                "oracle": "python_flow_reactor_port",
                "status": "PASS",
                "checkpoints": [],
                "metrics": {},
            }

            try:
                gas = ct.Solution(mech_name)
                gas.TPX = case["T"], case["P"], case["X"]
                r = ct.FlowReactor(gas)
                r.setArea(case["area"])
                r.setMassFlowRate(case["mass_flow_rate"])

                if case.get("new_area"):
                    r.setArea(case["new_area"])

                if "energy_enabled" in case:
                    r._energy = case["energy_enabled"]

                if "chemistry_enabled" in case:
                    r._chem = case["chemistry_enabled"]

                net = ct.ReactorNet([r])
                net.setTolerances(case["rtol"], case["atol"])

                if cat in ("flow_hydrodynamics", "flow_reacting", "flow_extreme_pressure",
                           "flow_geometry", "flow_stoichiometry", "flow_switches"):
                    for z_samp in case["sample_distances"]:
                        net.advance(z_samp)
                        cp = {
                            "distance": float(net.distance()),
                            "density": float(gas.density),
                            "speed": float(r.speed()),
                            "pressure": float(gas.P),
                            "temperature": float(gas.T),
                            "mass_fractions": [float(y) for y in gas.Y],
                            "mass_flow_rate": float(r.massFlowRate()),
                        }
                        result_entry["checkpoints"].append(cp)

                elif cat == "flow_stepping":
                    n_steps = case.get("n_steps", 5)
                    for step_idx in range(n_steps):
                        net.step()
                        cp = {
                            "step_index": step_idx,
                            "distance": float(net.distance()),
                            "density": float(gas.density),
                            "speed": float(r.speed()),
                            "pressure": float(gas.P),
                            "temperature": float(gas.T),
                            "mass_fractions": [float(y) for y in gas.Y],
                        }
                        result_entry["checkpoints"].append(cp)

                elif cat == "flow_introspection":
                    components = ["density", "speed", "pressure", "temperature"] + list(gas.species_names)
                    idx_map = {name: int(r.componentIndex(name)) for name in components}
                    name_map = {k: str(r.componentName(k)) for k in range(r.neq())}
                    result_entry["metrics"]["component_indices"] = idx_map
                    result_entry["metrics"]["component_names"] = name_map
                    result_entry["metrics"]["n_vars"] = int(r.neq())

                elif cat == "flow_contracts":
                    result_entry["metrics"]["type"] = str(r.type())
                    result_entry["metrics"]["is_ode"] = bool(r.isOde())
                    result_entry["metrics"]["time_is_independent"] = bool(r.timeIsIndependent())

                    # Verify getState raises NotImplementedError
                    threw_get_state = False
                    try:
                        r.getState(None)
                    except NotImplementedError:
                        threw_get_state = True
                    result_entry["metrics"]["threw_get_state"] = threw_get_state

                    # Verify eval raises NotImplementedError
                    threw_eval = False
                    try:
                        r.eval(0.0, None, None)
                    except NotImplementedError:
                        threw_eval = True
                    result_entry["metrics"]["threw_eval"] = threw_eval

                    # Verify initializeSteady raises CanteraError
                    threw_init_steady = False
                    try:
                        r.initializeSteady()
                    except ct.CanteraError:
                        threw_init_steady = True
                    result_entry["metrics"]["threw_init_steady"] = threw_init_steady

            except Exception as exc:
                result_entry["status"] = "ERROR"
                result_entry["error"] = str(exc)

            results.append(result_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Python FlowReactor port processed {len(results)} cases -> {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: run_python_flow_reactor_port.py <corpus_path> <output_path> <run_id>")
        sys.exit(1)
    run_cases(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
