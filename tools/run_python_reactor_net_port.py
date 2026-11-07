"""Pure Python ReactorNet differential runner."""

import json
import os
import sys
import math
from pathlib import Path

# Setup Python paths
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from cantera_python import (
    Solution,
    IdealGasReactor,
    IdealGasConstPressureReactor,
    Reservoir,
    MassFlowController,
    Valve,
    PressureController,
    Wall,
    ReactorNet,
)
import cantera_python as ct

def get_mechanism_path(mech_name):
    candidates = [
        repo_root / "data" / mech_name,
        repo_root.parent / "cantera-original" / "data" / mech_name,
        Path("./data") / mech_name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return mech_name

def main():
    if len(sys.argv) < 4:
        print("Usage: run_python_reactor_net_port.py <corpus_path> <output_path> <run_id>")
        sys.exit(1)

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
            mech_path = get_mechanism_path(case.get("mechanism", "gri30.yaml"))

            result_entry = {
                "case_id": case_id,
                "category": cat,
                "run_id": run_id,
                "oracle": "python_reactor_net_port",
                "status": "PASS",
                "checkpoints": [],
                "metrics": {},
            }

            try:
                if cat in ("constant_volume", "ignition", "rocket_high_pressure", "long_horizon"):
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T"], case["P"], case["X"]
                    vol = case.get("volume", 1.0)
                    r = IdealGasReactor(gas, volume=vol, name="Reactor0")
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    prev_T = r.phase.T
                    prev_t = 0.0
                    max_dTdt = 0.0
                    ign_delay = None

                    # Fine-grained stepping to capture ignition delay if requested
                    if case.get("compute_ignition_delay"):
                        t_curr = 0.0
                        dt_fine = 1e-6
                        t_max = max(case["sample_times"])
                        while t_curr < t_max:
                            t_curr = min(t_curr + dt_fine, t_max)
                            net.advance(t_curr)
                            dTdt = (r.phase.T - prev_T) / (t_curr - prev_t) if t_curr > prev_t else 0.0
                            if dTdt > max_dTdt:
                                max_dTdt = dTdt
                                ign_delay = t_curr
                            prev_T = r.phase.T
                            prev_t = t_curr
                        result_entry["metrics"]["ignition_delay_s"] = ign_delay
                        result_entry["metrics"]["max_dTdt"] = max_dTdt

                    # Re-run for exact sample times
                    gas.TPX = case["T"], case["P"], case["X"]
                    r = IdealGasReactor(gas, volume=vol, name="Reactor0")
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    for t_samp in case["sample_times"]:
                        net.advance(t_samp)
                        u_total = r.mass * r.phase.int_energy_mass
                        cp = {
                            "time": float(net.time),
                            "temperature": float(r.phase.T),
                            "pressure": float(r.phase.P),
                            "density": float(r.phase.density),
                            "volume": float(r.volume),
                            "mass": float(r.mass),
                            "int_energy": float(u_total),
                            "Y_major": [float(r.phase.Y[r.phase.species_index(s)]) for s in ["H2", "O2", "H2O", "CH4", "CO2"] if s in r.phase.species_names],
                        }
                        result_entry["checkpoints"].append(cp)

                elif cat == "constant_pressure":
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T"], case["P"], case["X"]
                    vol = case.get("volume", 1.0)
                    r = IdealGasConstPressureReactor(gas, volume=vol, name="ReactorCP")
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    for t_samp in case["sample_times"]:
                        net.advance(t_samp)
                        h_total = r.mass * r.phase.enthalpy_mass
                        cp = {
                            "time": float(net.time),
                            "temperature": float(r.phase.T),
                            "pressure": float(r.phase.P),
                            "density": float(r.phase.density),
                            "volume": float(r.volume),
                            "mass": float(r.mass),
                            "enthalpy": float(h_total),
                            "Y_major": [float(r.phase.Y[r.phase.species_index(s)]) for s in ["H2", "O2", "H2O", "N2"] if s in r.phase.species_names],
                        }
                        result_entry["checkpoints"].append(cp)

                elif cat == "flow_devices":
                    topo = case["topology"]
                    if topo == "reservoir_mfc_reactor":
                        gas_in = ct.Solution(mech_path)
                        gas_in.TPX = case["T_res"], case["P_res"], case["X_res"]
                        res = Reservoir(gas_in)

                        gas_r = ct.Solution(mech_path)
                        gas_r.TPX = case["T_r"], case["P_r"], case["X_r"]
                        r = IdealGasReactor(gas_r, volume=case["volume"])
                        mfc = MassFlowController(res, r, mdot=case["mdot"])
                        net = ReactorNet([r], solver_backend="native_python")
                        net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                        for t_samp in case["sample_times"]:
                            net.advance(t_samp)
                            result_entry["checkpoints"].append({
                                "time": float(net.time),
                                "mass": float(r.mass),
                                "temperature": float(r.phase.T),
                                "pressure": float(r.phase.P),
                            })

                    elif topo == "reactor_valve_reactor":
                        gas1 = ct.Solution(mech_path)
                        gas1.TPX = case["T1"], case["P1"], case["X1"]
                        r1 = IdealGasReactor(gas1, volume=case["V1"])

                        gas2 = ct.Solution(mech_path)
                        gas2.TPX = case["T2"], case["P2"], case["X2"]
                        r2 = IdealGasReactor(gas2, volume=case["V2"])

                        valve = Valve(r1, r2, Kv=case["Kv"])
                        net = ReactorNet([r1, r2], solver_backend="native_python")
                        net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                        for t_samp in case["sample_times"]:
                            net.advance(t_samp)
                            result_entry["checkpoints"].append({
                                "time": float(net.time),
                                "m1": float(r1.mass),
                                "m2": float(r2.mass),
                                "p1": float(r1.phase.P),
                                "p2": float(r2.phase.P),
                                "total_mass": float(r1.mass + r2.mass),
                            })

                    elif topo == "reactor_pressure_controller":
                        gas_in = ct.Solution(mech_path)
                        gas_in.TPX = 300.0, 101325.0, "N2:1"
                        res_in = Reservoir(gas_in)

                        gas_r = ct.Solution(mech_path)
                        gas_r.TPX = 300.0, 101325.0, "N2:1"
                        r = IdealGasReactor(gas_r, volume=1.0)
                        mfc = MassFlowController(res_in, r, mdot=0.05)

                        gas_out = ct.Solution(mech_path)
                        gas_out.TPX = 300.0, 101325.0, "N2:1"
                        res_out = Reservoir(gas_out)
                        pc = PressureController(r, res_out, master=mfc, Kv=case["Kv"])
                        net = ReactorNet([r], solver_backend="native_python")
                        net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                        for t_samp in case["sample_times"]:
                            net.advance(t_samp)
                            result_entry["checkpoints"].append({
                                "time": float(net.time),
                                "mass": float(r.mass),
                                "pressure": float(r.phase.P),
                            })

                elif cat == "walls":
                    topo = case["topology"]
                    if topo == "reactor_wall_heat":
                        gas1 = ct.Solution(mech_path)
                        gas1.TPX = case["T1"], case["P1"], "N2:1"
                        r1 = IdealGasReactor(gas1, volume=case["V1"])

                        gas2 = ct.Solution(mech_path)
                        gas2.TPX = case["T2"], case["P2"], "N2:1"
                        r2 = IdealGasReactor(gas2, volume=case["V2"])

                        wall = Wall(r1, r2, area=case["Area"], U=case["U"])
                        net = ReactorNet([r1, r2], solver_backend="native_python")
                        net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                        for t_samp in case["sample_times"]:
                            net.advance(t_samp)
                            u1 = r1.mass * r1.phase.int_energy_mass
                            u2 = r2.mass * r2.phase.int_energy_mass
                            result_entry["checkpoints"].append({
                                "time": float(net.time),
                                "T1": float(r1.phase.T),
                                "T2": float(r2.phase.T),
                                "u_total": float(u1 + u2),
                            })

                    elif topo == "reactor_wall_work":
                        gas1 = ct.Solution(mech_path)
                        gas1.TPX = case["T1"], case["P1"], "N2:1"
                        r1 = IdealGasReactor(gas1, volume=case["V1"])

                        gas2 = ct.Solution(mech_path)
                        gas2.TPX = case["T2"], case["P2"], "N2:1"
                        r2 = IdealGasReactor(gas2, volume=case["V2"])

                        wall = Wall(r1, r2, area=case["Area"], K=case["K_exp"])
                        net = ReactorNet([r1, r2], solver_backend="native_python")
                        net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                        for t_samp in case["sample_times"]:
                            net.advance(t_samp)
                            result_entry["checkpoints"].append({
                                "time": float(net.time),
                                "V1": float(r1.volume),
                                "V2": float(r2.volume),
                                "P1": float(r1.phase.P),
                                "P2": float(r2.phase.P),
                                "V_total": float(r1.volume + r2.volume),
                            })

                elif cat == "multi_reactor":
                    gas_in = ct.Solution(mech_path)
                    gas_in.TPX = 300.0, 101325.0, "CH4:1, O2:2, N2:7.52"
                    res_in = Reservoir(gas_in)

                    gas1 = ct.Solution(mech_path)
                    gas1.TPX = 1200.0, 101325.0, "CH4:1, O2:2, N2:7.52"
                    r1 = IdealGasReactor(gas1, volume=0.5, name="Combustor")

                    gas2 = ct.Solution(mech_path)
                    gas2.TPX = 1000.0, 101325.0, "N2:1"
                    r2 = IdealGasConstPressureReactor(gas2, volume=0.5, name="Plenum")

                    gas_out = ct.Solution(mech_path)
                    gas_out.TPX = 300.0, 101325.0, "N2:1"
                    res_out = Reservoir(gas_out)

                    mfc = MassFlowController(res_in, r1, mdot=0.02)
                    v1 = Valve(r1, r2, Kv=5e-5)
                    v2 = Valve(r2, res_out, Kv=5e-5)

                    net = ReactorNet([r1, r2], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    for t_samp in case["sample_times"]:
                        net.advance(t_samp)
                        result_entry["checkpoints"].append({
                            "time": float(net.time),
                            "T1": float(r1.phase.T),
                            "T2": float(r2.phase.T),
                            "m1": float(r1.mass),
                            "m2": float(r2.mass),
                            "P1": float(r1.phase.P),
                        })

                elif cat == "advance_step":
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T"], case["P"], case["X"]
                    r = IdealGasReactor(gas)
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    t_step1 = net.step()
                    t_step2 = net.step()
                    net.advance(0.001)
                    t_adv = net.time
                    result_entry["checkpoints"].append({
                        "step1_time": float(t_step1),
                        "step2_time": float(t_step2),
                        "advance_time": float(t_adv),
                        "final_temp": float(r.phase.T),
                    })

                elif cat == "advance_limits":
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T"], case["P"], case["X"]
                    r = IdealGasReactor(gas)
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])
                    lims = net.advance_limits
                    lims[net.global_component_index("temperature", 0)] = case.get("delta_T_limit", 50.0)
                    net.advance_limits = lims
                    net.advance(case["requested_tout"])
                    result_entry["checkpoints"].append({
                        "reached_time": float(net.time),
                        "final_temp": float(r.phase.T),
                    })

                elif cat == "reinitialization":
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T_initial"], 101325.0, "N2:1"
                    r = IdealGasReactor(gas)
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    net.advance(case["sample_times"][0])
                    t1 = float(net.time)
                    T1 = float(r.phase.T)

                    # Modify state and reinitialize
                    r.phase.TPX = case["T_reinit"], 101325.0, "N2:1"
                    r.syncState()
                    net.reinitialize()
                    net.advance(case["sample_times"][1])
                    t2 = float(net.time)
                    T2 = float(r.phase.T)

                    result_entry["checkpoints"].append({
                        "t1": t1,
                        "T1": T1,
                        "t2": t2,
                        "T2": T2,
                    })

                elif cat == "derivatives_estimates":
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T"], case["P"], case["X"]
                    r = IdealGasReactor(gas)
                    net = ReactorNet([r], solver_backend="native_python")
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])
                    net.advance(case["t_eval"])

                    result_entry["checkpoints"].append({
                        "time": float(net.time),
                        "temp": float(r.phase.T),
                    })

                elif cat == "sensitivities":
                    gas = ct.Solution(mech_path)
                    gas.TPX = case["T"], case["P"], case["X"]
                    r = IdealGasConstPressureReactor(gas)
                    net = ReactorNet([r], solver_backend="native_python")
                    for rxn in case["sensitivity_reactions"]:
                        r.add_sensitivity_reaction(rxn)
                    net.setTolerances(rtol=case["rtol"], atol=case["atol"])

                    for t_samp in case["sample_times"]:
                        net.advance(t_samp)
                        s0_T = float(net.sensitivity("temperature", 0))
                        s1_T = float(net.sensitivity("temperature", 1)) if len(case["sensitivity_reactions"]) > 1 else 0.0
                        result_entry["checkpoints"].append({
                            "time": float(net.time),
                            "temperature": float(r.phase.T),
                            "sens_rxn0_temp": s0_T,
                            "sens_rxn1_temp": s1_T,
                        })

                elif cat == "failure_semantics":
                    exp = case["expected_error"]
                    gas = ct.Solution(mech_path)
                    gas.TPX = 1000.0, 101325.0, "N2:1"
                    r = IdealGasReactor(gas)
                    net = ReactorNet([r], solver_backend="native_python")

                    caught = False
                    error_msg = ""
                    try:
                        if exp == "advance_backward_time":
                            net.advance(case["t_advance_1"])
                            net.advance(case["t_advance_2"])
                        elif exp == "negative_rtol":
                            net.setRelativeTolerance(case["invalid_rtol"])
                        elif exp == "reactor_index_out_of_range":
                            r_bad = net.reactors[case["invalid_index"]]
                        elif exp == "component_not_found":
                            net.global_component_index(case["invalid_name"], 0)
                    except Exception as exc:
                        caught = True
                        error_msg = str(exc)

                    result_entry["failure_caught"] = caught
                    result_entry["error_message"] = error_msg

            except Exception as e:
                result_entry["status"] = "ERROR"
                result_entry["error"] = str(e)

            results.append(result_entry)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Python port runner completed {len(results)} cases to {output_path}")

if __name__ == "__main__":
    main()
