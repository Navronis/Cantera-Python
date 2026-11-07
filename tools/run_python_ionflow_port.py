"""Python IonFlow port runner for differential validation against native oracle.

Reads:
  - audit/corpus_ionflow.jsonl
Writes:
  - audit/python_ionflow_port_results.json
"""
import sys
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cantera_python as ct
from cantera_python.oned import (
    IonFlow,
    Inlet1D,
    Outlet1D,
    Sim1D,
    c_offset_U,
    c_offset_V,
    c_offset_T,
    c_offset_L,
    c_offset_E,
    c_offset_Uo,
    c_offset_Y,
)


def run_port():
    corpus_path = root / "audit" / "corpus_ionflow.jsonl"
    output_path = root / "audit" / "python_ionflow_port_results.json"

    print(f"Loading IonFlow corpus from {corpus_path}")
    cases_out = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case_map = json.loads(line)
            case_id = case_map["case_id"]
            mech = case_map["mechanism"]
            domain_type = case_map["domain_type"]
            grid = case_map["grid"]
            pressure = case_map["pressure"]
            energy_enabled = case_map["energy_enabled"]
            efield_enabled = case_map["electric_field_enabled"]
            has_electron_transport = case_map.get("electron_transport", False)
            rdt = case_map["rdt"]

            n_pts = len(grid)
            sol = ct.Solution(mech)
            inlet = Inlet1D(sol, name="inlet")
            flow = IonFlow(sol, id="ion_flame", points=n_pts)
            outlet = Outlet1D(sol, name="outlet")

            flow.setupGrid(grid)
            flow.setPressure(pressure)

            if domain_type == "free-flow":
                flow.setFreeFlow()
            elif domain_type == "axisymmetric-flow":
                flow.setAxisymmetricFlow()
            elif domain_type == "unstrained-flow":
                flow.setUnstrainedFlow()

            flow.solveEnergyEqn(energy_enabled)

            if efield_enabled:
                flow.solveElectricField()
            else:
                flow.fixElectricField()

            if has_electron_transport:
                tfix = case_map["tfix"]
                diff_e = case_map["diff_e"]
                mobi_e = case_map["mobi_e"]
                flow.setElectronTransport(tfix, diff_e, mobi_e)

            sim = Sim1D((inlet, flow, outlet))
            sim.init()

            x_global = np.zeros(sim.size(), dtype=float)
            pts_data = case_map["state_profile"]

            flow_loc = flow.loc()
            nsp = sol.n_species

            for j in range(n_pts):
                pt = pts_data[j]
                u = pt["u"]
                V = pt["V"]
                T = pt["T"]
                L = pt["Lambda"]
                E_val = pt["eField"]
                Uo = pt["Uo"]
                Y_map = pt["Y"]

                x_global[flow_loc + flow.index(c_offset_U, j)] = u
                x_global[flow_loc + flow.index(c_offset_V, j)] = V
                x_global[flow_loc + flow.index(c_offset_T, j)] = T
                x_global[flow_loc + flow.index(c_offset_L, j)] = L
                x_global[flow_loc + flow.index(c_offset_E, j)] = E_val
                x_global[flow_loc + flow.index(c_offset_Uo, j)] = Uo

                for k in range(nsp):
                    sp_name = sol.species_name(k)
                    y_val = Y_map.get(sp_name, 0.0)
                    x_global[flow_loc + flow.index(c_offset_Y + k, j)] = y_val

            inlet.temperature = pts_data[0]["T"]
            inlet.mdot = 0.5

            rsd_global = np.zeros(sim.size(), dtype=float)
            diag_global = np.zeros(sim.size(), dtype=int)

            flow.eval(None, x_global, rsd_global, diag_global, rdt=rdt)

            res_points = []
            for j in range(n_pts):
                y_res = {}
                for k in range(nsp):
                    sp = sol.species_name(k)
                    y_res[sp] = float(rsd_global[flow_loc + flow.index(c_offset_Y + k, j)])

                res_points.append({
                    "point": j,
                    "z": grid[j],
                    "u": float(rsd_global[flow_loc + flow.index(c_offset_U, j)]),
                    "V": float(rsd_global[flow_loc + flow.index(c_offset_V, j)]),
                    "T": float(rsd_global[flow_loc + flow.index(c_offset_T, j)]),
                    "Lambda": float(rsd_global[flow_loc + flow.index(c_offset_L, j)]),
                    "eField": float(rsd_global[flow_loc + flow.index(c_offset_E, j)]),
                    "Uo": float(rsd_global[flow_loc + flow.index(c_offset_Uo, j)]),
                    "diag_u": int(diag_global[flow_loc + flow.index(c_offset_U, j)]),
                    "diag_T": int(diag_global[flow_loc + flow.index(c_offset_T, j)]),
                    "diag_E": int(diag_global[flow_loc + flow.index(c_offset_E, j)]),
                    "Y": y_res,
                })

            cases_out.append({
                "case_id": case_id,
                "flow_size": flow.size(),
                "flow_loc": flow_loc,
                "residuals": res_points,
            })

    output_data = {
        "cases": cases_out,
        "total_cases": len(cases_out),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Executed {len(cases_out)} cases successfully. Port results written to {output_path}")


if __name__ == "__main__":
    run_port()
