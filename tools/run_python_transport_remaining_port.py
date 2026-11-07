"""Pure Python Transport Remaining differential evaluator.

Executes the deterministic corpus in audit/corpus_transport_remaining.jsonl
using pure Python cantera_python (zero native C++ code), and writes
audit/python_transport_remaining_port_results.json.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cantera_python as ct
from cantera_python.transport import (
    UnityLewisTransport, WaterTransport, IonGasTransport, TransportFactory, newTransport
)


def evaluate_corpus(corpus_path: str, output_path: str):
    print(f"Executing Pure Python Transport Remaining Port against {corpus_path}")

    results = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    for case_idx, case in enumerate(cases, 1):
        case_id = case["case_id"]
        category = case["category"]
        print(f"  [{case_idx}/{len(cases)}] {case_id} ({category})...", end="", flush=True)

        res = {
            "case_id": case_id,
            "category": category
        }

        try:
            if category == "water_transport":
                T = float(case["temperature"])
                P = float(case["pressure"])
                rho = float(case["density"])
                kappa = float(case["isothermal_compressibility"])
                alpha = float(case["thermal_expansion_coeff"])

                # Create thermo state with exact density, kappa, alpha
                sol = ct.Solution("water.yaml")
                sol.TP = T, P
                # Attach WaterTransport with the exact state properties
                tran = WaterTransport(sol.thermo)
                sol.setTransport(tran)

                # Set density and derivative hooks on thermo to match state
                sol.thermo._density = rho
                sol.thermo._isothermal_compressibility = kappa
                sol.thermo._thermal_expansion_coeff = alpha

                mu = float(tran.viscosity)
                lam = float(tran.thermal_conductivity)
                bulk_mu = float(tran.bulk_viscosity)
                model = tran.transport_model

                res["transport_model"] = model
                res["viscosity"] = mu
                res["thermal_conductivity"] = lam
                res["bulk_viscosity"] = bulk_mu
                res["density"] = rho

            elif category == "unity_lewis_transport":
                input_file = case.get("input_file", "gri30.yaml")
                phase_name = case.get("phase", "gri30")
                T = float(case.get("temperature", 1500.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(input_file, transport="unity-Lewis-number")
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                tran = sol.transport
                nsp = sol.n_species
                d = [float(x) for x in tran.getMixDiffCoeffs()]
                dt = [float(x) for x in tran.getThermalDiffCoeffs()]
                mu = float(tran.viscosity)
                lam = float(tran.thermal_conductivity)
                model = tran.transport_model

                res["transport_model"] = model
                res["viscosity"] = mu
                res["thermal_conductivity"] = lam
                res["mix_diff_coeffs"] = d
                res["thermal_diff_coeffs"] = dt

            elif category == "ion_gas_transport":
                input_file = case.get("input_file", "ch4_ion.yaml")
                phase_name = case.get("phase", "gas")
                T = float(case.get("temperature", 1200.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(input_file)
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                tran = sol.transport
                nsp = sol.n_species
                mobi = [float(x) for x in tran.getMobilities()]
                d = [float(x) for x in tran.getMixDiffCoeffs()]
                mu = float(tran.viscosity)
                lam = float(tran.thermal_conductivity)
                sigma = float(tran.electrical_conductivity)
                model = tran.transport_model

                res["transport_model"] = model
                res["viscosity"] = mu
                res["thermal_conductivity"] = lam
                res["electrical_conductivity"] = sigma
                res["mobilities"] = mobi
                res["mix_diff_coeffs"] = d

            elif category == "transport_factory":
                model_req = case["model"]
                input_file = case.get("input_file", "gri30.yaml")
                phase_name = case.get("phase", "gri30")

                sol = ct.Solution(input_file, transport="none")
                fac = TransportFactory.factory()
                tr = fac.newTransport(model_req, sol.thermo)
                actual_model = tr.transport_model

                res["requested_model"] = model_req
                res["actual_model"] = actual_model

            print(" [OK]")
        except Exception as e:
            res["error"] = str(e)
            print(f" [ERROR: {e}]")

        results.append(res)

    out_data = {
        "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "evaluator": "pure_python_cantera_python",
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    evaluate_corpus("audit/corpus_transport_remaining.jsonl", "audit/python_transport_remaining_port_results.json")
