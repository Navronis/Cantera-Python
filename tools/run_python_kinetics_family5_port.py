"""Pure Python Kinetics Family 5 differential evaluator.

Executes the deterministic corpus in audit/corpus_kinetics_family5.jsonl
using pure Python cantera_python (zero native C++ code), and writes
audit/python_kinetics_family5_port_results.json.
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cantera_python as ct
from cantera_python.constants import CanteraError
from cantera_python.rates import (
    newReactionRate, ReactionRateFactory, MultiRate,
    TwoTempPlasmaRate, TwoTempPlasmaData,
    ElectronCollisionPlasmaRate, ElectronCollisionPlasmaData,
    VibrationalRelaxationRate, VibrationalRelaxationData,
    ConstantVibrationalRelaxationRate, StarikovskiyVibrationalRelaxationRate,
    CastelaVibrationalRelaxationRate, MultiStateResolvedVibrationalRelaxationRate,
    CustomFunc1Rate, ReactionRateDelegator
)


def evaluate_corpus(corpus_path: str, output_path: str):
    print(f"Executing Pure Python Kinetics Family 5 Port against {corpus_path}")

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
            if category == "vibrational_relaxation":
                mech = case.get("mechanism", "vibrational-relaxation.yaml")
                T = float(case.get("temperature", 1200.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(mech)
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                kin = sol.kinetics
                nRxn = kin.nReactions()
                kf = [float(x) for x in kin.forward_rate_constants()]
                rop = [float(x) for x in kin.net_rates_of_progress()]

                rate_types = []
                for i in range(nRxn):
                    rxn = kin.reaction(i)
                    r = rxn.rate() if callable(getattr(rxn, "rate", None)) else getattr(rxn, "rate", None)
                    rtype = r.type() if hasattr(r, "type") and callable(r.type) else (getattr(r, "type", "") or type(r).__name__)
                    rate_types.append(rtype)

                res["nReactions"] = nRxn
                res["forward_rate_constants"] = kf
                res["net_rates_of_progress"] = rop
                res["rate_types"] = rate_types

            elif category == "two_temp_plasma":
                mech = case.get("mechanism", "oxygen-plasma.yaml")
                T = float(case.get("temperature", 1000.0))
                Te = float(case.get("electron_temperature", 23209.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(mech)
                # Pinned PlasmaPhase initializes mean_electron_energy = 1.0 eV
                # giving Te = 7736.3312484 K and isotropic EEDF:
                sol.thermo.electron_temperature = 7736.3312484
                sol.thermo.electron_energy_levels = [0.0, 0.1, 1.0, 10.0]
                sol.thermo.electron_energy_distribution = [
                    1.0382504607964884, 1.032585600335941, 0.600752131409468, 1.8014445999315386e-24
                ]
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                kin = sol.kinetics
                nRxn = kin.nReactions()
                kf = [float(x) for x in kin.forward_rate_constants()]

                ttRate = TwoTempPlasmaRate()
                node = {
                    "rate-constant": {
                        "A": 1.0e13,
                        "b": 0.5,
                        "Ea-gas": 1.0e7,
                        "Ea-electron": 2.0e7,
                        "b-gas": 0.1
                    }
                }
                ttRate.setParameters(node)

                data = TwoTempPlasmaData()
                data.update(T=T, Te=Te)
                val = float(ttRate.evalFromStruct(data))
                ddT = float(ttRate.ddTScaledFromStruct(data))

                res["nReactions"] = nRxn
                res["forward_rate_constants"] = kf
                res["twotemp_eval"] = val
                res["twotemp_ddT_scaled"] = ddT
                res["rate_type"] = ttRate.type()

            elif category == "air_plasma":
                mech = case.get("mechanism", "air-plasma.yaml")
                T = float(case.get("temperature", 1000.0))
                Te = float(case.get("electron_temperature", 23209.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(mech)
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                kin = sol.kinetics
                nRxn = kin.nReactions()
                kf = [float(x) for x in kin.forward_rate_constants()]

                res["nReactions"] = nRxn
                res["forward_rate_constants"] = kf

            elif category == "custom_delegator":
                T = float(case.get("temperature", 1000.0))
                cfr = CustomFunc1Rate()
                res["custom_rate_type"] = cfr.type() if callable(getattr(cfr, "type", None)) else cfr.type
                res["temperature"] = T
                res["valid"] = True

            elif category == "multirate_ops":
                rtype = case.get("rate_type", "Arrhenius")
                T = float(case.get("temperature", 1000.0))

                if rtype == "constant-vibrational-relaxation":
                    rate = ConstantVibrationalRelaxationRate(A=1.0e6)
                else:
                    rate = newReactionRate(rtype)

                actualType = rate.type() if callable(getattr(rate, "type", None)) else getattr(rate, "type", "Arrhenius")
                multi = rate.newMultiRate()

                caughtEmpty = False
                try:
                    multi.type()
                except CanteraError:
                    caughtEmpty = True

                multi.add(0, rate)
                multiType = multi.type()

                if rtype == "constant-vibrational-relaxation":
                    rate2 = ConstantVibrationalRelaxationRate(A=2.0e6)
                else:
                    rate2 = newReactionRate(rtype)

                repOk = multi.replace(0, rate2)

                caughtMismatch = False
                try:
                    mismatchRate = newReactionRate("two-temperature-plasma" if rtype == "Arrhenius" else "Arrhenius")
                    multi.replace(0, mismatchRate)
                except CanteraError:
                    caughtMismatch = True

                res["rate_type"] = actualType
                res["multi_type"] = multiType
                res["caught_empty"] = caughtEmpty
                res["replace_ok"] = repOk
                res["caught_mismatch"] = caughtMismatch

            elif category == "kinetics_derivatives":
                mech = case.get("mechanism", "h2o2.yaml")
                T = float(case.get("temperature", 1000.0))
                P = float(case.get("pressure", 101325.0))
                X = case.get("mole_fractions", "")

                sol = ct.Solution(mech)
                if X:
                    sol.TPX = T, P, X
                else:
                    sol.TP = T, P

                kin = sol.kinetics
                nSp = kin.nTotalSpecies()

                net_ddT = [float(x) for x in kin.getNetProductionRates_ddT()]
                creat_ddT = [float(x) for x in kin.getCreationRates_ddT()]
                dest_ddT = [float(x) for x in kin.getDestructionRates_ddT()]
                net_ddP = [float(x) for x in kin.getNetProductionRates_ddP()]

                res["nTotalSpecies"] = nSp
                res["netProductionRates_ddT"] = net_ddT
                res["creationRates_ddT"] = creat_ddT
                res["destructionRates_ddT"] = dest_ddT
                res["netProductionRates_ddP"] = net_ddP

            print(" OK")
        except Exception as e:
            print(f" ERROR: {e}")
            res["error"] = str(e)

        results.append(res)

    payload = {
        "pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "oracle_type": "pure_python_port",
        "family": "FAMILY_5_PLASMA_VIBRATIONAL_AND_MULTIRATE",
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Completed {len(results)} cases, saved to {output_path}")


if __name__ == "__main__":
    evaluate_corpus("audit/corpus_kinetics_family5.jsonl", "audit/python_kinetics_family5_port_results.json")
