"""Pure Python Kinetics Family 4 differential evaluator.

Executes the deterministic corpus in audit/corpus_kinetics_family4.jsonl
using pure Python cantera_python (zero native C++ code), and writes
audit/python_kinetics_family4_port_results.json.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cantera_python as ct
from cantera_python.kinetics import (
    Reaction, Group, SpeciesNode, Path as KPath, ReactionPathBuilder,
    ReactionPathDiagram, newReactionPathDiagram, newKinetics, KineticsFactory
)
from cantera_python.rates import newReactionRate, ReactionRateFactory


def evaluate_corpus(corpus_path: str, output_path: str):
    print(f"Executing Pure Python Kinetics Family 4 Port against {corpus_path}")

    results = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    for case_idx, case in enumerate(cases, 1):
        case_id = case["case_id"]
        category = case["category"]
        print(f"  [{case_idx}] {case_id} ({category})...", end="", flush=True)

        res = {
            "case_id": case_id,
            "category": category
        }

        if category == "reaction_path_diagram":
            mech = case["mechanism"]
            phase_name = case.get("phase", "")
            element = case.get("element", "H")
            T = float(case.get("temperature", 1000.0))
            P = float(case.get("pressure", 101325.0))
            X = case.get("mole_fractions", "")
            flow_type = case.get("flow_type", "NetFlow")
            scale = float(case.get("scale", -1.0))
            threshold = float(case.get("threshold", 0.005))
            included = case.get("included", [])
            excluded = case.get("excluded", [])
            disp_only = int(case.get("display_only", -1))

            if phase_name:
                sol = ct.Solution(mech, phase_name)
            else:
                sol = ct.Solution(mech)

            if X:
                sol.TPX = T, P, X
            else:
                sol.TP = T, P

            diag = newReactionPathDiagram(sol.kinetics, element)
            diag.setFlowType(flow_type)
            diag.scale = scale
            diag.threshold = threshold
            for s in included:
                diag.include(s)
            for s in excluded:
                diag.exclude(s)
            if disp_only >= 0:
                diag.displayOnly(disp_only)

            diag.build()

            max_flow = float(diag.maxFlow())
            n_nodes = int(diag.nNodes())
            n_paths = int(diag.nPaths())
            sp = [int(k) for k in diag.species()]
            rxns = [int(i) for i in diag.reactions()]
            data_str = diag.getData()
            dot_str = diag.getDot()

            res["maxFlow"] = max_flow
            res["nNodes"] = n_nodes
            res["nPaths"] = n_paths
            res["species"] = sp
            res["reactions"] = rxns
            res["data"] = data_str
            res["dot"] = dot_str

            pairwise = []
            for i1 in range(len(sp)):
                k1 = sp[i1]
                for i2 in range(i1 + 1, len(sp)):
                    k2 = sp[i2]
                    f1 = float(diag.flow(k1, k2))
                    f2 = float(diag.flow(k2, k1))
                    net = float(diag.netFlow(k1, k2))
                    pairwise.append({
                        "k1": k1,
                        "k2": k2,
                        "flow_fwd": f1,
                        "flow_rev": f2,
                        "netFlow": net
                    })
            res["pairwise_flows"] = pairwise

        elif category == "reaction_inspection":
            mech = case["mechanism"]
            phase_name = case.get("phase", "")
            rxn_idx = int(case.get("reaction_index", 0))

            if phase_name:
                sol = ct.Solution(mech, phase_name)
            else:
                sol = ct.Solution(mech)

            kin = sol.kinetics
            rxn = kin.reaction(rxn_idx)

            res["equation"] = rxn.equation()
            res["reactantString"] = rxn.reactantString()
            res["productString"] = rxn.productString()
            res["type"] = rxn.type if isinstance(rxn.type, str) else rxn.type()
            res["reversible"] = bool(rxn.reversible)
            res["duplicate"] = bool(getattr(rxn, "duplicate", False))
            res["valid"] = bool(rxn.valid()) if callable(getattr(rxn, "valid", None)) else True
            res["usesThirdBody"] = bool(rxn.usesThirdBody())
            res["usesElectrochemistry"] = bool(rxn.usesElectrochemistry(kin))
            res["reactants"] = {k: float(v) for k, v in rxn.reactants.items()}
            res["products"] = {k: float(v) for k, v in rxn.products.items()}

        elif category == "group_algebra":
            comp1 = case.get("comp1", [])
            comp2 = case.get("comp2", [])
            elem_syms = case.get("elements", [])

            g1 = Group(comp1) if comp1 else Group()

            res["g1_valid"] = bool(g1.valid())
            res["g1_sign"] = int(g1.sign())
            res["g1_size"] = int(g1.size())
            res["g1_nAtoms"] = int(g1.nAtoms())
            res["g1_fmt"] = g1.fmt(elem_syms)

            if comp2:
                g2 = Group(comp2)
                g_sum = g1 + g2
                g_diff = g1 - g2

                res["sum_valid"] = bool(g_sum.valid())
                res["sum_sign"] = int(g_sum.sign())
                res["sum_nAtoms"] = int(g_sum.nAtoms())
                res["sum_fmt"] = g_sum.fmt(elem_syms)
                res["diff_valid"] = bool(g_diff.valid())
                res["diff_sign"] = int(g_diff.sign())
                res["diff_nAtoms"] = int(g_diff.nAtoms())
                res["diff_fmt"] = g_diff.fmt(elem_syms)

        elif category == "kinetics_factory":
            models = case.get("models", [])
            m_res = {}
            for m in models:
                kin = newKinetics(m)
                m_res[m] = kin.kineticsType() if kin else "null"
            res["models"] = m_res

        elif category == "rate_factory":
            rate_types = case.get("rate_types", [])
            rt_res = {}
            for rt in rate_types:
                rate = newReactionRate(rt)
                rname = rate.type() if (rate and callable(getattr(rate, "type", None))) else "null"
                rt_res[rt] = rname
            res["rate_types"] = rt_res

        results.append(res)
        print(" done.")

    output_obj = {
        "schema_version": "1.0.0",
        "subsystem": "kinetics",
        "closure_family": "FAMILY_4_REACTION_TOPOLOGY_AND_FACTORIES",
        "implementation": "pure_python",
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2)

    print(f"Finished executing pure python kinetics Family 4 port for {len(results)} cases.")


if __name__ == "__main__":
    corpus = "audit/corpus_kinetics_family4.jsonl"
    out = "audit/python_kinetics_family4_port_results.json"
    if len(sys.argv) >= 2:
        corpus = sys.argv[1]
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    evaluate_corpus(corpus, out)
