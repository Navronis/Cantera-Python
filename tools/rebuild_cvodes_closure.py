"""Rebuild true CVodes closure symbol set, runtime call graph, and equivalence contract."""

import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    provenance_dir = repo_root / "provenance"

    reconciliation_path = provenance_dir / "symbol_discovery_false_positive_reconciliation.json"
    with open(reconciliation_path, "r", encoding="utf-8") as f:
        rejected_rows = {r["stable_symbol_id"]: r for r in json.load(f).get("rejected_rows", [])}

    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    closure_symbols = {}
    runtime_sids = []
    non_runtime_sids = []

    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            cls = row.get("class") or ""
            src = row.get("source_file") or ""
            sid = row["stable_symbol_id"]
            if "CVodesIntegrator" in cls or "CVodesIntegrator" in src:
                is_rejected = sid in rejected_rows or row.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
                status = "NON_RUNTIME_CPP_ONLY_PROVEN" if is_rejected else row["current_status"]
                closure_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": cls,
                    "native_method": row["function_or_method"],
                    "signature": row.get("signature", ""),
                    "python_target": f"cantera_python.numerics.CVodesIntegrator.{row['function_or_method']}",
                    "current_status": status,
                    "scientific_runtime_relevance": "NON_RUNTIME_CPP_ONLY_PROVEN" if is_rejected else "SCIENTIFIC_RUNTIME_CANDIDATE",
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": (
                        "None (false-positive statement row proven non-runtime)" if is_rejected
                        else "Differential parity against pinned native Cantera CVodesIntegrator across stiff decay, oscillators, Robertson kinetics, reinitialization, step limits, and forward sensitivity"
                    ),
                }
                if is_rejected:
                    non_runtime_sids.append(sid)
                else:
                    runtime_sids.append(sid)

    closure_path = audit_dir / "cvodes_integrator_closure_symbol_set.json"
    with open(closure_path, "w", encoding="utf-8") as f:
        json.dump(closure_symbols, f, indent=2)

    # Call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "cvodes_integrator",
        "entry_points": [
            "cantera_python.numerics.CVodesIntegrator.__init__",
            "cantera_python.numerics.CVodesIntegrator.initialize",
            "cantera_python.numerics.CVodesIntegrator.reinitialize",
            "cantera_python.numerics.CVodesIntegrator.integrate",
            "cantera_python.numerics.CVodesIntegrator.step",
            "cantera_python.numerics.CVodesIntegrator.solution",
            "cantera_python.numerics.CVodesIntegrator.derivative",
            "cantera_python.numerics.CVodesIntegrator.setTolerances",
            "cantera_python.numerics.CVodesIntegrator.setSensitivityTolerances",
            "cantera_python.numerics.CVodesIntegrator.sensitivity",
            "cantera_python.numerics.CVodesIntegrator.lastOrder",
            "cantera_python.numerics.CVodesIntegrator.nEquations",
            "cantera_python.numerics.CVodesIntegrator.nEvals",
            "cantera_python.numerics.CVodesIntegrator.setMaxOrder",
            "cantera_python.numerics.CVodesIntegrator.setMethod",
            "cantera_python.numerics.CVodesIntegrator.setMaxStepSize",
            "cantera_python.numerics.CVodesIntegrator.setMinStepSize",
            "cantera_python.numerics.CVodesIntegrator.setMaxSteps",
            "cantera_python.numerics.CVodesIntegrator.maxSteps",
            "cantera_python.numerics.CVodesIntegrator.setMaxErrTestFails",
            "cantera_python.numerics.CVodesIntegrator.solverStats",
            "cantera_python.numerics.CVodesIntegrator.setLinearSolverType",
            "cantera_python.numerics.CVodesIntegrator.linearSolverType",
            "cantera_python.numerics.CVodesIntegrator.setBandwidth",
            "cantera_python.numerics.CVodesIntegrator.nSensParams",
            "cantera_python.numerics.CVodesIntegrator.getErrorInfo",
            "cantera_python.numerics.CVodesIntegrator.currentTime",
            "cantera_python.numerics.CVodesIntegrator.setRootFunctionCount",
        ],
        "nodes": {
            "cantera_python.numerics.CVodesIntegrator.initialize": {
                "callees": [
                    "cantera_python.numerics.FuncEval.getState",
                    "cantera_python.numerics.FuncEval.nparams",
                    "cantera_python.numerics.CVodesIntegrator.sensInit",
                    "cantera_python.numerics.CVodesIntegrator.applyOptions",
                ],
                "description": "Allocates state/derivative vectors, initializes BDF Nordsieck history, resets sensitivity if nparams > 0, and configures solver options",
            },
            "cantera_python.numerics.CVodesIntegrator.reinitialize": {
                "callees": [
                    "cantera_python.numerics.FuncEval.getState",
                    "cantera_python.numerics.CVodesIntegrator.applyOptions",
                ],
                "description": "Reloads initial state, clears old BDF step history to prevent contamination, and restarts integration at t0",
            },
            "cantera_python.numerics.CVodesIntegrator.integrate": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator.step",
                    "cantera_python.numerics.CVodesIntegrator._cv_interpolate",
                ],
                "description": "Advances solution through internal variable-step BDF steps until tInteg >= tout, then interpolates exactly to tout",
            },
            "cantera_python.numerics.CVodesIntegrator.step": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator._cv_step",
                ],
                "description": "Performs exactly one internal BDF step towards tout and returns the current internal time reached",
            },
            "cantera_python.numerics.CVodesIntegrator.derivative": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator._cv_dky",
                ],
                "description": "Evaluates the n-th derivative vector at tout using the Nordsieck polynomial interpolation",
            },
            "cantera_python.numerics.CVodesIntegrator.sensitivity": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator._cv_get_sens",
                ],
                "description": "Returns sensitivity of state component k with respect to parameter p (dy_k / dp_p) at current time",
            },
            "cantera_python.numerics.CVodesIntegrator._cv_step": {
                "callees": [
                    "cantera_python.numerics.CVodesIntegrator._cv_nordsieck_predict",
                    "cantera_python.numerics.CVodesIntegrator._cv_newton_corrector",
                    "cantera_python.numerics.CVodesIntegrator._cv_complete_step",
                    "cantera_python.numerics.CVodesIntegrator._cv_prepare_next_step",
                ],
                "description": "Core BDF variable-order (1-5) variable-step algorithm matching SUNDIALS cvStep",
            },
        },
    }

    call_graph_path = audit_dir / "cvodes_integrator_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)

    # Equivalence contract
    contract = {
        "schema_version": 1,
        "solver_family": "cvodes_integrator",
        "native_oracle_pinned_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "equivalence_definitions": {
            "public_observable_parity": {
                "state_at_target_time": "Solution y(t) evaluated via integrate(tout) must match pinned native within specified rtol and atol (max rel err < 1e-4 for stiff problems, < 1e-6 for analytic)",
                "one_step_advancement": "step(tout) must advance by one internal BDF step and return the new internal time tn, advancing monotonically towards tout",
                "derivative_evaluation": "derivative(tout, n) must match numerical derivatives and native CVodeGetDky to within BDF interpolation error",
                "forward_sensitivity": "sensitivity(k, p) must match analytic derivatives and native CVodeGetSens for parameterized ODEs",
                "limits_and_configuration": "Tolerances, max_step, min_step, max_steps, and max_err_test_fails must be strictly enforced with source-equivalent error exceptions",
                "reinitialization": "Calling reinitialize(t0, func) must clear history so subsequent solves are completely uncontaminated by prior runs",
            },
            "numerical_behavior": {
                "stiff_stability": "Must integrate stiff problems (Robertson kinetics, stiff exponential decay lambda=1e6) stably without unphysical oscillations",
                "tolerance_response": "Tightening rtol from 1e-4 down to 1e-10 must show expected error reduction",
                "bdf_orders": "Must support and exercise variable BDF orders 1 through 5",
                "long_horizon": "Must integrate over long time horizons (> 1000 time units) without artificial limit failures when configured with reasonable max_steps",
            },
            "internal_solver_behavior": {
                "step_sequences": "Because Python BDF and native SUNDIALS use adaptive step heuristics, internal step sizes and orders need not be bit-for-bit identical, provided trajectories and public observables match the contract",
            },
        },
    }

    contract_path = audit_dir / "cvodes_equivalence_contract.json"
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    print("Rebuilt CVodes closure symbol set, call graph, and equivalence contract:")
    print(f"  Closure set: {len(closure_symbols)} symbols ({len(runtime_sids)} runtime, {len(non_runtime_sids)} non-runtime)")
    print(f"  Call graph:  {len(call_graph['nodes'])} nodes")
    print(f"  Contract:    {contract_path}")

if __name__ == "__main__":
    main()
