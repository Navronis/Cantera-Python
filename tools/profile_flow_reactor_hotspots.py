"""Profile FlowReactor execution path to measure exact hotspot percentages."""

import cProfile
import pstats
import io
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from tools.run_python_flow_reactor_port import run_cases

def main():
    corpus_path = repo_root / "audit" / "corpus_flow_reactor.jsonl"
    temp_out = repo_root / "audit" / "profile_flow_out.jsonl"

    pr = cProfile.Profile()
    pr.enable()
    run_cases(corpus_path, temp_out, "profile_run")
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(50)
    profile_text = s.getvalue()

    # Categorize time
    total_time = ps.total_tt
    kinetics_time = 0.0
    thermo_time = 0.0
    dae_residual_time = 0.0
    jacobian_time = 0.0
    newton_solve_time = 0.0
    integrator_time = 0.0

    for func, (cc, nc, tt, ct, callers) in ps.stats.items():
        filename, line, name = func
        fn_lower = filename.lower()
        n_lower = name.lower()

        if "kinetics" in fn_lower or "rate" in n_lower or "reaction" in n_lower:
            kinetics_time += tt
        elif "thermo" in fn_lower or "phase" in fn_lower or "state" in fn_lower or "ideal_gas" in fn_lower:
            thermo_time += tt
        elif "evaldae" in n_lower:
            dae_residual_time += tt
        elif "jacobian" in n_lower or "fd" in n_lower:
            jacobian_time += tt
        elif "solve" in n_lower or "factor" in n_lower or "lu" in n_lower:
            newton_solve_time += tt
        elif "numerics" in fn_lower or "idas" in fn_lower or "bdf" in fn_lower:
            integrator_time += tt

    other_time = max(0.0, total_time - (kinetics_time + thermo_time + dae_residual_time + jacobian_time + newton_solve_time + integrator_time))

    breakdown = {
        "total_execution_seconds": round(total_time, 3),
        "hotspot_categories": {
            "thermochemistry_and_kinetics": {
                "seconds": round(kinetics_time + thermo_time, 3),
                "percentage": round(100.0 * (kinetics_time + thermo_time) / total_time, 1)
            },
            "finite_difference_jacobians": {
                "seconds": round(jacobian_time, 3),
                "percentage": round(100.0 * jacobian_time / total_time, 1)
            },
            "dae_residual_evaluations": {
                "seconds": round(dae_residual_time, 3),
                "percentage": round(100.0 * dae_residual_time / total_time, 1)
            },
            "newton_linear_solves_and_factorization": {
                "seconds": round(newton_solve_time, 3),
                "percentage": round(100.0 * newton_solve_time / total_time, 1)
            },
            "bdf_integrator_state_machine": {
                "seconds": round(integrator_time, 3),
                "percentage": round(100.0 * integrator_time / total_time, 1)
            },
            "python_call_overhead_and_other": {
                "seconds": round(other_time, 3),
                "percentage": round(100.0 * other_time / total_time, 1)
            }
        }
    }

    print(json.dumps(breakdown, indent=2))
    return breakdown

if __name__ == "__main__":
    main()
