import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"

# Critical parameters from TPX fluid models
FLUIDS = {
    "Water": {"Tc": 647.096, "Pc": 22.064e6},
    "Nitrogen": {"Tc": 126.2, "Pc": 3.3958e6},
    "Oxygen": {"Tc": 154.581, "Pc": 5.043e6},
    "Hydrogen": {"Tc": 33.145, "Pc": 1.2964e6},
    "Methane": {"Tc": 190.564, "Pc": 4.5992e6},
    "CarbonDioxide": {"Tc": 304.1282, "Pc": 7.3773e6},
    "Heptane": {"Tc": 540.2, "Pc": 2.74e6},
    "HFC134a": {"Tc": 374.18, "Pc": 4.0593e6}
}

def generate():
    cases = []
    case_idx = 1

    for fluid, params in FLUIDS.items():
        Tc = params["Tc"]
        Pc = params["Pc"]

        # 1. Exact Critical Point
        cases.append({
            "case_id": f"W3_SING_{fluid}_{case_idx:04d}",
            "fluid": fluid,
            "type": "EXACT_CRITICAL_POINT",
            "T": Tc,
            "P": Pc,
            "pair": "TP"
        })
        case_idx += 1

        # 2. Near-Critical Perturbations (approaching critical point)
        for d_rel in [-1e-2, -1e-4, -1e-6, 1e-6, 1e-4, 1e-2]:
            T_test = Tc * (1.0 + d_rel)
            P_test = Pc * (1.0 + d_rel)
            cases.append({
                "case_id": f"W3_SING_{fluid}_{case_idx:04d}",
                "fluid": fluid,
                "type": "NEAR_CRITICAL_PERTURBATION",
                "delta_relative": d_rel,
                "T": T_test,
                "P": P_test,
                "pair": "TP"
            })
            case_idx += 1

        # 3. Saturation curve points
        for frac in [0.6, 0.75, 0.9, 0.98]:
            T_sub = Tc * frac
            cases.append({
                "case_id": f"W3_SING_{fluid}_{case_idx:04d}",
                "fluid": fluid,
                "type": "SATURATION_CURVE",
                "T": T_sub,
                "pair": "TX",
                "x": 0.5 # Two-phase mixture
            })
            case_idx += 1

    out_file = AUDIT / "corpus_wave3_singular_jacobian.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")

    print(f"Generated {len(cases)} singular/near-singular test cases -> {out_file.name}")

if __name__ == "__main__":
    generate()
