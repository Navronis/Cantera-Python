"""Generate deterministic common differential case corpus for Waves 1, 2, and 3.

Produces:
- audit/corpus_wave1.jsonl: >=4,000 independent-start inverse cases + 100 adversarial failure cases
- audit/corpus_wave2.jsonl: >=2,000 Cubic EOS states (PR/RK) covering 1-root, 3-root, continuation, spinodals, failures
- audit/corpus_wave3_forward.jsonl: >=8,000 forward fluid states (1,000 per fluid) across liquid, vapor, supercritical
- audit/corpus_wave3_saturation.jsonl: 200 saturation curve points (25 per fluid)
- audit/corpus_wave3_quality.jsonl: 104 quality dome points across fluids
- audit/corpus_wave3_failures.jsonl: 80 boundary and failure cases across fluids
"""
from __future__ import annotations
import json, math, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"

SEED = 726522841

def generate_wave1_corpus(rng):
    cases = []
    solvers = ["HP", "UV", "SP", "SV"]
    models = ["gri30", "airNASA9", "CO2-PR", "CO2-RK"]

    # 4,000 nominal/independent cases (1,000 per solver)
    for solver_idx, solver in enumerate(solvers):
        for i in range(1000):
            model = models[(solver_idx * 1000 + i) % len(models)]
            case_id = f"W1_{solver}_{i+1:06d}"

            if "CO2" in model:
                t_min, t_max = 220.0, 600.0
                p_min, p_max = 5e4, 5e6
            elif model == "airNASA9":
                t_min, t_max = 300.0, 3000.0
                p_min, p_max = 2e4, 4e6
            else:  # gri30
                t_min, t_max = 300.0, 2800.0
                p_min, p_max = 2e4, 4e6

            t_target = t_min + 50.0 + (t_max - t_min - 100.0) * rng.random()
            p_target = p_min + (p_max - p_min) * rng.random()

            # Guaranteed separation for start state (|T_start - T_target| / T_target >= 0.15)
            offset = 0.25 * t_target
            if t_target + offset < t_max - 20.0:
                t_start = t_target + offset
            else:
                t_start = t_target - offset
            t_start = max(t_min + 10.0, min(t_max - 10.0, t_start))

            p_start = max(p_min, min(p_max, p_target * (0.4 + 1.6 * rng.random())))

            # Region tag
            if t_target > 0.8 * t_max:
                region = "high-T"
            elif t_target < t_min + 0.2 * (t_max - t_min):
                region = "low-T"
            elif p_target > 0.8 * p_max:
                region = "high-P"
            elif p_target < p_min + 0.2 * (p_max - p_min):
                region = "low-P"
            else:
                region = "nominal"

            cases.append({
                "case_id": case_id,
                "wave": 1,
                "model": model,
                "solver": solver,
                "type": "INVERSE_SOLVE",
                "t_target": t_target,
                "p_target": p_target,
                "t_start": t_start,
                "p_start": p_start,
                "region": region,
                "expected_success": True
            })

    # 100 Adversarial / Failure cases
    for i in range(100):
        case_id = f"W1_FAIL_{i+1:04d}"
        fail_type = ["INVALID_P_ZERO", "INVALID_P_NEGATIVE", "INVALID_T_LOW", "INVALID_T_HIGH"][i % 4]
        if fail_type == "INVALID_P_ZERO":
            p_target = 0.0
            t_target = 500.0
        elif fail_type == "INVALID_P_NEGATIVE":
            p_target = -101325.0
            t_target = 500.0
        elif fail_type == "INVALID_T_LOW":
            p_target = 101325.0
            t_target = -50.0
        else:
            p_target = 101325.0
            t_target = 100000.0

        cases.append({
            "case_id": case_id,
            "wave": 1,
            "model": "gri30",
            "solver": "HP",
            "type": "ADVERSARIAL_FAILURE",
            "fail_category": fail_type,
            "t_target": t_target,
            "p_target": p_target,
            "t_start": 400.0,
            "p_start": 101325.0,
            "region": "adversarial",
            "expected_success": False
        })

    return cases

def generate_wave2_corpus(rng):
    cases = []
    models = ["CO2-PR", "CO2-RK"]

    # 1. 1,600 Forward property and root states (800 PR, 800 RK)
    for model in models:
        for i in range(800):
            case_id = f"W2_{model}_{i+1:06d}"
            # Span subcritical, near-critical, and supercritical
            # CO2 Tc ~ 304.13 K, Pc ~ 7.377 MPa
            tc = 304.13
            pc = 7.377e6
            mode = i % 4
            if mode == 0:  # subcritical liquid/vapor coexistence candidate
                t = 250.0 + (tc - 250.0) * rng.random()
                p = 1e5 + (pc - 1e5) * rng.random()
                region = "coexistence_subcritical"
            elif mode == 1:  # near critical
                t = tc * (0.98 + 0.04 * rng.random())
                p = pc * (0.98 + 0.04 * rng.random())
                region = "near_critical"
            elif mode == 2:  # supercritical
                t = tc * (1.05 + 0.5 * rng.random())
                p = pc * (1.05 + 1.5 * rng.random())
                region = "supercritical"
            else:  # low-P gas
                t = 300.0 + 300.0 * rng.random()
                p = 1e4 + 5e5 * rng.random()
                region = "low_p_gas"

            cases.append({
                "case_id": case_id,
                "wave": 2,
                "model": model,
                "type": "CUBIC_FORWARD_ROOT",
                "temperature": t,
                "pressure": p,
                "region": region,
                "expected_success": True
            })

    # 2. 350 Phase continuation trajectories (vapor -> coexistence -> liquid & vice versa)
    for model in models:
        for traj_idx in range(5):
            t_fixed = 280.0 + 15.0 * traj_idx  # below Tc
            p_vals = [5e5 + k * 2e5 for k in range(35)]  # increasing P from vapor to liquid
            for step_idx, p in enumerate(p_vals):
                case_id = f"W2_{model}_TRAJ_{traj_idx+1}_{step_idx+1:03d}"
                cases.append({
                    "case_id": case_id,
                    "wave": 2,
                    "model": model,
                    "type": "CONTINUATION_TRAJECTORY",
                    "trajectory_id": f"{model}_traj_{traj_idx+1}",
                    "step_index": step_idx,
                    "temperature": t_fixed,
                    "pressure": p,
                    "region": "continuation",
                    "expected_success": True
                })

    # 3. 50 Failure / degenerate cases
    for i in range(50):
        case_id = f"W2_FAIL_{i+1:04d}"
        fail_type = ["NEGATIVE_P", "ZERO_P", "NEGATIVE_T", "EXCESSIVE_P"][i % 4]
        if fail_type == "NEGATIVE_P":
            p = -1e5
            t = 300.0
        elif fail_type == "ZERO_P":
            p = 0.0
            t = 300.0
        elif fail_type == "NEGATIVE_T":
            p = 101325.0
            t = -100.0
        else:
            p = 1e12
            t = 300.0

        cases.append({
            "case_id": case_id,
            "wave": 2,
            "model": "CO2-PR",
            "type": "ADVERSARIAL_FAILURE",
            "fail_category": fail_type,
            "temperature": t,
            "pressure": p,
            "region": "adversarial",
            "expected_success": False
        })

    return cases

def generate_wave3_corpus(rng):
    fluids = [
        ("Water", 273.16, 647.286, 22.089e6, 1600.0),
        ("CarbonDioxide", 216.54, 304.21, 7.3835e6, 1500.0),
        ("Nitrogen", 63.15, 126.2, 3.4e6, 2000.0),
        ("Oxygen", 54.34, 154.581, 5.0429e6, 2000.0),
        ("Hydrogen", 13.84, 32.938, 1.2838e6, 5000.0),
        ("Methane", 90.68, 190.555, 4.5988e6, 1700.0),
        ("Heptane", 182.57, 537.68, 2.6199e6, 1000.0),
        ("HFC134a", 170.0, 374.21, 4.05928e6, 455.0)
    ]

    forward_cases = []
    sat_cases = []
    qual_cases = []
    fail_cases = []

    for name, tmin, tc, pc, tmax in fluids:
        # 1. 1,000 forward cases per fluid (8,000 total)
        for i in range(1000):
            case_id = f"W3_FWD_{name.upper()}_{i+1:06d}"
            mode = i % 5
            if mode == 0:  # Compressed liquid
                t = tmin + 0.1 * (tc - tmin) + 0.3 * (tc - tmin) * rng.random()
                p = 0.1 * pc + 0.8 * pc * rng.random()
                region = "compressed_liquid"
            elif mode == 1:  # Subcritical vapor
                t = tc * (0.8 + 0.18 * rng.random())
                p = 0.01 * pc + 0.3 * pc * rng.random()
                region = "subcritical_vapor"
            elif mode == 2:  # Supercritical
                t = tc * (1.02 + 0.4 * rng.random())
                t = min(tmax - 20.0, t)
                p = pc * (1.05 + 1.5 * rng.random())
                region = "supercritical"
            elif mode == 3:  # Low-T liquid
                t = tmin + 5.0 + 0.15 * (tc - tmin) * rng.random()
                p = 1e5 + 0.5 * pc * rng.random()
                region = "low_t_liquid"
            else:  # High-T vapor
                t = tc + 0.2 * (tmax - tc) + 0.4 * (tmax - tc) * rng.random()
                t = min(tmax - 20.0, t)
                p = 0.05 * pc + 0.5 * pc * rng.random()
                region = "high_t_vapor"

            forward_cases.append({
                "case_id": case_id,
                "wave": 3,
                "fluid": name,
                "type": "TPX_FORWARD",
                "temperature": t,
                "pressure": p,
                "region": region,
                "expected_success": True
            })

        # 2. 25 Saturation points per fluid (200 total)
        for i in range(25):
            case_id = f"W3_SAT_{name.upper()}_{i+1:03d}"
            t = tmin + 0.1 * (tc - tmin) + (i / 24.0) * 0.85 * (tc - tmin)
            sat_cases.append({
                "case_id": case_id,
                "wave": 3,
                "fluid": name,
                "type": "TPX_SATURATION",
                "temperature": t,
                "region": "saturation",
                "expected_success": True
            })

        # 3. 13 Quality points across 2 saturation pressures (26 per fluid, 208 total)
        qualities = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
        for p_mult in [0.2, 0.5]:
            p_sat = p_mult * pc
            for q in qualities:
                case_id = f"W3_QUAL_{name.upper()}_P{int(p_mult*100)}_{int(q*1000):04d}"
                qual_cases.append({
                    "case_id": case_id,
                    "wave": 3,
                    "fluid": name,
                    "type": "TPX_QUALITY",
                    "pressure": p_sat,
                    "quality": q,
                    "region": "two_phase_quality",
                    "expected_success": True
                })

        # 4. 10 Failures per fluid (80 total)
        for i in range(10):
            case_id = f"W3_FAIL_{name.upper()}_{i+1:02d}"
            fail_type = ["T_BELOW_MIN", "T_ABOVE_MAX", "P_NEGATIVE", "P_ZERO", "Q_NEGATIVE", "Q_ABOVE_ONE"][i % 6]
            if fail_type == "T_BELOW_MIN":
                t, p, q = tmin - 50.0, 101325.0, 0.0
            elif fail_type == "T_ABOVE_MAX":
                t, p, q = tmax + 200.0, 101325.0, 0.0
            elif fail_type == "P_NEGATIVE":
                t, p, q = 300.0, -1e5, 0.0
            elif fail_type == "P_ZERO":
                t, p, q = 300.0, 0.0, 0.0
            elif fail_type == "Q_NEGATIVE":
                t, p, q = 0.5 * (tmin + tc), 0.2 * pc, -0.2
            else:
                t, p, q = 0.5 * (tmin + tc), 0.2 * pc, 1.5

            fail_cases.append({
                "case_id": case_id,
                "wave": 3,
                "fluid": name,
                "type": "ADVERSARIAL_FAILURE",
                "fail_category": fail_type,
                "temperature": t,
                "pressure": p,
                "quality": q,
                "region": "adversarial",
                "expected_success": False
            })

    return forward_cases, sat_cases, qual_cases, fail_cases

def main():
    print(f"Generating deterministic differential test corpus with seed={SEED}...")
    rng = random.Random(SEED)

    w1_cases = generate_wave1_corpus(rng)
    w2_cases = generate_wave2_corpus(rng)
    w3_fwd, w3_sat, w3_qual, w3_fail = generate_wave3_corpus(rng)

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(items):5d} cases -> {path.name}")

    write_jsonl(AUDIT / "corpus_wave1.jsonl", w1_cases)
    write_jsonl(AUDIT / "corpus_wave2.jsonl", w2_cases)
    write_jsonl(AUDIT / "corpus_wave3_forward.jsonl", w3_fwd)
    write_jsonl(AUDIT / "corpus_wave3_saturation.jsonl", w3_sat)
    write_jsonl(AUDIT / "corpus_wave3_quality.jsonl", w3_qual)
    write_jsonl(AUDIT / "corpus_wave3_failures.jsonl", w3_fail)

    print("Corpus generation complete!")

if __name__ == "__main__":
    main()
