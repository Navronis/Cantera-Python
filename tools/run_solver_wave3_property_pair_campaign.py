#!/usr/bin/env python3
"""Deterministic pure-Python round-trip campaign for TPX property pairs.

This is an implementation self-consistency campaign, not a native parity
oracle. Each target comes from a direct T-v or source-supported T-quality
state; the inverse setter starts from a separately constructed phase state.
Failures are preserved as records and are never filtered from the totals.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
# TPX itself only needs NumPy and constants; avoid importing the package
# facade, whose mechanism loader has separate YAML dependencies.
if "cantera_python" not in sys.modules:
    package = types.ModuleType("cantera_python")
    package.__path__ = [str(ROOT / "cantera_python")]
    sys.modules["cantera_python"] = package
from cantera_python.tpx import (  # noqa: E402
    CarbonDioxide, HFC134a, Heptane, Hydrogen, Methane, Nitrogen, Oxygen,
    PropertyPair, Water,
)

FLUIDS = {
    "water": Water,
    "carbon_dioxide": CarbonDioxide,
    "nitrogen": Nitrogen,
    "oxygen": Oxygen,
    "hydrogen": Hydrogen,
    "methane": Methane,
    "heptane": Heptane,
    "hfc134a": HFC134a,
}
PAIR_CODES = {
    "TV": (PropertyPair.TV, PropertyPair.VT),
    "HP": (PropertyPair.HP, PropertyPair.PH),
    "SP": (PropertyPair.SP, PropertyPair.PS),
    "PV": (PropertyPair.PV, PropertyPair.VP),
    "TP": (PropertyPair.TP, PropertyPair.PT),
    "UV": (PropertyPair.UV, PropertyPair.VU),
    "ST": (PropertyPair.ST, PropertyPair.TS),
    "SV": (PropertyPair.SV, PropertyPair.VS),
    "UP": (PropertyPair.UP, PropertyPair.PU),
    "VH": (PropertyPair.VH, PropertyPair.HV),
    "TH": (PropertyPair.TH, PropertyPair.HT),
    "SH": (PropertyPair.SH, PropertyPair.HS),
    "PX": (PropertyPair.PX, PropertyPair.XP),
    "TX": (PropertyPair.TX, PropertyPair.XT),
}
GUESS_STRATEGIES = (
    "opposite_phase", "low_density", "high_density", "far_below_temperature",
    "far_above_temperature", "near_saturation", "near_critical", "randomized",
)


class InvalidTargetState(ValueError):
    """The direct state sampler produced a non-physical/non-finite point."""


def get_properties(phase):
    return {
        "T": float(phase.T), "P": float(phase.P()), "rho": float(phase.Rho),
        "v": float(phase.v()), "u": float(phase.u()), "h": float(phase.h()),
        "s": float(phase.s()), "g": float(phase.g()), "quality": float(phase.x()),
        "phase": ("supercritical" if phase.T >= phase.Tcrit() else
                  "liquid" if phase.Rho >= phase.Rhf else
                  "vapor" if phase.Rho <= phase.Rhv else "two_phase"),
    }


def sample_target(factory, pair_name, rng):
    target = factory()
    tmin, tc = float(target.Tmin()), float(target.Tcrit())
    # Stay within the common subcritical saturation domain and away from the
    # critical singularity; targeted boundary suites are reported separately.
    t = rng.uniform(tmin + 0.08 * (tc - tmin), tc - 0.04 * (tc - tmin))
    if pair_name in ("PX", "TX"):
        quality = rng.uniform(0.001, 0.999)
        target.Set(PropertyPair.TX, t, quality)
    else:
        target.set_T(t)
        target.update_sat()
        if rng.random() < 0.5:
            # Vapor targets range from near saturation to a dilute gas.
            volume = (1.0 / target.Rhv) * math.exp(rng.uniform(0.015, 1.5))
        else:
            # Compressed-liquid targets remain on the liquid side of the dome.
            volume = (1.0 / target.Rhf) * rng.uniform(0.72, 0.985)
        target.Set(PropertyPair.TV, t, volume)
    state = get_properties(target)
    required = ("T", "P", "rho", "v", "u", "h", "s", "g", "quality")
    if any(not math.isfinite(state[name]) for name in required):
        raise InvalidTargetState("non-finite direct state property")
    if state["T"] <= 0.0 or state["P"] <= 0.0 or state["rho"] <= 0.0 or state["v"] <= 0.0:
        raise InvalidTargetState("non-positive T/P/density/volume")
    if not 0.0 <= state["quality"] <= 1.0:
        raise InvalidTargetState("quality outside [0,1]")
    return target


def pair_values(name, phase):
    funcs = {
        "TV": lambda: (phase.T, phase.v()),
        "HP": lambda: (phase.h(), phase.P()),
        "SP": lambda: (phase.s(), phase.P()),
        "PV": lambda: (phase.P(), phase.v()),
        "TP": lambda: (phase.T, phase.P()),
        "UV": lambda: (phase.u(), phase.v()),
        "ST": lambda: (phase.s(), phase.T),
        "SV": lambda: (phase.s(), phase.v()),
        "UP": lambda: (phase.u(), phase.P()),
        "VH": lambda: (phase.v(), phase.h()),
        "TH": lambda: (phase.T, phase.h()),
        "SH": lambda: (phase.s(), phase.h()),
        "PX": lambda: (phase.P(), phase.x()),
        "TX": lambda: (phase.T, phase.x()),
    }
    return tuple(float(x) for x in funcs[name]())


def start_guess(factory, target, rng, strategy):
    trial = factory()
    tmin, tmax, tc = float(trial.Tmin()), float(trial.Tmax()), float(trial.Tcrit())
    span = tc - tmin
    if strategy == "opposite_phase":
        tguess = min(tmax, max(tmin, target.T + rng.choice((-0.35, 0.35)) * span))
    elif strategy == "low_density" or strategy == "high_density":
        tguess = target.T
    elif strategy == "far_below_temperature":
        tguess = max(tmin, target.T - 0.45 * span)
    elif strategy == "far_above_temperature":
        tguess = min(tmax, target.T + 0.45 * span)
    elif strategy == "near_saturation":
        tguess = target.T
    elif strategy == "near_critical":
        # Some pure-fluid correlations stop just below the public critical
        # temperature (HFC134a is the current example). Keep this a genuinely
        # near-critical but valid single-phase initialization.
        tguess = max(tmin, tc - 1e-3 * span)
    else:
        tguess = rng.uniform(tmin, min(tmax, tc + 0.2 * span))
    trial.set_T(tguess)
    if strategy == "low_density":
        vguess = 20.0 * trial.Vcrit()
    elif strategy == "high_density":
        vguess = 0.35 * trial.Vcrit()
    elif strategy == "far_above_temperature" and tguess >= tc:
        vguess = trial.Vcrit() * rng.uniform(0.8, 1.2)
    elif strategy == "near_critical":
        vguess = trial.Vcrit()
    elif tguess < tc:
        trial.update_sat()
        if strategy == "near_saturation":
            is_target_vapor = target.Rho <= target.Rhv
            vguess = (1.0 / trial.Rhv * 1.0001 if is_target_vapor
                      else 1.0 / trial.Rhf * 0.9999)
        elif strategy == "randomized":
            vguess = trial.Vcrit() * math.exp(rng.uniform(-3.0, 3.0))
        elif target.Rho <= target.Rhv:
            # Deliberately initialize in the liquid branch.
            vguess = (1.0 / trial.Rhf) * rng.uniform(0.72, 0.95)
        else:
            # Deliberately initialize in the vapor branch.
            vguess = (1.0 / trial.Rhv) * math.exp(rng.uniform(0.05, 1.1))
    else:
        vguess = trial.Vcrit() * math.exp(rng.uniform(-1.2, 1.2))
    trial.Set(PropertyPair.TV, tguess, vguess)
    return trial


def safe_errors(target, solved):
    errors = {}
    for key in ("T", "P", "rho", "v", "u", "h", "s", "g", "quality"):
        a, b = target[key], solved[key]
        if math.isfinite(a) and math.isfinite(b):
            errors[key] = {"abs": abs(a - b), "rel": abs(a - b) / max(abs(a), 1e-300)}
        else:
            errors[key] = {"abs": None, "rel": None, "finite_pair": False}
    return errors


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def quantile(q):
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {"count": len(values), "p50": quantile(0.50), "p95": quantile(0.95),
            "p99": quantile(0.99), "max": values[-1]}


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 726522
    rng = random.Random(seed)
    output = ROOT / "audit/solver_wave3_property_pair_validation.jsonl"
    summary_path = ROOT / "audit/solver_wave3_property_pair_campaign_summary.json"
    totals = {"success": 0, "failure": 0, "nonfinite": 0, "valid_combinations": 0,
              "attempts": 0, "target_generation_failures": 0,
              "start_guess_failures": 0, "invalid_target_samples": 0}
    pair_summary = {}
    error_samples = defaultdict(list)
    invalid_samples = 0

    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for fluid_name, factory in FLUIDS.items():
            for pair_name, (forward_code, reverse_code) in PAIR_CODES.items():
                key = f"{fluid_name}:{pair_name}"
                pair_summary[key] = {"requested_strategy_trials": count, "success": 0, "failure": 0,
                                     "full_state_match": 0, "target_residual_match": 0,
                                     "target_generation_failures": 0,
                                     "start_guess_failures": 0,
                                     "invalid_target_samples": 0}
                totals["valid_combinations"] += 1
                case = 0
                attempts = 0
                attempt_limit = max(100, count * 100)
                while case < count and attempts < attempt_limit:
                    attempts += 1
                    totals["attempts"] += 1
                    case_id = f"W3PAIR_{fluid_name}_{pair_name}_{case:04d}_ATT_{attempts:06d}"
                    row = {"case_id": case_id, "fluid": fluid_name, "pair": pair_name,
                           "seed": seed, "case_index": case, "attempt_index": attempts,
                           "direction": "reverse" if case % 2 else "forward",
                           "guess_strategy": GUESS_STRATEGIES[case % len(GUESS_STRATEGIES)],
                           "iterations": None,
                           "campaign_scope": "PURE_PYTHON_ROUND_TRIP_NOT_NATIVE_PARITY"}
                    target_state = None
                    initial_state = None
                    try:
                        target_phase = sample_target(factory, pair_name, rng)
                        target_state = get_properties(target_phase)
                        target_args = pair_values(pair_name, target_phase)
                    except InvalidTargetState as err:
                        totals["invalid_target_samples"] += 1
                        invalid_samples += 1
                        pair_summary[key]["invalid_target_samples"] += 1
                        row.update({"status": "INVALID_TARGET_SAMPLE", "stage": "target",
                                    "reason": str(err)})
                        stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
                        continue
                    except Exception as err:
                        totals["target_generation_failures"] += 1
                        pair_summary[key]["target_generation_failures"] += 1
                        row.update({"status": "TARGET_GENERATION_FAILURE", "stage": "target",
                                    "exception_type": type(err).__name__, "failure": str(err)})
                        stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
                        continue
                    try:
                        trial = start_guess(factory, target_phase, rng, row["guess_strategy"])
                        initial_state = get_properties(trial)
                    except Exception as err:
                        totals["start_guess_failures"] += 1
                        pair_summary[key]["start_guess_failures"] += 1
                        row.update({"status": "START_GUESS_FAILURE", "stage": "initial_guess",
                                    "exception_type": type(err).__name__, "failure": str(err),
                                    "target": target_state})
                        stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
                        # A rejected start is itself a completed stress case.
                        # Advancing prevents one invalid strategy from being
                        # retried hundreds of times and starving later starts.
                        case += 1
                        continue
                    try:
                        code = reverse_code if case % 2 else forward_code
                        args = target_args[::-1] if case % 2 else target_args
                        trial._record_solver_iterations = True
                        trial._last_set_xy_iterations = 0
                        trial.Set(code, *args)
                        row["iterations"] = trial._last_set_xy_iterations
                        row["iterations_observed"] = True
                        solved_state = get_properties(trial)
                        errors = safe_errors(target_state, solved_state)
                        if any(value["rel"] is None for value in errors.values()):
                            totals["nonfinite"] += 1
                        for property_name, error in errors.items():
                            if error["rel"] is not None:
                                error_samples[property_name].append(error["rel"])
                        target_keys = {
                            "TV": ("T", "v"), "HP": ("h", "P"), "SP": ("s", "P"),
                            "PV": ("P", "v"), "TP": ("T", "P"), "UV": ("u", "v"),
                            "ST": ("s", "T"), "SV": ("s", "v"), "UP": ("u", "P"),
                            "VH": ("v", "h"), "TH": ("T", "h"), "SH": ("s", "h"),
                            "PX": ("P", "quality"), "TX": ("T", "quality"),
                        }[pair_name]
                        target_ok = all(errors[k]["rel"] is not None and errors[k]["rel"] <= 2e-6
                                        for k in target_keys)
                        full_ok = all(errors[k]["rel"] is not None and errors[k]["rel"] <= 2e-6
                                      for k in ("T", "P", "rho", "v", "u", "h", "s", "g", "quality"))
                        row.update({"status": "SUCCESS", "target": target_state,
                                    "start": initial_state, "solved": solved_state,
                                    "errors": errors, "target_residual_match": target_ok,
                                    "full_state_match": full_ok,
                                    "target_arguments": target_args})
                        totals["success"] += 1
                        pair_summary[key]["success"] += 1
                        pair_summary[key]["target_residual_match"] += int(target_ok)
                        pair_summary[key]["full_state_match"] += int(full_ok)
                        case += 1
                    except Exception as err:  # keep every failure in the corpus
                        row.update({"status": "INVERSION_FAILURE", "stage": "setter",
                                    "exception_type": type(err).__name__,
                                    "failure": str(err),
                                    "target": target_state, "start": initial_state})
                        totals["failure"] += 1
                        pair_summary[key]["failure"] += 1
                        case += 1
                    stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
                pair_summary[key]["target_sampling_attempts"] = attempts
                pair_summary[key]["requested_strategy_trials_completed"] = case == count
                print(f"{key}: valid={case}/{count} success={pair_summary[key]['success']} "
                      f"solver_failures={pair_summary[key]['failure']} "
                      f"invalid_samples={pair_summary[key]['invalid_target_samples']}", flush=True)

    summary = {
        "schema": "solver-wave3-property-pair-campaign-v2",
        "pinned_source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "seed": seed, "requested_strategy_trials_per_fluid_pair": count,
        "fluids": list(FLUIDS), "property_pairs": list(PAIR_CODES),
        "initial_guess_strategies": list(GUESS_STRATEGIES),
        "fluid_pair_combinations": totals["valid_combinations"],
        "total_cases": totals["success"] + totals["failure"], "totals": totals,
        "total_records": totals["attempts"],
        "invalid_target_samples": invalid_samples,
        "all_combinations_completed_requested_strategy_trials": all(
            item["requested_strategy_trials_completed"] for item in pair_summary.values()
        ),
        "full_state_relative_error_distributions": {
            name: distribution(values) for name, values in error_samples.items()
        },
        "pair_summary": pair_summary,
        "case_file": output.relative_to(ROOT).as_posix(),
        "case_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "runtime_class_source": "cantera_python.tpx via the same class family returned by newSubstance; tpx_exact duplicate classes are excluded",
        "runtime_tpx_sha256": hashlib.sha256((ROOT / "cantera_python/tpx.py").read_bytes()).hexdigest(),
        "campaign_runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Self-consistency stress campaign only; no native oracle comparison. Each recorded start strategy, including an invalid initialization, consumes one planned trial. Full-state differences may reflect non-unique solutions and require source-branch review.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total_cases": totals["success"] + totals["failure"], **totals,
                      "failed_combinations": sum(v["failure"] > 0 for v in pair_summary.values()),
                      "summary": str(summary_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
