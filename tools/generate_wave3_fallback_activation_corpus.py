#!/usr/bin/env python3
"""Test-only trace for exercising the TPX Jacobian fallback via public pairs."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
PACKAGE = "_wave3_fallback_probe"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "cantera_python")]
sys.modules[PACKAGE] = package
for module_name in ("constants", "tpx"):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{module_name}", ROOT / "cantera_python" / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
tpx = sys.modules[f"{PACKAGE}.tpx"]

FLUIDS = {
    "Water": tpx.Water,
    "Nitrogen": tpx.Nitrogen,
    "Oxygen": tpx.Oxygen,
    "Hydrogen": tpx.Hydrogen,
    "Methane": tpx.Methane,
    "CarbonDioxide": tpx.CarbonDioxide,
    "Heptane": tpx.Heptane,
    "HFC134a": tpx.HFC134a,
}
OFFSETS = {
    "Water": (-1.5970805414650720e7, 3.5199818593423611e3),
    "CarbonDioxide": (-9.369302835061580e6, 2.6341715836312655e3),
    "Nitrogen": (-4.597447218803429e5, 2.4282693624163958e3),
    "Oxygen": (-4.662646142116602e5, 2.0783004885140672e3),
    "Hydrogen": (-4.481631151352210e6, 5.1880892839525332e3),
    "Methane": (-5.630084513937902e6, 4.2269328154547165e3),
    "Heptane": (-2.236652426776766e6, 3.2908203067620075e3),
    "HFC134a": (-2.001178973281356e5, -2.6809005063933728e2),
}
PAIRS = [
    ("TP", tpx.PropertyPair.TP, tpx.PropertyFlag.T, tpx.PropertyFlag.P),
    ("HP", tpx.PropertyPair.HP, tpx.PropertyFlag.H, tpx.PropertyFlag.P),
    ("SP", tpx.PropertyPair.SP, tpx.PropertyFlag.S, tpx.PropertyFlag.P),
    ("UV", tpx.PropertyPair.UV, tpx.PropertyFlag.U, tpx.PropertyFlag.V),
    ("PV", tpx.PropertyPair.PV, tpx.PropertyFlag.P, tpx.PropertyFlag.V),
    ("VH", tpx.PropertyPair.VH, tpx.PropertyFlag.V, tpx.PropertyFlag.H),
    ("TH", tpx.PropertyPair.TH, tpx.PropertyFlag.T, tpx.PropertyFlag.H),
    ("ST", tpx.PropertyPair.ST, tpx.PropertyFlag.S, tpx.PropertyFlag.T),
    ("SV", tpx.PropertyPair.SV, tpx.PropertyFlag.S, tpx.PropertyFlag.V),
    ("UP", tpx.PropertyPair.UP, tpx.PropertyFlag.U, tpx.PropertyFlag.P),
    ("SH", tpx.PropertyPair.SH, tpx.PropertyFlag.S, tpx.PropertyFlag.H),
]
SOURCE, FIRST_LINE = inspect.getsourcelines(tpx.Substance._set_xy_source)
LINE = {line.strip(): FIRST_LINE + i for i, line in enumerate(SOURCE)}
ITERATION_LINE = LINE["for iteration in range(201):"]
NEWTON_LINE = LINE["dtn = _ieee_divide((X-x_here)*dydv - (Y-y_here)*dxdv, det)"]


def make_fluid(name):
    fluid = FLUIDS[name]()
    fluid.m_energy_offset += OFFSETS[name][0]
    fluid.m_entropy_offset += OFFSETS[name][1]
    return fluid


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def traced_set(fluid, pair, first, second):
    trace = {"iteration": 0, "determinants": []}

    def tracer(frame, event, arg):
        if event != "line" or frame.f_code.co_filename != tpx.__file__ or frame.f_code.co_name != "_set_xy_source":
            return tracer
        local = frame.f_locals
        if frame.f_lineno == ITERATION_LINE:
            trace["iteration"] += 1
        elif frame.f_lineno == NEWTON_LINE:
            determinant = local.get("det")
            if determinant is not None:
                trace["determinants"].append(float(determinant))
        return tracer

    sys.settrace(tracer)
    try:
        fluid.Set(pair, first, second)
        status, error = "SUCCESS", None
    except Exception as exc:  # probe records errors as observations
        status, error = "FAILURE", f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(None)
    finite_determinants = [abs(x) for x in trace["determinants"] if math.isfinite(x)]
    determinant_summary = {
        "evaluated_count": len(trace["determinants"]),
        "nonfinite_count": sum(not math.isfinite(x) for x in trace["determinants"]),
        "minimum_absolute_finite": min(finite_determinants) if finite_determinants else None,
    }
    return status, error, [], determinant_summary, trace["determinants"]


def main():
    attempts = []
    case_no = 0
    for name, cls in FLUIDS.items():
        reference = make_fluid(name)
        critical_t, critical_v = reference.Tcrit(), reference.Vcrit()
        # Resolve the critical neighborhood more finely than the initial
        # coarse probe, where no determinant came remotely close to the
        # absolute 1e-25 fallback threshold.
        for t_scale in (0.999999, 0.9999999, 1.0, 1.0000001, 1.000001):
            for v_scale in (0.999999, 0.9999999, 1.0, 1.0000001, 1.000001):
                target_t, target_v = critical_t * t_scale, critical_v * v_scale
                try:
                    target = make_fluid(name)
                    target.Set(tpx.PropertyPair.TV, target_t, target_v)
                    for pair_name, pair, xflag, yflag in PAIRS:
                        first, second = target.prop(xflag), target.prop(yflag)
                        for start_t_scale, start_v_scale in ((0.97, 1.03), (1.03, 0.97)):
                            case_no += 1
                            probe = make_fluid(name)
                            start_t = max(probe.Tmin(), min(probe.Tmax(), target_t * start_t_scale))
                            start_v = target_v * start_v_scale
                            case_id = f"W3_FB_{case_no:05d}"
                            try:
                                probe.Set(tpx.PropertyPair.TV, start_t, start_v)
                                status, error, events, determinant_summary, determinants = traced_set(probe, pair, first, second)
                                result = {
                                    "case_id": case_id,
                                    "fluid": name,
                                    "pair": pair_name,
                                    "target_temperature": target_t,
                                    "target_volume": target_v,
                                    "start_temperature": start_t,
                                    "start_volume": start_v,
                                    "target_first": first,
                                    "target_second": second,
                                    "status": status,
                                    "error": error,
                                    "fallback_activated": bool(events),
                                    "fallback_activation_count": len(events),
                                    "fallback_threshold": None,
                                    "python_only_regularization_present": False,
                                    "determinants": determinants,
                                    "fallback_iteration_numbers": [event["iteration"] for event in events],
                                    "fallback_pre_states": [event["state_before_fallback"] for event in events],
                                    "fallback_proposed_steps": [{"temperature": event.get("proposed_temperature_step"), "volume": event.get("proposed_volume_step")} for event in events],
                                    "fallback_post_states": [event.get("state_after_fallback") for event in events],
                                    "fallback_events": events,
                                    "determinant_summary": determinant_summary,
                                    "result": {
                                        "T": probe.Temp(), "P": probe.P(), "v": probe.v(),
                                        "rho": probe.Rho, "h": probe.h(), "u": probe.u(),
                                        "s": probe.s(), "quality": probe.x(),
                                    },
                                }
                                attempts.append(result)
                            except Exception as exc:
                                attempts.append({
                                    "case_id": case_id, "fluid": name, "pair": pair_name,
                                    "target_temperature": target_t, "target_volume": target_v,
                                    "start_temperature": start_t, "start_volume": start_v,
                                    "status": "SETUP_FAILURE", "error": f"{type(exc).__name__}: {exc}",
                                    "fallback_activated": False, "fallback_activation_count": 0,
                                    "fallback_threshold": None,
                                    "python_only_regularization_present": False,
                                    "determinants": [],
                                    "fallback_iteration_numbers": [], "fallback_pre_states": [],
                                    "fallback_proposed_steps": [], "fallback_post_states": [],
                                    "fallback_events": [], "determinant_summary": None,
                                })
                except Exception:
                    continue
    activated = [row for row in attempts if row.get("fallback_activated")]
    corpus = {
        "schema": "wave3-direct-jacobian-division-audit-v1",
        "instrumentation": "sys.settrace line probes; no scientific module source or numerical behavior is changed",
        "python_only_fallback_present": False,
        "python_jacobian_update": "unconditional IEEE-754 determinant division, matching pinned Sub.cpp:701-704",
        "activation_condition": None,
        "targeted_regions": [
            "critical and near-critical temperature/volume perturbations",
            "all 11 supported public property-pair directions",
            "two offset initial states per target/pair",
        ],
        "attempt_count": len(attempts),
        "activated_count": len(activated),
        "activated_case_ids": [row["case_id"] for row in activated],
        "activated_cases": activated,
        "all_attempts": attempts,
    }
    path = AUDIT / "wave3_fallback_activation_probe.json"
    path.write_text(json.dumps(json_safe(corpus), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Targeted cases: {len(attempts)}; fallback activations: {len(activated)}; wrote {path}")


if __name__ == "__main__":
    main()
