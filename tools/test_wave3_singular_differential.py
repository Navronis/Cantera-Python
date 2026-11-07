import json
import math
import subprocess
import importlib.util
import inspect
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"

import sys

def load_tpx_module():
    """Load TPX directly so this focused audit does not import unrelated 1D/SciPy modules."""
    package_name = "_wave3_tpx_audit"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "cantera_python")]
    sys.modules[package_name] = package
    for name in ("constants", "tpx"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{name}", ROOT / "cantera_python" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package_name}.tpx"]

tpx = load_tpx_module()
Water, Nitrogen, Oxygen = tpx.Water, tpx.Nitrogen, tpx.Oxygen
Hydrogen, Methane = tpx.Hydrogen, tpx.Methane
CarbonDioxide, Heptane, HFC134a = tpx.CarbonDioxide, tpx.Heptane, tpx.HFC134a
PropertyPair = tpx.PropertyPair
_SET_XY_SOURCE, _SET_XY_FIRST_LINE = inspect.getsourcelines(tpx.Substance._set_xy_source)
_SET_XY_LINES = {
    line.strip(): _SET_XY_FIRST_LINE + offset
    for offset, line in enumerate(_SET_XY_SOURCE)
    if line.strip() in {
        "for iteration in range(201):",
        "dtn = _ieee_divide((X-x_here)*dydv - (Y-y_here)*dxdv, det)",
    }
}
_ITERATION_LINE = _SET_XY_LINES["for iteration in range(201):"]
_DIRECT_DIVISION_LINE = _SET_XY_LINES["dtn = _ieee_divide((X-x_here)*dydv - (Y-y_here)*dxdv, det)"]

PY_FLUIDS = {
    "Water": Water,
    "Nitrogen": Nitrogen,
    "Oxygen": Oxygen,
    "Hydrogen": Hydrogen,
    "Methane": Methane,
    "CarbonDioxide": CarbonDioxide,
    "Heptane": Heptane,
    "HFC134a": HFC134a
}

# ct.Water, ct.Oxygen, etc. are PureFluidPhase objects. Their initialization
# applies the species reference-state offsets in PureFluidPhase::initThermo.
# The direct Python TPX classes default to zero offsets, so apply the same
# offsets here before comparing mass-specific h and s.
REFERENCE_OFFSETS = {
    "Water": (-1.5970805414650720e+07, 3.5199818593423611e+03),
    "CarbonDioxide": (-9.3693028350615799e+06, 2.6341715836312655e+03),
    "Nitrogen": (-4.5974472188034293e+05, 2.4282693624163958e+03),
    "Oxygen": (-4.6626461421166023e+05, 2.0783004885140672e+03),
    "Hydrogen": (-4.4816311513522100e+06, 5.1880892839525332e+03),
    "Methane": (-5.6300845139379026e+06, 4.2269328154547165e+03),
    "Heptane": (-2.2366524267767663e+06, 3.2908203067620075e+03),
    "HFC134a": (-2.0011789732813556e+05, -2.6809005063933728e+02),
}

PROPERTY_TOLERANCES = {
    "T": (1e-8, 1e-2),
    "P": (1e-7, 1.0),
    "v": (1e-5, 1e-10),
    "rho": (1e-5, 1e-8),
    "h": (5e-6, 1e-1),
    "u": (5e-6, 1e-1),
    "s": (5e-6, 1e-3),
    "quality": (1e-5, 1e-8),
}

def run_python_singular():
    cases = [json.loads(l) for l in (AUDIT / "corpus_wave3_singular_jacobian.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    py_results = []

    for c in cases:
        cid = c["case_id"]
        fluid_name = c["fluid"]
        pair = c["pair"]
        sub = PY_FLUIDS[fluid_name]()
        h_offset, s_offset = REFERENCE_OFFSETS[fluid_name]
        sub.m_energy_offset += h_offset
        sub.m_entropy_offset += s_offset
        trace_state = {"iteration": 0, "determinants": []}

        def trace_fallback(frame, event, arg):
            if (event != "line" or frame.f_code.co_filename != tpx.__file__
                    or frame.f_code.co_name != "_set_xy_source"):
                return trace_fallback
            if frame.f_lineno == _ITERATION_LINE:
                trace_state["iteration"] += 1
            if frame.f_lineno == _DIRECT_DIVISION_LINE:
                local = frame.f_locals
                trace_state["determinants"].append(local.get("det"))
            return trace_fallback

        try:
            sys.settrace(trace_fallback)
            if pair == "TP":
                sub.Set(PropertyPair.TP, c["T"], c["P"])
            elif pair == "TX":
                sub.Set(PropertyPair.TX, c["T"], c["x"])
            else:
                raise ValueError(f"Unknown pair: {pair}")

            py_results.append({
                "case_id": cid,
                "status": "SUCCESS",
                "T": sub.Temp(),
                "P": sub.P(),
                "v": sub.v(),
                "h": sub.h(),
                "s": sub.s(),
                "u": sub.u(),
                "rho": sub.Rho,
                "quality": sub.x(),
                "singular_fallback_activated": False,
                "fallback_activation_count": 0,
                "fallback_events": [],
                "python_only_regularization_present": False,
                "jacobian_determinants": trace_state["determinants"],
            })
        except Exception as e:
            py_results.append({
                "case_id": cid,
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "singular_fallback_activated": False,
                "fallback_activation_count": 0,
                "fallback_events": [],
                "python_only_regularization_present": False,
                "jacobian_determinants": trace_state["determinants"],
            })
        finally:
            sys.settrace(None)

    with open(AUDIT / "raw_python_wave3_singular.jsonl", "w", encoding="utf-8") as f:
        for r in py_results:
            f.write(json.dumps(r) + "\n")
    print(f"Python singular run complete: {len(py_results)} cases")

def run_native_singular():
    wsl_script = """
import json
from pathlib import Path
import cantera as ct

WSL_ROOT = Path('/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python')
AUDIT = WSL_ROOT / 'audit'

NAT_FLUIDS = {
    'Water': ct.Water,
    'Nitrogen': ct.Nitrogen,
    'Oxygen': ct.Oxygen,
    'Hydrogen': ct.Hydrogen,
    'Methane': ct.Methane,
    'CarbonDioxide': ct.CarbonDioxide,
    'Heptane': ct.Heptane,
    'HFC134a': ct.Hfc134a
}

cases = [json.loads(l) for l in (AUDIT / 'corpus_wave3_singular_jacobian.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
nat_results = []

for c in cases:
    cid = c['case_id']
    fluid_name = c['fluid']
    pair = c['pair']
    fluid = NAT_FLUIDS[fluid_name]()

    try:
        if pair == 'TP':
            fluid.TP = c['T'], c['P']
        elif pair == 'TX':
            fluid.TQ = c['T'], c['x']
        else:
            raise ValueError(f'Unknown pair: {pair}')

        nat_results.append({
            'case_id': cid,
            'status': 'SUCCESS',
            'T': float(fluid.T),
            'P': float(fluid.P),
            'v': 1.0 / float(fluid.density),
            'h': float(fluid.enthalpy_mass),
            's': float(fluid.entropy_mass)
        })
    except Exception as e:
        nat_results.append({
            'case_id': cid,
            'status': 'FAILURE',
            'error_type': type(e).__name__,
            'error_msg': str(e)
        })

with open(AUDIT / 'raw_native_wave3_singular.jsonl', 'w', encoding='utf-8') as f:
    for r in nat_results:
        f.write(json.dumps(r) + '\\n')
print(f'Native singular run complete: {len(nat_results)} cases')
"""
    cmd = ["wsl", "bash", "-c", f"/home/nebula/cantera-oracle-env/bin/python3 -c \"{wsl_script}\""]
    subprocess.run(cmd, check=True)

def compare(evidence_status):
    nat_recs = [json.loads(l) for l in (AUDIT / "raw_native_wave3_singular.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    py_recs = [json.loads(l) for l in (AUDIT / "raw_python_wave3_singular.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    nat_map = {r["case_id"]: r for r in nat_recs}
    py_map = {r["case_id"]: r for r in py_recs}
    corpus = {r["case_id"]: r for r in (
        json.loads(l) for l in (AUDIT / "corpus_wave3_singular_jacobian.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()
    )}

    if len(nat_recs) != len(py_recs):
        raise AssertionError(f"Record count mismatch: native={len(nat_recs)}, python={len(py_recs)}")
    if len(nat_map) != len(nat_recs) or len(py_map) != len(py_recs):
        raise AssertionError("Duplicate case IDs in singular-Jacobian records")
    if nat_map.keys() != py_map.keys():
        raise AssertionError("Native and Python singular-Jacobian case IDs differ")

    status_matches = 0
    status_mismatches = []
    property_errors = {key: [] for key in PROPERTY_TOLERANCES}
    absolute_errors = {key: [] for key in PROPERTY_TOLERANCES}
    property_mismatches = []
    unavailable_fields = {"internal_energy": 0, "quality": 0}
    residuals = {"T": {"native": [], "python": []}, "P": {"native": [], "python": []},
                 "quality": {"native": [], "python": []}}

    for cid, nat in nat_map.items():
        py = py_map[cid]
        if nat["status"] == py["status"]:
            status_matches += 1
            if nat["status"] == "SUCCESS":
                compared = dict(nat)
                compared_python = dict(py)
                if "rho" not in compared and "v" in compared:
                    compared["rho"] = 1.0 / float(compared["v"])
                if "rho" not in compared_python and "v" in compared_python:
                    compared_python["rho"] = 1.0 / float(compared_python["v"])
                if "u" not in compared or "u" not in compared_python:
                    unavailable_fields["internal_energy"] += 1
                if "quality" not in compared or "quality" not in compared_python:
                    unavailable_fields["quality"] += 1
                for key, (rtol, atol) in PROPERTY_TOLERANCES.items():
                    if key not in compared or key not in compared_python:
                        continue
                    left, right = float(compared[key]), float(compared_python[key])
                    error = abs(left - right)
                    limit = atol + rtol * abs(left)
                    rel = error / max(abs(left), abs(right), 1e-300)
                    property_errors[key].append(rel)
                    absolute_errors[key].append(error)
                    if not math.isfinite(error) or error > limit:
                        property_mismatches.append({
                            "case_id": cid, "property": key,
                            "native": left, "python": right,
                            "absolute_error": error, "relative_error": rel,
                            "allowed_error": limit,
                        })
                source_case = corpus.get(cid, {})
                pair = source_case.get("pair")
                if "T" in source_case:
                    for engine, record in (("native", compared), ("python", compared_python)):
                        if "T" in record:
                            residuals["T"][engine].append(float(record["T"]) - float(source_case["T"]))
                        if pair == "TP" and "P" in record:
                            residuals["P"][engine].append(float(record["P"]) - float(source_case["P"]))
                        if pair == "TX" and "quality" in record:
                            residuals["quality"][engine].append(float(record["quality"]) - float(source_case["x"]))
        else:
            status_mismatches.append({"case_id": cid, "nat_status": nat["status"], "py_status": py["status"], "nat_err": nat.get("error_msg"), "py_err": py.get("error_msg")})

    print(f"Singular Comparison: {len(nat_recs)} cases")
    print(f"  Status matches: {status_matches}/{len(nat_recs)}")
    print(f"  Status mismatches: {len(status_mismatches)}")
    for key, errors in property_errors.items():
        if errors:
            print(f"  Max {key} relative error on successes: {max(errors):.3e}")
    print(f"  Property mismatches: {len(property_mismatches)}")
    print(f"  Native-missing u/quality records: {unavailable_fields}")
    print(f"  Singular-fallback activation: {sum(bool(r.get('singular_fallback_activated')) for r in py_recs)} cases")

    report = {
        "cases": len(nat_recs),
        "status_matches": status_matches,
        "status_mismatches": status_mismatches,
        "property_tolerances": {
            key: {"rtol": rtol, "atol": atol}
            for key, (rtol, atol) in PROPERTY_TOLERANCES.items()
        },
        "observable_fields_compared": {key: len(errors) for key, errors in property_errors.items()},
        "unavailable_native_observables": unavailable_fields,
        "error_distributions": {
            key: {
                "count": len(values),
                "p50_absolute": sorted(values)[int((len(values)-1)*0.50)] if values else None,
                "p95_absolute": sorted(values)[int((len(values)-1)*0.95)] if values else None,
                "p99_absolute": sorted(values)[int((len(values)-1)*0.99)] if values else None,
                "max_absolute": max(values) if values else None,
                "p50_relative": sorted(property_errors[key])[int((len(property_errors[key])-1)*0.50)] if property_errors[key] else None,
                "p95_relative": sorted(property_errors[key])[int((len(property_errors[key])-1)*0.95)] if property_errors[key] else None,
                "p99_relative": sorted(property_errors[key])[int((len(property_errors[key])-1)*0.99)] if property_errors[key] else None,
                "max_relative": max(property_errors[key]) if property_errors[key] else None,
            } for key, values in absolute_errors.items()
        },
        "target_property_residuals": {
            key: {engine: {"count": len(values), "max_abs": max(map(abs, values)) if values else None}
                  for engine, values in engines.items()}
            for key, engines in residuals.items()
        },
        "property_mismatches": property_mismatches,
        "fallback_activation_instrumented": False,
        "direct_jacobian_division_instrumented": True,
        "python_only_regularization_present": False,
        "fallback_activation_cases": sorted(
            r["case_id"] for r in py_recs
            if r.get("singular_fallback_activated")
        ),
        "fallback_activation_count": sum(
            int(r.get("fallback_activation_count", int(bool(r.get("singular_fallback_activated"))))) for r in py_recs
        ),
        "fallback_activation_events": [
            {"case_id": r["case_id"], **event}
            for r in py_recs for event in r.get("fallback_events", [])
        ],
        "failure_messages_compared": False,
        "evidence_status": evidence_status,
    }
    (AUDIT / "wave3_singular_jacobian_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return len(status_mismatches) == 0 and not property_mismatches

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-only", action="store_true",
                        help="Refresh Python records without launching WSL/native Cantera")
    parser.add_argument("--compare-only", action="store_true",
                        help="Compare the existing native and Python records")
    args = parser.parse_args()
    if not args.compare_only:
        run_python_singular()
    if not args.python_only and not args.compare_only:
        run_native_singular()
    if not args.python_only:
        status = (
            "SAVED_NATIVE_AND_PYTHON_RECORDS_COMPARED"
            if args.compare_only else
            "PYTHON_RERUN_COMPARED_TO_UNCHANGED_NATIVE_RECORDS"
            if args.python_only else
            "LIVE_PINNED_NATIVE_AND_PYTHON_RERUN"
        )
        raise SystemExit(0 if compare(status) else 1)
