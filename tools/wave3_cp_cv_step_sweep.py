#!/usr/bin/env python3
"""Wave 3 cp/cv step-size sweep across dt/T = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5].
Evaluates both Python tpx and Native Cantera oracle across all 8 pure fluids.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cantera_python.tpx as py_tpx

AUDIT = ROOT / "audit"

FLUID_NAMES = [
    "Water", "Nitrogen", "Oxygen", "Hydrogen",
    "Methane", "CarbonDioxide", "Heptane", "HFC134a",
]

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

BENCHMARK_STATES = {
    "Water": [
        {"name": "liquid", "T": 300.0, "P": 101325.0},
        {"name": "supercritical", "T": 700.0, "P": 3.0e7},
        {"name": "vapor", "T": 450.0, "P": 101325.0},
    ],
    "Nitrogen": [
        {"name": "liquid", "T": 75.0, "P": 1.0e6},
        {"name": "supercritical", "T": 150.0, "P": 5.0e6},
        {"name": "gas", "T": 300.0, "P": 101325.0},
    ],
    "Oxygen": [
        {"name": "liquid", "T": 80.0, "P": 1.0e6},
        {"name": "supercritical", "T": 170.0, "P": 6.0e6},
        {"name": "gas", "T": 300.0, "P": 101325.0},
    ],
    "Hydrogen": [
        {"name": "liquid", "T": 25.0, "P": 1.0e6},
        {"name": "supercritical", "T": 50.0, "P": 3.0e6},
        {"name": "gas", "T": 300.0, "P": 101325.0},
    ],
    "Methane": [
        {"name": "liquid", "T": 100.0, "P": 1.0e6},
        {"name": "supercritical", "T": 200.0, "P": 6.0e6},
        {"name": "gas", "T": 300.0, "P": 101325.0},
    ],
    "CarbonDioxide": [
        {"name": "liquid", "T": 250.0, "P": 5.0e6},
        {"name": "supercritical", "T": 320.0, "P": 1.0e7},
        {"name": "gas", "T": 350.0, "P": 101325.0},
    ],
    "Heptane": [
        {"name": "liquid", "T": 300.0, "P": 101325.0},
        {"name": "supercritical", "T": 600.0, "P": 5.0e6},
        {"name": "gas", "T": 450.0, "P": 101325.0},
    ],
    "HFC134a": [
        {"name": "liquid", "T": 250.0, "P": 1.0e6},
        {"name": "supercritical", "T": 400.0, "P": 6.0e6},
        {"name": "gas", "T": 350.0, "P": 101325.0},
    ],
}

STEP_FACTORS = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]


def percentile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def compute_finite_diff_properties(sub, step_ratio):
    """Compute cv and cp using finite difference with specific step_ratio = dt/T."""
    T = sub.Temp()
    dt = step_ratio * T
    v = sub.v()
    P0 = sub.P()
    rho = 1.0 / v
    Tmin = sub.Tmin()
    Tmax = sub.Tmax()

    # Isochoric heat capacity: cv = T * (s2 - s1) / (T2 - T1) at constant v
    T1_cv = max(Tmin, T - dt)
    T2_cv = min(Tmax, T + dt)
    sub.Set(py_tpx.PropertyPair.TV, T1_cv, v)
    s1_cv = sub.s()
    sub.Set(py_tpx.PropertyPair.TV, T2_cv, v)
    s2_cv = sub.s()
    sub.Set(py_tpx.PropertyPair.TV, T, v)
    cv_val = T * (s2_cv - s1_cv) / (T2_cv - T1_cv)

    # Isobaric heat capacity: cp = T * (s2 - s1) / (T2 - T1) at constant P
    rhf = getattr(sub, 'Rhf', float('inf'))
    is_liquid = rho >= rhf
    if is_liquid:
        T1_cp = max(Tmin, T - dt)
        try:
            T2_cp = min(sub.Tsat(P0), min(Tmax, T + dt))
        except Exception:
            T2_cp = min(Tmax, T + dt)
        sub.Set(py_tpx.PropertyPair.TP, T1_cp, P0)
        s1_cp = sub.s()
        if T2_cp < T + dt:
            sub.Set(py_tpx.PropertyPair.TX, T2_cp, 0.0)
        else:
            sub.Set(py_tpx.PropertyPair.TP, T2_cp, P0)
        s2_cp = sub.s()
    else:
        try:
            T1_cp = max(sub.Tsat(P0), max(Tmin, T - dt))
        except Exception:
            T1_cp = max(Tmin, T - dt)
        T2_cp = min(Tmax, T + dt)
        if T1_cp > T - dt:
            sub.Set(py_tpx.PropertyPair.TX, T1_cp, 1.0)
        else:
            sub.Set(py_tpx.PropertyPair.TP, T1_cp, P0)
        s1_cp = sub.s()
        sub.Set(py_tpx.PropertyPair.TP, T2_cp, P0)
        s2_cp = sub.s()

    sub.Set(py_tpx.PropertyPair.TV, T, v)
    cp_val = T * (s2_cp - s1_cp) / (T2_cp - T1_cp)

    return cv_val, cp_val


def run_python_sweep():
    records = []
    for fluid_name in FLUID_NAMES:
        cls = getattr(py_tpx, fluid_name)
        hoff, soff = OFFSETS[fluid_name]
        states = BENCHMARK_STATES[fluid_name]

        for st in states:
            sub = cls()
            sub.m_energy_offset += hoff
            sub.m_entropy_offset += soff
            sub.Set(py_tpx.PropertyPair.TP, st["T"], st["P"])

            nom_cv = float(sub.cv())
            nom_cp = float(sub.cp())

            for step in STEP_FACTORS:
                cv_fd, cp_fd = compute_finite_diff_properties(sub, step)
                records.append({
                    "fluid": fluid_name,
                    "state_name": st["name"],
                    "T": st["T"],
                    "P": st["P"],
                    "step_ratio": step,
                    "py_cv_fd": cv_fd,
                    "py_cp_fd": cp_fd,
                    "py_cv_nominal": nom_cv,
                    "py_cp_nominal": nom_cp,
                })
    return records


def run_native_sweep():
    """Run native C++ probe in WSL."""
    wsl_script = """
import json, sys
import cantera as ct

FLUID_MAPPING = {
    'Water': 'water',
    'Nitrogen': 'nitrogen',
    'Oxygen': 'oxygen',
    'Hydrogen': 'hydrogen',
    'Methane': 'methane',
    'CarbonDioxide': 'carbon-dioxide',
    'Heptane': 'heptane',
    'HFC134a': 'HFC-134a'
}

BENCHMARK_STATES = """ + json.dumps(BENCHMARK_STATES) + """
STEP_FACTORS = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]

results = []
for fluid_name, states in BENCHMARK_STATES.items():
    ct_name = FLUID_MAPPING[fluid_name]
    for st in states:
        try:
            fl = ct.PureFluid('liquidvapor.yaml', ct_name)
            fl.TP = st['T'], st['P']
            nom_cv = float(fl.cv_mass)
            nom_cp = float(fl.cp_mass)
            v = 1.0 / fl.density
            T = fl.T
            P0 = fl.P
            
            for step in STEP_FACTORS:
                dt = step * T
                # cv finite difference
                fl.TD = T - dt, 1.0 / v
                s1 = fl.entropy_mass
                fl.TD = T + dt, 1.0 / v
                s2 = fl.entropy_mass
                nat_cv_fd = T * (s2 - s1) / (2.0 * dt)
                
                # cp finite difference
                fl.TP = T - dt, P0
                s1_p = fl.entropy_mass
                fl.TP = T + dt, P0
                s2_p = fl.entropy_mass
                nat_cp_fd = T * (s2_p - s1_p) / (2.0 * dt)
                
                fl.TP = T, P0
                results.append({
                    'fluid': fluid_name,
                    'state_name': st['name'],
                    'step_ratio': step,
                    'nat_cv_fd': nat_cv_fd,
                    'nat_cp_fd': nat_cp_fd,
                    'nat_cv_nominal': nom_cv,
                    'nat_cp_nominal': nom_cp
                })
        except Exception as e:
            for step in STEP_FACTORS:
                results.append({
                    'fluid': fluid_name,
                    'state_name': st['name'],
                    'step_ratio': step,
                    'error': str(e)
                })

print(json.dumps(results))
"""
    cmd = ["wsl", "/home/nebula/cantera-oracle-env/bin/python3", "-c", wsl_script]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip())


def main():
    print("Running Wave 3 cp/cv step-size sweep in pure-Python...")
    py_records = run_python_sweep()

    print("Running Wave 3 cp/cv step-size sweep against native oracle in WSL...")
    nat_records = run_native_sweep()

    # Index native records by (fluid, state_name, step_ratio)
    nat_map = {(r["fluid"], r["state_name"], r["step_ratio"]): r for r in nat_records}

    combined_rows = []
    step_stats = {s: {"cv_rel": [], "cp_rel": [], "cv_abs": [], "cp_abs": []} for s in STEP_FACTORS}

    for pr in py_records:
        key = (pr["fluid"], pr["state_name"], pr["step_ratio"])
        nr = nat_map.get(key, {})

        nat_cv = nr.get("nat_cv_fd")
        nat_cp = nr.get("nat_cp_fd")

        row = dict(pr)
        if nat_cv is not None and not math.isnan(nat_cv):
            row["nat_cv_fd"] = nat_cv
            row["nat_cp_fd"] = nat_cp
            row["nat_cv_nominal"] = nr.get("nat_cv_nominal")
            row["nat_cp_nominal"] = nr.get("nat_cp_nominal")

            cv_abs = abs(pr["py_cv_fd"] - nat_cv)
            cv_rel = cv_abs / max(abs(nat_cv), 1e-10)
            cp_abs = abs(pr["py_cp_fd"] - nat_cp)
            cp_rel = cp_abs / max(abs(nat_cp), 1e-10)

            row["cv_abs_err"] = cv_abs
            row["cv_rel_err"] = cv_rel
            row["cp_abs_err"] = cp_abs
            row["cp_rel_err"] = cp_rel

            s = pr["step_ratio"]
            step_stats[s]["cv_rel"].append(cv_rel)
            step_stats[s]["cp_rel"].append(cp_rel)
            step_stats[s]["cv_abs"].append(cv_abs)
            step_stats[s]["cp_abs"].append(cp_abs)

        combined_rows.append(row)

    # Write jsonl
    out_jsonl = AUDIT / "wave3_cp_cv_step_sweep.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in combined_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(combined_rows)} sweep records -> {out_jsonl.name}")

    # This is a measurement campaign using manually parameterized entropy
    # differences; it is not a parameterized invocation of production Sub::cp/cv.
    summary_lines = [
        "# Wave 3 `cp` / `cv` Step Sweep: Measurement Scope",
        "",
        f"Pinned source commit: `726522be4e2a13454d8415b7ef799d621f665cf3`. Records: {len(combined_rows)}.",
        "",
        "This campaign manually evaluates entropy differences at several step ratios. Its native evaluator uses centered `T-dT` / `T+dT` samples; the pinned `Sub.cpp` production methods use clipped and saturation-aware endpoints. The campaign therefore measures cross-engine property evaluation only and does not reproduce the native production `Sub::cp` / `Sub::cv` algorithm.",
        "",
        "The source formula audit separately maps native `Sub.cpp` and Python `tpx.py`, including nominal `dt = 1e-4*T`, entropy differences, held constraints, clipping, and restoration. This sweep does not make the production methods identical.",
        "",
        "",
        "## Measured relative differences by step size",
        "",
        "| `dt/T` | property | p50 | p95 | p99 | max | records |",
        "|---:|:---|---:|---:|---:|---:|---:|",
    ]

    for s in STEP_FACTORS:
        st = step_stats[s]
        for prop in ("cv", "cp"):
            values = st[f"{prop}_rel"]
            summary_lines.append(
                f"| `{s:.1e}` | `{prop}` | {percentile(values, .50):.3e} | {percentile(values, .95):.3e} | "
                f"{percentile(values, .99):.3e} | {max(values):.3e} | {len(values)} |"
            )

    summary_lines += [
        "",
        "## Causal conclusion",
        "",
        "These data do not isolate the cause of the saved `cp` / `cv` differences in the separate 9,608-case production comparison. Finite-difference truncation, roundoff, phase-boundary sensitivity, state-reset behavior, and implementation-path differences remain possible contributors; this experiment does not distinguish them.",
        "",
        "**Disposition: `CAUSE_NOT_FULLY_ESTABLISHED`.**",
        "",
    ]

    out_md = AUDIT / "wave3_cp_cv_step_sweep_summary.md"
    out_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    measured = {
        "schema": "wave3-cp-cv-measurement-only-v1",
        "status": "RUN_COMPLETE_MEASUREMENT_ONLY",
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "method_scope": "manual entropy-difference estimators; not production Sub::cp/cv endpoint parity",
        "requested_relative_steps": STEP_FACTORS,
        "record_count": len(combined_rows),
        "relative_error_distributions": {
            f"{s:.1e}": {
                prop: {"p50": percentile(step_stats[s][f"{prop}_rel"], .50),
                       "p95": percentile(step_stats[s][f"{prop}_rel"], .95),
                       "p99": percentile(step_stats[s][f"{prop}_rel"], .99),
                       "max": max(step_stats[s][f"{prop}_rel"], default=None)}
                for prop in ("cv", "cp")
            } for s in STEP_FACTORS
        },
        "cause_conclusion": "CAUSE_NOT_FULLY_ESTABLISHED",
        "sweep_records": "audit/wave3_cp_cv_step_sweep.jsonl",
        "summary_report": "audit/wave3_cp_cv_step_sweep_summary.md",
        "limitations": [
            "Native evaluation uses centered finite differences rather than Sub.cpp saturation-aware endpoints.",
            "This sweep does not attribute production 9,608-case cp/cv discrepancies.",
        ],
    }
    (AUDIT / "wave3_cp_cv_perturbation_results.json").write_text(
        json.dumps(measured, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote summary -> {out_md.name}")


if __name__ == "__main__":
    main()
