"""Regenerate Wave 3 property-pair cases from current source/runtime.

Cases are generated from valid direct T-v or T-quality states.  Failures are
recorded with the current state and exception; no result is promoted to
source-equivalence by this tool.
"""
from __future__ import annotations
import json, math, random
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cantera_python.tpx import (Water, CarbonDioxide, Nitrogen, Oxygen,
    Hydrogen, Methane, Heptane, HFC134a, PropertyPair, PropertyFlag, TPXError)
OUT = ROOT / "audit" / "solver_wave3_property_pair_validation.jsonl"
N = 200
FLUIDS = [Water, CarbonDioxide, Nitrogen, Oxygen, Hydrogen, Methane, Heptane, HFC134a]
PAIRS = [("TV", PropertyPair.TV, PropertyFlag.T, PropertyFlag.V),
         ("HP", PropertyPair.HP, PropertyFlag.H, PropertyFlag.P),
         ("SP", PropertyPair.SP, PropertyFlag.S, PropertyFlag.P),
         ("PV", PropertyPair.PV, PropertyFlag.P, PropertyFlag.V),
         ("TP", PropertyPair.TP, PropertyFlag.T, PropertyFlag.P),
         ("UV", PropertyPair.UV, PropertyFlag.U, PropertyFlag.V),
         ("ST", PropertyPair.ST, PropertyFlag.S, PropertyFlag.T),
         ("SV", PropertyPair.SV, PropertyFlag.S, PropertyFlag.V),
         ("UP", PropertyPair.UP, PropertyFlag.U, PropertyFlag.P),
         ("VH", PropertyPair.VH, PropertyFlag.V, PropertyFlag.H),
         ("TH", PropertyPair.TH, PropertyFlag.T, PropertyFlag.H),
         ("SH", PropertyPair.SH, PropertyFlag.S, PropertyFlag.H),
         ("PX", PropertyPair.PX, PropertyFlag.P, None),
         ("TX", PropertyPair.TX, PropertyFlag.T, None)]

def scalar(obj, flag):
    return obj.vprop(flag)

def state(obj):
    vals = {}
    for name, fn in (("T", lambda: obj.T), ("P", obj.P), ("rho", lambda: obj.Rho),
                     ("v", obj.v), ("u", obj.u), ("h", obj.h), ("s", obj.s),
                     ("g", obj.g), ("x", obj.x), ("two_phase", obj.TwoPhase)):
        try: vals[name] = float(fn()) if name != "two_phase" else bool(fn())
        except Exception: vals[name] = None
    return vals

def make_state(cls, rng, quality=False):
    o = cls()
    lo, hi = o.Tmin(), min(o.Tmax(), o.Tcrit()*0.97)
    if hi <= lo: hi = o.Tmax()
    T = lo + (hi-lo)*(0.10 + 0.80*rng.random())
    # Quality states are only used for the explicit TX/PX pairs.  Other
    # pairs are generated from a single-phase source-domain state because a
    # saturated endpoint is intentionally rejected by pinned TP.
    if quality and T < o.Tcrit():
        o.Set(PropertyPair.TX, T, rng.random())
    else:
        v = o.Vcrit() * (1.0e4 + 2.0e4*rng.random())
        o.Set(PropertyPair.TV, T, v)
        if o.TwoPhase():
            # Move to a clearly dilute gas state if the correlation's
            # saturation pressure still places the first draw in two phase.
            o.Set(PropertyPair.TV, T, o.Vcrit() * 1.0e6)
    return o

def worker(job):
    seed, cls, pair = job
    rng = random.Random(726522 + seed * 1009)
    pname, code, f1, f2 = pair
    rows = []
    for case in range(N):
        base = None
        try:
            base = make_state(cls, rng, quality=(pname in ("TX", "PX")))
            original = state(base)
            if pname == "TX": targets = (original["T"], original["x"])
            elif pname == "PX": targets = (original["P"], original["x"])
            else: targets = (scalar(base, f1), scalar(base, f2))
            start = make_state(cls, rng)
            start_state = state(start)
            start.Set(code, *targets)
            solved = state(start)
            errs = {k: (abs(solved[k]-original[k]) if isinstance(solved[k], float) and isinstance(original[k], float) else None)
                    for k in ("T","P","rho","v","u","h","s","g")}
            # Target residuals use source tolerances in the setter. The
            # remaining thermodynamic fields use a conservative relative
            # full-state tolerance so dilute specific-volume roundoff is not
            # mistaken for a phase-branch error.
            ok = all(v is None or v <= 2e-4*max(1.0, abs(original[k])) for k,v in errs.items())
            rows.append({"fluid": cls.__name__, "pair": pname, "case": case,
                         "target": list(targets), "original": original,
                         "start": start_state, "solved": solved, "errors": errs,
                         "converged": bool(ok), "failure_type": None if ok else "FULL_STATE_MISMATCH"})
        except Exception as exc:
            rows.append({"fluid": cls.__name__, "pair": pname, "case": case,
                         "target": None, "original": state(base) if base else None,
                         "start": None, "solved": None, "errors": None,
                         "converged": False, "failure_type": type(exc).__name__,
                         "exception": str(exc)})
    return rows

def main():
    jobs = [(i, cls, pair) for i, cls in enumerate(FLUIDS) for pair in PAIRS]
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futures = [pool.submit(worker, job) for job in jobs]
        for future in as_completed(futures): rows.extend(future.result())
    rows.sort(key=lambda r: (r["fluid"], r["pair"], r["case"]))
    OUT.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    summary = {"fluids": len(FLUIDS), "pairs": len(PAIRS), "cases": len(rows),
               "successes": sum(r["converged"] for r in rows),
               "unexpected_failures": sum(not r["converged"] for r in rows),
               "cases_per_combination": N}
    (ROOT / "audit" / "solver_wave3_property_pair_summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
