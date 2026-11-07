"""Differential comparison: cantera_python vs pinned native oracle baseline.

Reads provenance/oracle_baseline_pinned.json (AUTHORITATIVE_ORACLE) and
compares the pure-Python implementation, appending results to
provenance/pinned_native_differential_results.jsonl.

Run from cantera-python/ with the oracle baseline present:
    python tools/oracle/compare_pinned.py
"""
import json
import hashlib
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

import cantera_python as ct  # noqa: E402

BASELINE = ROOT / 'provenance' / 'oracle_baseline_pinned.json'
RESULTS = ROOT / 'provenance' / 'pinned_native_differential_results.jsonl'
DATA = ROOT.parent / 'cantera-original' / 'data' / 'gri30.yaml'

GAS_CONSTANT = 8314.46261815324  # J/(kmol K), pinned Cantera value
PINNED_COMMIT = '726522be4e2a13454d8415b7ef799d621f665cf3'
BASELINE_SHA256 = '4ffc6b1d07add187e71940f1dcbc14af39328b26abee7990713f8fb208e3623e'


def rel_err(a, b):
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return abs(a - b)
    return abs(a - b) / scale


def record(suite, case, quantity, py, native, max_rel, note=''):
    return {
        "suite": suite, "case": case, "quantity": quantity,
        "python_value": py, "oracle_value": native,
        "abs_error": abs(py - native) if isinstance(py, float) else None,
        "rel_error": rel_err(py, native) if isinstance(py, float) else None,
        "max_rel_error_group": max_rel,
        "oracle_role": "AUTHORITATIVE_ORACLE",
        "note": note,
    }


def main():
    digest = hashlib.sha256(BASELINE.read_bytes()).hexdigest()
    if digest != BASELINE_SHA256:
        raise RuntimeError(
            f'pinned baseline hash mismatch: expected {BASELINE_SHA256}, got {digest}')
    base = json.loads(BASELINE.read_text())
    reported_commit = str(base.get('oracle_commit', ''))
    source_commit = subprocess.check_output(
        ['git', '-C', str(ROOT.parent / 'cantera-original'), 'rev-parse', 'HEAD'],
        text=True).strip()
    try:
        resolved_commit = subprocess.check_output(
            ['git', '-C', str(ROOT.parent / 'cantera-original'), 'rev-parse',
             '--verify', f'{reported_commit}^{{commit}}'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        resolved_commit = None
    if source_commit != PINNED_COMMIT or resolved_commit != PINNED_COMMIT:
        raise RuntimeError(
            f'oracle/source identity mismatch: oracle={reported_commit!r}, '
            f'source={source_commit!r}, expected={PINNED_COMMIT!r}')
    gas = ct.Solution(DATA)
    results = []
    failures = 0

    # --- species thermo -----------------------------------------------------
    thermo = base["species_thermo"]
    for name, rows in thermo.items():
        for row in rows:
            T = row["T"]
            sp = [s for s in gas.species if s.name == name][0]
            # Python species thermo is nondimensional: cp/R, h/RT, s/R
            props = sp.thermo.properties(T)
            cp_py = props[0] * GAS_CONSTANT
            h_py = props[1] * GAS_CONSTANT * T
            s_py = props[2] * GAS_CONSTANT
            for q, py, nat in (("cp", cp_py, row["cp"]),
                               ("h", h_py, row["h"]),
                               ("s", s_py, row["s"])):
                e = rel_err(py, nat)
                results.append(record("species_thermo", name, f"{q}@{T}K",
                                      py, nat, e))
                if e > 1e-10:
                    failures += 1

    # --- phase states -------------------------------------------------------
    for label, snap in base["states"].items():
        T = snap["T"]
        P = snap["P"]
        # rebuild composition from oracle density + mean MW is not unique;
        # instead compare scalar structure with states we can reproduce
        pass

    state_defs = {
        "state_1200_ch4": (1200.0, 101325.0, 'CH4:1, O2:2, N2:7.52'),
        "state_2000_dilute": (2000.0, 101325.0, 'O2:0.3, N2:0.7'),
    }
    for label, (T, P, X) in state_defs.items():
        snap = base["states"][label]
        gas.TPX = T, P, X

        scalars = {
            "T": gas.temperature, "P": gas.pressure,
            "density": gas.density,
            "mean_MW": gas.mean_molecular_weight,
            "cp_mass": gas.cp_mass,
            "enthalpy_mass": gas.enthalpy_mass,
            "entropy_mass": gas.entropy_mass,
        }
        for q, py in scalars.items():
            nat = snap[q]
            e = rel_err(py, nat)
            results.append(record("phase_state", label, q, py, nat, e))
            if e > 1e-10:
                failures += 1

        # standard concentration: oracle stores P/(R*T) fallback or direct
        sc_nat = snap["standard_concentration"]
        sc_py = gas.standard_concentration
        e = rel_err(sc_py, sc_nat)
        results.append(record("phase_state", label, "standard_concentration",
                              sc_py, sc_nat, e))
        if e > 1e-10:
            failures += 1

        # vectors
        vectors = {
            "partial_molar_enthalpies": gas.partial_molar_enthalpies,
            "forward_rate_constants": gas.forward_rate_constants,
            "reverse_rate_constants": gas.reverse_rate_constants,
            "equilibrium_constants": gas.equilibrium_constants,
            "forward_rates_of_progress": gas.forward_rates_of_progress,
            "reverse_rates_of_progress": gas.reverse_rates_of_progress,
            "net_production_rates": gas.net_production_rates,
            "creation_rates": gas.creation_rates,
            "destruction_rates": gas.destruction_rates,
        }
        for q, pyvec in vectors.items():
            natvec = snap[q]
            pyvec = list(pyvec)
            if len(pyvec) != len(natvec):
                results.append(record("phase_state", label, q, f"len {len(pyvec)}",
                                      f"len {len(natvec)}", 1.0))
                failures += 1
                continue
            worst = max(rel_err(a, b) for a, b in zip(pyvec, natvec))
            results.append(record("phase_state", label, q, "vector", "vector",
                                  worst, note=f"max_rel={worst:.3e}"))
            if worst > 1e-8:
                failures += 1

        # standard Gibbs nondimensional
        mu0_py = [g / (GAS_CONSTANT * T)
                  for g in gas.standard_chemical_potentials]
        nat = snap["standard_gibbs_RT"]
        worst = max(rel_err(a, b) for a, b in zip(mu0_py, nat))
        results.append(record("phase_state", label, "standard_gibbs_RT",
                              "vector", "vector", worst,
                              note=f"max_rel={worst:.3e}"))
        if worst > 1e-10:
            failures += 1

    # --- falloff sample -----------------------------------------------------
    gas.TPX = 1500.0, 101325.0, 'CH4:0.1, O2:0.1, N2:0.8'
    kf = gas.forward_rate_constants
    for i, fsnap in base["falloff_sample"].items():
        idx = int(i)
        e = rel_err(kf[idx], fsnap["kf"])
        results.append(record("falloff", f"rxn_{i}_{fsnap['type']}", "kf",
                              kf[idx], fsnap["kf"], e))
        if e > 1e-8:
            failures += 1

    summary = {
        "oracle_role": "AUTHORITATIVE_ORACLE",
        "oracle_commit_reported": reported_commit,
        "oracle_commit_resolved": PINNED_COMMIT,
        "oracle_fixture_sha256": digest,
        "cases": len(results),
        "failures_over_tolerance": failures,
        "tolerance_policy": {
            "scalars_and_thermo": 1e-10,
            "rate_vectors": 1e-8,
        },
    }
    # A comparison artifact represents one run. Atomic replacement prevents
    # historical failed and successful runs from being mixed in one JSONL file.
    fd, temp_name = tempfile.mkstemp(
        dir=RESULTS.parent, prefix=RESULTS.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            for r in results:
                fh.write(json.dumps(r) + '\n')
            fh.write(json.dumps({"summary": summary}) + '\n')
        os.replace(temp_name, RESULTS)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
