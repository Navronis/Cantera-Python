"""Compare saved pinned-native and standalone-Python DustyGas probe records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "audit" / "dusty_transport_native.json"
PYTHON = ROOT / "audit" / "dusty_transport_python.json"
OUTPUT = ROOT / "audit" / "dusty_transport_comparison.json"
ORACLE_IDENTITY = ROOT / "audit" / "native_oracle_identity.json"
RTOL = 1e-10
ATOL = 1e-30


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _errors(native, python) -> dict:
    a = np.asarray(native, dtype=float)
    b = np.asarray(python, dtype=float)
    if a.shape != b.shape:
        return {"shape_match": False, "native_shape": list(a.shape),
                "python_shape": list(b.shape), "within_tolerance": False}
    abs_error = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    rel_error = np.divide(abs_error, scale, out=np.zeros_like(abs_error), where=scale != 0)
    within = np.isclose(a, b, rtol=RTOL, atol=ATOL)
    return {
        "shape_match": True,
        "elements": int(a.size),
        "mismatches": int(np.count_nonzero(~within)),
        "max_absolute_error": float(np.max(abs_error, initial=0.0)),
        "max_relative_error": float(np.max(rel_error, initial=0.0)),
        "rtol": RTOL,
        "atol": ATOL,
        "within_tolerance": bool(np.all(within)),
    }


def main() -> None:
    native = json.loads(NATIVE.read_text(encoding="utf-8"))
    python = json.loads(PYTHON.read_text(encoding="utf-8"))
    oracle_identity = json.loads(ORACLE_IDENTITY.read_text(encoding="utf-8"))
    native_cases = {item["case_id"]: item for item in native["cases"]}
    python_cases = {item["case_id"]: item for item in python["cases"]}
    ids_match = set(native_cases) == set(python_cases)
    results = []
    for case_id in sorted(set(native_cases) & set(python_cases)):
        a, b = native_cases[case_id], python_cases[case_id]
        state_keys = ("T", "P", "X", "porosity", "tortuosity",
                      "mean_pore_radius", "mean_particle_diameter", "permeability")
        state_match = all(a[key] == b[key] for key in state_keys)
        results.append({
            "case_id": case_id,
            "state_match": state_match,
            "species_names_match": a["species_names"] == b["species_names"],
            "multi_diff_coeffs": _errors(a["multi_diff_coeffs"], b["multi_diff_coeffs"]),
            "thermal_conductivity": _errors(
                [a["thermal_conductivity"]], [b["thermal_conductivity"]]),
            "molar_fluxes": _errors(a["molar_fluxes"], b["molar_fluxes"]),
        })
    pinned_oracle_matches = (
        oracle_identity.get("exact_commit") == "726522be4e2a13454d8415b7ef799d621f665cf3"
        and oracle_identity.get("NATIVE_ORACLE_IDENTITY_VERIFIED") is True
        and native["package_version"] == oracle_identity.get("cantera_version")
    )
    passed = pinned_oracle_matches and ids_match and all(
        item["state_match"] and item["species_names_match"]
        and item["multi_diff_coeffs"]["within_tolerance"]
        and item["thermal_conductivity"]["within_tolerance"]
        and item["molar_fluxes"]["within_tolerance"]
        for item in results
    )
    report = {
        "schema_version": 1,
        "pinned_native_version": native["package_version"],
        "pinned_native_commit": oracle_identity.get("exact_commit"),
        "pinned_oracle_identity_verified": pinned_oracle_matches,
        "native_extension_sha256": oracle_identity.get("native_extension_sha256"),
        "native_input_sha256": _sha256(NATIVE),
        "python_input_sha256": _sha256(PYTHON),
        "python_transport_source_sha256": _sha256(ROOT / "cantera_python" / "transport.py"),
        "probe_source_sha256": _sha256(ROOT / "tools" / "run_dusty_transport_probe.py"),
        "mechanism_sha256_match": native["mechanism_sha256"] == python["mechanism_sha256"],
        "case_ids_match": ids_match,
        "case_count": len(results),
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "results": results,
        "passed": passed,
        "scope_note": (
            "Four fixed states compare DustyGas multi-diffusion matrices and delegated "
            "thermal conductivity. This is targeted differential evidence, not full "
            "DustyGas API or domain validation."
        ),
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "case_count": len(results),
        "matrix_mismatch_counts": {
            item["case_id"]: item["multi_diff_coeffs"].get("mismatches")
            for item in results
        },
        "matrix_max_relative_errors": {
            item["case_id"]: item["multi_diff_coeffs"].get("max_relative_error")
            for item in results
        },
        "conductivity_mismatch_counts": {
            item["case_id"]: item["thermal_conductivity"].get("mismatches")
            for item in results
        },
        "molar_flux_mismatch_counts": {
            item["case_id"]: item["molar_fluxes"].get("mismatches")
            for item in results
        },
        "output": str(OUTPUT.relative_to(ROOT)),
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
