#!/usr/bin/env python3
"""Derive Wave 2 return-code and root-selection accounting from raw probe rows."""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_root(roots, volume):
    valid = [(i, float(root)) for i, root in enumerate(roots)
             if math.isfinite(float(root)) and float(root) > 0.0]
    if not valid:
        return None
    return min(valid, key=lambda pair: abs(pair[1] - volume))[0]


def main():
    identity_path = AUDIT / "waves_1_3_repository_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else {}
    native_path = AUDIT / "raw_native_wave2_internal.jsonl"
    python_path = AUDIT / "raw_python_wave2_internal.jsonl"
    native, python = read_jsonl(native_path), read_jsonl(python_path)
    nm, pm = {r["case_id"]: r for r in native}, {r["case_id"]: r for r in python}
    if len(nm) != len(native) or len(pm) != len(python) or nm.keys() != pm.keys():
        raise SystemExit("Raw native/Python Wave 2 case IDs are not unique and identical")

    distributions = {}
    outcomes = []
    mismatch_codes = 0
    mismatch_counts = 0
    mismatch_roots = 0
    mismatch_selected = 0
    mismatch_order = 0
    max_root_relative_error = 0.0
    root_tolerance = {"rtol": 1e-6, "atol": 1e-7, "basis": "absolute error <= atol + rtol * abs(native root)"}
    for label, records in (("native", nm), ("python", pm)):
        codes = collections.Counter(int(r["nsol"]) for r in records.values() if r.get("status") == "SUCCESS")
        single = collections.Counter(str(int(r["nsol"])) for r in records.values()
                                     if r.get("status") == "SUCCESS" and abs(int(r["nsol"])) == 1)
        multi = [r for r in records.values() if r.get("status") == "SUCCESS" and abs(int(r["nsol"])) == 3]
        selected = collections.Counter()
        for row in multi:
            index = selected_root(row["vroots"], float(row["molar_volume"]))
            selected[str(index)] += 1
        distributions[label] = {
            "successful_records": sum(codes.values()),
            "return_code_counts": {str(k): codes[k] for k in sorted(codes)},
            "single_root_return_code_counts": {k: single[k] for k in sorted(single)},
            "three_root_selected_index_counts": {k: selected[k] for k in ("0", "1", "2")},
            "three_root_selected_volume_class_counts": {
                "lowest_volume": selected["0"],
                "middle_volume": selected["1"],
                "highest_volume": selected["2"],
            },
            "source_defined_physical_branch_counts": {
                "single_root_gas_default": codes[1],
                "single_root_liquid_or_dense_supercritical": codes[-1],
                "three_root_selected_liquid_lowest_volume": selected["0"],
                "three_root_selected_middle_volume": selected["1"],
                "three_root_selected_vapor_highest_volume": selected["2"],
                "combined_gas_or_vapor": codes[1] + selected["2"],
                "combined_liquid_or_dense": codes[-1] + selected["0"],
            },
        }

    for cid in sorted(nm):
        n, p = nm[cid], pm[cid]
        if n.get("status") != p.get("status"):
            outcomes.append({"case_id": cid, "status_match": False, "reason": "status mismatch"})
            continue
        if n.get("status") != "SUCCESS":
            outcomes.append({"case_id": cid, "status_match": True, "status": "FAILURE"})
            continue
        nr, pr = int(n["nsol"]), int(p["nsol"])
        code_match = nr == pr
        mismatch_codes += not code_match
        n_roots = abs(nr) if abs(nr) in (1, 2, 3) else 0
        p_roots = abs(pr) if abs(pr) in (1, 2, 3) else 0
        count_match = n_roots == p_roots
        mismatch_counts += not count_match
        nvalid = [float(x) for x in n["vroots"][:n_roots]]
        pvalid = [float(x) for x in p["vroots"][:p_roots]]
        errors = []
        root_match = count_match and len(nvalid) == len(pvalid)
        if root_match:
            for a, b in zip(nvalid, pvalid):
                err = abs(a - b)
                rel = err / max(abs(a), 1e-300)
                max_root_relative_error = max(max_root_relative_error, rel)
                errors.append({"absolute": err, "relative": rel})
                if err > root_tolerance["atol"] + root_tolerance["rtol"] * abs(a):
                    root_match = False
        mismatch_roots += not root_match
        n_ordered = sorted(nvalid)
        p_ordered = sorted(pvalid)
        order_match = (nvalid == n_ordered and pvalid == p_ordered)
        mismatch_order += not order_match
        nselected = selected_root(n["vroots"], float(n["molar_volume"])) if n_roots == 3 else None
        pselected = selected_root(p["vroots"], float(p["molar_volume"])) if p_roots == 3 else None
        selected_match = n_roots != 3 or (p_roots == 3 and nselected == pselected)
        mismatch_selected += not selected_match
        outcomes.append({
            "case_id": cid,
            "status_match": True,
            "status": "SUCCESS",
            "native_return_code": nr,
            "python_return_code": pr,
            "return_code_match": code_match,
            "native_root_count": n_roots,
            "python_root_count": p_roots,
            "root_count_match": count_match,
            "native_valid_roots": nvalid,
            "python_valid_roots": pvalid,
            "native_roots_by_volume": ({"lowest": n_ordered[0], "middle": n_ordered[1], "highest": n_ordered[2]} if n_roots == 3 else None),
            "python_roots_by_volume": ({"lowest": p_ordered[0], "middle": p_ordered[1], "highest": p_ordered[2]} if p_roots == 3 else None),
            "root_arrays_are_volume_ordered": order_match,
            "root_errors": errors,
            "roots_within_tolerance": root_match,
            "native_selected_root_index": nselected,
            "python_selected_root_index": pselected,
            "native_selected_volume": float(n["molar_volume"]),
            "python_selected_volume": float(p["molar_volume"]),
            "native_selected_density": float(n["density"]),
            "python_selected_density": float(p["density"]),
            "selected_volume_relative_error": abs(float(n["molar_volume"]) - float(p["molar_volume"])) / max(abs(float(n["molar_volume"])), 1e-300),
            "selected_density_relative_error": abs(float(n["density"]) - float(p["density"])) / max(abs(float(n["density"])), 1e-300),
            "three_root_case": n_roots == 3,
            "single_root_source_phase": ("liquid_or_dense_supercritical" if nr == -1 else "gas_default" if nr == 1 else None),
        })

    native_codes = distributions["native"]["return_code_counts"]
    python_codes = distributions["python"]["return_code_counts"]
    provenance_path = AUDIT / "waves_1_3_native_rerun_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else {}
    fresh_bound = bool(
        provenance.get("pinned_native_commit") == "726522be4e2a13454d8415b7ef799d621f665cf3"
        and provenance.get("waves_1_3_snapshot_id") == identity.get("waves_1_3_snapshot_id")
        and provenance.get("raw_result_sha256", {}).get(str(native_path.relative_to(ROOT))) == sha256(native_path)
        and provenance.get("raw_result_sha256", {}).get(str(python_path.relative_to(ROOT))) == sha256(python_path)
        and set(provenance.get("completed_runs", [])) >= {"wave2_internal_native", "wave2_internal_python"}
    )
    results = {
        "artifact": "wave2_return_code_distribution",
        "waves_1_3_snapshot_id": identity.get("waves_1_3_snapshot_id"),
        "data_status": "LIVE_PINNED_NATIVE_AND_PYTHON_RERUN_HASH_BOUND" if fresh_bound else "SAVED_RAW_RECORDS_UNBOUND_TO_CURRENT_RUN",
        "rerun_provenance_sha256": sha256(provenance_path) if fresh_bound else None,
        "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "raw_inputs": {
            "native": {"path": str(native_path.relative_to(ROOT)), "sha256": sha256(native_path)},
            "python": {"path": str(python_path.relative_to(ROOT)), "sha256": sha256(python_path)},
        },
        "total_case_ids": len(nm),
        "successful_cases": sum(r.get("status") == "SUCCESS" for r in nm.values()),
        "failed_cases": sum(r.get("status") != "SUCCESS" for r in nm.values()),
        "native": distributions["native"],
        "python": distributions["python"],
        "return_code_count_dictionaries_match": native_codes == python_codes,
        "return_code_mismatches": mismatch_codes,
        "root_count_mismatches": mismatch_counts,
        "root_value_mismatch_cases": mismatch_roots,
        "selected_root_mismatches": mismatch_selected,
        "root_array_order_mismatches": mismatch_order,
        "root_tolerance": root_tolerance,
        "max_root_relative_error": max_root_relative_error,
        "single_root_semantics": "Use native signed return code: +1 gas-default; -1 liquid or dense supercritical per MixtureFugacityTP::solveCubic. Root index is not used as a phase label.",
        "three_root_semantics": "Index 0/1/2 describes lowest/middle/highest volume only after ordering the valid three roots; selected index is inferred from the returned molar volume.",
        "per_case_records": outcomes,
    }
    out = AUDIT / "wave2_return_code_distribution.json"
    out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}; cases={len(nm)}, codes={native_codes}, root mismatches={mismatch_roots}")
    if mismatch_codes or mismatch_counts or mismatch_roots or mismatch_selected or mismatch_order or native_codes != python_codes:
        raise SystemExit("Wave 2 raw-record parity check failed")


if __name__ == "__main__":
    main()
