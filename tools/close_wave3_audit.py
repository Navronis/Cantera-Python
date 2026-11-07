"""Generate conservative Wave 3 closure artifacts.

This tool deliberately preserves unresolved rows as unresolved.  It never
promotes a smoke result to source equivalence without a source-backed test.
"""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
PROV = ROOT / "provenance"
SRC = ROOT.parent / "cantera-original" / "src" / "tpx"

def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows), encoding="utf-8")

def pair_failures():
    rows = read_jsonl(AUDIT / "solver_wave3_property_pair_matrix.jsonl")
    failures = [r for r in rows if r.get("status") != "PASS"]
    out = []
    for r in failures:
        out.append({
            "fluid": r["fluid"], "pair": r["pair"],
            "forward_source_state": None, "target_property_1": None,
            "target_property_2": None, "initial_inverse_guess": None,
            "failure_location": r.get("python_path"), "solver_path": "source_set_xy_translation",
            "bracket": None, "iteration_count": None, "residual": None,
            "derivative": None, "exception_or_fallback": None,
            "physical_validity": "UNRECORDED_IN_LEGACY_SMOKE_ARTIFACT",
            "source_model_validity_domain": "REQUIRES_CASE_REPLAY",
            "pinned_source_should_accept": "UNDETERMINED",
            "classification": "UNRESOLVED_REQUIRES_REPLAY",
            "status": "IMPLEMENTATION_COMPLETE_FALSE",
            "evidence": "Legacy matrix contains no state, targets, or source outcome; preserving the gap is required."
        })
    write_jsonl(AUDIT / "solver_wave3_property_pair_failures.jsonl", out)
    return out

def source_pair_matrix():
    source = (SRC / "Sub.cpp").read_text(encoding="utf-8")
    section = source[source.index("void Substance::Set(PropertyPair::type"):source.index("//------------------ Protected")]
    pairs = re.findall(r"case PropertyPair::(\w+):", section)
    routes = {
        "TV":"direct T,v", "HP":"set_xy(H,P)", "SP":"set_xy(S,P)",
        "PV":"set_xy(P,V)", "TP":"phase select then set_xy(T,P)",
        "UV":"set_xy(U,V)", "ST":"set_xy(S,T)", "SV":"set_xy(S,V)",
        "UP":"set_xy(U,P)", "VH":"set_xy(V,H)", "TH":"set_xy(T,H)",
        "SH":"set_xy(S,H)", "PX":"Tsat(P), quality lever", "TX":"quality lever at T"
    }
    rows = []
    for pair in dict.fromkeys(pairs):
        rows.append({"pair": pair, "source_switch": True, "source_route": routes.get(pair, "source switch"),
                     "python_route": "Substance.Set/_set_xy_source", "status": "IMPLEMENTED_SOURCE_MAPPED"})
    write_jsonl(AUDIT / "solver_wave3_property_pair_source_mapping.jsonl", rows)
    return rows

def fallback_audit():
    files = [ROOT / "cantera_python" / "tpx.py", ROOT / "cantera_python" / "tpx_exact.py"]
    rows = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "fallback" in line.lower() or "underflow" in line.lower():
                rows.append({"file": str(path.relative_to(ROOT)), "line": i, "text": line.strip(),
                             "classification": "REQUIRES_SOURCE_REPLAY",
                             "status": "UNRESOLVED_IMPLEMENTATION_GAP"})
    write_jsonl(AUDIT / "solver_wave3_fallback_audit.jsonl", rows)
    return rows

def derivative_inventory():
    rows = []
    for path in sorted(SRC.glob("*.cpp")) + sorted(SRC.glob("*.h")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if any(tok in line for tok in ("dPsat", "dP", "Cp", "cv", "deriv", "Newton", "set_xy")):
                rows.append({"source_file": str(path.relative_to(SRC)), "line": i,
                             "source_expression": line.strip(), "python_implementation": "tpx.py/tpx_exact.py",
                             "consumer": "requires classification", "analytic_or_numeric": "UNCLASSIFIED",
                             "validation_status": "REQUIRES_INDEPENDENT_CAMPAIGN"})
    write_jsonl(PROV / "solver_wave3_derivative_inventory.jsonl", rows)
    return rows

def main():
    failures = pair_failures()
    mapping = source_pair_matrix()
    fallbacks = fallback_audit()
    derivatives = derivative_inventory()
    summary = json.loads((AUDIT / "solver_wave3_summary.json").read_text(encoding="utf-8"))
    pair_verification_path = AUDIT / "solver_wave3_property_pair_campaign_verification.json"
    pair_verification = (
        json.loads(pair_verification_path.read_text(encoding="utf-8"))
        if pair_verification_path.exists() else None
    )
    open_items = set(summary.get("open_items", []))
    open_items.discard("randomized 100-case-per-pair campaign not yet run")
    if pair_verification:
        statuses = pair_verification["status_counts"]
        summary.update({
            "property_pair_campaign_cases": pair_verification["trials"],
            "property_pair_campaign_successes": pair_verification[
                "successful_target_pair_residual_matches"],
            "property_pair_campaign_unexpected_failures": statuses.get(
                "INVERSION_FAILURE", 0),
            "property_pair_campaign_full_state_matches": pair_verification[
                "successful_full_state_matches"],
            "property_pair_campaign_strategy_coverage_complete": pair_verification[
                "strategy_coverage_complete"],
            "property_pair_campaign_native_parity": pair_verification[
                "native_parity"],
            "property_pair_campaign_verification": pair_verification_path.relative_to(
                ROOT).as_posix(),
        })
        open_items.add(
            "same-start pinned-native comparison required for property-pair Newton failures and alternate full states"
        )
    summary.update({
        "implementation": "SOURCE_TRANSLATED_WITH_CLOSURE_GAPS",
        "SOLVER_WAVE_3_IMPLEMENTATION_COMPLETE": False,
        "property_pair_failures_recorded": len(failures),
        "source_property_pair_cases_mapped": len(mapping),
        "fallback_records": len(fallbacks),
        "derivative_inventory_records": len(derivatives),
        "open_items": sorted(open_items | {
            "property-pair legacy cases lack replayable states and source outcomes",
            "fallback source equivalence not independently proven",
            "derivative inventory requires equation-level classification",
            "10,000-state adversarial campaign not yet run",
            "full upstream TPX replay not yet run"
        })
    })
    (AUDIT / "solver_wave3_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
