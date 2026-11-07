#!/usr/bin/env python3
"""Body-level review of the 71 legacy claims with unique Python name matches."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from revalidate_equivalence_claims import NATIVE, PROVENANCE, PYTHON, ROOT, digest, source_fragment


INPUT = PROVENANCE / "revalidation_421.jsonl"
OUTPUT = PROVENANCE / "revalidation_71_body_review.jsonl"
SUMMARY = PROVENANCE / "revalidation_71_body_review_summary.json"


def free_definition(symbol: str) -> list[dict]:
    """Resolve ordinary free-function bodies in C++ translation units."""
    found: list[dict] = []
    pattern = re.compile(
        r"^\s*(?!return\b)(?:[\w:<>,*&]+\s+)+" + re.escape(symbol) + r"\s*\("
    )
    for path in sorted(NATIVE.rglob("*.cpp")):
        if any(part in {"ext", "build"} for part in path.parts):
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not pattern.search(line):
                continue
            body, form, line_range = source_fragment(path, number)
            if form == "definition":
                found.append({
                    "source_file": path.relative_to(NATIVE).as_posix(),
                    "line_range": line_range,
                    "body_sha256": digest(body),
                })
    return found


def dimensions(default: str = "MATCH") -> dict:
    return {
        "equations": default,
        "branches": default,
        "units": "NOT_APPLICABLE",
        "state": "NOT_APPLICABLE",
        "exceptions": "MATCH",
        "mutation": "NOT_APPLICABLE",
    }


def classify(row: dict) -> tuple[str, dict, list[str], str, str]:
    ident = row["id"]
    source_file = row["source_file"]
    symbol = row["source_symbol"]
    cls = row.get("source_class")

    source_path = NATIVE / source_file
    source_line_number = int(ident.split(":")[-2])
    source_line = source_path.read_text(encoding="utf-8", errors="replace").splitlines()[source_line_number - 1]

    # The legacy regex treated calls and return expressions as declarations,
    # including calls inside inline methods in headers.
    call_expression = bool(re.match(r"^\s*return\s+", source_line))
    if call_expression:
        dims = dimensions("NOT_APPLICABLE")
        return (
            "INVENTORY_FALSE_POSITIVE",
            dims,
            ["Exclude invocation expressions from the source-symbol inventory."],
            "The inventoried line is a call site, not a C++ symbol declaration or definition.",
            "Add an inventory-parser regression asserting this source line is not emitted as a symbol.",
        )

    wrong_targets = {
        "include/cantera/kinetics/Arrhenius.h:121:order": (
            "ArrheniusBase::order returns reaction-rate order; the mapped Func1.order returns expression precedence.",
            "Add Arrhenius rate-order storage/update tests tied to UnitStack dimensions.",
        ),
        "include/cantera/kinetics/StoichManager.h:650:order": (
            "StoichManagerN::order reports stoichiometric order; the mapped Func1.order reports expression precedence.",
            "Add stoichiometric-order tests for integer and nonreactant-order pathways.",
        ),
        "include/cantera/thermo/IdealGasPhase.h:407:internalPressure": (
            "IdealGasPhase::internalPressure was mapped to StoichiometricSubstance.internal_pressure; IdealGasPhase lacks this API.",
            "Add an IdealGasPhase internal-pressure API test over several valid gas states.",
        ),
    }
    if ident in wrong_targets:
        note, test = wrong_targets[ident]
        return "WRONG_PYTHON_TARGET", dimensions("DIFFERENT"), [note], note, test

    if cls == "UnitSystem" and symbol == "convert":
        line = int(ident.split(":")[-2])
        dims = dimensions("DIFFERENT")
        dims["units"] = "PARTIAL" if line == 210 else "DIFFERENT"
        if line == 210:
            note = (
                "The explicit scalar string-unit equation matches, but the Python parser supports only a subset "
                "of pinned unit syntax and raises different exception types."
            )
            outcome = "FORMULA_MATCH_RESTRICTED_UNIT_PARSER"
        elif line == 211:
            note = "The C++ overload accepts Units objects; Python convert reparses strings and rejects Units objects."
            outcome = "MISSING_OVERLOAD_SEMANTICS"
        elif line in {227, 228}:
            note = "The C++ AnyValue overload parses dimensioned strings or applies UnitSystem defaults; no Python equivalent exists."
            outcome = "MISSING_OVERLOAD_SEMANTICS"
        else:
            note = "The C++ vector<AnyValue> overload converts each entry with UnitSystem defaults; Python convert is scalar-only."
            outcome = "MISSING_OVERLOAD_SEMANTICS"
        dims["exceptions"] = "DIFFERENT"
        dims["state"] = "DIFFERENT"
        return outcome, dims, [note], note, "Add scalar, Units-object, AnyValue/default-unit, and vector conversion parity tests."

    if ident == "include/cantera/base/Units.h:261:convertActivationEnergy":
        dims = dimensions("DIFFERENT")
        dims["units"] = "DIFFERENT"
        dims["state"] = "DIFFERENT"
        note = (
            "This row is the AnyValue-to-default-activation-energy overload. The mapped Python function is the "
            "three-argument explicit source/destination conversion and has no UnitSystem default state."
        )
        return "MISSING_OVERLOAD_SEMANTICS", dims, [note], note, "Test numeric and dimensioned-string activation energies under non-SI defaults."

    if ident == "include/cantera/kinetics/BlowersMaselRate.h:148:activationEnergy":
        dims = dimensions()
        dims["state"] = "DIFFERENT"
        dims["mutation"] = "DIFFERENT"
        note = "The three-branch equation matches, but Python requires delta_enthalpy as an argument while C++ reads stored state."
        return "FORMULA_MATCH_STATE_API_DIFFERENCE", dims, [note], note, "Compare branch boundaries and state-updated activation energy against the pinned implementation."

    if ident == "include/cantera/kinetics/Custom.h:65:setRateFunction":
        dims = dimensions()
        dims["state"] = "PARTIAL"
        dims["mutation"] = "PARTIAL"
        dims["exceptions"] = "DIFFERENT"
        note = "Python stores a callable or eval method but does not reproduce C++ shared ownership or the m_valid state flag."
        return "PARTIAL_STATE_AND_TYPE_SEMANTICS", dims, [note], note, "Test replacement, lifetime, validation state, null input, and exception propagation."

    if ident == "include/cantera/numerics/Func1.h:357:setMethod":
        dims = dimensions()
        dims["exceptions"] = "DIFFERENT"
        note = "The linear/previous branches and mutation match; invalid input raises ValueError instead of Cantera NotImplementedError."
        return "BODY_MATCH_EXCEPTION_DIFFERENCE", dims, [note], note, "Test both modes and invalid-method exception compatibility."

    if ident == "include/cantera/thermo/IdealGasPhase.h:369:setState_DP":
        dims = dimensions()
        dims["units"] = "MATCH"
        dims["state"] = "PARTIAL"
        dims["exceptions"] = "DIFFERENT"
        dims["mutation"] = "DIFFERENT"
        note = (
            "T=P*W/(R*rho) matches. Python adds finite/positive-density guards and returns an immutable state; C++ mutates "
            "density then temperature and only checks pressure in this body."
        )
        return "FORMULA_MATCH_STATE_AND_EXCEPTION_DIFFERENCE", dims, [note], note, "Test valid states plus zero, negative, NaN, and infinity with post-error state checks."

    # The remaining ideal-gas getters reproduce the equations through the state
    # object's concrete properties. Python additionally returns a tuple when a
    # caller does not provide an output buffer.
    if cls == "IdealGasPhase":
        dims = dimensions()
        dims["units"] = "MATCH"
        dims["state"] = "MATCH"
        dims["mutation"] = "MATCH" if symbol.startswith("get") else "NOT_APPLICABLE"
        note = "Equation and valid-state output-buffer behavior match on body inspection; Python also permits tuple return."
        return "BODY_MATCH_CANDIDATE_UNPROVEN", dims, [note], note, "Add symbol-linked pinned boundary and randomized composition tests."

    if cls == "Units" and symbol == "convertible":
        dims = dimensions()
        dims["units"] = "MATCH"
        note = "Both bodies compare all six base-dimension exponents for exact equality."
        return "BODY_MATCH_CANDIDATE_UNPROVEN", dims, [note], note, "Test fractional exponents and pressure/energy-derived dimensions against the pinned symbol."

    if cls in {"Func1", "Sum1", "Diff1", "Product1", "TimesConstant1", "PlusConstant1", "Ratio1", "Composite1"}:
        dims = dimensions()
        note = "Return expression and simplification branches match on body inspection for valid finite inputs."
        if symbol.startswith("new"):
            dims["exceptions"] = "DIFFERENT"
            note += " Exception classes and floating-point exceptional behavior remain Python-specific."
            outcome = "BODY_MATCH_EXCEPTION_DIFFERENCE"
        else:
            outcome = "BODY_MATCH_CANDIDATE_UNPROVEN"
        return outcome, dims, [note], note, "Add pinned identity, simplification-tree, aliasing, and exceptional-value tests."

    raise RuntimeError(f"No explicit body-review decision for {ident}")


def main() -> None:
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row["python_mapping_status"] == "RESOLVED_UNIQUE_NAME"]
    if len(rows) != 71:
        raise RuntimeError(f"Expected 71 unique-name mappings, found {len(rows)}")

    reviewed: list[dict] = []
    for row in rows:
        outcome, comparison, findings, rationale, test = classify(row)
        bodies = list(row["source_body_candidates"])
        if (
            outcome != "INVENTORY_FALSE_POSITIVE"
            and not bodies
            and row.get("source_class") in {None, "Func1", "Sum1", "Diff1", "TimesConstant1"}
        ):
            bodies = free_definition(row["source_symbol"])
        reviewed.append({
            "id": row["id"],
            "source_commit": row["source_commit"],
            "source_class": row.get("source_class"),
            "source_symbol": row["source_symbol"],
            "source_definition_candidates": bodies,
            "python_definition": row["python_candidates"][0],
            "body_review_outcome": outcome,
            "comparison": comparison,
            "findings": findings,
            "rationale": rationale,
            "recommended_test": test,
            "exact_pinned_symbol_linked_evidence": [],
            "equivalent_proven": False,
        })

    counts = Counter(row["body_review_outcome"] for row in reviewed)
    summary = {
        "schema_version": 1,
        "source_commit": rows[0]["source_commit"],
        "rows_reviewed": len(reviewed),
        "equivalent_proven_promoted": 0,
        "outcome_counts": dict(sorted(counts.items())),
        "finding": (
            "Unique name resolution is not semantic equivalence. Body review found false source symbols, wrong "
            "targets, missing overload semantics, and API/state differences. Rows whose valid-domain equations "
            "match remain unproven until symbol-linked exact-pinned evidence exists."
        ),
        "dependency_ordered_defects": [
            {
                "priority": 1,
                "area": "source inventory",
                "source_location": "tools/forensic_audit.py:23",
                "python_location": "tools/forensic_audit.py:304",
                "defect": (
                    "The method regex emits call expressions as symbols; "
                    f"{counts.get('INVENTORY_FALSE_POSITIVE', 0)} of these 71 rows are false positives."
                ),
                "recommended_test": "Assert representative return/call expressions in Units.cpp, Func1.cpp, and Func1.h are absent from the symbol inventory.",
            },
            {
                "priority": 2,
                "area": "unit-system semantics",
                "source_location": "include/cantera/base/Units.h:210",
                "python_location": "cantera_python/units.py:79",
                "defect": "UnitSystem default-state, Units-object, AnyValue, and vector conversion overloads are mapped to a scalar string conversion function.",
                "recommended_test": "Differentially test every overload, non-SI defaults, dimensioned strings, arrays, malformed input, and exception context.",
            },
            {
                "priority": 3,
                "area": "reaction-rate order",
                "source_location": "include/cantera/kinetics/Arrhenius.h:121",
                "python_location": "cantera_python/functions.py:109",
                "defect": "ArrheniusBase::order is falsely mapped to Func1 expression precedence; reaction-rate order is absent from the mapped class.",
                "recommended_test": "Verify rate order derived from UnitStack quantity dimensions and its updates after rate-unit changes.",
            },
            {
                "priority": 4,
                "area": "stoichiometric order",
                "source_location": "include/cantera/kinetics/StoichManager.h:650",
                "python_location": "cantera_python/functions.py:109",
                "defect": "StoichManagerN::order is also falsely mapped to Func1 expression precedence.",
                "recommended_test": "Verify integer, fractional, and nonreactant order handling through rates of progress.",
            },
            {
                "priority": 5,
                "area": "ideal-gas API",
                "source_location": "include/cantera/thermo/IdealGasPhase.h:407",
                "python_location": "cantera_python/stoich.py:98",
                "defect": "IdealGasPhase::internalPressure is mapped to another phase class and is absent on the ideal-gas implementation.",
                "recommended_test": "Assert zero internal pressure through both state and Solution ideal-gas APIs.",
            },
            {
                "priority": 6,
                "area": "state mutation and errors",
                "source_location": "include/cantera/thermo/IdealGasPhase.h:369",
                "python_location": "cantera_python/state.py:32",
                "defect": "setState_DP uses the same equation but differs in mutation atomicity and exceptional-input behavior.",
                "recommended_test": "Compare valid and invalid density/pressure cases, including the phase state after failure.",
            },
            {
                "priority": 7,
                "area": "stateful reaction rates",
                "source_location": "include/cantera/kinetics/BlowersMaselRate.h:148",
                "python_location": "cantera_python/rates.py:286",
                "defect": "Blowers-Masel activation energy takes delta H as a Python argument instead of reading state updated by kinetics.",
                "recommended_test": "Exercise all three branches while changing reaction enthalpy through the kinetics state-update path.",
            },
        ],
        "artifacts": {
            "input": INPUT.relative_to(ROOT).as_posix(),
            "rows": OUTPUT.relative_to(ROOT).as_posix(),
        },
    }
    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in reviewed), encoding="utf-8")
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
