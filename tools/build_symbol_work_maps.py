"""Build deterministic ownership and dependency maps for audited Cantera symbols.

The source inventory is lexical: it does not contain a native call graph. This tool
therefore records conservative dependencies between implementation capabilities
instead of fabricating symbol-to-symbol call edges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
SOURCE = PROVENANCE / "original_cantera_symbols.jsonl"
OWNERSHIP = PROVENANCE / "symbol_ownership.json"
DEPENDENCIES = PROVENANCE / "symbol_dependency_graph.json"

LANES = {
    "thermo_tpx_equilibrium": {
        "scope": ["thermo", "tpx", "equilibrium"],
        "purpose": "Thermodynamic models, real-fluid TPX models, and equilibrium algorithms.",
    },
    "kinetics_interfaces_transport": {
        "scope": ["kinetics", "transport"],
        "purpose": "Reaction kinetics, reacting interfaces, and transport models.",
    },
    "numerics_zeroD_oneD": {
        "scope": ["numerics", "zeroD", "oneD"],
        "purpose": "Numerical kernels, reactor ODE systems, and one-dimensional domains.",
    },
    "parser_api_provenance_tests": {
        "scope": ["foundation", "parser_api", "validation_tests"],
        "purpose": "Core utilities, parsers, language APIs, audit evidence, examples, and tests.",
    },
}

SUBSYSTEM_COMPONENT = {
    "base": "foundation",
    "thermo": "thermo",
    "tpx": "tpx",
    "equil": "equilibrium",
    "kinetics": "kinetics",
    "transport": "transport",
    "numerics": "numerics",
    "zeroD": "zeroD",
    "oneD": "oneD",
    "fortran": "parser_api",
    "cython": "parser_api",
    "Cabinet.h": "parser_api",
    "extensions": "parser_api",
}

COMPONENT_OWNER = {
    "foundation": "parser_api_provenance_tests",
    "parser_api": "parser_api_provenance_tests",
    "validation_tests": "parser_api_provenance_tests",
    "thermo": "thermo_tpx_equilibrium",
    "tpx": "thermo_tpx_equilibrium",
    "equilibrium": "thermo_tpx_equilibrium",
    "kinetics": "kinetics_interfaces_transport",
    "transport": "kinetics_interfaces_transport",
    "numerics": "numerics_zeroD_oneD",
    "zeroD": "numerics_zeroD_oneD",
    "oneD": "numerics_zeroD_oneD",
}

# These are implementation prerequisites, not a claim about individual C++ calls.
COMPONENT_DEPENDENCIES = {
    "foundation": [],
    "thermo": ["foundation"],
    "tpx": ["foundation", "thermo"],
    "kinetics": ["foundation", "thermo"],
    "transport": ["foundation", "thermo"],
    "numerics": ["foundation"],
    "equilibrium": ["foundation", "thermo", "numerics"],
    "zeroD": ["foundation", "thermo", "kinetics", "numerics"],
    "oneD": ["foundation", "thermo", "kinetics", "transport", "numerics"],
    "parser_api": ["foundation"],
    "validation_tests": ["foundation"],
}

COMPONENT_TESTS = {
    "foundation": ["tests/test_units.py", "tests/test_functions.py", "tests/test_composition.py", "tests/test_elements.py"],
    "thermo": ["tests/test_thermo.py", "tests/test_state.py", "tests/test_exact_pinned_benchmarks.py"],
    "tpx": ["tests/test_tpx.py", "tests/test_exact_pinned_benchmarks.py"],
    "equilibrium": ["tests/test_equilibrium_solver.py", "tests/test_equilibrium_modes.py", "tests/test_exact_pinned_benchmarks.py"],
    "kinetics": ["tests/test_rates.py", "tests/test_kinetics.py", "tests/test_exact_pinned_benchmarks.py"],
    "transport": ["tests/test_transport.py", "tests/test_exact_pinned_benchmarks.py"],
    "numerics": ["tests/test_numerics.py", "tests/test_exact_pinned_benchmarks.py"],
    "zeroD": ["tests/test_reactor_network.py", "tests/test_reactor_conservation.py", "tests/test_exact_pinned_benchmarks.py"],
    "oneD": ["tests/test_oned.py", "tests/test_exact_pinned_benchmarks.py"],
    "parser_api": ["tests/test_mechanism.py", "tests/test_solution_kinetics_reactor.py", "tests/test_runtime_purity.py"],
    "validation_tests": ["tests/test_upstream_parity.py", "tests/test_symbol_work_maps.py"],
}

COMPONENT_ORACLE = {
    "foundation": "Compare values, exception types, boundary behavior, and serialization semantics with the exact pinned native API.",
    "thermo": "Compare cp, h, s, g, chemical potentials, derivatives, and state inversion over a temperature-pressure-composition grid against the exact pinned native build.",
    "tpx": "Compare real-fluid states, saturation properties, phase boundaries, and failure behavior over the model-valid domain against the exact pinned native build.",
    "equilibrium": "Compare converged composition, thermodynamic invariants, residuals, iteration outcome, and failure behavior for each supported constraint against the exact pinned native build.",
    "kinetics": "Compare forward and reverse rate constants, rates of progress, production rates, derivatives, and error behavior over temperature-pressure-composition grids against the exact pinned native build.",
    "transport": "Compare mixture and multicomponent coefficients, mobilities, thermal diffusion terms, fluxes, and model errors against the exact pinned native build.",
    "numerics": "Run matched linear, nonlinear, integration, and failure-semantics benchmarks against the pinned native solver path with recorded tolerances and iteration outcomes.",
    "zeroD": "Compare reactor trajectories, conservation residuals, Jacobians, events, sensitivities, tolerances, and failure modes against the exact pinned native build.",
    "oneD": "Compare converged profiles, grid refinement, residuals, transport coupling, continuation behavior, and solver failures against the exact pinned native build.",
    "parser_api": "Compare parsing, round-trip serialization, factory selection, aliases, public signatures, return shapes, and exception behavior against the exact pinned native API.",
    "validation_tests": "Port the upstream test intent, run both implementations from identical inputs, and record matched assertions or an explicit justified exclusion.",
}


def _load_symbols() -> list[dict]:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf8").splitlines() if line]
    ids = [row["id"] for row in rows]
    if len(rows) != 5784:
        raise ValueError(f"Expected 5,784 audited symbols, found {len(rows):,}")
    if len(set(ids)) != len(ids):
        duplicates = [symbol_id for symbol_id, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate symbol IDs in source inventory: {duplicates[:5]}")
    return rows


def _component(row: dict) -> str:
    subsystem = row["subsystem"]
    if subsystem in SUBSYSTEM_COMPONENT:
        return SUBSYSTEM_COMPONENT[subsystem]
    source_file = row["source_file"]
    if subsystem == "other" and source_file.startswith(("doc/", "samples/", "test/", "test_problems/")):
        return "validation_tests"
    # Unknown and binding-layer locations are API/infrastructure work until a
    # body-level audit provides evidence for a scientific subsystem.
    return "parser_api"


def _destination(row: dict, component: str) -> str:
    """Infer a narrow Python destination without claiming that it is implemented."""
    haystack = f"{row['source_file']} {row.get('class') or ''} {row['symbol']}".lower()
    if component == "foundation":
        if "unit" in haystack:
            return "cantera_python/units.py"
        if "element" in haystack:
            return "cantera_python/elements.py"
        if any(key in haystack for key in ("func1", "poly1", "fourier", "gaussian")):
            return "cantera_python/functions.py"
        if any(key in haystack for key in ("anymap", "yaml", "inputfile")):
            return "cantera_python/mechanism.py"
        return "cantera_python/base.py"
    if component == "thermo":
        if "idealgas" in haystack:
            return "cantera_python/ideal_gas.py"
        if any(key in haystack for key in ("idealsolid", "lattice", "solid")):
            return "cantera_python/ideal_solid.py"
        if any(key in haystack for key in ("stoichsubstance", "fixedchem")):
            return "cantera_python/fixed_stoichiometry.py"
        if any(key in haystack for key in ("state", "phase.cpp", "phase.h")):
            return "cantera_python/state.py"
        return "cantera_python/thermo.py"
    if component == "tpx":
        return "cantera_python/tpx.py"
    if component == "equilibrium":
        return "cantera_python/equilibrium.py"
    if component == "kinetics":
        if any(key in haystack for key in ("interface", "edge", "surface", "sticking")):
            return "cantera_python/surface.py"
        if any(key in haystack for key in ("rate", "arrhenius", "falloff", "plog", "chebyshev")):
            return "cantera_python/rates.py"
        if "reaction" in haystack:
            return "cantera_python/reaction_equation.py"
        return "cantera_python/kinetics.py"
    if component == "transport":
        return "cantera_python/transport.py"
    if component == "numerics":
        return "cantera_python/numerics.py"
    if component == "zeroD":
        return "cantera_python/reactor.py"
    if component == "oneD":
        return "cantera_python/oned.py"
    if component == "validation_tests":
        stem = Path(row["source_file"]).stem.lower().replace("-", "_")
        return f"tests/upstream/test_{stem}.py"
    # Bindings and extensions route by their referenced public subsystem where
    # possible; otherwise solution.py is the public API integration point.
    api_routes = (
        (("reactor", "zerod"), "cantera_python/reactor.py"),
        (("transport",), "cantera_python/transport.py"),
        (("kinetics", "reaction", "rate"), "cantera_python/kinetics.py"),
        (("thermo", "phase", "species"), "cantera_python/thermo.py"),
        (("oned", "flame", "domain"), "cantera_python/oned.py"),
        (("yaml", "anymap", "factory"), "cantera_python/mechanism.py"),
    )
    for keywords, destination in api_routes:
        if any(keyword in haystack for keyword in keywords):
            return destination
    return "cantera_python/solution.py"


def _planned_test(path: str) -> dict:
    return {
        "path": path,
        "path_state": "EXISTS" if (ROOT / path).is_file() else "PLANNED_NEW_FILE",
        "requirement_state": "REQUIRED_NOT_CREDITED_AS_PASSING_BY_THIS_PLAN",
    }


def _implementation_plan(row: dict, component: str) -> dict:
    destination = _destination(row, component)
    return {
        "evidence_level": "INFERRED_FROM_SUBSYSTEM_AND_SYMBOL_METADATA",
        "implementation_state": "PLANNED_ONLY_DO_NOT_INFER_IMPLEMENTATION",
        "python_destination": {
            "path": destination,
            "path_state": "EXISTS" if (ROOT / destination).is_file() else "PLANNED_NEW_FILE",
        },
        "required_tests": [_planned_test(path) for path in COMPONENT_TESTS[component]],
        "oracle_validation": {
            "state": "PLANNED_NOT_EXECUTED",
            "reference": f"exact pinned Cantera commit {row['source_commit']}",
            "method": COMPONENT_ORACLE[component],
            "required_record": "Per-symbol or explicitly batched differential evidence with inputs, tolerances, outputs, and pass/fail result.",
        },
    }


def build_documents() -> tuple[dict, dict]:
    symbols = _load_symbols()
    inventory_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assignments = []
    nodes = []
    for row in symbols:
        component = _component(row)
        owner = COMPONENT_OWNER[component]
        plan = _implementation_plan(row, component)
        assignment = {
            "symbol_id": row["id"],
            "owner": owner,
            "component": component,
            "source_subsystem": row["subsystem"],
            "source_file": row["source_file"],
            "status": row["status"],
            "planning_evidence_level": plan["evidence_level"],
            "python_destination": plan["python_destination"],
            "required_tests": plan["required_tests"],
            "oracle_validation": plan["oracle_validation"],
        }
        assignments.append(assignment)
        nodes.append({
            "symbol_id": row["id"],
            "component": component,
            "owner": owner,
            "depends_on_components": COMPONENT_DEPENDENCIES[component],
            "dependency_semantics": "implementation_prerequisite",
            "source_proven_facts": {
                "source_file": row["source_file"],
                "source_sha256": row["source_sha256"],
                "line_range": row["line_range"],
                "symbol": row["symbol"],
                "class": row["class"],
                "type": row["type"],
            },
            "audit_recorded_facts": {
                "source_subsystem": row["subsystem"],
                "algorithm_category": row["algorithm_category"],
                "scientific_purpose": row["scientific_purpose"],
                "audit_status": row["status"],
                "recorded_python_equivalent": row["python_equivalent"],
            },
            "inferred_implementation_plan": plan,
        })

    owner_counts = Counter(item["owner"] for item in assignments)
    component_counts = Counter(item["component"] for item in assignments)
    common = {
        "schema_version": 2,
        "source_commit": symbols[0]["source_commit"],
        "source_inventory": "provenance/original_cantera_symbols.jsonl",
        "source_inventory_sha256": inventory_sha,
        "symbol_count": len(symbols),
        "evidence_labels": {
            "source_proven_facts": "Location and lexical identity copied from the pinned source inventory.",
            "audit_recorded_facts": "Prior audit classifications copied without revalidation by this planning tool.",
            "inferred_implementation_plan": "Proposed destination, tests, and oracle work; it is not implementation or validation evidence.",
        },
    }
    ownership = {
        **common,
        "invariants": {
            "expected_symbol_count": 5784,
            "unique_symbol_ids": True,
            "exactly_one_owner_per_symbol": True,
            "orphan_count": 0,
            "duplicate_assignment_count": 0,
        },
        "lanes": LANES,
        "owner_counts": dict(sorted(owner_counts.items())),
        "component_counts": dict(sorted(component_counts.items())),
        "assignment_policy": (
            "Scientific source subsystems map directly to their implementation lane. "
            "Base utilities, bindings, extensions, examples, and test-only symbols map "
            "to parser_api_provenance_tests. Unknown labels remain there pending audit."
        ),
        "assignments": assignments,
    }
    dependencies = {
        **common,
        "graph_kind": "capability_dependency_graph",
        "limitations": (
            "The lexical inventory does not encode a native call graph. Edges express "
            "conservative implementation prerequisites and must not be interpreted as "
            "symbol-level C++ call relationships."
        ),
        "components": {
            component: {
                "owner": COMPONENT_OWNER[component],
                "symbol_count": component_counts[component],
                "depends_on": dependencies,
            }
            for component, dependencies in COMPONENT_DEPENDENCIES.items()
        },
        "component_edges": [
            {"from": component, "requires": dependency}
            for component, dependencies in COMPONENT_DEPENDENCIES.items()
            for dependency in dependencies
        ],
        "nodes": nodes,
    }
    validate_documents(ownership, dependencies, symbols)
    return ownership, dependencies


def validate_documents(ownership: dict, dependencies: dict, symbols: list[dict] | None = None) -> None:
    symbols = symbols or _load_symbols()
    expected = {row["id"] for row in symbols}
    assignment_ids = [row["symbol_id"] for row in ownership["assignments"]]
    node_ids = [row["symbol_id"] for row in dependencies["nodes"]]
    for label, ids in (("ownership assignments", assignment_ids), ("dependency nodes", node_ids)):
        if len(ids) != 5784 or len(set(ids)) != 5784 or set(ids) != expected:
            raise ValueError(f"{label} do not cover each of the 5,784 symbols exactly once")
    valid_owners = set(LANES)
    if any(row["owner"] not in valid_owners for row in ownership["assignments"]):
        raise ValueError("Ownership document contains an unknown owner")
    valid_components = set(COMPONENT_DEPENDENCIES)
    for node in dependencies["nodes"]:
        if node["component"] not in valid_components:
            raise ValueError(f"Unknown component for {node['symbol_id']}")
        if any(item not in valid_components for item in node["depends_on_components"]):
            raise ValueError(f"Unknown dependency for {node['symbol_id']}")
        plan = node.get("inferred_implementation_plan", {})
        if plan.get("implementation_state") != "PLANNED_ONLY_DO_NOT_INFER_IMPLEMENTATION":
            raise ValueError(f"Missing non-implementation label for {node['symbol_id']}")
        if plan.get("oracle_validation", {}).get("state") != "PLANNED_NOT_EXECUTED":
            raise ValueError(f"Oracle plan incorrectly claims execution for {node['symbol_id']}")
        if not plan.get("python_destination", {}).get("path") or not plan.get("required_tests"):
            raise ValueError(f"Incomplete implementation plan for {node['symbol_id']}")


def _render(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Verify committed outputs are current")
    args = parser.parse_args()
    ownership, dependencies = build_documents()
    expected = {OWNERSHIP: _render(ownership), DEPENDENCIES: _render(dependencies)}
    if args.check:
        stale = [path for path, content in expected.items() if not path.exists() or path.read_text(encoding="utf8") != content]
        if stale:
            raise SystemExit("Stale symbol work map(s): " + ", ".join(str(path) for path in stale))
        print("Validated 5,784 unique symbol assignments and dependency nodes.")
        return 0
    for path, content in expected.items():
        path.write_text(content, encoding="utf8")
    print("Wrote ownership and dependency maps for 5,784 unique symbols.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
