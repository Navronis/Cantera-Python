from collections import Counter
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "provenance" / "canonical_5784_symbol_inventory.jsonl"
GENERATOR = ROOT / "tools" / "build_canonical_symbol_inventory.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("canonical_inventory_builder", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CanonicalInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [
            json.loads(line)
            for line in INVENTORY.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_artifact_is_current_and_preserves_denominator(self):
        self.assertEqual(len(self.rows), 5784)
        ids = [row["stable_symbol_id"] for row in self.rows]
        self.assertEqual(len(set(ids)), 5784)

    def test_required_schema_and_statuses(self):
        required = {
            "stable_symbol_id", "subsystem", "source_file", "line_range",
            "namespace", "class", "function_or_method", "signature",
            "visibility", "symbol_type", "source_hash",
            "scientific_runtime_relevance", "callers", "callees",
            "model_factory_registration", "python_mapping", "current_status",
        }
        self.assertTrue(all(required <= row.keys() for row in self.rows))
        self.assertEqual(
            Counter(row["current_status"] for row in self.rows),
            Counter({
                "NOT_PORTED": 2264,
                "PINNED_NATIVE_VALIDATED": 2487,
                "NON_RUNTIME_CPP_ONLY_PROVEN": 759,
                "PARTIAL": 274,
            }),
        )
        self.assertEqual(
            Counter(row["legacy_status"] for row in self.rows),
            Counter({
                "NOT_PORTED": 5027,
                "PARTIALLY_PORTED": 336,
                "EQUIVALENT_PROVEN": 421,
            }),
        )

    def test_deterministic_metadata_extraction_counts(self):
        status = lambda field: Counter(
            row["metadata_extraction_status"][field] for row in self.rows)
        self.assertEqual(status("namespace"), Counter({
            "LEXICAL_SCOPE": 5232,
            "DETERMINISTIC_NO_NAMESPACE_AT_SYMBOL": 552,
        }))
        self.assertEqual(status("class"), Counter({
            "LEXICAL_SCOPE": 4231,
            "DETERMINISTIC_NO_CLASS_AT_SYMBOL": 1553,
        }))
        self.assertEqual(status("signature"), Counter({
            "SOURCE_LINE_DECLARATION": 5784,
        }))
        self.assertEqual(status("visibility"), Counter({
            "LEXICAL_CLASS_ACCESS": 4231,
            "FREE_OR_NAMESPACE_FUNCTION": 1553,
        }))

    def test_relevance_is_candidate_routing_not_final_proof(self):
        self.assertEqual(
            Counter(row["scientific_runtime_relevance"] for row in self.rows),
            Counter({
                "SCIENTIFIC_RUNTIME_CANDIDATE": 4070,
                "NON_SYMBOL_INVENTORY_ARTIFACT": 759,
                "RUNTIME_UTILITY_CANDIDATE": 447,
                "VALIDATION_ONLY_CANDIDATE": 444,
                "NON_LIBRARY_SAMPLE_CANDIDATE": 54,
                "BINDING_RUNTIME_CANDIDATE": 8,
                "NON_LIBRARY_EXAMPLE_CANDIDATE": 2,
            }),
        )
        final_statuses = {
            "PINNED_NATIVE_VALIDATED",
            "SEMANTIC_PYTHON_EQUIVALENT_VALIDATED",
            "NON_RUNTIME_CPP_ONLY_PROVEN",
        }
        self.assertEqual(
            sum(row["current_status"] in final_statuses for row in self.rows), 3246)

    def test_ionflow_and_domain_factory_symbols_are_pinned_native_validated(self):
        ion_syms = [
            row for row in self.rows
            if "IonFlow" in row["source_file"]
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertEqual(len(ion_syms), 15)

        factory_syms = [
            row for row in self.rows
            if "DomainFactory" in row["source_file"]
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertEqual(len(factory_syms), 4)

        # Assert 100% of oneD subsystem is finalized with 0 open source symbols
        oned_open = [
            row for row in self.rows
            if row.get("subsystem") == "oneD"
            and not row.get("final_disposition")
        ]
        self.assertEqual(len(oned_open), 0)

    def test_kinetics_core_symbols_are_pinned_native_validated(self):
        kin_validated = [
            row for row in self.rows
            if row.get("subsystem") == "kinetics"
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertEqual(len(kin_validated), 667)

    def test_kinetics_subsystem_is_100_percent_finalized(self):
        kin_open = [
            row for row in self.rows
            if row.get("subsystem") == "kinetics"
            and not row.get("final_disposition")
        ]
        self.assertEqual(len(kin_open), 0)

    def test_family4_topology_symbols_are_pinned_native_validated(self):
        f4_validated = [
            row for row in self.rows
            if ("ReactionPath" in row["source_file"] or "Group" in row["source_file"])
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertGreaterEqual(len(f4_validated), 40)

    def test_interface_kinetics_symbols_are_pinned_native_validated(self):
        ik_validated = [
            row for row in self.rows
            if ("InterfaceKinetics" in row["source_file"] or "InterfaceRate" in row["source_file"])
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertGreaterEqual(len(ik_validated), 50)

    def test_domain1d_symbols_are_pinned_native_validated(self):
        dom_syms = [
            row for row in self.rows
            if ("Domain1D.h" in row["source_file"] or "Domain1D.cpp" in row["source_file"])
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertEqual(len(dom_syms), 93)

    def test_boundary1d_symbols_are_pinned_native_validated(self):
        bdry_syms = [
            row for row in self.rows
            if ("Boundary1D.h" in row["source_file"] or "Boundary1D.cpp" in row["source_file"])
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertEqual(len(bdry_syms), 70)

    def test_flow1d_symbols_are_pinned_native_validated(self):
        flow_syms = [
            row for row in self.rows
            if ("Flow1D" in row["source_file"] or "StFlow" in row["source_file"])
            and row["current_status"] == "PINNED_NATIVE_VALIDATED"
        ]
        self.assertEqual(len(flow_syms), 118)

    def test_source_reviewed_implementation_evidence_is_applied(self):
        dusty_gas = [
            row for row in self.rows
            if row["source_file"].endswith("DustyGasTransport.h")
        ]
        self.assertEqual(len(dusty_gas), 15)
        self.assertTrue(all(row["current_status"] == "PINNED_NATIVE_VALIDATED" for row in dusty_gas))
        self.assertTrue(all("tests/test_dusty_gas_transport.py" in row["validation_evidence"]
                            for row in dusty_gas))


if __name__ == "__main__":
    unittest.main()
