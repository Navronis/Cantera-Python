import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"


def jsonl(name):
    return [
        json.loads(line)
        for line in (PROVENANCE / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class SymbolAuditArtifactTests(unittest.TestCase):
    def test_canonical_inventory_schema_and_identity(self):
        rows = jsonl("canonical_5784_symbol_inventory.jsonl")
        required = {
            "stable_symbol_id", "subsystem", "source_file", "line_range",
            "namespace", "class", "function_or_method", "signature",
            "visibility", "symbol_type", "source_hash",
            "scientific_runtime_relevance", "callers", "callees",
            "model_factory_registration", "python_mapping", "current_status",
        }
        self.assertEqual(len(rows), 5784)
        self.assertEqual(len({row["stable_symbol_id"] for row in rows}), 5784)
        self.assertTrue(all(required <= row.keys() for row in rows))

    def test_legacy_equivalence_revalidation_is_complete(self):
        rows = jsonl("revalidation_421.jsonl")
        summary = json.loads(
            (PROVENANCE / "revalidation_421_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(rows), 421)
        self.assertEqual(len({row["id"] for row in rows}), 421)
        self.assertEqual(summary["legacy_claims_revalidated"], 421)
        self.assertEqual(summary["equivalent_proven_retained"], 0)
        self.assertEqual(summary["rows_with_explicit_per_symbol_exact_pinned_evidence"], 0)

    def test_work_maps_cover_the_same_denominator(self):
        ownership = json.loads(
            (PROVENANCE / "symbol_ownership.json").read_text(encoding="utf-8")
        )
        graph = json.loads(
            (PROVENANCE / "symbol_dependency_graph.json").read_text(encoding="utf-8")
        )
        assignments = ownership["assignments"]
        nodes = graph["nodes"]
        self.assertEqual(len(assignments), 5784)
        self.assertEqual(len(nodes), 5784)
        self.assertEqual(
            {row["symbol_id"] for row in assignments},
            {row["symbol_id"] for row in nodes},
        )


if __name__ == "__main__":
    unittest.main()
