import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"


class SymbolWorkMapTests(unittest.TestCase):
    def test_generated_maps_are_current_and_complete(self):
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "build_symbol_work_maps.py"), "--check"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        source_ids = {
            json.loads(line)["id"]
            for line in (PROVENANCE / "original_cantera_symbols.jsonl").read_text(encoding="utf8").splitlines()
            if line
        }
        ownership = json.loads((PROVENANCE / "symbol_ownership.json").read_text(encoding="utf8"))
        graph = json.loads((PROVENANCE / "symbol_dependency_graph.json").read_text(encoding="utf8"))
        assignment_ids = [row["symbol_id"] for row in ownership["assignments"]]
        node_ids = [row["symbol_id"] for row in graph["nodes"]]

        self.assertEqual(len(source_ids), 5784)
        self.assertEqual(Counter(assignment_ids), Counter(source_ids))
        self.assertEqual(Counter(node_ids), Counter(source_ids))
        self.assertEqual(sum(ownership["owner_counts"].values()), 5784)
        self.assertEqual(
            set(ownership["owner_counts"]),
            {
                "thermo_tpx_equilibrium",
                "kinetics_interfaces_transport",
                "numerics_zeroD_oneD",
                "parser_api_provenance_tests",
            },
        )
        for node in graph["nodes"]:
            self.assertEqual(
                node["inferred_implementation_plan"]["implementation_state"],
                "PLANNED_ONLY_DO_NOT_INFER_IMPLEMENTATION",
            )
            self.assertEqual(
                node["inferred_implementation_plan"]["oracle_validation"]["state"],
                "PLANNED_NOT_EXECUTED",
            )
            self.assertTrue(node["inferred_implementation_plan"]["python_destination"]["path"])
            self.assertTrue(node["inferred_implementation_plan"]["required_tests"])
            self.assertEqual(
                set(node["source_proven_facts"]),
                {"source_file", "source_sha256", "line_range", "symbol", "class", "type"},
            )
            self.assertIn("audit_status", node["audit_recorded_facts"])


if __name__ == "__main__":
    unittest.main()
