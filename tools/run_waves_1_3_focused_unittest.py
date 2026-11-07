#!/usr/bin/env python3
"""Run and preserve dependency-light Waves 1-3 inverse-solver regressions."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AUDIT = ROOT / "audit"
TESTS = ("tests.test_state", "tests.test_state_failure_mutation", "tests.test_cubic_inverse_source")


def main():
    # Avoid importing unrelated package __init__ modules that require optional
    # SciPy/mechanism-parser dependencies absent from this runtime.
    package = types.ModuleType("cantera_python")
    package.__path__ = [str(ROOT / "cantera_python")]
    sys.modules["cantera_python"] = package
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for module in TESTS:
        suite.addTests(loader.loadTestsFromName(module))
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    log = output.getvalue()
    (AUDIT / "waves_1_3_focused_unittest.log").write_text(log, encoding="utf-8")
    result_json = {
        "command": "python tools/run_waves_1_3_focused_unittest.py",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_modules": list(TESTS),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "exit_code": 0 if result.wasSuccessful() else 1,
        "scope_note": "Focused inverse-state and cubic inverse tests; this is not the full repository pytest suite.",
        "log_path": "audit/waves_1_3_focused_unittest.log",
    }
    (AUDIT / "waves_1_3_focused_unittest_results.json").write_text(json.dumps(result_json, indent=2) + "\n", encoding="utf-8")
    print(log, end="")
    return result_json["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
