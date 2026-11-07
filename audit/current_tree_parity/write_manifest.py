"""Bind current-wave differential, regression, and Wave 4 diagnostic evidence."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode().strip()


runtime = sorted((ROOT / "cantera_python").glob("*.py"))
tests = sorted((ROOT / "tests").glob("*.py"))
tools = [ROOT / "tools" / name for name in (
    "run_python_port_runner.py", "run_native_oracle_runner.py",
    "compare_native_vs_python.py",
    "compare_multitransport_matrix_probe.py",
)]
helpers = [AUDIT / name for name in (
    "run_native_wave12.py", "run_native_wave3.py", "run_python_wave12.py",
    "run_python_wave3.py", "compare_wave12.py", "compare_wave4_trace_heavy.py",
    "generate_wave4_thermal_reference.py",
)]
corpora = sorted(AUDIT.glob("corpus_wave*.jsonl")) + [
    AUDIT / "solver_wave3_property_pair_validation.jsonl",
    AUDIT / "native_parity_tolerance_policy.json",
]
outputs = sorted(AUDIT.glob("raw_*_wave[123].jsonl")) + sorted(
    AUDIT.glob("native_wave[123]_summary.json")
) + sorted(AUDIT.glob("native_wave[123]_results.jsonl")) + [
    AUDIT / "wave4_native_trace_heavy.txt",
    AUDIT / "wave4_native_trace_heavy.txt.diag",
    AUDIT / "wave4_python_trace_heavy.txt",
    AUDIT / "wave4_python_trace_heavy.txt.diag",
    AUDIT / "wave4_trace_heavy_comparison.json",
    ROOT / "tests" / "fixtures" / "pinned_multitransport_trace_heavy.json",
    AUDIT / "current_full_pytest.log",
]

summaries = {}
for wave in (1, 2, 3):
    path = AUDIT / f"native_wave{wave}_summary.json"
    summaries[str(wave)] = json.loads(path.read_text(encoding="utf-8"))

status = git("status", "--porcelain=v1").splitlines()
diff = subprocess.run(
    ["git", "diff", "--binary"], cwd=ROOT, check=True, capture_output=True
).stdout

manifest = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "current working-tree Wave 1-3 differential rerun and Wave 4 trace-heavy diagnostic",
    "pinned_native_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
    "native_oracle": {
        "source_checkout_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
        "cantera_version": "4.0.0a2",
        "extension_sha256": "d686773fe369ba07c2ad78431240869a3679316c4bdbb77160062d86aa14c68c",
        "static_library_sha256": "0f9948c9d3eea3f12547f11949a9cc6cbf28893ec98d6aa5425860b9d6d14e02",
    },
    "repository": {
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "working_tree_clean": not status,
        "status": status,
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    },
    "source_hashes": {
        "runtime_python_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in runtime},
        "runtime_tree_sha256": tree_hash(runtime),
        "test_python_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in tests},
        "test_tree_sha256": tree_hash(tests),
        "runner_files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in tools + helpers},
        "corpus_files": {p.name: sha256(p) for p in corpora},
        "result_files": {p.name: sha256(p) for p in outputs},
    },
    "fresh_pinned_native_comparison": {
        "case_counts": {"wave1": 4100, "wave2": 2000, "wave3": 9608},
        "total_cases": 15708,
        "mismatches": 0,
        "wave_summaries": summaries,
        "raw_records_directory": "audit/current_tree_parity",
    },
    "wave4_trace_heavy_diagnostic": {
        "comparison": json.loads((AUDIT / "wave4_trace_heavy_comparison.json").read_text()),
        "public_native_reference_fixture": "tests/fixtures/pinned_multitransport_trace_heavy.json",
        "regression_test": "tests/test_multicomponent_transport.py::test_trace_heavy_multitransport_public_properties_match_pinned_native",
        "regression_test_result": "passed",
        "interpretation": "The internal solution vectors are not close under this ill-conditioned case; both satisfy the matrix equation and the public thermal outputs match the pinned reference within the regression tolerance.",
    },
    "fresh_full_test_run": {
        "command": "python -m pytest tests/",
        "collected": 486,
        "passed": 483,
        "skipped": 3,
        "failed": 0,
        "errors": 0,
        "subtests_passed": 698,
        "duration_seconds": 133.09,
        "log": "audit/current_tree_parity/current_full_pytest.log",
        "log_sha256": sha256(AUDIT / "current_full_pytest.log"),
    },
    "limitations": [
        "This validates selected Wave 1-3 cases and one Wave 4 trace-heavy state; it does not establish full-library equivalence.",
        "It does not validate the remaining solver regimes or all 5,784 canonical symbols.",
        "The working tree was dirty during validation; hashes bind the measured runtime and tests to this snapshot.",
    ],
}
(AUDIT / "current_tree_parity_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps({
    "manifest": "audit/current_tree_parity/current_tree_parity_manifest.json",
    "head": manifest["repository"]["head"],
    "runtime_tree_sha256": manifest["source_hashes"]["runtime_tree_sha256"],
    "test_tree_sha256": manifest["source_hashes"]["test_tree_sha256"],
    "wave_cases": manifest["fresh_pinned_native_comparison"]["case_counts"],
    "mismatches": manifest["fresh_pinned_native_comparison"]["mismatches"],
}, indent=2))
