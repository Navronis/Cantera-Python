#!/usr/bin/env python3
"""Run and preserve the exact repository pytest command and exit status."""

from __future__ import annotations

import json
import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
LOG = AUDIT / "waves_1_3_full_pytest.log"
RESULT = AUDIT / "waves_1_3_test_results.json"
PROVENANCE = AUDIT / "waves_1_3_test_run_provenance.json"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def test_source_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "cantera_python").rglob("*.py"))
    paths += sorted((ROOT / "tests").rglob("*.py"))
    paths += [ROOT / "pyproject.toml", Path(__file__).resolve()]
    return {path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths if path.is_file()}


def main() -> int:
    command = [sys.executable, "-m", "pytest", "tests/"]
    started_at = datetime.now(timezone.utc).isoformat()
    source_hashes = test_source_hashes()
    source_digest = hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    run_commit = git("rev-parse", "HEAD")
    run_branch = git("branch", "--show-current")
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    CANONICAL_LOG = AUDIT / "canonical_full_pytest.log"
    CANONICAL_RESULT = AUDIT / "canonical_full_pytest_results.json"
    LOG.write_text(proc.stdout, encoding="utf-8")
    CANONICAL_LOG.write_text(proc.stdout, encoding="utf-8")
    match = re.search(r"(\d+) passed(?:, (\d+) skipped)?(?:, (\d+) failed)?(?:, (\d+) error)?", proc.stdout)
    result = {
        "command": command,
        "cwd": str(ROOT),
        "started_at_utc": started_at,
        "exit_code": proc.returncode,
        "log_path": str(LOG.relative_to(ROOT)),
        "pytest_available": "No module named pytest" not in proc.stdout,
        "collected": None,
        "passed": int(match.group(1)) if match else None,
        "skipped": int(match.group(2) or 0) if match else None,
        "failed": int(match.group(3) or 0) if match else None,
        "errors": int(match.group(4) or 0) if match else None,
        "summary_parsed": match is not None,
    }
    collected = re.search(r"collected\s+(\d+)\s+items?", proc.stdout)
    if collected:
        result["collected"] = int(collected.group(1))
    result["log_sha256"] = file_hash(LOG)
    result["git_commit"] = run_commit
    result["branch"] = run_branch
    result["test_source_sha256"] = source_digest
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    CANONICAL_RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "schema": "waves-1-3-test-run-provenance-v1",
        "command": command,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": run_commit,
        "branch": run_branch,
        "test_source_sha256": source_digest,
        "source_files": source_hashes,
        "exit_code": proc.returncode,
        "collected": result["collected"],
        "passed": result["passed"],
        "skipped": result["skipped"],
        "failed": result["failed"],
        "errors": result["errors"],
        "log_path": str(LOG.relative_to(ROOT)),
        "log_sha256": result["log_sha256"],
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(proc.stdout, end="")
    print(f"\nRecorded result: exit={proc.returncode}; pytest_available={result['pytest_available']}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
