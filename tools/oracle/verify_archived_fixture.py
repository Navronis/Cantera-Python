"""Verify identity and structural integrity of archived pinned-oracle fixtures.

This gate does not claim that a native oracle was regenerated.  It proves that
the checked-out source is the requested commit and that the archived fixtures
are byte-for-byte the artifacts whose hashes are recorded in this repository.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
SOURCE = ROOT.parent / "cantera-original"
REPORT = ROOT / "provenance" / "oracle_fixture_verification.json"
EXPECTED_COMMIT = "726522be4e2a13454d8415b7ef799d621f665cf3"
FIXTURES = {
    ROOT / "audit" / "wsl_oracle_data.json": {
        "sha256": "72701a5579ed60d1b9c7b55d8e47c66804f526dc8da52131ef9ff88b8369c57a",
        "commit_path": ("metadata", "cantera_commit"),
        "version_path": ("metadata", "cantera_version"),
        "required": {
            "thermo_10000": 10050,
            "state_setters": 8,
            "reactor_trajectories": 2,
        },
    },
    ROOT / "provenance" / "oracle_baseline_pinned.json": {
        "sha256": "4ffc6b1d07add187e71940f1dcbc14af39328b26abee7990713f8fb208e3623e",
        "commit_path": ("oracle_commit",),
        "version_path": ("oracle_version",),
        "required": {"species_thermo": 5, "states": 2},
    },
}


def nested(data, path):
    for key in path:
        data = data[key]
    return data


def resolve_commit_prefix(prefix: str) -> str | None:
    if len(prefix) < 7 or any(c not in "0123456789abcdefABCDEF" for c in prefix):
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "--verify", f"{prefix}^{{commit}}"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    checks = []
    try:
        source_commit = subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        source_commit = None
        checks.append({"check": "source_commit_readable", "pass": False,
                       "detail": str(exc)})
    else:
        checks.append({"check": "source_commit_exact", "pass": source_commit == EXPECTED_COMMIT,
                       "expected": EXPECTED_COMMIT, "actual": source_commit})

    fixture_rows = []
    for path, spec in FIXTURES.items():
        row = {"fixture": str(path.relative_to(ROOT))}
        if not path.is_file():
            row.update({"pass": False, "error": "missing fixture"})
            fixture_rows.append(row)
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding="utf-8"))
        reported_commit = str(nested(data, spec["commit_path"]))
        version = str(nested(data, spec["version_path"]))
        shape = {key: len(data[key]) for key in spec["required"]}
        hash_ok = digest == spec["sha256"]
        # Older Cantera exposed an abbreviated __git_commit__. Resolve that
        # abbreviation through Git, which also rejects ambiguous prefixes.
        resolved_commit = resolve_commit_prefix(reported_commit)
        commit_ok = (resolved_commit == EXPECTED_COMMIT
                     and source_commit == EXPECTED_COMMIT)
        shape_ok = shape == spec["required"]
        row.update({
            "sha256": digest,
            "expected_sha256": spec["sha256"],
            "hash_matches_record": hash_ok,
            "reported_commit": reported_commit,
            "resolved_commit": resolved_commit,
            "commit_matches_checkout": commit_ok,
            "reported_version": version,
            "observed_shape": shape,
            "expected_shape": spec["required"],
            "shape_matches": shape_ok,
            "pass": hash_ok and commit_ok and shape_ok,
        })
        fixture_rows.append(row)

    passed = all(item.get("pass", False) for item in checks + fixture_rows)
    report = {
        "schema_version": 1,
        "expected_pinned_commit": EXPECTED_COMMIT,
        "source_checkout": str(SOURCE),
        "source_commit": source_commit,
        "checks": checks,
        "fixtures": fixture_rows,
        "passed": passed,
        "native_fixture_regenerated_this_run": False,
        "evidentiary_limit": (
            "This verifies checkout identity, recorded artifact hashes, reported "
            "commit prefixes, and fixture structure. It does not independently "
            "reproduce the native build or prove how the archived bytes were generated."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
