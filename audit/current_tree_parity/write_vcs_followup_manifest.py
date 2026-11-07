"""Record the VCS safety fix and current full-suite evidence separately."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


log = AUDIT / "followup_full_pytest_20260929.log"
summary = log.read_text(encoding="utf-8").splitlines()[-1]
match = re.fullmatch(
    r"(\d+) passed, (\d+) skipped, (\d+) subtests passed in ([\d.]+)s(?: \(.*\))?",
    summary.strip(),
)
if not match:
    raise RuntimeError(f"Unrecognized pytest summary: {summary!r}")
targeted_log = AUDIT / "vcs_targeted_20260929.log"
targeted_summary = targeted_log.read_text(encoding="utf-8").splitlines()[-1]
targeted_match = re.fullmatch(r"(\d+) passed in [\d.]+s", targeted_summary.strip())
if not targeted_match:
    raise RuntimeError(f"Unrecognized targeted pytest summary: {targeted_summary!r}")

previous_manifest = AUDIT / "current_tree_parity_manifest.json"
paths = (
    ROOT / "cantera_python" / "vcs.py",
    ROOT / "tests" / "test_vcs_multiphase.py",
    ROOT / "tests" / "test_multiphase_container.py",
    ROOT / "tests" / "test_reactor_conservation.py",
    ROOT / "tests" / "fixtures" / "pinned_multiphase_container.json",
    AUDIT / "sensitivity_fd_diagnostic.py",
    AUDIT / "generate_native_multiphase_reference.py",
)
manifest = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "VCS API honesty and single-phase extensive-state regression follow-up",
    "pinned_source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
    "repository": {
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "working_tree_dirty": True,
    },
    "changed_file_sha256": {
        str(path.relative_to(ROOT).as_posix()): sha256(path) for path in paths
    },
    "vcs_disposition": {
        "multi_phase_vcs": "NOT_IMPLEMENTED",
        "behavior": "Explicitly raises before state mutation; no active-set result is fabricated.",
        "single_phase_convenience": "Delegates to the phase equilibrium implementation and scales phase amount to preserve the initial mass basis.",
        "solver_family_inventory_count_changed": False,
        "native_pinned_solver_parity_claimed": False,
        "container_parity": "One pinned-native two-phase state compares 54 species, 5 element totals, phase moles, composition, phase charge, and global charge.",
    },
    "targeted_tests": {
        "command": "python -m pytest tests/test_multiphase_container.py tests/test_vcs_multiphase.py -q",
        "summary": targeted_summary.strip(),
        "passed": int(targeted_match.group(1)),
        "failed": 0,
        "limitations": "The test file exercises existing single-phase equilibrium behavior and explicit rejection of unsupported multiphase VCS; it does not validate VCS parity.",
        "log": str(targeted_log.relative_to(ROOT).as_posix()),
        "log_sha256": sha256(targeted_log),
    },
    "full_test_run": {
        "command": "python -m pytest tests/ -q",
        "summary": summary.strip(),
        "passed": int(match.group(1)),
        "skipped": int(match.group(2)),
        "subtests_passed": int(match.group(3)),
        "duration_seconds": float(match.group(4)),
        "log": str(log.relative_to(ROOT).as_posix()),
        "log_sha256": sha256(log),
        "runtime": "WSL Ubuntu, /home/nebula/noyron_env/bin/python",
    },
    "sensitivity_reference_diagnostic": {
        "script": str((AUDIT / "sensitivity_fd_diagnostic.py").relative_to(ROOT).as_posix()),
        "finding": "Centered finite difference at step 1e-3 agrees with augmented sensitivity to 1.45e-8 relative; the prior one-sided step 1e-5 differed by 4.32e-6 due to first-order truncation.",
        "test_reference_updated": True,
    },
    "prior_wave1_to_wave4_evidence": {
        "manifest": str(previous_manifest.relative_to(ROOT).as_posix()),
        "manifest_sha256": sha256(previous_manifest),
        "note": "Historical evidence remains scoped to the exact runtime/test hashes in that manifest; this follow-up does not rebind or expand its native parity claims.",
    },
}
out = AUDIT / "vcs_followup_20260929.json"
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"manifest": str(out.relative_to(ROOT)), "summary": summary.strip()}, indent=2))
