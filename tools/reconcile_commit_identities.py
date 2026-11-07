#!/usr/bin/env python3
"""Audit and reconcile commit references across repository manifests and gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
OUT = AUDIT / "canonical_repository_identity_reconciliation.json"

REFERENCED_COMMITS = [
    "b539544bfa940a39fcf1cc4dd970b72aad89959f", # HEAD
    "4f44e60594eb544f078f2acd60666225c4011c74", # referenced by validation gate
    "3080be54b4d34f78faf83a961bccac9d1c26f601", # referenced by committed manifest
    "dfe32f8764ac1253908045c7d9bd7eb4de34b10f", # intermediate
    "417a58b2f6188999e7070327e417b67c987bdd9e", # intermediate
    "495f0225393cf651e89886cda825775d6b590b56", # intermediate
    "09201e2aa79cb481b4dcd5e763637942aaf86c50", # clean pytest commit
    "005f59140da86f838b21116274671447069c7aa2", # initial baseline commit
    "147b4c71476039634be17fa006066a254f6efc58", # pre-recertification freeze base
]

KEY_SCIENTIFIC_FILES = [
    "cantera_python/state.py",
    "cantera_python/thermo.py",
    "cantera_python/ideal_gas.py",
    "cantera_python/phases.py",
    "cantera_python/cubic_eos.py",
    "cantera_python/tpx.py",
    "cantera_python/tpx_exact.py",
    "cantera_python/numerics.py",
    "cantera_python/solution.py",
    "cantera_python/constants.py",
    "cantera_python/state_revision.py",
    "cantera_python/transport.py",
]

KEY_TEST_FILES = [
    "tests/test_state.py",
    "tests/test_ideal_gas.py",
    "tests/test_cubic_eos.py",
    "tests/test_cubic_inverse_source.py",
    "tests/test_tpx.py",
    "tests/test_state_failure_mutation.py",
    "tests/test_transport.py",
    "tests/test_multicomponent_transport.py",
]

KEY_TOOL_FILES = [
    "tools/compare_native_vs_python.py",
    "tools/wave1_failure_state_probe.py",
    "tools/compare_wave1_failure_state.py",
    "tools/compare_wave2_internals.py",
    "tools/run_python_cubic_internal_probe.py",
    "tools/run_native_cubic_internal_probe.cpp",
    "tools/wave3_cp_cv_step_sweep.py",
    "tools/generate_wave3_fallback_activation_corpus.py",
    "tools/validate_waves_1_3_current_head.py",
    "tools/build_waves_1_3_repository_identity.py",
    "tools/bind_waves_1_3_outputs.py",
]


def run_git(*args: str) -> str:
    res = subprocess.run(["git", *args], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        return ""
    return res.stdout.strip()


def git_file_content(commit: str, path: str) -> bytes | None:
    res = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return None
    return res.stdout


def sha256_bytes(b: bytes | None) -> str | None:
    if b is None:
        return None
    return hashlib.sha256(b).hexdigest()


def is_ancestor(ancestor: str, descendant: str) -> bool:
    res = subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=ROOT)
    return res.returncode == 0


def main():
    head = run_git("rev-parse", "HEAD")
    head_tree = run_git("rev-parse", "HEAD^{tree}")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--porcelain=v1", "--untracked-files=all")
    clean = not bool(status.strip())

    reconciliation = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_repository_state": {
            "HEAD": head,
            "HEAD_tree": head_tree,
            "branch": branch,
            "clean": clean,
            "status_porcelain": status,
        },
        "commit_records": {},
        "comparison_matrix": {},
    }

    # Gather head hashes for scientific files
    head_sci_hashes = {f: sha256_bytes(git_file_content(head, f)) for f in KEY_SCIENTIFIC_FILES}

    for rev in REFERENCED_COMMITS:
        c_full = run_git("rev-parse", "--verify", rev)
        if not c_full:
            reconciliation["commit_records"][rev] = {"error": "COMMIT_NOT_FOUND"}
            continue
        parent = run_git("rev-parse", f"{c_full}^")
        tree = run_git("rev-parse", f"{c_full}^{{tree}}")
        subject = run_git("log", "-n", "1", "--format=%s", c_full)
        author_date = run_git("log", "-n", "1", "--format=%ai", c_full)

        sci_hashes = {f: sha256_bytes(git_file_content(c_full, f)) for f in KEY_SCIENTIFIC_FILES}
        test_hashes = {f: sha256_bytes(git_file_content(c_full, f)) for f in KEY_TEST_FILES}
        tool_hashes = {f: sha256_bytes(git_file_content(c_full, f)) for f in KEY_TOOL_FILES}

        # Diff relative to HEAD for cantera_python/
        diff_sci = run_git("diff", c_full, head, "--", "cantera_python")
        diff_tests = run_git("diff", c_full, head, "--", "tests")
        diff_all = run_git("diff", c_full, head, "--stat")

        ancestor_of_head = is_ancestor(c_full, head)
        descendant_of_head = is_ancestor(head, c_full)

        # Classification
        if not diff_sci:
            if not diff_tests:
                if not diff_all:
                    classification = "IDENTICAL_SNAPSHOT"
                else:
                    classification = "IDENTICAL_SCIENTIFIC_SNAPSHOT"
            else:
                classification = "TEST_DIFFERENCE_IDENTICAL_SCIENTIFIC_SNAPSHOT"
        else:
            classification = "SCIENTIFIC_IMPLEMENTATION_DIFFERENCE"

        reconciliation["commit_records"][c_full] = {
            "short_sha": c_full[:7],
            "commit_sha": c_full,
            "parent_sha": parent,
            "tree_sha": tree,
            "subject": subject,
            "author_date": author_date,
            "ancestor_of_head": ancestor_of_head,
            "descendant_of_head": descendant_of_head,
            "classification": classification,
            "scientific_diff_empty": not bool(diff_sci.strip()),
            "tests_diff_empty": not bool(diff_tests.strip()),
            "scientific_hashes": sci_hashes,
            "test_hashes": test_hashes,
            "tool_hashes": tool_hashes,
            "diff_summary_vs_head": diff_all.splitlines()[-1] if diff_all else "No differences",
        }

    OUT.write_text(json.dumps(reconciliation, indent=2) + "\n", encoding="utf-8")
    print(f"Created {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
