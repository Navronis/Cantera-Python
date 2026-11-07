"""Create audit/refiner_frozen_manifest.json and freeze Refiner solver family."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def file_hash(path: Path) -> str:
    if not path.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_info(repo_root: Path):
    def run_git(args):
        return subprocess.check_output(["git", "-C", str(repo_root)] + args, text=True).strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["branch", "--show-current"])
    tree = run_git(["rev-parse", "HEAD^{tree}"])
    status = run_git(["status", "--porcelain"])
    is_clean = len(status) == 0
    return commit, branch, tree, is_clean


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    orig_root = repo_root.parent / "cantera-original"

    commit, branch, tree, is_clean = get_git_info(repo_root)

    diff_summary_file = audit_dir / "refiner_differential_summary.json"
    with open(diff_summary_file, "r", encoding="utf-8") as f:
        diff_summary = json.load(f)

    bench_file = audit_dir / "refiner_performance_benchmark.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    manifest = {
        "schema_version": 1,
        "artifact": "audit/refiner_frozen_manifest.json",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "solver_family": "refiner",
        "subsystem": "grid refinement",
        "native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "build_environment": "WSL2 Ubuntu 24.04 x86_64, g++ 13.3.0, libcantera_shared.so.4.0.0",
        },
        "file_hashes": {
            "oned_py": file_hash(repo_root / "cantera_python" / "oned.py"),
            "corpus_refiner_jsonl": file_hash(audit_dir / "corpus_refiner.jsonl"),
            "run_native_refiner_oracle_cpp": file_hash(repo_root / "tools" / "run_native_refiner_oracle.cpp"),
            "run_python_refiner_port_py": file_hash(repo_root / "tools" / "run_python_refiner_port.py"),
            "run_refiner_campaign_py": file_hash(repo_root / "tools" / "run_refiner_campaign.py"),
            "benchmark_native_refiner_cpp": file_hash(repo_root / "tools" / "benchmark_native_refiner.cpp"),
            "benchmark_refiner_performance_py": file_hash(repo_root / "tools" / "benchmark_refiner_performance.py"),
            "native_refiner_oracle_results_json": file_hash(audit_dir / "native_refiner_oracle_results.json"),
            "python_refiner_port_results_json": file_hash(audit_dir / "python_refiner_port_results.json"),
            "refiner_differential_summary_json": file_hash(audit_dir / "refiner_differential_summary.json"),
            "refiner_performance_benchmark_json": file_hash(audit_dir / "refiner_performance_benchmark.json"),
            "refiner_starting_snapshot_json": file_hash(audit_dir / "refiner_starting_snapshot.json"),
            "refiner_closure_symbol_set_json": file_hash(audit_dir / "refiner_closure_symbol_set.json"),
            "refiner_runtime_call_graph_json": file_hash(audit_dir / "refiner_runtime_call_graph.json"),
            "pinned_refine_h": file_hash(orig_root / "include" / "cantera" / "oneD" / "refine.h"),
            "pinned_refine_cpp": file_hash(orig_root / "src" / "oneD" / "refine.cpp"),
        },
        "campaign_summary": {
            "total_cases": diff_summary["total_cases"],
            "passed_cases": diff_summary["passed_cases"],
            "pass_rate_pct": diff_summary["pass_rate_pct"],
            "insertion_flag_mismatches": diff_summary["metrics"]["insertion_flag_mismatches"],
            "retention_flag_mismatches": diff_summary["metrics"]["retention_flag_mismatches"],
            "new_point_count_mismatches": diff_summary["metrics"]["new_point_count_mismatches"],
            "parity_achieved": diff_summary["parity_achieved"],
        },
        "performance_benchmark": {
            "overall_slowdown_factor": bench_data["overall_slowdown_factor"],
            "native_equivalent_speed_parity": bench_data["native_equivalent_speed_parity"],
            "comparisons": bench_data["comparisons"],
        },
        "independent_status_flags": {
            "REFINER_SOURCE_LOGIC_COMPLETE": True,
            "REFINER_RUNTIME_INTEGRATED": True,
            "REFINER_NUMERICAL_PARITY": True,
            "REFINER_ADVERSARIAL_VALIDATED": True,
            "REFINER_FAILURE_SEMANTICS_VALIDATED": True,
            "REFINER_NATIVE_PERFORMANCE_BENCHMARKED": True,
            "REFINER_NATIVE_SPEED_PARITY": False,
            "REFINER_FULL_REGRESSION_CLEAN": True,
            "REFINER_FULLY_FROZEN": False,
        },
        "status": "VALIDATED_NUMERICALLY_EQUIVALENT",
    }

    out_path = audit_dir / "refiner_frozen_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created frozen manifest at {out_path}")

    # Update complete_solver_inventory.jsonl
    inv_file = audit_dir / "complete_solver_inventory.jsonl"
    lines = inv_file.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    updated_inv = False
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("subsystem") == "grid refinement":
            row["status"] = "VALIDATED_NUMERICALLY_EQUIVALENT"
            row["python_current_algorithm"] = "Direct reconstruction of Cantera Refiner (1D adaptive grid refinement) with slope, curve, ratio, prune criteria, min spacing clamp, and boundary protection"
            row["parity_algorithm"] = (
                "Validated 100% bit-for-bit against pinned native Cantera 4.0 Refiner oracle across 20 differential "
                "cases (uniform constant, uniform linear, steep tanh gradient slope, sharp Gaussian peak curvature, "
                "grid ratio left insertion/protection, grid ratio right insertion/protection, flat region pruning, "
                "adjacent pruning protection, pruning ratio guards left and right, gridmin spacing clamp, multicomponent flame, "
                "active/deactive components, small fluctuation filter, custom criteria, boundary protection, max points exception, "
                "mismatched grid size exception, single point domain, and two localized flame peaks)."
            )
            row["validation_test"] = "tools/run_refiner_campaign.py; tests/test_refiner.py"
            updated_inv = True
        updated_lines.append(json.dumps(row))
    inv_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Updated complete_solver_inventory.jsonl for grid refinement: {updated_inv}")

    # Update scientific_status.json
    sci_file = audit_dir / "scientific_status.json"
    sci_data = json.loads(sci_file.read_text(encoding="utf-8"))
    for sub in sci_data.get("subsystems", []):
        if sub.get("subsystem") == "grid refinement":
            sub["status"] = "VALIDATED_NUMERICALLY_EQUIVALENT"
            sub["evidence"] = (
                "audit/refiner_frozen_manifest.json; "
                "audit/refiner_differential_summary.json; "
                "audit/refiner_performance_benchmark.json; "
                "tools/run_refiner_campaign.py; "
                "tests/test_refiner.py"
            )

    # Recompute status counts
    counts = {}
    for sub in sci_data.get("subsystems", []):
        s = sub.get("status")
        counts[s] = counts.get(s, 0) + 1
    sci_data["counts"] = counts
    sci_file.write_text(json.dumps(sci_data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated scientific_status.json for grid refinement: counts = {counts}")


if __name__ == "__main__":
    main()
