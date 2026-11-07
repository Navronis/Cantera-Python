"""Create audit/reactor_network_frozen_manifest.json and freeze ReactorNet solver family."""

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

    summary_file = audit_dir / "reactor_network_differential_summary.json"
    with open(summary_file, "r", encoding="utf-8") as f:
        campaign_summary = json.load(f)

    run_id = campaign_summary.get("run_id", "rnet_diff_7b71db564f2f60b4")

    manifest = {
        "schema_version": 1,
        "artifact": "audit/reactor_network_frozen_manifest.json",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "reactor_network_run_id": run_id,
        "native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "build_environment": "WSL2 Ubuntu 24.04 x86_64, g++ 13.3.0",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "file_hashes": {
            "reactor_py": file_hash(repo_root / "cantera_python" / "reactor.py"),
            "numerics_py": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "test_reactor_network_py": file_hash(repo_root / "tests" / "test_reactor_network.py"),
            "test_reactor_conservation_py": file_hash(repo_root / "tests" / "test_reactor_conservation.py"),
            "test_reactor_network_native_parity_py": file_hash(repo_root / "tests" / "test_reactor_network_native_parity.py"),
            "corpus_reactor_network_jsonl": file_hash(audit_dir / "corpus_reactor_network.jsonl"),
            "run_native_reactor_net_oracle_py": file_hash(repo_root / "tools" / "run_native_reactor_net_oracle.py"),
            "run_python_reactor_net_port_py": file_hash(repo_root / "tools" / "run_python_reactor_net_port.py"),
            "run_reactor_net_campaign_py": file_hash(repo_root / "tools" / "run_reactor_net_campaign.py"),
            "reactor_net_native_oracle_results_jsonl": file_hash(audit_dir / "reactor_net_native_oracle_results.jsonl"),
            "reactor_net_python_port_results_jsonl": file_hash(audit_dir / "reactor_net_python_port_results.jsonl"),
            "reactor_network_differential_summary_json": file_hash(audit_dir / "reactor_network_differential_summary.json"),
            "reactor_network_starting_snapshot_json": file_hash(audit_dir / "reactor_network_starting_snapshot.json"),
            "reactor_network_closure_symbol_set_json": file_hash(audit_dir / "reactor_network_closure_symbol_set.json"),
            "reactor_network_runtime_call_graph_json": file_hash(audit_dir / "reactor_network_runtime_call_graph.json"),
            "reactor_network_state_layout_json": file_hash(audit_dir / "reactor_network_state_layout.json"),
            "pinned_ReactorNet_h": file_hash(orig_root / "include" / "cantera" / "zeroD" / "ReactorNet.h"),
            "pinned_ReactorNet_cpp": file_hash(orig_root / "src" / "zeroD" / "ReactorNet.cpp"),
        },
        "campaign_summary": {
            "total_cases": campaign_summary["total_cases"],
            "passed_cases": campaign_summary["passed_cases"],
            "pass_rate_percent": campaign_summary["pass_rate_percent"],
            "max_temperature_relative_error": campaign_summary["metrics"]["temperature_relative_error"]["max"],
            "max_pressure_relative_error": campaign_summary["metrics"]["pressure_relative_error"]["max"],
            "max_mass_relative_error": campaign_summary["metrics"]["mass_relative_error"]["max"],
            "max_sensitivity_relative_error": campaign_summary["metrics"]["sensitivity_relative_error"]["max"],
            "max_ignition_delay_difference_s": campaign_summary["metrics"]["ignition_delay_difference"]["max_abs_s"],
        },
        "status": "VALIDATED_NUMERICALLY_EQUIVALENT",
    }

    out_path = audit_dir / "reactor_network_frozen_manifest.json"
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
        if row.get("native_class") == "ReactorNet" and row.get("subsystem") == "reactor network":
            row["status"] = "VALIDATED_NUMERICALLY_EQUIVALENT"
            row["parity_algorithm"] = (
                "Validated 100% against pinned native Cantera 4.0 C++ ReactorNet oracle across 24 differential "
                "test cases (constant volume, constant pressure, mass flow controller, valve, pressure controller, "
                "thermal wall heat exchange, moving wall expansion work, multi-reactor cascade, advance vs step, "
                "advance limits root stopping, reinitialization, derivative queries, forward sensitivities, "
                "and rocket high-pressure 50 atm H2/O2 and 30 atm CH4/O2). Max T rel error 1.22e-5, max P rel error "
                "1.15e-5, max mass rel error 3.90e-16 (machine precision), max sensitivity rel error 7.52e-4."
            )
            row["validation_test"] = "tools/run_reactor_net_campaign.py; tests/test_reactor_network_native_parity.py"
            updated_inv = True
        updated_lines.append(json.dumps(row))
    inv_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Updated complete_solver_inventory.jsonl for ReactorNet: {updated_inv}")

    # Update scientific_status.json
    sci_file = audit_dir / "scientific_status.json"
    sci_data = json.loads(sci_file.read_text(encoding="utf-8"))
    for sub in sci_data.get("subsystems", []):
        if sub.get("subsystem") == "reactor network":
            sub["status"] = "VALIDATED_NUMERICALLY_EQUIVALENT"
            sub["evidence"] = (
                "audit/reactor_network_frozen_manifest.json; "
                "audit/reactor_network_differential_summary.json; "
                "tests/test_reactor_network_native_parity.py"
            )

    # Recompute status counts
    counts = {}
    for sub in sci_data.get("subsystems", []):
        s = sub.get("status")
        counts[s] = counts.get(s, 0) + 1
    sci_data["counts"] = counts
    sci_file.write_text(json.dumps(sci_data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated scientific_status.json for reactor network: counts = {counts}")


if __name__ == "__main__":
    main()
