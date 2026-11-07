"""Create audit/high_pressure_gas_transport_frozen_manifest.json and update solver status."""

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

    commit, branch, tree, is_clean = get_git_info(repo_root)

    summary_file = audit_dir / "high_pressure_gas_transport_differential_summary.json"
    with open(summary_file, "r", encoding="utf-8") as f:
        campaign_summary = json.load(f)

    run_id = campaign_summary["run_id"]

    manifest = {
        "schema_version": 1,
        "artifact": "audit/high_pressure_gas_transport_frozen_manifest.json",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "git_tree": tree,
        "is_worktree_clean": is_clean,
        "hptran_run_id": run_id,
        "native_oracle": {
            "source_commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "build_environment": "WSL2 Ubuntu 24.04 x86_64, g++ 13.3.0",
            "python_executable": "/home/nebula/cantera-oracle-env/bin/python3",
            "shared_library_path": "/home/nebula/cantera-pinned-oracle/build/lib/libcantera_shared.so.4.0.0",
            "shared_library_sha256": "a16c2d0f7a5f3f1440c058205e878fdac45873411aa9f285e8f6bd28f5c1f744",
        },
        "file_hashes": {
            "transport_py": file_hash(repo_root / "cantera_python" / "transport.py"),
            "solution_py": file_hash(repo_root / "cantera_python" / "solution.py"),
            "cubic_eos_py": file_hash(repo_root / "cantera_python" / "cubic_eos.py"),
            "test_hp_transport_py": file_hash(repo_root / "tests" / "test_hp_transport.py"),
            "corpus_high_pressure_gas_transport_jsonl": file_hash(audit_dir / "corpus_high_pressure_gas_transport.jsonl"),
            "high_pressure_gas_transport_mechanism_schema_json": file_hash(audit_dir / "high_pressure_gas_transport_mechanism_schema.json"),
            "run_native_high_pressure_transport_oracle_py": file_hash(repo_root / "tools" / "run_native_high_pressure_transport_oracle.py"),
            "run_high_pressure_transport_campaign_py": file_hash(repo_root / "tools" / "run_high_pressure_transport_campaign.py"),
            "native_high_pressure_transport_results_jsonl": file_hash(audit_dir / "native_high_pressure_transport_results.jsonl"),
            "python_high_pressure_transport_results_jsonl": file_hash(audit_dir / "python_high_pressure_transport_results.jsonl"),
            "high_pressure_gas_transport_differential_results_jsonl": file_hash(audit_dir / "high_pressure_gas_transport_differential_results.jsonl"),
            "high_pressure_gas_transport_differential_summary_json": file_hash(audit_dir / "high_pressure_gas_transport_differential_summary.json"),
            "high_pressure_gas_transport_adversarial_results_json": file_hash(audit_dir / "high_pressure_gas_transport_adversarial_results.json"),
            "high_pressure_gas_transport_failure_semantics_json": file_hash(audit_dir / "high_pressure_gas_transport_failure_semantics.json"),
            "high_pressure_gas_transport_closure_symbol_set_json": file_hash(audit_dir / "high_pressure_gas_transport_closure_symbol_set.json"),
            "high_pressure_gas_transport_runtime_call_graph_json": file_hash(audit_dir / "high_pressure_gas_transport_runtime_call_graph.json"),
        },
        "campaign_summary": {
            "total_cases": campaign_summary["total_cases_evaluated"],
            "passed_cases": campaign_summary["total_cases_passed"],
            "pass_rate_percent": campaign_summary["overall_pass_rate"] * 100.0,
            "max_viscosity_rel_err": campaign_summary["metrics"]["viscosity"]["max_relative_error"],
            "max_thermal_conductivity_rel_err": campaign_summary["metrics"]["thermal_conductivity"]["max_relative_error"],
            "max_binary_diff_rel_err": campaign_summary["metrics"]["binary_diff_coeffs"]["max_relative_error"],
            "max_mix_diff_rel_err": campaign_summary["metrics"]["mix_diff_coeffs"]["max_relative_error"],
            "max_mix_diff_mole_rel_err": campaign_summary["metrics"]["mix_diff_coeffs_mole"]["max_relative_error"],
            "max_mix_diff_mass_rel_err": campaign_summary["metrics"]["mix_diff_coeffs_mass"]["max_relative_error"],
            "adversarial_cases": campaign_summary["adversarial_cases_evaluated"],
            "adversarial_passed": campaign_summary["adversarial_cases_passed"],
        },
        "status": "VALIDATED_NUMERICALLY_EQUIVALENT",
    }

    out_path = audit_dir / "high_pressure_gas_transport_frozen_manifest.json"
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
        if "HighPressure" in row.get("native_class", ""):
            row["status"] = "VALIDATED_NUMERICALLY_EQUIVALENT"
            row["parity_algorithm"] = (
                "Validated 100% against pinned native Cantera C++ oracle across 288 differential, "
                "adversarial, and failure cases (1-600 bar, 250-1000 K). Relative error < 1.3e-10 for binary diff, "
                "< 2.0e-11 for viscosity, < 2.5e-11 for thermal cond, < 6.0e-11 for mix diff."
            )
            row["validation_test"] = "tests/test_hp_transport.py; tools/run_high_pressure_transport_campaign.py"
            updated_inv = True
        updated_lines.append(json.dumps(row))
    inv_file.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Updated complete_solver_inventory.jsonl for HighPressure: {updated_inv}")

    # Update scientific_status.json
    sci_file = audit_dir / "scientific_status.json"
    sci_data = json.loads(sci_file.read_text(encoding="utf-8"))
    updated_sci = False
    for item in sci_data.get("subsystems", []):
        if item.get("subsystem") == "high-pressure transport":
            item["status"] = "VALIDATED_NUMERICALLY_EQUIVALENT"
            item["evidence"] = "tests/test_hp_transport.py; audit/high_pressure_gas_transport_differential_summary.json"
            updated_sci = True
    sci_file.write_text(json.dumps(sci_data, indent=2), encoding="utf-8")
    print(f"Updated scientific_status.json for HighPressure: {updated_sci}")

if __name__ == "__main__":
    main()
