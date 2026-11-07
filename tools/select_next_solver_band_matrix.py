"""Select BandMatrix (banded linear algebra) as next dependency-ready solver family and build its starting snapshot, closure set, and runtime call graph."""

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
    status = run_git(["status", "--porcelain"])
    is_clean = len(status) == 0
    return commit, branch, is_clean


def main():
    repo_root = Path(__file__).resolve().parent.parent
    audit_dir = repo_root / "audit"
    orig_root = repo_root.parent / "cantera-original"
    provenance_dir = repo_root / "provenance"

    commit, branch, is_clean = get_git_info(repo_root)

    # 1. Extract BandMatrix canonical symbols from inventory
    inventory_path = provenance_dir / "canonical_5784_symbol_inventory.jsonl"
    band_symbols = {}
    with open(inventory_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            src_file = row.get("source_file", "")
            if "BandMatrix.h" in src_file or ("BandMatrix.cpp" in src_file and row.get("stable_symbol_id") == "src/numerics/BandMatrix.cpp:282:m"):
                sid = row["stable_symbol_id"]
                method_name = sid.split(":")[-1]
                is_non_runtime = row.get("current_status") == "NON_RUNTIME_CPP_ONLY_PROVEN"
                band_symbols[sid] = {
                    "canonical_index": idx,
                    "stable_symbol_id": sid,
                    "native_file": row["source_file"],
                    "native_class": "BandMatrix",
                    "native_method": method_name,
                    "python_target": f"cantera_python.numerics.BandMatrix.{method_name}",
                    "current_status": row.get("current_status", "NOT_PORTED"),
                    "scientific_runtime_relevance": row.get("scientific_runtime_relevance", "SCIENTIFIC_RUNTIME_CANDIDATE"),
                    "callers": row.get("callers"),
                    "callees": row.get("callees"),
                    "remaining_validation_requirement": (
                        "NONE_FINAL_DISPOSITION" if is_non_runtime else
                        "Differential parity against pinned native Cantera BandMatrix probe across banded LU factorization, "
                        "banded back-substitution solve, condition number estimation, matrix-vector multiplication, "
                        "and row/column boundary checking."
                    )
                }

    closure_set_path = audit_dir / "band_matrix_closure_symbol_set.json"
    with open(closure_set_path, "w", encoding="utf-8") as f:
        json.dump(band_symbols, f, indent=2)
    print(f"Generated audit/band_matrix_closure_symbol_set.json ({len(band_symbols)} symbols).")

    open_runtime_count = sum(1 for s in band_symbols.values() if s.get("current_status") != "NON_RUNTIME_CPP_ONLY_PROVEN")
    non_runtime_count = len(band_symbols) - open_runtime_count
    not_ported_count = sum(1 for s in band_symbols.values() if s.get("current_status") == "NOT_PORTED")
    partial_count = sum(1 for s in band_symbols.values() if s.get("current_status") == "PARTIAL")
    implemented_unval_count = sum(1 for s in band_symbols.values() if s.get("current_status") == "IMPLEMENTED_UNVALIDATED")
    final_count = sum(1 for s in band_symbols.values() if s.get("current_status") in ("PINNED_NATIVE_VALIDATED", "NON_RUNTIME_CPP_ONLY_PROVEN"))

    print(f"BAND_MATRIX_RUNTIME_SYMBOLS_TOTAL: {open_runtime_count}")
    print(f"BAND_MATRIX_FINAL_BEFORE: {final_count}")
    print(f"BAND_MATRIX_IMPLEMENTED_UNVALIDATED_BEFORE: {implemented_unval_count}")
    print(f"BAND_MATRIX_PARTIAL_BEFORE: {partial_count}")
    print(f"BAND_MATRIX_NOT_PORTED_BEFORE: {not_ported_count}")
    print(f"BAND_MATRIX_NON_RUNTIME_ROWS: {non_runtime_count}")

    # 2. Update audit/next_solver_selection.json
    selection = {
        "schema_version": 1,
        "selected_solver_family": "band_matrix",
        "subsystem": "banded linear algebra",
        "native_class": "BandMatrix",
        "selection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rationale": (
            "With the 0D reactor network and flow reactor DAE family fully validated and frozen "
            "(CVodesIntegrator, ReactorNet, IdasIntegrator, FlowReactor), the next essential numerical foundation "
            "prior to unlocking 1D flame simulations (MultiNewton, OneDim, Sim1D, Refiner) is BandMatrix "
            "(banded linear algebra). BandMatrix provides the banded LU factorization (LAPACK DGBTRF / gbfa equivalent), "
            "banded solve (LAPACK DGBTRS / gbsl equivalent), 1-norm computation, and condition number reciprocal (rcond). "
            "It is already partially implemented in cantera_python.numerics and has 25+ canonical symbols ready for closure."
        ),
        "dependency_status": {
            "CVodesIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ReactorNet": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "IdasIntegrator": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "FlowReactor": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "ChemEquil": "VALIDATED_NUMERICALLY_EQUIVALENT",
            "MultiTransport": "VALIDATED_NUMERICALLY_EQUIVALENT"
        },
        "canonical_symbols_total": len(band_symbols),
        "open_runtime_symbols": open_runtime_count,
        "non_runtime_proven_symbols": non_runtime_count,
        "starting_snapshot_path": "audit/band_matrix_starting_snapshot.json",
        "closure_symbol_set_path": "audit/band_matrix_closure_symbol_set.json",
        "runtime_call_graph_path": "audit/band_matrix_runtime_call_graph.json"
    }

    selection_path = audit_dir / "next_solver_selection.json"
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Updated {selection_path}")

    # 3. Build starting snapshot
    snapshot = {
        "schema_version": 1,
        "artifact": "audit/band_matrix_starting_snapshot.json",
        "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "is_worktree_clean": is_clean,
        "solver_family": "band_matrix",
        "subsystem": "banded linear algebra",
        "native_source": {
            "commit": "726522be4e2a13454d8415b7ef799d621f665cf3",
            "header": "include/cantera/numerics/BandMatrix.h",
            "source": "src/numerics/BandMatrix.cpp",
            "header_sha256": file_hash(orig_root / "include" / "cantera" / "numerics" / "BandMatrix.h"),
            "source_sha256": file_hash(orig_root / "src" / "numerics" / "BandMatrix.cpp"),
        },
        "python_files": {
            "numerics_py_sha256": file_hash(repo_root / "cantera_python" / "numerics.py"),
            "test_numerics_py_sha256": file_hash(repo_root / "tests" / "test_numerics.py"),
            "compare_band_matrix_native_probe_py_sha256": file_hash(repo_root / "tools" / "compare_band_matrix_native_probe.py"),
        },
        "closure_set_sha256": file_hash(closure_set_path),
        "inventory_counts": {
            "total_canonical_symbols": len(band_symbols),
            "open_runtime_symbols": open_runtime_count,
            "non_runtime_proven_symbols": non_runtime_count,
            "not_ported": not_ported_count,
            "partial": partial_count,
            "implemented_unvalidated": implemented_unval_count,
            "final": final_count,
        }
    }

    snapshot_path = audit_dir / "band_matrix_starting_snapshot.json"
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Generated {snapshot_path}")

    # 4. Build runtime call graph
    call_graph = {
        "schema_version": 1,
        "solver_family": "band_matrix",
        "subsystem": "banded linear algebra",
        "public_entry_points": {
            "resize(n, kl, ku, v=0.0)": {
                "callees": ["_nrows_alloc calculation", "np.full allocation"],
                "scientific_purpose": "Allocate banded storage with 2*kl + ku + 1 rows for DGBTRF partial pivoting"
            },
            "bfill(v=0.0)": {
                "callees": ["self._data.fill"],
                "scientific_purpose": "Fill banded matrix storage with scalar value"
            },
            "zero()": {
                "callees": ["self.bfill(0.0)"],
                "scientific_purpose": "Zero all matrix entries"
            },
            "value(i, j)": {
                "callees": ["_band_row(i, j)"],
                "scientific_purpose": "Read matrix element A(i, j) in band storage or return 0 outside band"
            },
            "setValue(i, j, v)": {
                "callees": ["_band_row(i, j)"],
                "scientific_purpose": "Assign matrix element A(i, j) with boundary/bandwidth validation"
            },
            "mult(b, prod=None)": {
                "callees": ["value(i, j)"],
                "scientific_purpose": "Matrix-vector product prod = A * b using band limits"
            },
            "leftMult(b, prod=None)": {
                "callees": ["value(i, j)"],
                "scientific_purpose": "Transposed matrix-vector product prod = b * A"
            },
            "factor()": {
                "callees": ["_lu_value", "_set_lu_value", "DGBTRF partial pivoting"],
                "scientific_purpose": "Compute LU factorization with partial row pivoting in compact band storage"
            },
            "solve(b)": {
                "callees": ["factor()", "forward elimination with pivots (L*y = P*b)", "backward substitution (U*x = y)"],
                "scientific_purpose": "Solve linear system A * x = b for vector or multi-RHS matrix b"
            },
            "rcond(a1norm=0.0)": {
                "callees": ["to_dense()", "_reciprocal_condition_one_norm()"],
                "scientific_purpose": "Compute reciprocal 1-norm condition number estimate (LAPACK DGBCON / DGETRF)"
            },
            "oneNorm()": {
                "callees": ["to_dense()", "_matrix_one_norm()"],
                "scientific_purpose": "Compute matrix 1-norm (max column sum)"
            },
            "checkRows(value)": {
                "callees": ["abs(value(i, j))"],
                "scientific_purpose": "Check that each row contains at least one entry exceeding threshold"
            },
            "checkColumns(value)": {
                "callees": ["abs(value(i, j))"],
                "scientific_purpose": "Check that each column contains at least one entry exceeding threshold"
            }
        }
    }

    call_graph_path = audit_dir / "band_matrix_runtime_call_graph.json"
    with open(call_graph_path, "w", encoding="utf-8") as f:
        json.dump(call_graph, f, indent=2)
    print(f"Generated {call_graph_path}")


if __name__ == "__main__":
    main()
