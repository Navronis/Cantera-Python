"""Build the explicit scientific gap list from the conservative inventories."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOLVERS = ROOT / "audit" / "complete_solver_inventory.jsonl"
OUTPUT = ROOT / "audit" / "remaining_gaps.jsonl"


def main():
    rows = [json.loads(line) for line in SOLVERS.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    gaps = []
    for index, row in enumerate(rows, 1):
        if row["status"] in {"EXACT_DIRECT_PORT", "NATIVE_ALGORITHM_PORT",
                             "VALIDATED_NUMERICALLY_EQUIVALENT"}:
            continue
        gaps.append({
            "gap_id": f"SOLVER-{index:03d}",
            "subsystem": row["subsystem"],
            "status": row["status"],
            "native_source": f"{row['native_file']}::{row['native_function']}",
            "current": row["python_current_algorithm"],
            "required_parity_work": row["parity_algorithm"],
            "required_native_python_work": row["native_python_algorithm"],
            "required_validation": row["validation_test"] or
                "new exact-pinned native differential test required",
            "blocking_completion": True,
        })
    OUTPUT.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                              for row in gaps), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "open_solver_gaps": len(gaps)}))


if __name__ == "__main__":
    main()
