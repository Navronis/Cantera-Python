"""Compare fresh Wave 1-2 records in this isolated audit directory."""
import runpy
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
ns = runpy.run_path(str(repo / "tools" / "compare_native_vs_python.py"))
scratch = Path(__file__).resolve().parent
ns["compare_wave1"].__globals__["AUDIT"] = scratch
for wave in (1, 2, 3):
    summary, mismatches, _ = ns[f"compare_wave{wave}"]()
    print(f"WAVE {wave}: mismatch_count={len(mismatches)}")
