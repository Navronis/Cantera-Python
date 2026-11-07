"""Save the trace-heavy MultiTransport matrix/solve comparison as JSON."""
from contextlib import redirect_stdout
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
ns = runpy.run_path(str(ROOT / "tools" / "compare_multitransport_matrix_probe.py"))
sys.argv = [
    str(ROOT / "tools" / "compare_multitransport_matrix_probe.py"),
    str(AUDIT / "wave4_native_trace_heavy.txt"),
    str(AUDIT / "wave4_python_trace_heavy.txt"),
    "current source trace-heavy pinned rerun",
]
with (AUDIT / "wave4_trace_heavy_comparison.json").open("w", encoding="utf-8") as out:
    with redirect_stdout(out):
        ns["main"]()
print("audit/current_tree_parity/wave4_trace_heavy_comparison.json")
