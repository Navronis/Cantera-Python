"""Run pinned native Wave 3 corpus into this isolated audit directory."""
import runpy
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
ns = runpy.run_path(str(repo / "tools" / "run_native_oracle_runner.py"))
ns["run_wave3"].__globals__["AUDIT"] = Path(__file__).resolve().parent
ns["run_wave3"]()
