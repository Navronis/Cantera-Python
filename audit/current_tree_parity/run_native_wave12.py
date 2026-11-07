"""Run pinned native Waves 1-2 into this isolated audit directory."""
import runpy
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
ns = runpy.run_path(str(repo / "tools" / "run_native_oracle_runner.py"))
ns["run_wave1"].__globals__["AUDIT"] = Path(__file__).resolve().parent
ns["run_wave1"]()
ns["run_wave2"]()
